"""Check a third-party G2 policy server without robot-side dependencies."""

import argparse
import json
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from . import codec, protocol as p
from .samples import load_sample, prepare_images, synthetic_sample


def connection_uri(uri, encoding):
    parts = urlsplit(uri)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError("URL must start with ws:// or wss:// and include a host")
    query = dict(parse_qsl(parts.query))
    query["encoding"] = encoding
    return urlunsplit(parts._replace(query=urlencode(query)))


def check_connection(uri, encoding, sample, *, requests=2, timeout=30,
                     policy_hz=None, rtc=False, semantic_checks=True):
    from websockets.sync.client import connect

    result = {"encoding": encoding, "status": "failed", "requests": [], "warnings": []}
    try:
        with connect(connection_uri(uri, encoding), compression=None, max_size=None,
                     open_timeout=timeout) as ws:
            metadata = codec.loads(ws.recv(timeout=timeout), encoding=encoding)
            p.validate_metadata(metadata, policy_hz=policy_hz, require_rtc=rtc)
            result["metadata"] = metadata
            images = prepare_images(sample, metadata)
            if metadata["requires_joint_state"] and "joint_state" not in sample:
                raise ValueError("server requires joint_state but sample does not contain it")
            previous = None
            for request_id in range(1, requests + 1):
                request = p.build_request(
                    request_id=request_id, prompt=sample["prompt"], state=sample["state"],
                    images=images, metadata=metadata,
                    joint_state=sample.get("joint_state") if metadata["requires_joint_state"] else None,
                    rtc_enabled=rtc, inference_delay=0,
                    prev_chunk_left_over=previous if rtc else None,
                )
                start = time.perf_counter()
                ws.send(codec.dumps(request, encoding=encoding))
                response = codec.loads(ws.recv(timeout=timeout), encoding=encoding)
                actions = p.parse_response(response, request_id=request_id)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if semantic_checks:
                    check_action_semantics(actions)
                if actions.shape[0] != metadata["action_horizon"]:
                    warning = "returned T differs from action_horizon; confirm execution_steps and scheduling"
                    if warning not in result["warnings"]:
                        result["warnings"].append(warning)
                result["requests"].append({"request_id": request_id,
                    "shape": list(actions.shape), "round_trip_ms": round(elapsed_ms, 3)})
                previous = actions
            result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_action_semantics(actions):
    """Extra diagnostics beyond core protocol shape/finite checks."""
    if np.any((actions[:, [9, 19]] < 0) | (actions[:, [9, 19]] > 1)):
        raise ValueError("gripper values must lie in [0, 1]")
    for offset in (3, 13):
        first = actions[:, offset:offset + 3].astype(np.float64)
        second = actions[:, offset + 3:offset + 6].astype(np.float64)
        if np.any(np.linalg.norm(first, axis=1) < 1e-8):
            raise ValueError("rot6d has a zero first column")
        unit = first / np.linalg.norm(first, axis=1, keepdims=True)
        residual = second - np.sum(unit * second, axis=1, keepdims=True) * unit
        if np.any(np.linalg.norm(residual, axis=1) < 1e-8):
            raise ValueError("rot6d columns are degenerate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--encoding", choices=("all", "msgpack", "json"), default="all")
    parser.add_argument("--sample", type=Path, help="NPZ observation; default: synthetic fixture")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--policy-hz", type=float, help="Expected robot policy rate; omitted: accept server rate")
    parser.add_argument("--rtc", action="store_true", help="Exercise RTC prefix on requests after the first")
    parser.add_argument("--protocol-only", action="store_true", help="Skip extra gripper and rot6d diagnostics")
    parser.add_argument("--report", type=Path, help="Write a JSON report")
    args = parser.parse_args()
    if args.requests < 2 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--requests must be >= 2 and --timeout must be finite and > 0")
    report = {"status": "failed", "url": args.url,
        "scope": "Protocol integration only; no robot execution or model-quality validation",
        "rtc_tested": args.rtc, "extra_semantic_checks": not args.protocol_only, "connections": []}
    try:
        sample = load_sample(args.sample) if args.sample else synthetic_sample()
        report["sample"] = {"path": str(args.sample) if args.sample else None,
            "provenance": sample["provenance"]}
        encodings = ("msgpack", "json") if args.encoding == "all" else (args.encoding,)
        for encoding in encodings:
            report["connections"].append(check_connection(
                args.url, encoding, sample, requests=args.requests, timeout=args.timeout,
                policy_hz=args.policy_hz, rtc=args.rtc, semantic_checks=not args.protocol_only,
            ))
        if all(item["status"] == "passed" for item in report["connections"]):
            report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
