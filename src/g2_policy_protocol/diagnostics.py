"""Portable observation fixtures and a G2 policy server integration checker."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import protocol as p
from .runtime.images import prepare_images


def synthetic_sample():
    state = np.zeros(p.POSE_DIM, dtype=np.float32)
    state[[3, 7, 13, 17]] = 1
    state[:3] = [0.35, 0.2, 0.5]
    state[10:13] = [0.35, -0.2, 0.5]
    y, x = np.indices((240, 320))
    images = {}
    for index, camera in enumerate(p.CAMERA_IDS):
        image = np.empty((240, 320, 3), dtype=np.uint8)
        image[..., 0] = x % 256
        image[..., 1] = y % 256
        image[..., 2] = 40 + index * 70
        images[camera] = image
    return dict(
        state=state,
        joint_state=np.zeros(p.JOINT_DIM, dtype=np.float32),
        prompt="保持当前位姿",
        provenance="synthetic; not recorded from a robot",
        images=images,
    )


def validate_sample(sample):
    for key, shape in (("state", (p.POSE_DIM,)), ("joint_state", (p.JOINT_DIM,))):
        if key not in sample and key == "joint_state":
            continue
        value = np.asarray(sample[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"sample {key} must be finite {shape}")
    for key in ("prompt", "provenance"):
        if not isinstance(sample[key], str) or not sample[key].strip():
            raise ValueError(f"sample {key} must be nonempty text")
    if not sample["images"]:
        raise ValueError("sample must contain at least one image")
    for camera, image in sample["images"].items():
        if camera not in p.CAMERA_IDS:
            raise ValueError(f"unknown camera: {camera}")
        p.validate_image(image, encoding="raw")
    return sample


def save_sample(path, sample):
    validate_sample(sample)
    fields = {key: sample[key] for key in ("state", "prompt", "provenance")}
    if "joint_state" in sample:
        fields["joint_state"] = sample["joint_state"]
    fields.update(sample["images"])
    # File handle prevents NumPy from silently appending a second .npz suffix.
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **fields)


def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        allowed = {"state", "joint_state", "prompt", "provenance", *p.CAMERA_IDS}
        if set(data.files) - allowed:
            raise ValueError(f"unknown sample fields: {set(data.files) - allowed}")
        sample = {key: data[key].copy() for key in ("state", "joint_state") if key in data}
        for key in ("prompt", "provenance"):
            value = data[key]
            if value.shape != () or value.dtype.kind != "U":
                raise ValueError(f"sample {key} must be a scalar Unicode string")
            sample[key] = value.item()
        sample["images"] = {key: data[key].copy() for key in p.CAMERA_IDS if key in data}
    return validate_sample(sample)


def synthetic_context(sample, metadata):
    """Repeat fixture values for protocol diagnostics only; never a robot history provider."""
    observation = metadata["observation_spec"]
    result = {}
    for key in observation["context_keys"]:
        if key == "image_history":
            prepared = prepare_images(sample["images"], metadata)
            result[key] = {
                camera: (
                    [image] * observation["history_steps"]
                    if metadata["image_encoding"] == "jpeg"
                    else np.repeat(image[None], observation["history_steps"], axis=0)
                )
                for camera, image in prepared.items()
            }
        elif key == "base_to_parent":
            result[key] = np.eye(4)
        else:
            if "pose" in key:
                value = sample["state"]
            else:
                value = sample["joint_state"] if "joint_state" in sample else np.zeros(p.JOINT_DIM, np.float32)
            result[key] = np.tile(value, (observation["history_steps"], 1)) if key.endswith("_history") else value
    return result


def check_connection(uri, encoding, sample, *, requests=2, timeout=30, policy_hz=None, rtc=False, semantic_checks=True):
    from .runtime.client import PolicyClient

    result = {"encoding": encoding, "status": "failed", "requests": [], "warnings": []}
    try:
        with PolicyClient(
            uri, encoding=encoding, connect_timeout_s=timeout, policy_hz=policy_hz, require_rtc=rtc
        ) as client:
            metadata = client.metadata
            result["metadata"] = metadata
            context = synthetic_context(sample, metadata) if sample["provenance"].startswith("synthetic;") else None
            if metadata["requires_joint_state"] and "joint_state" not in sample:
                raise ValueError("server requires joint_state but sample does not contain it")
            previous = None
            for request_id in range(1, requests + 1):
                start = time.perf_counter()
                actions = client.infer(
                    prompt=sample["prompt"],
                    state=sample["state"],
                    images=sample["images"],
                    joint_state=sample.get("joint_state"),
                    rtc_enabled=rtc,
                    prev_chunk_left_over=previous if rtc else None,
                    observation_time_ns=time.monotonic_ns(),
                    context=context,
                    recv_timeout_s=timeout,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                if semantic_checks:
                    check_action_semantics(actions, metadata["action_spec"]["control_quantity"])
                if actions.shape[0] != metadata["action_horizon"]:
                    warning = "returned T differs from action_horizon; confirm execution_steps and scheduling"
                    if warning not in result["warnings"]:
                        result["warnings"].append(warning)
                result["requests"].append(
                    {"request_id": request_id, "shape": list(actions.shape), "round_trip_ms": round(elapsed_ms, 3)}
                )
                previous = actions
            result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_action_semantics(actions, control_quantity="pose_target"):
    """Extra diagnostics beyond core protocol shape/finite checks."""
    grips = [actions.shape[1] // 2 - 1, actions.shape[1] - 1]
    if np.any((actions[:, grips] < 0) | (actions[:, grips] > 1)):
        raise ValueError("gripper values must lie in [0, 1]")
    if control_quantity != "pose_target":
        return
    for offset in (3, 13):
        first = actions[:, offset : offset + 3].astype(np.float64)
        second = actions[:, offset + 3 : offset + 6].astype(np.float64)
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
    report = {
        "status": "failed",
        "url": args.url,
        "scope": "Protocol integration only; no robot execution or model-quality validation",
        "rtc_tested": args.rtc,
        "extra_semantic_checks": not args.protocol_only,
        "connections": [],
    }
    try:
        sample = load_sample(args.sample) if args.sample else synthetic_sample()
        report["sample"] = {"path": str(args.sample) if args.sample else None, "provenance": sample["provenance"]}
        encodings = ("msgpack", "json") if args.encoding == "all" else (args.encoding,)
        for encoding in encodings:
            report["connections"].append(
                check_connection(
                    args.url,
                    encoding,
                    sample,
                    requests=args.requests,
                    timeout=args.timeout,
                    policy_hz=args.policy_hz,
                    rtc=args.rtc,
                    semantic_checks=not args.protocol_only,
                )
            )
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
