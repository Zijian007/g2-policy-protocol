"""Run actual adapters in their native environments against wholebody's dev client.

No model weights, GPU or robot hardware are used. Each sibling repository needs
its own installed .venv. Run this with lerobot-finetune/.venv/bin/python.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


async def serve_worker(args):
    import numpy as np
    from g2_policy_tools.geometry import ActionCodec
    from g2_policy_tools.schema import resolve_policy
    from websockets.asyncio.server import serve

    root = args.stack_root / args.backend
    sys.path.insert(0, str(root / "src"))
    raw = {"timing": {"history_steps": args.history}}
    if args.quantity == "joint_position":
        raw["action"] = {"pose": {"enabled": False}, "joint": {"enabled": True}}
    else:
        raw["action"] = {
            "pose": {
                "delta": True,
                "method": "component",
                "reference": {"source": "state", "scope": "chunk"},
            }
        }
    settings = resolve_policy(raw, horizon=4)
    codec = ActionCodec(settings, "right")

    def prediction(obs):
        reference = obs[
            "observation/pose"
            if args.quantity == "pose_target"
            else "observation/joint"
        ]
        actions = np.tile(reference, (4, 1))
        actions[:, 10 if args.quantity == "pose_target" else 8] += np.arange(1, 5) / 100
        if args.history > 1:
            assert obs["observation/pose_history"].shape == (2, 20)
        return codec.encode(actions, obs)

    if args.backend == "lerobot-finetune":
        from g2_policy_protocol.server import PolicyServer
        from lerobot_finetune.deploy.g2_server import G2Backend, metadata_for

        runtime = SimpleNamespace(
            adapter={"policy": settings, "arm": "right", "cameras": ["head_color"]},
            config=SimpleNamespace(
                type="act",
                observation_delta_indices=[-1, 0] if args.history > 1 else None,
                input_features={
                    "observation.images.head_color": SimpleNamespace(shape=(3, 8, 8))
                },
            ),
            rename_map={},
            horizon=4,
        )

        def predict(batch, refs):
            assert batch["observation.state"].shape == (
                (1, 2, 10) if args.history > 1 else (1, 10)
            )
            assert batch["observation.images.head_color"].shape == (
                (1, 2, 3, 8, 8) if args.history > 1 else (1, 3, 8, 8)
            )
            return codec.decode(prediction(refs), refs)

        runtime.predict = predict
        metadata = metadata_for(runtime)
        server = PolicyServer(G2Backend(runtime, metadata), metadata=metadata)
    else:
        from openpi.adapters.g2.policy import G2Policy
        from openpi.adapters.g2.server import G2PolicyServer

        def infer(obs):
            image = obs["observation/image"]
            assert image.shape == (
                (2, 224, 224, 3) if args.history > 1 else (224, 224, 3)
            )
            return {"actions": prediction(obs)}

        native = SimpleNamespace(
            metadata={"arm": "right"}, infer=infer, reset=lambda: None
        )
        server = G2PolicyServer(
            G2Policy(native, settings),
            metadata={
                "policy_contract": settings,
                "arm": "right",
                "action_horizon": 4,
                "required_cameras": ["head_color"],
                "requires_prompt": False,
            },
        )
    # Let the OS assign the port, and report readiness only after the listener exists.
    async with serve(
        server._handler, "127.0.0.1", 0, compression=None, max_size=None
    ) as listener:
        print(listener.sockets[0].getsockname()[1], flush=True)
        await listener.serve_forever()


def check_stack(args):
    import numpy as np
    from g2_policy_protocol import protocol

    sys.path.insert(0, str(args.stack_root))
    from wholebody.external_services.policy.runtime.execution import execution_plan
    from wholebody.external_services.policy.runtime.transport import (
        G2WebSocketPolicyClient,
    )

    manifest = {}
    for name in ("wholebody", "lerobot-finetune", "openpi", "g2-policy-protocol"):
        root = args.stack_root / name
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip()
        paths = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
        ).split(b"\0")
        digest = hashlib.sha256()
        for relative in sorted(set(paths) - {b""}):
            path = root / relative.decode()
            if path.is_file():
                digest.update(relative + b"\0" + path.read_bytes())
        manifest[name] = {
            "commit": head,
            "branch": branch,
            "source_sha256": digest.hexdigest(),
        }
    if manifest["wholebody"]["branch"] != "gdk_3.2.3_collect_session_with_umi_dev":
        raise ValueError(
            "Stack check requires the wholebody collect_session_with_umi_dev branch"
        )
    results = []
    for backend, quantity, history in itertools.product(
        ("lerobot-finetune", "openpi"),
        ("pose_target", "joint_position"),
        (1, 2),
    ):
        command = [
            str(args.stack_root / backend / ".venv/bin/python"),
            str(Path(__file__).resolve()),
            "--stack-root",
            str(args.stack_root),
            "--backend",
            backend,
            "--quantity",
            quantity,
            "--history",
            str(history),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        try:
            line = process.stdout.readline().strip()
            if not line:
                raise RuntimeError(
                    f"{backend} worker exited during startup ({process.wait()})"
                )
            port = int(line)
            for encoding in ("json", "msgpack"):
                with G2WebSocketPolicyClient(port=port, encoding=encoding) as client:
                    metadata = client.metadata
                    assert metadata["extensions"]["response_mode"] == "full_chunk"
                    assert execution_plan(
                        metadata, fallback_horizon=4, execution_steps=2
                    ) == (2, 4)
                    pose = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], 2).astype(
                        np.float32
                    )
                    joint = np.zeros(16, np.float32)
                    image = np.full((8, 8, 3), 128, np.uint8)
                    context = {}
                    keys = protocol.observation_contract(metadata)["context_keys"]
                    if "start_pose" in keys:
                        context["start_pose"] = pose
                    if history > 1:
                        context["pose_history"] = np.tile(pose, (2, 1))
                        if "joint_history" in keys:
                            context["joint_history"] = np.tile(joint, (2, 1))
                        size = metadata["image_sizes"]["head_color"]
                        context["image_history"] = {
                            "head_color": np.full((2, *size, 3), 128, np.uint8)
                        }
                    request = {
                        "pose": pose,
                        "joint_state": joint,
                        "images": {"head_color": image},
                        "context": context,
                        "observation_time_ns": 1234567,
                    }
                    for _ in range(2):
                        actions = client.infer(**request)
                        assert actions.shape == (
                            4,
                            20 if quantity == "pose_target" else 16,
                        )
                        offset = 10 if quantity == "pose_target" else 8
                        np.testing.assert_allclose(
                            actions[:, offset], np.arange(1, 5) / 100, atol=1e-6
                        )
                        np.testing.assert_allclose(
                            actions[:, :offset],
                            np.tile(
                                pose[:offset] if offset == 10 else joint[:offset],
                                (4, 1),
                            ),
                        )
                    client.reset()
                    np.testing.assert_allclose(client.infer(**request), actions)
                    results.append(
                        {
                            "backend": backend,
                            "quantity": quantity,
                            "history": history,
                            "encoding": encoding,
                        }
                    )
        finally:
            process.terminate()
            process.wait(timeout=15)
    print(
        json.dumps(
            {
                "passed": len(results),
                "protocol_package": importlib.metadata.version("g2-policy-protocol"),
                "stack_manifest": manifest,
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["lerobot-finetune", "openpi"])
    parser.add_argument("--quantity", choices=["pose_target", "joint_position"])
    parser.add_argument("--history", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    if args.backend:
        asyncio.run(serve_worker(args))
    else:
        check_stack(args)
