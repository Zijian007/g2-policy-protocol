"""Dataset content and source provenance digests."""

import hashlib
import json
from pathlib import Path
from typing import Any


def mixture_fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps({"version": 1, "parts": parts}, sort_keys=True).encode()).hexdigest()


def dataset_identity(root: str | Path) -> str:
    """Hash metadata contents and payload file stats; do not read large videos."""
    root = Path(root)
    digest = hashlib.sha256(str(root.resolve()).encode())
    for directory in ("meta", "data", "videos"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            digest.update(json.dumps([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns]).encode())
            if directory == "meta":
                digest.update(path.read_bytes())
    return digest.hexdigest()


def source_digest() -> str:
    """Hash the shared package, including nested contracts and processing code."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        relative_name = f"g2_policy_protocol/{source_path.relative_to(package_root).as_posix()}"
        digest.update(relative_name.encode())
        digest.update(source_path.read_bytes())
    return digest.hexdigest()
