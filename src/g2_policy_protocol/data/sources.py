"""Local dataset descriptions and deterministic episode-level mixtures.

No backend imports: callers own format conversion, encoding and native metadata.
"""

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np


def image_size(data):
    value = data.get("image_size")
    if value is not None and (
        not isinstance(value, (list, tuple)) or len(value) != 2
        or any(type(v) is not int or v <= 0 for v in value)
    ):
        raise ValueError("data.image_size must be [height, width] with positive integers")
    return None if value is None else tuple(value)


def resolve_sources(data, *, base_dir):
    """Resolve paths against an explicit repository root, without accessing disk."""
    if not isinstance(data, dict) or data.keys() - {"root", "repo_id", "sources", "cameras", "image_size"}:
        raise ValueError("G2 data fields must be root, repo_id, sources, cameras and image_size")
    image_size(data)
    if ("repo_id" in data) == ("sources" in data):
        raise ValueError("Set exactly one of data.repo_id and data.sources")
    root = data.get("root", "./data")
    if not isinstance(root, str) or not root.strip() or "://" in root:
        raise ValueError("data.root must be a nonempty local path")
    root = Path(root).expanduser()
    root = (Path(base_dir) / root).resolve() if not root.is_absolute() else root.resolve()
    if "repo_id" in data:
        repo = data["repo_id"]
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError("data.repo_id must be a nonempty string")
        if Path(repo).is_absolute() or ".." in Path(repo).parts or Path(repo) == Path("."):
            raise ValueError("data.repo_id must be relative to data.root without '..'")
        return [{"name": repo, "path": str(root / repo), "repo_id": repo, "weight": None}]
    entries = data["sources"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("data.sources must be a nonempty list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.keys() - {"name", "path", "weight"}:
            raise ValueError("Each data.sources entry accepts name, path and weight")
        value = entry.get("path")
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError("Each data.sources.path must be a nonempty local path")
        name = entry.get("name", value)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each source name must be a nonempty string")
        weight = entry.get("weight")
        if "weight" in entry and (type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0):
            raise ValueError(f"Source {name}: weight must be a finite positive number")
        path = Path(value).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        result.append({"name": name, "path": str(path),
                       "repo_id": "local/" + hashlib.sha256(str(path).encode()).hexdigest()[:16],
                       "weight": weight})
    if len({source["name"] for source in result}) != len(result):
        raise ValueError("Source names must be unique")
    if any(source["weight"] is None for source in result) and not all(source["weight"] is None for source in result):
        raise ValueError("Set weight on all sources or omit it on all sources")
    return result


def repeat_plan(episode_lengths_per_source, weights=None, seed=0):
    """Repeat whole episodes; fractional copies use a fixed-seed Bernoulli draw.

    Weights are approximate at episode granularity. Report realized proportions.
    """
    lengths = [np.asarray(part, dtype=np.int64) for part in episode_lengths_per_source]
    if not lengths or any(part.ndim != 1 or not len(part) or np.any(part <= 0) for part in lengths):
        raise ValueError("Every source must contain positive effective episode lengths")
    if weights is not None and len(weights) != len(lengths):
        raise ValueError("weights must contain one value per source")
    if weights is None or all(value is None for value in weights):
        return [[1] * len(part) for part in lengths]
    if len(weights) != len(lengths) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights
    ):
        raise ValueError("weights must contain one finite positive number per source")
    # Normalize by max first to avoid overflowing when all weights are large.
    ratios = (np.asarray(weights, np.float64) / max(weights)) / np.asarray([sum(p) for p in lengths])
    copies = ratios / min(ratios)
    if not np.all(np.isfinite(copies)):
        raise ValueError("Source weight ratio is too large")
    rng = np.random.default_rng(seed)
    result = []
    for part, count in zip(lengths, copies):
        if count > 100:
            warnings.warn(f"A source is repeated approximately {count:.1f} times", stacklevel=2)
        integer = math.floor(float(count))
        result.append([integer + int(rng.random() < count - integer) for _ in part])
    return result


def expand_rows(episode_lengths_per_source, plan):
    """Build contiguous virtual intervals mapping to source-local episode rows."""
    rows, cursor = [], 0
    for source, (lengths, counts) in enumerate(zip(episode_lengths_per_source, plan, strict=True)):
        start = 0
        for episode, (length, copies) in enumerate(zip(lengths, counts, strict=True)):
            length = int(length)
            for _ in range(copies):
                rows.append({"source": source, "episode": episode, "start": start, "stop": start + length,
                             "from": cursor, "to": cursor + length})
                cursor += length
            start += length
    return rows


class MixtureIndex:
    """Small, pickleable lookup table; never materializes per-frame indices."""

    def __init__(self, lengths, weights=None, seed=0, *, plan=None):
        self.plan = repeat_plan(lengths, weights, seed) if plan is None else plan
        self.rows = expand_rows(lengths, self.plan)
        self.ends = np.asarray([row["to"] for row in self.rows], dtype=np.int64)
        totals = [sum(int(n) * c for n, c in zip(part, counts, strict=True))
                  for part, counts in zip(lengths, self.plan, strict=True)]
        self.proportions = [total / sum(totals) for total in totals]

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        row = self.rows[episode]
        return row["source"], row["start"] + index - row["from"], episode


def mixture_fingerprint(*parts):
    return hashlib.sha256(json.dumps({"version": 1, "parts": parts}, sort_keys=True).encode()).hexdigest()


def dataset_identity(root):
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
