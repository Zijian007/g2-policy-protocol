"""Resolve dataset paths and weights from training configuration."""

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict


class DatasetSource(TypedDict):
    name: str
    path: str
    repo_id: str
    weight: float | None


def image_size(dataset_settings: Mapping[str, Any]) -> tuple[int, int] | None:
    value = dataset_settings.get("image_size")
    if value is not None and (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError("data.image_size must be [height, width] with positive integers")
    return None if value is None else tuple(value)


def resolve_sources(dataset_settings: Mapping[str, Any], *, base_dir: str | Path) -> list[DatasetSource]:
    """Resolve paths against an explicit repository root, without accessing disk."""
    if not isinstance(dataset_settings, dict) or dataset_settings.keys() - {
        "root",
        "repo_id",
        "sources",
        "cameras",
        "image_size",
    }:
        raise ValueError("G2 data fields must be root, repo_id, sources, cameras and image_size")
    image_size(dataset_settings)
    if ("repo_id" in dataset_settings) == ("sources" in dataset_settings):
        raise ValueError("Set exactly one of data.repo_id and data.sources")
    root = dataset_settings.get("root", "./data")
    if not isinstance(root, str) or not root.strip() or "://" in root:
        raise ValueError("data.root must be a nonempty local path")
    root = Path(root).expanduser()
    root = (Path(base_dir) / root).resolve() if not root.is_absolute() else root.resolve()
    if "repo_id" in dataset_settings:
        repo = dataset_settings["repo_id"]
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError("data.repo_id must be a nonempty string")
        if Path(repo).is_absolute() or ".." in Path(repo).parts or Path(repo) == Path("."):
            raise ValueError("data.repo_id must be relative to data.root without '..'")
        return [{"name": repo, "path": str(root / repo), "repo_id": repo, "weight": None}]
    entries = dataset_settings["sources"]
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
        if (
            weight != "auto"
            and "weight" in entry
            and (type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0)
        ):
            raise ValueError(f"Source {name}: weight must be 'auto' or a finite positive number")
        path = Path(value).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        result.append(
            {
                "name": name,
                "path": str(path),
                "repo_id": "local/" + hashlib.sha256(str(path).encode()).hexdigest()[:16],
                "weight": weight,
            }
        )
    if len({source["name"] for source in result}) != len(result):
        raise ValueError("Source names must be unique")
    if any(source["weight"] == "auto" for source in result):
        if not all(source["weight"] == "auto" for source in result):
            raise ValueError(
                "Set weight: auto on all sources, omit it on all sources, or use numeric weights on all sources"
            )
        for source in result:
            source["weight"] = None
    if any(source["weight"] is None for source in result) and not all(source["weight"] is None for source in result):
        raise ValueError("Set weight on all sources or omit it on all sources")
    return result
