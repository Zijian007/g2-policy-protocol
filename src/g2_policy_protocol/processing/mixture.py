"""Deterministic whole-episode repeat plans and virtual source/frame lookup."""

import math
import warnings
from collections.abc import Sequence
from typing import TypedDict

import numpy as np

# Physical episode interval [start, stop), virtual mixture interval [from, to).
MixtureRow = TypedDict(
    "MixtureRow",
    {"source": int, "episode": int, "start": int, "stop": int, "from": int, "to": int},
)


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
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights):
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


def expand_rows(
    episode_lengths_per_source: Sequence[Sequence[int]],
    plan: Sequence[Sequence[int]],
) -> list[MixtureRow]:
    """Build contiguous virtual intervals mapping to source-local episode rows."""
    rows, cursor = [], 0
    for source, (lengths, counts) in enumerate(zip(episode_lengths_per_source, plan, strict=True)):
        start = 0
        for episode, (length, copies) in enumerate(zip(lengths, counts, strict=True)):
            length = int(length)
            for _ in range(copies):
                rows.append(
                    {
                        "source": source,
                        "episode": episode,
                        "start": start,
                        "stop": start + length,
                        "from": cursor,
                        "to": cursor + length,
                    }
                )
                cursor += length
            start += length
    return rows


class MixtureIndex:
    """Small, pickleable lookup table; never materializes per-frame indices."""

    def __init__(self, lengths, weights=None, seed=0, *, plan=None):
        self.plan = repeat_plan(lengths, weights, seed) if plan is None else plan
        self.rows = expand_rows(lengths, self.plan)
        self.ends = np.asarray([row["to"] for row in self.rows], dtype=np.int64)
        totals = [
            sum(int(n) * c for n, c in zip(part, counts, strict=True))
            for part, counts in zip(lengths, self.plan, strict=True)
        ]
        total_frames = sum(totals)
        self.proportions = [total / total_frames for total in totals]

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        row = self.rows[episode]
        return row["source"], row["start"] + index - row["from"], episode
