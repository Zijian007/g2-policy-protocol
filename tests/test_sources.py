import numpy as np
import pytest
from g2_policy_protocol.data.sources import (
    MixtureIndex,
    image_size,
    repeat_plan,
    resolve_sources,
)
from g2_policy_protocol.images import resize_rgb


def test_paths_and_legacy_configuration(tmp_path):
    legacy = resolve_sources({"root": "./data", "repo_id": "owner/task"}, base_dir=tmp_path)
    assert legacy[0]["path"] == str(tmp_path / "data/owner/task")
    assert legacy[0]["repo_id"] == "owner/task"
    sources = resolve_sources({"sources": [{"path": "a"}, {"path": str(tmp_path / "b")}]}, base_dir=tmp_path)
    assert [s["path"] for s in sources] == [str(tmp_path / "data/a"), str(tmp_path / "b")]


@pytest.mark.parametrize("data", [
    {}, {"repo_id": "a", "sources": [{"path": "b"}]}, {"sources": []},
    {"sources": [{"path": "a", "weight": 1}, {"path": "b"}]},
    {"sources": [{"path": "a", "weight": float("nan")}]},
    {"sources": [{"path": "a", "weight": True}]},
    {"sources": [{"path": "a"}, {"path": "a"}]},
    {"repo_id": "../a"}, {"sources": [{"path": "a", "repo_id": "b"}]},
    {"sources": [{"path": "a"}], "image_size": [0, 224]},
])
def test_invalid_sources(data, tmp_path):
    with pytest.raises(ValueError):
        resolve_sources(data, base_dir=tmp_path)


def test_proportional_and_weighted_episode_mixtures():
    assert repeat_plan([[3, 3], [3, 3]]) == [[1, 1], [1, 1]]
    index = MixtureIndex([[3, 3], [3, 3]], [3, 1], seed=7)
    assert index.proportions == [0.75, 0.25]
    assert len(index) == 24
    assert [index.locate(i)[0] for i in range(len(index))].count(0) == 18
    assert index.locate(23) == (1, 5, 7)
    assert index.rows == MixtureIndex([[3, 3], [3, 3]], [3, 1], seed=7).rows
    assert repeat_plan([[2] * 10, [3] * 10], [2.2, 1], 5) == repeat_plan([[2] * 10, [3] * 10], [2.2, 1], 5)
    with pytest.raises(IndexError):
        index.locate(24)


def test_resize_preserves_aspect_ratio_and_dtype():
    image = np.full((4, 8, 3), 255, np.uint8)
    result = resize_rgb(image, (8, 8))
    assert result.shape == (8, 8, 3)
    assert not result[:2].any()
    assert result[2:6].min() == 255
    floating = resize_rgb(image.astype(np.float32) / 255, (8, 8))
    np.testing.assert_array_equal(floating, result / 255)
    assert image_size({"image_size": [8, 8]}) == (8, 8)
