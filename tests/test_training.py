"""Common training fields retain their meaning across framework adapters."""

import copy
import subprocess
import sys

import pytest

from g2_policy_tools.training import from_native, to_native


@pytest.fixture
def source():
    return {
        "schema_version": 7,
        "backend": "lerobot",
        "experiment": {"name": "test", "arm": "right"},
        "data": {"root": "data", "repo_id": "org/task", "cameras": None},
        "policy": {"timing": {"action_horizon": 50}},
        "model": {"type": "pi05", "pretrained_path": "base"},
        "training": {
            "batch_size": 8,
            "steps": 10000,
            "num_workers": 4,
            "seed": 42,
            "log_interval": 50,
            "save_interval": 1000,
            "learning_rate": 2.5e-5,
        },
        "runtime": {"gpus": [0], "resume": False},
        "wandb": {"mode": "off", "project": "test"},
        "backend_options": {"lerobot": {"policy": {"gradient_checkpointing": True}}},
    }


def test_common_fields_map_without_mutating_source(source):
    original = copy.deepcopy(source)
    native = to_native(source, "lerobot")
    assert native["lerobot"]["steps"] == 10000
    assert native["lerobot"]["policy"]["optimizer_lr"] == 2.5e-5
    assert native["lerobot"]["policy"]["chunk_size"] == 50
    assert native["lerobot"]["policy"]["gradient_checkpointing"] is True
    assert native["lerobot"]["wandb"]["enable"] is False
    assert from_native(native, "lerobot") == source
    assert source == original


def test_openpi_snapshot_round_trips_all_training_settings(source):
    source["backend"] = "openpi"
    source["backend_options"] = {
        "openpi": {
            "training": {"warmup_steps": 1000, "keep_period": None},
            "model": {"discrete_state_input": True},
            "checkpoints": {"base_dir": "checkpoints", "assets_dir": "assets", "overwrite": False},
            "runtime": {"mem_fraction": 0.9},
        }
    }
    native = to_native(source, "openpi")
    assert native["training"]["num_train_steps"] == 10000
    assert native["training"]["peak_lr"] == 2.5e-5
    assert native["checkpoints"]["resume"] is False
    assert native["model"]["params_path"] == "base"
    assert from_native(native, "openpi") == source


@pytest.mark.parametrize("version", [None, 5, 6, "7", True])
def test_old_training_schemas_are_rejected(source, version):
    source["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version: 7"):
        to_native(source, "lerobot")


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 10},
        {"batch_size": 4},
        {"policy": {"optimizer_lr": 1e-5}},
        {"policy": {"chunk_size": 10}},
        {"policy": {"pretrained_path": "other"}},
    ],
)
def test_backend_options_cannot_shadow_common_fields(source, options):
    source["backend_options"]["lerobot"] = options
    with pytest.raises(ValueError, match="public|top level"):
        to_native(source, "lerobot")


def test_wrong_backend_is_rejected(source):
    with pytest.raises(ValueError, match="backend must be openpi"):
        to_native(source, "openpi")


def test_logging_cannot_be_hidden_in_backend_options(source):
    del source["wandb"]
    source["backend_options"]["lerobot"]["wandb"] = {"enable": True}
    with pytest.raises(ValueError, match="public G2 fields"):
        to_native(source, "lerobot")


def test_short_resume_does_not_create_a_model(source):
    short = {
        "schema_version": 7,
        "backend": "lerobot",
        "runtime": {"resume": "outputs/test"},
        "training": {"steps": 20000},
    }
    assert to_native(short, "lerobot") == {
        "runtime": {"resume": "outputs/test"},
        "lerobot": {"steps": 20000},
    }


def test_training_translation_does_not_import_frameworks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import g2_policy_tools.training; "
                "assert not {'jax', 'torch', 'lerobot', 'openpi'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )
