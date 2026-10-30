"""Translate the public G2 training schema at framework boundaries only."""

import copy

VERSION = 7
TRAINING_FIELDS = {
    "batch_size": ("batch_size", "batch_size"),
    "steps": ("num_train_steps", "steps"),
    "num_workers": ("num_workers", "num_workers"),
    "seed": ("seed", "seed"),
    "log_interval": ("log_interval", "log_freq"),
    "save_interval": ("save_interval", "save_freq"),
    "learning_rate": ("peak_lr", "optimizer_lr"),
}
PUBLIC_FIELDS = {
    "schema_version",
    "backend",
    "experiment",
    "data",
    "policy",
    "model",
    "training",
    "runtime",
    "wandb",
    "backend_options",
}
OPENPI_OPTIONS = {
    "finetuning": None,
    "run": {"smoke"},
    "model": {"discrete_state_input"},
    "training": {"fsdp_devices", "keep_period", "warmup_steps", "decay_steps", "decay_lr"},
    "checkpoints": {"base_dir", "assets_dir", "overwrite"},
    "normalization": {"mode", "max_frames"},
    "runtime": {"mem_fraction"},
}


def _fields(value, allowed, path):
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a YAML mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown YAML field: {path}.{min(unknown)}")


def to_native(source, backend):
    """Lower public fields without importing or initializing either training framework.

    Framework parsers own value validation. This boundary checks structure and field
    ownership so backend options cannot shadow a public setting.
    """
    _fields(source, PUBLIC_FIELDS, "settings")
    if source.get("schema_version") != VERSION:
        raise ValueError(f"G2 training requires schema_version: {VERSION}")
    if source.get("backend") != backend:
        raise ValueError(f"backend must be {backend} for this training entry point")
    for section, allowed in (
        ("model", {"type", "pretrained_path"}),
        ("training", TRAINING_FIELDS.keys()),
        ("runtime", {"gpus", "resume"}),
        ("wandb", {"mode", "project"}),
        ("backend_options", {backend}),
    ):
        _fields(source.get(section, {}), allowed, section)
    options = source.get("backend_options", {}).get(backend, {})
    if backend == "openpi":
        _fields(options, OPENPI_OPTIONS.keys(), "backend_options.openpi")
        for section, allowed in OPENPI_OPTIONS.items():
            if allowed is not None:
                _fields(options.get(section, {}), allowed, f"backend_options.openpi.{section}")
        model_type = source.get("model", {}).get("type", "pi05")
        if model_type != "pi05":
            raise ValueError("The OpenPI G2 training entry point supports model.type: pi05")
        result = copy.deepcopy(options)
        result["schema_version"] = VERSION
        for section in ("experiment", "data", "policy"):
            if section in source:
                result[section] = copy.deepcopy(source[section])
        training = result.setdefault("training", {})
        for key, value in source.get("training", {}).items():
            training[TRAINING_FIELDS[key][0]] = value
        if "pretrained_path" in source.get("model", {}):
            result.setdefault("model", {})["params_path"] = source["model"]["pretrained_path"]
        runtime = source.get("runtime", {})
        if "gpus" in runtime:
            result.setdefault("runtime", {})["gpus"] = copy.deepcopy(runtime["gpus"])
        if "resume" in runtime:
            result.setdefault("checkpoints", {})["resume"] = runtime["resume"]
        if "wandb" in source:
            result["wandb"] = copy.deepcopy(source["wandb"])
        return result

    if not isinstance(options, dict):
        raise TypeError("backend_options.lerobot must be a YAML mapping")
    reserved = {names[1] for names in TRAINING_FIELDS.values()} | {"resume", "output_dir", "job_name"}
    if reserved & options.keys():
        raise ValueError("Common training fields belong at the top level, not backend_options.lerobot")
    native = copy.deepcopy(options)
    policy = native.setdefault("policy", {})
    if not isinstance(policy, dict):
        raise TypeError("backend_options.lerobot.policy must be a YAML mapping")
    if {"type", "pretrained_path", "optimizer_lr", "chunk_size"} & policy.keys():
        raise ValueError("Set model, learning_rate and action_horizon in the public G2 fields")
    policy.update(copy.deepcopy(source.get("model", {})))
    for key, value in source.get("training", {}).items():
        if key == "learning_rate":
            policy["optimizer_lr"] = value
        else:
            native[TRAINING_FIELDS[key][1]] = value
    if "action_horizon" in source.get("policy", {}).get("timing", {}):
        policy["chunk_size"] = source["policy"]["timing"]["action_horizon"]
    if not policy:
        del native["policy"]
    if {"enable", "project"} & native.get("wandb", {}).keys():
        raise ValueError("Set wandb.mode/project in the public G2 fields")
    if "wandb" in source:
        logging = copy.deepcopy(source["wandb"])
        if "mode" in logging:
            mode = logging.pop("mode")
            if mode not in ("auto", "on", "off"):
                raise ValueError("wandb.mode must be auto, on or off")
            logging["enable"] = mode != "off"
        native.setdefault("wandb", {}).update(logging)
    result = {key: copy.deepcopy(source[key]) for key in ("runtime", "experiment", "data", "policy") if key in source}
    if "experiment" in source:
        result["adapter"] = {"type": "g2"}
    result["lerobot"] = native
    return result


def from_native(settings, backend):
    """Serialize framework settings as a self-contained public training snapshot."""
    native = copy.deepcopy(settings)
    result = {"schema_version": VERSION, "backend": backend}
    for section in ("experiment", "data", "policy"):
        if section in native:
            result[section] = native.pop(section)
    runtime = native.pop("runtime", {})
    result["runtime"] = {key: runtime.pop(key) for key in ("gpus", "resume") if key in runtime}
    training = {}
    if backend == "openpi":
        native.pop("schema_version", None)
        model = native["model"]
        result["model"] = {"type": "pi05", "pretrained_path": model.pop("params_path")}
        for key, names in TRAINING_FIELDS.items():
            if names[0] in native["training"]:
                training[key] = native["training"].pop(names[0])
        result["runtime"]["resume"] = native["checkpoints"].pop("resume")
        native["runtime"] = runtime
        result["wandb"] = native.pop("wandb")
    else:
        native.pop("adapter", None)
        native = native["lerobot"]
        policy = native.get("policy", {})
        model = {key: policy.pop(key) for key in ("type", "pretrained_path") if key in policy}
        if model:
            result["model"] = model
        for key, names in TRAINING_FIELDS.items():
            parent = policy if key == "learning_rate" else native
            if names[1] in parent:
                training[key] = parent.pop(names[1])
        if "chunk_size" in policy:
            result.setdefault("policy", {}).setdefault("timing", {})["action_horizon"] = policy.pop("chunk_size")
        if "wandb" in native:
            logging = native["wandb"]
            result["wandb"] = {"project": logging.pop("project")} if "project" in logging else {}
            if "enable" in logging:
                result["wandb"]["mode"] = "on" if logging.pop("enable") else "off"
            if not logging:
                del native["wandb"]
    result["training"] = training
    result["backend_options"] = {backend: native}
    return result
