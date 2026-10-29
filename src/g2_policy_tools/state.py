"""Model-independent G2 observation encoding."""

import numpy as np

from .geometry import encode_poses, joint_delta, source_joint


def encode_state(policy, observation, arm="both", *, flatten=True):
    """Encode canonical histories; references are fixed at observation time.

    ``flatten=False`` preserves (history, dimensions) for temporal policies.
    """
    history = policy["timing"]["history_steps"]
    states = []
    for kind, width in (("pose", 20), ("joint", 16)):
        spec = policy["state"][kind]
        if not spec["enabled"]:
            continue
        values = np.asarray(observation.get(f"observation/{kind}_history", observation[f"observation/{kind}"]))
        if values.ndim == 1:
            values = np.tile(values, (history, 1))
        if values.shape != (history, width):
            raise ValueError(f"{kind} history must have shape {(history, width)}")
        if kind == "pose":
            values = encode_poses(values, observation, spec)
        elif spec["delta"]:
            values = joint_delta(
                values, source_joint(observation, spec["reference"]["source"]), spec, inverse=False
            )
        selection = (
            slice(None)
            if arm == "both"
            else slice(0, values.shape[-1] // 2)
            if arm == "left"
            else slice(values.shape[-1] // 2, None)
        )
        states.append(values[..., selection])
    result = np.concatenate(states, axis=-1).astype(np.float32)
    return result.reshape(-1) if flatten else result


def references_from_sample(sample):
    """Map disk/sampler references to the shared geometry dictionary."""
    sources = {
        "state.pose": "pose",
        "state.joint": "joint",
        "g2.pose_history": "pose_history",
        "g2.joint_history": "joint_history",
        "g2.command_pose": "command_pose",
        "g2.command_joint": "command_joint",
        "reference.base_to_parent": "base_to_parent",
    }
    return {f"observation/{target}": np.asarray(sample[source]) for source, target in sources.items() if source in sample}
