"""Trajectory (plural) key names → runtime Observation v1.0 (singular) keys.

Irregular names are looked up first. Everything else that looks like a
time-series plural (trailing ``s``, but not ``ss``) drops the suffix.
Metadata keys such as ``instructions`` are left alone.
"""

from __future__ import annotations

from typing import Any

# Trajectory → observation. These are not a trailing-s pluralization.
IRREGULAR_TRAJ_TO_OBS = {
    "colors": "color",
    "depths": "depth",
    "extrinsic_matrix": "extrinsics_matrix",
    "extrinsics_matrices": "extrinsics_matrix",
    "intrinsic_matrices": "intrinsic_matrix",
}

# Look like they end in ``s`` but are not time-series field names.
_DO_NOT_STRIP = frozenset(
    {
        "instructions",
        "subtasks",
        "additional_info",
    }
)


def traj_key_to_obs_key(key: str) -> str:
    if key in IRREGULAR_TRAJ_TO_OBS:
        return IRREGULAR_TRAJ_TO_OBS[key]
    if key in _DO_NOT_STRIP:
        return key
    if key.endswith("s") and not key.endswith("ss"):
        return key[:-1]
    return key


def rename_traj_tree(node: Any) -> Any:
    """Rename keys in a nested dict; non-dicts are returned unchanged."""
    if not isinstance(node, dict):
        return node
    return {traj_key_to_obs_key(key): rename_traj_tree(value) for key, value in node.items()}
