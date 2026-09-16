"""Compare predicted action dicts against trajectory ``action/`` via pack_robot_state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.utils.process_data import (
    _get_state_keys,
    _validate_config,
    pack_robot_state,
)


def _with_pack_aliases(state_dict: dict[str, Any], source_type: str) -> dict[str, Any]:
    """Accept README ``arm_joint_state(s)`` as pack_robot_state's single-arm ``joint_state(s)``."""
    aliased = dict(state_dict)
    if source_type == "obs":
        if "arm_joint_state" in aliased and "joint_state" not in aliased:
            aliased["joint_state"] = aliased["arm_joint_state"]
    else:
        if "arm_joint_states" in aliased and "joint_states" not in aliased:
            aliased["joint_states"] = aliased["arm_joint_states"]
        if "arm_joint_state" in aliased and "joint_states" not in aliased:
            aliased["joint_states"] = aliased["arm_joint_state"]
    return aliased


def pack_action(
    action_dict: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict,
    *,
    source_type: str,
    state_type: str,
) -> np.ndarray | None:
    try:
        return pack_robot_state(
            {state_type: _with_pack_aliases(action_dict, source_type)},
            action_type,
            robot_action_dim_info,
            source_type=source_type,
            state_type=state_type,
        )
    except (KeyError, ValueError, TypeError):
        return None


def _check_vector(value: Any, key: str, expected_dim: int) -> None:
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError(f"action[{key!r}] must be array-like, got {type(value)}")
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"action[{key!r}] must be 1D, got shape {arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"action[{key!r}] dim mismatch: expected {expected_dim}, got {arr.shape}"
        )


def validate_pred_action(
    action: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict,
) -> None:
    """Check ``action_type`` keys only. Extra keys (e.g. mobile) are allowed."""
    if not isinstance(action, dict):
        raise TypeError(f"action must be a dict, got {type(action)}")

    arm_dims, ee_dims, num_arms = _validate_config(
        action_type, robot_action_dim_info, "obs"
    )
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, "obs")
    lookup = dict(action)
    if num_arms == 1 and action_type == "joint":
        if "joint_state" not in lookup and "arm_joint_state" in lookup:
            lookup["joint_state"] = lookup["arm_joint_state"]
        forbidden = [key for key in action if key.startswith(("left_", "right_"))]
        if forbidden:
            raise ValueError(
                f"single-arm action should not contain prefixed keys, got: {forbidden}"
            )

    for key, dim in zip(arm_keys, arm_dims):
        if key not in lookup:
            raise KeyError(f"action missing {key!r}")
        _check_vector(lookup[key], key, dim)
    for key, dim in zip(ee_keys, ee_dims):
        if key not in lookup:
            raise KeyError(f"action missing {key!r}")
        _check_vector(lookup[key], key, dim)


def compare_episode(
    pred_actions: list[dict[str, Any]],
    gt_actions: list[dict[str, Any] | None],
    action_type: str,
    robot_action_dim_info: dict,
) -> dict[str, Any]:
    pred_vecs: list[np.ndarray] = []
    gt_vecs: list[np.ndarray] = []
    for pred, gt in zip(pred_actions, gt_actions):
        if not isinstance(pred, dict) or not isinstance(gt, dict):
            continue
        packed_pred = pack_action(
            pred, action_type, robot_action_dim_info, source_type="obs", state_type="state"
        )
        packed_gt = pack_action(
            gt, action_type, robot_action_dim_info, source_type="dataset", state_type="action"
        )
        if packed_pred is None or packed_gt is None:
            continue
        pred_vec = np.asarray(packed_pred, dtype=np.float64).reshape(-1)
        gt_vec = np.asarray(packed_gt, dtype=np.float64).reshape(-1)
        if pred_vec.shape != gt_vec.shape:
            continue
        pred_vecs.append(pred_vec)
        gt_vecs.append(gt_vec)

    if not pred_vecs:
        return {
            "mae": None,
            "l2": None,
            "gt": np.zeros((0,), dtype=np.float64),
            "pred": np.zeros((0,), dtype=np.float64),
            "compared_steps": 0,
        }

    pred = np.stack(pred_vecs, axis=0)
    gt = np.stack(gt_vecs, axis=0)
    diff = pred - gt
    return {
        "mae": float(np.mean(np.abs(diff))),
        "l2": float(np.mean(np.linalg.norm(diff, axis=-1))),
        "gt": gt,
        "pred": pred,
        "compared_steps": int(pred.shape[0]),
    }


def save_traj_npz(path: str | Path, result: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gt": result.get("gt", np.zeros((0,))),
        "pred": result.get("pred", np.zeros((0,))),
        "compared_steps": np.asarray(result.get("compared_steps", 0)),
    }
    if result.get("mae") is not None:
        payload["mae"] = np.asarray(result["mae"])
        payload["l2"] = np.asarray(result["l2"])
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    return path
