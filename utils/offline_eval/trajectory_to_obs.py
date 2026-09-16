"""Convert one Trajectory v1.0 time index into Observation v1.0."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit

from XPolicyLab.utils.offline_eval.episode import (
    EpisodeSource,
    infer_episode_length,
    slice_time_tree,
)
from XPolicyLab.utils.offline_eval.keys import rename_traj_tree

_TOP_LEVEL_SKIP = frozenset(
    {
        "action",
        "subtasks",
        "vision",
        "state",
        "data_format_version",
        "instruction",
        "instructions",
        "additional_info",
        "env_idx",
    }
)


def _as_hw_shape(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    shape = np.asarray(value).reshape(-1)
    if shape.size < 2:
        return None
    return int(shape[0]), int(shape[1])


def _finalize_camera(camera: Any) -> dict[str, Any]:
    if not isinstance(camera, dict):
        color = decode_image_bit(camera)
        out: dict[str, Any] = {"color": color}
        if isinstance(color, np.ndarray) and color.ndim >= 2:
            out["shape"] = (int(color.shape[0]), int(color.shape[1]))
        return out

    out = dict(camera)
    if "colors" in out and "color" not in out:
        out["color"] = out.pop("colors")
    if "color" in out:
        out["color"] = decode_image_bit(out["color"])
        color = out["color"]
        if isinstance(color, np.ndarray) and color.ndim >= 2 and "shape" not in out:
            out["shape"] = (int(color.shape[0]), int(color.shape[1]))
    shape = _as_hw_shape(out.get("shape"))
    if shape is not None:
        out["shape"] = shape
    return out


def _finalize_vision(vision: Any) -> dict[str, Any]:
    if not isinstance(vision, dict):
        return {}
    return {name: _finalize_camera(camera) for name, camera in vision.items()}


def encode_obs_colors(obs: dict[str, Any]) -> dict[str, Any]:
    """Re-encode ``vision/*/color`` so the server-side decode path is exercised."""
    vision = obs.get("vision")
    if not isinstance(vision, dict):
        return obs
    for index, camera in enumerate(vision.values()):
        if not isinstance(camera, dict) or "color" not in camera:
            continue
        color = camera["color"]
        if not (isinstance(color, np.ndarray) and color.ndim == 3 and color.shape[-1] == 3):
            continue
        encoded = encode_image_bit(color)
        camera["color"] = encoded if index % 2 else np.frombuffer(encoded, np.uint8)
    return obs


def trajectory_to_obs(
    traj: dict[str, Any],
    index: int,
    *,
    env_idx: int = 0,
    episode_len: int | None = None,
    obs_encoded: bool = False,
) -> dict[str, Any]:
    """Slice trajectory time ``index`` and emit a runtime Observation v1.0 dict.

    ``action/`` and ``subtasks`` are never copied. Extra top-level groups
    (for example ``tactile/``) are sliced and key-renamed, not reshaped.
    """
    length = infer_episode_length(traj) if episode_len is None else int(episode_len)
    if index < 0 or index >= length:
        raise IndexError(f"trajectory index {index} out of range for length {length}")

    frame = slice_time_tree(traj, index, length)
    obs: dict[str, Any] = {
        "data_format_version": frame.get("data_format_version") or "v1.0",
        "env_idx": int(env_idx),
    }
    if "instruction" in frame:
        obs["instruction"] = frame["instruction"]
    if "instructions" in frame:
        obs["instructions"] = frame["instructions"]
    if "additional_info" in frame:
        obs["additional_info"] = frame["additional_info"]

    if isinstance(frame.get("vision"), dict):
        obs["vision"] = _finalize_vision(rename_traj_tree(frame["vision"]))
    if isinstance(frame.get("state"), dict):
        obs["state"] = rename_traj_tree(frame["state"])

    for key, value in frame.items():
        if key in _TOP_LEVEL_SKIP or not isinstance(value, dict):
            continue
        obs[key] = rename_traj_tree(value)

    if obs_encoded:
        encode_obs_colors(obs)
    return obs


def episode_obs(
    episode: EpisodeSource,
    step: int,
    *,
    env_idx: int = 0,
    obs_encoded: bool = False,
) -> dict[str, Any]:
    return trajectory_to_obs(
        episode.raw,
        episode.source_index(step),
        env_idx=env_idx,
        episode_len=episode.episode_len,
        obs_encoded=obs_encoded,
    )
