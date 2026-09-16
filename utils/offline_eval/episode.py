"""Load a native xspark / RoboDojo v1.0 episode and slice one time index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.utils.data_loader import load

EPISODE_SUFFIXES = (".hdf5", ".h5")
_TIME_GROUPS = ("vision", "state", "action", "tactile")
_STATIC_SHAPES = frozenset({(2,), (3,), (3, 3), (4, 4)})


def resolve_episode_paths(data_path: str | Path) -> list[Path]:
    path = Path(data_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in EPISODE_SUFFIXES:
            raise ValueError(f"not an HDF5 episode file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in EPISODE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"no .hdf5 / .h5 files under {path}")
        return files
    raise FileNotFoundError(f"episode path does not exist: {path}")


def default_data_dir(bench_name: str, task_name: str, env_cfg_type: str) -> Path:
    """``<parent-of-checkout>/data/<bench>/<task>/<env_cfg>/data``."""
    checkout = Path(__file__).resolve().parents[2]
    return checkout.parent / "data" / bench_name / task_name / env_cfg_type / "data"


def _vision_length(vision: Any) -> int | None:
    if not isinstance(vision, dict):
        return None
    for camera in vision.values():
        if isinstance(camera, dict):
            value = camera.get("colors", camera.get("color"))
        else:
            value = camera
        length = _sequence_length(value, prefer_image_stack=True)
        if length is not None:
            return length
    return None


def _sequence_length(value: Any, *, prefer_image_stack: bool = False) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value) if value else None
    if not isinstance(value, np.ndarray):
        return None
    if value.dtype.kind in {"S", "U", "O"}:
        return int(value.shape[0]) if value.ndim >= 1 else 1
    if prefer_image_stack:
        if value.ndim == 4:
            return int(value.shape[0])
        if value.ndim == 2 and value.dtype == np.uint8:
            return int(value.shape[0])
        if value.ndim == 1:
            return None
    if tuple(value.shape) in _STATIC_SHAPES:
        return None
    if value.ndim >= 2:
        return int(value.shape[0])
    return None


def _collect_group_lengths(node: Any, lengths: list[int]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _collect_group_lengths(value, lengths)
        return
    length = _sequence_length(node)
    if length is not None:
        lengths.append(length)


def infer_episode_length(traj: dict[str, Any]) -> int:
    vision_length = _vision_length(traj.get("vision"))
    if vision_length is not None:
        return vision_length

    lengths: list[int] = []
    for name in _TIME_GROUPS:
        group = traj.get(name)
        if isinstance(group, dict):
            _collect_group_lengths(group, lengths)
    state = traj.get("state")
    if isinstance(state, dict) and isinstance(state.get("mobile"), dict):
        _collect_group_lengths(state["mobile"], lengths)
    if not lengths:
        raise ValueError("cannot infer episode length: no time-series arrays in trajectory")
    return max(lengths)


def slice_time_tree(node: Any, index: int, episode_len: int) -> Any:
    """Take time ``index`` wherever an array's first axis equals ``episode_len``."""
    if isinstance(node, dict):
        return {key: slice_time_tree(value, index, episode_len) for key, value in node.items()}
    if isinstance(node, np.ndarray):
        # (3, 3) / (4, 4) camera matrices stay whole even when T equals 3 or 4.
        if tuple(node.shape) in _STATIC_SHAPES:
            return node
        if node.ndim >= 1 and node.shape[0] == episode_len:
            return node[index]
        return node
    if isinstance(node, (list, tuple)) and len(node) == episode_len:
        return node[index]
    return node


class EpisodeSource:
    """One HDF5 episode: raw trajectory, strided control steps, GT actions."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_type: str = "xspark",
        data_version: str = "v1.0",
        frame_stride: int = 1,
        traj: dict[str, Any] | None = None,
    ):
        if traj is None and path is None:
            raise ValueError("EpisodeSource requires path or traj")
        self.path = Path(path) if path is not None else Path(".")
        self.raw = traj if traj is not None else load(
            str(self.path), data_type=data_type, data_version=data_version
        )
        self.episode_len = infer_episode_length(self.raw)
        self.frame_stride = max(1, int(frame_stride))
        self.indices = list(range(0, self.episode_len, self.frame_stride))
        if not self.indices:
            raise ValueError(f"episode has no frames after stride={self.frame_stride}: {self.path}")

    @property
    def num_steps(self) -> int:
        return len(self.indices)

    def source_index(self, step: int) -> int:
        if not self.indices:
            return 0
        return self.indices[min(max(step, 0), len(self.indices) - 1)]

    def frame(self, step: int) -> dict[str, Any]:
        return slice_time_tree(self.raw, self.source_index(step), self.episode_len)

    def gt_action(self, step: int) -> dict[str, Any] | None:
        action = self.raw.get("action")
        if not isinstance(action, dict) or not action:
            return None
        return slice_time_tree(action, self.source_index(step), self.episode_len)
