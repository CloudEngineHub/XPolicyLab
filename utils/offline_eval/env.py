"""Offline TASK_ENV: replay native Trajectory v1.0 through the policy deploy loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.utils.process_data import get_robot_action_dim_info

from XPolicyLab.utils.offline_eval.episode import (
    EpisodeSource,
    default_data_dir,
    resolve_episode_paths,
)
from XPolicyLab.utils.offline_eval.metrics import (
    compare_episode,
    save_traj_npz,
    validate_pred_action,
)
from XPolicyLab.utils.offline_eval.trajectory_to_obs import episode_obs


class _DryRunClient:
    def call(self, **_kwargs):
        raise RuntimeError("dry-run-obs: no policy server")

    def close(self) -> None:
        return None


class OfflineEnv:
    """Same surface as ``utils.debug_env_client.TestEnv``, backed by HDF5 episodes."""

    def __init__(self, deploy_cfg: dict[str, Any], *, connect: bool = True):
        self.success_num, self.episode_num = 0, 0
        self._stop_check = None
        self.deploy_cfg = deploy_cfg
        self.obs_encoded = bool(deploy_cfg.get("obs_encoded", False))
        self.action_type = str(deploy_cfg.get("action_type") or "joint").strip().lower()
        if self.action_type not in {"joint", "ee"}:
            raise ValueError(f"unsupported action_type: {self.action_type!r}")
        self.max_steps = deploy_cfg.get("max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.frame_stride = max(1, int(deploy_cfg.get("frame_stride") or 1))
        self.data_type = deploy_cfg.get("data_type") or "xspark"
        self.data_version = deploy_cfg.get("data_version") or "v1.0"
        self.save_dir = Path(deploy_cfg.get("save_dir") or "offline_eval_results")

        if deploy_cfg.get("robot_action_dim_info") is not None:
            self.robot_action_dim_info = deploy_cfg["robot_action_dim_info"]
        else:
            self.robot_action_dim_info = get_robot_action_dim_info(deploy_cfg["env_cfg_type"])

        data_path = deploy_cfg.get("data_path")
        if not data_path:
            data_path = default_data_dir(
                deploy_cfg["bench_name"],
                deploy_cfg["task_name"],
                deploy_cfg["env_cfg_type"],
            )
        self.episode_paths = resolve_episode_paths(data_path)
        self.episode_idx = 0
        self.episode_step = 0
        self.pred_actions: list[dict[str, Any]] = []
        self.gt_actions: list[dict[str, Any] | None] = []
        self.last_metrics: dict[str, Any] | None = None
        self._load_current_episode()

        if connect:
            self.model_client = self._make_model_client(deploy_cfg)
        else:
            self.model_client = _DryRunClient()

    @staticmethod
    def _make_model_client(deploy_cfg: dict[str, Any]):
        if deploy_cfg.get("protocol", "ws") == "ws":
            from client_server.ws import WsModelClient

            policy_server_url = deploy_cfg.get("policy_server_url")
            if policy_server_url is None:
                policy_server_url = f"ws://{deploy_cfg['host']}:{deploy_cfg['port']}"
            return WsModelClient(
                url=policy_server_url,
                evaluation_id=deploy_cfg.get("evaluation_id") or "offline-eval",
                trial_id=deploy_cfg.get("trial_id") or deploy_cfg.get("task_name") or "offline-trial",
                action_case_id=deploy_cfg.get("action_case_id"),
                repeat_index=deploy_cfg.get("repeat_index"),
                ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s", 20.0),
                ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s", 20.0),
                connect_timeout_s=deploy_cfg.get("connect_timeout_s"),
                handshake_timeout_s=deploy_cfg.get("handshake_timeout_s"),
                request_timeout_s=deploy_cfg.get("request_timeout_s"),
                max_connect_attempts=deploy_cfg.get("max_connect_attempts"),
                connect_retry_delay_s=deploy_cfg.get("connect_retry_delay_s"),
                max_connect_seconds=deploy_cfg.get("max_connect_seconds"),
                close_timeout_s=deploy_cfg.get("close_timeout_s"),
            )
        from client_server.tcp.model_client import ModelClient

        return ModelClient(host=deploy_cfg["host"], port=deploy_cfg["port"])

    def _load_current_episode(self) -> None:
        path = self.episode_paths[self.episode_idx]
        self.episode = EpisodeSource(
            path,
            data_type=self.data_type,
            data_version=self.data_version,
            frame_stride=self.frame_stride,
        )
        self.episode_step = 0
        self.pred_actions = []
        self.gt_actions = []
        print(
            f"[OfflineEnv] loaded {path} "
            f"({self.episode.episode_len} raw frames, {self.episode.num_steps} steps, "
            f"stride={self.frame_stride})"
        )

    def set_stop_check(self, stop_check) -> None:
        self._stop_check = stop_check

    def get_obs(self, env_idx=0):
        return episode_obs(
            self.episode,
            self.episode_step,
            env_idx=env_idx,
            obs_encoded=self.obs_encoded,
        )

    def get_obs_batch(self, env_idx_list):
        return [self.get_obs(env_idx) for env_idx in env_idx_list]

    def eval_one_episode(self):
        policy_name = self.deploy_cfg["policy_name"]
        try:
            eval_module = __import__(
                f"XPolicyLab.policy.{policy_name}.deploy",
                fromlist=["eval_one_episode"],
            )
        except ImportError as exc:
            print(
                "[OfflineEnv]",
                f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {exc}",
                "ERROR",
            )
            raise
        if not hasattr(eval_module, "eval_one_episode"):
            raise AttributeError(f"Missing eval_one_episode in {policy_name}.deploy")
        eval_module.eval_one_episode(TASK_ENV=self, model_client=self.model_client)

    def eval_one_episode_batch(self):
        policy_name = self.deploy_cfg["policy_name"]
        try:
            eval_module = __import__(
                f"XPolicyLab.policy.{policy_name}.deploy",
                fromlist=["eval_one_episode_batch"],
            )
        except ImportError as exc:
            print(
                "[OfflineEnv]",
                f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {exc}",
                "ERROR",
            )
            raise
        if not hasattr(eval_module, "eval_one_episode_batch"):
            raise AttributeError(f"Missing eval_one_episode_batch in {policy_name}.deploy")
        eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

    def reset(self) -> None:
        if self.model_client is not None:
            self.model_client.call(func_name="reset")
        self.episode_step = 0
        self.pred_actions = []
        self.gt_actions = []

    def take_action(self, action) -> None:
        limit = self._step_limit()
        print(f"[OfflineEnv] Action Step: {self.episode_step} / {limit}")
        validate_pred_action(action, self.action_type, self.robot_action_dim_info)
        self.pred_actions.append(action)
        self.gt_actions.append(self.episode.gt_action(self.episode_step))
        self.episode_step += 1

    def take_action_batch(self, action_list, env_idx_list) -> None:
        if len(action_list) != len(env_idx_list):
            raise ValueError(
                f"action num != env num: {len(action_list)} != {len(env_idx_list)}"
            )
        for action in action_list:
            validate_pred_action(action, self.action_type, self.robot_action_dim_info)
        # Replay is single-episode; record the first env's action and advance once.
        self.take_action(action_list[0])

    def _step_limit(self) -> int:
        if self.max_steps is None:
            return self.episode.num_steps
        return min(int(self.max_steps), self.episode.num_steps)

    def is_episode_end(self) -> bool:
        if self._stop_check is not None and self._stop_check():
            print("[OfflineEnv] Check Episode End: stop requested")
            return True
        ended = self.episode_step >= self._step_limit()
        print("[OfflineEnv] Check Episode End:", ended)
        return ended

    def finish_episode(self) -> None:
        self.episode_num += 1
        result = compare_episode(
            self.pred_actions,
            self.gt_actions,
            self.action_type,
            self.robot_action_dim_info,
        )
        self.last_metrics = result
        policy_name = self.deploy_cfg.get("policy_name") or "policy"
        stem = self.episode.path.stem
        npz_path = self.save_dir / policy_name / f"{stem}_traj.npz"
        save_traj_npz(
            npz_path,
            result,
            extra={"episode_path": np.asarray(str(self.episode.path))},
        )
        if result["compared_steps"]:
            print(
                f"[OfflineEnv] Episode finished mae={result['mae']:.6g} "
                f"l2={result['l2']:.6g} steps={result['compared_steps']} npz={npz_path}"
            )
        else:
            print(
                f"[OfflineEnv] Episode finished (no GT comparison) npz={npz_path}"
            )

        self.episode_idx = (self.episode_idx + 1) % len(self.episode_paths)
        self._load_current_episode()

    def get_running_env_idx_list(self):
        return [0]

    def close(self) -> None:
        close = getattr(self.model_client, "close", None)
        if callable(close):
            close()
