#!/usr/bin/env python3
"""Replay native XPolicyLab Trajectory v1.0 episodes against a policy server.

Same ``get_obs`` / ``take_action`` / ``eval_one_episode`` loop as live deploy.
The client only emits Observation v1.0 — it does not build per-policy wire obs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# client_server/ and the XPolicyLab package are both rooted at the checkout;
# insert it so this script also works when invoked directly by path.
_XPOLICYLAB_ROOT = Path(__file__).resolve().parents[1]
if str(_XPOLICYLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPOLICYLAB_ROOT))

from XPolicyLab.utils.offline_eval.env import OfflineEnv


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_additional_info(raw: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not raw:
        return parsed
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _describe_obs(obs: dict) -> str:
    import numpy as np

    vision = {}
    for name, camera in (obs.get("vision") or {}).items():
        color = camera.get("color") if isinstance(camera, dict) else camera
        if hasattr(color, "shape"):
            vision[name] = (
                type(color).__name__,
                tuple(color.shape),
                str(getattr(color, "dtype", "")),
            )
        else:
            vision[name] = type(color).__name__
    state = {
        name: tuple(np.asarray(value).shape)
        for name, value in (obs.get("state") or {}).items()
        if name != "mobile"
    }
    if isinstance((obs.get("state") or {}).get("mobile"), dict):
        state["mobile"] = {
            key: tuple(np.asarray(value).shape)
            for key, value in obs["state"]["mobile"].items()
        }
    extras = sorted(set(obs) - {"env_idx", "instruction", "instructions", "vision", "state"})
    return (
        f"instruction={obs.get('instruction')!r} vision={vision} "
        f"state={state} extras={extras}"
    )


def build_deploy_cfg(args: argparse.Namespace) -> dict:
    extra = parse_additional_info(args.additional_info)
    action_type = args.action_type or extra.get("action_type") or "joint"
    data_path = args.data_path or os.environ.get("XPL_OFFLINE_DATA_PATH") or None
    frame_stride = args.frame_stride
    if frame_stride is None and os.environ.get("XPL_OFFLINE_FRAME_STRIDE"):
        frame_stride = int(os.environ["XPL_OFFLINE_FRAME_STRIDE"])
    return {
        "bench_name": args.bench_name,
        "task_name": args.task_name,
        "env_cfg_type": args.env_cfg_type,
        "policy_name": args.policy_name,
        "action_type": action_type,
        "protocol": args.protocol,
        "host": args.host,
        "port": args.port,
        "policy_server_url": args.policy_server_url,
        "evaluation_id": args.evaluation_id,
        "action_case_id": args.action_case_id,
        "trial_id": args.trial_id or args.task_name,
        "repeat_index": args.repeat_index,
        "eval_episode_num": args.eval_episode_num,
        "eval_batch": args.eval_batch,
        "obs_encoded": args.obs_encoded,
        "data_path": data_path,
        "data_type": args.data_type,
        "frame_stride": frame_stride or 1,
        "max_steps": args.max_steps,
        "save_dir": args.save_dir,
    }


def run_dry_run(deploy_cfg: dict) -> int:
    env = OfflineEnv(deploy_cfg, connect=False)
    try:
        obs = env.get_obs()
        if obs.get("data_format_version") != "v1.0":
            raise AssertionError(f"expected data_format_version v1.0, got {obs.get('data_format_version')!r}")
        for camera in (obs.get("vision") or {}).values():
            if isinstance(camera, dict) and "colors" in camera:
                raise AssertionError("trajectory plural key 'colors' leaked into obs")
        print(f"[offline] {_describe_obs(obs)}", flush=True)
        print("[offline] dry-run-obs OK", flush=True)
    finally:
        env.close()
    return 0


def run_eval(deploy_cfg: dict) -> int:
    env = OfflineEnv(deploy_cfg, connect=True)
    eval_batch = bool(deploy_cfg.get("eval_batch"))
    try:
        for idx in range(int(deploy_cfg["eval_episode_num"])):
            print(f"\033[94m[offline] Running Episode {idx}\033[0m")
            env.reset()
            if not eval_batch:
                env.eval_one_episode()
            else:
                env.eval_one_episode_batch()
            env.finish_episode()
    finally:
        env.close()
    print("[offline] OK", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay native XPolicyLab HDF5 episodes against a policy server"
    )
    parser.add_argument("--bench_name", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--env_cfg_type", type=str, required=True)
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("--action_type", choices=("joint", "ee"), default=None)
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), default="ws")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--policy_server_url", type=str)
    parser.add_argument("--evaluation_id", type=str, default="offline-eval")
    parser.add_argument("--action_case_id", type=str)
    parser.add_argument("--trial_id", type=str)
    parser.add_argument("--repeat_index", type=int)
    parser.add_argument("--eval_episode_num", type=int, default=1)
    parser.add_argument("--eval_batch", type=str2bool, default=False)
    parser.add_argument(
        "--obs_encoded",
        type=str2bool,
        default=os.environ.get("DEBUG_OBS_ENCODED", "0"),
        help="send encoded camera colors to exercise the server-side decode path",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="xspark")
    parser.add_argument("--frame_stride", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="offline_eval_results")
    parser.add_argument("--additional_info", type=str, default=None)
    parser.add_argument(
        "--dry_run_obs",
        action="store_true",
        help="build the first Observation v1.0 and exit without connecting",
    )
    args = parser.parse_args()
    if not args.dry_run_obs and args.port is None and not args.policy_server_url:
        parser.error("--port is required unless --dry_run_obs or --policy_server_url is set")
    if args.port is None:
        args.port = 0
    deploy_cfg = build_deploy_cfg(args)
    if args.dry_run_obs:
        return run_dry_run(deploy_cfg)
    return run_eval(deploy_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
