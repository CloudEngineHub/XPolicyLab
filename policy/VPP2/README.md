# VPP2 — RoboDojo Adapter

**Contributor:** [Haodong Yan](https://github.com/Haodong-Yan) | **Paper:** Video Prediction Policy 2: Predict Better, Act Better | **arXiv:** [2610.10270](https://arxiv.org/abs/2610.10270) | **Original code:** [Official VPP2 repository](https://github.com/roboterax/video-prediction-policy-2)

**Project page:** [Video Prediction Policy 2](https://robert-gyj.github.io/video-prediction-policy-2/)

VPP2 combines a pretrained video model with an action expert for robot control.
This adapter supports training and evaluation of the joint + 2B 100k policy on
**RoboDojo / arx_x5 / absolute EE16**.
The model and trainer are installed from a pinned public VPP2 revision in
`upstream/`. Small local entry points reuse its training code and configuration.
The [official repository](https://github.com/roboterax/video-prediction-policy-2)
also includes **LIBERO, LIBERO-OOD and LIBERO-PRO**.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

From a RoboDojo workspace with its `env_cfg/` beside the XPolicyLab checkout:

```bash
cd XPolicyLab/policy/VPP2
bash install.sh vpp2
conda activate vpp2
bash download_checkpoints.sh
python launch_policy.py --dry-run
```

The installer pins the official code at `ae9afab`, uses Python 3.10 and defaults
to PyTorch 2.11 / CUDA 13.0. Select a compatible driver/build with
`TORCH_VERSION`, `TORCHVISION_VERSION` and `TORCH_CUDA` if necessary.
The reference GPU is one 96 GiB RTX PRO 6000; smaller devices are unverified.
Install the simulator separately using the RoboDojo instructions.

For a new training environment, use `bash install.sh vpp2 train` instead.
To add training dependencies to an existing policy environment:

```bash
python -m pip install -r upstream/requirements-train.txt -c upstream/environment-reference.txt
```

## Data Processing

Start with the public [RoboDojo EE16 LeRobot v3.0 export](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo_ee_lerobot_v30_video).
`process_data.sh convert` calls the official VPP2 converter to generate one
native-frame EE16 Parquet and RGB T-shaped video per episode. It preserves the
source action/state alignment and uses each camera's own episode timestamps.
The result is a custom full-episode layout, not XPolicyLab's generic LeRobot
v2.1/v3.0 converter output. Joint14 and raw HDF5 inputs are not supported.

From `policy/VPP2` in the activated policy environment:

```bash
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --include 'data/RoboDojo_ee_lerobot_v30_video/**' --local-dir /data/robodojo_download
export VPP2_SOURCE=/data/robodojo_download/data/RoboDojo_ee_lerobot_v30_video
export VPP2_MEDIA_ROOT="$PWD/data/robodojo_source"
export VPP2_PREPARED="$PWD/data/robodojo"
bash process_data.sh convert --source "$VPP2_SOURCE" --output "$VPP2_MEDIA_ROOT" --workers 4
bash process_data.sh prepare \
  --metadata "$VPP2_MEDIA_ROOT/full_episode_metadata.csv" \
  --media-root "$VPP2_MEDIA_ROOT" --output "$VPP2_PREPARED"
```

Use fresh output directories. Conversion requires `ffmpeg` with `libx264`,
3500 episodes / 1,856,102 frames at 25 Hz, and the released task inventory.
Preparation builds the fixed 3466/34 split and critical sampling index, using
the supplied normalization statistics. See the [data guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/training.md#convert-the-public-dataset)
for camera composition and a small conversion smoke test.

## Training

The recipe is **history-conditioned Video-10k → joint Video + fresh Action2B**,
with **global batch size 288** and **100,000 steps**. The local
[`train_100k.yaml`](train_100k.yaml) inherits the official configuration and only
sets adapter-local paths. Scripts accept the same arguments as the
[official RoboDojo entry points](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/robodojo.md).
Use absolute input/output paths; other relative paths resolve from `upstream/`.

After data preparation, download the initializer and shared encoders, cache the
text prompts, then initialize Action2B:

```bash
bash download_checkpoints.sh --stage train
export VPP2_WAN_ROOT="$PWD/checkpoints/Wan2.1-I2V-14B-480P"
export VPP2_VIDEO_INIT="$PWD/checkpoints/initialization/robodojo_his10k.pt"
export VPP2_ACTION_INIT="$PWD/checkpoints/action2b_init.pt"
bash text_cache.sh --data "$VPP2_PREPARED" --wan "$VPP2_WAN_ROOT"
bash init_action.sh --video "$VPP2_VIDEO_INIT" --output "$VPP2_ACTION_INIT"
```

Set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and
`REQUIRE_RDMA` for your runtime. Adjust `batch_size` and
`gradient_accumulation_steps` to retain global batch 288. Check live GPU
processes and tmux sessions before text caching or training. Run the training
launcher on every GPU-bearing role, sharing paths and the output directory.

```bash
export VPP2_OUTPUT_DIR="$PWD/runs/robodojo_joint2b_100k"
bash train.sh --dry-run
bash train.sh
# Resume full training state after an interruption:
bash train.sh resume="$VPP2_OUTPUT_DIR/checkpoints/state/step_090000"
```

`--dry-run` checks the configuration without allocating GPUs. Use a fresh
output directory for a new run. For the required short preflight before a
formal run, follow the [training guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/robodojo.md#3-train-joint-0100k).
Export the final paired weights to the adapter's named-bundle layout:

```bash
bash export.sh \
  --checkpoint "$VPP2_OUTPUT_DIR/checkpoints/weights/step_100000.pt" \
  --config "$VPP2_OUTPUT_DIR/config.yaml" \
  --stats "$VPP2_PREPARED/dataset_stats.json" \
  --output "$PWD/checkpoints/my_joint2b_s100000"
bash eval.sh RoboDojo stack_bowls my_joint2b_s100000 arx_x5 ee 1 0 1 vpp2 RoboDojo
```

The exporter refuses an existing destination. Keep `action.pt`, `video.pt`,
`dataset_stats.json` and `manifest.json` together; the existing checkpoint
resolver accepts the bundle name in `eval.sh`.

## Evaluation

```bash
cd XPolicyLab/policy/VPP2
bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 1 vpp2 RoboDojo

# Wiring checks using real model weights, without a simulator:
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
```

Replace `RoboDojo` with your simulator conda environment name. Both environment
names and absolute conda prefixes are accepted. Debug checks verify wiring and
action shapes; they do not measure task success. For separate policy and
simulator machines, use the [shared deployment flow](../../README.md#-deployment-flow).

## Model Assets

The downloader defaults to public [Hugging Face](https://huggingface.co/Haodong082399/VPP2).
Use `--stage train` for his10k plus the shared encoders, or `--stage all` for
both training and evaluation assets:

```bash
bash download_checkpoints.sh                      # evaluation bundle + encoders
bash download_checkpoints.sh --stage train         # his10k + encoders
bash download_checkpoints.sh --source modelscope   # alternative mirror
```

[ModelScope](https://modelscope.cn/models/haodong123/VPP2_preview) requires an
authorized account; use its SDK login or set `MODELSCOPE_API_TOKEN` privately.
The upstream robot-video pretrained checkpoint remains a future release;
training here starts from the public his10k initializer.

```text
checkpoints/
├── joint2b_s100000/
│   ├── action.pt
│   ├── video.pt
│   ├── dataset_stats.json
│   └── manifest.json
└── Wan2.1-I2V-14B-480P/
    ├── Wan2.1_VAE.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    └── google/umt5-xxl/...
```

This named bundle is supported by the shared checkpoint resolver, alongside its
conventional run-directory layout and explicit paths. Keep its paired video,
action and normalization files together. `VPP2_BUNDLE` and `VPP2_WAN_ROOT` can
point to existing asset directories; `launch_policy.py --dry-run` checks file
sizes, manifest step and encoder inventory without allocating a GPU.

## Configuration

`deploy.yml` records the released defaults:

| Setting | Value |
| --- | --- |
| Action inference | 10 Euler steps, sigma shift 1, seed 1 |
| Predicted / executed actions | 32 / 24 |
| Video context | History 8, native stride 25, episode anchor |
| RGB input | Native views → T-shaped composition → 240 × 416 |
| Action representation | Absolute poses `[x,y,z,qw,qx,qy,qz]` and gripper, left then right |
| Normalization | Checkpoint z-score statistics, including grippers |
| Precision | `device: cuda`, `mixed_precision: bf16` |

The model runtime performs camera composition and resizing. The adapter accepts
RGB arrays already decoded by XPolicyLab. `deployment_adapter: robodojo_ee16`,
`action_dim: 16`, `action_horizon: 32`, `history_stride: 25` and
`video_seed_offset: 1000003` describe this checkpoint contract.
`checkpoint_path`, `video_checkpoint_path`, `dataset_stats_path` and
`wan_model_dir` default to the bundle above. `default_instruction` is used only
when an observation supplies no instruction.

For diagnostic comparisons, launch scripts accept `VPP2_NUM_INFERENCE_STEPS`,
`VPP2_SIGMA_SHIFT` and `VPP2_REPLAN_STEPS`; these override
`num_inference_steps`, `sigma_shift` and `replan_steps` respectively.
Changing them changes the evaluation setting.

Observation history is recorded once per action and isolated by client ID.
Keep `eval_batch: false`: vectorized environment batch methods are explicitly
unsupported. Multiple independent clients can share one policy server.
