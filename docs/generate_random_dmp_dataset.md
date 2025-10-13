# `generate_random_dmp_dataset.py`

## Overview
- Samples Planar DMP configurations and executes them inside the PushT environment to build supervision data for conditional flow models.
- Records the intended DMP targets and the full environment interaction as rewards and serialised per-step `info` (instead of saving raw observations).
- Optionally captures rollout videos for quick qualitative inspection. Progress is displayed with `tqdm` to show how many trajectories have finished.

## Runtime Dependencies
- `gymnasium` and `gym-pusht` for environment simulation.
- `imageio` + `imageio-ffmpeg` for video export when `--video-samples > 0`.
- `numpy` for sampling and tensor manipulation.
- `tqdm` for progress bars.
- Project-local modules: `nsf_rl.dmp` (Planar DMP implementation) and `nsf_rl.utils.pymunk_compat` (collision handler patch for modern PyMunk releases).

## Command-Line Interface
```
uv run -- python scripts/generate_random_dmp_dataset.py \
  --num-samples 1000 \
  --seed 42 \
  --output-dir data/random_dmp_npz \
  --video-dir videos/random_dmp_samples \
  --video-samples 10 \
  --scale-forcing
```

Argument summary:
- `--output-dir`: Root directory for NPZ samples and `index.jsonl` (default `data/random_dmp_npz`). Subdir `samples/` is created automatically.
- `--video-dir`: Directory for MP4 rollouts. Only used when `--video-samples > 0`.
- `--num-samples`: Number of trajectories to generate.
- `--video-samples`: Number of trajectories that should receive video captures (starting from index 0, capped at 10 inside the script).
- `--seed`: Seed for `numpy.random.default_rng`, ensuring reproducible DMP parameter draws and environment resets.
- `--scale-forcing`: When set, multiplies the DMP forcing term by the goal displacement to mirror the Marimo explorer toggle.

## Generation Pipeline
1. Initialise RNG and gym environment (`render_mode="rgb_array"` so frames can be captured if needed).
2. For each requested sample (`tqdm` progress bar):
   - Reset the environment with a fresh seed.
   - Sample start/goal locations, duration, stiffness, and RBF weights (now from `[-60, 60]`) and build `DMPConfig`/`DMPParams`.
   - Roll out the DMP with sub-steps, recovering normalized positions, velocities, canonical phase `s`, and the forcing term.
   - Down-sample to the environment control frequency (every `DMP_DT_SUBSTEPS` points) to obtain actions and timestamps.
   - Step the environment with these actions, logging observations, rewards, and `info` dicts frame-by-frame.
   - Serialise reset info and per-step info (agent pose, block pose, goal pose, contacts, success flag, coverage) alongside the DMP signals.
   - Optionally save a video for the first `--video-samples` trajectories (hard-capped at 10 renders).
3. Save each trajectory as a compressed NPZ file under `OUTPUT_DIR/samples/{idx:06d}.npz` and append per-trajectory metadata as a single JSON line to `OUTPUT_DIR/index.jsonl`.
4. Close the environment in a `finally` block.

## Output Format (Simple, Training-Friendly)

Directory layout
- `OUTPUT_DIR/index.jsonl`: One JSON object per trajectory (metadata for filtering/sampling)
- `OUTPUT_DIR/samples/{idx:06d}.npz`: Compressed arrays for the trajectory
- `OUTPUT_DIR/samples/{idx:06d}.json`: Serialised `info` dicts for reset and each step
- Optional videos under `VIDEO_DIR/` when `--video-samples > 0` (plus an auto-combined clip)

NPZ contents per trajectory
- `act`: float32, shape `[T, act_dim]` — executed actions in pixel space
- `rew`: float32, shape `[T]` — per-step rewards
- `done`: bool, shape `[T]` — last element True iff terminated or truncated
- `phase`: float32, shape `[T+1]` — canonical phase `s(t) = exp(-alpha_s * t / duration)` aligned with the reset/steps timeline
- `time`: float32, shape `[T+1]` — wall-clock times in seconds aligned with observations (`time[0]=0.0`, `time[t+1]=time[t]+env_dt`)

Index JSONL fields per trajectory
- `id` (int), `path` (str relative to `OUTPUT_DIR`), `info_path` (str), `len` (int = T), `success` (bool)
- `seed` (int), `duration` (float), `stiffness` (float), `damping` (float)
- `terminated` (bool), `truncated` (bool)
- `dmp_dt` (float), `dmp_alpha_s` (float), `dmp_n_basis` (int)
- `scale_forcing_by_goal_delta` (bool)
- `start_pixels` (list[2] of float), `goal_pixels` (list[2] of float)
- `goal_pose` (list[3] of float) — validated constant across all steps
- `weights` (list[2][n_basis] of float)

Infos JSON contents per trajectory
- `reset_info`: serialised `info` dict returned by `env.reset(...)`
- `step_infos`: list of serialised `info` dicts returned by each `env.step(...)`

Consistency guarantee
- The script validates that `goal_pose` is identical (within 1e-5) across `reset_info` and every element of `step_infos`. If any discrepancy is found, the script terminates with an error for that sample.

Loading examples
```python
import json, numpy as np
from pathlib import Path

root = Path("data/random_dmp_npz/train")
for line in (root/"index.jsonl").open():
    meta = json.loads(line)
    with np.load(root/meta["path"]) as npz:
        act = npz["act"]; rew = npz["rew"]; done = npz["done"]; phase = npz["phase"]; time = npz["time"]
    infos = json.loads((root/meta["info_path"]).read_text(encoding="utf-8"))
    reset_info = infos["reset_info"]
    step_infos = infos["step_infos"]
    # filter by meta e.g., stiffness range, success, length, etc.
```

### What is inside `reset_info` and `step_infos`?
- These are serialised mirrors of the environment `info` dicts. Typical keys include `pos_agent`, `vel_agent`, `block_pose`, `goal_pose`, `n_contacts`, `is_success`, and possibly `coverage`.
- Arrays are converted to lists for JSON compatibility.

## Practical Tips
- For rapid checks, run with `--num-samples 10 --video-samples 0`; this avoids video encoding and writes a small `index.jsonl` with a handful of `.npz` files.
- Split generation: use `scripts/generate_random_dmp_splits.sh` to create train/validation/test under `data/random_dmp_npz/<split>/`.
- Filtering is fast by scanning `index.jsonl` first; load only the matching `.npz` files you need for a batch.
- Video rendering stops after the first ten samples even if `--video-samples` is larger, avoiding runaway encoding time.
