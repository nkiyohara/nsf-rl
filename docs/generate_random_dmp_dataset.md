# `generate_random_dmp_dataset.py`

## Overview
- Samples planar DMP parameters and executes them in `gym_pusht/PushT-v0` to build training data for conditional flow models.
- Persists minimal arrays per trajectory (actions, rewards, done flags, phase, time) and serialised `reset_info`/`step_infos` with full state.
- Optionally captures rollout videos. Shows progress with `tqdm`.

## Runtime Dependencies
- `gymnasium`, `gym-pusht`
- `imageio`, `imageio-ffmpeg`
- `numpy`, `tqdm`
- Project-local: `nsf_rl.dmp`, `nsf_rl.utils.pymunk_compat`

## CLI
```bash
uv run -- python scripts/generate_random_dmp_dataset.py \
  --num-samples 1000 \
  --seed 42 \
  --output-dir data/random_dmp_npz \
  --video-dir videos/random_dmp_samples \
  --video-samples 10 \
  --scale-forcing
```

Arguments:
- `--output-dir` (Path, default `data/random_dmp_npz`): Root; creates `samples/` and `index.jsonl`.
- `--video-dir` (Path, default `videos/random_dmp_samples`): Where MP4s are written; ignored if `--video-samples 0`.
- `--num-samples` (int, default 128): Number of trajectories.
- `--video-samples` (int, default 10): Number of videos to render (hard cap 10).
- `--seed` (int, default 0): RNG seed for reproducibility.
- `--scale-forcing` (flag): Scale DMP forcing term by goal displacement.

## Generation details
1. Create PushT env (`render_mode="rgb_array"`), compute `env_dt` from control frequency, set DMP `dt = env_dt / 8` for sub-steps.
2. Per sample:
   - Reset env with fresh seed; derive start position from reset info/observation.
   - Sample `goal`, `duration [1,4]`, `stiffness [5,35]`, and `weights ∈ [-60,60]^{2×3}`.
   - Roll out DMP, downsample to env steps, step env with pixel-space waypoints.
   - Record rewards and serialise `info` dicts; optionally render frames.
   - Validate `goal_pose` consistency across reset and all steps.
3. Save arrays (`act, rew, done, phase, time`) to `samples/{idx:06d}.npz` and write `samples/{idx:06d}.json` with infos.
4. Append JSON line to `index.jsonl` with filtering metadata.
5. Optionally, write individual MP4s and a combined clip once the first N are rendered.

## Outputs

Directory layout:
- `OUTPUT_DIR/index.jsonl`
- `OUTPUT_DIR/samples/{idx:06d}.npz`
- `OUTPUT_DIR/samples/{idx:06d}.json`
- Optional videos under `VIDEO_DIR/` and a combined `samples_###-###.mp4`

NPZ per trajectory:
- `act` `[T, act_dim]` (float32)
- `rew` `[T]` (float32)
- `done` `[T]` (bool)
- `phase` `[T+1]` (float32), with `phase[t] = exp(-alpha_s * t / duration)`
- `time` `[T+1]` (float32), `time[0]=0.0`, step `env_dt`

Index JSONL fields:
- `id`, `path`, `info_path`, `len`, `success`, `seed`, `duration`, `stiffness`, `damping`, `terminated`, `truncated`,
- `dmp_dt`, `dmp_alpha_s`, `dmp_n_basis`, `scale_forcing_by_goal_delta`,
- `start_pixels`, `goal_pixels`, `goal_pose`, `weights`

Infos JSON:
- `reset_info` and `step_infos` (serialised env infos)

Consistency:
- `goal_pose` must match across reset and all steps (checked; otherwise the script exits for that sample).

## Tips
- Quick smoke test: `--num-samples 10 --video-samples 0` to avoid encoding.
- Use `scripts/generate_random_dmp_splits.sh` to produce train/validation/test subfolders under `data/random_dmp_npz/`.
- Filter by scanning `index.jsonl` first; load only matching NPZs.
- Video rendering is capped at the first 10 samples to keep runtime predictable.
