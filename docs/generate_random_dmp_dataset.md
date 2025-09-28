# `generate_random_dmp_dataset.py`

## Overview
- Samples Planar DMP configurations and executes them inside the PushT environment to build supervision data for conditional flow models.
- Records both the intended DMP targets and the full environment interaction (observations, rewards, per-step info) so models can learn from raw rollouts.
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
  --output-json data/random_dmp_dataset.json \
  --video-dir videos/random_dmp_samples \
  --video-samples 10 \
  --scale-forcing
```

Argument summary:
- `--output-json`: Target JSON file (default `data/random_dmp_dataset.json`). Parent dirs are created automatically.
- `--video-dir`: Directory for MP4 rollouts. Only used when `--video-samples > 0`.
- `--num-samples`: Number of trajectories to generate.
- `--video-samples`: Number of trajectories that should receive video captures (starting from index 0).
- `--seed`: Seed for `numpy.random.default_rng`, ensuring reproducible DMP parameter draws and environment resets.
- `--scale-forcing`: When set, multiplies the DMP forcing term by the goal displacement to mirror the Marimo explorer toggle.

## Generation Pipeline
1. Initialise RNG and gym environment (`render_mode="rgb_array"` so frames can be captured if needed).
2. For each requested sample (`tqdm` progress bar):
   - Reset the environment with a fresh seed.
   - Sample start/goal locations, duration, stiffness, and RBF weights; construct `DMPConfig`/`DMPParams`.
   - Roll out the DMP with sub-steps, recovering detailed signals: normalized positions, velocities, canonical phase `s`, and forcing term.
   - Down-sample to environment control frequency (every `DMP_DT_SUBSTEPS` points) to obtain actions and timestamps.
   - Step the environment with these actions, logging observations, rewards, and `info` dicts frame-by-frame.
   - Serialise reset info and per-step info (agent pose, block pose, goal pose, contacts, success flag, coverage) alongside the DMP signals.
   - Optionally save a video for the first `--video-samples` trajectories.
3. Dump the collected list of samples to JSON with indentation.
4. Close the environment in a `finally` block.

## Stored Fields per Trajectory
- **Metadata**: `seed`, `duration`, `stiffness`, `damping`, `timestamps`, `executed_steps`, `terminated`, `truncated`.
- **DMP configuration**: `weights`, `start_*`, `goal_*`, `dmp_dt`, `dmp_alpha_s`, `dmp_n_basis`, `dmp_waypoints_pixels`, `dmp_waypoints_normalized`.
- **DMP dynamics**: `dmp_velocity_normalized`, `dmp_forcing_normalized`, `dmp_canonical_phase_actions`, `dmp_canonical_phase_observations`.
- **Environment rollouts**: `executed_actions_pixels` + `executed_action_timestamps`, `observations` + `observation_timestamps`, `rewards`, `reset_info`, raw `step_infos`.
- **Convenience sequences** (aligned with observations): `env_pos_agent`, `env_vel_agent`, `env_block_pose`, `env_goal_pose`, `env_n_contacts`, `env_is_success`, `env_coverage`.
- **Video reference**: `video_path` when the trajectory was selected for capture.

## Practical Tips
- For rapid checks, run with `--num-samples 10 --video-samples 0` to keep the JSON small and skip video encoding.
- When creating large datasets, splitting runs across multiple files (e.g., batches of 1k trajectories with different seeds) avoids single massive JSON blobs.
- The environment observations are already stored as float32 lists; converting to NumPy arrays on load is straightforward via `np.asarray(entry["observations"], dtype=np.float32)`.

