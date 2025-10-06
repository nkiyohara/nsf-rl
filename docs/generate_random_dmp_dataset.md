# `generate_random_dmp_dataset.py`

## Overview
- Samples Planar DMP configurations and executes them inside the PushT environment to build supervision data for conditional flow models.
- Records both the intended DMP targets and the full environment interaction (observations, rewards, per-step info) so models can learn from raw rollouts.
- Optionally captures rollout videos for quick qualitative inspection. Progress is displayed with `tqdm` to show how many trajectories have finished.

## Runtime Dependencies
- `gymnasium` and `gym-pusht` for environment simulation.
- `h5py` for writing structured HDF5 datasets.
- `imageio` + `imageio-ffmpeg` for video export when `--video-samples > 0`.
- `numpy` for sampling and tensor manipulation.
- `tqdm` for progress bars.
- Project-local modules: `nsf_rl.dmp` (Planar DMP implementation) and `nsf_rl.utils.pymunk_compat` (collision handler patch for modern PyMunk releases).

## Command-Line Interface
```
uv run -- python scripts/generate_random_dmp_dataset.py \
  --num-samples 1000 \
  --seed 42 \
  --output-h5 data/random_dmp_dataset.h5 \
  --video-dir videos/random_dmp_samples \
  --video-samples 10 \
  --scale-forcing
```

Argument summary:
- `--output-h5`: Target HDF5 file (default `data/random_dmp_dataset.h5`). Parent dirs are created automatically.
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
3. Write each sample into a dedicated `sample_<idx>` group in the HDF5 file, storing arrays as float32 datasets and nested metadata as UTF-8 JSON blobs or attributes.
4. Close the environment in a `finally` block.

## Stored Fields per Trajectory
- **Group attributes**: `seed`, `duration`, `stiffness`, `damping`, `executed_steps`, `terminated`, `truncated`, `dmp_dt`, `dmp_alpha_s`, `dmp_n_basis`, plus `video_path` when a capture exists.
- **DMP configuration datasets**: `weights`, `start_pixels`, `start_normalized`, `goal_pixels`, `goal_normalized`, `timestamps`.
- **DMP dynamics datasets**: `dmp_waypoints_pixels`, `dmp_waypoints_normalized`, `dmp_velocity_normalized`, `dmp_forcing_normalized`, `dmp_canonical_phase_actions`, `dmp_canonical_phase_observations`.
- **Environment rollout datasets**: `executed_actions_pixels`, `executed_action_timestamps`, `observations`, `observation_timestamps`, `rewards`.
- **Structured metadata**: JSON-encoded UTF-8 datasets for `reset_info`, `env_info`, `step_infos`, `env_pos_agent`, `env_vel_agent`, `env_block_pose`, `env_goal_pose`, `env_n_contacts`, `env_is_success`, `env_coverage`.

## Practical Tips
- For rapid checks, run with `--num-samples 10 --video-samples 0`; the resulting HDF5 file stays lightweight and skips video encoding.
- Each trajectory can be streamed directly: `with h5py.File(path) as f: traj = f["sample_00042"]`.
- Decode any JSON-style dataset with `json.loads(traj["reset_info"][()])` to regain dict structures.
- Video rendering stops after the first ten samples even if `--video-samples` is larger, avoiding runaway encoding time.
