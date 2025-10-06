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
- **Attributes**
  - `seed` (int): RNG draw used to reset the environment for this rollout.
  - `duration` (float): Planned DMP execution time in seconds.
  - `stiffness` (float): Spring constant passed to the DMP and controller.
  - `damping` (float): Critical-damping term `2*sqrt(stiffness)` carried with the sample.
  - `executed_steps` (int): Number of environment steps actually taken before stopping.
  - `terminated` (bool): Whether the environment emitted a terminal signal.
  - `truncated` (bool): Whether the rollout ended because of the environment horizon.
  - `dmp_dt` (float): Integration step used inside the DMP (seconds per sub-step).
  - `dmp_alpha_s` (float): Canonical system decay parameter from the Planar DMP.
  - `dmp_n_basis` (int): Number of radial basis functions in the DMP.
  - `video_path` (str, optional): Absolute path to the per-trajectory MP4 when recorded.
- **DMP configuration (datasets)**
  - `weights` (float32[2, 3]): RBF weights for x/y channels, ordered by basis index.
  - `start_pixels` (float32[2]): Initial planar end-effector location in PushT pixels.
  - `start_normalized` (float32[2]): Same start position rescaled to the normalized workspace.
  - `goal_pixels` (float32[2]): Sampled goal position in pixels.
  - `goal_normalized` (float32[2]): Goal position in normalized coordinates.
  - `timestamps` (float32[T_dmp]): Absolute times (seconds) for each DMP waypoint kept after sub-sampling to the control frequency.
- **DMP rollout (datasets)**
  - `dmp_waypoints_pixels` (float32[T_dmp, 2]): Position targets fed to the environment each control step.
  - `dmp_waypoints_normalized` (float32[T_dmp, 2]): Same waypoints in normalized space.
  - `dmp_velocity_normalized` (float32[T_dmp, 2]): Normalized velocity trace emitted by the DMP integrator.
  - `dmp_forcing_normalized` (float32[T_dmp, 2]): Normalized forcing term after optional goal scaling.
  - `dmp_canonical_phase_actions` (float32[T_dmp]): Canonical phase values aligned with each control action time.
  - `dmp_canonical_phase_observations` (float32[T_obs]): Canonical phase evaluated at every observation timestamp (`T_obs = executed_steps + 1`).
- **Environment rollout (datasets)**
  - `executed_actions_pixels` (float32[T_env, 2]): Actions actually applied to the environment (`T_env = executed_steps`). Can be shorter than `T_dmp` if the rollout stopped early.
  - `executed_action_timestamps` (float32[T_env]): Elapsed control times `t = n * env_dt` for each executed action.
  - `observations` (float32[T_obs, obs_dim]): Observed PushT state sequence starting with the reset state; `obs_dim` is the environment's observation dimensionality.
  - `observation_timestamps` (float32[T_obs]): Elapsed times sampled at every observation.
  - `rewards` (float32[T_env]): Step-wise reward signal aligned with executed actions.
- **Structured metadata (datasets, JSON-encoded UTF-8 strings)**
  - `reset_info`: Serialized info dict returned by `env.reset()`.
  - `env_info`: Final info dict observed when the rollout ended (empty if unavailable).
  - `step_infos`: List of per-step info dicts (`len = T_env`), matching the environment transitions.
  - `env_pos_agent`: List of agent position entries collected from reset/step infos (`len = T_obs`; entries may be `null` when missing).
  - `env_vel_agent`: List of agent velocity entries pulled from infos (`len = T_obs`).
  - `env_block_pose`: List of block pose entries from infos (`len = T_obs`).
  - `env_goal_pose`: List of goal pose entries from infos (`len = T_obs`).
  - `env_n_contacts`: List of contact counts from infos (`len = T_obs`).
  - `env_is_success`: List of success flags from infos (`len = T_obs`).
  - `env_coverage`: List of coverage metrics from infos (`len = T_obs`).

## Practical Tips
- For rapid checks, run with `--num-samples 10 --video-samples 0`; the resulting HDF5 file stays lightweight and skips video encoding.
- Each trajectory can be streamed directly: `with h5py.File(path) as f: traj = f["sample_00042"]`.
- Decode any JSON-style dataset with `json.loads(traj["reset_info"][()])` to regain dict structures.
- Video rendering stops after the first ten samples even if `--video-samples` is larger, avoiding runaway encoding time.
