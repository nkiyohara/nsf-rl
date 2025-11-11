# Conditional Neural Stochastic Flow for PushT

This document describes the conditional flow components and how they relate to the current training scripts.

## Background
- We model transitions with autonomous neural stochastic flows trained with MLL and bidirectional flow consistency objectives, conditioned on DMP parameters.

## Data
- Primary dataset path used by the current workflow is generated via `scripts/generate_random_dmp_dataset.py` (JSONL index + per-trajectory NPZ). Each NPZ now also stores normalized DMP waypoints and velocities (`waypoints_norm`, `waypoint_vel_norm`) sampled at the environment control rate so that we can condition directly on the command trajectory.
- The legacy padded NPZ dataset in `src/nsf_rl/data/generate.py` is still available but not used in the main scripts.

## Architecture
- See `src/nsf_rl/models/conditional_flow.py` for `ConditionalNeuralStochasticFlow` and `ConditionalAuxiliaryFlow`. `FlowNetworkConfig` just groups the flow hyperparameters that we unpack into the model constructor.
- Bijectors live in `src/nsf_rl/models/bijectors.py`; distributions in `src/nsf_rl/models/distributions.py`; losses in `src/nsf_rl/models/losses.py`.
- Time handling is autonomous: the flow reduces to identity at zero interval, and losses evaluate at both data and sampled times.

## Training
- The primary entrypoint is `scripts/train_nsf_affine_dmp.py`, which builds pairwise datasets from the JSONL/NPZ format (`src/nsf_rl/data/dmp_pairs.py`). State vectors default to the DMP waypoint positions + velocities + phase (`--state-source waypoint`), but you can switch back to reconstructed environment observations via `--state-source env`. Condition vectors are optionally standardized and stats are stored in `<checkpoint_dir>/condition_stats.json`.
- The alternative legacy path is `src/nsf_rl/cli.py` + `src/nsf_rl/training.py` + `src/nsf_rl/data/generate.py`, which targets a single padded `.npz` dataset. Prefer the `scripts/` path unless you specifically need that format.

## CLI
- Preferred: run training and comparison via the scripts in `scripts/`. See `docs/train_nsf_affine_dmp.md` and `docs/compare_flow_vs_gt.md`.
- Legacy: `python -m nsf_rl.cli generate-data ...` and `python -m nsf_rl.cli train ...` exist but are not the default path here.

## Interactive Explorer
- `marimo/dmp_explorer.py` lets you experiment with DMPs and an optional PushT simulation. The data-generation script includes a `--scale-forcing` flag that mirrors the explorer’s goal-scaling toggle.

## Notes
- Before training, run `uv sync` to install dependencies.
- For GPU, ensure your JAX wheel matches your CUDA setup.
