# Conditional Neural Stochastic Flow for PushT

## Background
- The original Neural Stochastic Flows paper emphasises single-step sampling from learned SDE solutions using both maximum likelihood (MLL) and forward/backward flow consistency losses. Those objectives are retained here without the latent-variable block to target conditional transition modelling.
- Conditioning is introduced through Dynamic Movement Primitive (DMP) parameters that act as control context while the flow models the environment state evolution.

## Data Generation
- `nsf_rl/dmp.py:11-160` contains a planar DMP implementation that samples start/goal locations and RBF weights, then constructs the forcing term as a normalised linear combination of radial basis functions acting on the canonical variable. `DMPParams.as_vector()` therefore packs only duration, start/goal, weight matrix, and damping-related scalars. The new `DMPConfig.scale_forcing_by_goal_delta` flag controls whether the forcing term is stretched along the goal offset (`True`, the default) or each coordinate is driven independently (`False`).
- `nsf_rl/data/generate.py:19-208` wraps PushT rollouts. It calls `ensure_add_collision_handler()` and registers `gym_pusht`, samples DMP parameters, executes the PD-controlled policy, and stores padded tensors (observations, actions, masks, times, rewards, condition vectors). Their global min/max are logged to the metadata JSON. When `video_dir` is provided it also records `rgb_array` frames to MP4 so trajectories can be inspected visually.
- `nsf_rl/utils/pymunk_compat.py:8-64` patches modern PyMunk releases so the legacy `Space.add_collision_handler()` API required by PushT is restored, pre-populating handler data slots for collision callbacks.

## Model Architecture
- `nsf_rl/models/conditional_flow.py:22-233` defines the flow components: `ConditionalGaussianTransition` produces autonomous base Gaussians that depend only on the elapsed time; `ConditionalAuxiliaryFlow` and `ConditionalNeuralStochasticFlow` then apply `ContinuousAffineCoupling` layers whose shifts/scales are scaled by the effective time interval (`α` for the bridge, `Δt` for transitions), guaranteeing they collapse to the identity whenever the interval length is zero or we evaluate the bridge endpoints (`nsf_rl/models/bijectors.py:1-150`).
- Probability primitives (`MultivariateNormalDiag`, `ContinuousNormalizingFlow`) live in `nsf_rl/models/distributions.py:1-206`, retained from the reference implementation with minimal trimming.
- Flow losses are implemented in `nsf_rl/models/losses.py:12-139`, combining the forward and reverse consistency objectives with MLL, exactly mirroring the paper’s equations but now conditioned on the DMP vector.

## Training Pipeline
- Dataset access is provided by `nsf_rl/data/dataset.py:15-75`, which precomputes valid transition indices and delivers batches of `(state_t, state_{t+1}, t_t, t_{t+1}, condition)` tuples.
- `nsf_rl/training.py:24-265` orchestrates optimisation: `TrainState` stores a partitioned parameter/static split to keep non-array fields out of Optax; `_loss_and_metrics()` evaluates MLL and flow losses; `train_step()` applies filtered gradients; and `train()` wires everything together, handles W&B logging, and periodically evaluates random transition batches. Passing `WANDB_MODE=offline` is respected so runs can stay local when needed.

## Command Line Interface
- `nsf_rl/cli.py:1-123` exposes two subcommands:
  - `generate-data` writes a dataset: `uv run -- python -m nsf_rl.cli generate-data --output data/pusht.npz --num-trajectories 512 --video-dir videos/` to also capture MP4 rollouts.
  - `train` starts optimisation with W&B logging: `uv run -- python -m nsf_rl.cli train --dataset data/pusht.npz --steps 50000 --batch-size 256`.
  Hyperparameters such as hidden sizes, number of flow layers, and auxiliary conditioner/scale choices can be tuned via flags (see `--aux-conditioner-hidden-size`, `--aux-num-flow-layers`, `--aux-scale-fn`).

## Interactive Explorer
- `marimo/dmp_explorer.py` provides a Marimo app for experimenting with DMP parameters in a normalized \([-1, 1]^2\) control space. Sliders control start/goal positions (in pixels), duration, stiffness, and individual RBF weights. A checkbox toggles whether the forcing term is scaled by the goal offset so you can compare goal-direction modulation on and off while keeping the UI sliders in the same numeric range. The app can also stream PushT rollouts when `simulate PushT` is enabled.

## Dataset Layout
- The saved `.npz` archive from `generate_pusht_dataset()` contains fields documented in `nsf_rl/data/generate.py:169-186`. Observations and actions are time-major padded arrays with accompanying masks; `state_times`/`action_times` store the physical timestamps (step size `dt = 0.1` by default); `conditions` is the DMP parameter matrix. A matching `.meta.json` summarises trajectory count, success statistics, and dimensionalities.

## Usage Notes
- Before training, run `uv sync` to materialise dependencies (`jax`, `equinox`, `optax`, `wandb`, `gym-pusht`).
- When training on larger datasets, consider enabling W&B online sync or keep `WANDB_MODE=offline` if experiments must stay local.
- The current design keeps the model non-jitted to simplify parameter partitioning. For production-scale runs you can re-enable `eqx.filter_jit` once static arguments are hashable (e.g., by wrapping static state in dataclasses with deterministic hashing).
