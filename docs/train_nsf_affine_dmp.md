# Train Autonomous Neural Stochastic Flow (Affine Coupling) on DMP Pairs

This script trains an autonomous neural stochastic flow (affine coupling) conditioned on constant DMP parameters using random two-step pairs extracted from the generated PushT dataset.

## State and Condition
- State (7D): `[pos_agent_x, pos_agent_y, block_x, block_y, sin(theta_deg), cos(theta_deg), phase]`
  - Pixel-like components are mapped to [-1, 1]. Angle is converted from degrees to radians, then expanded to `sin, cos`.
- Condition (excluding duration): `stiffness, damping, dmp_dt, dmp_alpha_s, dmp_n_basis, scale_forcing_by_goal_delta, start_pixels(2), goal_pixels(2), weights(2*n_basis)`
  - Pixel-like `start_pixels, goal_pixels` are mapped to [-1, 1]. Others standardized from train split (toggleable).
- Time uses saved `time[T+1]` from the dataset (real seconds).

## Loss
- Negative log-likelihood with symmetric flow loss at data times and sampled times.

## Requirements
- Generated dataset with `time` included (regenerate with the updated `scripts/generate_random_dmp_dataset.py` if missing).
- JAX + Equinox + Optax, and optionally WandB for logging.

## Usage
```bash
uv run -- python scripts/train_nsf_affine_dmp.py \
  --data-root data/random_dmp_npz \
  --train-split train \
  --val-split validation \
  --batch-size 256 \
  --epochs 100 \
  --lr 1e-3 \
  --hidden-size 64 --depth 2 \
  --conditioner-hidden-size 64 --conditioner-depth 2 \
  --num-flow-layers 4 \
  --scale-fn tanh_exp \
  --flow-loss-data-weight 0.2 \
  --flow-loss-sampled-weight 0.2 \
  --flow-1-2-weight 1.0 \
  --flow-2-1-weight 1.0 \
  --checkpoint-dir models/nsf_affine_dmp \
  --wandb-project your_project --wandb-entity your_entity --run-name dmp-nsf-affine
```

## Checkpoints and Stats
- Condition standardization stats saved to: `models/nsf_affine_dmp/condition_stats.json`.
- Latest and best model parameters saved as `latest.eqx` and `best.eqx` under the checkpoint directory.

## Notes
- Pair sampling is uniform over i < j within each trajectory. `t_middle` is uniform in `[t_init, t_final]`.
- Sampled-time flow loss uses times uniformly drawn over the per-batch max of `t_final`.
- No dependency on the embedded `latent-neural-stochastic-flows` code; the implementation uses only `src/nsf_rl/models/*`.


