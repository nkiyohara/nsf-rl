# nsf-rl

Neural Stochastic Flows for PushT with Dynamic Movement Primitive (DMP) conditioning.

This repo provides:
- Data generation for PushT using sampled planar DMPs (per-trajectory NPZ + JSONL index).
- Training of an autonomous conditional neural stochastic flow (affine coupling) on pairwise transitions.
- Side-by-side video comparison of ground-truth vs model predictions.

## Installation

We recommend using `uv` for fast, reproducible setup:

```bash
uv sync
```

Alternatively, standard editable install with pip:

```bash
pip install -e .
```

Python 3.13+ is required. JAX with CUDA 12 wheels are specified in `pyproject.toml`. Adjust extras if needed for your platform.

## Quickstart

### 1) Generate dataset (NPZ per trajectory + JSONL index)

```bash
uv run -- python scripts/generate_random_dmp_dataset.py \
  --num-samples 1000 \
  --seed 42 \
  --output-dir data/random_dmp_npz \
  --video-dir videos/random_dmp_samples \
  --video-samples 10 \
  --scale-forcing
```

Outputs under `data/random_dmp_npz/`:
- `index.jsonl` (metadata per trajectory)
- `samples/{idx:06d}.npz` with arrays: `act`, `rew`, `done`, `phase`, `time`
- `samples/{idx:06d}.json` with serialised `reset_info` and `step_infos`

See `docs/generate_random_dmp_dataset.md` for full details.

### 2) Train conditional NSF on DMP pairs

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
  --wandb-project NSF-PushT --run-name dmp-nsf-affine
```

Notes:
- Condition standardization stats are saved to `<checkpoint_dir>/condition_stats.json` when enabled (default True).
- Parameters are saved to `<checkpoint_dir>/latest.eqx`, `<checkpoint_dir>/best.eqx`, and periodic `epoch_XXX.eqx` if `--checkpoint-interval` > 0.

See `docs/train_nsf_affine_dmp.md` for state/condition definitions and loss details.

### 3) Compare predictions vs ground truth (videos)

```bash
uv run -- python scripts/compare_flow_vs_gt.py \
  --data-root data/random_dmp_npz \
  --test-split test \
  --checkpoint-dir models/nsf_affine_dmp \
  --use-best \
  --num-trajs 10 \
  --output-dir videos/compare_flow_predictions \
  --fps 10
```

You can also select a specific checkpoint epoch using `--epoch 50`. By default, outputs are written to `videos/<checkpoint_name>/<epoch-tag>/`.

See `docs/compare_flow_vs_gt.md` for options.

## Repo structure

- `scripts/`
  - `generate_random_dmp_dataset.py`: Create JSONL-indexed dataset with per-trajectory NPZ files and optional videos.
  - `train_nsf_affine_dmp.py`: Train the conditional autonomous NSF on DMP pairs.
  - `compare_flow_vs_gt.py`: Render side-by-side GT vs predictions.
- `src/nsf_rl/`
  - `dmp.py`: Planar DMP implementation and configs.
  - `data/dmp_pairs.py`: Dataset for sampling transition pairs and building condition vectors; stats for standardization.
  - `models/*`: Conditional flow architecture, bijectors, distributions, losses.
  - `training.py`: Alternate training pipeline (paired with `src/nsf_rl/cli.py`).
- `marimo/dmp_explorer.py`: Interactive DMP explorer (optional).

The legacy CLI in `src/nsf_rl/cli.py` wraps an older NPZ dataset format (`src/nsf_rl/data/generate.py`) and an alternate training entrypoint. The primary workflow in this repo uses the `scripts/` described above.

## Videos

Generated videos will appear under `videos/`. The dataset generator can also optionally combine the first N rendered samples into a single clip for quick inspection.

## Troubleshooting

- If you see rendering or collision handler errors, ensure `gym-pusht` is installed and `nsf_rl.utils.pymunk_compat.ensure_add_collision_handler()` is called (handled by scripts).
- For JAX GPU builds, verify the installed CUDA/CuDNN versions match the `jax` wheel you selected.

## License

MIT or project-specific license here.
