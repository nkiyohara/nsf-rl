#!/usr/bin/env python3
"""Training script for Latent Neural Stochastic Flow on PushT task.

This script trains a Latent NSF model for the PushT environment with:
- Partial observations: agent position and velocity (4D)
- Hidden state: block pose (4D) inferred via latent space
- Conditioning: DMP parameters (stiffness, damping, weights, etc.)
- Time axis: absolute time in seconds

The model learns to:
1. Infer the hidden block state from partial observations
2. Model the dynamics of the full state in latent space
3. Predict future observations from latent states

Usage:
    python scripts/train_latent_nsf_pusht.py --data-root data/random_dmp_npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from nsf_rl.data.pusht_latent_dataset import PushTLatentDataset, SequenceBatch
from nsf_rl.models.latent_stochastic_flow import (
    LatentStochasticFlow,
    LossComponents,
    build_latent_stochastic_flow,
)


def _safe_import_wandb():
    """Safely import wandb, returning None if not available."""
    try:
        import wandb

        return wandb
    except Exception:
        return None


def _to_wandb_safe(obj):
    """Recursively convert objects to a wandb-serializable form."""
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return _to_wandb_safe(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_wandb_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_wandb_safe(v) for v in obj]
    return str(obj)


@dataclass
class TrainConfig:
    """Configuration for training Latent NSF on PushT."""

    # Data
    data_root: Path = Path("data/random_dmp_npz")
    train_split: str = "train"
    val_split: str = "validation"
    seq_len: int = 16
    batch_size: int = 64
    epochs: int = 10000
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42

    # Irregular sampling (essential for NSF training)
    irregular_sampling: bool = True  # Randomly subsample for variable time intervals

    # Model architecture
    latent_dim: int = 8  # Latent state dimension
    encoder_embedding_dim: int = 64
    encoder_hidden_size: int = 64
    encoder_depth: int = 2
    encoder_type: Literal["mlp", "identity"] = "mlp"
    posterior_type: Literal["mlp", "mlp_residual", "gru"] = "gru"  # GRU for temporal context
    posterior_hidden_size: int = 64
    posterior_depth: int = 2
    gru_hidden_size: int = 64  # GRU hidden state size
    flow_type: Literal["mvn", "affine_coupling"] = "affine_coupling"
    flow_hidden_size: int = 64
    flow_depth: int = 2
    flow_num_layers: int = 4
    bridge_type: Literal["mvn", "affine_coupling"] = "mvn"
    bridge_hidden_size: int = 64
    bridge_depth: int = 2
    decoder_hidden_size: int = 64
    decoder_depth: int = 2

    # Condition routing
    use_condition_in_encoder: bool = True
    use_condition_in_flow: bool = True
    autonomous: bool = True  # Time-homogeneous flow

    # Loss weights
    flow_loss_weight: float = 0.1
    kl_weight: float = 1.0  # Target weight for KL divergence
    kl_anneal_epochs: int = 0  # Number of epochs to anneal KL from 0 to kl_weight

    # Normalization
    standardize_condition: bool = True

    # Logging
    checkpoint_dir: Optional[Path] = None
    wandb_project: Optional[str] = "Latent-NSF-PushT"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_interval: int = 100
    log_interval: int = 10


def build_model(
    *,
    obs_dim: int,
    condition_dim: int,
    cfg: TrainConfig,
    key: jax.Array,
) -> LatentStochasticFlow:
    """Build Latent NSF model from configuration."""
    return build_latent_stochastic_flow(
        latent_dim=cfg.latent_dim,
        obs_dim=obs_dim,
        condition_dim=condition_dim,
        encoder_embedding_dim=cfg.encoder_embedding_dim,
        encoder_hidden_size=cfg.encoder_hidden_size,
        encoder_depth=cfg.encoder_depth,
        encoder_type=cfg.encoder_type,
        posterior_type=cfg.posterior_type,
        posterior_hidden_size=cfg.posterior_hidden_size,
        posterior_depth=cfg.posterior_depth,
        gru_hidden_size=cfg.gru_hidden_size,
        flow_type=cfg.flow_type,
        flow_hidden_size=cfg.flow_hidden_size,
        flow_depth=cfg.flow_depth,
        flow_num_layers=cfg.flow_num_layers,
        bridge_type=cfg.bridge_type,
        bridge_hidden_size=cfg.bridge_hidden_size,
        bridge_depth=cfg.bridge_depth,
        decoder_hidden_size=cfg.decoder_hidden_size,
        decoder_depth=cfg.decoder_depth,
        use_condition_in_encoder=cfg.use_condition_in_encoder,
        use_condition_in_flow=cfg.use_condition_in_flow,
        autonomous=cfg.autonomous,
        key=key,
    )


def _as_batch_arrays(batch: SequenceBatch) -> dict:
    """Convert SequenceBatch to dictionary of jnp arrays."""
    return {
        "observations": jnp.asarray(batch.observations),  # [B, T, obs_dim]
        "full_states": jnp.asarray(batch.full_states),
        "times": jnp.asarray(batch.times),
        "condition": jnp.asarray(batch.condition),
    }


def compute_elbo_loss(
    model: LatentStochasticFlow,
    batch: dict,
    key: jax.Array,
    kl_weight: float = 1.0,
) -> tuple[jnp.ndarray, LossComponents]:
    """Compute ELBO loss for a batch.

    Args:
        model: Latent NSF model.
        batch: Dictionary with observations, times, condition.
        key: PRNG key.
        kl_weight: Weight for KL divergence term.

    Returns:
        Tuple of (total_loss, loss_components).
    """
    observations = batch["observations"]  # (batch, T, obs_dim)
    times = batch["times"]  # (batch, T)
    condition = batch["condition"]  # (batch, condition_dim)
    batch_size = condition.shape[0]

    # Split keys for each batch element
    keys = jax.random.split(key, batch_size)

    # Compute ELBO for each sequence using vmap
    # model.elbo expects: obs (T, obs_dim), times (T,), condition (condition_dim,), key
    # Return tuple instead of LossComponents for vmap compatibility
    def elbo_as_tuple(obs, t, cond, k):
        lc = model.elbo(obs=obs, times=t, condition=cond, key=k)
        return (lc.elbo, lc.reconstruction_loss, lc.kl_divergence, lc.flow_1_to_2_loss, lc.flow_2_to_1_loss)

    results = jax.vmap(elbo_as_tuple)(observations, times, condition, keys)
    elbo_batch, recon_batch, kl_batch, flow12_batch, flow21_batch = results

    # Average over batch
    avg_loss_components = LossComponents(
        elbo=elbo_batch.mean(),
        reconstruction_loss=recon_batch.mean(),
        kl_divergence=kl_batch.mean(),
        flow_1_to_2_loss=flow12_batch.mean(),
        flow_2_to_1_loss=flow21_batch.mean(),
    )

    # Total loss with weighted KL
    total_loss = (
        avg_loss_components.reconstruction_loss
        + kl_weight * avg_loss_components.kl_divergence
    )

    return total_loss, avg_loss_components


def train(cfg: TrainConfig):
    """Main training loop."""
    # RNG
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Data paths
    train_root = cfg.data_root / cfg.train_split
    val_root = cfg.data_root / cfg.val_split

    # Initialize datasets (without stats path first)
    train_ds = PushTLatentDataset(
        root=train_root,
        rng=rng,
        seq_len=cfg.seq_len,
        irregular_sampling=cfg.irregular_sampling,
        standardize=cfg.standardize_condition,
        stats_path=None,
    )

    # Get dimensions
    obs_dim = train_ds.obs_dim
    condition_dim = train_ds.condition_dim

    print(f"Observation dim: {obs_dim}")
    print(f"Condition dim: {condition_dim}")
    print(f"Latent dim: {cfg.latent_dim}")

    # Build model
    key, model_key = jax.random.split(key)
    model = build_model(
        obs_dim=obs_dim,
        condition_dim=condition_dim,
        cfg=cfg,
        key=model_key,
    )

    # Optimizer
    params = eqx.filter(model, eqx.is_array)
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.lr),
    )
    opt_state = optim.init(params)

    # WandB
    wandb = _safe_import_wandb()
    if cfg.wandb_project and wandb is not None:
        full_cfg = {
            **asdict(cfg),
            "obs_dim": obs_dim,
            "condition_dim": condition_dim,
        }
        wandb_config = _to_wandb_safe(full_cfg)
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config=wandb_config,
        )
    else:
        run = None

    # Checkpoint directory
    if cfg.checkpoint_dir is None:
        if run is not None and hasattr(run, "name") and run.name:
            checkpoint_dir = Path("models") / f"latent_nsf_{run.name}"
        else:
            checkpoint_dir = Path("models") / "latent_nsf_local"
    else:
        checkpoint_dir = cfg.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(_to_wandb_safe(asdict(cfg)), f, indent=2)

    # Setup datasets with stats path
    stats_path = checkpoint_dir / "condition_stats.json"
    train_ds = PushTLatentDataset(
        root=train_root,
        rng=rng,
        seq_len=cfg.seq_len,
        standardize=cfg.standardize_condition,
        stats_path=stats_path,
        irregular_sampling=cfg.irregular_sampling,
    )
    val_ds = PushTLatentDataset(
        root=val_root,
        rng=rng,
        seq_len=cfg.seq_len,
        irregular_sampling=cfg.irregular_sampling,
        standardize=cfg.standardize_condition,
        stats_path=stats_path,
    )

    # Compute condition statistics
    if cfg.standardize_condition:
        train_ds.compute_condition_stats()

    # Training step
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(params, batch, key, current_kl_weight):
        m = eqx.combine(params, model)
        total_loss, loss_components = compute_elbo_loss(
            m, batch, key, current_kl_weight
        )
        # Add flow loss
        total_loss = total_loss + cfg.flow_loss_weight * (
            loss_components.flow_1_to_2_loss + loss_components.flow_2_to_1_loss
        )
        metrics = {
            "total_loss": total_loss,
            "reconstruction_loss": loss_components.reconstruction_loss,
            "kl_divergence": loss_components.kl_divergence,
            "elbo": loss_components.elbo,
            "flow_1_to_2_loss": loss_components.flow_1_to_2_loss,
            "flow_2_to_1_loss": loss_components.flow_2_to_1_loss,
            "kl_weight": current_kl_weight,
        }
        return total_loss, metrics

    @eqx.filter_jit
    def train_step(params, opt_state, batch, key, current_kl_weight):
        (loss_value, metrics), grads = loss_fn(params, batch, key, current_kl_weight)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, metrics

    @eqx.filter_jit
    def eval_step(params, batch, key, current_kl_weight):
        m = eqx.combine(params, model)
        total_loss, loss_components = compute_elbo_loss(
            m, batch, key, current_kl_weight
        )
        return {
            "total_loss": total_loss,
            "reconstruction_loss": loss_components.reconstruction_loss,
            "kl_divergence": loss_components.kl_divergence,
            "elbo": loss_components.elbo,
        }

    # Training loop
    best_val_loss = float("inf")
    train_iter = train_ds.sequence_batches(batch_size=cfg.batch_size, repeat=True)
    steps_per_epoch = max(1, len(train_ds.index) // cfg.batch_size)

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        # Calculate KL weight
        if cfg.kl_anneal_epochs > 0:
            # Linear annealing from 0 to cfg.kl_weight
            progress = min(1.0, epoch / cfg.kl_anneal_epochs)
            current_kl_weight = jnp.array(progress * cfg.kl_weight)
        else:
            current_kl_weight = jnp.array(cfg.kl_weight)

        # Training
        train_metrics_sum = None
        for step in range(steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch = _as_batch_arrays(next(train_iter))

            params, opt_state, metrics = train_step(
                params, opt_state, batch, step_key, current_kl_weight
            )

            if train_metrics_sum is None:
                train_metrics_sum = {k: jnp.array(v) for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    train_metrics_sum[k] = train_metrics_sum[k] + jnp.array(v)


        # Average training metrics
        train_metrics = {k: float(v / steps_per_epoch) for k, v in train_metrics_sum.items()}

        # Validation
        val_iter = val_ds.sequence_batches(batch_size=cfg.batch_size, repeat=False)
        val_metrics_sum = None
        val_steps = 0

        for batch in val_iter:
            key, val_key = jax.random.split(key)
            batch = _as_batch_arrays(batch)
            metrics = eval_step(params, batch, val_key, current_kl_weight)

            if val_metrics_sum is None:
                val_metrics_sum = {k: jnp.array(v) for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum[k] + jnp.array(v)
            val_steps += 1

        if val_steps > 0 and val_metrics_sum is not None:
            val_metrics = {k: float(v / val_steps) for k, v in val_metrics_sum.items()}
        else:
            val_metrics = train_metrics  # Fallback if no validation data

        # Update progress bar
        epoch_bar.set_postfix({
            "train_loss": f"{train_metrics['total_loss']:.4f}",
            "val_loss": f"{val_metrics['total_loss']:.4f}",
            "elbo": f"{val_metrics['elbo']:.4f}",
            "kl_w": f"{current_kl_weight:.2f}",
        })

        # Logging
        if epoch % cfg.log_interval == 0 and run is not None:
            run.log(
                {
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "kl_weight": current_kl_weight,
                    "epoch": epoch,
                },
                step=epoch,
            )

        # Checkpointing
        if epoch % cfg.checkpoint_interval == 0:
            eqx.tree_serialise_leaves(
                checkpoint_dir / f"epoch_{epoch:05d}.eqx", params
            )

        # Save latest
        eqx.tree_serialise_leaves(checkpoint_dir / "latest.eqx", params)

        # Save best
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            eqx.tree_serialise_leaves(checkpoint_dir / "best.eqx", params)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def parse_args() -> TrainConfig:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train Latent NSF on PushT with partial observations"
    )

    # Data args
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-split", type=str, default=None)
    p.add_argument("--val-split", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Model args
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--encoder-embedding-dim", type=int, default=None)
    p.add_argument("--encoder-hidden-size", type=int, default=None)
    p.add_argument("--encoder-depth", type=int, default=None)
    p.add_argument("--encoder-type", type=str, choices=["mlp", "identity"], default=None)
    p.add_argument("--posterior-type", type=str, choices=["mlp", "mlp_residual", "gru"], default=None)
    p.add_argument("--posterior-hidden-size", type=int, default=None)
    p.add_argument("--posterior-depth", type=int, default=None)
    p.add_argument("--gru-hidden-size", type=int, default=None)
    p.add_argument("--flow-type", type=str, choices=["mvn", "affine_coupling"], default=None)
    p.add_argument("--flow-hidden-size", type=int, default=None)
    p.add_argument("--flow-depth", type=int, default=None)
    p.add_argument("--flow-num-layers", type=int, default=None)
    p.add_argument("--bridge-type", type=str, choices=["mvn", "affine_coupling"], default=None)
    p.add_argument("--decoder-hidden-size", type=int, default=None)
    p.add_argument("--decoder-depth", type=int, default=None)

    # Condition routing
    p.add_argument("--use-condition-in-encoder", action="store_true", dest="use_condition_in_encoder")
    p.add_argument("--no-condition-in-encoder", action="store_false", dest="use_condition_in_encoder")
    p.set_defaults(use_condition_in_encoder=None)
    p.add_argument("--use-condition-in-flow", action="store_true", dest="use_condition_in_flow")
    p.add_argument("--no-condition-in-flow", action="store_false", dest="use_condition_in_flow")
    p.set_defaults(use_condition_in_flow=None)

    # Loss weights
    p.add_argument("--flow-loss-weight", type=float, default=None)
    p.add_argument("--kl-weight", type=float, default=None)
    p.add_argument("--kl-anneal-epochs", type=int, default=None)

    # Normalization
    p.add_argument("--standardize-condition", action="store_true", dest="standardize_condition")
    p.add_argument("--no-standardize-condition", action="store_false", dest="standardize_condition")
    p.set_defaults(standardize_condition=None)

    # Logging
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)

    args = p.parse_args()

    # Build config with defaults, override with CLI args
    cfg = TrainConfig()
    for field_name in TrainConfig.__dataclass_fields__:
        cli_value = getattr(args, field_name, None)
        if cli_value is not None:
            setattr(cfg, field_name, cli_value)

    return cfg


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()

