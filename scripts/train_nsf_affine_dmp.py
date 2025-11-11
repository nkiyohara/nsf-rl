#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import equinox as eqx
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax

from nsf_rl.models.conditional_flow import (
    ConditionalAuxiliaryFlow,
    ConditionalNeuralStochasticFlow,
    FlowNetworkConfig,
)
from nsf_rl.models.losses import flow_1_to_2_loss, flow_2_to_1_loss
from nsf_rl.data.dmp_pairs import DmpPairwiseDataset


def _safe_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def _to_wandb_safe(obj):
    """Recursively convert objects to a wandb-serializable form.

    - Convert Path to str
    - Handle dataclasses, dicts, lists/tuples
    - Keep primitives and None as-is
    - Fallback to str for unknown types
    """
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
    data_root: Path = Path("data/random_dmp_npz")
    train_split: str = "train"
    val_split: str = "validation"
    batch_size: int = 256
    epochs: int = 5000
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42
    # model
    hidden_size: int = 64
    depth: int = 2
    conditioner_hidden_size: int = 64
    conditioner_depth: int = 2
    num_flow_layers: int = 4
    activation: str = "tanh"
    scale_fn: str = "tanh_exp"
    # losses
    flow_loss_data_weight: float = 0.2
    flow_loss_sampled_weight: float = 0.2
    flow_1_2_weight: float = 1.0
    flow_2_1_weight: float = 1.0
    t_max_flow: Optional[float] = None
    eps: float = 1e-6
    # normalization
    standardize_condition: bool = True
    state_source: str = "waypoint"
    # logging
    checkpoint_dir: Optional[Path] = None
    wandb_project: Optional[str] = "NSF-PushT"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_interval: int = 50


def build_models(*, state_dim: int, condition_dim: int, cfg: TrainConfig, key: jax.Array):
    key_flow, key_aux = jax.random.split(key, 2)
    flow_cfg = FlowNetworkConfig(
        state_dim=state_dim,
        condition_dim=condition_dim,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        activation=cfg.activation,
        num_flow_layers=cfg.num_flow_layers,
        conditioner_hidden_size=cfg.conditioner_hidden_size,
        conditioner_depth=cfg.conditioner_depth,
        scale_fn=cfg.scale_fn,  # type: ignore
        include_initial_time=False,
    )
    flow_model = ConditionalNeuralStochasticFlow(key=key_flow, **asdict(flow_cfg))
    aux_model = ConditionalAuxiliaryFlow(
        state_dim=state_dim,
        condition_dim=condition_dim,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        activation=jnp.tanh if cfg.activation == "tanh" else jax.nn.tanh,
        include_initial_time=False,
        include_time_ratio=True,
        conditioner_hidden_size=cfg.conditioner_hidden_size,
        conditioner_depth=cfg.conditioner_depth,
        num_flow_layers=cfg.num_flow_layers,
        scale_fn=cfg.scale_fn,  # type: ignore
        key=key_aux,
    )
    return flow_model, aux_model


def _as_batch_dict(batch):
    # Convert PairBatch instance to a dict of jnp arrays (PyTree-friendly)
    if isinstance(batch, dict):
        return {k: jnp.asarray(v) for k, v in batch.items()}
    return {
        "x_init": jnp.asarray(batch.x_init),
        "x_final": jnp.asarray(batch.x_final),
        "t_init": jnp.asarray(batch.t_init),
        "t_final": jnp.asarray(batch.t_final),
        "t_middle": jnp.asarray(batch.t_middle),
        "condition": jnp.asarray(batch.condition),
    }


def nll_loss(flow_model: ConditionalNeuralStochasticFlow, batch):
    dist = flow_model(x_init=batch["x_init"], t_init=batch["t_init"], t_final=batch["t_final"], condition=batch["condition"])
    return -dist.log_prob(batch["x_final"])


def compute_flow_losses(flow_model: ConditionalNeuralStochasticFlow, aux_model: ConditionalAuxiliaryFlow, batch, key: jax.Array, w12: float, w21: float):
    keys = jax.random.split(key, batch["x_init"].shape[0])
    def per_example(xi, ti, tm, tf, cond, k):
        l12 = flow_1_to_2_loss(
            stochastic_flow=flow_model,
            auxiliary_model=aux_model,
            x_init=xi,
            t_init=ti,
            t_middle=tm,
            t_final=tf,
            condition=cond,
            key=k,
        )
        l21 = flow_2_to_1_loss(
            stochastic_flow=flow_model,
            auxiliary_model=aux_model,
            x_init=xi,
            t_init=ti,
            t_middle=tm,
            t_final=tf,
            condition=cond,
            key=k,
        )
        return w12 * l12 + w21 * l21, l12, l21
    total, l12s, l21s = jax.vmap(per_example)(batch["x_init"], batch["t_init"], batch["t_middle"], batch["t_final"], batch["condition"], keys)
    return total, l12s, l21s


def train(cfg: TrainConfig):
    # RNG
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Data (created early to determine dims; stats written later after checkpoint_dir is decided)
    train_root = cfg.data_root / cfg.train_split
    val_root = cfg.data_root / cfg.val_split
    train_ds = DmpPairwiseDataset(
        root=train_root,
        rng=rng,
        standardize=cfg.standardize_condition,
        stats_path=None,
        state_source=cfg.state_source,
    )
    val_ds = DmpPairwiseDataset(
        root=val_root,
        rng=rng,
        standardize=cfg.standardize_condition,
        stats_path=None,
        state_source=cfg.state_source,
    )

    # Inspect dims to build models
    # Peek one batch
    batch_iter = train_ds.batches(batch_size=1, repeat=True)
    peek = next(batch_iter)
    state_dim = int(peek.x_init.shape[-1])
    condition_dim = int(peek.condition.shape[-1])

    # Models and optimizers
    flow_model, aux_model = build_models(state_dim=state_dim, condition_dim=condition_dim, cfg=cfg, key=key)
    params = eqx.filter(flow_model, eqx.is_array)
    aux_params = eqx.filter(aux_model, eqx.is_array)
    optim = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
    opt_state = optim.init((params, aux_params))

    # WandB (now have dims for config)
    wandb = _safe_import_wandb()
    if cfg.wandb_project and wandb is not None:
        # Serialize full config safely (convert Paths, etc.) and include derived dims
        full_cfg = {**asdict(cfg), "state_dim": state_dim, "condition_dim": condition_dim}
        wandb_config = _to_wandb_safe(full_cfg)
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_name, config=wandb_config)
    else:
        run = None

    # Checkpoint directory policy (after run is available)
    if cfg.checkpoint_dir is None:
        if run is not None and hasattr(run, "name") and run.name:
            checkpoint_dir = Path("models") / str(run.name)
        else:
            checkpoint_dir = Path("models") / "local_run"
    else:
        checkpoint_dir = cfg.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Recreate datasets with stats_path tied to checkpoint_dir and compute stats if requested
    stats_path = checkpoint_dir / "condition_stats.json"
    train_ds = DmpPairwiseDataset(
        root=train_root,
        rng=rng,
        standardize=cfg.standardize_condition,
        stats_path=stats_path,
        state_source=cfg.state_source,
    )
    val_ds = DmpPairwiseDataset(
        root=val_root,
        rng=rng,
        standardize=cfg.standardize_condition,
        stats_path=stats_path,
        state_source=cfg.state_source,
    )
    if cfg.standardize_condition:
        # Recompute stats on the final dataset object so they persist under checkpoint_dir
        train_ds = DmpPairwiseDataset(
            root=train_root,
            rng=rng,
            standardize=cfg.standardize_condition,
            stats_path=stats_path,
            state_source=cfg.state_source,
        )
        train_ds.compute_condition_stats()
        # Recreate val_ds to ensure it uses the same stats_path for reading
        val_ds = DmpPairwiseDataset(
            root=val_root,
            rng=rng,
            standardize=cfg.standardize_condition,
            stats_path=stats_path,
            state_source=cfg.state_source,
        )

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(params_tuple, batch, key):
        flow_params, aux_params = params_tuple
        fm = eqx.combine(flow_params, flow_model)
        am = eqx.combine(aux_params, aux_model)
        nll = nll_loss(fm, batch).mean()
        # data times flow loss
        flow_data, l12d, l21d = compute_flow_losses(fm, am, batch, key, cfg.flow_1_2_weight, cfg.flow_2_1_weight)
        flow_data = flow_data.mean()
        # sampled times flow loss (autonomous): introduce eps margins to avoid degeneracy
        # t_init fixed at 0, sample t_final in [3*eps, tT], t_middle in [eps, t_final - eps]
        t0 = jnp.zeros_like(batch["t_init"])  
        tT_batch = batch["t_final"].max()  # coarse upper bound per-batch
        tT = jnp.minimum(tT_batch, jnp.array(cfg.t_max_flow)) if cfg.t_max_flow is not None else tT_batch
        key_s = jax.random.split(key, 3)
        t_init_flow = t0
        t_final_flow = jax.random.uniform(
            key_s[1],
            shape=batch["t_final"].shape,
            minval=t_init_flow + 3.0 * cfg.eps,
            maxval=tT,
        )
        t_middle_flow = jax.random.uniform(
            key_s[2],
            shape=batch["t_final"].shape,
            minval=t_init_flow + cfg.eps,
            maxval=t_final_flow - cfg.eps,
        )
        batch_flow = {
            "x_init": batch["x_init"],
            "x_final": batch["x_final"],
            "t_init": t_init_flow,
            "t_final": t_final_flow,
            "t_middle": t_middle_flow,
            "condition": batch["condition"],
        }
        flow_samp, l12s, l21s = compute_flow_losses(fm, am, batch_flow, key, cfg.flow_1_2_weight, cfg.flow_2_1_weight)
        flow_samp = flow_samp.mean()
        total = nll + cfg.flow_loss_data_weight * flow_data + cfg.flow_loss_sampled_weight * flow_samp
        metrics = {
            "nll": nll,
            "flow_data": flow_data,
            "flow_sampled": flow_samp,
            "flow_1_2_data": l12d.mean(),
            "flow_2_1_data": l21d.mean(),
            "flow_1_2_samp": l12s.mean(),
            "flow_2_1_samp": l21s.mean(),
            "total": total,
        }
        return total, metrics

    @eqx.filter_jit
    def train_step(flow_params, aux_params, opt_state, batch, key):
        params_tuple = (flow_params, aux_params)
        (loss_value, metrics), grads = loss_fn(params_tuple, batch, key)
        updates, opt_state = optim.update(grads, opt_state, params_tuple)
        flow_params, aux_params = eqx.apply_updates(params_tuple, updates)
        return flow_params, aux_params, opt_state, metrics

    @eqx.filter_jit
    def eval_step(flow_params, aux_params, batch, key):
        fm = eqx.combine(flow_params, flow_model)
        am = eqx.combine(aux_params, aux_model)
        nll = nll_loss(fm, batch).mean()
        flow_data, _, _ = compute_flow_losses(fm, am, batch, key, cfg.flow_1_2_weight, cfg.flow_2_1_weight)
        return nll, flow_data.mean()

    best_val = float("inf")

    def run_epoch(root: Path, is_train: bool, epoch: int, epoch_bar=None):
        nonlocal params, aux_params, opt_state
        ds = train_ds if is_train else val_ds
        it = ds.batches(batch_size=cfg.batch_size, repeat=True)
        steps = max(1, len(ds.index) // cfg.batch_size)
        # Stable seeding; avoid negative fold_in values
        key_epoch = jax.random.fold_in(key, int(epoch))
        if not is_train:
            key_epoch = jax.random.fold_in(key_epoch, 1)
        metrics_sum = None
        for step in range(steps):
            batch = _as_batch_dict(next(it))
            k = jax.random.fold_in(key_epoch, step)
            if epoch_bar is not None:
                phase = "train" if is_train else "val"
                epoch_bar.set_postfix_str(f"{phase} {step + 1}/{steps}", refresh=True)
            if is_train:
                params, aux_params, opt_state, metrics = train_step(params, aux_params, opt_state, batch, k)
            else:
                # eval: no opt step
                nll, flow_data = eval_step(params, aux_params, batch, k)
                metrics = {"nll": nll, "flow_data": flow_data, "flow_sampled": jnp.array(0.0), "total": nll + cfg.flow_loss_data_weight * flow_data}
            if metrics_sum is None:
                metrics_sum = {k: jnp.array(v) for k, v in metrics.items()}
            else:
                for k2, v2 in metrics.items():
                    metrics_sum[k2] = metrics_sum[k2] + jnp.array(v2)
        # average
        metrics_avg = {k: float(v / steps) for k, v in metrics_sum.items()}
        # log (no console prints)
        split = "train" if is_train else "val"
        if run is not None:
            run.log({f"{split}/{k}": v for k, v in metrics_avg.items()}, step=epoch)
        return metrics_avg

    epoch_bar = tqdm(range(1, cfg.epochs + 1), total=cfg.epochs, desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        epoch_bar.set_postfix_str("train 0/?", refresh=True)
        run_epoch(train_root, True, epoch, epoch_bar)
        epoch_bar.set_postfix_str("val 0/?", refresh=True)
        val_metrics = run_epoch(val_root, False, epoch, epoch_bar)
        val_total = val_metrics.get("total", val_metrics.get("nll", 0.0))
        # checkpoint
        eqx.tree_serialise_leaves(checkpoint_dir / "latest.eqx", (params, aux_params))
        if val_total < best_val:
            best_val = val_total
            eqx.tree_serialise_leaves(checkpoint_dir / "best.eqx", (params, aux_params))
        # periodic checkpoints every N epochs (default: 50)
        if cfg.checkpoint_interval and (epoch % cfg.checkpoint_interval == 0):
            eqx.tree_serialise_leaves(checkpoint_dir / f"epoch_{epoch:03d}.eqx", (params, aux_params))


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train autonomous NSF (affine coupling) on DMP pairs")
    # Use None defaults so TrainConfig() remains the single source of truth.
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--train-split", type=str, default=None)
    p.add_argument("--val-split", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--conditioner-hidden-size", type=int, default=None)
    p.add_argument("--conditioner-depth", type=int, default=None)
    p.add_argument("--num-flow-layers", type=int, default=None)
    p.add_argument("--activation", type=str, default=None)
    p.add_argument("--scale-fn", type=str, default=None)
    p.add_argument("--flow-loss-data-weight", type=float, default=None)
    p.add_argument("--flow-loss-sampled-weight", type=float, default=None)
    p.add_argument("--flow-1-2-weight", type=float, default=None)
    p.add_argument("--flow-2-1-weight", type=float, default=None)
    p.add_argument("--t-max-flow", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--standardize-condition", dest="standardize_condition", action="store_true")
    p.add_argument("--no-standardize-condition", dest="standardize_condition", action="store_false")
    p.set_defaults(standardize_condition=None)
    p.add_argument("--state-source", type=str, choices=("waypoint", "env"), default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    # Only override fields when CLI provides a non-None value
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.val_split is not None:
        cfg.val_split = args.val_split
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.seed is not None:
        cfg.seed = args.seed
    if args.hidden_size is not None:
        cfg.hidden_size = args.hidden_size
    if args.depth is not None:
        cfg.depth = args.depth
    if args.conditioner_hidden_size is not None:
        cfg.conditioner_hidden_size = args.conditioner_hidden_size
    if args.conditioner_depth is not None:
        cfg.conditioner_depth = args.conditioner_depth
    if args.num_flow_layers is not None:
        cfg.num_flow_layers = args.num_flow_layers
    if args.activation is not None:
        cfg.activation = args.activation
    if args.scale_fn is not None:
        cfg.scale_fn = args.scale_fn
    if args.flow_loss_data_weight is not None:
        cfg.flow_loss_data_weight = args.flow_loss_data_weight
    if args.flow_loss_sampled_weight is not None:
        cfg.flow_loss_sampled_weight = args.flow_loss_sampled_weight
    if args.flow_1_2_weight is not None:
        cfg.flow_1_2_weight = args.flow_1_2_weight
    if args.flow_2_1_weight is not None:
        cfg.flow_2_1_weight = args.flow_2_1_weight
    if args.t_max_flow is not None:
        cfg.t_max_flow = args.t_max_flow
    if args.eps is not None:
        cfg.eps = args.eps
    if args.standardize_condition is not None:
        cfg.standardize_condition = args.standardize_condition
    if args.state_source is not None:
        cfg.state_source = args.state_source
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.wandb_entity = args.wandb_entity
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.checkpoint_interval is not None:
        cfg.checkpoint_interval = args.checkpoint_interval

    return cfg


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
