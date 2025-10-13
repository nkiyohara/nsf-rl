#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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


@dataclass
class TrainConfig:
    data_root: Path
    train_split: str = "train"
    val_split: str = "validation"
    batch_size: int = 256
    epochs: int = 100
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
    # normalization
    standardize_condition: bool = True
    # logging
    checkpoint_dir: Path = Path("models/nsf_affine_dmp")
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None


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
    flow_model = ConditionalNeuralStochasticFlow(config=flow_cfg, key=key_flow)
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

    # Data
    train_root = cfg.data_root / cfg.train_split
    val_root = cfg.data_root / cfg.val_split
    stats_path = cfg.checkpoint_dir / "condition_stats.json"
    train_ds = DmpPairwiseDataset(root=train_root, rng=rng, standardize=cfg.standardize_condition, stats_path=stats_path)
    val_ds = DmpPairwiseDataset(root=val_root, rng=rng, standardize=cfg.standardize_condition, stats_path=stats_path)
    if cfg.standardize_condition:
        train_ds.compute_condition_stats()

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

    # WandB
    wandb = _safe_import_wandb()
    if cfg.wandb_project and wandb is not None:
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.run_name, config={**asdict(cfg), "state_dim": state_dim, "condition_dim": condition_dim})
    else:
        run = None

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(params_tuple, batch, key):
        flow_params, aux_params = params_tuple
        fm = eqx.combine(flow_params, flow_model)
        am = eqx.combine(aux_params, aux_model)
        nll = nll_loss(fm, batch).mean()
        # data times flow loss
        flow_data, l12d, l21d = compute_flow_losses(fm, am, batch, key, cfg.flow_1_2_weight, cfg.flow_2_1_weight)
        flow_data = flow_data.mean()
        # sampled times flow loss
        # sample t_init_flow, t_final_flow, t_middle_flow uniformly over [0, time[-1]] for each sample
        t0 = jnp.zeros_like(batch["t_init"]) 
        tT = batch["t_final"].max()  # coarse upper bound per-batch
        key_s = jax.random.split(key, 3)
        t_init_flow = jax.random.uniform(key_s[0], shape=batch["t_init"].shape, minval=t0, maxval=tT)
        t_final_flow = jax.random.uniform(key_s[1], shape=batch["t_final"].shape, minval=t_init_flow, maxval=tT)
        t_middle_flow = jax.random.uniform(key_s[2], shape=batch["t_final"].shape, minval=t_init_flow, maxval=t_final_flow)
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

    # Checkpoint dir
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    def run_epoch(root: Path, is_train: bool, epoch: int):
        nonlocal params, aux_params, opt_state
        ds = train_ds if is_train else val_ds
        it = ds.batches(batch_size=cfg.batch_size, repeat=True)
        steps = max(1, len(ds.index) // cfg.batch_size)
        # Stable seeding; avoid negative fold_in values
        key_epoch = jax.random.fold_in(key, int(epoch))
        if not is_train:
            key_epoch = jax.random.fold_in(key_epoch, 1)
        metrics_sum = None
        for step in tqdm(range(steps), total=steps, desc=("train" if is_train else "val") + f" {epoch}", leave=False):
            batch = _as_batch_dict(next(it))
            k = jax.random.fold_in(key_epoch, step)
            if is_train:
                params, aux_params, opt_state, metrics = train_step(params, aux_params, opt_state, batch, k)
            else:
                # eval: no opt step
                fm = eqx.combine(params, flow_model)
                am = eqx.combine(aux_params, aux_model)
                nll = nll_loss(fm, batch).mean()
                flow_data, _, _ = compute_flow_losses(fm, am, batch, k, cfg.flow_1_2_weight, cfg.flow_2_1_weight)
                flow_data = flow_data.mean()
                metrics = {"nll": nll, "flow_data": flow_data, "flow_sampled": jnp.array(0.0), "total": nll + cfg.flow_loss_data_weight * flow_data}
            if metrics_sum is None:
                metrics_sum = {k: jnp.array(v) for k, v in metrics.items()}
            else:
                for k2, v2 in metrics.items():
                    metrics_sum[k2] = metrics_sum[k2] + jnp.array(v2)
        # average
        metrics_avg = {k: float(v / steps) for k, v in metrics_sum.items()}
        # log
        split = "train" if is_train else "val"
        print(f"epoch {epoch} {split}: ", {k: round(v, 4) for k, v in metrics_avg.items()})
        if run is not None:
            run.log({f"{split}/{k}": v for k, v in metrics_avg.items()}, step=epoch)
        return metrics_avg

    for epoch in range(1, cfg.epochs + 1):
        run_epoch(train_root, True, epoch)
        val_metrics = run_epoch(val_root, False, epoch)
        val_total = val_metrics.get("total", val_metrics.get("nll", 0.0))
        # checkpoint
        eqx.tree_serialise_leaves(cfg.checkpoint_dir / "latest.eqx", (params, aux_params))
        if val_total < best_val:
            best_val = val_total
            eqx.tree_serialise_leaves(cfg.checkpoint_dir / "best.eqx", (params, aux_params))


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train autonomous NSF (affine coupling) on DMP pairs")
    p.add_argument("--data-root", type=Path, default=Path("data/random_dmp_npz"))
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="validation")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--conditioner-hidden-size", type=int, default=64)
    p.add_argument("--conditioner-depth", type=int, default=2)
    p.add_argument("--num-flow-layers", type=int, default=4)
    p.add_argument("--activation", type=str, default="tanh")
    p.add_argument("--scale-fn", type=str, default="tanh_exp")
    p.add_argument("--flow-loss-data-weight", type=float, default=0.2)
    p.add_argument("--flow-loss-sampled-weight", type=float, default=0.2)
    p.add_argument("--flow-1-2-weight", type=float, default=1.0)
    p.add_argument("--flow-2-1-weight", type=float, default=1.0)
    p.add_argument("--t-max-flow", type=float, default=None)
    p.add_argument("--standardize-condition", action="store_true")
    p.add_argument("--no-standardize-condition", dest="standardize_condition", action="store_false")
    p.set_defaults(standardize_condition=True)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("models/nsf_affine_dmp"))
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        seed=args.seed,
        hidden_size=args.hidden_size,
        depth=args.depth,
        conditioner_hidden_size=args.conditioner_hidden_size,
        conditioner_depth=args.conditioner_depth,
        num_flow_layers=args.num_flow_layers,
        activation=args.activation,
        scale_fn=args.scale_fn,
        flow_loss_data_weight=args.flow_loss_data_weight,
        flow_loss_sampled_weight=args.flow_loss_sampled_weight,
        flow_1_2_weight=args.flow_1_2_weight,
        flow_2_1_weight=args.flow_2_1_weight,
        t_max_flow=args.t_max_flow,
        standardize_condition=args.standardize_condition,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()


