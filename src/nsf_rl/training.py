"""Training utilities for conditional neural stochastic flows on PushT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb

from nsf_rl.data.dataset import TransitionBatch, TransitionDataset
from nsf_rl.models.conditional_flow import (
    ConditionalAuxiliaryFlow,
    ConditionalNeuralStochasticFlow,
    FlowNetworkConfig,
)
from nsf_rl.models.losses import flow_loss, maximum_log_likelihood, time_between


@dataclass
class TrainingConfig:
    dataset_path: Path
    num_steps: int = 10000
    batch_size: int = 128
    learning_rate: float = 3e-4
    flow_weight: float = 0.5
    flow_12_weight: float = 1.0
    flow_21_weight: float = 1.0
    seed: int = 0
    log_every: int = 50
    eval_every: int = 500
    eval_samples: int = 2048
    wandb_project: str = "conditional-nsf"
    wandb_run_name: Optional[str] = None


@dataclass
class AuxiliaryConfig:
    hidden_size: int = 128
    depth: int = 2
    activation: str = "tanh"
    include_initial_time: bool = False
    include_time_ratio: bool = True
    conditioner_hidden_size: int = 128
    conditioner_depth: int = 2
    num_flow_layers: int = 2
    scale_fn: str = "tanh_exp"


@dataclass
class TrainState:
    params: eqx.Module
    static: eqx.Module
    opt_state: optax.OptState

    def combine(self) -> tuple[ConditionalNeuralStochasticFlow, ConditionalAuxiliaryFlow]:
        return eqx.combine(self.params, self.static)


def _activation(name: str):
    mapping = {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "elu": jax.nn.elu,
        "swish": jax.nn.swish,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation {name}")
    return mapping[name]


def setup_training_state(
    dataset: TransitionDataset,
    flow_config: FlowNetworkConfig,
    aux_config: AuxiliaryConfig,
    optimizer: optax.GradientTransformation,
    key: jax.Array,
) -> TrainState:
    key_model, key_aux = jax.random.split(key, 2)
    model = ConditionalNeuralStochasticFlow(key=key_model, **asdict(flow_config))
    auxiliary = ConditionalAuxiliaryFlow(
        state_dim=dataset.state_dim,
        condition_dim=dataset.condition_dim,
        hidden_size=aux_config.hidden_size,
        depth=aux_config.depth,
        activation=_activation(aux_config.activation),
        include_initial_time=aux_config.include_initial_time,
        include_time_ratio=aux_config.include_time_ratio,
        conditioner_hidden_size=aux_config.conditioner_hidden_size,
        conditioner_depth=aux_config.conditioner_depth,
        num_flow_layers=aux_config.num_flow_layers,
        scale_fn=aux_config.scale_fn,
        key=key_aux,
    )
    params, static = eqx.partition((model, auxiliary), eqx.is_array)
    opt_state = optimizer.init(params)
    return TrainState(params=params, static=static, opt_state=opt_state)


def make_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    return optax.adam(config.learning_rate)


def _loss_and_metrics(
    model: ConditionalNeuralStochasticFlow,
    auxiliary: ConditionalAuxiliaryFlow,
    batch: TransitionBatch,
    *,
    flow_weight: float,
    flow_12_weight: float,
    flow_21_weight: float,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    log_prob = maximum_log_likelihood(
        stochastic_flow=model,
        x_init=batch.x_init,
        x_final=batch.x_final,
        t_init=batch.t_init,
        t_final=batch.t_final,
        condition=batch.condition,
    )
    mll_loss = -jnp.mean(log_prob)
    key_flow, key_mid = jax.random.split(key)
    t_middle = time_between(key=key_mid, t_init=batch.t_init, t_final=batch.t_final)
    flow_total, flow_12, flow_21 = flow_loss(
        stochastic_flow=model,
        auxiliary_model=auxiliary,
        x_init=batch.x_init,
        t_init=batch.t_init,
        t_middle=t_middle,
        t_final=batch.t_final,
        condition=batch.condition,
        key=key_flow,
        weight_1_to_2=flow_12_weight,
        weight_2_to_1=flow_21_weight,
    )
    flow_loss_mean = jnp.mean(flow_total)
    total_loss = mll_loss + flow_weight * flow_loss_mean
    metrics = {
        "loss": total_loss,
        "neg_log_likelihood": mll_loss,
        "mean_log_prob": jnp.mean(log_prob),
        "flow_loss": flow_loss_mean,
        "flow_1_to_2": jnp.mean(flow_12),
        "flow_2_to_1": jnp.mean(flow_21),
    }
    return total_loss, metrics


def train_step(
    state: TrainState,
    batch: TransitionBatch,
    optimizer: optax.GradientTransformation,
    flow_weight: float,
    flow_12_weight: float,
    flow_21_weight: float,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(train_params):
        model, auxiliary = eqx.combine(train_params, state.static)
        return _loss_and_metrics(
            model=model,
            auxiliary=auxiliary,
            batch=batch,
            flow_weight=flow_weight,
            flow_12_weight=flow_12_weight,
            flow_21_weight=flow_21_weight,
            key=key,
        )

    (loss_value, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = optimizer.update(grads, state.opt_state)
    params = eqx.apply_updates(state.params, updates)
    new_state = TrainState(params=params, static=state.static, opt_state=opt_state)
    metrics = {"loss": loss_value, **metrics}
    return new_state, metrics


def evaluate(
    model: ConditionalNeuralStochasticFlow,
    auxiliary: ConditionalAuxiliaryFlow,
    batch: TransitionBatch,
    key: jax.Array,
) -> dict[str, jax.Array]:
    log_prob = maximum_log_likelihood(
        stochastic_flow=model,
        x_init=batch.x_init,
        x_final=batch.x_final,
        t_init=batch.t_init,
        t_final=batch.t_final,
        condition=batch.condition,
    )
    t_middle = time_between(key=key, t_init=batch.t_init, t_final=batch.t_final)
    flow_total, flow_12, flow_21 = flow_loss(
        stochastic_flow=model,
        auxiliary_model=auxiliary,
        x_init=batch.x_init,
        t_init=batch.t_init,
        t_middle=t_middle,
        t_final=batch.t_final,
        condition=batch.condition,
        key=key,
    )
    return {
        "mean_log_prob": jnp.mean(log_prob),
        "flow_loss": jnp.mean(flow_total),
        "flow_1_to_2": jnp.mean(flow_12),
        "flow_2_to_1": jnp.mean(flow_21),
    }


def train(
    training_config: TrainingConfig,
    flow_config: FlowNetworkConfig,
    aux_config: AuxiliaryConfig,
) -> TrainState:
    dataset = TransitionDataset(training_config.dataset_path)
    optimizer = make_optimizer(training_config)
    if flow_config.state_dim != dataset.state_dim or flow_config.condition_dim != dataset.condition_dim:
        flow_config = replace(flow_config, state_dim=dataset.state_dim, condition_dim=dataset.condition_dim)
    key = jax.random.PRNGKey(training_config.seed)
    key, init_key = jax.random.split(key)

    state = setup_training_state(dataset, flow_config, aux_config, optimizer, init_key)

    wandb_run = wandb.init(
        project=training_config.wandb_project,
        name=training_config.wandb_run_name,
        config={
            **asdict(training_config),
            **asdict(flow_config),
            **asdict(aux_config),
            "state_dim": dataset.state_dim,
            "condition_dim": dataset.condition_dim,
            "num_transitions": dataset.num_transitions,
        },
    )

    key_loop = key
    for step in range(training_config.num_steps + 1):
        key_loop, key_batch, key_loss = jax.random.split(key_loop, 3)
        batch = dataset.sample(key_batch, training_config.batch_size)
        state, metrics = train_step(
            state,
            batch,
            optimizer,
            training_config.flow_weight,
            training_config.flow_12_weight,
            training_config.flow_21_weight,
            key_loss,
        )
        if step % training_config.log_every == 0:
            wandb_run.log({k: float(v) for k, v in metrics.items()}, step=step)

        if step % training_config.eval_every == 0 and step > 0:
            key_loop, key_eval_sample = jax.random.split(key_loop)
            eval_size = min(training_config.eval_samples, dataset.num_transitions)
            eval_batch = dataset.sample(key_eval_sample, eval_size)
            key_loop, key_eval_metrics = jax.random.split(key_loop)
            eval_model, eval_aux = state.combine()
            eval_metrics = evaluate(eval_model, eval_aux, eval_batch, key_eval_metrics)
            wandb_run.log({f"eval/{k}": float(v) for k, v in eval_metrics.items()}, step=step)

    wandb_run.finish()
    return state


__all__ = ["TrainingConfig", "AuxiliaryConfig", "train", "setup_training_state", "make_optimizer"]
