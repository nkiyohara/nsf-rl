"""Conditional neural stochastic flow model for PushT."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.models.bijectors import (
    AffineCouplingScaleFn,
    Conditioner,
    ContinuousAffineCoupling,
)
from nsf_rl.models.distributions import ContinuousNormalizingFlow, MultivariateNormalDiag


def _apply_mlp(mlp: eqx.nn.MLP, inputs: Array) -> Array:
    flat = inputs.reshape((-1, inputs.shape[-1]))
    outputs = jax.vmap(mlp)(flat)
    return outputs.reshape(inputs.shape[:-1] + (outputs.shape[-1],))


@dataclass
class FlowNetworkConfig:
    state_dim: int = 0
    condition_dim: int = 0
    hidden_size: int = 128
    depth: int = 2
    activation: str = "tanh"
    num_flow_layers: int = 4
    conditioner_hidden_size: int = 128
    conditioner_depth: int = 2
    scale_fn: AffineCouplingScaleFn = "tanh_exp"
    include_initial_time: bool = False


def _activation(name: str):
    mapping = {
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "elu": jax.nn.elu,
        "swish": jax.nn.swish,
        "softplus": jax.nn.softplus,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation {name}")
    return mapping[name]


class ConditionalGaussianTransition(eqx.Module):
    net: eqx.nn.MLP
    state_dim: int
    condition_dim: int
    include_initial_time: bool
    eps: float = eqx.field(static=True, default=1e-4)

    def __init__(self, *, state_dim: int, condition_dim: int, hidden_size: int, depth: int, activation, include_initial_time: bool, key: PRNGKeyArray) -> None:
        extra = 1 + (1 if include_initial_time else 0)
        self.net = eqx.nn.MLP(
            in_size=state_dim + condition_dim + extra,
            out_size=2 * state_dim,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.include_initial_time = include_initial_time

    def __call__(self, x_init: Float[Array, "..."], t_init: Float[Array, "..."], t_final: Float[Array, "..."], condition: Float[Array, "..."]) -> MultivariateNormalDiag:
        time_diff = (t_final - t_init)[..., None]
        inputs = [x_init, condition, time_diff]
        if self.include_initial_time:
            inputs.append(t_init[..., None])
        features = jnp.concatenate(inputs, axis=-1)
        params = _apply_mlp(self.net, features)
        mean_shift, raw_scale = jnp.split(params, 2, axis=-1)
        mean = x_init + mean_shift * time_diff
        scale = jax.nn.softplus(raw_scale) + self.eps
        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


class ConditionalAuxiliaryGaussian(eqx.Module):
    net: eqx.nn.MLP
    state_dim: int
    condition_dim: int
    include_initial_time: bool
    include_time_ratio: bool
    eps: float = eqx.field(static=True, default=1e-4)
    feature_dim: int = eqx.field(static=True)

    def __init__(self, *, state_dim: int, condition_dim: int, hidden_size: int, depth: int, activation, include_initial_time: bool, include_time_ratio: bool, key: PRNGKeyArray) -> None:
        feature_dim = 2 * state_dim + condition_dim + 1
        if include_initial_time:
            feature_dim += 1
        if include_time_ratio:
            feature_dim += 1
        self.net = eqx.nn.MLP(
            in_size=feature_dim,
            out_size=2 * state_dim,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.include_initial_time = include_initial_time
        self.include_time_ratio = include_time_ratio
        self.feature_dim = feature_dim

    def _features(
        self,
        x_init: Float[Array, "..."],
        x_final: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        t_eval: Float[Array, "..."],
        condition: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        time_diff = (t_final - t_init)[..., None]
        denom = jnp.maximum(time_diff.squeeze(-1), 1e-6)
        ratio = ((t_eval - t_init) / denom)[..., None]
        base = x_init + ratio * (x_final - x_init)
        parts = [x_init, x_final, condition, time_diff]
        if self.include_initial_time:
            parts.append(t_init[..., None])
        if self.include_time_ratio:
            parts.append(ratio)
        features = jnp.concatenate(parts, axis=-1)
        return features, base, time_diff, ratio

    def _distribution(
        self,
        features: Float[Array, "..."],
        base: Float[Array, "..."],
        time_diff: Float[Array, "..."],
    ) -> MultivariateNormalDiag:
        params = _apply_mlp(self.net, features)
        mean_shift, raw_scale = jnp.split(params, 2, axis=-1)
        mean = base + mean_shift * time_diff
        scale = jax.nn.softplus(raw_scale) + self.eps
        return MultivariateNormalDiag(loc=mean, scale_diag=scale)

    def __call__(self, x_init: Float[Array, "..."], x_final: Float[Array, "..."], t_init: Float[Array, "..."], t_final: Float[Array, "..."], t_eval: Float[Array, "..."], condition: Float[Array, "..."]) -> MultivariateNormalDiag:
        features, base, time_diff, _ = self._features(x_init, x_final, t_init, t_final, t_eval, condition)
        return self._distribution(features, base, time_diff)


class ConditionalAuxiliaryFlow(eqx.Module):
    base: ConditionalAuxiliaryGaussian
    bijectors: list[ContinuousAffineCoupling]
    state_dim: int
    feature_dim: int

    def __init__(
        self,
        *,
        state_dim: int,
        condition_dim: int,
        hidden_size: int,
        depth: int,
        activation,
        include_initial_time: bool,
        include_time_ratio: bool,
        conditioner_hidden_size: int,
        conditioner_depth: int,
        num_flow_layers: int,
        scale_fn: AffineCouplingScaleFn,
        key: PRNGKeyArray,
    ) -> None:
        key_base, *bijector_keys = jax.random.split(key, num_flow_layers + 1)
        self.base = ConditionalAuxiliaryGaussian(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_size=hidden_size,
            depth=depth,
            activation=activation,
            include_initial_time=include_initial_time,
            include_time_ratio=include_time_ratio,
            key=key_base,
        )
        mask = (jnp.arange(state_dim) % 2).astype(jnp.bool_)
        bijectors: list[ContinuousAffineCoupling] = []
        for k in bijector_keys:
            conditioner = Conditioner(
                dim=state_dim,
                condition_dim=self.base.feature_dim,
                width_size=conditioner_hidden_size,
                depth=conditioner_depth,
                activation=activation,
                key=k,
            )
            bijectors.append(
                ContinuousAffineCoupling(
                    mask=mask,
                    conditioner=conditioner,
                    scale_fn=scale_fn,
                )
            )
            mask = ~mask
        self.bijectors = bijectors
        self.state_dim = state_dim
        self.feature_dim = self.base.feature_dim

    def __call__(
        self,
        x_init: Float[Array, "..."],
        x_final: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        t_eval: Float[Array, "..."],
        condition: Float[Array, "..."],
    ) -> ContinuousNormalizingFlow:
        features, base, time_diff, ratio = self.base._features(x_init, x_final, t_init, t_final, t_eval, condition)
        base_dist = self.base._distribution(features, base, time_diff)
        alpha = jnp.clip((ratio * (1.0 - ratio)).squeeze(-1), a_min=0.0)
        return ContinuousNormalizingFlow(
            base_distribution=base_dist,
            bijectors=self.bijectors,
            time_diff=alpha,
            condition=features,
        )


class ConditionalNeuralStochasticFlow(eqx.Module):
    base: ConditionalGaussianTransition
    bijectors: list[ContinuousAffineCoupling]
    condition_dim: int
    include_initial_time: bool
    state_dim: int

    def __init__(
        self,
        *,
        state_dim: int,
        condition_dim: int,
        hidden_size: int,
        depth: int,
        activation: str,
        num_flow_layers: int,
        conditioner_hidden_size: int,
        conditioner_depth: int,
        scale_fn: AffineCouplingScaleFn,
        include_initial_time: bool,
        key: PRNGKeyArray,
    ) -> None:
        if state_dim <= 0 or condition_dim <= 0:
            raise ValueError("state_dim and condition_dim must be positive")
        activation_fn = _activation(activation)
        key_base, *bijector_keys = jax.random.split(key, num_flow_layers + 1)
        self.base = ConditionalGaussianTransition(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_size=hidden_size,
            depth=depth,
            activation=activation_fn,
            include_initial_time=include_initial_time,
            key=key_base,
        )
        mask = (jnp.arange(state_dim) % 2).astype(jnp.bool_)
        self.bijectors = []
        conditioner_dim = state_dim + condition_dim + 1 + (1 if include_initial_time else 0)
        for k in bijector_keys:
            conditioner = Conditioner(
                dim=state_dim,
                condition_dim=conditioner_dim,
                width_size=conditioner_hidden_size,
                depth=conditioner_depth,
                activation=activation_fn,
                key=k,
            )
            self.bijectors.append(
                ContinuousAffineCoupling(
                    mask=mask,
                    conditioner=conditioner,
                    scale_fn=scale_fn,
                )
            )
            mask = ~mask
        self.condition_dim = condition_dim
        self.include_initial_time = include_initial_time
        self.state_dim = state_dim

    def _flow_condition(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        condition: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        time_diff = (t_final - t_init)[..., None]
        parts = [x_init, condition, time_diff]
        if self.include_initial_time:
            parts.append(t_init[..., None])
        return jnp.concatenate(parts, axis=-1)

    def __call__(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        condition: Float[Array, "..."],
    ) -> ContinuousNormalizingFlow:
        base_dist = self.base(x_init=x_init, t_init=t_init, t_final=t_final, condition=condition)
        flow_condition = self._flow_condition(x_init, t_init, t_final, condition)
        time_diff = jnp.maximum(t_final - t_init, 0.0)
        return ContinuousNormalizingFlow(
            base_distribution=base_dist,
            bijectors=self.bijectors,
            time_diff=time_diff,
            condition=flow_condition,
        )


__all__ = [
    "FlowNetworkConfig",
    "ConditionalNeuralStochasticFlow",
    "ConditionalAuxiliaryGaussian",
    "ConditionalAuxiliaryFlow",
]
