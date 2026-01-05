"""Bridge models for flow loss computation.

Bridge models provide an auxiliary distribution q(x_middle | x_init, x_final, t_init, t_middle, t_final)
used to compute the bidirectional flow consistency loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

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


class BridgeModel(eqx.Module, ABC):
    """Base class for bridge models.

    A bridge model approximates the distribution of intermediate states
    given the initial and final states: q(x_t | x_0, x_T, t).
    """

    @abstractmethod
    def __call__(
        self,
        x_init: Float[Array, "*batch D"],
        t_init: Float[Array, "*batch"],
        x_final: Float[Array, "*batch D"],
        t_final: Float[Array, "*batch"],
        t: Float[Array, "*batch"],
        condition: Optional[Float[Array, "*batch C"]] = None,
    ) -> MultivariateNormalDiag | ContinuousNormalizingFlow:
        """Compute the bridge distribution at time t.

        Args:
            x_init: Initial state.
            t_init: Initial time.
            x_final: Final state.
            t_final: Final time.
            t: Query time (t_init < t < t_final).
            condition: Optional conditioning vector.

        Returns:
            Distribution over the state at time t.
        """
        pass


class MultivariateNormalDiagBridgeModel(BridgeModel):
    """Bridge model with diagonal Gaussian distribution.

    The bridge distribution is parameterized as a Gaussian whose mean
    interpolates between x_init and x_final, with learned corrections.
    """

    net: eqx.nn.MLP
    state_dim: int
    condition_dim: int
    autonomous: bool
    eps: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        *,
        state_dim: int,
        condition_dim: int = 0,
        autonomous: bool = True,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        eps: float = 1e-6,
    ):
        """Initialize the bridge model.

        Args:
            state_dim: Dimension of the state.
            condition_dim: Dimension of conditioning vector.
            autonomous: If True, uses only time differences.
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key for initialization.
            eps: Small constant for numerical stability.
        """
        # Input: x_init, x_final, time_features, condition
        # Time features: (t - t_init), (t_final - t), (t_final - t_init)
        # Plus t_init if not autonomous
        time_dims = 3 if autonomous else 4
        in_size = 2 * state_dim + time_dims + condition_dim

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=2 * state_dim,  # mean and scale
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.autonomous = autonomous
        self.eps = eps

    def __call__(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        x_final: Float[Array, "..."],
        t_final: Float[Array, "..."],
        t: Float[Array, "..."],
        condition: Optional[Float[Array, "..."]] = None,
    ) -> MultivariateNormalDiag:
        """Compute bridge distribution at time t."""
        t_init = jnp.atleast_1d(t_init)
        t_final = jnp.atleast_1d(t_final)
        t = jnp.atleast_1d(t)

        # Time features
        dt_init = (t - t_init)[..., None]  # time since init
        dt_final = (t_final - t)[..., None]  # time until final
        dt_total = (t_final - t_init)[..., None]  # total interval

        # Build input
        if self.autonomous:
            time_features = jnp.concatenate([dt_init, dt_final, dt_total], axis=-1)
        else:
            time_features = jnp.concatenate(
                [dt_init, dt_final, dt_total, t_init[..., None]], axis=-1
            )

        inputs = [x_init, x_final, time_features]
        if condition is not None:
            inputs.append(condition)

        features = jnp.concatenate(inputs, axis=-1)

        # Handle batched inputs
        batch_shape = features.shape[:-1]
        flat_features = features.reshape((-1, features.shape[-1]))
        params = jax.vmap(self.net)(flat_features)
        params = params.reshape(batch_shape + (2 * self.state_dim,))

        mean_shift, raw_scale = jnp.split(params, 2, axis=-1)

        # Linear interpolation baseline
        ratio = dt_init / jnp.maximum(dt_total, self.eps)
        base_mean = x_init + ratio * (x_final - x_init)

        # Add learned correction
        mean = base_mean + mean_shift * dt_init * dt_final / jnp.maximum(dt_total, self.eps)

        # Scale proportional to geometric mean of time intervals
        scale = jax.nn.softplus(raw_scale) + self.eps
        scale = scale * jnp.sqrt(dt_init * dt_final / jnp.maximum(dt_total, self.eps) + self.eps)

        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


class AffineCouplingBridgeModel(BridgeModel):
    """Bridge model with affine coupling layers for more expressive distributions."""

    base_bridge: MultivariateNormalDiagBridgeModel
    bijectors: list[ContinuousAffineCoupling]
    state_dim: int
    condition_dim: int
    autonomous: bool

    def __init__(
        self,
        *,
        state_dim: int,
        condition_dim: int = 0,
        autonomous: bool = True,
        mvn_hidden_size: int = 64,
        mvn_depth: int = 2,
        mvn_activation: Callable = jax.nn.tanh,
        conditioner_hidden_size: int = 64,
        conditioner_depth: int = 2,
        conditioner_activation: Callable = jax.nn.tanh,
        num_flow_layers: int = 4,
        scale_fn: AffineCouplingScaleFn = "tanh_exp",
        key: PRNGKeyArray,
    ):
        """Initialize the affine coupling bridge model.

        Args:
            state_dim: Dimension of the state.
            condition_dim: Dimension of conditioning vector.
            autonomous: If True, uses only time differences.
            mvn_hidden_size: Hidden size for base MVN network.
            mvn_depth: Depth of base MVN network.
            mvn_activation: Activation for base MVN network.
            conditioner_hidden_size: Hidden size for conditioner networks.
            conditioner_depth: Depth of conditioner networks.
            conditioner_activation: Activation for conditioner networks.
            num_flow_layers: Number of affine coupling layers.
            scale_fn: Scale function for affine coupling.
            key: PRNG key for initialization.
        """
        key_base, *bijector_keys = jax.random.split(key, num_flow_layers + 1)

        self.base_bridge = MultivariateNormalDiagBridgeModel(
            state_dim=state_dim,
            condition_dim=condition_dim,
            autonomous=autonomous,
            hidden_size=mvn_hidden_size,
            depth=mvn_depth,
            activation=mvn_activation,
            key=key_base,
        )

        # Conditioner input: x_init, x_final, time_features, condition
        time_dims = 3 if autonomous else 4
        flow_condition_dim = 2 * state_dim + time_dims + condition_dim

        # Build affine coupling layers
        mask = (jnp.arange(state_dim) % 2).astype(jnp.bool_)
        bijectors: list[ContinuousAffineCoupling] = []

        for k in bijector_keys:
            conditioner = Conditioner(
                dim=state_dim,
                condition_dim=flow_condition_dim,
                width_size=conditioner_hidden_size,
                depth=conditioner_depth,
                activation=conditioner_activation,
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
        self.condition_dim = condition_dim
        self.autonomous = autonomous

    def __call__(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        x_final: Float[Array, "..."],
        t_final: Float[Array, "..."],
        t: Float[Array, "..."],
        condition: Optional[Float[Array, "..."]] = None,
    ) -> ContinuousNormalizingFlow:
        """Compute bridge distribution with normalizing flow."""
        t_init = jnp.atleast_1d(t_init)
        t_final = jnp.atleast_1d(t_final)
        t = jnp.atleast_1d(t)

        # Get base distribution
        base_dist = self.base_bridge(x_init, t_init, x_final, t_final, t, condition)

        # Time features for flow condition
        dt_init = (t - t_init)[..., None]
        dt_final = (t_final - t)[..., None]
        dt_total = (t_final - t_init)[..., None]

        if self.autonomous:
            time_features = jnp.concatenate([dt_init, dt_final, dt_total], axis=-1)
        else:
            time_features = jnp.concatenate(
                [dt_init, dt_final, dt_total, t_init[..., None]], axis=-1
            )

        flow_condition_parts = [x_init, x_final, time_features]
        if condition is not None:
            flow_condition_parts.append(condition)

        flow_condition = jnp.concatenate(flow_condition_parts, axis=-1)

        # Time scaling factor: α(1-α) where α = (t - t_init) / (t_final - t_init)
        alpha = dt_init / jnp.maximum(dt_total, 1e-6)
        time_scale = jnp.clip(alpha * (1.0 - alpha), a_min=0.0).squeeze(-1)

        return ContinuousNormalizingFlow(
            base_distribution=base_dist,
            bijectors=self.bijectors,
            time_diff=time_scale,
            condition=flow_condition,
        )


__all__ = [
    "BridgeModel",
    "MultivariateNormalDiagBridgeModel",
    "AffineCouplingBridgeModel",
]

