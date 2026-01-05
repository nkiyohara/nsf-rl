"""Stochastic flow models for latent state evolution.

This module provides stochastic flow implementations that model the temporal
evolution of latent states as stochastic differential equations (SDEs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Optional, Union

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


class StochasticFlow(eqx.Module, ABC):
    """Base class for stochastic flows modeling latent state evolution.

    A stochastic flow models p(z_t | z_s) for s < t, representing
    how the latent state evolves over time.
    """

    @abstractmethod
    def __call__(
        self,
        x_init: Float[Array, "*batch D"],
        t_init: Float[Array, "*batch"],
        t_final: Float[Array, "*batch"],
        condition: Optional[Float[Array, "*batch C"]] = None,
    ) -> MultivariateNormalDiag | ContinuousNormalizingFlow:
        """Compute the distribution at t_final given x_init at t_init.

        Args:
            x_init: Initial state of shape (batch_shape, state_dim).
            t_init: Initial time of shape (batch_shape,).
            t_final: Final time of shape (batch_shape,).
            condition: Optional conditioning vector of shape (batch_shape, condition_dim).

        Returns:
            A Distribution representing the state at time t_final.
        """
        pass


class MultivariateNormalDiagStochasticFlow(StochasticFlow):
    """Stochastic flow with diagonal Gaussian transition.

    Models p(z_t | z_s) = N(z_t; μ(z_s, Δt), σ(z_s, Δt)²)
    where Δt = t - s and the mean/variance are predicted by a neural network.

    Attributes:
        net: MLP that predicts mean and scale parameters.
        state_dim: Dimension of the latent state.
        condition_dim: Dimension of the conditioning vector (0 if unconditional).
        autonomous: If True, the flow is time-homogeneous (depends only on Δt).
        eps: Small constant for numerical stability.
    """

    net: eqx.nn.MLP
    state_dim: int
    condition_dim: int
    autonomous: bool
    eps: float = eqx.field(static=True, default=1e-6)
    scale_fn: Callable[[Union[float, Array]], Union[float, Array]] = eqx.field(static=True)

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
        scale_fn: Literal["softplus", "exp"] = "softplus",
    ):
        """Initialize the stochastic flow.

        Args:
            state_dim: Dimension of the latent state.
            condition_dim: Dimension of conditioning vector (0 for unconditional).
            autonomous: If True, depends only on time difference (not absolute time).
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key for initialization.
            eps: Small constant for numerical stability.
            scale_fn: Function to ensure positive scale ("softplus" or "exp").
        """
        # Input: x_init + time_diff + (t_init if not autonomous) + condition
        time_dims = 1 if autonomous else 2  # Δt only, or (Δt, t_init)
        in_size = state_dim + time_dims + condition_dim

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=2 * state_dim,  # mean and log_scale
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

        if scale_fn == "softplus":
            self.scale_fn = jax.nn.softplus
        elif scale_fn == "exp":
            self.scale_fn = jnp.exp
        else:
            raise ValueError(f"Unknown scale function: {scale_fn}")

    def __call__(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        condition: Optional[Float[Array, "..."]] = None,
    ) -> MultivariateNormalDiag:
        """Compute transition distribution."""
        t_init = jnp.atleast_1d(t_init)
        t_final = jnp.atleast_1d(t_final)

        time_diff = (t_final - t_init)[..., None]

        # Build input features
        if self.autonomous:
            inputs = [x_init, time_diff]
        else:
            inputs = [x_init, time_diff, t_init[..., None]]

        if condition is not None:
            inputs.append(condition)

        features = jnp.concatenate(inputs, axis=-1)

        # Handle batched inputs
        batch_shape = features.shape[:-1]
        flat_features = features.reshape((-1, features.shape[-1]))
        params = jax.vmap(self.net)(flat_features)
        params = params.reshape(batch_shape + (2 * self.state_dim,))

        mean_shift, raw_scale = jnp.split(params, 2, axis=-1)

        # Mean: x_init + shift * Δt (linear interpolation baseline)
        mean = x_init + mean_shift * time_diff

        # Scale: proportional to sqrt(Δt) (Brownian motion scaling)
        scale = (self.scale_fn(raw_scale) + self.eps) * jnp.sqrt(jnp.maximum(time_diff, self.eps))

        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


class AffineCouplingStochasticFlow(StochasticFlow):
    """Stochastic flow with affine coupling layers.

    Combines a base Gaussian distribution with affine coupling bijectors
    for more expressive transition distributions.

    Attributes:
        base_flow: Base MultivariateNormalDiag stochastic flow.
        bijectors: List of affine coupling layers.
        state_dim: Dimension of the latent state.
        condition_dim: Dimension of the conditioning vector.
        autonomous: If True, the flow is time-homogeneous.
    """

    base_flow: MultivariateNormalDiagStochasticFlow
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
        """Initialize the affine coupling stochastic flow.

        Args:
            state_dim: Dimension of the latent state.
            condition_dim: Dimension of conditioning vector (0 for unconditional).
            autonomous: If True, depends only on time difference.
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

        # Base Gaussian flow
        self.base_flow = MultivariateNormalDiagStochasticFlow(
            state_dim=state_dim,
            condition_dim=condition_dim,
            autonomous=autonomous,
            hidden_size=mvn_hidden_size,
            depth=mvn_depth,
            activation=mvn_activation,
            key=key_base,
        )

        # Conditioner input: state + time_diff + (t_init if not autonomous) + condition
        time_dims = 1 if autonomous else 2
        flow_condition_dim = state_dim + time_dims + condition_dim

        # Build affine coupling layers with alternating masks
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
            mask = ~mask  # Alternate mask

        self.bijectors = bijectors
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.autonomous = autonomous

    def __call__(
        self,
        x_init: Float[Array, "..."],
        t_init: Float[Array, "..."],
        t_final: Float[Array, "..."],
        condition: Optional[Float[Array, "..."]] = None,
    ) -> ContinuousNormalizingFlow:
        """Compute transition distribution with normalizing flow."""
        t_init = jnp.atleast_1d(t_init)
        t_final = jnp.atleast_1d(t_final)

        # Get base distribution
        base_dist = self.base_flow(x_init, t_init, t_final, condition)

        time_diff = t_final - t_init

        # Build flow condition
        if self.autonomous:
            flow_condition_parts = [x_init, jnp.expand_dims(time_diff, axis=-1)]
        else:
            flow_condition_parts = [
                x_init,
                jnp.expand_dims(time_diff, axis=-1),
                jnp.expand_dims(t_init, axis=-1),
            ]

        if condition is not None:
            flow_condition_parts.append(condition)

        flow_condition = jnp.concatenate(flow_condition_parts, axis=-1)

        return ContinuousNormalizingFlow(
            base_distribution=base_dist,
            bijectors=self.bijectors,
            time_diff=time_diff,
            condition=flow_condition,
        )


__all__ = [
    "StochasticFlow",
    "MultivariateNormalDiagStochasticFlow",
    "AffineCouplingStochasticFlow",
]

