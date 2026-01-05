"""MLP-based posterior models.

These posteriors use MLPs to predict posterior distribution parameters
from encoded observations and prior distributions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.models.distributions import Distribution, MultivariateNormalDiag
from nsf_rl.models.posteriors.base import PosteriorCarry, PriorConditionedPosterior


class MLPPosterior(PriorConditionedPosterior):
    """MLP-based posterior that predicts mean and scale directly.

    Takes the embedding, prior parameters, time_diff, and optional condition as input,
    and outputs posterior mean and scale.

    Attributes:
        net: MLP network for parameter prediction.
        latent_dim: Dimension of the latent state.
        eps: Small constant for numerical stability.
        autonomous: If True, only uses time_diff; if False, also uses absolute time.
    """

    net: eqx.nn.MLP
    latent_dim: int
    autonomous: bool
    eps: float = eqx.field(static=True, default=1e-6)
    scale_fn: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        condition_dim: int = 0,
        autonomous: bool = True,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        eps: float = 1e-6,
        scale_fn: str = "softplus",
    ):
        """Initialize the MLP posterior.

        Args:
            embedding_dim: Dimension of the encoder embedding.
            latent_dim: Dimension of the latent state.
            condition_dim: Dimension of conditioning vector.
            autonomous: If True, only time_diff is used; if False, also uses abs time.
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key for initialization.
            eps: Small constant for numerical stability.
            scale_fn: Function to ensure positive scale.
        """
        # Input: embedding + prior_mean + prior_scale + time_diff + (time if not autonomous) + condition
        time_dims = 1 if autonomous else 2
        in_size = embedding_dim + 2 * latent_dim + time_dims + condition_dim

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=2 * latent_dim,  # mean and scale
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.latent_dim = latent_dim
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
        embedding: Float[Array, "*batch embedding_dim"],
        prior: Distribution,
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Compute posterior distribution (stateless, assumes time_diff=0).

        For proper sequential inference, use step() instead.
        """
        # Default time_diff = 0 for backward compatibility
        time_diff = jnp.zeros(())
        time = jnp.zeros(())
        prior_initial_state = prior.loc
        dist, _ = self.step(
            prior_dist=prior,
            embedding=embedding,
            time=time,
            prior_initial_state=prior_initial_state,
            time_diff=time_diff,
            condition=condition,
            carry=None,
        )
        return dist

    def step(
        self,
        prior_dist: Distribution,
        embedding: Float[Array, "embedding_dim"],
        time: Float[Array, ""],
        prior_initial_state: Float[Array, "state_dim"],
        time_diff: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[MultivariateNormalDiag, None]:
        """Compute posterior at a single time step."""
        # Extract prior parameters
        prior_mean = prior_dist.loc
        prior_scale = prior_dist.scale_diag

        # Build input features
        if self.autonomous:
            time_features = jnp.atleast_1d(time_diff)
        else:
            time_features = jnp.concatenate([jnp.atleast_1d(time_diff), jnp.atleast_1d(time)], axis=-1)

        inputs = [embedding, prior_mean, prior_scale, time_features]
        if condition is not None:
            inputs.append(condition)

        features = jnp.concatenate(inputs, axis=-1)

        # Handle batched inputs
        if features.ndim > 1:
            batch_shape = features.shape[:-1]
            flat_features = features.reshape((-1, features.shape[-1]))
            params = jax.vmap(self.net)(flat_features)
            params = params.reshape(batch_shape + (2 * self.latent_dim,))
        else:
            params = self.net(features)

        mean, raw_scale = jnp.split(params, 2, axis=-1)
        scale = self.scale_fn(raw_scale) + self.eps

        return MultivariateNormalDiag(loc=mean, scale_diag=scale), None


class MLPResidualPosterior(PriorConditionedPosterior):
    """MLP posterior with residual connection to the prior.

    Predicts adjustments to the prior mean rather than the posterior mean directly.
    This can help with training stability when the posterior should be close to the prior.

    Attributes:
        net: MLP network for residual prediction.
        latent_dim: Dimension of the latent state.
        eps: Small constant for numerical stability.
        scale_factor: Factor to scale the residual correction.
        autonomous: If True, only uses time_diff; if False, also uses absolute time.
    """

    net: eqx.nn.MLP
    latent_dim: int
    autonomous: bool
    eps: float = eqx.field(static=True, default=1e-6)
    scale_eps: float = eqx.field(static=True, default=1e-2)
    scale_fn: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        condition_dim: int = 0,
        autonomous: bool = True,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        eps: float = 1e-6,
        scale_eps: float = 1e-2,
        scale_fn: str = "softplus",
    ):
        """Initialize the residual MLP posterior.

        Args:
            embedding_dim: Dimension of the encoder embedding.
            latent_dim: Dimension of the latent state.
            condition_dim: Dimension of conditioning vector.
            autonomous: If True, only time_diff is used; if False, also uses abs time.
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key for initialization.
            eps: Small constant for numerical stability.
            scale_eps: Epsilon added to scale.
            scale_fn: Function to ensure positive scale.
        """
        # Input: embedding + prior_initial_state + time_diff + (time if not autonomous) + condition
        time_dims = 1 if autonomous else 2
        in_size = embedding_dim + latent_dim + time_dims + condition_dim

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=2 * latent_dim,  # mean residual and scale
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.latent_dim = latent_dim
        self.autonomous = autonomous
        self.eps = eps
        self.scale_eps = scale_eps

        if scale_fn == "softplus":
            self.scale_fn = jax.nn.softplus
        elif scale_fn == "exp":
            self.scale_fn = jnp.exp
        else:
            raise ValueError(f"Unknown scale function: {scale_fn}")

    def __call__(
        self,
        embedding: Float[Array, "*batch embedding_dim"],
        prior: Distribution,
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Compute posterior distribution (stateless, assumes time_diff=0).

        For proper sequential inference, use step() instead.
        """
        time_diff = jnp.zeros(())
        time = jnp.zeros(())
        prior_initial_state = prior.loc
        dist, _ = self.step(
            prior_dist=prior,
            embedding=embedding,
            time=time,
            prior_initial_state=prior_initial_state,
            time_diff=time_diff,
            condition=condition,
            carry=None,
        )
        return dist

    def step(
        self,
        prior_dist: Distribution,
        embedding: Float[Array, "embedding_dim"],
        time: Float[Array, ""],
        prior_initial_state: Float[Array, "state_dim"],
        time_diff: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[MultivariateNormalDiag, None]:
        """Compute posterior with residual connection to prior."""
        prior_mean = prior_dist.loc
        prior_scale = prior_dist.scale_diag

        # Build input features
        if self.autonomous:
            time_features = jnp.atleast_1d(time_diff)
        else:
            time_features = jnp.concatenate([jnp.atleast_1d(time_diff), jnp.atleast_1d(time)], axis=-1)

        inputs = [embedding, prior_initial_state, time_features]
        if condition is not None:
            inputs.append(condition)

        features = jnp.concatenate(inputs, axis=-1)

        # Handle batched inputs
        if features.ndim > 1:
            batch_shape = features.shape[:-1]
            flat_features = features.reshape((-1, features.shape[-1]))
            params = jax.vmap(self.net)(flat_features)
            params = params.reshape(batch_shape + (2 * self.latent_dim,))
        else:
            params = self.net(features)

        mean_residual, log_scale = jnp.split(params, 2, axis=-1)

        # Posterior mean = prior mean + residual
        mean = prior_mean + mean_residual

        # Posterior scale: prior_scale * learned_scale_factor
        scale = prior_scale * self.scale_fn(log_scale) + self.scale_eps

        return MultivariateNormalDiag(loc=mean, scale_diag=scale), None


__all__ = [
    "MLPPosterior",
    "MLPResidualPosterior",
]
