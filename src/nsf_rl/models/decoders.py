"""Decoder models for mapping latent states to observation distributions.

Decoders map latent states back to the observation space, typically
producing a probability distribution over observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.models.distributions import Distribution, MultivariateNormalDiag


class Decoder(eqx.Module, ABC):
    """Base class for decoders.

    A decoder maps a latent state (and optional condition) to a distribution
    over observations.
    """

    @abstractmethod
    def __call__(
        self,
        z: Float[Array, "*batch latent_dim"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Distribution:
        """Decode latent state to observation distribution.

        Args:
            z: Latent state.
            condition: Optional conditioning vector.

        Returns:
            A Distribution over observations.
        """
        pass


class MLPMultivariateNormalDiagDecoder(Decoder):
    """Decoder that outputs a diagonal Gaussian distribution over observations.

    Uses an MLP to predict the mean and diagonal covariance from the latent state.

    Attributes:
        net: MLP network for predicting distribution parameters.
        obs_dim: Dimension of the observation space.
        eps: Small constant for numerical stability.
    """

    net: eqx.nn.MLP
    obs_dim: int
    eps: float = eqx.field(static=True, default=1e-6)
    scale_fn: Callable[[Array], Array] = eqx.field(static=True)
    learn_scale: bool

    def __init__(
        self,
        *,
        latent_dim: int,
        obs_dim: int,
        condition_dim: int = 0,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        eps: float = 1e-6,
        scale_fn: str = "softplus",
        learn_scale: bool = True,
    ):
        """Initialize the MLP decoder.

        Args:
            latent_dim: Dimension of the latent state.
            obs_dim: Dimension of the observation.
            condition_dim: Dimension of conditioning vector (0 to disable).
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key for initialization.
            eps: Small constant for numerical stability.
            scale_fn: Function to ensure positive scale ("softplus" or "exp").
            learn_scale: Whether to learn the scale (if False, uses fixed scale).
        """
        in_size = latent_dim + condition_dim
        out_size = obs_dim * 2 if learn_scale else obs_dim

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.obs_dim = obs_dim
        self.eps = eps
        self.learn_scale = learn_scale

        if scale_fn == "softplus":
            self.scale_fn = jax.nn.softplus
        elif scale_fn == "exp":
            self.scale_fn = jnp.exp
        else:
            raise ValueError(f"Unknown scale function: {scale_fn}")

    def __call__(
        self,
        z: Float[Array, "*batch latent_dim"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Decode latent state to Gaussian distribution over observations."""
        # Build input
        if condition is not None:
            inputs = jnp.concatenate([z, condition], axis=-1)
        else:
            inputs = z

        # Handle batched inputs
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape((-1, inputs.shape[-1]))
        params = jax.vmap(self.net)(flat_inputs)

        if self.learn_scale:
            params = params.reshape(batch_shape + (2 * self.obs_dim,))
            mean, raw_scale = jnp.split(params, 2, axis=-1)
            scale = self.scale_fn(raw_scale) + self.eps
        else:
            params = params.reshape(batch_shape + (self.obs_dim,))
            mean = params
            scale = jnp.ones_like(mean) * 0.1  # Fixed small scale

        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


class LinearDecoder(Decoder):
    """Simple linear decoder.

    Maps latent state to observation mean with a linear transformation.
    Outputs a Gaussian with fixed variance.

    Attributes:
        linear: Linear transformation layer.
        obs_dim: Dimension of observations.
        fixed_scale: Fixed standard deviation for the output distribution.
    """

    linear: eqx.nn.Linear
    obs_dim: int
    fixed_scale: float

    def __init__(
        self,
        *,
        latent_dim: int,
        obs_dim: int,
        condition_dim: int = 0,
        fixed_scale: float = 0.1,
        key: PRNGKeyArray,
    ):
        """Initialize the linear decoder.

        Args:
            latent_dim: Dimension of the latent state.
            obs_dim: Dimension of the observation.
            condition_dim: Dimension of conditioning vector.
            fixed_scale: Fixed standard deviation.
            key: PRNG key for initialization.
        """
        in_size = latent_dim + condition_dim
        self.linear = eqx.nn.Linear(in_size, obs_dim, key=key)
        self.obs_dim = obs_dim
        self.fixed_scale = fixed_scale

    def __call__(
        self,
        z: Float[Array, "*batch latent_dim"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Decode latent state with linear transformation."""
        if condition is not None:
            inputs = jnp.concatenate([z, condition], axis=-1)
        else:
            inputs = z

        # Handle batched inputs
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape((-1, inputs.shape[-1]))
        mean = jax.vmap(self.linear)(flat_inputs)
        mean = mean.reshape(batch_shape + (self.obs_dim,))

        scale = jnp.ones_like(mean) * self.fixed_scale

        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


class ResidualMLPDecoder(Decoder):
    """Decoder with residual connection from latent to observation.

    Useful when the latent space is designed to be similar to the observation space.
    The MLP learns residual corrections.

    Attributes:
        net: MLP for residual prediction.
        obs_dim: Dimension of observations.
        latent_to_obs_indices: Indices mapping latent dims to observation dims.
    """

    net: eqx.nn.MLP
    obs_dim: int
    latent_dim: int
    eps: float = eqx.field(static=True, default=1e-6)

    def __init__(
        self,
        *,
        latent_dim: int,
        obs_dim: int,
        condition_dim: int = 0,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        eps: float = 1e-6,
    ):
        """Initialize the residual MLP decoder.

        Args:
            latent_dim: Dimension of latent state.
            obs_dim: Dimension of observations.
            condition_dim: Dimension of conditioning vector.
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key.
            eps: Small constant for numerical stability.
        """
        in_size = latent_dim + condition_dim
        out_size = 2 * obs_dim  # mean residual + scale

        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.eps = eps

    def __call__(
        self,
        z: Float[Array, "*batch latent_dim"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Decode with residual connection."""
        if condition is not None:
            inputs = jnp.concatenate([z, condition], axis=-1)
        else:
            inputs = z

        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape((-1, inputs.shape[-1]))
        params = jax.vmap(self.net)(flat_inputs)
        params = params.reshape(batch_shape + (2 * self.obs_dim,))

        residual, raw_scale = jnp.split(params, 2, axis=-1)

        # Use first obs_dim latent dimensions as base (identity-like mapping)
        if self.latent_dim >= self.obs_dim:
            base = z[..., : self.obs_dim]
        else:
            # Pad with zeros if latent is smaller
            pad_shape = z.shape[:-1] + (self.obs_dim - self.latent_dim,)
            base = jnp.concatenate([z, jnp.zeros(pad_shape, dtype=z.dtype)], axis=-1)

        mean = base + residual
        scale = jax.nn.softplus(raw_scale) + self.eps

        return MultivariateNormalDiag(loc=mean, scale_diag=scale)


__all__ = [
    "Decoder",
    "MLPMultivariateNormalDiagDecoder",
    "LinearDecoder",
    "ResidualMLPDecoder",
]

