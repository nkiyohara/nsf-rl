"""Encoder models for mapping observations to embeddings.

Encoders transform raw observations into fixed-dimensional embeddings
that are used by the posterior model for inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Encoder(eqx.Module, ABC):
    """Base class for encoders.

    An encoder maps observations to embeddings that are used by the
    posterior model to infer latent states.
    """

    @abstractmethod
    def __call__(
        self,
        obs: Float[Array, "*obs_shape"],
        condition: Optional[Float[Array, "*condition_shape"]] = None,
    ) -> Float[Array, "*embedding_shape"]:
        """Encode an observation into an embedding.

        Args:
            obs: Observation data.
            condition: Optional conditioning vector (e.g., DMP parameters).

        Returns:
            Embedding vector.
        """
        pass


class IdentityEncoder(Encoder):
    """Identity encoder that passes through the observation unchanged."""

    def __call__(
        self,
        obs: Float[Array, "*obs_shape"],
        condition: Optional[Float[Array, "*condition_shape"]] = None,
    ) -> Float[Array, "*obs_shape"]:
        """Return observation unchanged (optionally with condition appended)."""
        if condition is not None:
            return jnp.concatenate([obs.flatten(), condition.flatten()], axis=-1)
        return obs


class FlattenEncoder(Encoder):
    """Encoder that flattens the observation."""

    def __call__(
        self,
        obs: Float[Array, "*obs_shape"],
        condition: Optional[Float[Array, "*condition_shape"]] = None,
    ) -> Float[Array, "flat_dim"]:
        """Flatten observation and optionally append condition."""
        flat = jnp.ravel(obs)
        if condition is not None:
            flat = jnp.concatenate([flat, jnp.ravel(condition)], axis=-1)
        return flat


class MLPEncoder(Encoder):
    """MLP-based encoder.

    Maps observations (and optional conditions) to a fixed-dimensional embedding
    through a multilayer perceptron.

    Attributes:
        net: The MLP network.
        embedding_dim: Dimension of the output embedding.
        use_condition: Whether to expect and use conditioning input.
    """

    net: eqx.nn.MLP
    embedding_dim: int
    use_condition: bool

    def __init__(
        self,
        *,
        obs_dim: int,
        embedding_dim: int,
        condition_dim: int = 0,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = lambda x: x,
        key: PRNGKeyArray,
    ):
        """Initialize the MLP encoder.

        Args:
            obs_dim: Dimension of the observation input.
            embedding_dim: Dimension of the output embedding.
            condition_dim: Dimension of conditioning vector (0 to disable).
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            final_activation: Activation for the output layer.
            key: PRNG key for initialization.
        """
        in_size = obs_dim + condition_dim
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=embedding_dim,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )
        self.embedding_dim = embedding_dim
        self.use_condition = condition_dim > 0

    def __call__(
        self,
        obs: Float[Array, "obs_dim"],
        condition: Optional[Float[Array, "condition_dim"]] = None,
    ) -> Float[Array, "embedding_dim"]:
        """Encode observation to embedding."""
        if self.use_condition:
            if condition is None:
                raise ValueError("Condition required but not provided")
            inputs = jnp.concatenate([obs.flatten(), condition.flatten()], axis=-1)
        else:
            inputs = obs.flatten()

        return self.net(inputs)


class ConditionalMLPEncoder(Encoder):
    """MLP encoder with optional condition injection.

    This encoder allows flexible handling of conditioning:
    - Can work with or without conditions at runtime
    - Conditions are concatenated with observations before encoding

    Attributes:
        net_with_condition: MLP used when condition is provided.
        net_without_condition: MLP used when no condition is provided.
        embedding_dim: Output embedding dimension.
    """

    net_with_condition: Optional[eqx.nn.MLP]
    net_without_condition: eqx.nn.MLP
    embedding_dim: int
    condition_dim: int

    def __init__(
        self,
        *,
        obs_dim: int,
        embedding_dim: int,
        condition_dim: int = 0,
        hidden_size: int = 64,
        depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
    ):
        """Initialize the conditional MLP encoder.

        Args:
            obs_dim: Dimension of observation.
            embedding_dim: Output embedding dimension.
            condition_dim: Dimension of optional condition vector.
            hidden_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNG key.
        """
        key1, key2 = jax.random.split(key)

        # Network without condition (always available)
        self.net_without_condition = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=embedding_dim,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            key=key1,
        )

        # Network with condition (if condition_dim > 0)
        if condition_dim > 0:
            self.net_with_condition = eqx.nn.MLP(
                in_size=obs_dim + condition_dim,
                out_size=embedding_dim,
                width_size=hidden_size,
                depth=depth,
                activation=activation,
                key=key2,
            )
        else:
            self.net_with_condition = None

        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim

    def __call__(
        self,
        obs: Float[Array, "obs_dim"],
        condition: Optional[Float[Array, "condition_dim"]] = None,
    ) -> Float[Array, "embedding_dim"]:
        """Encode observation, optionally using condition."""
        obs_flat = obs.flatten()

        if condition is not None and self.net_with_condition is not None:
            inputs = jnp.concatenate([obs_flat, condition.flatten()], axis=-1)
            return self.net_with_condition(inputs)
        else:
            return self.net_without_condition(obs_flat)


__all__ = [
    "Encoder",
    "IdentityEncoder",
    "FlattenEncoder",
    "MLPEncoder",
    "ConditionalMLPEncoder",
]

