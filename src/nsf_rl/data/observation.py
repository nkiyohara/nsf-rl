"""Observation classes for Latent Neural Stochastic Flows.

This module provides abstractions for handling observations in latent variable models,
including support for masked (partially observed) data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class ObservationBase(eqx.Module, ABC):
    """Abstract base class for observation contexts.

    Implementations define specific transformations for different model components.
    This abstraction allows for flexible handling of observations, including
    masked observations for partially observed data.
    """

    @property
    def dtype(self) -> jnp.dtype:
        return self.value.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def encoder_input_shape(self) -> tuple[int, ...]:
        return self.encoder_input.shape

    @property
    @abstractmethod
    def value(self) -> Float[Array, "..."]:
        """Get observation data values. Used for reconstruction loss."""
        pass

    @property
    @abstractmethod
    def encoder_input(self) -> Float[Array, "..."]:
        """Get encoder input data. Used for posterior computation."""
        pass

    @property
    def mask(self) -> Optional[Bool[Array, "..."]]:
        """Get mask data. Used for masked observations."""
        return None

    def __getitem__(self, idx: Any) -> "ObservationBase":
        return jax.tree.map(lambda leaf: leaf[idx], self)


class Observation(ObservationBase):
    """Simple observation that stores raw data.

    Used when observations are fully observed (no masking).

    Attributes:
        _value: The observation data array.
    """

    _value: Float[Array, "*obs_shape"]

    def __init__(self, value: Float[Array, "*obs_shape"]):
        self._value = value

    @property
    def value(self) -> Float[Array, "*obs_shape"]:
        return self._value

    @property
    def encoder_input(self) -> Float[Array, "*obs_shape"]:
        return self._value


class MaskedObservation(ObservationBase):
    """Observation with masking for partially observed data.

    Used when some dimensions or time steps may be missing.
    The mask indicates which values are observed (True) vs missing (False).

    Attributes:
        _value: The observation data array.
        _mask: Boolean mask array (True = observed, False = missing).
    """

    _value: Float[Array, "*obs_shape"]
    _mask: Bool[Array, "*obs_shape"]

    def __init__(
        self,
        value: Float[Array, "*obs_shape"],
        mask: Bool[Array, "*obs_shape"],
    ):
        self._value = value
        self._mask = mask

    @property
    def value(self) -> Float[Array, "*obs_shape"]:
        return self._value

    @property
    def mask(self) -> Bool[Array, "*obs_shape"]:
        return self._mask

    @property
    def encoder_input(self) -> Float[Array, "*obs_shape"]:
        """Return masked observation (unobserved values zeroed out)."""
        return self._value * self._mask


class PushTObservation(ObservationBase):
    """Observation class for PushT environment.

    Stores agent observation (position, velocity) which is the observed part,
    while block state is hidden (inferred via latent state).

    Attributes:
        _agent_pos: Agent position [x, y].
        _agent_vel: Agent velocity [vx, vy].
    """

    _agent_pos: Float[Array, "... 2"]
    _agent_vel: Float[Array, "... 2"]

    def __init__(
        self,
        agent_pos: Float[Array, "... 2"],
        agent_vel: Float[Array, "... 2"],
    ):
        self._agent_pos = agent_pos
        self._agent_vel = agent_vel

    @property
    def value(self) -> Float[Array, "... 4"]:
        """Full observation vector [pos_x, pos_y, vel_x, vel_y]."""
        return jnp.concatenate([self._agent_pos, self._agent_vel], axis=-1)

    @property
    def encoder_input(self) -> Float[Array, "... 4"]:
        """Encoder input is same as value for PushT."""
        return self.value

    @property
    def agent_pos(self) -> Float[Array, "... 2"]:
        return self._agent_pos

    @property
    def agent_vel(self) -> Float[Array, "... 2"]:
        return self._agent_vel


__all__ = [
    "ObservationBase",
    "Observation",
    "MaskedObservation",
    "PushTObservation",
]

