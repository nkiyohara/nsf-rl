"""Probability distributions reused from the reference neural stochastic flow implementation."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, PRNGKeyArray


SCALE_FLOOR = 1e-6


class Distribution(eqx.Module):
    """Base distribution class supporting sampling and log-densities."""

    _event_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._event_shape

    @property
    def event_axes(self) -> tuple[int, ...]:
        return tuple(range(-len(self._event_shape), 0))

    @property
    def num_event_elements(self) -> int:
        return int(jnp.prod(jnp.array(self._event_shape)))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.sample_shape[: -len(self._event_shape)]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def transform(self, eps: Float[Array, "..."]) -> Float[Array, "..."]:
        raise NotImplementedError

    def inverse_transform(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        raise NotImplementedError

    def sample(self, key: PRNGKeyArray) -> Float[Array, "..."]:
        raise NotImplementedError

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        raise NotImplementedError

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        sample = self.sample(key)
        return sample, self.log_prob(sample)


class MultivariateNormalDiag(Distribution):
    loc: Float[Array, "..."]
    _log_scale: Float[Array, "..."]

    def __init__(self, *, loc: Float[Array, "..."], scale_diag: Float[Array, "..."], event_shape: Optional[tuple[int, ...]] = None):
        if loc.shape != scale_diag.shape:
            raise ValueError("`loc` and `scale_diag` must have identical shapes.")
        self.loc = loc
        self._log_scale = jnp.log(jnp.maximum(scale_diag, SCALE_FLOOR))
        self._event_shape = event_shape or (loc.shape[-1],)

    @property
    def scale_diag(self) -> Float[Array, "..."]:
        return jnp.exp(self._log_scale) + SCALE_FLOOR

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.loc.shape

    def _inv_scale(self) -> Float[Array, "..."]:
        return 1.0 / self.scale_diag

    def transform(self, eps: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.loc + self.scale_diag * eps

    def inverse_transform(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        return (value - self.loc) * self._inv_scale()

    def sample(self, key: PRNGKeyArray) -> Float[Array, "..."]:
        eps = jax.random.normal(key, shape=self.loc.shape, dtype=self.loc.dtype)
        return self.transform(eps)

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        diff = (value - self.loc) * self._inv_scale()
        diff = jnp.clip(diff, -1e5, 1e5)
        log_det = jnp.sum(jnp.log(self.scale_diag), axis=self.event_axes)
        return jnp.sum(jsp.stats.norm.logpdf(diff), axis=self.event_axes) - log_det

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        eps = jax.random.normal(key, self.loc.shape, dtype=self.loc.dtype)
        sample = self.transform(eps)
        log_prob = jnp.sum(jsp.stats.norm.logpdf(eps), axis=self.event_axes) - jnp.sum(
            jnp.log(self.scale_diag), axis=self.event_axes
        )
        return sample, log_prob


class ContinuousNormalizingFlow(Distribution):
    base_distribution: Distribution
    bijectors: list
    time_diff: Float[Array, "..."]
    condition: Optional[Float[Array, "..."]]

    def __init__(self, *, base_distribution: Distribution, bijectors: list, time_diff: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> None:
        self.base_distribution = base_distribution
        self.bijectors = bijectors
        self.time_diff = time_diff
        self.condition = condition
        self._event_shape = base_distribution.event_shape

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.base_distribution.sample_shape

    def sample(self, key: PRNGKeyArray) -> Float[Array, "..."]:
        z = self.base_distribution.sample(key)
        for bijector in self.bijectors:
            z = bijector.forward(x=z, time_diff=self.time_diff, condition=self.condition)
        return z

    def transform(self, eps: Float[Array, "..."]) -> Float[Array, "..."]:
        z = self.base_distribution.transform(eps)
        for bijector in self.bijectors:
            z = bijector.forward(x=z, time_diff=self.time_diff, condition=self.condition)
        return z

    def inverse_transform(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        z = value
        for bijector in reversed(self.bijectors):
            z = bijector.inverse(y=z, time_diff=self.time_diff, condition=self.condition)
        return self.base_distribution.inverse_transform(z)

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        z = value
        log_det = jnp.zeros(value.shape[: -len(self.event_shape)])
        for bijector in reversed(self.bijectors):
            z, ldj = bijector.inverse_and_log_det_jacobian(y=z, time_diff=self.time_diff, condition=self.condition)
            log_det += ldj
        base_log_prob = self.base_distribution.log_prob(z)
        return base_log_prob - log_det

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        z, base_log_prob = self.base_distribution.sample_and_log_prob(key)
        log_det = jnp.zeros_like(base_log_prob)
        for bijector in self.bijectors:
            z, ldj = bijector.forward_and_log_det_jacobian(x=z, time_diff=self.time_diff, condition=self.condition)
            log_det += ldj
        return z, base_log_prob - log_det


class NormalizingFlow(Distribution):
    base_distribution: Distribution
    bijectors: list
    condition: Optional[Float[Array, "..."]]

    def __init__(self, *, base_distribution: Distribution, bijectors: list, condition: Optional[Float[Array, "..."]] = None) -> None:
        self.base_distribution = base_distribution
        self.bijectors = bijectors
        self.condition = condition
        self._event_shape = base_distribution.event_shape

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.base_distribution.sample_shape

    def sample(self, key: PRNGKeyArray) -> Float[Array, "..."]:
        z = self.base_distribution.sample(key)
        for bijector in self.bijectors:
            z = bijector.forward(z, condition=self.condition)
        return z

    def transform(self, eps: Float[Array, "..."]) -> Float[Array, "..."]:
        z = self.base_distribution.transform(eps)
        for bijector in self.bijectors:
            z = bijector.forward(z, condition=self.condition)
        return z

    def inverse_transform(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        z = value
        for bijector in reversed(self.bijectors):
            z = bijector.inverse(z, condition=self.condition)
        return self.base_distribution.inverse_transform(z)

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        z = value
        log_det = jnp.zeros(value.shape[: -len(self.event_shape)])
        for bijector in reversed(self.bijectors):
            z, ldj = bijector.inverse_and_log_det_jacobian(z, condition=self.condition)
            log_det += ldj
        base_log_prob = self.base_distribution.log_prob(z)
        return base_log_prob - log_det

    def sample_and_log_prob(self, key: PRNGKeyArray) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        z, base_log_prob = self.base_distribution.sample_and_log_prob(key)
        log_det = jnp.zeros_like(base_log_prob)
        for bijector in self.bijectors:
            z, ldj = bijector.forward_and_log_det_jacobian(z, condition=self.condition)
            log_det += ldj
        return z, base_log_prob - log_det


__all__ = [
    "Distribution",
    "MultivariateNormalDiag",
    "ContinuousNormalizingFlow",
    "NormalizingFlow",
    "SCALE_FLOOR",
]
