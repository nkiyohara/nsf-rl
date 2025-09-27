"""Affine coupling layers with optional conditioning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

AffineCouplingScaleFn = str


class Conditioner(eqx.Module):
    net: eqx.nn.MLP
    condition_dim: int

    def __init__(self, *, dim: int, condition_dim: int, width_size: int, depth: int, activation: Callable[[Float[Array, "..."]], Float[Array, "..."]], key: PRNGKeyArray) -> None:
        self.net = eqx.nn.MLP(
            in_size=dim + condition_dim,
            out_size=2 * dim,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,
            key=key,
        )
        self.condition_dim = condition_dim

    def __call__(self, x: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        inputs = x if condition is None else jnp.concatenate([x, condition], axis=-1)
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape((-1, inputs.shape[-1]))
        params = jax.vmap(self.net)(flat_inputs)
        shift, log_scale = jnp.split(params, 2, axis=-1)
        shift = shift.reshape((*batch_shape, -1))
        log_scale = log_scale.reshape((*batch_shape, -1))
        return shift, log_scale


class ContinuousAffineCoupling(eqx.Module):
    mask: Array
    conditioner: Conditioner
    scale_fn_name: AffineCouplingScaleFn
    scale_scale: Optional[Array]
    event_shape: tuple[int, ...]
    event_axes: tuple[int, ...]
    num_event_elements: int

    def __init__(self, *, mask: Array, conditioner: Conditioner, scale_fn: AffineCouplingScaleFn) -> None:
        self.mask = mask.astype(jnp.bool_)
        self.conditioner = conditioner
        self.scale_fn_name = scale_fn
        self.event_shape = tuple(mask.shape)
        self.event_axes = tuple(range(-len(self.event_shape), 0))
        self.num_event_elements = int(np.prod(self.event_shape))
        if self.scale_fn_name == "tanh_exp":
            self.scale_scale = jnp.ones(self.event_shape, dtype=jnp.float32)
        else:
            self.scale_scale = None

    # ------------------------------------------------------------------
    def _broadcast(self, arr: Array, reference: Array) -> Array:
        while arr.ndim < reference.ndim:
            arr = jnp.expand_dims(arr, axis=0)
        target_shape = reference.shape[: -len(self.event_shape)] + self.event_shape
        return jnp.broadcast_to(arr, target_shape)

    def scale_fn(self, s: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        s = jnp.clip(s, -1e5, 1e5)
        if self.scale_fn_name == "exp":
            s_safe = jnp.clip(s, -88.0, 88.0)
            scale = jnp.exp(s_safe)
            return scale, s_safe
        if self.scale_fn_name == "softplus":
            val = jax.nn.softplus(jnp.clip(s, -50.0, 50.0))
            scale = val + 1e-6
            return scale, jnp.log(scale)
        if self.scale_fn_name == "tanh_exp":
            scale_scale = self._broadcast(self.scale_scale, s) if self.scale_scale is not None else 1.0
            scaled = jnp.clip(scale_scale * jnp.tanh(s), -88.0, 88.0)
            scale = jnp.exp(scaled)
            return scale, scaled
        raise ValueError(f"Unknown scale function {self.scale_fn_name}")

    def forward(self, *, x: Float[Array, "..."], time_diff: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> Float[Array, "..."]:
        expanded_dt = time_diff
        for _ in range(len(self.event_shape)):
            expanded_dt = jnp.expand_dims(expanded_dt, axis=-1)

        mask = self._broadcast(self.mask, x)
        x1 = x * mask
        x2 = x * (1 - mask)
        s, t = self.conditioner(x1, condition)
        s = jnp.broadcast_to(s, x.shape)
        t = jnp.broadcast_to(t, x.shape)
        s = s * expanded_dt
        t = t * expanded_dt
        scale, _ = self.scale_fn(s)
        y2 = x2 * scale + t
        return x1 + y2 * (1 - mask)

    def inverse(self, *, y: Float[Array, "..."], time_diff: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> Float[Array, "..."]:
        expanded_dt = time_diff
        for _ in range(len(self.event_shape)):
            expanded_dt = jnp.expand_dims(expanded_dt, axis=-1)
        mask = self._broadcast(self.mask, y)
        y1 = y * mask
        y2 = y * (1 - mask)
        s, t = self.conditioner(y1, condition)
        s = jnp.broadcast_to(s, y.shape) * expanded_dt
        t = jnp.broadcast_to(t, y.shape) * expanded_dt
        scale, _ = self.scale_fn(s)
        x2 = (y2 - t) / scale
        return y1 + x2 * (1 - mask)

    def forward_and_log_det_jacobian(self, *, x: Float[Array, "..."], time_diff: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        expanded_dt = time_diff
        for _ in range(len(self.event_shape)):
            expanded_dt = jnp.expand_dims(expanded_dt, axis=-1)
        mask = self._broadcast(self.mask, x)
        x1 = x * mask
        x2 = x * (1 - mask)
        s, t = self.conditioner(x1, condition)
        s = jnp.broadcast_to(s, x.shape) * expanded_dt
        t = jnp.broadcast_to(t, x.shape) * expanded_dt
        scale, log_scale = self.scale_fn(s)
        y2 = x2 * scale + t
        y = x1 + y2 * (1 - mask)
        log_det = jnp.sum(log_scale * (1 - mask), axis=self.event_axes)
        return y, log_det

    def inverse_and_log_det_jacobian(self, *, y: Float[Array, "..."], time_diff: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        expanded_dt = time_diff
        for _ in range(len(self.event_shape)):
            expanded_dt = jnp.expand_dims(expanded_dt, axis=-1)
        mask = self._broadcast(self.mask, y)
        y1 = y * mask
        y2 = y * (1 - mask)
        s, t = self.conditioner(y1, condition)
        s = jnp.broadcast_to(s, y.shape) * expanded_dt
        t = jnp.broadcast_to(t, y.shape) * expanded_dt
        scale, log_scale = self.scale_fn(s)
        x2 = (y2 - t) / scale
        x = y1 + x2 * (1 - mask)
        log_det = -jnp.sum(log_scale * (1 - mask), axis=self.event_axes)
        return x, log_det


class AffineCoupling(eqx.Module):
    mask: Array
    conditioner: Conditioner
    scale_fn_name: AffineCouplingScaleFn
    scale_scale: Optional[Array]
    event_shape: tuple[int, ...]
    event_axes: tuple[int, ...]
    num_event_elements: int

    def __init__(self, *, mask: Array, conditioner: Conditioner, scale_fn: AffineCouplingScaleFn) -> None:
        self.mask = mask.astype(jnp.bool_)
        self.conditioner = conditioner
        self.scale_fn_name = scale_fn
        self.event_shape = tuple(mask.shape)
        self.event_axes = tuple(range(-len(self.event_shape), 0))
        self.num_event_elements = int(np.prod(self.event_shape))
        if self.scale_fn_name == "tanh_exp":
            self.scale_scale = jnp.ones(self.event_shape, dtype=jnp.float32)
        else:
            self.scale_scale = None

    def _broadcast(self, arr: Array, reference: Array) -> Array:
        while arr.ndim < reference.ndim:
            arr = jnp.expand_dims(arr, axis=0)
        target_shape = reference.shape[: -len(self.event_shape)] + self.event_shape
        return jnp.broadcast_to(arr, target_shape)

    def scale_fn(self, s: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        s = jnp.clip(s, -1e5, 1e5)
        if self.scale_fn_name == "exp":
            s_safe = jnp.clip(s, -88.0, 88.0)
            scale = jnp.exp(s_safe)
            return scale, s_safe
        if self.scale_fn_name == "softplus":
            val = jax.nn.softplus(jnp.clip(s, -50.0, 50.0))
            scale = val + 1e-6
            return scale, jnp.log(scale)
        if self.scale_fn_name == "tanh_exp":
            scale_scale = self._broadcast(self.scale_scale, s) if self.scale_scale is not None else 1.0
            scaled = jnp.clip(scale_scale * jnp.tanh(s), -88.0, 88.0)
            scale = jnp.exp(scaled)
            return scale, scaled
        raise ValueError(f"Unknown scale function {self.scale_fn_name}")

    def forward(self, x: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> Float[Array, "..."]:
        mask = self._broadcast(self.mask, x)
        x1 = x * mask
        x2 = x * (1 - mask)
        s, t = self.conditioner(x1, condition)
        s = jnp.broadcast_to(s, x.shape)
        t = jnp.broadcast_to(t, x.shape)
        scale, _ = self.scale_fn(s)
        y2 = x2 * scale + t
        return x1 + y2 * (1 - mask)

    def inverse(self, y: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None) -> Float[Array, "..."]:
        mask = self._broadcast(self.mask, y)
        y1 = y * mask
        y2 = y * (1 - mask)
        s, t = self.conditioner(y1, condition)
        s = jnp.broadcast_to(s, y.shape)
        t = jnp.broadcast_to(t, y.shape)
        scale, _ = self.scale_fn(s)
        x2 = (y2 - t) / scale
        return y1 + x2 * (1 - mask)

    def forward_and_log_det_jacobian(
        self, x: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        mask = self._broadcast(self.mask, x)
        x1 = x * mask
        x2 = x * (1 - mask)
        s, t = self.conditioner(x1, condition)
        s = jnp.broadcast_to(s, x.shape)
        t = jnp.broadcast_to(t, x.shape)
        scale, log_scale = self.scale_fn(s)
        y2 = x2 * scale + t
        y = x1 + y2 * (1 - mask)
        log_det = jnp.sum(log_scale * (1 - mask), axis=self.event_axes)
        return y, log_det

    def inverse_and_log_det_jacobian(
        self, y: Float[Array, "..."], condition: Optional[Float[Array, "..."]] = None
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        mask = self._broadcast(self.mask, y)
        y1 = y * mask
        y2 = y * (1 - mask)
        s, t = self.conditioner(y1, condition)
        s = jnp.broadcast_to(s, y.shape)
        t = jnp.broadcast_to(t, y.shape)
        scale, log_scale = self.scale_fn(s)
        x2 = (y2 - t) / scale
        x = y1 + x2 * (1 - mask)
        log_det = -jnp.sum(log_scale * (1 - mask), axis=self.event_axes)
        return x, log_det


__all__ = ["Conditioner", "ContinuousAffineCoupling", "AffineCoupling", "AffineCouplingScaleFn"]
