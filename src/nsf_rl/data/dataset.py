"""Dataset utilities for conditional NSF training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


@dataclass
class TransitionBatch:
    x_init: Float[Array, "B D"]
    x_final: Float[Array, "B D"]
    t_init: Float[Array, "B"]
    t_final: Float[Array, "B"]
    condition: Float[Array, "B C"]


class TransitionDataset(eqx.Module):
    observations: Float[Array, "N T D"]
    state_times: Float[Array, "N T"]
    mask: Array
    conditions: Float[Array, "N C"]
    indices: Int[Array, "M 2"]

    def __init__(self, path: Path | str) -> None:
        with np.load(path) as data:
            observations = data["observations"].astype(np.float32)
            state_times = data["state_times"].astype(np.float32)
            mask = data["observation_mask"].astype(bool)
            conditions = data["conditions"].astype(np.float32)
        valid = mask[:, :-1] & mask[:, 1:]
        idx0, idx1 = np.nonzero(valid)
        indices = np.stack([idx0, idx1], axis=-1).astype(np.int32)
        self.observations = jnp.asarray(observations)
        self.state_times = jnp.asarray(state_times)
        self.mask = jnp.asarray(mask)
        self.conditions = jnp.asarray(conditions)
        self.indices = jnp.asarray(indices)

    @property
    def num_transitions(self) -> int:
        return int(self.indices.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def condition_dim(self) -> int:
        return int(self.conditions.shape[-1])

    def sample(self, key: jax.Array, batch_size: int) -> TransitionBatch:
        replace = batch_size > self.indices.shape[0]
        batch_idx = jax.random.choice(key, self.indices.shape[0], shape=(batch_size,), replace=replace)
        idx = self.indices[batch_idx]
        traj_idx = idx[:, 0]
        step_idx = idx[:, 1]
        x_init = self.observations[traj_idx, step_idx]
        x_final = self.observations[traj_idx, step_idx + 1]
        t_init = self.state_times[traj_idx, step_idx]
        t_final = self.state_times[traj_idx, step_idx + 1]
        condition = self.conditions[traj_idx]
        return TransitionBatch(
            x_init=x_init,
            x_final=x_final,
            t_init=t_init,
            t_final=t_final,
            condition=condition,
        )


__all__ = ["TransitionDataset", "TransitionBatch"]
