"""Planar Dynamic Movement Primitives used for sampling PushT policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DMPParams:
    """Runtime parameters describing one planar DMP rollout."""

    duration: float
    start: np.ndarray
    goal: np.ndarray
    weights: np.ndarray
    stiffness: float
    damping: float

    def as_vector(self) -> np.ndarray:
        """Flatten parameters into a single vector for conditioning."""
        flat = [
            np.asarray(self.duration)[None],
            self.start,
            self.goal,
            self.weights.reshape(-1),
            np.asarray(self.stiffness)[None],
            np.asarray(self.damping)[None],
        ]
        return np.concatenate(flat)


@dataclass
class DMPConfig:
    """Sampling configuration for planar DMP policies."""

    n_basis: int = 10
    min_duration: float = 1.8
    max_duration: float = 3.6
    workspace_low: float = 180.0
    workspace_high: float = 440.0
    weight_scale: float = 0.5
    goal_noise: float = 120.0
    start_noise: float = 30.0
    stiffness: float = 25.0
    alpha_s: float = 1.2
    scale_forcing_by_goal_delta: bool = True


class PlanarDMP:
    """Discrete-time planar DMP generator for PushT control."""

    def __init__(self, *, dt: float, config: DMPConfig | None = None) -> None:
        self.dt = float(dt)
        self.config = config or DMPConfig()
        self._centres = np.linspace(0.0, 1.0, self.config.n_basis)
        # Use squared exponential basis with widths chosen so neighbouring centres overlap.
        spacing = np.diff(self._centres).mean() if self.config.n_basis > 1 else 1.0
        self._widths = np.ones_like(self._centres) * (1.0 / (spacing**2))
        self._widths[-1] = self._widths[-2] if self.config.n_basis > 1 else self._widths[-1]
        self._alpha_z = self.config.stiffness
        self._beta_z = self._alpha_z / 4.0
        self._alpha_s = self.config.alpha_s
        self._scale_forcing_by_goal_delta = bool(self.config.scale_forcing_by_goal_delta)

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def sample_parameters(self, rng: np.random.Generator, start: np.ndarray | None = None) -> DMPParams:
        cfg = self.config
        duration = float(rng.uniform(cfg.min_duration, cfg.max_duration))
        workspace_low, workspace_high = cfg.workspace_low, cfg.workspace_high
        margin = 0.1 * (workspace_high - workspace_low)
        start_low = workspace_low + margin
        start_high = workspace_high - margin
        if start is None:
            start = rng.uniform(start_low, start_high, size=2)
            start = np.clip(start, workspace_low, workspace_high)
        start = np.asarray(start, dtype=np.float32)
        span = workspace_high - workspace_low
        radius = rng.uniform(0.3 * span, 0.6 * span)
        angle = rng.uniform(-np.pi, np.pi)
        goal_offset = np.array([np.cos(angle), np.sin(angle)]) * radius
        goal = start + goal_offset
        goal = np.clip(goal, workspace_low, workspace_high).astype(np.float32)

        weights = rng.normal(scale=cfg.weight_scale, size=(2, cfg.n_basis))
        return DMPParams(
            duration=duration,
            start=start.astype(np.float32),
            goal=goal.astype(np.float32),
            weights=weights.astype(np.float32),
            stiffness=cfg.stiffness,
            damping=2.0 * np.sqrt(cfg.stiffness),
        )

    # ------------------------------------------------------------------
    def _rollout_detailed(self, params: DMPParams) -> DMPRollout:
        """Integrate the DMP and expose the full internal signals."""
        timesteps = max(2, int(np.ceil(params.duration / self.dt)))
        y = params.start.astype(np.float32).copy()
        z = np.zeros_like(y)
        s = 1.0
        tau = params.duration
        dt_scaled = self.dt / tau

        raw_positions = np.zeros((timesteps, 2), dtype=np.float32)
        raw_positions[0] = params.start
        velocities = np.zeros((timesteps, 2), dtype=np.float32)
        forcing_terms = np.zeros((timesteps, 2), dtype=np.float32)
        phases = np.zeros(timesteps, dtype=np.float32)
        phases[0] = 1.0
        times = np.arange(timesteps, dtype=np.float32) * self.dt

        for i in range(1, timesteps):
            psi = np.exp(-self._widths * (s - self._centres) ** 2)
            psi_sum = psi.sum() + 1e-6
            forcing = (psi / psi_sum) @ params.weights.T
            if self._scale_forcing_by_goal_delta:
                forcing = forcing * s * (params.goal - params.start)
            else:
                forcing = forcing * s

            z_dot = self._alpha_z * (self._beta_z * (params.goal - y) - z) + forcing
            z = z + z_dot * dt_scaled
            y = y + z * dt_scaled
            velocities[i] = z.astype(np.float32)
            forcing_terms[i] = forcing.astype(np.float32)

            s = s - self._alpha_s * s * dt_scaled
            s = max(s, 0.0)
            phases[i] = np.float32(s)

            y_clipped = np.clip(y, self.config.workspace_low, self.config.workspace_high)
            raw_positions[i] = y_clipped

        positions = raw_positions
        span = self.config.workspace_high - self.config.workspace_low
        base_max_step = 0.18 * span
        decay = np.linspace(1.0, 0.4, timesteps)
        for i in range(1, timesteps - 1):
            delta = positions[i] - positions[i - 1]
            max_step = base_max_step * decay[i]
            norm = float(np.linalg.norm(delta))
            if norm > max_step and norm > 0.0:
                positions[i] = positions[i - 1] + delta * (max_step / norm)
        if timesteps >= 2:
            delta = params.goal - positions[-2]
            max_step = base_max_step * decay[-1]
            norm = float(np.linalg.norm(delta))
            if norm > max_step and norm > 0.0:
                positions[-1] = positions[-2] + delta * (max_step / norm)
            else:
                positions[-1] = params.goal
        positions = np.clip(positions, self.config.workspace_low, self.config.workspace_high)
        positions[0] = params.start

        return DMPRollout(
            positions=positions,
            velocities=velocities,
            canonical_phase=phases,
            forcing=forcing_terms,
            times=times,
        )

    def rollout(self, params: DMPParams) -> tuple[np.ndarray, np.ndarray]:
        """Generate a sequence of Cartesian targets and their timestamps."""
        rollout = self._rollout_detailed(params)
        return rollout.positions, rollout.times

    def rollout_detailed(self, params: DMPParams) -> DMPRollout:
        """Public helper to access the full DMP rollout signals."""
        return self._rollout_detailed(params)

    # ------------------------------------------------------------------
    def condition_vector_size(self) -> int:
        dummy = self.sample_parameters(np.random.default_rng(0))
        return dummy.as_vector().size


def batch_stack(items: Iterable[np.ndarray]) -> np.ndarray:
    """Stack arrays with consistent shapes into a new dimension."""
    stacked = list(items)
    return np.stack(stacked) if stacked else np.empty((0,), dtype=np.float32)


__all__ = ["PlanarDMP", "DMPConfig", "DMPParams", "batch_stack"]
@dataclass
class DMPRollout:
    """Full state of a planar DMP rollout sampled at self.dt."""

    positions: np.ndarray
    velocities: np.ndarray
    canonical_phase: np.ndarray
    forcing: np.ndarray
    times: np.ndarray
