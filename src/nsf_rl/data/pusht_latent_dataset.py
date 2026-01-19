"""PushT dataset for Latent Neural Stochastic Flow.

This module provides a dataset for training Latent NSF on the PushT task.
The observation is partially observed (agent position and velocity only),
while the full state (including block pose) is learned in the latent space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np

from nsf_rl.data.observation import Observation, PushTObservation


PIXEL_LOW = 0.0
PIXEL_HIGH = 512.0
SPAN = PIXEL_HIGH - PIXEL_LOW
HALF_SPAN = SPAN / 2.0
CENTER = PIXEL_LOW + HALF_SPAN


def _to_normalized_pixels(arr: np.ndarray) -> np.ndarray:
    """Normalize pixel coordinates to [-1, 1] range."""
    a = np.asarray(arr, dtype=np.float32)
    return np.clip((a - CENTER) / HALF_SPAN, -1.0, 1.0)


@dataclass
class NormalizationStats:
    """Statistics for standardizing condition vectors."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_data(cls, x: np.ndarray) -> "NormalizationStats":
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-8
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class SequenceBatch:
    """A batch of sequences for training Latent NSF.

    Attributes:
        observations: Observation arrays [batch, T, obs_dim].
        full_states: Full state arrays [batch, T, state_dim] for supervision.
        times: Time array [batch, T] in seconds.
        condition: Condition array [batch, condition_dim].
        z_init: Initial latent state [batch, latent_dim] (if available).
    """

    observations: np.ndarray  # [B, T, obs_dim]
    full_states: np.ndarray  # [B, T, state_dim]
    times: np.ndarray  # [B, T]
    condition: np.ndarray  # [B, C]
    z_init: Optional[np.ndarray] = None  # [B, latent_dim]


class PushTLatentDataset:
    """Dataset for training Latent NSF on PushT task.

    This dataset provides:
    - Partial observations: agent position and velocity (4D)
    - Full states: agent pos/vel + block pose (9D) for supervision
    - Condition: DMP parameters (stiffness, damping, weights, etc.)
    - Time: absolute time in seconds (IRREGULAR sampling)

    The latent NSF learns to:
    1. Infer the hidden block state from partial observations
    2. Model the dynamics of the full state in latent space
    3. Predict future observations from latent states

    **Important**: This dataset uses IRREGULAR time sampling to train
    the model to handle variable time_diff values. This is essential
    for Neural Stochastic Flows to generalize.

    Attributes:
        root: Path to the dataset directory.
        rng: Random number generator for sampling.
        index: List of sample metadata dictionaries.
        seq_len: Fixed sequence length for batching.
        irregular_sampling: Whether to use irregular time sampling.
        min_skip: Minimum frames to skip between observations.
        max_skip: Maximum frames to skip between observations.
        obs_dim: Dimension of observations (4: agent pos + vel).
        state_dim: Dimension of full state (8: agent + block).
    """

    OBS_DIM = 4  # agent_pos (2) + agent_vel (2)
    STATE_DIM = 9  # agent_pos (2) + agent_vel (2) + block_xy (2) + block_sin_cos_theta (2) + 1 extra = 8, but we use 9 for padding

    def __init__(
        self,
        *,
        root: Path,
        rng: np.random.Generator,
        seq_len: int = 16,
        standardize: bool = True,
        stats_path: Optional[Path] = None,
        irregular_sampling: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Path to dataset directory containing index.jsonl and npz files.
            rng: Random number generator.
            seq_len: Length of sequences to sample.
            standardize: Whether to standardize condition vectors.
            stats_path: Optional path to save/load condition statistics.
            irregular_sampling: If True, randomly subsample seq_len frames (variable time intervals).
                               If False, use contiguous regular sampling.
        """
        self.root = Path(root)
        self.rng = rng
        self.index = [json.loads(l) for l in (self.root / "index.jsonl").open()]
        self.seq_len = seq_len
        self.standardize = standardize
        self.stats_path = stats_path
        self._stats: Optional[NormalizationStats] = None
        self.irregular_sampling = irregular_sampling

    def _condition_vector(self, meta: dict) -> np.ndarray:
        """Build condition vector from sample metadata.

        Includes: stiffness, damping, dmp_dt, dmp_alpha_s, n_basis, scale_flag,
                  start_norm(2), goal_norm(2), weights.flatten()

        Note: Duration (tau) is NOT included - the model learns to generalize
        across different durations via the time input.
        """
        stiffness = float(meta["stiffness"])
        damping = float(meta["damping"])
        dmp_dt = float(meta["dmp_dt"])
        dmp_alpha_s = float(meta["dmp_alpha_s"])
        n_basis = int(meta["dmp_n_basis"])
        scale_flag = 1.0 if bool(meta["scale_forcing_by_goal_delta"]) else 0.0
        start_norm = _to_normalized_pixels(meta["start_pixels"])
        goal_norm = _to_normalized_pixels(meta["goal_pixels"])
        weights = np.asarray(meta["weights"], dtype=np.float32)

        cond = [
            np.array(
                [stiffness, damping, dmp_dt, dmp_alpha_s, float(n_basis), float(scale_flag)],
                dtype=np.float32,
            ),
            start_norm.astype(np.float32),
            goal_norm.astype(np.float32),
            weights.reshape(-1).astype(np.float32),
        ]
        return np.concatenate(cond, axis=0)

    def _load_infos(self, info_path: Path) -> dict:
        """Load step info JSON file."""
        return json.loads(info_path.read_text(encoding="utf-8"))

    def _state_from_info(self, info: dict) -> np.ndarray:
        """Extract full state from step info.

        Full state: [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y,
                     block_x, block_y, block_sin_theta, block_cos_theta]
        """
        pos_agent = np.asarray(info.get("pos_agent", [CENTER, CENTER]), dtype=np.float32)
        pos_agent_norm = _to_normalized_pixels(pos_agent[:2])
        vel_agent = np.asarray(info.get("vel_agent", [0.0, 0.0]), dtype=np.float32)
        vel_agent_norm = vel_agent[:2] / HALF_SPAN

        block_pose = np.asarray(info.get("block_pose", [CENTER, CENTER, 0.0]), dtype=np.float32)
        block_xy_norm = _to_normalized_pixels(block_pose[:2])
        theta_rad = np.deg2rad(float(block_pose[2]))
        sin_theta = np.sin(theta_rad).astype(np.float32)
        cos_theta = np.cos(theta_rad).astype(np.float32)

        return np.concatenate(
            [
                pos_agent_norm,
                vel_agent_norm.astype(np.float32),
                block_xy_norm,
                np.array([sin_theta, cos_theta], dtype=np.float32),
            ],
            axis=0,
        )

    def _observation_from_state(self, state: np.ndarray) -> np.ndarray:
        """Extract partial observation from full state.

        Observation: [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y]
        """
        return state[:4]

    def _load_sample_arrays(self, meta: dict) -> tuple[np.ndarray, np.ndarray]:
        """Load time and states from a sample.

        Args:
            meta: Sample metadata dictionary.

        Returns:
            Tuple of (times, states) arrays.
        """
        with np.load(self.root / meta["path"]) as npz:
            time = np.asarray(npz["time"], dtype=np.float32)

        infos = self._load_infos(self.root / meta["info_path"])

        # Build state sequence from step infos
        reset_info = infos["reset_info"]
        step_infos = infos["step_infos"]

        states = [self._state_from_info(reset_info)]
        states.extend(self._state_from_info(info) for info in step_infos)
        states = np.stack(states, axis=0)

        if states.shape[0] != time.shape[0]:
            raise RuntimeError(
                f"Mismatch in sequence lengths for sample {meta.get('id', '?')}: "
                f"time={time.shape[0]}, states={states.shape[0]}"
            )

        return time, states

    def _sample_subsequence(
        self, time: np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a subsequence with irregular time intervals.

        If irregular_sampling is True, samples seq_len observations with
        variable time gaps (controlled by min_skip and max_skip).
        This is essential for training NSF models that need to handle
        variable time_diff values.

        If irregular_sampling is False, returns a contiguous subsequence
        (for debugging or comparison purposes).

        Args:
            time: Full time array [T_full].
            states: Full state array [T_full, state_dim].

        Returns:
            Tuple of (time_subseq, states_subseq) of length seq_len.
        """
        T_full = len(time)

        if not self.irregular_sampling:
            # Regular (contiguous) sampling - NOT RECOMMENDED for NSF training
            if T_full < self.seq_len:
                pad_len = self.seq_len - T_full
                time = np.concatenate([time, np.full(pad_len, time[-1], dtype=np.float32)])
                states = np.concatenate(
                    [states, np.tile(states[-1:], (pad_len, 1))], axis=0
                )
                return time, states

            max_start = T_full - self.seq_len
            start_idx = int(self.rng.integers(0, max_start + 1))
            return (
                time[start_idx : start_idx + self.seq_len],
                states[start_idx : start_idx + self.seq_len],
            )

        # Irregular sampling: random subsampling
        # Simple approach: randomly select seq_len unique indices, then sort
        
        if T_full < self.seq_len:
            # Not enough frames - pad with small time increments
            pad_len = self.seq_len - T_full
            last_time = time[-1]
            dt_pad = 0.001  # Small positive increment to avoid dt=0
            pad_times = last_time + dt_pad * np.arange(1, pad_len + 1, dtype=np.float32)
            time = np.concatenate([time, pad_times])
            states = np.concatenate(
                [states, np.tile(states[-1:], (pad_len, 1))], axis=0
            )
            return time, states

        # Random subsampling: pick seq_len unique indices, sort for temporal order
        indices = np.sort(self.rng.choice(T_full, size=self.seq_len, replace=False))
        
        return time[indices], states[indices]

    def compute_condition_stats(self) -> Optional[NormalizationStats]:
        """Compute and optionally save condition normalization statistics."""
        if not self.standardize:
            return None

        all_cond: list[np.ndarray] = []
        for meta in self.index:
            cond = self._condition_vector(meta)
            all_cond.append(cond)

        cond_mat = np.stack(all_cond, axis=0)
        stats = NormalizationStats.from_data(cond_mat)
        self._stats = stats

        if self.stats_path is not None:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"mean": stats.mean.tolist(), "std": stats.std.tolist()}, fh
                )

        return stats

    def _maybe_load_stats(self) -> None:
        """Load condition statistics if available."""
        if not self.standardize:
            self._stats = None
            return
        if self._stats is not None:
            return
        if self.stats_path is not None and self.stats_path.exists():
            obj = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self._stats = NormalizationStats(
                mean=np.asarray(obj["mean"], dtype=np.float32),
                std=np.asarray(obj["std"], dtype=np.float32),
            )

    def load_sequence(self, idx: int) -> dict:
        """Load a complete sequence for evaluation.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with time, states, observations, condition, and metadata.
        """
        if not (0 <= idx < len(self.index)):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.index)}")

        self._maybe_load_stats()
        meta = self.index[idx]
        time, states = self._load_sample_arrays(meta)

        # Extract observations from states
        observations = np.array([self._observation_from_state(s) for s in states])

        cond = self._condition_vector(meta)
        if self._stats is not None:
            cond = self._stats.apply(cond)

        return {
            "time": time,
            "states": states,
            "observations": observations,
            "condition": cond,
            "meta": meta,
        }

    def sequence_batches(
        self, *, batch_size: int, repeat: bool = True
    ) -> Iterator[SequenceBatch]:
        """Generate batches of fixed-length sequences.

        Args:
            batch_size: Number of sequences per batch.
            repeat: Whether to repeat indefinitely.

        Yields:
            SequenceBatch objects for training.
        """
        self._maybe_load_stats()
        idxs = np.arange(len(self.index))

        while True:
            self.rng.shuffle(idxs)

            for start in range(0, len(idxs), batch_size):
                metas = [self.index[i] for i in idxs[start : start + batch_size]]

                # Collect batch data
                obs_sequences: list[np.ndarray] = []  # List of [seq_len, obs_dim]
                state_sequences: list[np.ndarray] = []  # [batch, seq_len, state_dim]
                time_sequences: list[np.ndarray] = []  # [batch, seq_len]
                conditions: list[np.ndarray] = []  # [batch, cond_dim]

                for meta in metas:
                    time, states = self._load_sample_arrays(meta)
                    time_subseq, states_subseq = self._sample_subsequence(time, states)

                    # Extract observations
                    obs_subseq = np.array(
                        [self._observation_from_state(s) for s in states_subseq]
                    )

                    cond = self._condition_vector(meta)
                    if self._stats is not None:
                        cond = self._stats.apply(cond)

                    obs_sequences.append(obs_subseq)
                    state_sequences.append(states_subseq)
                    time_sequences.append(time_subseq)
                    conditions.append(cond)

                if not obs_sequences:
                    continue

                # Stack into batch arrays
                obs_batch = np.stack(obs_sequences, axis=0)  # [B, T, obs_dim]
                states_batch = np.stack(state_sequences, axis=0)  # [B, T, state_dim]
                times_batch = np.stack(time_sequences, axis=0)  # [B, T]
                cond_batch = np.stack(conditions, axis=0)  # [B, cond_dim]

                yield SequenceBatch(
                    observations=obs_batch,  # [B, T, obs_dim] - array, not list
                    full_states=states_batch,
                    times=times_batch,
                    condition=cond_batch,
                )

            if not repeat:
                break

    @property
    def obs_dim(self) -> int:
        """Dimension of partial observations."""
        return self.OBS_DIM

    @property
    def state_dim(self) -> int:
        """Dimension of full states."""
        return 8  # agent (4) + block (4)

    @property
    def condition_dim(self) -> int:
        """Dimension of condition vector."""
        if len(self.index) == 0:
            return 0
        return len(self._condition_vector(self.index[0]))


__all__ = [
    "PushTLatentDataset",
    "SequenceBatch",
    "NormalizationStats",
]

