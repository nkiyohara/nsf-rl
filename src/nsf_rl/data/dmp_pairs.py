from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np


PIXEL_LOW = 0.0
PIXEL_HIGH = 512.0
SPAN = PIXEL_HIGH - PIXEL_LOW
HALF_SPAN = SPAN / 2.0
CENTER = PIXEL_LOW + HALF_SPAN


def _to_normalized_pixels(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    return np.clip((a - CENTER) / HALF_SPAN, -1.0, 1.0)


@dataclass
class NormalizationStats:
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
class PairBatch:
    x_init: np.ndarray  # [B, 7]
    x_final: np.ndarray  # [B, 7]
    t_init: np.ndarray  # [B]
    t_final: np.ndarray  # [B]
    t_middle: np.ndarray  # [B]
    condition: np.ndarray  # [B, C]


StateSource = Literal["waypoint", "env"]


class DmpPairwiseDataset:
    def __init__(
        self,
        *,
        root: Path,
        rng: np.random.Generator,
        standardize: bool = True,
        stats_path: Optional[Path] = None,
        state_source: StateSource = "waypoint",
    ) -> None:
        self.root = Path(root)
        self.rng = rng
        self.index = [json.loads(l) for l in (self.root / "index.jsonl").open()]  # list of dicts
        self.standardize = standardize
        self.stats_path = stats_path
        self._stats: Optional[NormalizationStats] = None
        if state_source not in ("waypoint", "env"):
            raise ValueError(f"state_source must be 'waypoint' or 'env', got {state_source}")
        self.state_source: StateSource = state_source

    def _condition_vector(self, meta: dict) -> np.ndarray:
        # Order: stiffness, damping, dmp_dt, dmp_alpha_s, n_basis, scale_flag,
        #        start_norm(2), goal_norm(2), weights.flatten(2 * n_basis)
        stiffness = float(meta["stiffness"])  # scalar
        damping = float(meta["damping"])  # scalar
        dmp_dt = float(meta["dmp_dt"])  # scalar
        dmp_alpha_s = float(meta["dmp_alpha_s"])  # scalar
        n_basis = int(meta["dmp_n_basis"])  # scalar (cast to float later)
        scale_flag = 1.0 if bool(meta["scale_forcing_by_goal_delta"]) else 0.0
        start_norm = _to_normalized_pixels(meta["start_pixels"])  # [2]
        goal_norm = _to_normalized_pixels(meta["goal_pixels"])  # [2]
        weights = np.asarray(meta["weights"], dtype=np.float32)  # [2, n_basis]
        cond = [
            np.array([stiffness, damping, dmp_dt, dmp_alpha_s, float(n_basis), float(scale_flag)], dtype=np.float32),
            start_norm.astype(np.float32),
            goal_norm.astype(np.float32),
            weights.reshape(-1).astype(np.float32),
        ]
        return np.concatenate(cond, axis=0)

    def _load_infos(self, info_path: Path) -> dict:
        return json.loads(info_path.read_text(encoding="utf-8"))

    def _sample_pair(self, T_plus_1: int) -> tuple[int, int]:
        # sample 0 <= i < j <= T where T_plus_1 = T+1
        T = T_plus_1 - 1
        if T < 1:
            return 0, 0  # degenerate
        i = int(self.rng.integers(0, T))
        j = int(self.rng.integers(i + 1, T + 1))
        return i, j

    def _state_from_info(self, info: dict) -> np.ndarray:
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

    def _states_from_infos(self, infos: dict) -> np.ndarray:
        reset = infos["reset_info"]
        steps = infos["step_infos"]
        states = [self._state_from_info(reset)]
        states.extend(self._state_from_info(info) for info in steps)
        return np.stack(states, axis=0)

    def _states_from_waypoints(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        if positions.shape != velocities.shape:
            raise ValueError("Waypoint positions and velocities must have matching shapes")
        return np.concatenate([positions.astype(np.float32), velocities.astype(np.float32)], axis=-1)

    def _load_sample_arrays(self, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.root / meta["path"]) as npz:
            time = np.asarray(npz["time"], dtype=np.float32)
            phase = np.asarray(npz["phase"], dtype=np.float32)
            waypoint_positions = np.asarray(npz["waypoints_norm"], dtype=np.float32) if "waypoints_norm" in npz else None
            waypoint_velocities = (
                np.asarray(npz["waypoint_vel_norm"], dtype=np.float32) if "waypoint_vel_norm" in npz else None
            )
        infos = self._load_infos(self.root / meta["info_path"])
        if self.state_source == "waypoint":
            if waypoint_positions is None or waypoint_velocities is None:
                raise RuntimeError(
                    "Waypoint state_source requested but dataset was generated without waypoint arrays. "
                    "Regenerate data or set state_source='env'."
                )
            states = self._states_from_waypoints(waypoint_positions, waypoint_velocities)
        else:
            states = self._states_from_infos(infos)
        if states.shape[0] != time.shape[0] or phase.shape[0] != time.shape[0]:
            raise RuntimeError(
                f"Mismatch in sequence lengths for sample {meta.get('id', '?')}: "
                f"time={time.shape[0]}, phase={phase.shape[0]}, states={states.shape[0]}"
            )
        return time, phase, states

    def load_sequence(self, idx: int) -> dict:
        """Return complete sequences (time, phase, states, condition) for a dataset entry."""
        if not (0 <= idx < len(self.index)):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.index)}")
        self._maybe_load_stats()
        meta = self.index[idx]
        time, phase, states = self._load_sample_arrays(meta)
        cond = self._condition_vector(meta)
        if self._stats is not None:
            cond = self._stats.apply(cond)
        return {
            "time": time,
            "phase": phase,
            "states": states,
            "condition": cond,
            "meta": meta,
        }

    def compute_condition_stats(self) -> Optional[NormalizationStats]:
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
                json.dump({"mean": self._stats.mean.tolist(), "std": self._stats.std.tolist()}, fh)
        return stats

    def _maybe_load_stats(self) -> None:
        if not self.standardize:
            self._stats = None
            return
        if self._stats is not None:
            return
        if self.stats_path is not None and self.stats_path.exists():
            obj = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self._stats = NormalizationStats(mean=np.asarray(obj["mean"], dtype=np.float32), std=np.asarray(obj["std"], dtype=np.float32))

    def batches(self, *, batch_size: int, repeat: bool = True) -> Iterator[PairBatch]:
        self._maybe_load_stats()
        idxs = np.arange(len(self.index))
        while True:
            self.rng.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                metas = [self.index[i] for i in idxs[start : start + batch_size]]
                x_inits: list[np.ndarray] = []
                x_finals: list[np.ndarray] = []
                t_inits: list[float] = []
                t_finals: list[float] = []
                t_middles: list[float] = []
                conds: list[np.ndarray] = []
                for meta in metas:
                    time, phase, states = self._load_sample_arrays(meta)
                    i, j = self._sample_pair(len(time))
                    state_i = states[i]
                    state_j = states[j]
                    x_init = np.concatenate([state_i, np.array([phase[i]], dtype=np.float32)], axis=0)
                    x_final = np.concatenate([state_j, np.array([phase[j]], dtype=np.float32)], axis=0)
                    t_init = float(time[i])
                    t_final = float(time[j])
                    if t_final <= t_init:
                        t_middle = t_init
                    else:
                        t_middle = float(self.rng.uniform(t_init, t_final))
                    cond = self._condition_vector(meta)
                    if self._stats is not None:
                        cond = self._stats.apply(cond)
                    x_inits.append(x_init)
                    x_finals.append(x_final)
                    t_inits.append(t_init)
                    t_finals.append(t_final)
                    t_middles.append(t_middle)
                    conds.append(cond)
                if not x_inits:
                    continue
                batch = PairBatch(
                    x_init=np.stack(x_inits, axis=0),
                    x_final=np.stack(x_finals, axis=0),
                    t_init=np.asarray(t_inits, dtype=np.float32),
                    t_final=np.asarray(t_finals, dtype=np.float32),
                    t_middle=np.asarray(t_middles, dtype=np.float32),
                    condition=np.stack(conds, axis=0),
                )
                yield batch
            if not repeat:
                break
