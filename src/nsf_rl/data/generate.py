"""Utility to generate PuSHt rollouts conditioned on DMP parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import gym_pusht  # noqa: F401 -- registers environments
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from nsf_rl.dmp import DMPConfig, PlanarDMP
from nsf_rl.utils.pymunk_compat import ensure_add_collision_handler


@dataclass
class DatasetConfig:
    """Configuration for the PuSHt experience dataset."""

    output_path: Path
    num_trajectories: int = 512
    env_id: str = "gym_pusht/PushT-v0"
    seed: int = 0
    dmp_basis: int = 10
    min_duration: float = 1.8
    max_duration: float = 3.6
    weight_scale: float = 0.5
    start_noise: float = 30.0
    goal_noise: float = 120.0
    video_dir: Path | None = None

    @property
    def path(self) -> Path:
        return Path(self.output_path)


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    state_times: np.ndarray
    action_times: np.ndarray
    dmp_vector: np.ndarray
    info: dict


def _pad_sequences(sequences: Sequence[np.ndarray], value: float = 0.0) -> np.ndarray:
    max_len = max(seq.shape[0] for seq in sequences)
    out_shape = (len(sequences), max_len) + sequences[0].shape[1:]
    out = np.full(out_shape, value, dtype=sequences[0].dtype)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for idx, seq in enumerate(sequences):
        length = seq.shape[0]
        out[idx, :length] = seq
        mask[idx, :length] = True
    return out, mask


def _stack_metadata(config: DatasetConfig, trajectories: Sequence[Trajectory], dt: float) -> dict:
    metadata = {
        "num_trajectories": len(trajectories),
        "dt": dt,
        "env_id": config.env_id,
        "dmp_basis": config.dmp_basis,
        "min_duration": config.min_duration,
        "max_duration": config.max_duration,
        "weight_scale": config.weight_scale,
        "start_noise": config.start_noise,
        "goal_noise": config.goal_noise,
        "observation_dim": int(trajectories[0].observations.shape[-1]),
        "action_dim": int(trajectories[0].actions.shape[-1]),
        "condition_dim": int(trajectories[0].dmp_vector.shape[-1]),
    }
    success_rate = np.mean([traj.info["is_success"] for traj in trajectories])
    metadata["success_rate"] = float(success_rate)
    coverage = [traj.info.get("coverage", 0.0) for traj in trajectories]
    metadata["coverage_mean"] = float(np.mean(coverage))
    metadata["coverage_std"] = float(np.std(coverage))
    return metadata


def generate_pusht_dataset(config: DatasetConfig) -> Path:
    """Roll out PuSHt with randomly sampled DMP policies and save the dataset."""
    ensure_add_collision_handler()
    rng = np.random.default_rng(config.seed)

    env = gym.make(config.env_id, render_mode="rgb_array")
    unwrapped = env.unwrapped
    dt = 1.0 / float(unwrapped.control_hz)

    dmp = PlanarDMP(
        dt=dt,
        config=DMPConfig(
            n_basis=config.dmp_basis,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            weight_scale=config.weight_scale,
            start_noise=config.start_noise,
            goal_noise=config.goal_noise,
        ),
    )

    video_dir = None
    if config.video_dir is not None:
        video_dir = Path(config.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    trajectories: list[Trajectory] = []
    target_min = np.array([np.inf, np.inf], dtype=np.float32)
    target_max = np.array([-np.inf, -np.inf], dtype=np.float32)

    for rollout_idx in tqdm(range(config.num_trajectories), desc="Generating PuSHt rollouts"):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        obs = obs.astype(np.float32)
        params = dmp.sample_parameters(rng, start=obs[:2])
        dmp_actions, action_times = dmp.rollout(params)

        obs_seq = [obs]
        act_seq = []
        rew_seq = []
        step_info = {}
        terminated = False
        frames: list[np.ndarray] = []

        for action in dmp_actions:
            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            act_seq.append(action.astype(np.float32))
            rew_seq.append(np.float32(reward))
            obs_seq.append(obs.astype(np.float32))
            step_info = info
            if video_dir is not None:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            if terminated or truncated:
                break

        obs_arr = np.asarray(obs_seq, dtype=np.float32)
        act_arr = np.asarray(act_seq, dtype=np.float32)
        rew_arr = np.asarray(rew_seq, dtype=np.float32)
        state_times = np.arange(obs_arr.shape[0], dtype=np.float32) * dt
        action_times = np.arange(act_arr.shape[0], dtype=np.float32) * dt

        info_dict = {
            "is_success": bool(step_info.get("is_success", False)),
            "coverage": float(step_info.get("coverage", 0.0)),
            "steps": int(act_arr.shape[0]),
        }

        target_min = np.minimum(target_min, np.min(dmp_actions, axis=0))
        target_max = np.maximum(target_max, np.max(dmp_actions, axis=0))

        trajectories.append(
            Trajectory(
                observations=obs_arr,
                actions=act_arr,
                rewards=rew_arr,
                state_times=state_times,
                action_times=action_times,
                dmp_vector=params.as_vector().astype(np.float32),
                info=info_dict,
            )
        )
        if video_dir is not None and frames:
            fps = env.metadata.get("render_fps", 10)
            video_path = video_dir / f"trajectory_{rollout_idx:05d}.mp4"
            with imageio.get_writer(video_path, format="FFMPEG", mode="I", fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

    env.close()

    obs_padded, obs_mask = _pad_sequences([traj.observations for traj in trajectories], value=0.0)
    state_times, _ = _pad_sequences([traj.state_times[:, None] for traj in trajectories], value=0.0)
    state_times = state_times.squeeze(-1)

    actions_padded, action_mask = _pad_sequences([traj.actions for traj in trajectories], value=0.0)
    rewards_padded, _ = _pad_sequences([traj.rewards[:, None] for traj in trajectories], value=0.0)
    rewards_padded = rewards_padded.squeeze(-1)
    action_times, _ = _pad_sequences([traj.action_times[:, None] for traj in trajectories], value=0.0)
    action_times = action_times.squeeze(-1)

    condition_matrix = np.stack([traj.dmp_vector for traj in trajectories]).astype(np.float32)

    metadata = _stack_metadata(config, trajectories, dt)
    metadata["target_min"] = target_min.tolist()
    metadata["target_max"] = target_max.tolist()

    output_path = config.path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=obs_padded,
        observation_mask=obs_mask,
        state_times=state_times,
        actions=actions_padded,
        action_mask=action_mask,
        action_times=action_times,
        rewards=rewards_padded,
        conditions=condition_matrix,
    )
    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return output_path


__all__ = ["DatasetConfig", "generate_pusht_dataset"]
