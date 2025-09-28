#!/usr/bin/env python3
"""Generate PushT trajectories from slider-range DMP samples and save videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from nsf_rl.dmp import DMPConfig, DMPParams, PlanarDMP
from nsf_rl.utils.pymunk_compat import ensure_add_collision_handler

try:
    import gymnasium as gym
except ModuleNotFoundError as exc:  # pragma: no cover - surface nice error for users
    raise SystemExit(
        "gymnasium is required to run this script. Install project dependencies first."
    ) from exc

try:
    import gym_pusht  # noqa: F401  # pylint: disable=unused-import
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "gym-pusht is required to run this script. Install project dependencies first."
    ) from exc

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "imageio is required to write videos. Install project dependencies first."
    ) from exc


DMP_DT_SUBSTEPS = 8  # Integrate DMP with this many substeps per env control step

PIXEL_LOW = 0.0
PIXEL_HIGH = 512.0
SPAN = PIXEL_HIGH - PIXEL_LOW
HALF_SPAN = SPAN / 2.0
CENTER = PIXEL_LOW + HALF_SPAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample DMP parameters and generate PushT data")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/random_dmp_dataset.json"),
        help="Path to write the JSON dataset",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos/random_dmp_samples"),
        help="Directory to store rendered rollout videos",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--video-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale-forcing",
        action="store_true",
        help="Scale the DMP forcing term by the goal displacement (matches Marimo checkbox)",
    )
    return parser.parse_args()


def to_normalized(pix: np.ndarray) -> np.ndarray:
    arr = np.asarray(pix, dtype=np.float32)
    return np.clip((arr - CENTER) / HALF_SPAN, -1.0, 1.0)


def to_pixels(norm: np.ndarray) -> np.ndarray:
    arr = np.asarray(norm, dtype=np.float32)
    return np.clip(arr * HALF_SPAN + CENTER, PIXEL_LOW, PIXEL_HIGH)


def _serialise_info(info: dict[str, Any] | None) -> dict[str, Any]:
    if not info:
        return {}
    out: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif hasattr(value, "tolist"):
            out[key] = value.tolist()  # type: ignore[assignment]
        else:
            out[key] = repr(value)
    return out


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.video_samples < 0:
        raise SystemExit("--video-samples cannot be negative")

    rng = np.random.default_rng(args.seed)
    ensure_add_collision_handler()
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")

    try:
        control_hz = float(getattr(env.unwrapped, "control_hz", 10.0))
        env_dt = 1.0 / control_hz
        dmp_dt = env_dt / DMP_DT_SUBSTEPS

        video_dir = args.video_dir if args.video_samples > 0 else None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        dataset: list[dict[str, Any]] = []

        for idx in tqdm(range(args.num_samples), desc="Generating samples", unit="traj"):
            rollout_seed = int(rng.integers(0, 1_000_000))
            obs, info = env.reset(seed=rollout_seed)
            obs = np.asarray(obs, dtype=np.float32)
            base_state = info.get("state")
            if base_state is None:
                base_state = obs
            base_state = np.asarray(base_state, dtype=np.float32)
            start_pixels = np.clip(base_state[:2], PIXEL_LOW, PIXEL_HIGH)
            start_norm = to_normalized(start_pixels)

            reset_info = _serialise_info(info)
            step_observations: list[np.ndarray] = [obs]
            step_rewards: list[float] = []
            step_infos: list[dict[str, Any]] = []

            goal_pixels = rng.uniform(PIXEL_LOW, PIXEL_HIGH, size=2).astype(np.float32)
            goal_norm = to_normalized(goal_pixels)

            duration = float(rng.uniform(1.0, 4.0))
            stiffness = float(rng.uniform(5.0, 35.0))
            weights = rng.uniform(-30.0, 30.0, size=(2, 3)).astype(np.float32)

            cfg = DMPConfig(
                n_basis=3,
                min_duration=duration,
                max_duration=duration,
                workspace_low=-1.0,
                workspace_high=1.0,
                weight_scale=0.0,
                goal_noise=0.0,
                start_noise=0.0,
                stiffness=stiffness,
                scale_forcing_by_goal_delta=args.scale_forcing,
            )

            dmp = PlanarDMP(dt=dmp_dt, config=cfg)
            params = DMPParams(
                duration=duration,
                start=start_norm,
                goal=goal_norm,
                weights=weights,
                stiffness=stiffness,
                damping=2.0 * np.sqrt(stiffness),
            )

            rollout = dmp.rollout_detailed(params)
            waypoints_norm_full = rollout.positions
            times_full = rollout.times
            phase_full = rollout.canonical_phase
            velocity_full = rollout.velocities
            forcing_full = rollout.forcing
            waypoints_pixels_full = to_pixels(waypoints_norm_full)

            env_indices = np.arange(0, waypoints_pixels_full.shape[0], DMP_DT_SUBSTEPS, dtype=int)
            if env_indices.size == 0 or env_indices[-1] != waypoints_pixels_full.shape[0] - 1:
                env_indices = np.append(env_indices, waypoints_pixels_full.shape[0] - 1)

            waypoints_norm = waypoints_norm_full[env_indices]
            waypoints_pixels = waypoints_pixels_full[env_indices]
            times = times_full[env_indices]
            phase_actions = phase_full[env_indices]
            velocity_norm = velocity_full[env_indices]
            forcing_norm = forcing_full[env_indices]

            frames: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            episode_info: dict[str, Any] = {}
            terminated = False
            truncated = False

            for action in waypoints_pixels:
                obs, reward, terminated, truncated, step_info = env.step(action.astype(np.float32))
                actions.append(action.astype(np.float32))
                step_rewards.append(float(reward))
                obs = np.asarray(obs, dtype=np.float32)
                step_observations.append(obs)
                step_info_serialized = _serialise_info(step_info)
                step_infos.append(step_info_serialized)
                episode_info = step_info
                if video_dir is not None and idx < args.video_samples:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                if terminated or truncated:
                    break

            num_actions = len(actions)
            action_times = env_dt * np.arange(num_actions, dtype=np.float32)
            observation_times = env_dt * np.arange(len(step_observations), dtype=np.float32)
            phase_observations = np.exp(-dmp.config.alpha_s * observation_times / params.duration).astype(np.float32)

            def _sequence_from_infos(key: str) -> list[Any]:
                sequence: list[Any] = [reset_info.get(key)]
                sequence.extend(info.get(key) for info in step_infos)
                return sequence

            env_pos_agent = _sequence_from_infos("pos_agent")
            env_vel_agent = _sequence_from_infos("vel_agent")
            env_block_pose = _sequence_from_infos("block_pose")
            env_goal_pose = _sequence_from_infos("goal_pose")
            env_contacts = _sequence_from_infos("n_contacts")
            env_success = _sequence_from_infos("is_success")
            env_coverage = _sequence_from_infos("coverage")

            sample_entry = {
                "seed": rollout_seed,
                "duration": duration,
                "stiffness": stiffness,
                "damping": float(params.damping),
                "weights": weights.tolist(),
                "start_pixels": start_pixels.tolist(),
                "start_normalized": start_norm.tolist(),
                "goal_pixels": goal_pixels.tolist(),
                "goal_normalized": goal_norm.tolist(),
                "dmp_waypoints_pixels": waypoints_pixels.astype(np.float32).tolist(),
                "dmp_waypoints_normalized": waypoints_norm.astype(np.float32).tolist(),
                "dmp_velocity_normalized": velocity_norm.astype(np.float32).tolist(),
                "dmp_forcing_normalized": forcing_norm.astype(np.float32).tolist(),
                "dmp_canonical_phase_actions": phase_actions.astype(np.float32).tolist(),
                "dmp_canonical_phase_observations": phase_observations.tolist(),
                "dmp_dt": float(dmp.dt),
                "dmp_alpha_s": float(dmp.config.alpha_s),
                "dmp_n_basis": int(dmp.config.n_basis),
                "timestamps": times.astype(np.float32).tolist(),
                "executed_steps": len(actions),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_info": _serialise_info(episode_info),
                "reset_info": reset_info,
                "executed_actions_pixels": np.asarray(actions, dtype=np.float32).tolist(),
                "executed_action_timestamps": action_times.tolist(),
                "observations": np.asarray(step_observations, dtype=np.float32).tolist(),
                "observation_timestamps": observation_times.tolist(),
                "rewards": step_rewards,
                "step_infos": step_infos,
                "env_pos_agent": env_pos_agent,
                "env_vel_agent": env_vel_agent,
                "env_block_pose": env_block_pose,
                "env_goal_pose": env_goal_pose,
                "env_n_contacts": env_contacts,
                "env_is_success": env_success,
                "env_coverage": env_coverage,
            }

            if video_dir is not None and idx < args.video_samples and frames:
                fps = env.metadata.get("render_fps", 10)
                video_path = video_dir / f"sample_{idx:03d}.mp4"
                with imageio.get_writer(video_path, format="FFMPEG", mode="I", fps=fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                sample_entry["video_path"] = str(video_path)

            dataset.append(sample_entry)

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(dataset, fh, indent=2)
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
