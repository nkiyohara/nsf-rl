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
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "h5py is required to run this script. Install project dependencies first."
    ) from exc

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


def _pad_frame_to_macro_block(frame: np.ndarray, macro_block: int = 16) -> np.ndarray:
    """Pad frame so height/width are multiples of macro_block."""
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    pad_h = (-height) % macro_block
    pad_w = (-width) % macro_block
    if pad_h == 0 and pad_w == 0:
        return frame
    pad = ((0, pad_h), (0, pad_w)) + tuple((0, 0) for _ in range(frame.ndim - 2))
    return np.pad(frame, pad, mode="edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample DMP parameters and generate PushT data")
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=Path("data/random_dmp_dataset.h5"),
        help="Path to write the HDF5 dataset",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos/random_dmp_samples"),
        help="Directory to store rendered rollout videos",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument(
        "--video-samples",
        type=int,
        default=10,
        help="Number of trajectories to render to video (capped at 10)",
    )
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

        video_limit = min(args.video_samples, 10)
        video_dir = args.video_dir if video_limit > 0 else None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        args.output_h5.parent.mkdir(parents=True, exist_ok=True)
        json_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(args.output_h5, "w") as dataset_file:
            dataset_file.attrs["num_samples"] = int(args.num_samples)
            dataset_file.attrs["seed"] = int(args.seed)
            dataset_file.attrs["scale_forcing_by_goal_delta"] = bool(args.scale_forcing)
            dataset_file.attrs["dmp_dt_substeps"] = int(DMP_DT_SUBSTEPS)
            dataset_file.attrs["video_limit"] = int(video_limit)
            if video_dir is not None:
                dataset_file.attrs["video_directory"] = str(video_dir)

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
                weights = rng.uniform(-60.0, 60.0, size=(2, 3)).astype(np.float32)

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
                    if video_dir is not None and idx < video_limit:
                        frame = env.render()
                        if frame is not None:
                            padded = _pad_frame_to_macro_block(np.asarray(frame))
                            frames.append(padded)
                    if terminated or truncated:
                        break

                num_actions = len(actions)
                action_times = env_dt * np.arange(num_actions, dtype=np.float32)
                observation_times = env_dt * np.arange(len(step_observations), dtype=np.float32)
                phase_observations = np.exp(
                    -dmp.config.alpha_s * observation_times / params.duration
                ).astype(np.float32)

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

                actions_array = np.asarray(actions, dtype=np.float32)
                if actions_array.size == 0:
                    actions_array = np.zeros((0, waypoints_pixels.shape[1]), dtype=np.float32)

                rewards_array = np.asarray(step_rewards, dtype=np.float32)
                observations_array = np.asarray(step_observations, dtype=np.float32)
                phase_actions_array = phase_actions.astype(np.float32)
                phase_observations_array = phase_observations.astype(np.float32)

                sample_group = dataset_file.create_group(f"sample_{idx:05d}")
                sample_group.attrs["seed"] = int(rollout_seed)
                sample_group.attrs["duration"] = float(duration)
                sample_group.attrs["stiffness"] = float(stiffness)
                sample_group.attrs["damping"] = float(params.damping)
                sample_group.attrs["terminated"] = bool(terminated)
                sample_group.attrs["truncated"] = bool(truncated)
                sample_group.attrs["executed_steps"] = int(len(actions))

                sample_group.create_dataset("weights", data=weights, dtype=np.float32)
                sample_group.create_dataset("start_pixels", data=start_pixels.astype(np.float32))
                sample_group.create_dataset("start_normalized", data=start_norm.astype(np.float32))
                sample_group.create_dataset("goal_pixels", data=goal_pixels.astype(np.float32))
                sample_group.create_dataset("goal_normalized", data=goal_norm.astype(np.float32))
                sample_group.create_dataset("dmp_waypoints_pixels", data=waypoints_pixels.astype(np.float32))
                sample_group.create_dataset("dmp_waypoints_normalized", data=waypoints_norm.astype(np.float32))
                sample_group.create_dataset("dmp_velocity_normalized", data=velocity_norm.astype(np.float32))
                sample_group.create_dataset("dmp_forcing_normalized", data=forcing_norm.astype(np.float32))
                sample_group.create_dataset("dmp_canonical_phase_actions", data=phase_actions_array)
                sample_group.create_dataset("dmp_canonical_phase_observations", data=phase_observations_array)
                sample_group.create_dataset("timestamps", data=times.astype(np.float32))
                sample_group.create_dataset("executed_actions_pixels", data=actions_array)
                sample_group.create_dataset("executed_action_timestamps", data=action_times.astype(np.float32))
                sample_group.create_dataset("observations", data=observations_array)
                sample_group.create_dataset("observation_timestamps", data=observation_times.astype(np.float32))
                sample_group.create_dataset("rewards", data=rewards_array)

                sample_group.create_dataset("reset_info", data=json.dumps(reset_info), dtype=json_dtype)
                sample_group.create_dataset(
                    "env_info", data=json.dumps(_serialise_info(episode_info)), dtype=json_dtype
                )
                sample_group.create_dataset("step_infos", data=json.dumps(step_infos), dtype=json_dtype)
                sample_group.create_dataset("env_pos_agent", data=json.dumps(env_pos_agent), dtype=json_dtype)
                sample_group.create_dataset("env_vel_agent", data=json.dumps(env_vel_agent), dtype=json_dtype)
                sample_group.create_dataset("env_block_pose", data=json.dumps(env_block_pose), dtype=json_dtype)
                sample_group.create_dataset("env_goal_pose", data=json.dumps(env_goal_pose), dtype=json_dtype)
                sample_group.create_dataset("env_n_contacts", data=json.dumps(env_contacts), dtype=json_dtype)
                sample_group.create_dataset("env_is_success", data=json.dumps(env_success), dtype=json_dtype)
                sample_group.create_dataset("env_coverage", data=json.dumps(env_coverage), dtype=json_dtype)

                sample_group.attrs["dmp_dt"] = float(dmp.dt)
                sample_group.attrs["dmp_alpha_s"] = float(dmp.config.alpha_s)
                sample_group.attrs["dmp_n_basis"] = int(dmp.config.n_basis)

                if video_dir is not None and idx < video_limit and frames:
                    fps = env.metadata.get("render_fps", 10)
                    video_path = video_dir / f"sample_{idx:03d}.mp4"
                    with imageio.get_writer(video_path, format="FFMPEG", mode="I", fps=fps) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    sample_group.attrs["video_path"] = str(video_path)
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
