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

import gymnasium as gym

import gym_pusht  # noqa: F401

import imageio.v2 as imageio


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
        "--output-dir",
        type=Path,
        default=Path("data/random_dmp_npz"),
        help="Directory to write NPZ samples and index.jsonl",
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


def _get_goal_pose_from_info(info: dict[str, Any]) -> np.ndarray | None:
    """Extract goal pose as a float32 array if present in a serialised info dict.

    The environment is expected to populate `goal_pose` consistently across all
    steps. If the key is absent, return None.
    """
    if "goal_pose" not in info:
        return None
    value = info["goal_pose"]
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    return arr


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

        # Output directories for NPZ samples and index.jsonl
        args.output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = args.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        rendered_videos: list[tuple[int, Path]] = []
        combined_fps: float | None = None
        combined_written = False

        def write_combined_video() -> None:
            nonlocal combined_written
            if combined_written:
                return
            if (
                video_dir is None
                or combined_fps is None
                or not rendered_videos
                or video_limit <= 0
                or len(rendered_videos) < video_limit
            ):
                return
            combined_sequence = sorted(rendered_videos, key=lambda item: item[0])
            first_idx = combined_sequence[0][0] + 1
            last_idx = combined_sequence[-1][0] + 1
            combined_name = f"samples_{first_idx:03d}-{last_idx:03d}.mp4"
            combined_path = video_dir / combined_name
            with imageio.get_writer(
                combined_path,
                format="FFMPEG",
                mode="I",
                fps=combined_fps,
            ) as writer:
                for _, path in combined_sequence:
                    with imageio.get_reader(path, format="FFMPEG") as reader:
                        for frame in reader:
                            writer.append_data(frame)
            combined_written = True

        index_path = args.output_dir / "index.jsonl"
        with open(index_path, "w", encoding="utf-8") as index_fh:
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
                waypoints_pixels_full = to_pixels(waypoints_norm_full)

                env_indices = np.arange(0, waypoints_pixels_full.shape[0], DMP_DT_SUBSTEPS, dtype=int)
                if env_indices.size == 0 or env_indices[-1] != waypoints_pixels_full.shape[0] - 1:
                    env_indices = np.append(env_indices, waypoints_pixels_full.shape[0] - 1)

                waypoints_pixels = waypoints_pixels_full[env_indices]
                # We only need pixel-space waypoints for control

                frames: list[np.ndarray] = []
                actions: list[np.ndarray] = []
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
                    if video_dir is not None and idx < video_limit:
                        frame = env.render()
                        if frame is not None:
                            padded = _pad_frame_to_macro_block(np.asarray(frame))
                            frames.append(padded)
                    if terminated or truncated:
                        break

                num_actions = len(actions)
                # Timestamps and canonical phase aligned with observations (length T+1)
                observation_times = env_dt * np.arange(len(step_observations), dtype=np.float32)
                phase_observations = np.exp(
                    -dmp.config.alpha_s * observation_times / params.duration
                ).astype(np.float32)

                # Verify goal consistency across all steps and save serialised infos to JSON
                reset_goal = _get_goal_pose_from_info(reset_info)
                if reset_goal is None:
                    raise SystemExit(f"reset info for sample {idx} is missing goal_pose")
                for t, si in enumerate(step_infos):
                    step_goal = _get_goal_pose_from_info(si)
                    if step_goal is None:
                        raise SystemExit(f"step {t} info for sample {idx} is missing goal_pose")
                    if not np.allclose(step_goal, reset_goal, rtol=1e-5, atol=1e-5):
                        raise SystemExit(
                            f"Inconsistent goal_pose at step {t} for sample {idx}: "
                            f"reset {reset_goal.tolist()} vs step {step_goal.tolist()}"
                        )

                def _sequence_from_infos(key: str) -> list[Any]:
                    sequence: list[Any] = [reset_info.get(key)]
                    sequence.extend(info.get(key) for info in step_infos)
                    return sequence

                env_success = _sequence_from_infos("is_success")
                env_coverage = _sequence_from_infos("coverage")

                actions_array = np.asarray(actions, dtype=np.float32)
                if actions_array.size == 0:
                    actions_array = np.zeros((0, waypoints_pixels.shape[1]), dtype=np.float32)

                rewards_array = np.asarray(step_rewards, dtype=np.float32)
                # Minimal arrays only (we no longer persist observations; infos carry the full state)

                # Build minimal arrays for NPZ
                done = np.zeros((num_actions,), dtype=np.bool_)
                if num_actions > 0:
                    done[-1] = bool(terminated or truncated)

                sample_name = f"{idx:06d}.npz"
                sample_rel_path = f"samples/{sample_name}"
                sample_path = samples_dir / sample_name
                sample_info_name = f"{idx:06d}.json"
                sample_info_rel_path = f"samples/{sample_info_name}"
                sample_info_path = samples_dir / sample_info_name

                np.savez_compressed(
                    sample_path,
                    act=actions_array,
                    rew=rewards_array,
                    done=done,
                    phase=phase_observations,
                    time=observation_times,
                )

                # Write per-trajectory infos (reset + per-step) to JSON
                with open(sample_info_path, "w", encoding="utf-8") as info_fh:
                    json.dump(
                        {
                            "reset_info": reset_info,
                            "step_infos": step_infos,
                            "goal_pose": reset_goal.tolist(),
                        },
                        info_fh,
                        ensure_ascii=False,
                    )

                # Minimal metadata per sample for filtering
                success_val = env_success[-1] if env_success else None
                success = bool(success_val) if success_val is not None else False
                coverage_val = env_coverage[-1] if env_coverage else None
                try:
                    coverage = float(coverage_val) if coverage_val is not None else None
                except Exception:
                    coverage = None

                index_record = {
                    "id": int(idx),
                    "path": sample_rel_path,
                    "info_path": sample_info_rel_path,
                    "len": int(num_actions),
                    "success": bool(success),
                    # Per-trajectory metadata for filtering/sampling
                    "seed": int(rollout_seed),
                    "duration": float(duration),
                    "stiffness": float(stiffness),
                    "damping": float(params.damping),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    # DMP configuration summary
                    "dmp_dt": float(dmp.dt),
                    "dmp_alpha_s": float(dmp.config.alpha_s),
                    "dmp_n_basis": int(dmp.config.n_basis),
                    "scale_forcing_by_goal_delta": bool(args.scale_forcing),
                    # Start/goal and weights (small arrays)
                    "start_pixels": np.asarray(start_pixels, dtype=np.float32).tolist(),
                    "goal_pixels": np.asarray(goal_pixels, dtype=np.float32).tolist(),
                    "goal_pose": reset_goal.tolist(),
                    "weights": np.asarray(weights, dtype=np.float32).tolist(),
                }
                if coverage is not None:
                    index_record["coverage"] = float(coverage)
                index_fh.write(json.dumps(index_record, ensure_ascii=False) + "\n")

                if video_dir is not None and idx < video_limit and frames:
                    fps = env.metadata.get("render_fps", 10)
                    video_path = video_dir / f"sample_{idx:03d}.mp4"
                    with imageio.get_writer(video_path, format="FFMPEG", mode="I", fps=fps) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    rendered_videos.append((idx, video_path))
                    if combined_fps is None:
                        combined_fps = float(fps)
                    write_combined_video()

            if not combined_written:
                write_combined_video()
    finally:
        env.close()


if __name__ == "__main__":  # pragma: no cover
    main()
