#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Tuple

import equinox as eqx
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

import gymnasium as gym
import gym_pusht  # noqa: F401

from nsf_rl.models.conditional_flow import (
    ConditionalNeuralStochasticFlow,
    FlowNetworkConfig,
)
from nsf_rl.data.dmp_pairs import DmpPairwiseDataset, NormalizationStats


PIXEL_LOW = 0.0
PIXEL_HIGH = 512.0
SPAN = PIXEL_HIGH - PIXEL_LOW
HALF_SPAN = SPAN / 2.0
CENTER = PIXEL_LOW + HALF_SPAN


def _to_pixels(norm: np.ndarray) -> np.ndarray:
    a = np.asarray(norm, dtype=np.float32)
    return np.clip(a * HALF_SPAN + CENTER, PIXEL_LOW, PIXEL_HIGH)


def _draw_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]):
    h, w = img.shape[:2]
    cy, cx = int(center[1]), int(center[0])
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = color


def _draw_square(img: np.ndarray, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    half = size // 2
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def _draw_line(img: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], color: Tuple[int, int, int]):
    # Simple Bresenham
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _render_prediction_frame(pred_state: np.ndarray, goal_pixels: np.ndarray, canvas_size: int = 512) -> np.ndarray:
    # pred_state: [7] = [agent_x, agent_y, block_x, block_y, sin, cos, phase] in normalized space
    agent_xy = _to_pixels(pred_state[:2])
    block_xy = _to_pixels(pred_state[2:4])
    sin_t, cos_t = float(pred_state[4]), float(pred_state[5])
    theta = np.arctan2(sin_t, cos_t)
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    # Agent: blue circle
    _draw_circle(img, (int(agent_xy[0]), int(agent_xy[1])), radius=6, color=(40, 120, 240))
    # Block: green square with heading
    _draw_square(img, (int(block_xy[0]), int(block_xy[1])), size=16, color=(40, 200, 40))
    hx = int(block_xy[0] + 12 * np.cos(theta))
    hy = int(block_xy[1] + 12 * np.sin(theta))
    _draw_line(img, (int(block_xy[0]), int(block_xy[1])), (hx, hy), color=(0, 0, 0))
    # Goal: red crosshair
    gx, gy = int(goal_pixels[0]), int(goal_pixels[1])
    _draw_line(img, (gx - 8, gy), (gx + 8, gy), color=(220, 40, 40))
    _draw_line(img, (gx, gy - 8), (gx, gy + 8), color=(220, 40, 40))
    return img


def _load_meta_and_infos(root: Path, meta: dict) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(root / meta["path"]) as npz:
        time = np.asarray(npz["time"], dtype=np.float32)
        phase = np.asarray(npz["phase"], dtype=np.float32)
    infos = json.loads((root / meta["info_path"]).read_text(encoding="utf-8"))
    return meta, infos, time, phase


def _state_from_info(info: dict[str, Any]) -> np.ndarray:
    # Mirrors DmpPairwiseDataset._state_from_info and then we append phase externally
    CENTER = 256.0
    HALF_SPAN = 256.0
    def to_norm(pix):
        a = np.asarray(pix, dtype=np.float32)
        return np.clip((a - CENTER) / HALF_SPAN, -1.0, 1.0)
    pos_agent = np.asarray(info.get("pos_agent", [CENTER, CENTER]), dtype=np.float32)
    pos_agent_norm = to_norm(pos_agent[:2])
    block_pose = np.asarray(info.get("block_pose", [CENTER, CENTER, 0.0]), dtype=np.float32)
    block_xy_norm = to_norm(block_pose[:2])
    theta_rad = np.deg2rad(float(block_pose[2]))
    sin_theta = np.sin(theta_rad).astype(np.float32)
    cos_theta = np.cos(theta_rad).astype(np.float32)
    return np.concatenate([pos_agent_norm, block_xy_norm, np.array([sin_theta, cos_theta], dtype=np.float32)], axis=0)


def _condition_vector_from_meta(meta: dict[str, Any]) -> np.ndarray:
    # Match DmpPairwiseDataset._condition_vector
    stiffness = float(meta["stiffness"])  # scalar
    damping = float(meta["damping"])  # scalar
    dmp_dt = float(meta["dmp_dt"])  # scalar
    dmp_alpha_s = float(meta["dmp_alpha_s"])  # scalar
    n_basis = int(meta["dmp_n_basis"])  # scalar (cast to float later)
    scale_flag = 1.0 if bool(meta["scale_forcing_by_goal_delta"]) else 0.0
    start_norm = (np.asarray(meta["start_pixels"], dtype=np.float32) - CENTER) / HALF_SPAN
    goal_norm = (np.asarray(meta["goal_pixels"], dtype=np.float32) - CENTER) / HALF_SPAN
    weights = np.asarray(meta["weights"], dtype=np.float32)
    cond = [
        np.array([stiffness, damping, dmp_dt, dmp_alpha_s, float(n_basis), float(scale_flag)], dtype=np.float32),
        start_norm.astype(np.float32),
        goal_norm.astype(np.float32),
        weights.reshape(-1).astype(np.float32),
    ]
    return np.concatenate(cond, axis=0)


def _load_condition_stats(stats_path: Path) -> NormalizationStats | None:
    if not stats_path.exists():
        return None
    obj = json.loads(stats_path.read_text(encoding="utf-8"))
    return NormalizationStats(mean=np.asarray(obj["mean"], dtype=np.float32), std=np.asarray(obj["std"], dtype=np.float32))


def _predict_states_for_times(
    flow_model: ConditionalNeuralStochasticFlow,
    params,
    x0: np.ndarray,  # [7]
    times: np.ndarray,  # [T+1]
    condition_vec: np.ndarray,  # [C]
) -> np.ndarray:
    # Deterministic prediction via base-mean pushed through flow: z = transform(eps=0)
    # Prepare batched inputs
    B = int(times.shape[0])
    x_init = jnp.repeat(jnp.asarray(x0)[None, :], B, axis=0)
    t_init = jnp.zeros((B,), dtype=jnp.float32)
    t_final = jnp.asarray(times, dtype=jnp.float32)
    condition = jnp.repeat(jnp.asarray(condition_vec)[None, :], B, axis=0)

    fm = eqx.combine(params, flow_model)

    def per_time(ti):
        dist = fm(x_init=x_init[0], t_init=jnp.array(0.0, dtype=jnp.float32), t_final=ti, condition=condition[0])
        # Push mean of base through flow layers: set eps=0 in base space
        # Base is Gaussian with loc, scale; its mean in base space is loc
        # But we need to go through bijectors; ContinuousNormalizingFlow.transform expects eps in base noise space.
        # To get mean, we pass eps=0 and then add base loc via transform implementation (loc + scale*0 = loc)
        zeros = jnp.zeros(dist.base_distribution.sample_shape, dtype=jnp.float32)
        return dist.transform(zeros)

    preds = jax.vmap(per_time)(t_final)
    return np.asarray(preds, dtype=np.float32)


def build_flow_model(state_dim: int, condition_dim: int, *, hidden_size: int = 64, depth: int = 2, conditioner_hidden_size: int = 64, conditioner_depth: int = 2, num_flow_layers: int = 4, activation: str = "tanh", scale_fn: str = "tanh_exp", key: jax.Array = jax.random.PRNGKey(0)) -> ConditionalNeuralStochasticFlow:
    cfg = FlowNetworkConfig(
        state_dim=state_dim,
        condition_dim=condition_dim,
        hidden_size=hidden_size,
        depth=depth,
        activation=activation,
        num_flow_layers=num_flow_layers,
        conditioner_hidden_size=conditioner_hidden_size,
        conditioner_depth=conditioner_depth,
        scale_fn=scale_fn,  # type: ignore
        include_initial_time=False,
    )
    return ConditionalNeuralStochasticFlow(config=cfg, key=key)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ground-truth vs NSF predictions on test split")
    p.add_argument("--data-root", type=Path, default=Path("data/random_dmp_npz"))
    p.add_argument("--test-split", type=str, default="test")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--use-best", action="store_true", help="Load best.eqx instead of latest.eqx")
    p.add_argument("--num-trajs", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("videos/compare_flow_predictions"))
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load index and pick first N metas from test split
    test_root = args.data_root / args.test_split
    index = [json.loads(l) for l in (test_root / "index.jsonl").open()]
    metas = index[: max(0, args.num_trajs)]
    if not metas:
        raise SystemExit("No trajectories in test split")

    # Build dataset once to probe dims and load stats
    rng = np.random.default_rng(0)
    ds = DmpPairwiseDataset(root=test_root, rng=rng, standardize=True, stats_path=args.checkpoint_dir / "condition_stats.json")
    # Force stats load
    _ = next(ds.batches(batch_size=1, repeat=True))
    # Probe dims with a batch
    peek = next(ds.batches(batch_size=1, repeat=True))
    state_dim = int(peek.x_init.shape[-1])
    condition_dim = int(peek.condition.shape[-1])

    # Build model with default hyperparams (as per training defaults)
    model = build_flow_model(state_dim=state_dim, condition_dim=condition_dim, key=jax.random.PRNGKey(0))

    # Load parameters
    params_path = args.checkpoint_dir / ("best.eqx" if args.use_best else "latest.eqx")
    if not params_path.exists():
        raise SystemExit(f"Missing checkpoint: {params_path}")
    params = eqx.tree_deserialise_leaves(params_path, eqx.filter(model, eqx.is_array))

    # Load condition normalization stats (if available)
    stats_path = args.checkpoint_dir / "condition_stats.json"
    stats = _load_condition_stats(stats_path)

    # Prepare env for GT rendering
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    fps = args.fps
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, meta in enumerate(metas):
            # Load meta + infos + time/phase
            meta, infos, times, phase = _load_meta_and_infos(test_root, meta)
            reset_info = infos["reset_info"]
            step_infos = infos["step_infos"]
            T = len(times) - 1
            # Ground-truth rollout frames by replaying stored actions
            with np.load(test_root / meta["path"]) as npz:
                actions = np.asarray(npz["act"], dtype=np.float32)
            # Reset env with seed from meta for consistency
            obs, info = env.reset(seed=int(meta["seed"]))
            gt_frames: list[np.ndarray] = []
            for a in actions:
                obs, reward, terminated, truncated, step_info = env.step(a)
                frame = env.render()
                if frame is not None:
                    gt_frames.append(np.asarray(frame))
                if terminated or truncated:
                    break
            if not gt_frames:
                continue
            # Match video length to actions
            # Build initial normalized state x0 (append phase[0])
            x0_core = _state_from_info(reset_info)
            x0 = np.concatenate([x0_core, np.array([phase[0]], dtype=np.float32)], axis=0)
            # Build condition vector from meta, then standardize
            cond = _condition_vector_from_meta(meta)
            if stats is not None:
                cond = stats.apply(cond)
            # Predict states for each time using one-shot from t=0
            preds = _predict_states_for_times(flow_model=model, params=params, x0=x0, times=times, condition_vec=cond)
            # Render predicted frames on blank canvas, using meta goal for overlay
            goal_pixels = np.asarray(meta["goal_pixels"], dtype=np.float32)
            pred_frames: list[np.ndarray] = []
            # We skip t=0 for visual comparison to align with env step frames (length T)
            for t in range(1, len(times)):
                frame_pred = _render_prediction_frame(preds[t], goal_pixels, canvas_size=512)
                pred_frames.append(frame_pred)
            # Align lengths
            L = min(len(gt_frames), len(pred_frames))
            gt_frames = gt_frames[:L]
            pred_frames = pred_frames[:L]
            # Concatenate side-by-side
            side_by_side = [np.concatenate([gt, pred], axis=1) for gt, pred in zip(gt_frames, pred_frames)]
            # Write video
            out_path = args.output_dir / f"compare_{idx:03d}.mp4"
            with imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=fps) as writer:
                for fr in side_by_side:
                    writer.append_data(fr)
            print(f"Wrote {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()


