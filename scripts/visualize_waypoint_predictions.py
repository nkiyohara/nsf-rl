#!/usr/bin/env python3
"""Visualize how well the NSF reproduces DMP waypoint trajectories."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Tuple

import equinox as eqx
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from nsf_rl.data.dmp_pairs import DmpPairwiseDataset
from nsf_rl.models.conditional_flow import ConditionalNeuralStochasticFlow, FlowNetworkConfig


PIXEL_LOW = 0.0
PIXEL_HIGH = 512.0
SPAN = PIXEL_HIGH - PIXEL_LOW
HALF_SPAN = SPAN / 2.0
CENTER = PIXEL_LOW + HALF_SPAN

COLOR_BG = 255
COLOR_GRID = (230, 230, 230)
COLOR_CMD_FULL = (200, 200, 200)
COLOR_CMD_ACTIVE = (66, 135, 245)
COLOR_PRED = (255, 120, 40)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render NSF predictions vs DMP waypoints in 2D")
    p.add_argument("--data-root", type=Path, default=Path("data/random_dmp_npz"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--use-best", action="store_true", help="Load best.eqx instead of latest.eqx")
    p.add_argument("--epoch", type=int, default=None, help="Load epoch_XXX.eqx for a specific epoch number")
    p.add_argument("--start-index", type=int, default=0, help="Index in index.jsonl to start rendering from")
    p.add_argument("--num-samples", type=int, default=1, help="Number of trajectories to render")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--frame-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0, help="RNG seed used only for dataset shuffling helpers")
    return p.parse_args()


def _to_pixels(norm_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(norm_xy, dtype=np.float32)
    return np.clip(arr * HALF_SPAN + CENTER, PIXEL_LOW, PIXEL_HIGH)


def _draw_line(canvas: np.ndarray, p0: Sequence[float], p1: Sequence[float], color: Tuple[int, int, int]) -> None:
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= y0 < canvas.shape[0] and 0 <= x0 < canvas.shape[1]:
            canvas[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_circle(canvas: np.ndarray, center: Sequence[float], radius: int, color: Tuple[int, int, int]) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[mask] = color


def _draw_cross(canvas: np.ndarray, center: Sequence[float], size: int, color: Tuple[int, int, int]) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = size // 2
    for dx in range(-half, half + 1):
        x = cx + dx
        y = cy + dx
        y2 = cy - dx
        if 0 <= x < canvas.shape[1]:
            if 0 <= y < canvas.shape[0]:
                canvas[y, x] = color
            if 0 <= y2 < canvas.shape[0]:
                canvas[y2, x] = color


def _draw_grid(canvas: np.ndarray, step: int = 64) -> None:
    for x in range(0, canvas.shape[1], step):
        canvas[:, x] = COLOR_GRID
    for y in range(0, canvas.shape[0], step):
        canvas[y, :] = COLOR_GRID


def _draw_polyline(canvas: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    if points.shape[0] < 2:
        return
    for i in range(1, points.shape[0]):
        _draw_line(canvas, points[i - 1], points[i], color)


def _build_frame(
    command_xy: np.ndarray,
    pred_xy: np.ndarray,
    t: int,
    frame_size: int,
) -> np.ndarray:
    canvas = np.full((frame_size, frame_size, 3), COLOR_BG, dtype=np.uint8)
    _draw_grid(canvas)
    _draw_polyline(canvas, command_xy, COLOR_CMD_FULL)
    _draw_polyline(canvas, command_xy[: t + 1], COLOR_CMD_ACTIVE)
    _draw_polyline(canvas, pred_xy[: t + 1], COLOR_PRED)
    _draw_circle(canvas, command_xy[t], radius=4, color=COLOR_CMD_ACTIVE)
    _draw_cross(canvas, pred_xy[t], size=8, color=COLOR_PRED)
    return canvas


def _predict_states_for_times(
    flow_model: ConditionalNeuralStochasticFlow,
    params,
    x0: np.ndarray,
    times: np.ndarray,
    condition_vec: np.ndarray,
) -> np.ndarray:
    fm = eqx.combine(params, flow_model)
    x_init = jnp.asarray(x0)
    condition = jnp.asarray(condition_vec)

    def per_time(tf):
        dist = fm(x_init=x_init, t_init=jnp.array(0.0, dtype=jnp.float32), t_final=jnp.array(tf, dtype=jnp.float32), condition=condition)
        zeros = jnp.zeros(dist.base_distribution.sample_shape, dtype=jnp.float32)
        return dist.transform(zeros)

    preds = jax.vmap(per_time)(jnp.asarray(times, dtype=jnp.float32))
    return np.asarray(preds, dtype=np.float32)


def _build_flow_model(state_dim: int, condition_dim: int) -> ConditionalNeuralStochasticFlow:
    cfg = FlowNetworkConfig(
        state_dim=state_dim,
        condition_dim=condition_dim,
        hidden_size=64,
        depth=2,
        activation="tanh",
        num_flow_layers=4,
        conditioner_hidden_size=64,
        conditioner_depth=2,
        scale_fn="tanh_exp",  # type: ignore[arg-type]
        include_initial_time=False,
    )
    return ConditionalNeuralStochasticFlow(key=jax.random.PRNGKey(0), **asdict(cfg))


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    stats_path = args.checkpoint_dir / "condition_stats.json"
    rng = np.random.default_rng(args.seed)

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    ds = DmpPairwiseDataset(
        root=split_root,
        rng=rng,
        standardize=True,
        stats_path=stats_path,
        state_source="waypoint",
    )

    # Probe dims for constructing the flow module
    peek = next(ds.batches(batch_size=1, repeat=True))
    state_dim = int(peek.x_init.shape[-1])
    condition_dim = int(peek.condition.shape[-1])
    model = _build_flow_model(state_dim=state_dim, condition_dim=condition_dim)

    if args.epoch is not None:
        params_path = args.checkpoint_dir / f"epoch_{args.epoch:03d}.eqx"
        epoch_tag = f"epoch-{args.epoch:03d}"
    else:
        params_path = args.checkpoint_dir / ("best.eqx" if args.use_best else "latest.eqx")
        epoch_tag = "epoch-best" if args.use_best else "epoch-latest"
    if not params_path.exists():
        raise SystemExit(f"Checkpoint not found: {params_path}")
    params = eqx.tree_deserialise_leaves(params_path, eqx.filter(model, eqx.is_array))

    output_dir = args.output_dir or (Path("videos") / args.checkpoint_dir.name / epoch_tag / "waypoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = max(0, args.start_index)
    end = min(len(ds.index), start + args.num_samples)
    if start >= end:
        raise SystemExit(f"No samples available in range [{start}, {start + args.num_samples})")

    for sample_idx in range(start, end):
        seq = ds.load_sequence(sample_idx)
        time = seq["time"]
        phase = seq["phase"]
        states_core = seq["states"]
        if states_core.shape[0] != time.shape[0]:
            raise RuntimeError(f"Sequence length mismatch for sample {sample_idx}")
        state_with_phase = np.concatenate([states_core, phase[:, None]], axis=-1)
        if state_with_phase.shape[1] != state_dim:
            raise RuntimeError(
                f"State dimension mismatch (expected {state_dim}, got {state_with_phase.shape[1]}) "
                " — ensure the dataset and checkpoint were created with state_source='waypoint'."
            )
        x0 = state_with_phase[0].astype(np.float32)
        cond = np.asarray(seq["condition"], dtype=np.float32)
        preds = _predict_states_for_times(flow_model=model, params=params, x0=x0, times=time, condition_vec=cond)

        command_xy = _to_pixels(states_core[:, :2])
        pred_xy = _to_pixels(preds[:, :2])
        if command_xy.shape[0] < 2:
            continue

        sample_name = seq["meta"].get("id", f"{sample_idx:06d}")
        out_path = output_dir / f"waypoints_{int(sample_idx):06d}_{sample_name}.mp4"
        with imageio.get_writer(out_path, format="FFMPEG", mode="I", fps=args.fps) as writer:
            for t in range(1, command_xy.shape[0]):
                frame = _build_frame(command_xy, pred_xy, t, args.frame_size)
                writer.append_data(frame)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
