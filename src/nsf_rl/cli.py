"""Command line interface for dataset generation and training."""

from __future__ import annotations

import argparse
from pathlib import Path

from nsf_rl.data.generate import DatasetConfig, generate_pusht_dataset
from nsf_rl.models.conditional_flow import FlowNetworkConfig
from nsf_rl.training import AuxiliaryConfig, TrainingConfig, train


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conditional neural stochastic flow utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-data", help="Generate PushT rollouts with DMP policies")
    gen.add_argument("--output", type=Path, default=Path("data/pusht_dmp_dataset.npz"), help="Output .npz path")
    gen.add_argument("--num-trajectories", type=_positive_int, default=512)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--dmp-basis", type=_positive_int, default=10)
    gen.add_argument("--min-duration", type=float, default=1.8)
    gen.add_argument("--max-duration", type=float, default=3.6)
    gen.add_argument("--weight-scale", type=float, default=0.5)
    gen.add_argument("--start-noise", type=float, default=30.0)
    gen.add_argument("--goal-noise", type=float, default=120.0)
    gen.add_argument("--video-dir", type=Path, default=None, help="Optional directory to write rollout videos")

    train_parser = subparsers.add_parser("train", help="Train conditional NSF on generated data")
    train_parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset .npz")
    train_parser.add_argument("--steps", type=_positive_int, default=20000, help="Number of optimisation steps")
    train_parser.add_argument("--batch-size", type=_positive_int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--flow-weight", type=float, default=0.5)
    train_parser.add_argument("--flow12-weight", type=float, default=1.0)
    train_parser.add_argument("--flow21-weight", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-every", type=_positive_int, default=50)
    train_parser.add_argument("--eval-every", type=_positive_int, default=500)
    train_parser.add_argument("--eval-samples", type=_positive_int, default=2048)
    train_parser.add_argument("--wandb-project", type=str, default="conditional-nsf")
    train_parser.add_argument("--wandb-run-name", type=str, default=None)
    train_parser.add_argument("--hidden-size", type=_positive_int, default=128, help="Flow MLP hidden size")
    train_parser.add_argument("--depth", type=_positive_int, default=2, help="Flow MLP depth")
    train_parser.add_argument("--num-flow-layers", type=_positive_int, default=4)
    train_parser.add_argument("--conditioner-hidden-size", type=_positive_int, default=128)
    train_parser.add_argument("--conditioner-depth", type=_positive_int, default=2)
    train_parser.add_argument("--activation", type=str, default="tanh")
    train_parser.add_argument("--aux-hidden-size", type=_positive_int, default=128)
    train_parser.add_argument("--aux-depth", type=_positive_int, default=2)
    train_parser.add_argument("--aux-activation", type=str, default="tanh")
    train_parser.add_argument("--aux-conditioner-hidden-size", type=_positive_int, default=128)
    train_parser.add_argument("--aux-conditioner-depth", type=_positive_int, default=2)
    train_parser.add_argument("--aux-num-flow-layers", type=_positive_int, default=2)
    train_parser.add_argument("--aux-scale-fn", type=str, default="tanh_exp")
    train_parser.add_argument("--no-aux-time-ratio", action="store_true", help="Disable time-ratio feature in auxiliary model")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-data":
        config = DatasetConfig(
            output_path=args.output,
            num_trajectories=args.num_trajectories,
            seed=args.seed,
            dmp_basis=args.dmp_basis,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            weight_scale=args.weight_scale,
            start_noise=args.start_noise,
            goal_noise=args.goal_noise,
            video_dir=args.video_dir,
        )
        path = generate_pusht_dataset(config)
        print(f"Dataset saved to {path}")
        return

    if args.command == "train":
        training_config = TrainingConfig(
            dataset_path=args.dataset,
            num_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            flow_weight=args.flow_weight,
            flow_12_weight=args.flow12_weight,
            flow_21_weight=args.flow21_weight,
            seed=args.seed,
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_samples=args.eval_samples,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        flow_config = FlowNetworkConfig(
            state_dim=0,
            condition_dim=0,
            hidden_size=args.hidden_size,
            depth=args.depth,
            activation=args.activation,
            num_flow_layers=args.num_flow_layers,
            conditioner_hidden_size=args.conditioner_hidden_size,
            conditioner_depth=args.conditioner_depth,
            include_initial_time=False,
        )
        aux_config = AuxiliaryConfig(
            hidden_size=args.aux_hidden_size,
            depth=args.aux_depth,
            activation=args.aux_activation,
            include_initial_time=False,
            include_time_ratio=not args.no_aux_time_ratio,
            conditioner_hidden_size=args.aux_conditioner_hidden_size,
            conditioner_depth=args.aux_conditioner_depth,
            num_flow_layers=args.aux_num_flow_layers,
            scale_fn=args.aux_scale_fn,
        )
        train(training_config, flow_config, aux_config)
        return

    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
