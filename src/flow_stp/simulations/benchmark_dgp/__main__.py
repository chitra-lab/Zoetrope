"""Command-line entry point for the comparative benchmark simulator."""

from __future__ import annotations

import argparse

from .config import BenchmarkSimulationConfig
from .simulation import run_benchmark_simulation, save_benchmark_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a method-independent spatial-TI benchmark dataset."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cells-per-tissue", type=int, default=3000)
    parser.add_argument("--n-genes", type=int, default=128)
    parser.add_argument("--replicates-per-time", type=int, default=1)
    parser.add_argument("--off-axis-strength", type=float, default=0.0)
    parser.add_argument("--parallel-strength", type=float, default=0.0)
    parser.add_argument(
        "--observation-model",
        choices=("negative_binomial", "poisson"),
        default="negative_binomial",
    )
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--no-expected-mean",
        action="store_true",
        help="Do not save hidden expected count means.",
    )
    args = parser.parse_args()

    config = BenchmarkSimulationConfig(
        cells_per_tissue=args.cells_per_tissue,
        n_genes=args.n_genes,
        replicates_per_time=args.replicates_per_time,
        off_axis_strength=args.off_axis_strength,
        parallel_strength=args.parallel_strength,
        observation_model=args.observation_model,
        random_seed=args.seed,
    )
    result = run_benchmark_simulation(config)
    output = save_benchmark_simulation(
        result,
        args.output_dir,
        save_expected_mean=not args.no_expected_mean,
    )
    print(f"Saved benchmark simulation to {output}")


if __name__ == "__main__":
    main()
