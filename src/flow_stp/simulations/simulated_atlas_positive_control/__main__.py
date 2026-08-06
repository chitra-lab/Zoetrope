"""Command-line entry point for the positive-control atlas."""

from __future__ import annotations

import argparse
import json

from .config import SimulationConfig
from .io import save_simulation
from .plotting import save_diagnostic_plots
from .simulation import run_simulation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Flow-STP simulated cortical atlas positive control."
    )
    parser.add_argument(
        "--output-dir",
        default="simulated_atlas_positive_control",
        help="Directory receiving CSV/NPZ truth and observation files.",
    )
    parser.add_argument("--cells-per-tissue", type=int, default=10_000)
    parser.add_argument("--timepoints", type=int, default=6)
    parser.add_argument("--max-bend-deg", type=float, default=120.0)
    parser.add_argument("--n-z", type=int, default=401)

    parser.add_argument(
        "--constant-q",
        type=float,
        default=0.98,
        help=(
            "Constant developmental source q=d(rho)/dt "
            "used for all populations and times."
        ),
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--no-expected-mean",
        action="store_true",
        help="Do not write the large expected_mean.csv.gz file.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save diagnostic PNG plots under OUTPUT_DIR/plots.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = SimulationConfig(
        n_timepoints=args.timepoints,
        n_cells_per_tissue=args.cells_per_tissue,
        maximum_bend_deg=args.max_bend_deg,
        n_z=args.n_z,
        constant_q=args.constant_q,
        random_seed=args.seed,
    )
    result = run_simulation(config)
    output = save_simulation(
        result,
        args.output_dir,
        save_expected_mean=not args.no_expected_mean,
    )
    if args.plots:
        save_diagnostic_plots(result, output / "plots")
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    print(f"Saved simulation to: {output.resolve()}")


if __name__ == "__main__":
    main()
