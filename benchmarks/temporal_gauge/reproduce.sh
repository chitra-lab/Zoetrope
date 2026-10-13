#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
# Run from repository root. Generate datasets once before comparisons.
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output unused --generate-only
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_A_heldout --output unused --regime A --data-seed 20261009 --generate-only
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B_4ages --output unused --ages 4 --generate-only
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B_4ages_rotated --output unused --ages 4 --selected-tissues 0,2,5,9 --generate-only
for seed in 20260622 20260623 20260624; do
  for candidate in baseline tissue_range; do
    python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output "experiments/temporal_gauge/${candidate}_s${seed}" --candidate "$candidate" --seed "$seed"
  done
done
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/eikonal_s20260622 --candidate eikonal --weight .01
python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/normal_hessian_s20260622 --candidate normal_hessian --weight .001
for candidate in baseline tissue_range; do
  python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_A_heldout --output "experiments/temporal_gauge/${candidate}_A_heldout_s20260622" --candidate "$candidate" --regime A --data-seed 20261009
  python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B_4ages --output "experiments/temporal_gauge/${candidate}_B_4ages_s20260622" --candidate "$candidate" --ages 4
  python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B_4ages_rotated --output "experiments/temporal_gauge/${candidate}_B_4ages_rotated_s20260622" --candidate "$candidate" --ages 4 --selected-tissues 0,2,5,9
done
python -m pytest -q benchmarks/temporal_gauge/test_gauge.py tests/test_stage1a_defaults.py tests/test_stage2_dynamics.py tests/test_cortical_benchmark.py
python benchmarks/temporal_gauge/analyze.py --output temporal_gauge_report
