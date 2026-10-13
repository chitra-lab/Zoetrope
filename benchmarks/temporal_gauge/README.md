# Temporal anatomical-gauge benchmark

Experimental benchmark based on main commit `76bb952a841c3671ec35161149e780167fd9b7b1`.
Production Stage 1, frozen crescent generator, and Stage 2 are unchanged.

The runner uses public raw counts and XY only. Hidden truth enters evaluation,
including equal-depth calibration, scale-invariant gauge distortion, held-out
anatomical information, gauge-aware material flow, actual source-point-to-target
learned contour correspondence, molecular progression, and age-balanced NLL.
See `analyze.py` for the complete mathematical explanation and reporting protocol.

`experimental_stage1.py` copies the selected source into an isolated module and
injects one objective hook or replaces its affine gauge refresh. Candidates:

- `tissue_range`: per-tissue robust spatial-reference range, with a common-support
  assumption. This is a control, not an anatomy-landmark gauge.
- `eikonal`: gradient-norm variation penalty, weight 0.01.
- `normal_hessian`: normal derivative of gradient magnitude, weight 0.001.

Run `bash benchmarks/temporal_gauge/reproduce.sh` from the repository root.
Use new output directories for new training. Existing completed runs reload
their checkpoints. Generated data and all model outputs stay under the ignored
`experiments/temporal_gauge/` directory. Generate datasets once before starting
parallel fits. Report output is specified independently with `analyze.py --output`.

The comparison uses 1,500 epochs and 256 cells per tissue per update; it is an
economical experiment, not a reproduction of the default 1,024-cell batch.
Development B has three paired training seeds. Additional checks use independent
Regime A (zero true progression) and four observed ages. These additional settings
are single-seed checks. No withheld physical-tissue generalization is claimed.

Gauge drift is not equivalent to correspondence error: exact Fisher flow obeys
`V_tilde = h_z V + h_t`, the flow map transforms covariantly, and the residual and
its integrated speed are invariant. The benchmark reports numerical gauge and
material correspondence separately. Unsupported calibration inversions are
counted rather than silently clipped. The physical contour oracle does not claim
unique XY trajectory recovery. Evaluation-v1 artifacts, if present, are retained;
only v2 uses each physical source point's actual learned level.
