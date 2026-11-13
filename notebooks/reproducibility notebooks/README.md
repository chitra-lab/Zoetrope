# Zoetrope paper experiments

These notebooks accompany Figures 3–6 and Appendix E of *Zoetrope efficiently maps spatial trajectories in time-series spatial transcriptomics data*. The directory layout preserves the original division between synthetic and PFC experiments. Start with the figure notebooks and frozen inputs to replay the manuscript. The model-training notebooks are available for rerunning individual fits.

## Configure paths and kernels

Set the experiment root and source checkout before starting Jupyter. Use absolute paths.

```bash
export STP_PROJECT="/path/to/experiment-workspace"
export FLOW_STP_REPO="/path/to/zoetrope-checkout"
export QIAN_PFC_DATA_ROOT="/path/to/zenodo_14422018_processed"
export STVCR_REPO="/path/to/stVCR"
export STVCR_BENCHMARK_PYTHON="/path/to/stvcr-env/bin/python"
```

On Skipjack, the original `STP_PROJECT` is the `STP flow` directory above `Final experiments`. Keep the existing relative layout of `experiments/`, `paper_figures/`, and `Experiments/simulation validation/`. The notebooks also honor their more specific path overrides. Keep the original experiment receipts and verify their hashes against the associated files when copying inputs. Use a separate experiment workspace for fresh training.

Use the Zoetrope analysis kernel for the generator, Zoetrope fits and most evaluations. Install the repository's `flow_stp` package for cells that import it, alongside the analysis packages used by the notebooks (NumPy, pandas, SciPy, Matplotlib, PyTorch, h5py, Scanpy and Shapely as applicable). The original synthetic fit and GPU timing code check the Zoetrope revision `76bb952a841c3671ec35161149e780167fd9b7b1`. Use the original method-specific GPU kernels for STORIES, moscot and stVCR training and native exports. The PFC native moscot forecast uses moscot 0.5.2 and its private interpolation API. The synthetic stVCR endpoint exporter checks upstream revision `26aa79a63eba7a5e21726b1eb95bf6bb61cfe699`. Preserve the environments recorded in the original fit manifests for an exact replay.

## Paper run order

| Result | Notebook under the dataset directory | Inputs and expected result |
|---|---|---|
| Figure 3 | Synthetic `synthetic data generation`, then `Figures/Synthetic data overview` | Seed 20261023. The generator produces 36,425 observed cells across ten sections and a separate held-out reference. It writes public training files and evaluator-only truth in separate trees. |
| Figure 4A | Synthetic `Figures/common coordinate axis synthetic dataset validation` | Frozen ten-tissue Zoetrope fit and its per-cell coordinate table. Mean absolute Spearman 0.9967, minimum 0.9928. Run before the Figure 4A–B plotting notebook. |
| Figure 4B | Synthetic `Figures/held out expression benchmark` | Frozen four-method forecasts and bootstrap tables, original source counts, independent held-out expected rates, and Figure 4A tables. MMD 0.025, 0.076, 0.081, 0.103 in Zoetrope, STORIES, stVCR, moscot order. The moscot panel recomputes the conditional-barycentric MMD and 200 bootstrap replicates. |
| Figure 4C–D | Synthetic `Figures/spatial trajectories synthetic dataset benchmark` | Frozen endpoint NPZ, auxiliary transverse endpoint NPZ, endpoint evaluation CSV and simulator truth. The `Experiments/spatial trajectories benchmark` notebook exports these inputs from the fitted models. The NRMSE cohort requires finite predictions from the three plotted methods and the auxiliary Zoetrope transverse adapter. The notebook exports its exact pair IDs in `figure_4c_cohort.csv`. Expect 1,403 common pairs and pooled NRMSE 0.691, 3.169, 3.822. The figure notebook also reports the 1,620-pair centroid-fallback sensitivity. |
| Figure 4E–G | Synthetic `Figures/trajectory progression synthetic dataset benchmark` | Frozen progression values, coordinate data and provenance. Pooled signed Spearman 0.895 for Zoetrope and −0.470 for STORIES (−J). |
| Figure 5A–B | PFC `Figures/Anatomical axis PFC dataset figures` | Nine-section fit, integrated atlas and saved Poisson input bins. Run the matching cell before the coordinate figure and the regression cell before the profile figure. GW20 Tissue 1 ARI is 0.746 for NEUROD2 and 0.707 for NEUROD6. |
| Figure 5C–D | PFC `Figures/runtime comparison` and `Figures/held-out expression PFC benchmark` | Run the timing notebook to replay the original four core timing receipts, then the figure cell with the frozen score CSV. For fresh scoring follow the prepare/score instructions in the notebook. MMD 0.238, 0.427, 0.383, 0.179 in Zoetrope, STORIES, stVCR, moscot order. |
| Figure 6A–B | PFC `Figures/Spatial Trajectories PFC benchmark` | Nine-section fit, integrated atlas, and original cohort, corrected stVCR endpoint, and moscot barycenter tables. Figure 6A uses 2,130 selected cells and Figure 6B uses 402. Zoetrope reports contour-length fractions and comparators report endpoint fractions. |
| Figure 6C | PFC `Figures/Trajectory Progression` | Frozen matched-cell score CSV. The scoring cell uses within-age ranks for display and averages six within-age, pairwise AUCs. Expect 1,454 cells and AUC 0.818 for Zoetrope, 0.459 for STORIES. |

## Inputs to carry over from the original experiment workspace

The source ZIP contained the notebooks. Exact replay also needs the archived data and intermediate files below. Paths in this section are relative to `STP_PROJECT` unless noted.

- **Synthetic data and trained fits.** Keep the original `synthetic_developmental_crescent_v1` tree (or run its generator) and the relevant model folders under `experiments/synthetic_benchmark/`. For strict byte-level provenance, copy the original dataset alongside the original predictions. A fresh generation uses the same seed, but serialization can vary with the software environment.
- **Synthetic held-out expression.** Copy `experiments/synthetic_benchmark/publication_results/heldout_expression_t2p5_seed20261023_decisive_diagnostics_v1/`, including `all_source_forecasts_FROZEN_BEFORE_TRUTH.npz`, `independent_target_population_MMD.csv`, and the summary and `bootstrap_draws.csv` under `revised_distributional_MMD_and_matched_RMSE_v1/`. The original bootstrap tables for Zoetrope, STORIES, and stVCR are inputs to the final plotting cell.
- **Synthetic spatial endpoints and progression.** Copy `frozen_public_native_and_adapter_endpoints.npz` and `per_cell_spatial_endpoints.csv.gz` from `experiments/synthetic_benchmark/synthetic_developmental_crescent_v1_displacement_final_v1/` for direct figure replay. Under that same directory, copy `geometry_only_harmonic_transverse_all_tissues_v1/geometry_only_transverse_all_tissues_PUBLIC_ONLY.npz`. Copy `experiments/synthetic_benchmark/publication_results/progression_TI_vs_STORIES_seed20261023_native_potential_v3/` with the evaluation-only per-cell progression table, original Spearman tables, and provenance.
- **PFC atlas and anatomical fits.** Provide `gw15.h5ad`, `gw20.h5ad`, `gw22.h5ad`, and `merscope_integrated_855.h5ad` in the processed Qian data root. Copy the nine-section run at `experiments/qian_pfc_spatial_ti_all9_synthetic_defaults_v1/CPU_FULL/` and the original section-wise bin/count and ranking files under `paper_figures/pfc_z_joint_poisson_dp_9sections_v1/`.
- **PFC held-out expression.** For plotting, copy `GW20_PRIMARY_AND_BASELINES.csv` under `experiments/qian_pfc_publication_4way_20261024/GW20_heldout_fourway_Fisher_flow_vs_native_v1/GW20_NATIVE_MOSCOT_INTERPOLATOR_GENE_SHARE_v1/`. For fresh scoring, copy the full frozen GW20 benchmark tree, including the original prediction hashes, the five-section GW15 Tissue 2 forecast and receipt, the original training-only bandwidth manifest and cohort table, and the native moscot forecast and receipt. Also retain the five-section Zoetrope run with its gene-order file. Use a working copy and move the existing primary score CSV out of the score output directory before rescoring. Keep the freeze receipts. The preparation phase precedes any GW20 evaluation reads.
- **PFC timing.** Copy the original `TIMING_SUCCESS.json` under `experiments/qian_pfc_timing_three_sections_1_2_7_v1/GPU_FULL/` and the three comparator receipts referenced by the timing notebook, or reuse the existing `PFC_three_age_core_training_times.csv`. The timing notebook replays receipts by default. Set `ZOETROPE_RERUN_TIMING=1` in a separate GPU experiment workspace to run the Zoetrope timing experiment.
- **PFC spatial trajectories.** Copy `paper_figures/qian_ENETSP_spatial_transport_LATEST_3AGE_BARYCENTRIC_v1/` and `paper_figures/qian_PFC_H2_H3_Fisher_transition_matrices_TEXT_v1/`, including the frozen cohorts, barycentric tables, corrected native stVCR endpoints, frame-validation receipt, and `TOP10_CONTOUR_PHYSICAL_BAND_SCORES.csv`. The optional export can rebuild the EN-ET-SP cohort and moscot barycenters. The H2 moscot endpoint and cohort tables are archived inputs.
- **PFC progression.** Copy `paper_figures/qian_PFC_Zoetrope_vs_STORIES_ordering_v1/` with the frozen matched score table. The optional scoring replay additionally requires `PART_A_LOCKED_MATCHED_CELLS.csv` and the full three-age STORIES checkpoint and preprocessing artifacts.

The paper's nine-section PFC fit and three-age benchmark models include GW20. Use the dedicated GW15/GW22 fits for held-out expression. The generator and all four Figure 3 panel exports were run locally after cleanup. The remaining benchmarks require the archived experiment inputs or their original compute environments.
