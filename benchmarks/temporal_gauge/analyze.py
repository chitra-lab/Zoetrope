"""Aggregate completed evaluation-v2 runs and produce a reviewable report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_COMMIT='76bb952a841c3671ec35161149e780167fd9b7b1'
BRANCH='experiments/temporal-gauge-20261013'

def table(frame, columns):
    view=frame[columns].copy()
    for c in columns:
        if pd.api.types.is_numeric_dtype(view[c]):view[c]=view[c].map(lambda x:f'{x:.4f}' if np.isfinite(x) else '—')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in view.to_numpy())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('experiments/temporal_gauge'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    expected=[f'{candidate}_s{seed}' for seed in [20260622,20260623,20260624] for candidate in ['baseline','tissue_range']]
    expected+=['eikonal_s20260622','normal_hessian_s20260622']
    expected+=[f'{candidate}_{setting}_s20260622' for setting in ['A_heldout','B_4ages','B_4ages_rotated'] for candidate in ['baseline','tissue_range']]
    for name in expected:
        d=json.loads((a.root/name/'gauge_results.json').read_text())
        assert d['evaluation_version']==2, name
        settings=json.loads((a.root/name/'experiment_settings.json').read_text())
        row=dict(run=name,setting='A held-out' if 'A_heldout' in name else 'B 4 ages rotated' if 'B_4ages_rotated' in name else 'B 4 ages' if 'B_4ages' in name else 'B 10 ages',
            candidate=settings['candidate'],seed=settings['seed'],status='completed',
            epochs=settings['epochs'],batch=settings['batch'],weight=settings['weight'],
            gauge=d['gauge_distortion'],ordering=d['ordering_spearman_mean'],anatomy=d['heldout_anatomy_rmse'],
            anatomy_coverage=d['heldout_anatomy_coverage'],fixed_F=d['fixed_gauge_flow_nrmse'],F=d['material_flow_nrmse'],
            F_coverage=d['material_flow_coverage'],contour=d['physical_contour_oracle_nrmse'],contour_coverage=d['physical_contour_coverage'],
            rho=d['rho_relative_rmse'] if d['rho_relative_rmse'] is not None else np.nan,rho_absolute=d['rho_absolute_rmse'],
            NLL=d['balanced_validation_poisson_nll'],pooled_NLL=d['stage1_summary']['validation_poisson_nll'],
            training_seconds=d['stage1_summary']['training_elapsed_sec'])
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(a.output/'comparison.csv',index=False)
    pilot=df[(df.setting=='B 10 ages')&(df.seed==20260622)]
    reps=df[(df.setting=='B 10 ages')&df.candidate.isin(['baseline','tissue_range'])]
    stats=reps.groupby('candidate')[['gauge','F','contour','rho','NLL']].agg(['mean','std'])
    stats.to_csv(a.output/'seed_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,name in zip(axes.ravel(),['baseline_s20260622','tissue_range_s20260622','eikonal_s20260622','normal_hessian_s20260622']):
        arrays=np.load(a.root/name/'evaluation_arrays.npz');hs=arrays['calibration'];hm=hs.mean(0);span=hm[-1]-hm[0]
        for time,h in zip(arrays['times'],hs):ax.plot(arrays['grid'],(h-hm[0])/span,color=plt.cm.viridis(time),lw=1.5)
        row=df[df.run==name].iloc[0];ax.set_title(f'{row.candidate}: gauge={row.gauge:.4f}')
        ax.set_xlabel('True radial anatomical depth');ax.set_ylabel('Learned coordinate / pooled span');ax.grid(alpha=.15)
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0,1)),ax=axes.ravel().tolist(),label='Developmental time',fraction=.025,pad=.02)
    fig.suptitle('Age-specific calibration on common depth support [0.05, 0.95]')
    fig.subplots_adjust(top=.90,right=.9,hspace=.28,wspace=.22);fig.savefig(a.output/'calibration_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,title in zip(axes,['gauge','F','contour'],['Gauge distortion','Material-flow NRMSE','Physical-contour NRMSE']):
        for seed in sorted(reps.seed.unique()):
            pair=reps[reps.seed==seed].set_index('candidate')
            ax.plot([0,1],[pair.loc['baseline',metric],pair.loc['tissue_range',metric]],'o-',label=str(seed))
        ax.set(xticks=[0,1],xticklabels=['Baseline','Tissue range'],title=title);ax.grid(alpha=.15)
    axes[-1].legend(title='Training seed',fontsize=8);fig.tight_layout();fig.savefig(a.output/'seed_comparison.png',dpi=180);plt.close(fig)
    summaryrows=[]
    for candidate in ['baseline','tissue_range']:
        g=reps[reps.candidate==candidate];r={'candidate':candidate}
        for metric in ['gauge','F','contour','rho','NLL']:r[metric]=f'{g[metric].mean():.4f} ± {g[metric].std(ddof=1):.4f}'
        summaryrows.append(r)
    summarytable=table(pd.DataFrame(summaryrows),['candidate','gauge','F','contour','rho','NLL'])
    development=table(pilot,['candidate','gauge','ordering','anatomy','F','contour','rho','NLL'])
    confirmation=table(df[df.setting!='B 10 ages'],['setting','candidate','gauge','ordering','F','contour','rho_absolute','NLL'])
    allruns=table(df,['run','gauge','fixed_F','F','F_coverage','contour','rho_absolute','NLL'])
    content=f'''# Temporal gauge consistency in Flow-STP Stage 1

**Recommendation: keep the selected production model unchanged.** No tested modification reliably satisfies the combined gauge, anatomy, material-flow, progression, likelihood, and seed-stability criteria. Tissue-range normalization is an exploratory control, not a validated PFC replacement.

## Mathematical analysis

Write the learned coordinate as w=h(z,t), with h_z nonzero and a consistent orientation. The decoder can absorb the inverse transformation: Ψ̃(w,t)=Ψ(h⁻¹(w,t),t). At corresponding points,

Ψ̃_w=Ψ_z/h_z and Ψ̃_t=Ψ_t−(h_t/h_z)Ψ_z.

Ignoring the numerical denominator floor, the Fisher projection therefore gives Ṽ=h_z V+h_t. The orthogonal residual satisfies q̃=q. Consequently, exact flow transforms as F̃(h(a,0),t)=h(F(a,t),t), and molecular arc length ρ=∫‖q‖dt is invariant along corresponding trajectories. The fixed epsilon slightly breaks exact covariance when tangent energy is small.

Temporal gauge drift alters the numerical interpretation of velocity; it does **not**, by itself, prove failure of material correspondence. Finite-sample fitting, derivative error, weak tangent energy, and inaccurate contour geometry can cause practical failures. Neither a shared decoder nor expression likelihood identifies a unique anatomical gauge. Minimizing velocity can select material coordinates and erase real anatomical migration. Spatial penalties constrain parameterization but are not independent biological anchors. Tissue-support endpoints are informative only if they represent corresponding anatomical boundaries across ages.

## Source and data audit

- Source inspected on main at `{BASE_COMMIT}`. Experimental code is isolated under `benchmarks/temporal_gauge/`; selected Stage 1 and every Stage-2 source file remain unchanged.
- The requested `experiments/synthetic_benchmark/final_method_validation/` and historical trained crescent datasets are not tracked in main and were unavailable here. Existing archived directional-Hessian code was inspected, but its historical experimental outcomes could not be reproduced from absent files.
- Used the existing **frozen cortical crescent v1.0 generator**, without rewriting its geometry or expression mechanism. Generated new copies because existing data/checkpoints were unavailable. Development: Regime B, data seed 20261008, 10 ages, default growing population and physical-area sampling, 48 genes, 1,800 initial cells and 36,425 total cells. Check manifests for exact counts.
- `z_true` is the radial anatomical depth on [0,1] across every age. `u_true` is the transverse/longitudinal coordinate, not the anatomical estimand called u in the task description. Rotation, translation, curvature and thickness change across tissues. Each tissue spans the same intrinsic depth support; evaluation uses [0.05,0.95].
- Training receives only public raw counts, observed XY, time, tissue identity and gene names. Hidden material labels, depths, rates and velocities enter evaluation only. Public exposure offsets are exactly the production count-total offsets; true synthetic library factors are never used for fitting.
- True migration follows dz/dt=1.8 z(1−z)[1+0.45(2z−1)][1−0.9(3t²−2t³)]. True F is its RK4 solution from material a. Regime B adds orthogonal differentiation: exact intrinsic Fisher flow agrees with prescribed migration. True ρ integrates the molecular residual speed along material a, rather than inferring it from observed counts.
- Held-out synthetic setting: Regime A (pure transport), independent data seed 20261009, same crescent geometry and 10 ages. True molecular progression is zero; absolute ρ error is reported. Four-age sensitivity uses original tissues 0,3,6,9 and a second subset 0,2,5,9, with unchanged time values and physical truth. The first excludes all rotated sections; the second retains two rotated sections. Neither gives a pure effect of time-point count with geometry fully controlled.

## Protocol and evaluation

All fits use 1,500 epochs, coordinate MLP (64,64) tanh, decoder (128,128) GELU, AdamW 0.002, weight decay 10⁻⁵, gradient clipping 5, and 256 cells per tissue per update. This economical batch differs from the production default 1,024; all comparisons use the same batch and epochs. Validation fraction is 0.1. Paired seeds are 20260622, 20260623, 20260624. Stage 2 uses unmodified Fisher derivatives and RK4 with 72 steps. No optional velocity refinement is applied.

Estimate h_t from equal-depth bins of evaluation truth on training-cell locations, with monotone regression and one global orientation. No hidden truth is involved in model training. Evaluate all curves on a common 181-point depth grid. Set R to the pooled calibration span on that grid and compute mean normalized squared deviation across ages. This makes the refined gauge score invariant to global affine rescaling. Report span, minimum ordering, held-out pooled-inverse anatomy error and coverage separately; collapsed calibrations are rejected. A fixed nonlinear monotone calibration is allowed. Ages have equal weight. This dataset has one physical tissue per age, so it does not test replicated-tissue balancing.

Gauge bootstrap intervals resample cells within each tissue (100 repeats); these describe sampling uncertainty conditional on a fitted model, not training-seed uncertainty. Independent training seeds provide the latter. Anatomical information is checked using held-out cells and a pooled inverse calibration. Geometry and library offsets use all public cells, so held-out expression validation is transductive with respect to observed tissue geometry.

**Fixed-gauge F** uses the inverse pooled curve. **Material F** uses the target age’s inverse calibration and compares to known material trajectories. Both accept fixed nonlinear reparameterization. Unsupported inverse predictions remain NaN and are explicitly counted; reported F errors are conditional on coverage, so coverage must accompany scores. Anchors span true initial depths 0.10–0.75.

**Physical contour correspondence** starts from actual source points at five transverse locations and 51 depths, computes each point’s actual learned coordinate, integrates the frozen Stage-2 field, and measures distance from the true material destination to the actual learned target level set. Target contours are evaluated on a 257×193 intrinsic mesh mapped into registered physical geometry; scalar distance uses closest line-segment points. It is a contour oracle diagnostic, **not** a prediction of a unique XY endpoint or an arclength reconstruction. No sequencer rotation/translation is counted as migration. Coverage and aggregate physical distance are retained, together with calibration curves and axial trajectory arrays. This diagnostic uses truth geometry for evaluation only.

ρ is integrated by trapezoidal quadrature of the unmodified Stage-2 orthogonal speed along stored RK4 nodes. Regime-B scores compare with original generating biological rates. Production count-total exposures can absorb biological variation in total expected expression, so absolute ρ errors cannot be attributed exclusively to gauge. NLL below is balanced equally across ages; the cell-weighted production NLL is also preserved in CSV/JSON.

## Development tournament (seed 20260622)

{development}

Lower gauge, anatomy RMSE, F NRMSE, contour NRMSE, ρ relative RMSE and NLL are better; higher ordering is better. The baseline gauge bootstrap CI is recorded in its result JSON. Eikonal weight is 0.01; normal-Hessian weight is 0.001. Both penalties use 64 fixed observed reference cells per tissue and isotropic spatial units derived from observed XY scales. Their purpose is to reduce gradient-magnitude variation, not to minimize V. This is a small pilot subset and a single weight per penalty, not a systematic negative result for every possible regularizer.

Tissue-range normalization replaces the pooled detached robust affine ruler with tissue-specific 0.5%–99.5% spatial-reference endpoints. It does not force complete cell-coordinate distributions to match, but its endpoints still depend on coverage and reference sampling. It is not an anatomy-labelled boundary constraint. This experiment's common depth support makes it a useful control; partial PFC sections need a different justification.

## Three-seed confirmation (Regime B)

Mean ± sample standard deviation across the three paired seeds:

{summarytable}

The first-seed range-normalization gains do not generalize across seeds. One additional seed substantially worsens material-flow recovery. No successful candidate is selected or promoted. Gradient and Hessian penalties were screened on one seed only after failing the development gate; their untested weights/seeds are hypotheses, not measured results.

## Held-out regime and number of ages

{confirmation}

These are completed single-seed robustness checks, not multi-seed generalization evidence. The candidate was chosen from development B before inspecting the held-out A outcome; no subsequent tuning used A. Regime A changes molecular dynamics and random realization within the same geometry family. The first four-age subset removes all rigidly rotated sections, which confounds any apparent gain from fewer ages. The rotated four-age check addresses this partially but does not retain every original geometry. No missing-time truth enters fitting. No genuinely held-out physical-tissue benchmark is claimed: each age has one independently transformed section and no supplied correspondence/registration for a withheld slice.

The rotated four-age check is a positive, context-dependent result for tissue-range normalization: material-flow NRMSE falls from 0.3574 to 0.1711, physical-contour NRMSE from 0.6767 to 0.1242, and progression relative RMSE stays similar (0.5332 versus 0.5314). This supports further controlled development of an anatomy-anchored ruler, but does not override the failed three-seed ten-age confirmation. With one tissue per age, the selected model uses time-only coordinate conditioning; discontinuously rotated sections must be represented through that smooth time-conditioned map. The difference between four and ten ages warrants an architectural/optimization diagnostic, but it does not establish that rotations alone caused the ten-age failures.

## All completed runs and coverage

{allruns}

Every row has status completed. Detailed settings, timing, per-run calibration span, uncertainty, ordering, likelihood, and coverage are in `comparison.csv`, `experiment_settings.json`, manifests and `gauge_results.json`. Earlier evaluation-v1 score files are preserved under each run's `evaluation_v1/`; they used a median source calibration for the physical-contour metric and should not be used in final comparisons. Evaluation v2 uses actual source-point levels consistently for every row above.

## Reproduction and artifacts

Experimental source branch: https://github.com/JamesOQ/Anatomical-Axis-Dynamics-Model/tree/{BRANCH}

From the repository root:

```bash
PYTHONPATH=src python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/baseline_s20260622 --epochs 1500 --batch 256
PYTHONPATH=src python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/tissue_range_s20260622 --candidate tissue_range
PYTHONPATH=src python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/eikonal_s20260622 --candidate eikonal --weight .01
PYTHONPATH=src python benchmarks/temporal_gauge/run.py --dataset experiments/temporal_gauge/data_B --output experiments/temporal_gauge/normal_hessian_s20260622 --candidate normal_hessian --weight .001
PYTHONPATH=src python benchmarks/temporal_gauge/analyze.py --output temporal_gauge_report
```

Use a new output directory for a new fit; an already completed directory reloads its frozen checkpoint for evaluation. Generate each dataset once before running parallel comparisons. Data creation uses a temporary staging directory and atomic rename. Commands for every seed, held-out A and four ages are included in `reproduce.sh`. Archived outputs include public/hidden data, trained final and last checkpoints, frozen split indices, settings, logs, calibration plots, evaluated flows and progression arrays. Scientific tests check global-scale invariance, gauge-covariant material flow and unsupported inverse predictions. Existing Stage-1 defaults, Stage-2 dynamics and frozen cortical generator tests were also run. Runtime: Python 3.12, PyTorch 2.14.1 CPU, two intra-op threads per fit.

Initial baseline training succeeded but the first evaluator invocation failed on a split-file path; loading the preserved checkpoint and correcting the path completed evaluation. Initial four-age launches read an incompletely written new dataset and failed before training; after generation finished they were rerun successfully. These failed logs are preserved. No old user experiment, production source, or Stage-2 method was overwritten.

## Decision and next question

Do not retrain PFC using these candidates as a new primary model. Retain the selected baseline. A true anatomy-anchored candidate should use independently meaningful corresponding boundaries/landmarks and should be tested under deliberately incomplete, unequal section coverage. Direct temporal penalties at matching standardized XY were not tested because changing curvature and rigid rotations make those locations incomparable. Broad penalty sweeps and direct velocity minimization were not performed.

The decisive next question is whether an observed-data anatomical anchor improves correspondence robustly while preserving expression and progression, rather than merely making numerical velocity or calibration curves smoother. The present experiment establishes a reusable way to ask that question and an informative failure of the tested simple modifications; it does not identify the biological gauge of real PFC tissue.
'''
    (a.output/'temporal_gauge_report.md').write_text(content)
    print(df[['run','gauge','F','contour','rho_absolute','NLL']].to_string(index=False))

if __name__=='__main__':main()
