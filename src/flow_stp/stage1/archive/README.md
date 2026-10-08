# Historical Stage-1 implementations (archived)

These are preserved, **unchanged from main before the freeze**, for
reproducibility. They are not imported or executed by the selected Stage-1 API.

- `stage1a.py`: old independent/per-tissue and shared-time direct-Psi
  implementations, with the formerly selected direct Fisher-flat expression
  decoder `Psi = b + f(z,t)`. Historical method version:
  `stage1a_library_offset_fisher_flat_v1`.
- `stage1a_optimized.py`: optimizer preset of the direct-Psi model.
- `stage1a_shared.py`: shared-time per-tissue/global gauge wrappers
  around the direct-Psi model.
- `legacy_test_stage1a_defaults.py`: preservation copy of the old
  direct-Psi default tests (not collected by pytest).

The active selected method is now `flow_stp.stage1.stage1a`:
`alpha(x,y,t)` with **one tissue per time** or
`alpha(x,y,t, onehot(physical_tissue_id))` with **repeated timepoints**,
always with a global gauge and log-E Poisson decoder.

Archived checkpoints and current log-E checkpoints are **not** mutually
loadable. Old entry-point imports should explicitly target
`flow_stp.stage1.archive`, or use the historical Git commit for exact
reproduction. Other prior models no longer present in the repository remain
recoverable from Git history.
