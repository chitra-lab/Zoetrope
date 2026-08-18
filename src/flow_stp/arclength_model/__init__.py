"""Independent arclength-model prototype.

This namespace is intentionally separate from ``flow_stp.stage1`` and
``flow_stp.stage2``.  The arclength model jointly learns an anatomical
coordinate and a spatiotemporal mean-expression field, then derives
registration and transcriptomic arclength quantities from that fitted field.

Nothing in this package should silently import or reuse the staged Stage 1 / 
Stage 2 models unless that dependency is made explicit in an experiment.
"""

ARCLENGTH_MODEL_VERSION = "arclength-prototype-v0-2026-08-18"

__all__ = ["ARCLENGTH_MODEL_VERSION"]
