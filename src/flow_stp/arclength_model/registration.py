"""Smooth one-dimensional registration for the arclength model.

Given fixed M_z and M_t, this module will solve the regularized quadratic
problem for V(z,t) that minimizes residual temporal expression change
M_t + V M_z subject to magnitude and smoothness penalties.
"""
