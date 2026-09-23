import dataclasses

from flow_stp.zoo import (
    METHODS,
    PRIMARY_DEFAULT_CONFIG,
    SECONDARY_DEFAULT_CONFIG,
    fit_primary,
    fit_secondary,
)


def test_frozen_registry_contains_only_primary_and_secondary():
    assert set(METHODS) == {"primary", "secondary"}
    assert METHODS["primary"] is fit_primary
    assert METHODS["secondary"] is fit_secondary


def test_primary_defaults_are_frozen_and_deterministic():
    assert dataclasses.is_dataclass(PRIMARY_DEFAULT_CONFIG)
    assert PRIMARY_DEFAULT_CONFIG.deterministic is True
    assert PRIMARY_DEFAULT_CONFIG.seed == 20260622
    assert PRIMARY_DEFAULT_CONFIG.learning_rate == 1e-3
    assert PRIMARY_DEFAULT_CONFIG.weight_decay == 1e-5
    assert PRIMARY_DEFAULT_CONFIG.coord_hidden == (64, 64)
    assert PRIMARY_DEFAULT_CONFIG.decoder_hidden == (128, 128)


def test_secondary_defaults_are_fixed_for_validation():
    assert SECONDARY_DEFAULT_CONFIG.seed == 20260908
    assert SECONDARY_DEFAULT_CONFIG.local_k == 16
    assert SECONDARY_DEFAULT_CONFIG.pca_rank == 12
    assert SECONDARY_DEFAULT_CONFIG.curve_knots == 12
    assert SECONDARY_DEFAULT_CONFIG.registration_steps == 2000
