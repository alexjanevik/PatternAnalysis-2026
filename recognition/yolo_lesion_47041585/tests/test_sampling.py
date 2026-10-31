"""Sampler checks."""

import pytest

pytestmark = pytest.mark.skip(reason="implement EpochSampler first")


def test_epoch_length_is_fixed():
    """Draw count equals the training split size regardless of target fraction."""


def test_malignant_fraction_is_close_to_target():
    """Allow stochastic variation; do not assert exact proportions."""


def test_epochs_differ_but_are_reproducible():
    """set_epoch(n) gives the same draws every time but different draws per epoch."""
