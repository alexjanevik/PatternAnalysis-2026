"""Routing branch checks."""

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet")


def test_no_detection_routes_to_monitor():
    """Absent detections are monitored, not cleared."""


def test_all_monitor_and_all_clear_policies():
    """Denominators stay correct at the extremes."""
