"""Box geometry checks."""

import pytest

pytestmark = pytest.mark.skip(reason="needs predict.py's coordinate mapping")


def test_resize_and_pad_round_trip():
    """Letterboxed boxes map back to the original frame, including edge-touching boxes."""
