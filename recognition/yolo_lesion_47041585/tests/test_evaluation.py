"""Detection-matching sanity cases."""

import pytest

pytestmark = pytest.mark.skip(reason="implement match_detections first")


def test_duplicate_prediction_matches_once():
    """Two boxes on one lesion: one match, one false positive."""


def test_wrong_class_is_not_a_match():
    """Right box, wrong class counts as a miss."""


def test_low_iou_is_not_a_match():
    """IoU below 0.5 counts as a miss."""


def test_no_predictions():
    """Every ground-truth box is missed; no error."""
