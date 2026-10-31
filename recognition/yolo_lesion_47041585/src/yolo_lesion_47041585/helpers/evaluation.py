"""
Scoring: box matching, detection metrics, routing outcomes, uncertainty and timing.

`match_detections` is needed first; the other functions are placeholders for the
interfaces the later experiments will use.
"""

from collections.abc import Sequence

from yolo_lesion_47041585.predict import Detection


def match_detections(
    predictions: Sequence[Detection],
    truths: Sequence[Detection],
    iou_threshold: float = 0.5,
):
    """One-to-one, class-aware matching of predictions to ground truth by IoU.

    Duplicates, wrong classes, IoU below threshold and empty predictions must all count
    as failures; a ground-truth box is matched at most once.
    """
    raise NotImplementedError("TODO")


def evaluate_detection(*args, **kwargs):
    """mAP50, mAP50:95, precision, recall and per-class matched IoU."""
    raise NotImplementedError("TODO")


def evaluate_routing(*args, **kwargs):
    """Malignant sensitivity and benign escalation under a PolicySpec."""
    raise NotImplementedError("TODO")


def bootstrap_by_patient(*args, **kwargs):
    """Patient-level bootstrap intervals."""
    raise NotImplementedError("TODO")


def profile_inference(*args, **kwargs):
    """Batch-1 latency including preprocessing, transfers and post-processing."""
    raise NotImplementedError("TODO")
