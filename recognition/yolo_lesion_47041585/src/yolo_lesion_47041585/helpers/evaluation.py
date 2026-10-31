"""
Scoring: box matching, detection metrics, routing outcomes, uncertainty and timing.

`match_detections` is needed first; the other functions are placeholders for the
interfaces the later experiments will use.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from yolo_lesion_47041585.dataset import NormalizedBox
from yolo_lesion_47041585.predict import Detection


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching predictions to ground truth.

    Attributes:
        matches (tuple[tuple[Detection, Detection, float], ...]): (prediction, truth,
            IoU) for every class-correct match.
        false_positives (tuple[Detection, ...]): Predictions with no match, including
            duplicates and wrong-class boxes.
        misses (tuple[Detection, ...]): Ground-truth boxes with no match.
    """

    matches: tuple[tuple[Detection, Detection, float], ...]
    false_positives: tuple[Detection, ...]
    misses: tuple[Detection, ...]


def box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """IoU of two normalised centre-format boxes.

    Normalising by the same image width and height scales the intersection and the union
    by the same factor, so the result equals the pixel IoU.

    Args:
        a (Detection): First box.
        b (Detection): Second box.

    Returns:
        float: Intersection over union in [0, 1].
    """
    # compute box edge coordinates
    ax1, ax2 = a.x_center - a.width / 2, a.x_center + a.width / 2
    ay1, ay2 = a.y_center - a.height / 2, a.y_center + a.height / 2
    bx1, bx2 = b.x_center - b.width / 2, b.x_center + b.width / 2
    by1, by2 = b.y_center - b.height / 2, b.y_center + b.height / 2

    # compute overlap along each axis
    overlap_w = min(ax2, bx2) - max(ax1, bx1)
    overlap_h = min(ay2, by2) - max(ay1, by1)

    # if there is no overlap, return 0.0
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0

    # compute IoU
    intersection = overlap_w * overlap_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union


def match_detections(
    predictions: Sequence[Detection],
    truths: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """One-to-one, class-aware matching of predictions to ground truth by IoU.

    Boxes are only compared within the same image. Predictions are taken in order of
    decreasing confidence and each claims the unmatched same-class ground truth it
    overlaps most, provided the IoU is at least `iou_threshold`. A ground-truth box is
    matched at most once, so duplicates become false positives, and a wrong-class box
    is both a false positive and a miss.

    Args:
        predictions (Sequence[Detection]): Predicted boxes for any number of images.
        truths (Sequence[Detection]): Ground-truth boxes. Their confidence is ignored.
        iou_threshold (float): Minimum IoU for a match.

    Returns:
        MatchResult: The matches, unmatched predictions and unmatched ground truth.
    """
    # group ground-truth boxes by image for faster lookup
    truths_by_image: dict[str, list[Detection]] = defaultdict(list)
    for truth in truths:
        truths_by_image[truth.image_id].append(truth)

    claimed: set[int] = set()  # indices into `truths` already matched
    index_of = {id(truth): i for i, truth in enumerate(truths)}

    matches = []
    false_positives = []
    # go through predictions from most to least confident
    for pred in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best, best_iou = None, 0.0
        for truth in truths_by_image.get(pred.image_id, ()):
            # if the classes don't match or the truth is already claimed, skip it
            if truth.class_id != pred.class_id or index_of[id(truth)] in claimed:
                continue
            iou = box_iou(pred.box, truth.box)
            # if IoU is above the threshold and it's the best so far, remember it
            if iou >= iou_threshold and iou > best_iou:
                best, best_iou = truth, iou
        # if there is no match, it's a false positive, otherwise claim the truth and
        # record the match
        if best is None:
            false_positives.append(pred)
        else:
            claimed.add(index_of[id(best)])
            matches.append((pred, best, best_iou))

    # any unmatched truths are misses
    misses = tuple(t for i, t in enumerate(truths) if i not in claimed)

    return MatchResult(tuple(matches), tuple(false_positives), misses)


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
