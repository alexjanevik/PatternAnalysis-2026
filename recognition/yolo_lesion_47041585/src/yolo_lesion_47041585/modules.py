"""
Detector construction for the lesion experiments.

Wraps the Ultralytics YOLO11n detector (COCO-pretrained `yolo11n.pt`) and the trainer
subclass that plugs the project's sampler into it. The detector architecture, loss and
NMS are supplied by the framework and are attributed as such in the README. This module
does not split data, choose thresholds or draw figures.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from yolo_lesion_47041585.dataset import EpochSampler


def build_detector(checkpoint: str = "yolo11n.pt") -> YOLO:
    """Load a detection checkpoint (not a classification or segmentation variant).

    Args:
        checkpoint (str): Path or name of the weights to load.

    Returns:
        YOLO: The loaded detector.
    """
    raise NotImplementedError("TODO")


def validate_model_contract(model: YOLO, class_names: Sequence[str]) -> None:
    """Check the model is a detector whose class order matches the project mapping.

    Raises:
        ValueError: If the task or class names differ from `class_names`.
    """
    raise NotImplementedError("TODO")


class LesionDetectionTrainer(DetectionTrainer):
    """DetectionTrainer whose training loader uses the project's EpochSampler.

    Inspect `DetectionTrainer.get_dataloader` in the pinned Ultralytics version before
    overriding: the stock `build_dataloader` only honours a sampler when `rank != -1`,
    so the sampler must be passed to the DataLoader explicitly. Validation must stay
    untouched, unaugmented and unweighted.
    """

    sampler: EpochSampler | None = None

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"
    ):
        raise NotImplementedError("TODO: inject the sampler for mode == 'train'")


def trainer_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a resolved run config into the keyword overrides Ultralytics expects."""
    raise NotImplementedError("TODO")
