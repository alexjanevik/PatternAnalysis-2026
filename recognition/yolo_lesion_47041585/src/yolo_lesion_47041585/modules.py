"""
Detector construction for the lesion experiments.

Wraps the Ultralytics YOLO11n detector (COCO-pretrained `yolo11n.pt`) and the trainer
subclass that plugs the project's sampler into it. The detector architecture, loss and
NMS are supplied by the framework and are attributed as such in the README. This module
does not split data, choose thresholds or draw figures.
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from yolo_lesion_47041585.dataset import (
    EpochSampler,
    SplitRecord,
    build_sampling_weights,
)


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
    """DetectionTrainer whose training loader draws images with the project's sampler.

    The stock `build_dataloader` only honours a sampler when training across processes,
    so for the "replacement" scheme this builds the training `InfiniteDataLoader` itself
    and hands it an `EpochSampler`, drawing with replacement (uniformly if there is no
    target fraction). The "stock" scheme, and validation under either, use the stock
    loader unchanged, which shuffles without replacement and is never weighted.

    Args:
        cfg: Ultralytics default configuration.
        overrides (dict[str, Any] | None): Training arguments for Ultralytics.
        _callbacks (dict[str, Any] | None): Ultralytics callbacks.
        train_records (Sequence[SplitRecord]): The training split's records, whose
            ISIC IDs must match the images in the training folder.
        scheme (Literal["stock", "replacement"]): How training images are drawn.
        target_fraction (float | None): Expected malignant share of each epoch, or None
            to keep the natural class balance. Only valid with "replacement".

    Raises:
        ValueError: If a target fraction is given with the "stock" scheme.

    Attributes:
        train_sampler (EpochSampler | None): The sampler last given to the training
            loader, kept so a run can log what it drew.
    """

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, Any] | None = None,
        *,
        train_records: Sequence[SplitRecord],
        scheme: Literal["stock", "replacement"],
        target_fraction: float | None,
    ) -> None:
        if scheme == "stock" and target_fraction is not None:
            raise ValueError("The stock loader cannot sample to a malignant fraction.")
        # pyright doesnt know DetectionTrainer.__init__ takes overrides and _callbacks
        super().__init__(cfg, overrides, _callbacks)  # pyright: ignore[reportUnknownMemberType]
        self.train_sampler: EpochSampler | None = None
        self._weight_by_id: dict[str, float] | None = None
        if scheme == "replacement":
            weights = build_sampling_weights(train_records, target_fraction)
            ids = [r.audited.record.isic_id for r in train_records]
            self._weight_by_id = dict(zip(ids, weights.tolist()))

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> InfiniteDataLoader:
        # only use weighted loader for training
        if mode != "train" or self._weight_by_id is None:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1:
            raise ValueError("Weighted sampling supports single-process training only.")

        # build the dataset and line up the weights with the order it loaded
        dataset = cast(YOLODataset, self.build_dataset(dataset_path, mode, batch_size))

        # weights are per record, so line them up with the order the dataset loaded
        # pyright doesnt know YOLODataset has an im_files attribute
        files = cast(list[str], dataset.im_files)  # pyright: ignore[reportUnknownMemberType]
        ids = [Path(file).stem for file in files]
        if set(ids) != self._weight_by_id.keys() or len(ids) != len(set(ids)):
            raise ValueError(
                f"Training images do not match the training records: found {len(ids)} "
                f"images for {len(self._weight_by_id)} records."
            )
        weights = np.array([self._weight_by_id[i] for i in ids], dtype=np.float64)

        # pyright doesnt know self.args has seed and workers attributes
        args = cast(Any, self.args)  # pyright: ignore[reportUnknownMemberType]
        seed, workers = int(args.seed), min(os.cpu_count() or 1, int(args.workers))
        self.train_sampler = EpochSampler(weights, len(dataset), seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            sampler=self.train_sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=cast(Any, dataset).collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            prefetch_factor=4 if workers > 0 else None,
        )


def trainer_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a resolved run config into the keyword overrides Ultralytics expects."""
    raise NotImplementedError("TODO")
