"""
Inference driver: images in, boxes and routes out. Needs no masks or labels.

Usage:
    uv run predict --checkpoint runs/<run>/best.pt --images <dir> --output <dir>
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ultralytics.engine.results import Results


@dataclass(frozen=True)
class Detection:
    """A predicted box in the original image's frame.

    Attributes:
        image_id (str): Image identifier (file stem).
        x_center (float): Normalised box centre x.
        y_center (float): Normalised box centre y.
        width (float): Normalised box width.
        height (float): Normalised box height.
        class_id (int): 0 Benign, 1 Malignant.
        confidence (float): Class confidence.
    """

    image_id: str
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    confidence: float


def detections_from_result(result: Results) -> list[Detection]:
    """Convert one Ultralytics `Results` into Detections in the original image's frame.

    Ultralytics has already undone the resize and padding, and `boxes.xywhn` is
    normalised by the original image size, so no further geometry is needed here.

    Args:
        result (ultralytics.engine.results.Results): Prediction for a single image.

    Returns:
        list[Detection]: One entry per predicted box, with the image ID taken from the
            file name.
    """
    image_id = Path(result.path).stem
    boxes = result.boxes
    return [
        Detection(image_id, x, y, w, h, int(class_id), float(conf))
        for (x, y, w, h), class_id, conf in zip(
            boxes.xywhn.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        )
    ]


def load_run(checkpoint: Path):
    """Load a trained model plus its saved inference settings (conf, NMS, max_det)."""
    raise NotImplementedError("TODO")


def predict_cases(model, images: Sequence[Path]) -> list[Detection]:
    """Detect lesions, undoing resize/padding so boxes are normalised by the original
    image size. This is the single inference path shared by validation, test and demo."""
    raise NotImplementedError("TODO")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise SystemExit(main())
