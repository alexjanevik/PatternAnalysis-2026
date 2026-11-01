"""
Figures and tables for the report. Each output should name the run and split it came from.
Only box overlays are needed for the pilot run; the rest are placeholders.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from yolo_lesion_47041585.predict import Detection

# one colour per class ID: benign green, malignant red
CLASS_COLOURS = ("#2ca02c", "#d62728")


def draw_detections(
    image: Image.Image, detections: Sequence["Detection"], class_names: Sequence[str]
) -> Image.Image:
    """Draw predicted boxes, labelled with class and confidence, on a copy of the image.

    Args:
        image (Image.Image): The original image the detections were made on.
        detections (Sequence[Detection]): Boxes normalised by that image's size.
        class_names (Sequence[str]): The class names in class ID order.

    Returns:
        Image.Image: A new RGB image with the boxes drawn.
    """
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    thickness = max(2, round(max(width, height) / 400))

    for detection in detections:
        box = detection.box
        left = (box.x_center - box.width / 2) * width
        right = (box.x_center + box.width / 2) * width
        top = (box.y_center - box.height / 2) * height
        bottom = (box.y_center + box.height / 2) * height

        colour = CLASS_COLOURS[detection.class_id % len(CLASS_COLOURS)]
        draw.rectangle((left, top, right, bottom), outline=colour, width=thickness)
        draw.text(
            (left + thickness, top + thickness),
            f"{class_names[detection.class_id]} {detection.confidence:.2f}",
            fill=colour,
        )
    return canvas


def render_case_comparison(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: overlays of native/transformed/predicted boxes")


def plot_training_history(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def plot_tradeoff(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def summarise_results(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")
