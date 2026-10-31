"""
Pure routing rules: detections in, clear / monitor / urgent review out.

No model or file access here, so the branches are easy to test.
"""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

from yolo_lesion_47041585.predict import Detection

Route = Literal["clear", "monitor", "urgent"]


@dataclass(frozen=True)
class PolicySpec:
    """Thresholds and rule version for one operating point.

    Attributes:
        conf_floor (float): Detections below this confidence are ignored.
        malignant_threshold (float): Malignant confidence at or above which to escalate.
        benign_threshold (float): Benign confidence at or above which to clear.
        version (str): Routing-rule version.
    """

    conf_floor: float
    malignant_threshold: float
    benign_threshold: float
    version: str = "v0"


def route_case(detections: Sequence[Detection], spec: PolicySpec) -> Route:
    """Absent, conflicting or low-confidence detections are routed to monitor."""
    raise NotImplementedError("TODO")


def search_policy(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def select_operating_point(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def select_candidate(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")
