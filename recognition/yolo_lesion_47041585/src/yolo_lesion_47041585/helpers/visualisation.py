"""
Figures and tables for the report. Each output should name the run and split it came from.
Only box overlays are needed for the pilot run; the rest are placeholders.
"""

from typing import Any


def render_case_comparison(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO: overlays of native/transformed/predicted boxes")


def plot_training_history(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def plot_tradeoff(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")


def summarise_results(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("TODO")
