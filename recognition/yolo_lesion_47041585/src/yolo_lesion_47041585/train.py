"""
Training entry point: resolve configs, run one experiment, record its provenance.

Normal runs train and validate only. Test evaluation lives behind the guarded path in
helpers/evaluation.py to be used once the selection manifest is frozen.

Usage:
    uv run train --config configs/base.yaml --config configs/b0.yaml [--pilot]
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

SAMPLING_SCHEMES = ("stock", "replacement")


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = _merge(
                dict(cast(Mapping[str, Any], current)), cast(Mapping[str, Any], value)
            )
        else:
            merged[key] = value
    return merged


def resolve_training_config(
    config_paths: Sequence[Path], pilot: bool = False
) -> dict[str, Any]:
    """Merge YAML configs left to right into one fully resolved config.

    Args:
        config_paths (Sequence[Path]): Base config first, then the condition override.
        pilot (bool): Shorten the run for the smoke test and label it as a pilot.

    Returns:
        dict[str, Any]: The resolved config, to be saved with the run.
    """
    config: dict[str, Any] = {}
    for path in config_paths:
        config = _merge(config, yaml.safe_load(path.read_text()) or {})
    sampling = config["sampling"]
    if sampling["scheme"] not in SAMPLING_SCHEMES:
        raise ValueError(f"Unknown sampling scheme: {sampling['scheme']!r}")
    if sampling["scheme"] == "stock" and sampling["malignant_fraction"] is not None:
        raise ValueError("The stock loader cannot sample to a malignant fraction.")
    if pilot:
        config["pilot"] = True
        config["train"]["epochs"] = 2
    return config


def collect_run_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance for a run: source revision, split hash, package/CUDA/GPU versions,
    starting-checkpoint hash and seed."""
    raise NotImplementedError("TODO")


def run_experiment(config: Mapping[str, Any]) -> Path:
    """Train and validate one condition and return its run directory.

    Should build the detector, attach the sampler to LesionDetectionTrainer, train with
    best-checkpoint selection on validation mAP50:95 (earliest epoch on ties), and save
    the resolved config, metadata and history under runs/.
    """
    raise NotImplementedError("TODO")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--pilot", action="store_true", help="short smoke-test run")
    args = parser.parse_args()
    run_experiment(resolve_training_config(args.config, args.pilot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
