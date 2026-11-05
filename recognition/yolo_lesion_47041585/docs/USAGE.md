# Usage Guide

<div align="center">

[README](../README.md) • [Installation](./INSTALLATION.md)

</div>

---

This page covers how to train and evaluate the YOLO11 model after installing the project. All commands are to be run from the project folder (`yolo_lesion_47041585/`)

## Run the data audit

`audit.py` acts as an interface for `dataset.py`. It scours the dataset for invalid records. This includes missing or invalid ISIC identifiers, patient identifiers, diagnosis labels, images and masks. The remaining data is used to build independence groups for maintaining patient-independence between sets before splitting the dataset into training, validation and test sets (**70/15/15**). It is also responsible for deriving bounding boxes from the training masks and exporting the dataset in a format usable for YOLO training.

To run the audit, use:

```bash
uv run python -m yolo_lesion_47041585.audit [--source dataset] [--output artifacts/isic2018]
```

By default, data is sourced from `dataset/` and written to `artifacts/isic2018`. The output folder must be new or empty. The output folder contains:

| File | Contents |
|:-----|:---------|
| `yolo/` | `dataset.yaml`, `images/` and `labels/` directories for `train`, `val` and `test` splits |
| `split_manifest.csv` | Split, patient, class and box for every retained image |
| `cohort.csv`, `exclusions.csv` | Retained images and excluded images + exclusion reasons |
| `summary.json` | Counts and hashes of the raw metadata and split |

>[!IMPORTANT]
> Images in `yolo/` are symlinks to the absolute image paths. **Run this on the machine you will train on**. Do not copy `artifacts/` between machines, it will not work.

## Training the YOLO11 model

Each training run accepts a config file and a set of flags. The config parameter is a `yaml` file containing a set of training conditions, found in `configs/`.

```bash
uv run train {--config PATH_TO_CONFIG.yaml} [--flags]

# or if using the slurm script:

sbatch scripts/train.slurm {--config PATH_TO_CONFIG.yaml} [--flags]
```

The available flags are:

| Flag | Description |
|:-----|:------------|
| `--config configs/{config}.yaml` | **REQUIRED**. Specifies the config file to use for training |
| `--seed N` | Sets the random seed for reproducibility. Default = `47041585` |
| `--pilot` | A 3-epoch smoke test. Is labelled `...-pilot/` in `runs/` |
| `--device D` | The Ultralytics device to use in training (`0`, `cpu`, `mps`). Default = `0` (first GPU) |

A list of tested configs have been supplied under `configs/`, with their differences described in the [readme](../README.md#benchmarks-and-results).

> [!NOTE]
> `b0` differs from `a0` in both augmentation and sampling. `b0` vs `a0` measures these two together. Single-change comparisons are made against `a0` and are thus more informative.

Configs can be easily modified or added in the `configs/` directory. For a list of all modifyable config parameters, see the [Ultralytics documentation](https://docs.ultralytics.com/usage/cfg).
