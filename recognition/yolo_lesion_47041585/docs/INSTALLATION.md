# Installation Guide

<div align="center">

[README](../README.md) • [Usage](./USAGE.md)

</div>

---

This page covers the project's dependencies and how to install the project itself. This guide covers installation with `uv` for dependency management, but steps are also provided for using `pip` and `venv` directly.

## Dependencies

Training was performed on the UQ Rangpur cluster (Alma Linux 8.10) using an NVIDIA A100 GPU (40GB VRAM) with CUDA 13.1.
> [!NOTE]
> Pilot tests were also performed using MPS on an M1 MacBook Air, though these results are not reported in the README due to the huge discrepancy in performance. Just keep in mind that it is possible to run this project using MPS, though not recommended. Windows compatibility has not been tested, and thus is not explicitly supported.

| Dependency    | Version   | Use-case                                                      |
|---------------|----------:|---------------------------------------------------------------|
| Python        | 3.12      | Runtime environment                                           |
| ultralytics   | 8.4.182   | YOLO11 detector, trainer, validator                           |
| torch         | 2.14.1    | Tensor computation, GPU use                                   |
| torchvision   | 0.29.1    | Installed with ultralytics and torch (not imported directly)  |
| numpy         | 2.5.4     | Sampling, box geometry                                        |
| pillow        | 13.0.0    | Image reading and overlays                                    |
| pyyaml        | 6.0.3     | Configuration files                                           |
| rich          | 15.0.0    | Console output formatting                                     |

Dependencies for these packages are all automatically installed when using a virtual environment either in `pip` with `requirements.txt` or `uv` (or equivalent) with `pyproject.toml`. The `uv.lock` file is included to ensure that the same versions of dependencies are installed across different environments.

## Building the project

Start by cloning the repository and moving into the project folder:

```bash
git clone https://github.com/alexjanevik/PatternAnalysis-2026.git
cd PatternAnalysis-2026/recognition/yolo_lesion_47041585
```

### Installation with `uv`

If you have `uv` installed, you can create a virtual environment and install the dependencies with:

```bash
uv venv
uv sync --frozen # --frozen ensures exact versions from uv.lock are installed 
```

This will create a virtual environment in `.venv/` and install the dependencies listed in `pyproject.toml` at the pinned versions listed in `uv.lock`.

With this, you can run project modules with `uv`, for example:

```bash
uv run -m yolo_lesion_47041585.audit --help
```

### Installation with `pip` and `venv`

Alongside the `pyproject.toml` and `uv.lock` files, a `requirements.txt` file is included to allow installation of the same dependencies using `pip` and `venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer this method, note that all commands in [Usage](./USAGE.md) will need to be run as:

```bash
python3 -m yolo_lesion_47041585.{module}
# e.g.
python3 -m yolo_lesion_47041585.audit --help
```

## Retrieve the data

Before beginning any training, you will need to ensure the project has access to the ISIC2018 dataset. This differs depending on whether or not you are running the project on the UQ Rangpur cluster.

```bash
# if running on UQ Rangpur cluster, copy files from shared storage
# and only download a single archive from ISIC
./scripts/fetch_data.sh --rangpur

# otherwise, download all necessary files directly from ISIC
./scripts/fetch_data.sh
```

This will place the ISIC2018 dataset into `dataset/` alongside the provided `metadata_raw.csv`.

> [!NOTE]
> The training images and masks available on Rangpur are `_x2` versions. These are not available from the public ISIC archive, so the results described in this report are not entirely reproducible without upscaled images. Using the original images will likely provide better results due to the masks being truly binary.
