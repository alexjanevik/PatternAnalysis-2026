#!/bin/bash
#SBATCH --job-name=yolo-lesion
#SBATCH --partition=a100 # TODO: confirm the COMP3710 partition/account on Rangpur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00 # TODO: set from the measured pilot projection
#SBATCH --output=runs/slurm/%x-%j.out

# Usage: mkdir -p runs/slurm && sbatch scripts/train.slurm configs/o35.yaml [--pilot]
# The runs/slurm folder must exist first, since Slurm opens the output file before this runs.
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/train.slurm <condition config> [--pilot]}"

cd "$(dirname "$0")/.."
mkdir -p runs/slurm
nvidia-smi
uv run --frozen train --config configs/base.yaml --config "$CONFIG" "${@:2}"
