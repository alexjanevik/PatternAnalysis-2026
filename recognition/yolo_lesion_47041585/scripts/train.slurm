#!/bin/bash
#SBATCH --job-name=yolo-lesion
#SBATCH --partition=a100 # TODO: confirm the COMP3710 partition/account on Rangpur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00 # TODO: set from the measured pilot projection
#SBATCH --output=runs/slurm/%x-%j.out

# Usage: sbatch scripts/train.slurm configs/b0.yaml
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/train.slurm <condition config>}"

cd "$(dirname "$0")/.."
mkdir -p runs/slurm
nvidia-smi
uv run --frozen train --config configs/base.yaml --config "$CONFIG"
