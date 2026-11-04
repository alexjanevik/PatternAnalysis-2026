#!/bin/bash
#SBATCH --job-name=yolo-lesion
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Usage, from the project folder: sbatch scripts/train.slurm configs/o35.yaml [--pilot]
# Slurm starts the job in the folder sbatch was run from, so no cd is needed.
module load cuda/11.1
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/train.slurm <condition config> [--pilot]}"

[[ -f pyproject.toml ]] || { echo "Run sbatch from the project folder, not $PWD" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH" # where uv usually installs, and batch shells may not have it
nvidia-smi
uv run --frozen train --config "$CONFIG" "${@:2}"
