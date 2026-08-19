#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
PY_ENV=${PY_ENV:-cuda6}

set +u
source "$CONDA_SH"
conda activate "$PY_ENV"
set -u

echo "[INFO] host=$(hostname)"
echo "[INFO] date=$(date)"
echo "[INFO] PY_ENV=$PY_ENV"
echo "[INFO] python=$(which python)"
python --version

echo "[RUN] python -u $*"
python -u "$@"

echo "[DONE] date=$(date)"
