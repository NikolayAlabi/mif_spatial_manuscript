#!/bin/bash
#SBATCH --job-name=s2a_compress_z
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=upgrade
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs/s2a_compress_z_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs/s2a_compress_z_%j.err

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a_rank_compress_candidates_v1.py}"
CONFIG="${CONFIG:-$MODULE_DIR/configs/stage2a_rank_compress_candidates_zscore.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"

mkdir -p /projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$MODULE_DIR"

python "$SCRIPT" --config "$CONFIG" "$@"
