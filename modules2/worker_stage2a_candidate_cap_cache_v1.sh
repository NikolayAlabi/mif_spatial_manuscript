#!/bin/bash
#SBATCH --job-name=s2acap_cache
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
export PS1="${PS1-}"

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a_candidate_cap_sensitivity_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a_candidate_cap_sensitivity_v1.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SCRIPT" cache-worker \
  --config "$CONFIG_JSON" \
  --cache-id "$SLURM_ARRAY_TASK_ID"
