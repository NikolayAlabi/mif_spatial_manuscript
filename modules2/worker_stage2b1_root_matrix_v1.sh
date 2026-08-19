#!/bin/bash
set -euo pipefail

CONFIG_JSON="${1:?Usage: worker_stage2b1_root_matrix_v1.sh CONFIG_JSON}"
SCRIPT="${SCRIPT:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2/stage2b1_root_consensus_v1.py}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"

export PS1="${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python "$SCRIPT" matrix-worker --config "$CONFIG_JSON"
