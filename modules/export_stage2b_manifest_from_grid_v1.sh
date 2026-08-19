#!/bin/bash
set -euo pipefail

MODULE_DIR=${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}
SCRIPT=${SCRIPT:-$MODULE_DIR/stage2a5_cap_rho_grid_v1.py}
CONFIG_JSON=${CONFIG_JSON:-$MODULE_DIR/configs/stage2a5_cap_rho_grid_config_v1.json}
CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuda6}

CAP=${1:?Usage: $0 <candidate_cap> <semantic_rho> [S1|S2|S2E] [output_dir]}
RHO=${2:?Usage: $0 <candidate_cap> <semantic_rho> [S1|S2|S2E] [output_dir]}
SUPPORT_SET=${3:-S2E}
OUTPUT_DIR=${4:-}

export PS1="${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

ARGS=(
  export
  --config "$CONFIG_JSON"
  --candidate-cap "$CAP"
  --semantic-rho "$RHO"
  --support-set "$SUPPORT_SET"
)
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

python "$SCRIPT" "${ARGS[@]}"
