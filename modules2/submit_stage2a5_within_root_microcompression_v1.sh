#!/bin/bash
# Finalized root-aware Stage 2A-5.
# Explicit modes; no automatic dependency chains.
#
# Usage:
#   MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 MODE=validate  bash submit_stage2a5_within_root_microcompression_v1.sh
#   MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 MODE=inventory bash submit_stage2a5_within_root_microcompression_v1.sh
#   MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 MODE=workers   bash submit_stage2a5_within_root_microcompression_v1.sh
#   MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 MODE=aggregate bash submit_stage2a5_within_root_microcompression_v1.sh

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a5_within_root_microcompression_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a5_within_root_microcompression_v1.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
MODE="${MODE:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"

VALIDATE_MEM="${VALIDATE_MEM:-4G}"
VALIDATE_TIME="${VALIDATE_TIME:-00:30:00}"
INVENTORY_MEM="${INVENTORY_MEM:-4G}"
INVENTORY_TIME="${INVENTORY_TIME:-00:30:00}"
WORKER_MEM="${WORKER_MEM:-8G}"
WORKER_TIME="${WORKER_TIME:-02:00:00}"
AGG_MEM="${AGG_MEM:-8G}"
AGG_TIME="${AGG_TIME:-01:00:00}"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] Missing config: $CONFIG_JSON" >&2; exit 1; }
[[ -n "$MODE" ]] || { echo "[ERROR] Set MODE=validate|inventory|workers|aggregate" >&2; exit 1; }

OUTPUT_ROOT=$(python - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["output_root"])
PY
)
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs}"
mkdir -p "$OUTPUT_ROOT" "$LOGDIR"

write_common() {
  cat <<EOF
set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
EOF
}

case "$MODE" in
  validate)
    SLURM="$LOGDIR/stage2a5_validate.slurm.sh"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --job-name=s2a5_val'
      echo '#SBATCH --cpus-per-task=1'
      echo "#SBATCH --mem=$VALIDATE_MEM"
      echo "#SBATCH --time=$VALIDATE_TIME"
      echo "#SBATCH --output=$LOGDIR/validate_%j.out"
      echo "#SBATCH --error=$LOGDIR/validate_%j.err"
      write_common
      echo "python \"$SCRIPT\" validate --config \"$CONFIG_JSON\""
    } > "$SLURM"
    chmod +x "$SLURM"
    sbatch "$SLURM"
    ;;

  inventory)
    SLURM="$LOGDIR/stage2a5_inventory.slurm.sh"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --job-name=s2a5_inv'
      echo '#SBATCH --cpus-per-task=1'
      echo "#SBATCH --mem=$INVENTORY_MEM"
      echo "#SBATCH --time=$INVENTORY_TIME"
      echo "#SBATCH --output=$LOGDIR/inventory_%j.out"
      echo "#SBATCH --error=$LOGDIR/inventory_%j.err"
      write_common
      echo "python \"$SCRIPT\" inventory --config \"$CONFIG_JSON\""
    } > "$SLURM"
    chmod +x "$SLURM"
    sbatch "$SLURM"
    ;;

  workers)
    INDEX="$OUTPUT_ROOT/stage2a5_context_index.csv"
    [[ -s "$INDEX" ]] || { echo "[ERROR] Missing $INDEX. Run MODE=inventory and wait for it to finish." >&2; exit 1; }
    N_CONTEXTS=$(awk 'END {print NR - 1}' "$INDEX")
    [[ "$N_CONTEXTS" -gt 0 ]] || { echo "[ERROR] No contexts in $INDEX" >&2; exit 1; }
    ARRAY_MAX=$((N_CONTEXTS - 1))

    SLURM="$LOGDIR/stage2a5_workers.slurm.sh"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --job-name=s2a5_root'
      echo '#SBATCH --cpus-per-task=1'
      echo "#SBATCH --mem=$WORKER_MEM"
      echo "#SBATCH --time=$WORKER_TIME"
      echo "#SBATCH --output=$LOGDIR/worker_%A_%a.out"
      echo "#SBATCH --error=$LOGDIR/worker_%A_%a.err"
      write_common
      echo "python \"$SCRIPT\" worker --config \"$CONFIG_JSON\" --array-id \"\$SLURM_ARRAY_TASK_ID\""
    } > "$SLURM"
    chmod +x "$SLURM"
    sbatch --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" "$SLURM"
    echo "[SUBMIT] contexts=$N_CONTEXTS one_CPU_per_context"
    ;;

  aggregate)
    SLURM="$LOGDIR/stage2a5_aggregate.slurm.sh"
    {
      echo '#!/bin/bash'
      echo '#SBATCH --job-name=s2a5_agg'
      echo '#SBATCH --cpus-per-task=1'
      echo "#SBATCH --mem=$AGG_MEM"
      echo "#SBATCH --time=$AGG_TIME"
      echo "#SBATCH --output=$LOGDIR/aggregate_%j.out"
      echo "#SBATCH --error=$LOGDIR/aggregate_%j.err"
      write_common
      echo "python \"$SCRIPT\" aggregate --config \"$CONFIG_JSON\""
    } > "$SLURM"
    chmod +x "$SLURM"
    sbatch "$SLURM"
    ;;

  *)
    echo "[ERROR] Unknown MODE=$MODE; use validate|inventory|workers|aggregate" >&2
    exit 1
    ;;
esac

echo "[OUTPUT] $OUTPUT_ROOT"
