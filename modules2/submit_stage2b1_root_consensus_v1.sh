#!/bin/bash
# Root-aware Stage 2B-1 submission helper.
# Explicit modes; intentionally no dependency chaining.
#
# Usage:
#   MODE=validate  bash submit_stage2b1_root_consensus_v1.sh
#   MODE=setup     bash submit_stage2b1_root_consensus_v1.sh
#   MODE=matrices  bash submit_stage2b1_root_consensus_v1.sh
#   MODE=consensus bash submit_stage2b1_root_consensus_v1.sh
#   MODE=aggregate bash submit_stage2b1_root_consensus_v1.sh

set -euo pipefail

MODE="${MODE:-}"
if [[ -z "$MODE" ]]; then
  echo "[ERROR] Set MODE=validate|setup|matrices|consensus|aggregate" >&2
  exit 1
fi

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2b1_root_consensus_v1.json}"
SCRIPT="$MODULE_DIR/stage2b1_root_consensus_v1.py"
MATRIX_WORKER="$MODULE_DIR/worker_stage2b1_root_matrix_v1.sh"
CONSENSUS_WORKER="$MODULE_DIR/worker_stage2b1_root_consensus_v1.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2b1_root_consensus_v1}"
LOGDIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOGDIR"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"

for f in "$CONFIG_JSON" "$SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing: $f" >&2
    exit 1
  fi
done

submit_single () {
  local command="$1"
  local jobname="$2"
  local mem="$3"
  local time="$4"
  local slurm="$LOGDIR/${jobname}.slurm.sh"

  cat > "$slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --cpus-per-task=1
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH --output=$LOGDIR/${jobname}_%j.out
#SBATCH --error=$LOGDIR/${jobname}_%j.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
python "$SCRIPT" $command --config "$CONFIG_JSON"
EOF
  chmod +x "$slurm"
  sbatch "$slurm"
}

case "$MODE" in
  validate)
    submit_single "validate" "s2b1_valid" "8G" "00:30:00"
    ;;

  setup)
    submit_single "setup" "s2b1_setup" "24G" "02:00:00"
    ;;

  matrices)
    INDEX="$OUTPUT_ROOT/stage2b1_matrix_worker_index.csv"
    if [[ ! -f "$INDEX" ]]; then
      echo "[ERROR] Missing matrix index: $INDEX. Run MODE=setup first." >&2
      exit 1
    fi
    if [[ ! -f "$MATRIX_WORKER" ]]; then
      echo "[ERROR] Missing worker: $MATRIX_WORKER" >&2
      exit 1
    fi
    N=$(awk 'END {print NR-1}' "$INDEX")
    if [[ "$N" -lt 1 ]]; then
      echo "[ERROR] No matrix workers in $INDEX" >&2
      exit 1
    fi
    MEM="${MATRIX_MEM:-32G}"
    TIME="${MATRIX_TIME:-08:00:00}"
    MAX_PARALLEL="${MATRIX_MAX_PARALLEL:-8}"
    sbatch \
      --job-name=s2b1_matrix \
      --cpus-per-task=1 \
      --mem="$MEM" \
      --time="$TIME" \
      --array="1-${N}%${MAX_PARALLEL}" \
      --output="$LOGDIR/matrix_%A_%a.out" \
      --error="$LOGDIR/matrix_%A_%a.err" \
      "$MATRIX_WORKER" "$CONFIG_JSON"
    ;;

  consensus)
    INDEX="$OUTPUT_ROOT/stage2b1_consensus_worker_index.csv"
    MATRIX_INDEX="$OUTPUT_ROOT/stage2b1_matrix_worker_index.csv"
    if [[ ! -f "$INDEX" || ! -f "$MATRIX_INDEX" ]]; then
      echo "[ERROR] Missing setup indices. Run MODE=setup first." >&2
      exit 1
    fi
    EXPECTED=$(awk 'END {print NR-1}' "$MATRIX_INDEX")
    COMPLETED=$(find "$OUTPUT_ROOT/cohort_root_matrices" -name .done 2>/dev/null | wc -l)
    echo "[CHECK] matrix workers expected=$EXPECTED completed=$COMPLETED"
    if [[ "$EXPECTED" -ne "$COMPLETED" ]]; then
      echo "[ERROR] Matrix workers are incomplete. Do not start consensus yet." >&2
      exit 1
    fi
    if [[ ! -f "$CONSENSUS_WORKER" ]]; then
      echo "[ERROR] Missing worker: $CONSENSUS_WORKER" >&2
      exit 1
    fi
    N=$(awk 'END {print NR-1}' "$INDEX")
    MEM="${CONSENSUS_MEM:-24G}"
    TIME="${CONSENSUS_TIME:-04:00:00}"
    MAX_PARALLEL="${CONSENSUS_MAX_PARALLEL:-7}"
    sbatch \
      --job-name=s2b1_cons \
      --cpus-per-task=1 \
      --mem="$MEM" \
      --time="$TIME" \
      --array="1-${N}%${MAX_PARALLEL}" \
      --output="$LOGDIR/consensus_%A_%a.out" \
      --error="$LOGDIR/consensus_%A_%a.err" \
      "$CONSENSUS_WORKER" "$CONFIG_JSON"
    ;;

  aggregate)
    CINDEX="$OUTPUT_ROOT/stage2b1_consensus_worker_index.csv"
    if [[ ! -f "$CINDEX" ]]; then
      echo "[ERROR] Missing consensus index. Run MODE=setup first." >&2
      exit 1
    fi
    EXPECTED=$(awk 'END {print NR-1}' "$CINDEX")
    COMPLETED=$(find "$OUTPUT_ROOT/root_consensus" -name .done 2>/dev/null | wc -l)
    echo "[CHECK] consensus workers expected=$EXPECTED completed=$COMPLETED"
    if [[ "$EXPECTED" -ne "$COMPLETED" ]]; then
      echo "[ERROR] Consensus workers are incomplete. Do not aggregate yet." >&2
      exit 1
    fi
    submit_single "aggregate" "s2b1_agg" "16G" "01:00:00"
    ;;

  *)
    echo "[ERROR] Unknown MODE=$MODE" >&2
    exit 1
    ;;
esac
