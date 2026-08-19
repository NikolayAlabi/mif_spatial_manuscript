#!/bin/bash
# Finalized root-aware Stage 2A-4 submitter.
#
# Explicit modes (run sequentially):
#   MODE=validate  bash submit_stage2a4_root_nomination_v1.sh
#   MODE=workers   bash submit_stage2a4_root_nomination_v1.sh
#   MODE=aggregate bash submit_stage2a4_root_nomination_v1.sh
#
# No cross-root rescue or compression is performed here.

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a4_root_nomination_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a4_root_nomination_v1.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
MODE="${MODE:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"
WORKER_MEM="${WORKER_MEM:-12G}"
WORKER_TIME="${WORKER_TIME:-03:00:00}"
AGG_MEM="${AGG_MEM:-8G}"
AGG_TIME="${AGG_TIME:-01:00:00}"

STAGE2_ROOT="${STAGE2_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1}"
LOGDIR="${LOGDIR:-$STAGE2_ROOT/logs/stage2a4_root_nomination_v1}"
mkdir -p "$LOGDIR"

if [[ -z "$MODE" ]]; then
  echo "[ERROR] Set MODE=validate, MODE=workers, or MODE=aggregate" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Missing script: $SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "[ERROR] Missing config: $CONFIG_JSON" >&2
  exit 1
fi

CAP_ROOT=$(python - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["cap_sensitivity_output_root"])
PY
)
CONTEXT_INDEX="$CAP_ROOT/context_index.csv"
if [[ ! -s "$CONTEXT_INDEX" ]]; then
  echo "[ERROR] Missing upstream context index: $CONTEXT_INDEX" >&2
  exit 1
fi

case "$MODE" in
  validate)
    export PS1="${PS1-}"
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    python "$SCRIPT" validate --config "$CONFIG_JSON"
    ;;

  workers)
    N_CONTEXTS=$(awk 'END {print NR - 1}' "$CONTEXT_INDEX")
    if [[ "$N_CONTEXTS" -lt 1 ]]; then
      echo "[ERROR] No contexts in $CONTEXT_INDEX" >&2
      exit 1
    fi
    ARRAY_MAX=$((N_CONTEXTS - 1))
    WORKER_SLURM="$LOGDIR/workers.slurm.sh"
    cat > "$WORKER_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a4_root
#SBATCH --cpus-per-task=1
#SBATCH --mem=$WORKER_MEM
#SBATCH --time=$WORKER_TIME
#SBATCH --output=$LOGDIR/worker_%A_%a.out
#SBATCH --error=$LOGDIR/worker_%A_%a.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "$SCRIPT" worker --config "$CONFIG_JSON" --array-id "\$SLURM_ARRAY_TASK_ID"
EOF
    chmod +x "$WORKER_SLURM"
    JOB=$(sbatch --parsable --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" "$WORKER_SLURM")
    echo "[SUBMIT] Stage2A4 root workers job=$JOB contexts=$N_CONTEXTS"
    ;;

  aggregate)
    AGG_SLURM="$LOGDIR/aggregate.slurm.sh"
    cat > "$AGG_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a4_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=$AGG_MEM
#SBATCH --time=$AGG_TIME
#SBATCH --output=$LOGDIR/aggregate_%j.out
#SBATCH --error=$LOGDIR/aggregate_%j.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "$SCRIPT" aggregate --config "$CONFIG_JSON"
EOF
    chmod +x "$AGG_SLURM"
    JOB=$(sbatch --parsable "$AGG_SLURM")
    echo "[SUBMIT] Stage2A4 aggregate job=$JOB"
    ;;

  *)
    echo "[ERROR] Unknown MODE=$MODE" >&2
    exit 1
    ;;
esac
