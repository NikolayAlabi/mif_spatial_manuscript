#!/bin/bash
# Stage 2A steps 1-3 root-aware v1.
#
# Explicit sequencing is intentional:
#   MODE=inventory  -> submit lightweight inventory only
#   MODE=workers    -> submit one worker per context after inventory finishes
#   MODE=aggregate  -> submit aggregation only after all workers finish
#
# Examples:
#   MODE=inventory bash submit_stage2a_steps1_3_rootaware_v1.sh
#   MODE=workers   bash submit_stage2a_steps1_3_rootaware_v1.sh
#   MODE=aggregate bash submit_stage2a_steps1_3_rootaware_v1.sh
#
# No #SBATCH --partition directive is written.

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a_steps1_3_rootaware_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a_steps1_3_rootaware_v1.json}"
MODE="${MODE:-inventory}"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"

LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/logs/stage2a_steps1_3_rootaware_v1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"

INVENTORY_MEM="${INVENTORY_MEM:-8G}"
INVENTORY_TIME="${INVENTORY_TIME:-01:00:00}"

WORKER_MEM="${WORKER_MEM:-32G}"
WORKER_TIME="${WORKER_TIME:-08:00:00}"

AGG_MEM="${AGG_MEM:-16G}"
AGG_TIME="${AGG_TIME:-02:00:00}"

mkdir -p "$LOGDIR"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] Missing config: $CONFIG_JSON" >&2; exit 1; }

# Read output_root without requiring pandas on the login node.
OUTPUT_ROOT="$(
python - "$CONFIG_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f)["output_root"])
PY
)"
mkdir -p "$OUTPUT_ROOT"

case "$MODE" in

  inventory)
    SLURM_SCRIPT="$LOGDIR/inventory.slurm.sh"

    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13root_inv
#SBATCH --cpus-per-task=1
#SBATCH --mem=$INVENTORY_MEM
#SBATCH --time=$INVENTORY_TIME
#SBATCH --output=$LOGDIR/inventory_%j.out
#SBATCH --error=$LOGDIR/inventory_%j.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SCRIPT" inventory --config "$CONFIG_JSON"
EOF

    chmod +x "$SLURM_SCRIPT"
    JOB=$(sbatch --parsable "$SLURM_SCRIPT")
    echo "[SUBMIT] inventory job=$JOB"
    echo "[NEXT] After it finishes:"
    echo "       MODE=workers bash $0"
    ;;

  workers)
    CONTEXT_INDEX="$OUTPUT_ROOT/stage2a_context_index.csv"

    [[ -s "$CONTEXT_INDEX" ]] || {
      echo "[ERROR] Missing context index: $CONTEXT_INDEX" >&2
      echo "Run MODE=inventory first and wait for it to finish." >&2
      exit 1
    }

    # Header + N data rows. Avoid pandas on login node.
    N_CONTEXTS=$(awk 'END {print NR - 1}' "$CONTEXT_INDEX")

    [[ "$N_CONTEXTS" -gt 0 ]] || {
      echo "[ERROR] No contexts found in $CONTEXT_INDEX" >&2
      exit 1
    }

    ARRAY_MAX=$((N_CONTEXTS - 1))
    SLURM_SCRIPT="$LOGDIR/workers.slurm.sh"

    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13root
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

python "$SCRIPT" worker \
  --config "$CONFIG_JSON" \
  --array-id "\${SLURM_ARRAY_TASK_ID}"
EOF

    chmod +x "$SLURM_SCRIPT"

    JOB=$(sbatch --parsable \
      --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
      "$SLURM_SCRIPT")

    echo "[SUBMIT] worker array job=$JOB contexts=$N_CONTEXTS"
    echo "[NEXT] After every worker finishes:"
    echo "       MODE=aggregate bash $0"
    ;;

  aggregate)
    SLURM_SCRIPT="$LOGDIR/aggregate.slurm.sh"

    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13root_agg
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

    chmod +x "$SLURM_SCRIPT"

    JOB=$(sbatch --parsable "$SLURM_SCRIPT")
    echo "[SUBMIT] aggregate job=$JOB"
    echo "[OUTPUT] $OUTPUT_ROOT"
    ;;

  *)
    echo "[ERROR] MODE must be inventory, workers, or aggregate; got: $MODE" >&2
    exit 2
    ;;
esac
