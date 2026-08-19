#!/bin/bash
set -euo pipefail

MODULE_DIR=${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}
SCRIPT=${SCRIPT:-$MODULE_DIR/stage2a5_cap_rho_grid_v1.py}
CONFIG_JSON=${CONFIG_JSON:-$MODULE_DIR/configs/stage2a5_cap_rho_grid_config_v1.json}
CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuda6}
PARTITION=${PARTITION:-upgrade}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
WORKER_MEM=${WORKER_MEM:-16G}
WORKER_TIME=${WORKER_TIME:-16:00:00}
AGG_MEM=${AGG_MEM:-32G}
AGG_TIME=${AGG_TIME:-06:00:00}
SKIP_INVENTORY=${SKIP_INVENTORY:-0}
FORCE_WORKERS=${FORCE_WORKERS:-0}

[[ -s "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 1; }
[[ -s "$CONFIG_JSON" ]] || { echo "[ERROR] Missing config: $CONFIG_JSON" >&2; exit 1; }

# Standard-library Python only; no pandas/Conda needed on the login node.
OUTPUT_ROOT=$(python3 - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["output_root"])
PY
)

SLURM_DIR="$OUTPUT_ROOT/slurm"
LOGDIR="$OUTPUT_ROOT/logs"
mkdir -p "$SLURM_DIR" "$LOGDIR"

if [[ "$SKIP_INVENTORY" != "1" ]]; then
  INVENTORY_SCRIPT="$SLURM_DIR/stage2a5_grid_inventory.slurm.sh"
  cat > "$INVENTORY_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5grid_inv
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=$LOGDIR/inventory_%j.out
#SBATCH --error=$LOGDIR/inventory_%j.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" inventory --config "$CONFIG_JSON"
EOF
  chmod +x "$INVENTORY_SCRIPT"
  echo "[SUBMIT] inventory"
  sbatch --wait "$INVENTORY_SCRIPT"
else
  echo "[SKIP] inventory; reusing existing context index"
fi

CONTEXT_INDEX="$OUTPUT_ROOT/stage2a5_grid_context_index.csv"
[[ -s "$CONTEXT_INDEX" ]] || { echo "[ERROR] Missing $CONTEXT_INDEX" >&2; exit 1; }
N_CONTEXTS=$(awk 'END {print NR - 1}' "$CONTEXT_INDEX")
[[ "$N_CONTEXTS" -gt 0 ]] || { echo "[ERROR] No contexts in $CONTEXT_INDEX" >&2; exit 1; }
LAST_ARRAY_ID=$((N_CONTEXTS - 1))

WORKER_SCRIPT="$SLURM_DIR/stage2a5_grid_worker.slurm.sh"
FORCE_ARG=""
if [[ "$FORCE_WORKERS" == "1" ]]; then
  FORCE_ARG="--force"
fi
cat > "$WORKER_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5grid
#SBATCH --cpus-per-task=1
#SBATCH --mem=$WORKER_MEM
#SBATCH --time=$WORKER_TIME
#SBATCH --array=0-${LAST_ARRAY_ID}%${MAX_CONCURRENT}
#SBATCH --output=$LOGDIR/worker_%A_%a.out
#SBATCH --error=$LOGDIR/worker_%A_%a.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" worker --config "$CONFIG_JSON" --array-id "\${SLURM_ARRAY_TASK_ID}" $FORCE_ARG
EOF
chmod +x "$WORKER_SCRIPT"

ARRAY_JOB_ID=$(sbatch --parsable "$WORKER_SCRIPT")
echo "[SUBMIT] worker array job: $ARRAY_JOB_ID | contexts=$N_CONTEXTS | max_concurrent=$MAX_CONCURRENT"

AGG_SCRIPT="$SLURM_DIR/stage2a5_grid_aggregate.slurm.sh"
cat > "$AGG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5grid_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=$AGG_MEM
#SBATCH --time=$AGG_TIME
#SBATCH --dependency=afterok:$ARRAY_JOB_ID
#SBATCH --output=$LOGDIR/aggregate_%j.out
#SBATCH --error=$LOGDIR/aggregate_%j.err

set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" aggregate --config "$CONFIG_JSON"
EOF
chmod +x "$AGG_SCRIPT"
AGG_JOB_ID=$(sbatch --parsable "$AGG_SCRIPT")
echo "[SUBMIT] aggregate job: $AGG_JOB_ID (afterok:$ARRAY_JOB_ID)"
echo "[OUTPUT] $OUTPUT_ROOT"
