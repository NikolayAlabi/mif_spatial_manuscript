#!/bin/bash
# Submit Stage 2A steps 1-3 as:
#   1) one inventory job;
#   2) one Slurm-array task per context, using one CPU each;
#   3) one dependent aggregation job.
#
# Usage:
#   CONFIG_JSON=/path/to/stage2a_steps1_3_context_review.json \
#   bash submit_stage2a_steps1_3_context_review_v3.sh

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a_steps1_3_context_review_v3.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a_steps1_3_context_review.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
PARTITION="${PARTITION:-upgrade}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs/stage2a_steps1_3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

mkdir -p "$LOGDIR"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Missing script: $SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "[ERROR] Missing config: $CONFIG_JSON" >&2
  exit 1
fi

OUTPUT_ROOT=$(python - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["output_root"])
PY
)
mkdir -p "$OUTPUT_ROOT"

INVENTORY_SLURM="$LOGDIR/stage2a_steps1_3_inventory.slurm.sh"
cat > "$INVENTORY_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13_inventory
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=$PARTITION
#SBATCH --output=$LOGDIR/s2a13_inventory_%j.out
#SBATCH --error=$LOGDIR/s2a13_inventory_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "$SCRIPT" inventory --config "$CONFIG_JSON"
EOF
chmod +x "$INVENTORY_SLURM"

# Wait because the context count is created by the inventory step.
echo "[SUBMIT] Inventory"
sbatch --wait "$INVENTORY_SLURM"

CONTEXT_INDEX="$OUTPUT_ROOT/stage2a_context_index.csv"
if [[ ! -s "$CONTEXT_INDEX" ]]; then
  echo "[ERROR] Missing or empty context index: $CONTEXT_INDEX" >&2
  exit 1
fi

N_CONTEXTS=$(python - <<PY
import pandas as pd
x = pd.read_csv("$CONTEXT_INDEX")
print(len(x))
PY
)
if [[ "$N_CONTEXTS" -lt 1 ]]; then
  echo "[ERROR] No contexts found" >&2
  exit 1
fi
ARRAY_MAX=$((N_CONTEXTS - 1))

WORKER_SLURM="$LOGDIR/stage2a_steps1_3_worker.slurm.sh"
cat > "$WORKER_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13_ctx
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=$PARTITION
#SBATCH --output=$LOGDIR/s2a13_ctx_%A_%a.out
#SBATCH --error=$LOGDIR/s2a13_ctx_%A_%a.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "$SCRIPT" worker --config "$CONFIG_JSON" --array-id "\$SLURM_ARRAY_TASK_ID"
EOF
chmod +x "$WORKER_SLURM"

WORKER_JOB=$(sbatch --parsable --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" "$WORKER_SLURM")
echo "[SUBMIT] Context array job=$WORKER_JOB contexts=$N_CONTEXTS"

AGG_SLURM="$LOGDIR/stage2a_steps1_3_aggregate.slurm.sh"
cat > "$AGG_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a13_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=$PARTITION
#SBATCH --output=$LOGDIR/s2a13_aggregate_%j.out
#SBATCH --error=$LOGDIR/s2a13_aggregate_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python "$SCRIPT" aggregate --config "$CONFIG_JSON"
EOF
chmod +x "$AGG_SLURM"

AGG_JOB=$(sbatch --parsable --dependency="afterok:$WORKER_JOB" "$AGG_SLURM")
echo "[SUBMIT] Aggregate job=$AGG_JOB (afterok:$WORKER_JOB)"
echo "[OUTPUT] $OUTPUT_ROOT"
