#!/bin/bash
# Stage 2A-5: build context index, run one microcompression worker per context
# with one CPU, then aggregate.
#
# Usage:
#   CONFIG_JSON=/path/to/stage2a5_config.json \
#   MAX_CONCURRENT=24 \
#   bash submit_stage2a5_interpretable_microcompression_v1.sh

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a5_interpretable_microcompression_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a5_interpretable_microcompression.json}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
PARTITION="${PARTITION:-upgrade}"
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"
WORKER_MEM="${WORKER_MEM:-16G}"
WORKER_TIME="${WORKER_TIME:-04:00:00}"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] Missing $CONFIG_JSON" >&2; exit 1; }

OUTPUT_ROOT=$(python - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["output_root"])
PY
)
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs}"
mkdir -p "$OUTPUT_ROOT" "$LOGDIR"

export PS1="${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" inventory --config "$CONFIG_JSON"

CONTEXT_INDEX="$OUTPUT_ROOT/stage2a5_context_index.csv"
[[ -s "$CONTEXT_INDEX" ]] || { echo "[ERROR] Missing $CONTEXT_INDEX" >&2; exit 1; }
N_CONTEXTS=$(python - <<PY
import pandas as pd
print(len(pd.read_csv("$CONTEXT_INDEX")))
PY
)
[[ "$N_CONTEXTS" -gt 0 ]] || { echo "[ERROR] No Stage 2A-5 contexts" >&2; exit 1; }
ARRAY_MAX=$((N_CONTEXTS - 1))

WORKER_SLURM="$LOGDIR/stage2a5_worker.slurm.sh"
cat > "$WORKER_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5_ctx
#SBATCH --cpus-per-task=1
#SBATCH --mem=$WORKER_MEM
#SBATCH --time=$WORKER_TIME
#SBATCH --output=$LOGDIR/s2a5_ctx_%A_%a.out
#SBATCH --error=$LOGDIR/s2a5_ctx_%A_%a.err
set -euo pipefail
export PS1="${PS1-}"
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
echo "[SUBMIT] Stage 2A-5 worker array=$WORKER_JOB contexts=$N_CONTEXTS one_CPU_per_context"

AGG_SLURM="$LOGDIR/stage2a5_aggregate.slurm.sh"
cat > "$AGG_SLURM" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=$LOGDIR/s2a5_agg_%j.out
#SBATCH --error=$LOGDIR/s2a5_agg_%j.err
set -euo pipefail
export PS1="${PS1-}"
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
echo "[SUBMIT] Stage 2A-5 aggregate=$AGG_JOB dependency=afterok:$WORKER_JOB"
echo "[OUTPUT] $OUTPUT_ROOT"
