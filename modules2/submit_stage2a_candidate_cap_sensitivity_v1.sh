#!/bin/bash
# Explicit four-step submission; no dependencies are created automatically.
#
# Usage:
#   MODE=setup     bash submit_stage2a_candidate_cap_sensitivity_v1.sh
#   MODE=cache     bash submit_stage2a_candidate_cap_sensitivity_v1.sh
#   MODE=workers   bash submit_stage2a_candidate_cap_sensitivity_v1.sh
#   MODE=aggregate bash submit_stage2a_candidate_cap_sensitivity_v1.sh

set -euo pipefail
export PS1="${PS1-}"

MODE="${MODE:-}"
MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a_candidate_cap_sensitivity_v1.py}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a_candidate_cap_sensitivity_v1.json}"
CACHE_WORKER="${CACHE_WORKER:-$MODULE_DIR/worker_stage2a_candidate_cap_cache_v1.sh}"
CONTEXT_WORKER="${CONTEXT_WORKER:-$MODULE_DIR/worker_stage2a_candidate_cap_context_v1.sh}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
MAX_CONCURRENT_CACHE="${MAX_CONCURRENT_CACHE:-8}"
MAX_CONCURRENT_CONTEXT="${MAX_CONCURRENT_CONTEXT:-24}"

if [[ -z "$MODE" ]]; then
  echo "Set MODE=setup, cache, workers, or aggregate" >&2
  exit 1
fi

OUTPUT_ROOT=$(python - <<PY
import json
with open("$CONFIG_JSON") as f:
    print(json.load(f)["output_root"])
PY
)
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs}"
mkdir -p "$LOGDIR"

if [[ "$MODE" == "setup" ]]; then
  JOBFILE="$LOGDIR/setup.slurm.sh"
  cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=s2acap_setup
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGDIR/setup_%j.out
#SBATCH --error=$LOGDIR/setup_%j.err
set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" setup --config "$CONFIG_JSON"
EOF
  chmod +x "$JOBFILE"
  sbatch "$JOBFILE"
  exit 0
fi

if [[ "$MODE" == "cache" ]]; then
  INDEX="$OUTPUT_ROOT/cache_index.csv"
  if [[ ! -s "$INDEX" ]]; then
    echo "Missing $INDEX. Run MODE=setup and wait for completion first." >&2
    exit 1
  fi
  N=$(awk 'END {print NR-1}' "$INDEX")
  if [[ "$N" -lt 1 ]]; then
    echo "No cache contexts found" >&2
    exit 1
  fi
  MAX=$((N-1))
  sbatch \
    --array="0-${MAX}%${MAX_CONCURRENT_CACHE}" \
    --output="$LOGDIR/cache_%A_%a.out" \
    --error="$LOGDIR/cache_%A_%a.err" \
    --export=ALL,MODULE_DIR="$MODULE_DIR",SCRIPT="$SCRIPT",CONFIG_JSON="$CONFIG_JSON",CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV" \
    "$CACHE_WORKER"
  echo "Submitted $N shared cache workers. Wait for all to finish before MODE=workers."
  exit 0
fi

if [[ "$MODE" == "workers" ]]; then
  INDEX="$OUTPUT_ROOT/context_index.csv"
  if [[ ! -s "$INDEX" ]]; then
    echo "Missing $INDEX. Run MODE=setup first." >&2
    exit 1
  fi
  # Verify all shared caches are present before context workers start.
  N_CACHE=$(awk 'END {print NR-1}' "$OUTPUT_ROOT/cache_index.csv")
  N_DONE=$(find "$OUTPUT_ROOT/shared_cache" -name .done 2>/dev/null | wc -l)
  if [[ "$N_DONE" -lt "$N_CACHE" ]]; then
    echo "Only $N_DONE/$N_CACHE shared caches complete. Wait for MODE=cache jobs." >&2
    exit 1
  fi
  N=$(awk 'END {print NR-1}' "$INDEX")
  MAX=$((N-1))
  sbatch \
    --array="0-${MAX}%${MAX_CONCURRENT_CONTEXT}" \
    --output="$LOGDIR/context_%A_%a.out" \
    --error="$LOGDIR/context_%A_%a.err" \
    --export=ALL,MODULE_DIR="$MODULE_DIR",SCRIPT="$SCRIPT",CONFIG_JSON="$CONFIG_JSON",CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV" \
    "$CONTEXT_WORKER"
  echo "Submitted $N context workers. Wait for all to finish before MODE=aggregate."
  exit 0
fi

if [[ "$MODE" == "aggregate" ]]; then
  JOBFILE="$LOGDIR/aggregate.slurm.sh"
  cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=s2acap_agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGDIR/aggregate_%j.out
#SBATCH --error=$LOGDIR/aggregate_%j.err
set -euo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python "$SCRIPT" aggregate --config "$CONFIG_JSON"
EOF
  chmod +x "$JOBFILE"
  sbatch "$JOBFILE"
  exit 0
fi

echo "Unknown MODE=$MODE" >&2
exit 1
