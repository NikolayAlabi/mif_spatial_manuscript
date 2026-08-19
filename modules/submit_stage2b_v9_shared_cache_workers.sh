#!/bin/bash
# Submit eight corrected v9 Stage 2B shared-cache workers.
# One worker = one cohort x panel, one CPU.
# No dependency job is submitted. Aggregate manually afterward.

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2b_v9_cap_sensitivity.json}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2b_v9_build_shared_cache_worker.py}"
STAGE2B_ROOT="${STAGE2B_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v9/stage2b_cap_sensitivity}"
INDEX="${INDEX:-$STAGE2B_ROOT/setup/shared_cache_worker_index.csv}"
LOGDIR="${LOGDIR:-$STAGE2B_ROOT/logs/shared_cache}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MEM="${MEM:-24G}"
TIME="${TIME:-08:00:00}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

mkdir -p "$LOGDIR"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing script: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] missing config: $CONFIG_JSON" >&2; exit 1; }
[[ -s "$INDEX" ]] || { echo "[ERROR] missing worker index: $INDEX" >&2; exit 1; }

N=$(awk 'END {print NR - 1}' "$INDEX")
[[ "$N" -gt 0 ]] || { echo "[ERROR] zero cache workers" >&2; exit 1; }
LAST=$((N - 1))

SLURM_SCRIPT="$LOGDIR/stage2b_v9_cache_workers.slurm.sh"

cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2b9cache
#SBATCH --cpus-per-task=1
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
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

python "$SCRIPT" \
  --config "$CONFIG_JSON" \
  --array-id "\${SLURM_ARRAY_TASK_ID}"
EOF

chmod +x "$SLURM_SCRIPT"

echo "[INFO] workers=$N array=0-$LAST%$MAX_CONCURRENT"
echo "[INFO] logs=$LOGDIR"

sbatch --array="0-${LAST}%${MAX_CONCURRENT}" "$SLURM_SCRIPT"
