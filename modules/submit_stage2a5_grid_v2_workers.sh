#!/bin/bash
# Submit corrected Stage 2A-5 cap x rho grid workers only.
# Run inventory before this script and aggregate manually afterward.
# No dependency job is created.

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2a5_cap_rho_grid_v2.json}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2a5_cap_rho_grid_v2.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v9/stage2a5_cap_rho_grid_v2}"
CONTEXT_INDEX="${CONTEXT_INDEX:-$OUTPUT_ROOT/stage2a5_grid_context_index.csv}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v9/logs/stage2a5_grid_v2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
MEM="${MEM:-12G}"
TIME="${TIME:-06:00:00}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

mkdir -p "$LOGDIR"

[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] missing config: $CONFIG_JSON" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing script: $SCRIPT" >&2; exit 1; }
[[ -s "$CONTEXT_INDEX" ]] || {
  echo "[ERROR] missing grid context index: $CONTEXT_INDEX" >&2
  echo "[HINT] run: python $SCRIPT inventory --config $CONFIG_JSON" >&2
  exit 1
}

N_CONTEXTS=$(awk 'END {print NR - 1}' "$CONTEXT_INDEX")
[[ "$N_CONTEXTS" -gt 0 ]] || { echo "[ERROR] no contexts found" >&2; exit 1; }
LAST=$((N_CONTEXTS - 1))

SLURM_SCRIPT="$LOGDIR/stage2a5_grid_v2_workers.slurm.sh"

cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2a5g2
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

python "$SCRIPT" worker \
  --config "$CONFIG_JSON" \
  --array-id "\${SLURM_ARRAY_TASK_ID}"
EOF

chmod +x "$SLURM_SCRIPT"

echo "[INFO] contexts=$N_CONTEXTS array=0-$LAST%$MAX_CONCURRENT"
echo "[INFO] config=$CONFIG_JSON"
echo "[INFO] logs=$LOGDIR"

sbatch --array="0-${LAST}%${MAX_CONCURRENT}" "$SLURM_SCRIPT"
