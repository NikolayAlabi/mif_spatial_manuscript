#!/bin/bash
# Submit corrected v9 Stage 2B statistical preparation for cap 10/15/20.
# Each cap independently reuses the same shared matrix cache.
# No dependency job is submitted.

set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage2b_v9_cap_sensitivity.json}"
SCRIPT="${SCRIPT:-$MODULE_DIR/stage2b_v9_prepare_cap.py}"
STAGE2B_ROOT="${STAGE2B_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v9/stage2b_cap_sensitivity}"
LOGDIR="${LOGDIR:-$STAGE2B_ROOT/logs/prepare_caps}"
CAPS="${CAPS:-10 15 20}"
MEM="${MEM:-24G}"
TIME="${TIME:-04:00:00}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

mkdir -p "$LOGDIR"

for CAP in $CAPS; do
  SLURM_SCRIPT="$LOGDIR/cap$(printf '%03d' "$CAP").slurm.sh"

  cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2b9c${CAP}
#SBATCH --cpus-per-task=1
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGDIR/cap$(printf '%03d' "$CAP").out
#SBATCH --error=$LOGDIR/cap$(printf '%03d' "$CAP").err

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
  --cap "$CAP"
EOF

  chmod +x "$SLURM_SCRIPT"
  sbatch "$SLURM_SCRIPT"
done
