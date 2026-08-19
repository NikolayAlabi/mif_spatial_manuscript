#!/bin/bash
# Submit final module freezing after choosing k.
# Usage:
#   FINAL_K="AR=24,BT=12" bash submit_stage2c_finalize_global_modules_v7.sh primary_turbt_expanded

set -euo pipefail

PRESET="${1:-primary_turbt_expanded}"
MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
STAGE2_ROOT="${STAGE2_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v7}"
LOGDIR="${LOGDIR:-$STAGE2_ROOT/logs}"
mkdir -p "$LOGDIR"

SCRIPT="$MODULE_DIR/stage2c_finalize_global_modules_v7.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Missing script: $SCRIPT" >&2
  exit 1
fi

case "$PRESET" in
  primary_turbt_expanded)
    PREPARED_ROOT="${PREPARED_ROOT:-$STAGE2_ROOT/global_module_discovery_primary_turbt_expanded}"
    ;;
  phenotype_only_repro)
    PREPARED_ROOT="${PREPARED_ROOT:-$STAGE2_ROOT/global_module_discovery_phenotype_only_reproducibility}"
    ;;
  rc_exploratory)
    PREPARED_ROOT="${PREPARED_ROOT:-$STAGE2_ROOT/global_module_discovery_rc_exploratory}"
    ;;
  custom)
    if [[ -z "${PREPARED_ROOT:-}" ]]; then
      echo "[ERROR] custom preset requires PREPARED_ROOT=/path/to/prepared/root" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Unknown preset: $PRESET" >&2
    exit 1
    ;;
esac

FINAL_K="${FINAL_K:-}"
if [[ -z "$FINAL_K" ]]; then
  echo "[ERROR] Set FINAL_K, e.g. FINAL_K='AR=24,BT=12'" >&2
  exit 1
fi

JOB_NAME="s2c_${PRESET}"
SLURM_SCRIPT="$LOGDIR/${JOB_NAME}.slurm.sh"
OUT_LOG="$LOGDIR/${JOB_NAME}.out"
ERR_LOG="$LOGDIR/${JOB_NAME}.err"
CONDA_ENV="${CONDA_ENV:-cuda6}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
PARTITION="${PARTITION:-upgrade}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-2}"

cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
#SBATCH --output=$OUT_LOG
#SBATCH --error=$ERR_LOG

set -eo pipefail
export PS1="${PS1:-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$MODULE_DIR"

echo "[INFO] host=\$(hostname)"
echo "[INFO] date=\$(date)"
echo "[INFO] python=\$(which python)"

python "$SCRIPT" \
  --prepared-root "$PREPARED_ROOT" \
  --final-k "$FINAL_K"

echo "[DONE] date=\$(date)"
EOF
chmod +x "$SLURM_SCRIPT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Would submit: sbatch $SLURM_SCRIPT"
  sed -n '1,160p' "$SLURM_SCRIPT"
  exit 0
fi

sbatch "$SLURM_SCRIPT"
