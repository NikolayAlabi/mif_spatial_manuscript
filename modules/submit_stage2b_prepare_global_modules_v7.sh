#!/bin/bash
# Submit Stage 2B global module preparation.
# Usage:
#   bash submit_stage2b_prepare_global_modules_v7.sh primary_turbt_expanded
#   bash submit_stage2b_prepare_global_modules_v7.sh phenotype_only_repro
#   bash submit_stage2b_prepare_global_modules_v7.sh rc_exploratory
#   CONFIG_JSON=/path/to/config.json bash submit_stage2b_prepare_global_modules_v7.sh custom

set -euo pipefail

PRESET="${1:-primary_turbt_expanded}"

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}"
STAGE2_ROOT="${STAGE2_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v7}"
LOGDIR="${LOGDIR:-$STAGE2_ROOT/logs}"
mkdir -p "$LOGDIR"

SCRIPT="$MODULE_DIR/stage2b_prepare_global_module_inputs_v7.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Missing script: $SCRIPT" >&2
  exit 1
fi

CONFIG_DIR="$MODULE_DIR/configs"
case "$PRESET" in
  primary_turbt_expanded)
    CONFIG_JSON="${CONFIG_JSON:-$CONFIG_DIR/primary_turbt_expanded.json}"
    ;;
  phenotype_only_repro)
    CONFIG_JSON="${CONFIG_JSON:-$CONFIG_DIR/phenotype_only_reproducibility.json}"
    ;;
  rc_exploratory)
    CONFIG_JSON="${CONFIG_JSON:-$CONFIG_DIR/rc_exploratory.json}"
    ;;
  custom)
    if [[ -z "${CONFIG_JSON:-}" ]]; then
      echo "[ERROR] PRESET=custom requires CONFIG_JSON=/path/to/config.json" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Unknown preset: $PRESET" >&2
    echo "Allowed: primary_turbt_expanded, phenotype_only_repro, rc_exploratory, custom" >&2
    exit 1
    ;;
esac

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "[ERROR] Missing config: $CONFIG_JSON" >&2
  exit 1
fi

JOB_NAME="s2b_${PRESET}"
SLURM_SCRIPT="$LOGDIR/${JOB_NAME}.slurm.sh"
OUT_LOG="$LOGDIR/${JOB_NAME}.out"
ERR_LOG="$LOGDIR/${JOB_NAME}.err"

MEM="${MEM:-128G}"
CPUS="${CPUS:-8}"
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-upgrade}"
CONDA_ENV="${CONDA_ENV:-cuda6}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
FORCE_FLAG=""
if [[ "${FORCE_REBUILD_MATRICES:-0}" == "1" ]]; then
  FORCE_FLAG="--force-rebuild-matrices"
fi

cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=$TIME
#SBATCH --output=$OUT_LOG
#SBATCH --error=$ERR_LOG
##SBATCH --partition=$PARTITION

set -eo pipefail
export PS1="${PS1:-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

cd "$MODULE_DIR"

echo "[INFO] host=\$(hostname)"
echo "[INFO] date=\$(date)"
echo "[INFO] python=\$(which python)"
python - <<'PY'
import sys
import numpy, pandas, scipy, sklearn
print('[INFO] python_version', sys.version)
print('[INFO] numpy', numpy.__version__)
print('[INFO] pandas', pandas.__version__)
print('[INFO] scipy', scipy.__version__)
print('[INFO] sklearn', sklearn.__version__)
PY

python "$SCRIPT" \
  --config "$CONFIG_JSON" \
  $FORCE_FLAG

echo "[DONE] date=\$(date)"
EOF

chmod +x "$SLURM_SCRIPT"

echo "[INFO] preset=$PRESET"
echo "[INFO] config=$CONFIG_JSON"
echo "[INFO] slurm_script=$SLURM_SCRIPT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Would submit: sbatch $SLURM_SCRIPT"
  sed -n '1,160p' "$SLURM_SCRIPT"
  exit 0
fi

sbatch "$SLURM_SCRIPT"
