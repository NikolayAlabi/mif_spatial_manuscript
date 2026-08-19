#!/bin/bash
# Submit cap 10, 15, and 20 Stage 2B cached preparation jobs.
# Run only after the shared matrix cache job has completed successfully.
# Intentionally omits any #SBATCH partition directive.

set -euo pipefail

MODULE_DIR=${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}
CONFIG_DIR=${CONFIG_DIR:-$MODULE_DIR/configs/stage2b_cap_sensitivity}
SCRIPT=${SCRIPT:-$MODULE_DIR/stage2b_prepare_global_module_inputs_cached_v1.py}
LOGDIR=${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs/stage2b_cap_sensitivity}
CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuda6}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
TIME=${TIME:-12:00:00}
RHO_TOKEN=${RHO_TOKEN:-0p9}
CAPS=${CAPS:-"10 15 20"}

mkdir -p "$LOGDIR"
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 1; }

for CAP in $CAPS; do
  CAP_PAD=$(printf "%03d" "$CAP")
  CONFIG_JSON="$CONFIG_DIR/stage2b_cap${CAP_PAD}_rho${RHO_TOKEN}.json"
  [[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] Missing config: $CONFIG_JSON" >&2; exit 1; }

  SLURM_SCRIPT="$LOGDIR/stage2b_cap${CAP_PAD}.slurm.sh"
  cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2b_c${CAP}
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGDIR/stage2b_cap${CAP_PAD}_%j.out
#SBATCH --error=$LOGDIR/stage2b_cap${CAP_PAD}_%j.err

set -eo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

python "$SCRIPT" --config "$CONFIG_JSON"
EOF
  chmod +x "$SLURM_SCRIPT"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "===== cap $CAP ====="
    cat "$SLURM_SCRIPT"
  else
    sbatch "$SLURM_SCRIPT"
  fi
done
