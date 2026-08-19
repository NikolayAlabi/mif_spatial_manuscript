#!/bin/bash
# Build the shared cap-20 union matrix cache.
# Intentionally omits any #SBATCH partition directive.

set -euo pipefail

MODULE_DIR=${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}
CONFIG_JSON=${CONFIG_JSON:-$MODULE_DIR/configs/stage2b_cap_sensitivity/stage2b_shared_matrix_cache.json}
SCRIPT=${SCRIPT:-$MODULE_DIR/stage2b_build_shared_matrix_cache_v1.py}
LOGDIR=${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/logs/stage2b_cap_sensitivity}
CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-cuda6}
CPUS=${CPUS:-8}
MEM=${MEM:-64G}
TIME=${TIME:-12:00:00}
FORCE=${FORCE:-0}

mkdir -p "$LOGDIR"
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG_JSON" ]] || { echo "[ERROR] Missing config: $CONFIG_JSON" >&2; exit 1; }

SLURM_SCRIPT="$LOGDIR/stage2b_shared_cache.slurm.sh"
cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=s2b_cache
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGDIR/stage2b_shared_cache_%j.out
#SBATCH --error=$LOGDIR/stage2b_shared_cache_%j.err

set -eo pipefail
export PS1="\${PS1-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export NUMEXPR_NUM_THREADS=$CPUS

ARGS=(--config "$CONFIG_JSON")
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi
python "$SCRIPT" "\${ARGS[@]}"
EOF
chmod +x "$SLURM_SCRIPT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat "$SLURM_SCRIPT"
  exit 0
fi

sbatch "$SLURM_SCRIPT"
