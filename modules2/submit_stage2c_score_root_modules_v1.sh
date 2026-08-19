#!/bin/bash
set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2}"
CONFIG="${CONFIG:-$MODULE_DIR/configs/stage2c_score_root_modules_v1.json}"
SCRIPT="$MODULE_DIR/stage2c_score_root_modules_v1.py"
MODE="${MODE:-validate}"

ROOT="/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2c_root_module_scores_v1"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

case "$MODE" in
  validate)
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=s2c_validate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=$LOGDIR/validate_%j.out
#SBATCH --error=$LOGDIR/validate_%j.err
set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python "$SCRIPT" --config "$CONFIG" validate
EOF
    ;;

  setup)
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=s2c_setup
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=$LOGDIR/setup_%j.out
#SBATCH --error=$LOGDIR/setup_%j.err
set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python "$SCRIPT" --config "$CONFIG" setup
EOF
    ;;

  workers)
    INDEX="$ROOT/stage2c_worker_index.csv"
    if [[ ! -f "$INDEX" ]]; then
      echo "Missing $INDEX. Run MODE=setup first." >&2
      exit 1
    fi
    N=$(awk 'END {print NR-1}' "$INDEX")
    if [[ "$N" -lt 1 ]]; then
      echo "No Stage2C workers indexed." >&2
      exit 1
    fi
    sbatch --array="1-${N}%8" <<EOF
#!/bin/bash
#SBATCH --job-name=s2c_score
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=$LOGDIR/worker_%A_%a.out
#SBATCH --error=$LOGDIR/worker_%A_%a.err
set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python "$SCRIPT" --config "$CONFIG" worker --array-id "\$SLURM_ARRAY_TASK_ID"
EOF
    ;;

  aggregate)
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=s2c_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=$LOGDIR/aggregate_%j.out
#SBATCH --error=$LOGDIR/aggregate_%j.err
set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python "$SCRIPT" --config "$CONFIG" aggregate
EOF
    ;;

  *)
    echo "MODE must be validate, setup, workers, or aggregate" >&2
    exit 1
    ;;
esac
