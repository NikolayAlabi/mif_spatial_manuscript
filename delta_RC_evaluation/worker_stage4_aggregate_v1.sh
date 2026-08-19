#!/bin/bash
#SBATCH --cpus-per-task=1

set -euo pipefail
export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

CONFIG="$1"
MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/delta_RC_evaluation}"

python "$MODULE_DIR/stage4e_aggregate_delta_rc_v1.py" --config "$CONFIG"
