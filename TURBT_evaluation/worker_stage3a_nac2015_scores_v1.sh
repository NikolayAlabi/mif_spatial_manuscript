#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
export PS1="${PS1-}"

source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/TURBT_evaluation}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage3a_turbt_module_univariate_v1.json}"

python "$MODULE_DIR/stage3a_turbt_module_univariate_v1.py" \
  nac2015-worker \
  --config "$CONFIG_JSON"
