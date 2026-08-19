#!/bin/bash
#SBATCH --cpus-per-task=1

set -euo pipefail
export PS1="${PS1-}"

source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

SCRIPT="$1"
COMMAND="$2"
CONFIG="$3"

shift 3

python "$SCRIPT" "$COMMAND" --config "$CONFIG" "$@"
