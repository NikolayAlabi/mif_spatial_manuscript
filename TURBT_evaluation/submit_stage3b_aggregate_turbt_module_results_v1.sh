#!/bin/bash
set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/TURBT_evaluation}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage3a_turbt_module_univariate_v1.json}"

OUTROOT="/projects/ovcare/users/nikolay_alabi/immuno/stage3_turbt_module_evaluation_v1/stage3b_aggregate"
mkdir -p "$OUTROOT/logs"

sbatch \
  --job-name=stage3b_aggregate \
  --cpus-per-task=1 --mem=16G --time=02:00:00 \
  --output="$OUTROOT/logs/aggregate_%j.out" \
  --error="$OUTROOT/logs/aggregate_%j.err" \
  --wrap="export PS1=\"\${PS1-}\"; source /home/nalabi/miniconda3/etc/profile.d/conda.sh; conda activate cuda6; python '$MODULE_DIR/stage3b_aggregate_turbt_module_results_v1.py' --config '$CONFIG_JSON'"
