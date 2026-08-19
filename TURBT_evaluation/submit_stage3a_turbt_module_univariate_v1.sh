#!/bin/bash
set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/TURBT_evaluation}"
CONFIG_JSON="${CONFIG_JSON:-$MODULE_DIR/configs/stage3a_turbt_module_univariate_v1.json}"
MODE="${MODE:-validate}"

OUTROOT="/projects/ovcare/users/nikolay_alabi/immuno/stage3_turbt_module_evaluation_v1/stage3a_univariate"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

case "$MODE" in

  validate)
    sbatch \
      --job-name=s3a_validate \
      --cpus-per-task=1 --mem=8G --time=01:00:00 \
      --output="$LOGDIR/validate_%j.out" \
      --error="$LOGDIR/validate_%j.err" \
      --wrap="bash -lc 'export PS1=\"\${PS1-}\"; source /home/nalabi/miniconda3/etc/profile.d/conda.sh; conda activate cuda6; python \"$MODULE_DIR/stage3a_turbt_module_univariate_v1.py\" validate --config \"$CONFIG_JSON\"'"
    ;;

  setup)
    sbatch \
      --job-name=s3a_setup \
      --cpus-per-task=1 --mem=16G --time=02:00:00 \
      --output="$LOGDIR/setup_%j.out" \
      --error="$LOGDIR/setup_%j.err" \
      --wrap="bash -lc 'export PS1=\"\${PS1-}\"; source /home/nalabi/miniconda3/etc/profile.d/conda.sh; conda activate cuda6; python \"$MODULE_DIR/stage3a_turbt_module_univariate_v1.py\" setup --config \"$CONFIG_JSON\"'"
    ;;

  nac2015)
    IDX="$OUTROOT/stage3a_nac2015_score_worker_index.csv"
    [[ -f "$IDX" ]] || { echo "Missing $IDX. Run MODE=setup first."; exit 1; }
    N=$(awk 'END {print NR-1}' "$IDX")
    [[ "$N" -gt 0 ]] || { echo "No NAC2015 workers."; exit 1; }
    sbatch \
      --job-name=s3a_nac2015 \
      --array="1-${N}" \
      --cpus-per-task=1 --mem=32G --time=12:00:00 \
      --output="$LOGDIR/nac2015_%A_%a.out" \
      --error="$LOGDIR/nac2015_%A_%a.err" \
      --export=ALL,MODULE_DIR="$MODULE_DIR",CONFIG_JSON="$CONFIG_JSON" \
      "$MODULE_DIR/worker_stage3a_nac2015_scores_v1.sh"
    ;;

  finalize)
    sbatch \
      --job-name=s3a_finalize \
      --cpus-per-task=1 --mem=8G --time=01:00:00 \
      --output="$LOGDIR/finalize_%j.out" \
      --error="$LOGDIR/finalize_%j.err" \
      --wrap="bash -lc 'export PS1=\"\${PS1-}\"; source /home/nalabi/miniconda3/etc/profile.d/conda.sh; conda activate cuda6; python \"$MODULE_DIR/stage3a_turbt_module_univariate_v1.py\" finalize --config \"$CONFIG_JSON\"'"
    ;;

  workers)
    IDX="$OUTROOT/stage3a_context_index.csv"
    [[ -f "$IDX" ]] || { echo "Missing $IDX. Run MODE=finalize first."; exit 1; }
    N=$(awk 'END {print NR-1}' "$IDX")
    [[ "$N" -gt 0 ]] || { echo "No Stage3A contexts."; exit 1; }
    sbatch \
      --job-name=s3a_uni \
      --array="1-${N}%12" \
      --cpus-per-task=1 --mem=8G --time=04:00:00 \
      --output="$LOGDIR/worker_%A_%a.out" \
      --error="$LOGDIR/worker_%A_%a.err" \
      --export=ALL,MODULE_DIR="$MODULE_DIR",CONFIG_JSON="$CONFIG_JSON" \
      "$MODULE_DIR/worker_stage3a_univariate_v1.sh"
    ;;

  *)
    echo "MODE must be one of: validate setup nac2015 finalize workers"
    exit 1
    ;;
esac