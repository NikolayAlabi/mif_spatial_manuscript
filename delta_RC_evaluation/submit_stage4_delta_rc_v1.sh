#!/bin/bash
set -euo pipefail

MODULE_DIR="${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/delta_RC_evaluation}"
CONFIG="${CONFIG:-$MODULE_DIR/configs/stage4_delta_rc_v1.json}"
MODE="${MODE:-validate}"

OUTROOT="/projects/ovcare/users/nikolay_alabi/immuno/stage4_delta_rc_evaluation_v1"
LOGDIR="$OUTROOT/logs"
mkdir -p "$LOGDIR"

GENERIC="$MODULE_DIR/worker_stage4_generic_v1.sh"

submit_single () {
  local name="$1"; local mem="$2"; local time="$3"; local script="$4"; local command="$5"
  sbatch \
    --job-name="$name" \
    --cpus-per-task=1 --mem="$mem" --time="$time" \
    --output="$LOGDIR/${name}_%j.out" \
    --error="$LOGDIR/${name}_%j.err" \
    --export=ALL \
    "$GENERIC" "$script" "$command" "$CONFIG"
}

submit_array () {
  local name="$1"; local mem="$2"; local time="$3"; local script="$4"; local command="$5"; local index="$6"; local maxrun="${7:-8}"
  [[ -f "$index" ]] || { echo "Missing index: $index"; exit 1; }
  local n
  n=$(awk 'END {print NR-1}' "$index")
  [[ "$n" -gt 0 ]] || { echo "No workers in $index"; exit 1; }
  sbatch \
    --job-name="$name" \
    --array="1-${n}%${maxrun}" \
    --cpus-per-task=1 --mem="$mem" --time="$time" \
    --output="$LOGDIR/${name}_%A_%a.out" \
    --error="$LOGDIR/${name}_%A_%a.err" \
    --export=ALL \
    "$GENERIC" "$script" "$command" "$CONFIG"
}

case "$MODE" in
  validate)
    submit_single "s4_validate" 8G 01:00:00 "$MODULE_DIR/stage4a_score_programs_v1.py" validate
    ;;
  score_setup)
    submit_single "s4_score_setup" 8G 01:00:00 "$MODULE_DIR/stage4a_score_programs_v1.py" setup
    ;;
  score_workers)
    submit_array "s4_score" 32G 12:00:00 "$MODULE_DIR/stage4a_score_programs_v1.py" worker "$OUTROOT/stage4a_score_worker_index.csv" 6
    ;;
  shift_setup)
    submit_single "s4_shift_setup" 8G 01:00:00 "$MODULE_DIR/stage4b_matched_shift_v1.py" setup
    ;;
  shift_workers)
    submit_array "s4_shift" 8G 02:00:00 "$MODULE_DIR/stage4b_matched_shift_v1.py" worker "$OUTROOT/stage4b_shift_worker_index.csv" 10
    ;;
  delta_setup)
    submit_single "s4_delta_setup" 8G 01:00:00 "$MODULE_DIR/stage4c_delta_outcomes_v1.py" setup
    ;;
  delta_workers)
    submit_array "s4_delta" 8G 04:00:00 "$MODULE_DIR/stage4c_delta_outcomes_v1.py" worker "$OUTROOT/stage4c_delta_worker_index.csv" 12
    ;;
  rc_setup)
    submit_single "s4_rc_setup" 8G 01:00:00 "$MODULE_DIR/stage4d_rc_only_univariate_v1.py" setup
    ;;
  rc_workers)
    submit_array "s4_rc" 8G 04:00:00 "$MODULE_DIR/stage4d_rc_only_univariate_v1.py" worker "$OUTROOT/stage4d_rc_worker_index.csv" 12
    ;;
  aggregate)
    sbatch \
      --job-name="s4_aggregate" \
      --cpus-per-task=1 --mem=16G --time=02:00:00 \
      --output="$LOGDIR/s4_aggregate_%j.out" \
      --error="$LOGDIR/s4_aggregate_%j.err" \
      --export=ALL \
      "$MODULE_DIR/worker_stage4_aggregate_v1.sh" "$CONFIG"
    ;;
  *)
    echo "MODE must be: validate score_setup score_workers shift_setup shift_workers delta_setup delta_workers rc_setup rc_workers aggregate"
    exit 1
    ;;
esac
