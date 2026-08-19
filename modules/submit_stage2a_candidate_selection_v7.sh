#!/bin/bash
# Submit Stage-2A candidate-selection jobs for global module discovery v7.
# Fixed version: writes an explicit bash Slurm script instead of using sbatch --wrap,
# because --wrap may run under /bin/sh on some clusters and fail on `source`.
#
# Example:
#   bash submit_stage2a_candidate_selection_v7_1.sh audit_all_median
#   bash submit_stage2a_candidate_selection_v7_1.sh discovery_primary_median
#   bash submit_stage2a_candidate_selection_v7_1.sh aggregation_sensitivity_main4
#
# Useful overrides:
#   TRANSFORM_MODES_CSV=zscore bash submit_stage2a_candidate_selection_v7_1.sh audit_all_median
#   TRANSFORM_MODES_CSV=zscore,log1p_zscore bash submit_stage2a_candidate_selection_v7_1.sh audit_all_median
#   DRY_RUN=1 bash submit_stage2a_candidate_selection_v7_1.sh audit_all_median

set -euo pipefail

MODE=${1:-audit_all_median}

MODULE_DIR=${MODULE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules}
SCRIPT=${SCRIPT:-$MODULE_DIR/stage2_module_pipeline_v7.py}

STAGE1_ROOT=${STAGE1_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/results}
STAGE2_ROOT=${STAGE2_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v7}
LOGDIR=${LOGDIR:-$STAGE2_ROOT/logs}

CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
PY_ENV=${PY_ENV:-cuda6}

PARTITION=${PARTITION:-upgrade}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
MEM=${MEM:-64G}
TIME=${TIME:-08:00:00}

# At the moment, if log1p median jobs are not finished, use TRANSFORM_MODES_CSV=zscore.
# Once log1p jobs are complete, set TRANSFORM_MODES_CSV=zscore,log1p_zscore.
TRANSFORM_MODES_CSV=${TRANSFORM_MODES_CSV:-zscore}

# Optional knobs passed through to the Python script.
TOP_PER_SOURCE_GROUP=${TOP_PER_SOURCE_GROUP:-5}
TOP_PER_SOURCE=${TOP_PER_SOURCE:-0}
MAX_CANDIDATES_PER_CONTEXT=${MAX_CANDIDATES_PER_CONTEXT:-80}
MIN_CANDIDATES_PER_CONTEXT=${MIN_CANDIDATES_PER_CONTEXT:-20}
USE_CV_STD_FILTER=${USE_CV_STD_FILTER:-1}
USE_PANEL_SOURCE_MAP=${USE_PANEL_SOURCE_MAP:-1}

mkdir -p "$LOGDIR"

case "$MODE" in
  audit_all_median)
    OUTDIR="$STAGE2_ROOT/audit_all_cohorts_median_best_transform"
    COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC,NAC2015,KOLL"
    AGG_SUMMARY_COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    SAMPLE_TYPES_CSV="TURBT,RC"
    PATIENT_SUBSETS_CSV="all,no_adj_chemo"
    AGGS_CSV="median"
    EVALUATE_ALL_FLAG="--evaluate-all-cohorts"
    ;;

  discovery_primary_median)
    OUTDIR="$STAGE2_ROOT/discovery_primary_median_best_transform"
    COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    AGG_SUMMARY_COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    SAMPLE_TYPES_CSV="TURBT"
    PATIENT_SUBSETS_CSV="all"
    AGGS_CSV="median"
    EVALUATE_ALL_FLAG=""
    ;;

  aggregation_sensitivity_main4)
    OUTDIR="$STAGE2_ROOT/aggregation_sensitivity_main4_best_transform"
    COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    AGG_SUMMARY_COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    SAMPLE_TYPES_CSV="TURBT,RC"
    PATIENT_SUBSETS_CSV="all"
    AGGS_CSV="mean,max,min"
    EVALUATE_ALL_FLAG=""
    ;;

  log1p_median_main4)
    OUTDIR="$STAGE2_ROOT/log1p_median_main4"
    COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    AGG_SUMMARY_COHORTS_CSV="NAC2020,PURE01,BLASST,No-NAC"
    SAMPLE_TYPES_CSV="TURBT,RC"
    PATIENT_SUBSETS_CSV="all"
    AGGS_CSV="median"
    TRANSFORM_MODES_CSV="log1p_zscore"
    EVALUATE_ALL_FLAG=""
    ;;

  custom)
    OUTDIR=${OUTDIR:?For MODE=custom, set OUTDIR=/path/to/output}
    COHORTS_CSV=${COHORTS_CSV:?For MODE=custom, set COHORTS_CSV}
    AGG_SUMMARY_COHORTS_CSV=${AGG_SUMMARY_COHORTS_CSV:-$COHORTS_CSV}
    SAMPLE_TYPES_CSV=${SAMPLE_TYPES_CSV:-TURBT}
    PATIENT_SUBSETS_CSV=${PATIENT_SUBSETS_CSV:-all}
    AGGS_CSV=${AGGS_CSV:-median}
    EVALUATE_ALL_FLAG=${EVALUATE_ALL_FLAG:-}
    ;;

  *)
    echo "[ERROR] Unknown MODE=$MODE" >&2
    echo "Allowed: audit_all_median, discovery_primary_median, aggregation_sensitivity_main4, log1p_median_main4, custom" >&2
    exit 1
    ;;
esac

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Script not found: $SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] Conda profile script not found: $CONDA_SH" >&2
  exit 1
fi

JOB_NAME="s2a_v7_${MODE}"
LOGOUT="$LOGDIR/${JOB_NAME}.out"
LOGERR="$LOGDIR/${JOB_NAME}.err"
SLURM_SCRIPT="$LOGDIR/${JOB_NAME}.slurm.sh"

PANEL_SOURCE_FLAG="--use-panel-source-map"
if [[ "$USE_PANEL_SOURCE_MAP" == "0" ]]; then
  PANEL_SOURCE_FLAG="--no-use-panel-source-map"
fi

CV_STD_FLAG="--use-cv-std-filter"
if [[ "$USE_CV_STD_FILTER" == "0" ]]; then
  CV_STD_FLAG="--no-use-cv-std-filter"
fi

cat > "$SLURM_SCRIPT" <<EOS
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOGOUT
#SBATCH --error=$LOGERR
##SBATCH -p $PARTITION

set -euo pipefail

hostname
date

echo "[INFO] SLURM_JOB_ID=\
\${SLURM_JOB_ID:-NA}"
echo "[INFO] CONDA_SH=$CONDA_SH"
echo "[INFO] PY_ENV=$PY_ENV"

set +u
source "$CONDA_SH"
conda activate "$PY_ENV"
set -u

echo "[INFO] python=\$(which python)"
python --version
python - <<'PY'
import sys
import numpy
import pandas
print('[INFO] numpy', numpy.__version__)
print('[INFO] pandas', pandas.__version__)
PY

python -u "$SCRIPT" \
  --stage1-root "$STAGE1_ROOT" \
  --output-root "$OUTDIR" \
  --cohorts "$COHORTS_CSV" \
  --aggregation-summary-cohorts "$AGG_SUMMARY_COHORTS_CSV" \
  --sample-types "$SAMPLE_TYPES_CSV" \
  --patient-subsets "$PATIENT_SUBSETS_CSV" \
  --aggs "$AGGS_CSV" \
  --transform-modes "$TRANSFORM_MODES_CSV" \
  --top-per-source-group "$TOP_PER_SOURCE_GROUP" \
  --top-per-source "$TOP_PER_SOURCE" \
  --max-candidates-per-context "$MAX_CANDIDATES_PER_CONTEXT" \
  --min-candidates-per-context "$MIN_CANDIDATES_PER_CONTEXT" \
  $PANEL_SOURCE_FLAG \
  $CV_STD_FLAG \
  $EVALUATE_ALL_FLAG

date
echo "[DONE] $JOB_NAME"
EOS

chmod +x "$SLURM_SCRIPT"

echo "[INFO] MODE=$MODE"
echo "[INFO] SCRIPT=$SCRIPT"
echo "[INFO] STAGE1_ROOT=$STAGE1_ROOT"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] COHORTS_CSV=$COHORTS_CSV"
echo "[INFO] AGG_SUMMARY_COHORTS_CSV=$AGG_SUMMARY_COHORTS_CSV"
echo "[INFO] SAMPLE_TYPES_CSV=$SAMPLE_TYPES_CSV"
echo "[INFO] PATIENT_SUBSETS_CSV=$PATIENT_SUBSETS_CSV"
echo "[INFO] AGGS_CSV=$AGGS_CSV"
echo "[INFO] TRANSFORM_MODES_CSV=$TRANSFORM_MODES_CSV"
echo "[INFO] LOGOUT=$LOGOUT"
echo "[INFO] LOGERR=$LOGERR"
echo "[INFO] SLURM_SCRIPT=$SLURM_SCRIPT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] sbatch $SLURM_SCRIPT"
  echo "---------- $SLURM_SCRIPT ----------"
  sed -n '1,220p' "$SLURM_SCRIPT"
  exit 0
fi

sbatch "$SLURM_SCRIPT"
