#!/bin/bash
# Submit Stage-1 univariate CV v6 jobs with source/subset/transform/sample-type ordering.
#
# Default priority order:
#   1) zscore + median + TURBT
#   2) zscore + median + RC
#   3) log1p_zscore + median + TURBT
#   4) log1p_zscore + median + RC
#   5) zscore + mean/max/min + TURBT/RC
#   6) log1p_zscore + mean/max/min + TURBT/RC
#
# Use STAGE1_PHASE to submit one phase at a time if you want the earlier phase
# to truly finish before later phases:
#   primary_turbt, primary_rc, log1p_turbt, log1p_rc,
#   other_aggs_zscore, other_aggs_log1p, all_ordered, custom

set -euo pipefail

FEATURE_DIR=${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/univariate_screening}
SCRIPT=${SCRIPT:-$FEATURE_DIR/stage1_univariate_cv_screen_v6.py}
WORKER=${WORKER:-$FEATURE_DIR/run_stage1_v6_worker.sh}
LOGDIR=${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/logs}
OUTDIR_BASE=${OUTDIR_BASE:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/results}

CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
PY_ENV=${PY_ENV:-cuda6}

# Main modeling knobs
N_SPLITS=${N_SPLITS:-5}
N_REPEATS=${N_REPEATS:-5}
N_FEATURE_CHUNKS=${N_FEATURE_CHUNKS:-1}
QC_ACCEPTABILITY=${QC_ACCEPTABILITY:-acceptable_or_borderline}
MIN_EPI_FRACTION=${MIN_EPI_FRACTION:-0.05}

# Coverage filters after patient aggregation/endpoint filtering
MIN_FEATURE_NONMISSING_FRAC=${MIN_FEATURE_NONMISSING_FRAC:-0.70}
MIN_FEATURE_UNIQUE=${MIN_FEATURE_UNIQUE:-3}
MIN_FEATURE_NONZERO=${MIN_FEATURE_NONZERO:-5}

# Slurm resources per job
MEM=${MEM:-16G}
TIME=${TIME:-12:00:00}
PARTITION=${PARTITION:-upgrade}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

# Comma-separated overrides
COHORTS_CSV=${COHORTS_CSV:-NAC2020,PURE01,No-NAC,NAC2015,BLASST,KOLL}
ENDPOINTS_CSV=${ENDPOINTS_CSV:-complete_response,any_response,OS,RFS}
FEATURE_GROUPS_CSV=${FEATURE_GROUPS_CSV:-NN,athena,cell_features,triads}
FEATURE_SOURCES_CSV=${FEATURE_SOURCES_CSV:-phenotype_only,AR_state,AR_checkpoint_state,compartment,compartment_state}

# Patient subset mode:
#   auto = all everywhere, plus all/no_adj_chemo for No-NAC/KOLL TURBT and RC OS/RFS
#   otherwise use explicit CSV everywhere contexts are not skipped, e.g. PATIENT_SUBSETS_CSV=all,no_adj_chemo
PATIENT_SUBSETS_CSV=${PATIENT_SUBSETS_CSV:-auto}

# KOLL core metadata/crosswalk built from Summary_UBC TMA + CE_summary_UBC.
# Required only for KOLL jobs.
KOLL_METADATA_CSV=${KOLL_METADATA_CSV:-/projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/KOLL_core_metadata.csv}

# Submitter phase control.
STAGE1_PHASE=${STAGE1_PHASE:-all_ordered}
# Used only when STAGE1_PHASE=custom
TRANSFORM_MODES_CSV=${TRANSFORM_MODES_CSV:-zscore}
AGGS_CSV=${AGGS_CSV:-median}
SAMPLE_TYPES_CSV=${SAMPLE_TYPES_CSV:-TURBT}

SKIP_EXISTING=${SKIP_EXISTING:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$LOGDIR" "$OUTDIR_BASE"

need_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
need_file "$SCRIPT"
need_file "$WORKER"

sanitize() {
  echo "$1" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_\+/_/g' | sed 's/^_//;s/_$//' | cut -c1-120
}

csv_to_array() {
  # Bash-3/4.2-compatible CSV splitter.
  # Avoids `local -n` namerefs, which are unavailable on some clusters.
  local csv="$1"
  local arr_name="$2"
  local old_ifs="$IFS"
  local __tmp_csv_arr=()
  local item

  IFS=',' read -r -a __tmp_csv_arr <<< "$csv"
  IFS="$old_ifs"

  eval "$arr_name=()"
  for item in "${__tmp_csv_arr[@]}"; do
    item="$(echo "$item" | xargs)"
    eval "$arr_name+=(\"$item\")"
  done
}

source_panels_csv() {
  local src="$1"
  case "$src" in
    phenotype_only|compartment) echo "AR,BT" ;;
    AR_state|AR_checkpoint_state|compartment_state) echo "AR" ;;
    *) echo "[ERROR] Unknown feature source: $src" >&2; return 1 ;;
  esac
}

should_skip_context() {
  local cohort="$1" endpoint="$2" sample_type="$3"

  # No-NAC and KOLL have no neoadjuvant response endpoints in this screen.
  if [[ ( "$cohort" == "No-NAC" || "$cohort" == "KOLL" ) && ( "$endpoint" == "complete_response" || "$endpoint" == "any_response" ) ]]; then
    return 0
  fi

  return 1
}

patient_subsets_for_context() {
  local cohort="$1" endpoint="$2" sample_type="$3"

  if [[ "$PATIENT_SUBSETS_CSV" != "auto" ]]; then
    echo "$PATIENT_SUBSETS_CSV"
    return 0
  fi

  if [[ ( "$cohort" == "No-NAC" || "$cohort" == "KOLL" ) && ( "$endpoint" == "OS" || "$endpoint" == "RFS" ) && ( "$sample_type" == "TURBT" || "$sample_type" == "RC" || "$sample_type" == "all" ) ]]; then
    echo "all,no_adj_chemo"
  else
    echo "all"
  fi
}

submit_one() {
  local feature_source="$1" panel="$2" feature_group="$3" cohort="$4" endpoint="$5" sample_type="$6" agg="$7" transform_mode="$8" patient_subset="$9" chunk_idx="${10}"

  local outdir="$OUTDIR_BASE/$feature_source/$panel/$feature_group/$cohort/$endpoint/$sample_type/agg-$agg/patient_subset-$patient_subset/transform-$transform_mode"
  mkdir -p "$outdir"

  local stem="${cohort}__${panel}__${feature_source}__${feature_group}__${endpoint}__${sample_type}__${patient_subset}__agg-${agg}__transform-${transform_mode}__chunk$(printf '%03d' "$chunk_idx")of$(printf '%03d' "$N_FEATURE_CHUNKS")"
  local expected_summary="$outdir/${stem}__summary.csv"

  if [[ "$SKIP_EXISTING" == "1" && -s "$expected_summary" ]]; then
    echo "[SKIP_EXISTING] $expected_summary"
    return 0
  fi

  local jobtag
  jobtag=$(sanitize "s1v6_${feature_source}_${panel}_${feature_group}_${cohort}_${endpoint}_${sample_type}_${patient_subset}_${agg}_${transform_mode}_c${chunk_idx}of${N_FEATURE_CHUNKS}")
  local logout="$LOGDIR/${jobtag}.out"
  local logerr="$LOGDIR/${jobtag}.err"

  local args=(
    "$SCRIPT"
    --feature-source "$feature_source"
    --cohort "$cohort"
    --panel "$panel"
    --feature-group "$feature_group"
    --endpoint "$endpoint"
    --sample-type "$sample_type"
    --patient-subset "$patient_subset"
    --qc-acceptability "$QC_ACCEPTABILITY"
    --agg "$agg"
    --n-splits "$N_SPLITS"
    --n-repeats "$N_REPEATS"
    --chunk-idx "$chunk_idx"
    --n-chunks "$N_FEATURE_CHUNKS"
    --outdir "$outdir"
    --transform-mode "$transform_mode"
    --koll-metadata-csv "$KOLL_METADATA_CSV"
    --min-feature-nonmissing-frac "$MIN_FEATURE_NONMISSING_FRAC"
    --min-feature-unique "$MIN_FEATURE_UNIQUE"
    --min-feature-nonzero "$MIN_FEATURE_NONZERO"
  )

  if [[ -n "$MIN_EPI_FRACTION" && "$MIN_EPI_FRACTION" != "none" && "$MIN_EPI_FRACTION" != "None" ]]; then
    args+=(--min-epi-fraction "$MIN_EPI_FRACTION")
  fi

  echo "[SUBMIT] $jobtag"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'sbatch --job-name=%q --cpus-per-task=%q --mem=%q --time=%q -p %q --output=%q --error=%q --export=ALL,CONDA_SH=%q,PY_ENV=%q %q' \
      "$jobtag" "$CPUS_PER_TASK" "$MEM" "$TIME" "$PARTITION" "$logout" "$logerr" "$CONDA_SH" "$PY_ENV" "$WORKER"
    printf ' %q' "${args[@]}"
    printf '\n'
  else
    sbatch \
      --job-name="$jobtag" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME" \
      -p "$PARTITION" \
      --output="$logout" \
      --error="$logerr" \
      --export=ALL,CONDA_SH="$CONDA_SH",PY_ENV="$PY_ENV" \
      "$WORKER" "${args[@]}"
  fi
}

run_phase() {
  local phase_name="$1" transform_csv="$2" agg_csv="$3" sample_csv="$4"
  local TRANSFORMS AGGS SAMPLES COHORTS ENDPOINTS FEATURE_GROUPS FEATURE_SOURCES PANELS SUBSETS
  csv_to_array "$transform_csv" TRANSFORMS
  csv_to_array "$agg_csv" AGGS
  csv_to_array "$sample_csv" SAMPLES
  csv_to_array "$COHORTS_CSV" COHORTS
  csv_to_array "$ENDPOINTS_CSV" ENDPOINTS
  csv_to_array "$FEATURE_GROUPS_CSV" FEATURE_GROUPS
  csv_to_array "$FEATURE_SOURCES_CSV" FEATURE_SOURCES

  echo ""
  echo "======================================================================"
  echo "[PHASE] $phase_name | transforms=$transform_csv | aggs=$agg_csv | sample_types=$sample_csv"
  echo "======================================================================"

  for TRANSFORM_MODE in "${TRANSFORMS[@]}"; do
    for AGG in "${AGGS[@]}"; do
      for SAMPLE_TYPE in "${SAMPLES[@]}"; do
        for FEATURE_SOURCE in "${FEATURE_SOURCES[@]}"; do
          local panels_csv
          panels_csv=$(source_panels_csv "$FEATURE_SOURCE")
          csv_to_array "$panels_csv" PANELS

          for PANEL in "${PANELS[@]}"; do
            for FEATURE_GROUP in "${FEATURE_GROUPS[@]}"; do
              for COHORT in "${COHORTS[@]}"; do
                for ENDPOINT in "${ENDPOINTS[@]}"; do
                  if should_skip_context "$COHORT" "$ENDPOINT" "$SAMPLE_TYPE"; then
                    skip_count=$((skip_count + 1))
                    continue
                  fi

                  local subsets_csv
                  subsets_csv=$(patient_subsets_for_context "$COHORT" "$ENDPOINT" "$SAMPLE_TYPE")
                  csv_to_array "$subsets_csv" SUBSETS

                  for PATIENT_SUBSET in "${SUBSETS[@]}"; do
                    for ((CHUNK_IDX=0; CHUNK_IDX<N_FEATURE_CHUNKS; CHUNK_IDX++)); do
                      submit_one "$FEATURE_SOURCE" "$PANEL" "$FEATURE_GROUP" "$COHORT" "$ENDPOINT" "$SAMPLE_TYPE" "$AGG" "$TRANSFORM_MODE" "$PATIENT_SUBSET" "$CHUNK_IDX"
                      submit_count=$((submit_count + 1))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

submit_count=0
skip_count=0

cat <<INFO
[INFO] STAGE1_PHASE=$STAGE1_PHASE
[INFO] FEATURE_SOURCES_CSV=$FEATURE_SOURCES_CSV
[INFO] FEATURE_GROUPS_CSV=$FEATURE_GROUPS_CSV
[INFO] COHORTS_CSV=$COHORTS_CSV
[INFO] ENDPOINTS_CSV=$ENDPOINTS_CSV
[INFO] PATIENT_SUBSETS_CSV=$PATIENT_SUBSETS_CSV
[INFO] KOLL_METADATA_CSV=$KOLL_METADATA_CSV
[INFO] OUTDIR_BASE=$OUTDIR_BASE
[INFO] LOGDIR=$LOGDIR
[INFO] DRY_RUN=$DRY_RUN; SKIP_EXISTING=$SKIP_EXISTING
INFO

case "$STAGE1_PHASE" in
  primary_turbt)
    run_phase "primary_turbt" "zscore" "median" "TURBT"
    ;;
  primary_rc)
    run_phase "primary_rc" "zscore" "median" "RC"
    ;;
  log1p_turbt)
    run_phase "log1p_turbt" "log1p_zscore" "median" "TURBT"
    ;;
  log1p_rc)
    run_phase "log1p_rc" "log1p_zscore" "median" "RC"
    ;;
  other_aggs_zscore)
    run_phase "other_aggs_zscore" "zscore" "mean,max,min" "TURBT,RC"
    ;;
  other_aggs_log1p)
    run_phase "other_aggs_log1p" "log1p_zscore" "mean,max,min" "TURBT,RC"
    ;;
  all_ordered)
    run_phase "primary_turbt" "zscore" "median" "TURBT"
    run_phase "primary_rc" "zscore" "median" "RC"
    run_phase "log1p_turbt" "log1p_zscore" "median" "TURBT"
    run_phase "log1p_rc" "log1p_zscore" "median" "RC"
    run_phase "other_aggs_zscore" "zscore" "mean,max,min" "TURBT,RC"
    run_phase "other_aggs_log1p" "log1p_zscore" "mean,max,min" "TURBT,RC"
    ;;
  custom)
    run_phase "custom" "$TRANSFORM_MODES_CSV" "$AGGS_CSV" "$SAMPLE_TYPES_CSV"
    ;;
  *)
    echo "[ERROR] Unknown STAGE1_PHASE=$STAGE1_PHASE" >&2
    exit 1
    ;;
esac

echo "[DONE] submitted=$submit_count skipped=$skip_count"
