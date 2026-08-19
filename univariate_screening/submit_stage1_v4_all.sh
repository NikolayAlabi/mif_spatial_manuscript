#!/bin/bash
# Submit Stage-1 univariate CV v4 jobs.
# One CPU per job. Jobs are emitted by feature_source x panel x feature_group x cohort x endpoint x sample_type x agg x feature_chunk.

set -euo pipefail

FEATURE_DIR=${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/univariate_screening}
SCRIPT=${SCRIPT:-$FEATURE_DIR/stage1_univariate_cv_screen_v4.py}
WORKER=${WORKER:-$FEATURE_DIR/run_stage1_v4_worker.sh}
LOGDIR=${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v4/logs}
OUTDIR_BASE=${OUTDIR_BASE:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v4/results}

CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
PY_ENV=${PY_ENV:-cuda6}

# Main modeling knobs
N_SPLITS=${N_SPLITS:-5}
N_REPEATS=${N_REPEATS:-5}
N_FEATURE_CHUNKS=${N_FEATURE_CHUNKS:-1}
TRANSFORM_MODE=${TRANSFORM_MODE:-zscore}
QC_ACCEPTABILITY=${QC_ACCEPTABILITY:-acceptable_or_borderline}
MIN_EPI_FRACTION=${MIN_EPI_FRACTION:-0.05}
PATIENT_SUBSET=${PATIENT_SUBSET:-all}

# Coverage filters after patient aggregation/endpoint filtering
MIN_FEATURE_NONMISSING_FRAC=${MIN_FEATURE_NONMISSING_FRAC:-0.70}
MIN_FEATURE_UNIQUE=${MIN_FEATURE_UNIQUE:-3}
MIN_FEATURE_NONZERO=${MIN_FEATURE_NONZERO:-5}

# Slurm resources per job
MEM=${MEM:-16G}
TIME=${TIME:-12:00:00}
PARTITION=${PARTITION:-upgrade}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

# Comma-separated overrides, e.g. COHORTS_CSV="NAC2020,PURE01"
COHORTS_CSV=${COHORTS_CSV:-NAC2020,PURE01,No-NAC,NAC2015,BLASST}
ENDPOINTS_CSV=${ENDPOINTS_CSV:-complete_response,any_response,OS,RFS}
AGGS_CSV=${AGGS_CSV:-median,mean,max,min}
SAMPLE_TYPES_CSV=${SAMPLE_TYPES_CSV:-TURBT}
FEATURE_GROUPS_CSV=${FEATURE_GROUPS_CSV:-NN,athena,cell_features,triads}
FEATURE_SOURCES_CSV=${FEATURE_SOURCES_CSV:-phenotype_only,AR_state}

IFS=',' read -r -a COHORTS <<< "$COHORTS_CSV"
IFS=',' read -r -a ENDPOINTS <<< "$ENDPOINTS_CSV"
IFS=',' read -r -a AGGS <<< "$AGGS_CSV"
IFS=',' read -r -a SAMPLE_TYPES <<< "$SAMPLE_TYPES_CSV"
IFS=',' read -r -a FEATURE_GROUPS <<< "$FEATURE_GROUPS_CSV"
IFS=',' read -r -a FEATURE_SOURCES <<< "$FEATURE_SOURCES_CSV"

mkdir -p "$LOGDIR" "$OUTDIR_BASE"

submit_count=0
skip_count=0

sanitize() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

for FEATURE_SOURCE in "${FEATURE_SOURCES[@]}"; do
  if [[ "$FEATURE_SOURCE" == "phenotype_only" ]]; then
    PANELS=(AR BT)
  elif [[ "$FEATURE_SOURCE" == "AR_state" ]]; then
    PANELS=(AR)
  else
    echo "[ERROR] Unknown FEATURE_SOURCE=$FEATURE_SOURCE" >&2
    exit 1
  fi

  for PANEL in "${PANELS[@]}"; do
    for FEATURE_GROUP in "${FEATURE_GROUPS[@]}"; do
      for COHORT in "${COHORTS[@]}"; do
        for ENDPOINT in "${ENDPOINTS[@]}"; do
          # No-NAC has no neoadjuvant response endpoint; skip rather than submit guaranteed-empty jobs.
          if [[ "$COHORT" == "No-NAC" && ( "$ENDPOINT" == "complete_response" || "$ENDPOINT" == "any_response" ) ]]; then
            skip_count=$((skip_count + 1))
            continue
          fi

          for SAMPLE_TYPE in "${SAMPLE_TYPES[@]}"; do
            for AGG in "${AGGS[@]}"; do
              for ((CHUNK_IDX=0; CHUNK_IDX<N_FEATURE_CHUNKS; CHUNK_IDX++)); do
                OUTDIR="$OUTDIR_BASE/$FEATURE_SOURCE/$PANEL/$FEATURE_GROUP/$COHORT/$ENDPOINT/$SAMPLE_TYPE/agg-$AGG"
                mkdir -p "$OUTDIR"

                JOBTAG=$(sanitize "s1v4_${FEATURE_SOURCE}_${PANEL}_${FEATURE_GROUP}_${COHORT}_${ENDPOINT}_${SAMPLE_TYPE}_${AGG}_c${CHUNK_IDX}of${N_FEATURE_CHUNKS}")
                LOGOUT="$LOGDIR/${JOBTAG}.out"
                LOGERR="$LOGDIR/${JOBTAG}.err"

                ARGS=(
                  "$SCRIPT"
                  --feature-source "$FEATURE_SOURCE"
                  --cohort "$COHORT"
                  --panel "$PANEL"
                  --feature-group "$FEATURE_GROUP"
                  --endpoint "$ENDPOINT"
                  --sample-type "$SAMPLE_TYPE"
                  --patient-subset "$PATIENT_SUBSET"
                  --qc-acceptability "$QC_ACCEPTABILITY"
                  --agg "$AGG"
                  --n-splits "$N_SPLITS"
                  --n-repeats "$N_REPEATS"
                  --chunk-idx "$CHUNK_IDX"
                  --n-chunks "$N_FEATURE_CHUNKS"
                  --outdir "$OUTDIR"
                  --transform-mode "$TRANSFORM_MODE"
                  --min-feature-nonmissing-frac "$MIN_FEATURE_NONMISSING_FRAC"
                  --min-feature-unique "$MIN_FEATURE_UNIQUE"
                  --min-feature-nonzero "$MIN_FEATURE_NONZERO"
                )

                if [[ -n "$MIN_EPI_FRACTION" && "$MIN_EPI_FRACTION" != "none" && "$MIN_EPI_FRACTION" != "None" ]]; then
                  ARGS+=(--min-epi-fraction "$MIN_EPI_FRACTION")
                fi

                echo "[SUBMIT] $JOBTAG"
                if [[ "${DRY_RUN:-0}" == "1" ]]; then
                  printf 'sbatch --job-name=%q --cpus-per-task=%q --mem=%q --time=%q -p %q --output=%q --error=%q --export=ALL,CONDA_SH=%q,PY_ENV=%q %q' \
                    "$JOBTAG" "$CPUS_PER_TASK" "$MEM" "$TIME" "$PARTITION" "$LOGOUT" "$LOGERR" "$CONDA_SH" "$PY_ENV" "$WORKER"
                  printf ' %q' "${ARGS[@]}"
                  printf '\n'
                else
                  sbatch \
                    --job-name="$JOBTAG" \
                    --cpus-per-task="$CPUS_PER_TASK" \
                    --mem="$MEM" \
                    --time="$TIME" \
                    -p "$PARTITION" \
                    --output="$LOGOUT" \
                    --error="$LOGERR" \
                    --export=ALL,CONDA_SH="$CONDA_SH",PY_ENV="$PY_ENV" \
                    "$WORKER" "${ARGS[@]}"
                fi
                submit_count=$((submit_count + 1))
              done
            done
          done
        done
      done
    done
  done
done

echo "[DONE] submitted=$submit_count skipped=$skip_count"
