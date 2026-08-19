#!/usr/bin/env bash
# Submit all-label triad feature generation as one Slurm array task per prep chunk,
# then submit dependent concat jobs for phenotype-only and AR-state runs.

set -euo pipefail

FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"
CELL_FEATURE_ROOT="${CELL_FEATURE_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed}"

PHENO_RUN="${PHENO_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only}"
ARSTATE_RUN="${ARSTATE_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state}"

TRIAD_SCRIPT="${TRIAD_SCRIPT:-${FEATURE_DIR}/make_triad_features_from_prep.py}"
TRIAD_WORKER="${TRIAD_WORKER:-${FEATURE_DIR}/run_triad_feature_chunk_v4.sh}"
CONCAT_SCRIPT="${CONCAT_SCRIPT:-${FEATURE_DIR}/concat_chunk_feature_tables_v3.py}"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
PY_ENV="${PY_ENV:-cuda6}"

TRIAD_MAX_CONCURRENT="${TRIAD_MAX_CONCURRENT:-24}"
TRIAD_CPUS="${TRIAD_CPUS:-1}"
TRIAD_MEM="${TRIAD_MEM:-24G}"
TRIAD_TIME="${TRIAD_TIME:-24:00:00}"
TRIAD_THRESHOLD="${TRIAD_THRESHOLD:-100}"
TRIAD_REGIONS="${TRIAD_REGIONS:-All Tumor Stroma}"
# Empty means all non-excluded labels are centers. This is the default for v4.
TRIAD_CENTER_REGEX="${TRIAD_CENTER_REGEX:-}"
# Include ALL_NEG. Exclude only labels that should not be biological participants.
TRIAD_EXCLUDE_LABELS="${TRIAD_EXCLUDE_LABELS:-artifact unresolved mixed_lineage}"
TRIAD_ALLOW_CENTER_AS_NEIGHBOR="${TRIAD_ALLOW_CENTER_AS_NEIGHBOR:-0}"
TRIAD_WRITE_LONG="${TRIAD_WRITE_LONG:-0}"
TRIAD_MIN_CELLS_SAMPLE_REGION="${TRIAD_MIN_CELLS_SAMPLE_REGION:-20}"
TRIAD_MIN_CENTER_CELLS="${TRIAD_MIN_CENTER_CELLS:-1}"

CONCAT_MEM="${CONCAT_MEM:-32G}"
CONCAT_TIME="${CONCAT_TIME:-04:00:00}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOGDIR" "$CELL_FEATURE_ROOT"

need_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "[ERROR] Missing dir: $1" >&2; exit 1; }; }

need_file "$TRIAD_SCRIPT"
need_file "$TRIAD_WORKER"
need_file "$CONCAT_SCRIPT"
need_dir "$PHENO_RUN"
need_dir "$ARSTATE_RUN"

submit_array() {
    local label="$1"
    local run_root="$2"
    local manifest="${run_root}/spatial_feature_chunk_manifest.tsv"
    local job_name="triad_${label}_alllabels"

    need_file "$manifest"
    local n
    n=$(wc -l < "$manifest")
    if [[ "$n" -lt 1 ]]; then
        echo "[ERROR] Empty manifest: $manifest" >&2
        exit 1
    fi

    echo "[INFO] Submitting ${job_name}: ${n} chunks, max concurrent ${TRIAD_MAX_CONCURRENT}"
    echo "[INFO] centers: ${TRIAD_CENTER_REGEX:-ALL_NON_EXCLUDED_LABELS}"
    echo "[INFO] exclude labels: ${TRIAD_EXCLUDE_LABELS}"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] sbatch array for $manifest"
        echo "DRYRUN_${label}"
        return 0
    fi

    sbatch --parsable \
        --job-name="$job_name" \
        --array="1-${n}%${TRIAD_MAX_CONCURRENT}" \
        --cpus-per-task="$TRIAD_CPUS" \
        --mem="$TRIAD_MEM" \
        --time="$TRIAD_TIME" \
        --output="${LOGDIR}/${job_name}_%A_%a.out" \
        --error="${LOGDIR}/${job_name}_%A_%a.err" \
        --export="ALL,MANIFEST=${manifest},TRIAD_SCRIPT=${TRIAD_SCRIPT},CONDA_SH=${CONDA_SH},PY_ENV=${PY_ENV},TRIAD_THRESHOLD=${TRIAD_THRESHOLD},TRIAD_REGIONS=${TRIAD_REGIONS},TRIAD_CENTER_REGEX=${TRIAD_CENTER_REGEX},TRIAD_EXCLUDE_LABELS=${TRIAD_EXCLUDE_LABELS},TRIAD_ALLOW_CENTER_AS_NEIGHBOR=${TRIAD_ALLOW_CENTER_AS_NEIGHBOR},TRIAD_WRITE_LONG=${TRIAD_WRITE_LONG},TRIAD_MIN_CELLS_SAMPLE_REGION=${TRIAD_MIN_CELLS_SAMPLE_REGION},TRIAD_MIN_CENTER_CELLS=${TRIAD_MIN_CENTER_CELLS}" \
        "$TRIAD_WORKER"
}

submit_concat() {
    local label="$1"
    local run_root="$2"
    local dep_jobid="$3"
    local outdir="$4"
    local outfile="$5"
    local job_name="concat_triad_${label}_alllabels"
    local job_script="${LOGDIR}/${job_name}.sh"

    mkdir -p "$outdir"

    cat > "$job_script" <<EOF_JOB
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=1
#SBATCH --mem=${CONCAT_MEM}
#SBATCH --time=${CONCAT_TIME}
#SBATCH --output=${LOGDIR}/${job_name}_%j.out
#SBATCH --error=${LOGDIR}/${job_name}_%j.err

set -euo pipefail
set +u
source "${CONDA_SH}"
conda activate "${PY_ENV}"
set -u

echo "[INFO] python=\$(command -v python)"

python -u "${CONCAT_SCRIPT}" \
  --root "${run_root}" \
  --filename "triad_features_chunk.csv" \
  --outdir "${outdir}" \
  --outfile "${outfile}"
EOF_JOB

    chmod +x "$job_script"

    echo "[INFO] Submitting concat job ${job_name} after ${dep_jobid}"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] sbatch concat $job_script"
        return 0
    fi

    sbatch --dependency="afterok:${dep_jobid}" "$job_script"
}

PHENO_ID=$(submit_array "pheno" "$PHENO_RUN")
ARSTATE_ID=$(submit_array "arstate" "$ARSTATE_RUN")

echo "[INFO] Phenotype-only all-label triad array job: ${PHENO_ID}"
echo "[INFO] AR-state all-label triad array job:       ${ARSTATE_ID}"

submit_concat \
  "pheno" \
  "$PHENO_RUN" \
  "$PHENO_ID" \
  "${CELL_FEATURE_ROOT}/triads_phenotype_only_all_labels" \
  "triad_features_phenotype_only_all_labels.csv"

submit_concat \
  "arstate" \
  "$ARSTATE_RUN" \
  "$ARSTATE_ID" \
  "${CELL_FEATURE_ROOT}/triads_AR_state_all_labels" \
  "triad_features_AR_state_all_labels.csv"

echo "[DONE] Submitted all-label parallel triad arrays and dependent concat jobs."
echo "Check with: squeue -u \$USER"
