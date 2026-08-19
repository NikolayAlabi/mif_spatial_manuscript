#!/usr/bin/env bash
# Submit all reviewed feature-generation jobs for the manuscript pipeline.
#
# This submits, for BOTH phenotype-only and AR-state prep roots:
#   1) NNstats from 1NN_prep.tsv
#   2) ATHENA v3 from 1NN_prep.tsv, with interactions ON and Ripley OFF by default
#   3) Cell count/proportion/ratio features from all 1NN_prep.tsv files
#   4) Centered triad/motif features from all 1NN_prep.tsv files
#
# Run from login node:
#   bash submit_all_feature_generation_v3.sh
#
# Common overrides:
#   FEATURE_DIR=/path/to/scripts \
#   PHENO_RUN=/path/to/run_reviewed_phenotype_only \
#   ARSTATE_RUN=/path/to/run_reviewed_AR_state \
#   MAX_CONCURRENT=24 \
#   bash submit_all_feature_generation_v3.sh

set -euo pipefail

# ---------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------
FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"
SUBMIT_SCRIPT_DIR="${SUBMIT_SCRIPT_DIR:-${LOGDIR}/submitted_feature_jobs}"

PHENO_RUN="${PHENO_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only}"
ARSTATE_RUN="${ARSTATE_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state}"

# Output roots for non-spatial summary feature tables
CELL_FEATURE_ROOT="${CELL_FEATURE_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed}"

# ---------------------------------------------------------------------
# Script paths
# ---------------------------------------------------------------------
MANIFEST_SCRIPT="${MANIFEST_SCRIPT:-${FEATURE_DIR}/build_spatial_feature_manifest_v3.sh}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${FEATURE_DIR}/run_spatial_feature_chunk_v3.sh}"
CELL_SCRIPT="${CELL_SCRIPT:-${FEATURE_DIR}/make_cell_feature_table_from_prep.py}"
TRIAD_SCRIPT="${TRIAD_SCRIPT:-${FEATURE_DIR}/make_triad_features_from_prep.py}"
NN_SCRIPT="${NN_SCRIPT:-${FEATURE_DIR}/step1_nnstats_from_prep_v3.py}"
ATHENA_SCRIPT="${ATHENA_SCRIPT:-${FEATURE_DIR}/athena_run_v3.py}"

# ---------------------------------------------------------------------
# Conda / runtime settings
# ---------------------------------------------------------------------
# PY_ENV can be blank if normal python on the node has pandas/scipy/sklearn.
PY_ENV="${PY_ENV:-}"
ATHENA_ENV="${ATHENA_ENV:-/projects/ovcare/users/nikolay_alabi/packages/athena}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

# Spatial array resource settings
MAX_CONCURRENT="${MAX_CONCURRENT:-24}"
SPATIAL_CPUS="${SPATIAL_CPUS:-1}"
SPATIAL_MEM="${SPATIAL_MEM:-32G}"
SPATIAL_TIME="${SPATIAL_TIME:-48:00:00}"

# ATHENA settings
RUN_NNSTATS="${RUN_NNSTATS:-1}"
RUN_ATHENA="${RUN_ATHENA:-1}"
RUN_ATHENA_INTERACTIONS="${RUN_ATHENA_INTERACTIONS:-1}"
RUN_ATHENA_RIPLEY="${RUN_ATHENA_RIPLEY:-0}"
ATHENA_RADIUS="${ATHENA_RADIUS:-40}"
ATHENA_MIN_CELLS="${ATHENA_MIN_CELLS:-20}"
ATHENA_REGIONS="${ATHENA_REGIONS:-Tumor Stroma All}"
ATHENA_INTERACTION_PERMUTATIONS="${ATHENA_INTERACTION_PERMUTATIONS:-100}"

# Cell-feature job resource settings
CELL_CPUS="${CELL_CPUS:-1}"
CELL_MEM="${CELL_MEM:-32G}"
CELL_TIME="${CELL_TIME:-08:00:00}"

# Triad job resource settings
TRIAD_CPUS="${TRIAD_CPUS:-1}"
TRIAD_MEM="${TRIAD_MEM:-64G}"
TRIAD_TIME="${TRIAD_TIME:-24:00:00}"
TRIAD_THRESHOLD="${TRIAD_THRESHOLD:-100}"
TRIAD_REGIONS="${TRIAD_REGIONS:-All Tumor Stroma}"
TRIAD_CENTER_REGEX="${TRIAD_CENTER_REGEX:-macrophage}"
TRIAD_WRITE_LONG="${TRIAD_WRITE_LONG:-0}"

# Set to 1 to only print what would be submitted.
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOGDIR" "$SUBMIT_SCRIPT_DIR" "$CELL_FEATURE_ROOT"

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------
need_file() {
    local fp="$1"
    if [[ ! -f "$fp" ]]; then
        echo "[ERROR] Missing required file: $fp" >&2
        exit 1
    fi
}

need_dir() {
    local dp="$1"
    if [[ ! -d "$dp" ]]; then
        echo "[ERROR] Missing required directory: $dp" >&2
        exit 1
    fi
}

need_file "$MANIFEST_SCRIPT"
need_file "$WORKER_SCRIPT"
need_file "$CELL_SCRIPT"
need_file "$TRIAD_SCRIPT"
need_file "$NN_SCRIPT"
need_file "$ATHENA_SCRIPT"
need_dir "$PHENO_RUN"
need_dir "$ARSTATE_RUN"

run_cmd() {
    echo "[CMD] $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

submit_sbatch() {
    echo "[SBATCH] $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        sbatch "$@"
    fi
}

# ---------------------------------------------------------------------
# Manifest + spatial array submission
# ---------------------------------------------------------------------
build_manifest_for_run() {
    local run_root="$1"
    local manifest="$2"
    echo ""
    echo "======================================================================"
    echo "Building manifest"
    echo "RUN_ROOT:  $run_root"
    echo "MANIFEST:  $manifest"
    echo "======================================================================"

    if [[ "$DRY_RUN" != "1" ]]; then
        RUN_ROOT="$run_root" MANIFEST="$manifest" LOGDIR="$LOGDIR" bash "$MANIFEST_SCRIPT"
        local n
        n=$(wc -l < "$manifest")
        echo "[INFO] Manifest rows: $n"
        if [[ "$n" -lt 1 ]]; then
            echo "[ERROR] Manifest has no rows: $manifest" >&2
            exit 1
        fi
    else
        echo "[DRY_RUN] Would run manifest script."
    fi
}

submit_spatial_array_for_run() {
    local run_label="$1"
    local manifest="$2"
    local job_name="spatial_${run_label}"

    local n="1"
    if [[ "$DRY_RUN" != "1" ]]; then
        n=$(wc -l < "$manifest")
    fi

    echo ""
    echo "======================================================================"
    echo "Submitting spatial array: $run_label"
    echo "Manifest: $manifest"
    echo "Tasks:    $n"
    echo "Runs:     NNstats=${RUN_NNSTATS}, ATHENA=${RUN_ATHENA}, interactions=${RUN_ATHENA_INTERACTIONS}, ripley=${RUN_ATHENA_RIPLEY}"
    echo "======================================================================"

    submit_sbatch \
        --job-name="$job_name" \
        --array="1-${n}%${MAX_CONCURRENT}" \
        --cpus-per-task="$SPATIAL_CPUS" \
        --mem="$SPATIAL_MEM" \
        --time="$SPATIAL_TIME" \
        --output="${LOGDIR}/${job_name}_%A_%a.out" \
        --error="${LOGDIR}/${job_name}_%A_%a.err" \
        --export="ALL,MANIFEST=${manifest},FEATURE_DIR=${FEATURE_DIR},NN_SCRIPT=${NN_SCRIPT},ATHENA_SCRIPT=${ATHENA_SCRIPT},PY_ENV=${PY_ENV},ATHENA_ENV=${ATHENA_ENV},RUN_NNSTATS=${RUN_NNSTATS},RUN_ATHENA=${RUN_ATHENA},RUN_ATHENA_INTERACTIONS=${RUN_ATHENA_INTERACTIONS},RUN_ATHENA_RIPLEY=${RUN_ATHENA_RIPLEY},ATHENA_RADIUS=${ATHENA_RADIUS},ATHENA_MIN_CELLS=${ATHENA_MIN_CELLS},ATHENA_REGIONS=${ATHENA_REGIONS},ATHENA_INTERACTION_PERMUTATIONS=${ATHENA_INTERACTION_PERMUTATIONS},MASTER_LOGDIR=${LOGDIR}" \
        "$WORKER_SCRIPT"
}

# ---------------------------------------------------------------------
# Single-job feature table submissions
# ---------------------------------------------------------------------
write_python_job_header() {
    local fp="$1"
    local job_name="$2"
    local cpus="$3"
    local mem="$4"
    local time_lim="$5"

    cat > "$fp" <<EOF_JOB
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_lim}
#SBATCH --output=${LOGDIR}/${job_name}_%j.out
#SBATCH --error=${LOGDIR}/${job_name}_%j.err

set -euo pipefail

FEATURE_DIR="${FEATURE_DIR}"
PY_ENV="${PY_ENV}"
CONDA_SH="${CONDA_SH}"

if [[ -n "\${PY_ENV}" ]]; then
    set +u
    source "\${CONDA_SH}"
    conda activate "\${PY_ENV}"
    set -u
fi

EOF_JOB
}

submit_cell_feature_job() {
    local run_label="$1"
    local run_root="$2"
    local include_panels="$3"
    local outdir="$4"
    local outfile="$5"
    local job_name="cellfeat_${run_label}"
    local job_script="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"

    write_python_job_header "$job_script" "$job_name" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME"

    cat >> "$job_script" <<EOF_JOB
python -u "${CELL_SCRIPT}" \
    --prep-roots "${run_root}" \
    --outdir "${outdir}" \
    --outfile "${outfile}" \
    --include-panels ${include_panels} \
    --exclude-panels MY \
    --exclude-all-neg-from-ratios

EOF_JOB

    chmod +x "$job_script"
    echo ""
    echo "======================================================================"
    echo "Submitting cell feature job: $run_label"
    echo "Script: $job_script"
    echo "======================================================================"
    submit_sbatch "$job_script"
}

submit_triad_job() {
    local run_label="$1"
    local run_root="$2"
    local include_panels="$3"
    local outdir="$4"
    local outfile="$5"
    local job_name="triads_${run_label}"
    local job_script="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"
    local long_flag=""

    if [[ "$TRIAD_WRITE_LONG" == "1" ]]; then
        long_flag="--write-long"
    fi

    write_python_job_header "$job_script" "$job_name" "$TRIAD_CPUS" "$TRIAD_MEM" "$TRIAD_TIME"

    cat >> "$job_script" <<EOF_JOB
python -u "${TRIAD_SCRIPT}" \
    --prep-roots "${run_root}" \
    --outdir "${outdir}" \
    --outfile "${outfile}" \
    --include-panels ${include_panels} \
    --exclude-panels MY \
    --center-regex "${TRIAD_CENTER_REGEX}" \
    --threshold "${TRIAD_THRESHOLD}" \
    --regions ${TRIAD_REGIONS} \
    ${long_flag}

EOF_JOB

    chmod +x "$job_script"
    echo ""
    echo "======================================================================"
    echo "Submitting triad job: $run_label"
    echo "Script: $job_script"
    echo "======================================================================"
    submit_sbatch "$job_script"
}

# ---------------------------------------------------------------------
# Main submission
# ---------------------------------------------------------------------
echo "======================================================================"
echo "Submitting all feature generation jobs"
echo "FEATURE_DIR:  $FEATURE_DIR"
echo "PHENO_RUN:    $PHENO_RUN"
echo "ARSTATE_RUN:  $ARSTATE_RUN"
echo "LOGDIR:       $LOGDIR"
echo "Interactions: $RUN_ATHENA_INTERACTIONS"
echo "Ripley:       $RUN_ATHENA_RIPLEY"
echo "======================================================================"

PHENO_MANIFEST="${PHENO_RUN}/spatial_feature_chunk_manifest.tsv"
ARSTATE_MANIFEST="${ARSTATE_RUN}/spatial_feature_chunk_manifest.tsv"

build_manifest_for_run "$PHENO_RUN" "$PHENO_MANIFEST"
build_manifest_for_run "$ARSTATE_RUN" "$ARSTATE_MANIFEST"

submit_spatial_array_for_run "pheno" "$PHENO_MANIFEST"
submit_spatial_array_for_run "arstate" "$ARSTATE_MANIFEST"

submit_cell_feature_job \
    "pheno" \
    "$PHENO_RUN" \
    "AR BT" \
    "${CELL_FEATURE_ROOT}/phenotype_only_from_prep" \
    "cell_features_phenotype_only_from_prep.csv"

submit_cell_feature_job \
    "arstate" \
    "$ARSTATE_RUN" \
    "AR" \
    "${CELL_FEATURE_ROOT}/AR_state_from_prep" \
    "cell_features_AR_state_from_prep.csv"

submit_triad_job \
    "pheno" \
    "$PHENO_RUN" \
    "AR BT" \
    "${CELL_FEATURE_ROOT}/triads_phenotype_only" \
    "triad_features_phenotype_only.csv"

submit_triad_job \
    "arstate" \
    "$ARSTATE_RUN" \
    "AR" \
    "${CELL_FEATURE_ROOT}/triads_AR_state" \
    "triad_features_AR_state.csv"

echo ""
echo "======================================================================"
echo "Submitted all jobs. Check status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check outputs with:"
echo "  find \"$PHENO_RUN\" -name 'NNstats.tsv' | wc -l"
echo "  find \"$PHENO_RUN\" -name 'athena_features.csv' | wc -l"
echo "  find \"$ARSTATE_RUN\" -name 'NNstats.tsv' | wc -l"
echo "  find \"$ARSTATE_RUN\" -name 'athena_features.csv' | wc -l"
echo "  ls -lh \"${CELL_FEATURE_ROOT}/phenotype_only_from_prep\""
echo "  ls -lh \"${CELL_FEATURE_ROOT}/AR_state_from_prep\""
echo "  ls -lh \"${CELL_FEATURE_ROOT}/triads_phenotype_only\""
echo "  ls -lh \"${CELL_FEATURE_ROOT}/triads_AR_state\""
echo "======================================================================"
