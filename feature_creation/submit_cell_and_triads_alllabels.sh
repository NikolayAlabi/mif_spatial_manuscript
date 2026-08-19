#!/usr/bin/env bash
# Submit the lightweight, prep-based cell-feature and triad feature jobs.
# Includes ALL_NEG in ratios/triads and uses all non-excluded labels as triad centers by default.
# Does NOT submit NNstats or ATHENA jobs.

set -euo pipefail

FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"
SUBMIT_SCRIPT_DIR="${SUBMIT_SCRIPT_DIR:-${LOGDIR}/submitted_feature_jobs}"

PHENO_RUN="${PHENO_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only}"
ARSTATE_RUN="${ARSTATE_RUN:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state}"
CELL_FEATURE_ROOT="${CELL_FEATURE_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed}"

CELL_SCRIPT="${CELL_SCRIPT:-${FEATURE_DIR}/make_cell_feature_table_from_prep.py}"
TRIAD_SCRIPT="${TRIAD_SCRIPT:-${FEATURE_DIR}/make_triad_features_from_prep.py}"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
PY_ENV="${PY_ENV:-cuda6}"

CELL_CPUS="${CELL_CPUS:-1}"
CELL_MEM="${CELL_MEM:-32G}"
CELL_TIME="${CELL_TIME:-08:00:00}"

TRIAD_CPUS="${TRIAD_CPUS:-1}"
TRIAD_MEM="${TRIAD_MEM:-64G}"
TRIAD_TIME="${TRIAD_TIME:-24:00:00}"
TRIAD_THRESHOLD="${TRIAD_THRESHOLD:-100}"
TRIAD_REGIONS="${TRIAD_REGIONS:-All Tumor Stroma}"
# Empty = all non-excluded labels can be centers.
TRIAD_CENTER_REGEX="${TRIAD_CENTER_REGEX:-}"
# Updated exclusion rule: include ALL_NEG; exclude only these labels.
TRIAD_EXCLUDE_LABELS="${TRIAD_EXCLUDE_LABELS:-artifact unresolved mixed_lineage}"
TRIAD_WRITE_LONG="${TRIAD_WRITE_LONG:-0}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOGDIR" "$SUBMIT_SCRIPT_DIR" "$CELL_FEATURE_ROOT"

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

need_file "$CELL_SCRIPT"
need_file "$TRIAD_SCRIPT"
need_file "$CONDA_SH"
need_dir "$PHENO_RUN"
need_dir "$ARSTATE_RUN"

submit_or_print() {
    echo "[SBATCH] $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        sbatch "$@"
    fi
}

write_header() {
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

set +u
source "${CONDA_SH}"
conda activate "${PY_ENV}"
set -u

echo "[INFO] Activated PY_ENV=${PY_ENV}"
echo "[INFO] which python: \$(command -v python)"
python --version

EOF_JOB
}

make_cell_job() {
    local run_label="$1"
    local prep_root="$2"
    local outdir="$3"
    local outfile="$4"
    local include_panels="$5"
    local job_name="cellfeat_${run_label}_allneg"
    local job_file="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"

    mkdir -p "$outdir"
    write_header "$job_file" "$job_name" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME"

    cat >> "$job_file" <<EOF_JOB
python -u "${CELL_SCRIPT}" \\
  --prep-roots "${prep_root}" \\
  --outdir "${outdir}" \\
  --outfile "${outfile}" \\
  --include-panels ${include_panels} \\
  --exclude-panels MY
EOF_JOB

    chmod +x "$job_file"
    submit_or_print "$job_file"
}

make_triad_job() {
    local run_label="$1"
    local prep_root="$2"
    local outdir="$3"
    local outfile="$4"
    local include_panels="$5"
    local job_name="triads_${run_label}_alllabels"
    local job_file="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"

    mkdir -p "$outdir"
    write_header "$job_file" "$job_name" "$TRIAD_CPUS" "$TRIAD_MEM" "$TRIAD_TIME"

    cat >> "$job_file" <<EOF_JOB
CMD="python -u ${TRIAD_SCRIPT} \
  --prep-roots ${prep_root} \
  --outdir ${outdir} \
  --outfile ${outfile} \
  --include-panels ${include_panels} \
  --exclude-panels MY \
  --threshold ${TRIAD_THRESHOLD} \
  --regions ${TRIAD_REGIONS} \
  --exclude-labels ${TRIAD_EXCLUDE_LABELS}"

if [[ -n "${TRIAD_CENTER_REGEX}" ]]; then
  CMD="\${CMD} --center-regex ${TRIAD_CENTER_REGEX}"
fi
if [[ "${TRIAD_WRITE_LONG}" == "1" ]]; then
  CMD="\${CMD} --write-long"
fi

echo "[RUN] \${CMD}"
eval "\${CMD}"
EOF_JOB

    chmod +x "$job_file"
    submit_or_print "$job_file"
}

echo "[INFO] Submitting cell-feature and triad jobs only"
echo "[INFO] ALL_NEG will be included in ratios and triads"
echo "[INFO] Triad centers: all non-excluded labels unless TRIAD_CENTER_REGEX is set"
echo "[INFO] Triad center regex: ${TRIAD_CENTER_REGEX:-<ALL>}"
echo "[INFO] Triad excluded labels: ${TRIAD_EXCLUDE_LABELS}"

make_cell_job \
  "pheno" \
  "$PHENO_RUN" \
  "${CELL_FEATURE_ROOT}/phenotype_only_from_prep_allneg_ratios" \
  "cell_features_phenotype_only_from_prep_allneg_ratios.csv" \
  "AR BT"

make_cell_job \
  "arstate" \
  "$ARSTATE_RUN" \
  "${CELL_FEATURE_ROOT}/AR_state_from_prep_allneg_ratios" \
  "cell_features_AR_state_from_prep_allneg_ratios.csv" \
  "AR"

make_triad_job \
  "pheno" \
  "$PHENO_RUN" \
  "${CELL_FEATURE_ROOT}/triads_phenotype_only_all_labels" \
  "triad_features_phenotype_only_all_labels.csv" \
  "AR BT"

make_triad_job \
  "arstate" \
  "$ARSTATE_RUN" \
  "${CELL_FEATURE_ROOT}/triads_AR_state_all_labels" \
  "triad_features_AR_state_all_labels.csv" \
  "AR"

echo "[DONE] Submitted cell-feature and triad jobs."
