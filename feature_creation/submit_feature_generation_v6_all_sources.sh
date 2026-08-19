#!/usr/bin/env bash
#SBATCH --job-name=submit_feat_v6
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/submit_feat_v6_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/submit_feat_v6_%j.err

set -euo pipefail

# Final unified feature-generation submitter for the reviewed STROMA/checkpoint recode.
# It handles all five feature sources:
#   phenotype_only, AR_state, AR_checkpoint_state, compartment, compartment_state
# For each source it submits:
#   1) one Slurm array for NNstats + ATHENA over prep chunks
#   2) one cell feature table job
#   3) one triad feature table job

FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
WEIBULL_ROOT="${WEIBULL_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/weibull}"
CELL_FEATURE_ROOT="${CELL_FEATURE_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"
SUBMIT_SCRIPT_DIR="${SUBMIT_SCRIPT_DIR:-${LOGDIR}/submitted_feature_jobs_v6}"

WORKER_SCRIPT="${WORKER_SCRIPT:-${FEATURE_DIR}/run_spatial_feature_chunk_v6.sh}"
NN_SCRIPT="${NN_SCRIPT:-${FEATURE_DIR}/step1_nnstats_from_prep_v3.py}"
ATHENA_SCRIPT="${ATHENA_SCRIPT:-${FEATURE_DIR}/athena_run_v3.py}"
CELL_SCRIPT="${CELL_SCRIPT:-${FEATURE_DIR}/make_cell_feature_table_from_prep.py}"
TRIAD_SCRIPT="${TRIAD_SCRIPT:-${FEATURE_DIR}/make_triad_features_from_prep.py}"

PY_ENV="${PY_ENV:-cuda6}"
ATHENA_ENV="${ATHENA_ENV:-/projects/ovcare/users/nikolay_alabi/packages/athena}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

FEATURE_SOURCES_CSV="${FEATURE_SOURCES_CSV:-phenotype_only,AR_state,AR_checkpoint_state,compartment,compartment_state}"

RUN_SPATIAL="${RUN_SPATIAL:-1}"
RUN_NNSTATS="${RUN_NNSTATS:-1}"
RUN_ATHENA="${RUN_ATHENA:-1}"
RUN_CELL_FEATURES="${RUN_CELL_FEATURES:-1}"
RUN_TRIADS="${RUN_TRIADS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

SPATIAL_MAX_CONCURRENT="${SPATIAL_MAX_CONCURRENT:-24}"
SPATIAL_CPUS="${SPATIAL_CPUS:-1}"
SPATIAL_MEM="${SPATIAL_MEM:-32G}"
SPATIAL_TIME="${SPATIAL_TIME:-48:00:00}"

RUN_ATHENA_INTERACTIONS="${RUN_ATHENA_INTERACTIONS:-1}"
RUN_ATHENA_RIPLEY="${RUN_ATHENA_RIPLEY:-0}"
ATHENA_RADIUS="${ATHENA_RADIUS:-40}"
ATHENA_MIN_CELLS="${ATHENA_MIN_CELLS:-20}"
ATHENA_REGIONS="${ATHENA_REGIONS:-Tumor Stroma All}"
ATHENA_INTERACTION_PERMUTATIONS="${ATHENA_INTERACTION_PERMUTATIONS:-100}"

CELL_CPUS="${CELL_CPUS:-1}"
CELL_MEM="${CELL_MEM:-32G}"
CELL_TIME="${CELL_TIME:-08:00:00}"
CELL_MIN_CELLS_PER_SAMPLE="${CELL_MIN_CELLS_PER_SAMPLE:-1}"

TRIAD_CPUS="${TRIAD_CPUS:-1}"
TRIAD_MEM="${TRIAD_MEM:-64G}"
TRIAD_TIME="${TRIAD_TIME:-24:00:00}"
TRIAD_THRESHOLD="${TRIAD_THRESHOLD:-100}"
TRIAD_REGIONS="${TRIAD_REGIONS:-All Tumor Stroma}"
# Empty = all non-excluded labels as centers.
TRIAD_CENTER_REGEX="${TRIAD_CENTER_REGEX:-}"
TRIAD_EXCLUDE_LABELS="${TRIAD_EXCLUDE_LABELS:-artifact unresolved mixed_lineage}"
TRIAD_ALLOW_CENTER_AS_NEIGHBOR="${TRIAD_ALLOW_CENTER_AS_NEIGHBOR:-0}"
TRIAD_ALLOW_SAME_NEIGHBOR_TYPE="${TRIAD_ALLOW_SAME_NEIGHBOR_TYPE:-0}"
TRIAD_WRITE_LONG="${TRIAD_WRITE_LONG:-0}"
TRIAD_MIN_CELLS_SAMPLE_REGION="${TRIAD_MIN_CELLS_SAMPLE_REGION:-20}"
TRIAD_MIN_CENTER_CELLS="${TRIAD_MIN_CENTER_CELLS:-1}"

mkdir -p "$LOGDIR" "$SUBMIT_SCRIPT_DIR" "$CELL_FEATURE_ROOT"

need_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "[ERROR] Missing dir: $1" >&2; exit 1; }; }

for fp in "$WORKER_SCRIPT" "$NN_SCRIPT" "$ATHENA_SCRIPT" "$CELL_SCRIPT" "$TRIAD_SCRIPT" "$CONDA_SH"; do
    need_file "$fp"
done

sanitize() {
    echo "$1" | tr -c 'A-Za-z0-9_' '_' | sed 's/_\+/_/g' | sed 's/^_//;s/_$//' | cut -c1-90
}

submit_or_echo() {
    echo "[SBATCH] $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        sbatch "$@"
    fi
}

build_manifest() {
    local run_root="$1"
    local manifest="$2"

    need_dir "$run_root"
    mkdir -p "$(dirname "$manifest")"
    : > "$manifest"

    find "$run_root" -type f -name "1NN_prep.tsv" | sort | while read -r prep; do
        local chunk_dir panel cohort dataset chunk tissue
        chunk_dir="$(dirname "$prep")"
        panel="$(basename "$(dirname "$chunk_dir")")"
        cohort="$(basename "$(dirname "$(dirname "$chunk_dir")")")"
        dataset="$(basename "$(dirname "$(dirname "$(dirname "$chunk_dir")")")")"
        chunk="$(basename "$chunk_dir")"
        tissue="${chunk_dir}/tissue_prep.tsv"

        if [[ ! -f "$tissue" ]]; then
            echo "[WARN] Missing tissue_prep.tsv for $chunk_dir, skipping" >&2
            continue
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$dataset" "$cohort" "$panel" "$chunk" "$prep" "$tissue" >> "$manifest"
    done

    local n
    n=$(wc -l < "$manifest")
    echo "[INFO] Manifest: $manifest"
    echo "[INFO] Manifest rows: $n"
    if [[ "$n" -lt 1 ]]; then
        echo "[ERROR] Empty manifest for $run_root" >&2
        exit 1
    fi
}

write_conda_job_header() {
    local job_script="$1" job_name="$2" cpus="$3" mem="$4" time_lim="$5"
    cat > "$job_script" <<EOF_JOB
#!/usr/bin/env bash
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

echo "[INFO] python=\$(command -v python)"
python - <<'PY'
import sys
print('[INFO] python_version=' + sys.version.replace('\\n', ' '))
PY

EOF_JOB
}

submit_spatial_array() {
    local source="$1" manifest="$2"
    local n job_name
    n=$(wc -l < "$manifest")
    job_name="$(sanitize "spatial_${source}")"

    if [[ "$RUN_SPATIAL" != "1" ]]; then
        echo "[SKIP] spatial array disabled for $source"
        return 0
    fi

    submit_or_echo \
        --job-name="$job_name" \
        --array="1-${n}%${SPATIAL_MAX_CONCURRENT}" \
        --cpus-per-task="$SPATIAL_CPUS" \
        --mem="$SPATIAL_MEM" \
        --time="$SPATIAL_TIME" \
        --output="${LOGDIR}/${job_name}_%A_%a.out" \
        --error="${LOGDIR}/${job_name}_%A_%a.err" \
        --export="ALL,MANIFEST=${manifest},FEATURE_DIR=${FEATURE_DIR},NN_SCRIPT=${NN_SCRIPT},ATHENA_SCRIPT=${ATHENA_SCRIPT},PY_ENV=${PY_ENV},ATHENA_ENV=${ATHENA_ENV},CONDA_SH=${CONDA_SH},RUN_NNSTATS=${RUN_NNSTATS},RUN_ATHENA=${RUN_ATHENA},RUN_ATHENA_INTERACTIONS=${RUN_ATHENA_INTERACTIONS},RUN_ATHENA_RIPLEY=${RUN_ATHENA_RIPLEY},ATHENA_RADIUS=${ATHENA_RADIUS},ATHENA_MIN_CELLS=${ATHENA_MIN_CELLS},ATHENA_REGIONS=${ATHENA_REGIONS},ATHENA_INTERACTION_PERMUTATIONS=${ATHENA_INTERACTION_PERMUTATIONS},MASTER_LOGDIR=${LOGDIR},SKIP_EXISTING=${SKIP_EXISTING}" \
        "$WORKER_SCRIPT"
}

submit_cell_features() {
    local source="$1" run_root="$2" include_panels="$3" outdir="$4" outfile="$5"
    local job_name job_script
    job_name="$(sanitize "cellfeat_${source}")"
    job_script="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"

    if [[ "$RUN_CELL_FEATURES" != "1" ]]; then
        echo "[SKIP] cell features disabled for $source"
        return 0
    fi

    mkdir -p "$outdir"
    write_conda_job_header "$job_script" "$job_name" "$CELL_CPUS" "$CELL_MEM" "$CELL_TIME"
    cat >> "$job_script" <<EOF_JOB
python -u "${CELL_SCRIPT}" \
  --prep-roots "${run_root}" \
  --outdir "${outdir}" \
  --outfile "${outfile}" \
  --include-panels ${include_panels} \
  --exclude-panels MY \
  --min-cells-per-sample "${CELL_MIN_CELLS_PER_SAMPLE}"
EOF_JOB
    chmod +x "$job_script"
    submit_or_echo "$job_script"
}

submit_triads() {
    local source="$1" run_root="$2" include_panels="$3" outdir="$4" outfile="$5"
    local job_name job_script
    job_name="$(sanitize "triads_${source}")"
    job_script="${SUBMIT_SCRIPT_DIR}/${job_name}.sh"

    if [[ "$RUN_TRIADS" != "1" ]]; then
        echo "[SKIP] triads disabled for $source"
        return 0
    fi

    mkdir -p "$outdir"
    write_conda_job_header "$job_script" "$job_name" "$TRIAD_CPUS" "$TRIAD_MEM" "$TRIAD_TIME"
    cat >> "$job_script" <<EOF_JOB
TRIAD_CMD=(
  python -u "${TRIAD_SCRIPT}"
  --prep-roots "${run_root}"
  --outdir "${outdir}"
  --outfile "${outfile}"
  --include-panels ${include_panels}
  --exclude-panels MY
  --threshold "${TRIAD_THRESHOLD}"
  --regions ${TRIAD_REGIONS}
  --exclude-labels ${TRIAD_EXCLUDE_LABELS}
  --min-cells-sample-region "${TRIAD_MIN_CELLS_SAMPLE_REGION}"
  --min-center-cells "${TRIAD_MIN_CENTER_CELLS}"
)
EOF_JOB
    if [[ -n "$TRIAD_CENTER_REGEX" ]]; then
        cat >> "$job_script" <<EOF_JOB
TRIAD_CMD+=(--center-regex "${TRIAD_CENTER_REGEX}")
EOF_JOB
    fi
    if [[ "$TRIAD_ALLOW_CENTER_AS_NEIGHBOR" == "1" ]]; then
        cat >> "$job_script" <<'EOF_JOB'
TRIAD_CMD+=(--allow-center-as-neighbor)
EOF_JOB
    fi
    if [[ "$TRIAD_ALLOW_SAME_NEIGHBOR_TYPE" == "1" ]]; then
        cat >> "$job_script" <<'EOF_JOB'
TRIAD_CMD+=(--allow-same-neighbor-type)
EOF_JOB
    fi
    if [[ "$TRIAD_WRITE_LONG" == "1" ]]; then
        cat >> "$job_script" <<'EOF_JOB'
TRIAD_CMD+=(--write-long)
EOF_JOB
    fi
    cat >> "$job_script" <<'EOF_JOB'
printf '[INFO] TRIAD_CMD:'
printf ' %q' "${TRIAD_CMD[@]}"
printf '\n'
"${TRIAD_CMD[@]}"
EOF_JOB
    chmod +x "$job_script"
    submit_or_echo "$job_script"
}

# Format:
# source|run_suffix|include_panels|cell_subdir|cell_file|triad_subdir|triad_file
ALL_SOURCES=(
  "phenotype_only|run_reviewed_phenotype_only|AR BT|phenotype_only|cell_features_phenotype_only.csv|triads_phenotype_only|triad_features_phenotype_only.csv"
  "AR_state|run_reviewed_AR_state|AR|AR_state|cell_features_AR_state.csv|triads_AR_state|triad_features_AR_state.csv"
  "AR_checkpoint_state|run_reviewed_AR_checkpoint_state|AR|AR_checkpoint_state|cell_features_AR_checkpoint_state.csv|triads_AR_checkpoint_state|triad_features_AR_checkpoint_state.csv"
  "compartment|run_reviewed_compartment|AR BT|compartment|cell_features_compartment.csv|triads_compartment|triad_features_compartment.csv"
  "compartment_state|run_reviewed_compartment_state|AR|compartment_state|cell_features_compartment_state.csv|triads_compartment_state|triad_features_compartment_state.csv"
)

IFS=',' read -r -a REQUESTED_SOURCES <<< "$FEATURE_SOURCES_CSV"
requested_contains() {
    local target="$1"
    local x
    for x in "${REQUESTED_SOURCES[@]}"; do
        x="$(echo "$x" | xargs)"
        [[ "$x" == "$target" ]] && return 0
    done
    return 1
}

cat <<INFO
[INFO] FEATURE_SOURCES_CSV=${FEATURE_SOURCES_CSV}
[INFO] RUN_SPATIAL=${RUN_SPATIAL}; RUN_NNSTATS=${RUN_NNSTATS}; RUN_ATHENA=${RUN_ATHENA}
[INFO] RUN_CELL_FEATURES=${RUN_CELL_FEATURES}; RUN_TRIADS=${RUN_TRIADS}
[INFO] SKIP_EXISTING=${SKIP_EXISTING}; DRY_RUN=${DRY_RUN}
[INFO] WEIBULL_ROOT=${WEIBULL_ROOT}
[INFO] CELL_FEATURE_ROOT=${CELL_FEATURE_ROOT}
[INFO] LOGDIR=${LOGDIR}
INFO

for row in "${ALL_SOURCES[@]}"; do
    IFS='|' read -r SOURCE RUN_SUFFIX INCLUDE_PANELS CELL_SUBDIR CELL_FILE TRIAD_SUBDIR TRIAD_FILE <<< "$row"
    if ! requested_contains "$SOURCE"; then
        continue
    fi

    RUN_ROOT="${WEIBULL_ROOT}/${RUN_SUFFIX}"
    MANIFEST="${RUN_ROOT}/spatial_feature_chunk_manifest.tsv"
    CELL_OUTDIR="${CELL_FEATURE_ROOT}/${CELL_SUBDIR}"
    TRIAD_OUTDIR="${CELL_FEATURE_ROOT}/${TRIAD_SUBDIR}"

    echo ""
    echo "======================================================================"
    echo "[SOURCE] ${SOURCE}"
    echo "[RUN_ROOT] ${RUN_ROOT}"
    echo "======================================================================"

    need_dir "$RUN_ROOT"
    build_manifest "$RUN_ROOT" "$MANIFEST"
    submit_spatial_array "$SOURCE" "$MANIFEST"
    submit_cell_features "$SOURCE" "$RUN_ROOT" "$INCLUDE_PANELS" "$CELL_OUTDIR" "$CELL_FILE"
    submit_triads "$SOURCE" "$RUN_ROOT" "$INCLUDE_PANELS" "$TRIAD_OUTDIR" "$TRIAD_FILE"
done

echo ""
echo "[DONE] Submitted feature-generation jobs for selected sources."
echo "Check with: squeue -u $USER"
echo "After completion, example checks:"
echo "  find ${WEIBULL_ROOT}/run_reviewed_phenotype_only -name NNstats.tsv | wc -l"
echo "  find ${WEIBULL_ROOT}/run_reviewed_phenotype_only -name athena_features.csv | wc -l"
echo "  ls -lh ${CELL_FEATURE_ROOT}/phenotype_only/cell_features_phenotype_only.csv"
echo "  ls -lh ${CELL_FEATURE_ROOT}/triads_phenotype_only/triad_features_phenotype_only.csv"
