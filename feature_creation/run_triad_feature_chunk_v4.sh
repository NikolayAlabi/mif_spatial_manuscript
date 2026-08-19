#!/usr/bin/env bash
#SBATCH --job-name=triad_chunk
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

# Required env vars:
#   MANIFEST, TRIAD_SCRIPT
# Optional env vars:
#   CONDA_SH, PY_ENV, TRIAD_THRESHOLD, TRIAD_REGIONS, TRIAD_CENTER_REGEX,
#   TRIAD_EXCLUDE_LABELS, TRIAD_ALLOW_CENTER_AS_NEIGHBOR, TRIAD_WRITE_LONG,
#   TRIAD_MIN_CELLS_SAMPLE_REGION, TRIAD_MIN_CENTER_CELLS, TRIAD_OUTFILE

MANIFEST="${MANIFEST:?MANIFEST is required}"
TRIAD_SCRIPT="${TRIAD_SCRIPT:?TRIAD_SCRIPT is required}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
PY_ENV="${PY_ENV:-cuda6}"

TRIAD_THRESHOLD="${TRIAD_THRESHOLD:-100}"
TRIAD_REGIONS="${TRIAD_REGIONS:-All Tumor Stroma}"
# Empty means all non-excluded labels can be centres.
TRIAD_CENTER_REGEX="${TRIAD_CENTER_REGEX:-}"
# By default include ALL_NEG and any other reviewed primary labels.
TRIAD_EXCLUDE_LABELS="${TRIAD_EXCLUDE_LABELS:-artifact unresolved mixed_lineage}"
TRIAD_ALLOW_CENTER_AS_NEIGHBOR="${TRIAD_ALLOW_CENTER_AS_NEIGHBOR:-0}"
TRIAD_WRITE_LONG="${TRIAD_WRITE_LONG:-0}"
TRIAD_MIN_CELLS_SAMPLE_REGION="${TRIAD_MIN_CELLS_SAMPLE_REGION:-20}"
TRIAD_MIN_CENTER_CELLS="${TRIAD_MIN_CENTER_CELLS:-1}"
TRIAD_OUTFILE="${TRIAD_OUTFILE:-triad_features_chunk.csv}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] Manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$TRIAD_SCRIPT" ]]; then
    echo "[ERROR] Triad script not found: $TRIAD_SCRIPT" >&2
    exit 1
fi

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "[ERROR] No manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r DATASET COHORT PANEL CHUNK PREP_FILE TISSUE_FILE <<< "$line"
CHUNK_DIR="$(dirname "$PREP_FILE")"
OUTDIR="${CHUNK_DIR}/triad_features"
LOGDIR="${CHUNK_DIR}/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

job_tag="${DATASET}_${COHORT}_${PANEL}_${CHUNK}"

echo "[INFO] job_tag=${job_tag}"
echo "[INFO] PREP_FILE=${PREP_FILE}"
echo "[INFO] OUTDIR=${OUTDIR}"
echo "[INFO] PANEL=${PANEL}"
echo "[INFO] threshold=${TRIAD_THRESHOLD}"
echo "[INFO] regions=${TRIAD_REGIONS}"
echo "[INFO] center_regex=${TRIAD_CENTER_REGEX:-ALL_NON_EXCLUDED_LABELS}"
echo "[INFO] exclude_labels=${TRIAD_EXCLUDE_LABELS}"
echo "[INFO] allow_center_as_neighbor=${TRIAD_ALLOW_CENTER_AS_NEIGHBOR}"

# Safe conda activation with set -u disabled to avoid PS1/unbound-variable errors.
if [[ -n "$PY_ENV" ]]; then
    set +u
    source "$CONDA_SH"
    conda activate "$PY_ENV"
    set -u
    echo "[INFO] Activated PY_ENV=${PY_ENV}"
    echo "[INFO] python=$(command -v python)"
fi

args=(
    --prep-files "$PREP_FILE"
    --outdir "$OUTDIR"
    --outfile "$TRIAD_OUTFILE"
    --include-panels "$PANEL"
    --exclude-panels MY
    --threshold "$TRIAD_THRESHOLD"
    --regions $TRIAD_REGIONS
    --exclude-labels $TRIAD_EXCLUDE_LABELS
    --min-cells-sample-region "$TRIAD_MIN_CELLS_SAMPLE_REGION"
    --min-center-cells "$TRIAD_MIN_CENTER_CELLS"
)

if [[ -n "$TRIAD_CENTER_REGEX" ]]; then
    args+=(--center-regex "$TRIAD_CENTER_REGEX")
fi
if [[ "$TRIAD_ALLOW_CENTER_AS_NEIGHBOR" == "1" ]]; then
    args+=(--allow-center-as-neighbor)
fi
if [[ "$TRIAD_WRITE_LONG" == "1" ]]; then
    args+=(--write-long)
fi

python -u "$TRIAD_SCRIPT" "${args[@]}" \
    > "${LOGDIR}/triad_${job_tag}.out" \
    2> "${LOGDIR}/triad_${job_tag}.err"

if [[ -n "$PY_ENV" ]]; then
    set +u
    conda deactivate || true
    set -u
fi

echo "[DONE] ${job_tag}"
echo "[DONE] ${OUTDIR}/${TRIAD_OUTFILE}"
