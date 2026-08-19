#!/bin/bash
#SBATCH --job-name=spatial_feat_v3
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/%x_%A_%a.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/%x_%A_%a.err

set -euo pipefail

# This worker runs per-chunk spatial features from 1NN_prep.tsv:
#   1) NNstats-only Python step (default on)
#   2) ATHENA v3 features (default off unless RUN_ATHENA=1)
#
# Required env:
#   MANIFEST=/path/to/spatial_feature_chunk_manifest.tsv
#
# Useful env:
#   FEATURE_DIR=/projects/.../manuscript/feature_creation
#   PY_ENV=/path/to/conda/env/with/scipy/pandas
#   ATHENA_ENV=/path/to/conda/env/with/athena
#   RUN_NNSTATS=1
#   RUN_ATHENA=0/1
#   RUN_ATHENA_INTERACTIONS=0/1
#   RUN_ATHENA_RIPLEY=0/1
#   ATHENA_RADIUS=40
#   ATHENA_MIN_CELLS=20
#   ATHENA_REGIONS="Tumor Stroma All"

MANIFEST="${MANIFEST:?MANIFEST env variable is required}"
FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
NN_SCRIPT="${NN_SCRIPT:-${FEATURE_DIR}/step1_nnstats_from_prep_v3.py}"
ATHENA_SCRIPT="${ATHENA_SCRIPT:-${FEATURE_DIR}/athena_run_v3.py}"

RUN_NNSTATS="${RUN_NNSTATS:-1}"
RUN_ATHENA="${RUN_ATHENA:-0}"
RUN_ATHENA_INTERACTIONS="${RUN_ATHENA_INTERACTIONS:-0}"
RUN_ATHENA_RIPLEY="${RUN_ATHENA_RIPLEY:-0}"

PY_ENV="${PY_ENV:-cuda6}"
ATHENA_ENV="${ATHENA_ENV:-/projects/ovcare/users/nikolay_alabi/packages/athena}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

ATHENA_RADIUS="${ATHENA_RADIUS:-40}"
ATHENA_MIN_CELLS="${ATHENA_MIN_CELLS:-20}"
ATHENA_REGIONS="${ATHENA_REGIONS:-Tumor Stroma All}"
ATHENA_INTERACTION_PERMUTATIONS="${ATHENA_INTERACTION_PERMUTATIONS:-100}"

MASTER_LOGDIR="${MASTER_LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"
mkdir -p "$MASTER_LOGDIR"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "No manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r DATASET COHORT PANEL CHUNK PREP_FILE TISSUE_FILE <<< "$line"
CHUNK_DIR="$(dirname "$PREP_FILE")"
CHUNK_LOGDIR="${CHUNK_DIR}/logs"
mkdir -p "$CHUNK_LOGDIR"

job_tag="${DATASET}_${COHORT}_${PANEL}_${CHUNK}"

echo "[INFO] DATASET=${DATASET}"
echo "[INFO] COHORT=${COHORT}"
echo "[INFO] PANEL=${PANEL}"
echo "[INFO] CHUNK=${CHUNK}"
echo "[INFO] PREP_FILE=${PREP_FILE}"
echo "[INFO] TISSUE_FILE=${TISSUE_FILE}"
echo "[INFO] CHUNK_DIR=${CHUNK_DIR}"

# ------------------------------------------------------------
# Conda helpers
# ------------------------------------------------------------
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda setup file not found: $CONDA_SH" >&2
    exit 1
fi

# Source conda with nounset disabled to avoid common PS1/unbound-variable errors.
set +u
source "$CONDA_SH"
set -u

activate_env() {
    local env_name="$1"
    if [[ -z "$env_name" ]]; then
        return 0
    fi
    echo "[INFO] Activating conda env: ${env_name}"
    set +u
    conda activate "$env_name"
    set -u
    echo "[INFO] python=$(command -v python)"
    python - <<'PY_CHECK'
import sys
print('[INFO] python_version=' + sys.version.replace('
', ' '))
PY_CHECK
}

deactivate_env() {
    set +u
    conda deactivate >/dev/null 2>&1 || true
    set -u
}

# ------------------------------------------------------------
# NNstats-only Python step
# ------------------------------------------------------------
if [[ "$RUN_NNSTATS" == "1" ]]; then
    if [[ ! -f "$NN_SCRIPT" ]]; then
        echo "NN script not found: $NN_SCRIPT" >&2
        exit 1
    fi

    echo "[RUN] NNstats for ${job_tag}"
    activate_env "$PY_ENV"

    python -u "$NN_SCRIPT" \
        --prep-file "$PREP_FILE" \
        --out-file "${CHUNK_DIR}/NNstats.tsv" \
        --jobs "${SLURM_CPUS_PER_TASK:-1}" \
        > "${CHUNK_LOGDIR}/nnstats_${job_tag}.out" \
        2> "${CHUNK_LOGDIR}/nnstats_${job_tag}.err"

    deactivate_env
fi

# ------------------------------------------------------------
# ATHENA v3
# ------------------------------------------------------------
if [[ "$RUN_ATHENA" == "1" ]]; then
    if [[ ! -f "$ATHENA_SCRIPT" ]]; then
        echo "ATHENA script not found: $ATHENA_SCRIPT" >&2
        exit 1
    fi

    echo "[RUN] ATHENA for ${job_tag}"
    activate_env "$ATHENA_ENV"

    ATHENA_CMD=(
        python -u "$ATHENA_SCRIPT"
        --prep-file "$PREP_FILE"
        --tissue-file "$TISSUE_FILE"
        --out-file "${CHUNK_DIR}/athena_features.csv"
        --panel "$PANEL"
        --radius "$ATHENA_RADIUS"
        --min_cells_sample "$ATHENA_MIN_CELLS"
        --regions $ATHENA_REGIONS
    )

    if [[ "$RUN_ATHENA_INTERACTIONS" == "1" ]]; then
        ATHENA_CMD+=(--run-interactions --interaction-permutations "$ATHENA_INTERACTION_PERMUTATIONS")
    fi
    if [[ "$RUN_ATHENA_RIPLEY" == "1" ]]; then
        ATHENA_CMD+=(--run-ripley)
    fi

    "${ATHENA_CMD[@]}" \
        > "${CHUNK_LOGDIR}/athena_${job_tag}.out" \
        2> "${CHUNK_LOGDIR}/athena_${job_tag}.err"

    deactivate_env
fi

echo "[DONE] ${job_tag}"
