#!/usr/bin/env bash
#SBATCH --job-name=athena_one_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/athena_one_test_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/athena_one_test_%j.err

set -euo pipefail

# Submit example:
#   RUN_ROOT=/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state \
#   TASK_ID=1 \
#   sbatch submit_single_athena_chunk_test.sh
#
# Optional overrides:
#   ATHENA_ENV=/projects/ovcare/users/nikolay_alabi/packages/athena
#   CONDA_SH=/home/nalabi/miniconda3/etc/profile.d/conda.sh
#   FEATURE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation
#   RUN_INTERACTIONS=1
#   INTERACTION_PERMUTATIONS=25

FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
ATHENA_SCRIPT="${ATHENA_SCRIPT:-${FEATURE_DIR}/athena_run_v3.py}"

RUN_ROOT="${RUN_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/spatial_feature_chunk_manifest.tsv}"
TASK_ID="${TASK_ID:-1}"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
ATHENA_ENV="${ATHENA_ENV:-/projects/ovcare/users/nikolay_alabi/packages/athena}"

ATHENA_RADIUS="${ATHENA_RADIUS:-40}"
ATHENA_MIN_CELLS="${ATHENA_MIN_CELLS:-20}"
ATHENA_REGIONS="${ATHENA_REGIONS:-Tumor Stroma All}"
RUN_INTERACTIONS="${RUN_INTERACTIONS:-1}"
INTERACTION_PERMUTATIONS="${INTERACTION_PERMUTATIONS:-25}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] Manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$ATHENA_SCRIPT" ]]; then
    echo "[ERROR] ATHENA script not found: $ATHENA_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] Conda setup script not found: $CONDA_SH" >&2
    exit 1
fi

line="$(sed -n "${TASK_ID}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "[ERROR] No manifest line for TASK_ID=${TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r DATASET COHORT PANEL CHUNK PREP_FILE TISSUE_FILE <<< "$line"
CHUNK_DIR="$(dirname "$PREP_FILE")"
CHUNK_LOGDIR="${CHUNK_DIR}/logs"
mkdir -p "$CHUNK_LOGDIR"

OUT_FILE="${OUT_FILE:-${CHUNK_DIR}/athena_features_test.csv}"
JOB_TAG="${DATASET}_${COHORT}_${PANEL}_${CHUNK}_athena_test"

cat <<INFO
[INFO] RUN_ROOT=${RUN_ROOT}
[INFO] MANIFEST=${MANIFEST}
[INFO] TASK_ID=${TASK_ID}
[INFO] DATASET=${DATASET}
[INFO] COHORT=${COHORT}
[INFO] PANEL=${PANEL}
[INFO] CHUNK=${CHUNK}
[INFO] PREP_FILE=${PREP_FILE}
[INFO] TISSUE_FILE=${TISSUE_FILE}
[INFO] CHUNK_DIR=${CHUNK_DIR}
[INFO] OUT_FILE=${OUT_FILE}
[INFO] ATHENA_ENV=${ATHENA_ENV}
[INFO] RUN_INTERACTIONS=${RUN_INTERACTIONS}
[INFO] INTERACTION_PERMUTATIONS=${INTERACTION_PERMUTATIONS}
INFO

# Source conda with nounset disabled to avoid PS1/unbound-variable errors.
set +u
source "$CONDA_SH"
conda activate "$ATHENA_ENV"
set -u

echo "[INFO] which python: $(command -v python)"
python --version

# Import test first. This should fail quickly if the environment is wrong.
python - <<'PY'
import sys
print("[INFO] Python executable:", sys.executable)
try:
    import numpy as np
    import pandas as pd
    import athena as ath
    import spatialOmics
    from spatialOmics import SpatialOmics
    print("[INFO] numpy:", np.__version__)
    print("[INFO] pandas:", pd.__version__)
    print("[INFO] athena import: OK")
    print("[INFO] spatialOmics import: OK")
except Exception as e:
    print("[ERROR] ATHENA environment import test failed:", repr(e))
    raise
PY

ATHENA_CMD=(
    python -u "$ATHENA_SCRIPT"
    --prep-file "$PREP_FILE"
    --tissue-file "$TISSUE_FILE"
    --out-file "$OUT_FILE"
    --panel "$PANEL"
    --radius "$ATHENA_RADIUS"
    --min_cells_sample "$ATHENA_MIN_CELLS"
    --regions $ATHENA_REGIONS
)

if [[ "$RUN_INTERACTIONS" == "1" ]]; then
    ATHENA_CMD+=(--run-interactions --interaction-permutations "$INTERACTION_PERMUTATIONS")
fi

echo "[RUN] ${ATHENA_CMD[*]}"
"${ATHENA_CMD[@]}" \
    > "${CHUNK_LOGDIR}/athena_${JOB_TAG}.out" \
    2> "${CHUNK_LOGDIR}/athena_${JOB_TAG}.err"

echo "[DONE] ATHENA test finished"
ls -lh "$OUT_FILE"
