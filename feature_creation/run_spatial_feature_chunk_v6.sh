#!/usr/bin/env bash
#SBATCH --job-name=spatial_feat_v6
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/%x_%A_%a.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/%x_%A_%a.err

set -euo pipefail

# Per-chunk spatial feature worker.
# Reads one line from MANIFEST using SLURM_ARRAY_TASK_ID.
# Manifest columns:
#   dataset cohort panel chunk prep_file tissue_file
# Writes into each chunk directory:
#   NNstats.tsv
#   athena_features.csv

MANIFEST="${MANIFEST:?MANIFEST env variable is required}"
FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
NN_SCRIPT="${NN_SCRIPT:-${FEATURE_DIR}/step1_nnstats_from_prep_v3.py}"
ATHENA_SCRIPT="${ATHENA_SCRIPT:-${FEATURE_DIR}/athena_run_v3.py}"

RUN_NNSTATS="${RUN_NNSTATS:-1}"
RUN_ATHENA="${RUN_ATHENA:-1}"
RUN_ATHENA_INTERACTIONS="${RUN_ATHENA_INTERACTIONS:-1}"
RUN_ATHENA_RIPLEY="${RUN_ATHENA_RIPLEY:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

PY_ENV="${PY_ENV:-cuda6}"
ATHENA_ENV="${ATHENA_ENV:-/projects/ovcare/users/nikolay_alabi/packages/athena}"
CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"

ATHENA_RADIUS="${ATHENA_RADIUS:-40}"
ATHENA_MIN_CELLS="${ATHENA_MIN_CELLS:-20}"
ATHENA_REGIONS="${ATHENA_REGIONS:-Tumor Stroma All}"
ATHENA_INTERACTION_PERMUTATIONS="${ATHENA_INTERACTION_PERMUTATIONS:-100}"
MASTER_LOGDIR="${MASTER_LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"

mkdir -p "$MASTER_LOGDIR"

need_file() {
    [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }
}

need_file "$MANIFEST"
need_file "$CONDA_SH"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
line="$(sed -n "${TASK_ID}p" "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "[ERROR] No manifest line for task ${TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r DATASET COHORT PANEL CHUNK PREP_FILE TISSUE_FILE <<< "$line"
CHUNK_DIR="$(dirname "$PREP_FILE")"
CHUNK_LOGDIR="${CHUNK_DIR}/logs"
mkdir -p "$CHUNK_LOGDIR"

job_tag="${DATASET}_${COHORT}_${PANEL}_${CHUNK}"
NN_OUT="${CHUNK_DIR}/NNstats.tsv"
ATHENA_OUT="${CHUNK_DIR}/athena_features.csv"

cat <<INFO
[INFO] job_tag=${job_tag}
[INFO] TASK_ID=${TASK_ID}
[INFO] DATASET=${DATASET}
[INFO] COHORT=${COHORT}
[INFO] PANEL=${PANEL}
[INFO] CHUNK=${CHUNK}
[INFO] PREP_FILE=${PREP_FILE}
[INFO] TISSUE_FILE=${TISSUE_FILE}
[INFO] CHUNK_DIR=${CHUNK_DIR}
[INFO] RUN_NNSTATS=${RUN_NNSTATS}
[INFO] RUN_ATHENA=${RUN_ATHENA}
[INFO] RUN_ATHENA_INTERACTIONS=${RUN_ATHENA_INTERACTIONS}
[INFO] RUN_ATHENA_RIPLEY=${RUN_ATHENA_RIPLEY}
[INFO] SKIP_EXISTING=${SKIP_EXISTING}
INFO

need_file "$PREP_FILE"
need_file "$TISSUE_FILE"

# Conda setup. Keep nounset disabled during conda activation to avoid PS1 errors.
set +u
source "$CONDA_SH"
set -u

activate_env() {
    local env_name="$1"
    echo "[INFO] Activating conda env: ${env_name}"
    set +u
    conda activate "$env_name"
    set -u
    echo "[INFO] python=$(command -v python)"
    python - <<'PY_CHECK'
import sys
print('[INFO] python_version=' + sys.version.replace('\n', ' '))
PY_CHECK
}

deactivate_env() {
    set +u
    conda deactivate >/dev/null 2>&1 || true
    set -u
}

if [[ "$RUN_NNSTATS" == "1" ]]; then
    need_file "$NN_SCRIPT"
    if [[ "$SKIP_EXISTING" == "1" && -s "$NN_OUT" ]]; then
        echo "[SKIP] NNstats exists: $NN_OUT"
    else
        echo "[RUN] NNstats for ${job_tag}"
        activate_env "$PY_ENV"
        python -u "$NN_SCRIPT" \
            --prep-file "$PREP_FILE" \
            --out-file "$NN_OUT" \
            --jobs "${SLURM_CPUS_PER_TASK:-1}" \
            > "${CHUNK_LOGDIR}/nnstats_${job_tag}.out" \
            2> "${CHUNK_LOGDIR}/nnstats_${job_tag}.err"
        deactivate_env
        [[ -s "$NN_OUT" ]] || { echo "[ERROR] NNstats output missing/empty: $NN_OUT" >&2; exit 1; }
    fi
fi

if [[ "$RUN_ATHENA" == "1" ]]; then
    need_file "$ATHENA_SCRIPT"
    if [[ "$SKIP_EXISTING" == "1" && -s "$ATHENA_OUT" ]]; then
        echo "[SKIP] ATHENA exists: $ATHENA_OUT"
    else
        echo "[RUN] ATHENA for ${job_tag}"
        activate_env "$ATHENA_ENV"

        read -r -a ATHENA_REGION_ARRAY <<< "$ATHENA_REGIONS"
        ATHENA_CMD=(
            python -u "$ATHENA_SCRIPT"
            --prep-file "$PREP_FILE"
            --tissue-file "$TISSUE_FILE"
            --out-file "$ATHENA_OUT"
            --panel "$PANEL"
            --radius "$ATHENA_RADIUS"
            --min_cells_sample "$ATHENA_MIN_CELLS"
            --regions "${ATHENA_REGION_ARRAY[@]}"
        )

        if [[ "$RUN_ATHENA_INTERACTIONS" == "1" ]]; then
            ATHENA_CMD+=(--run-interactions --interaction-permutations "$ATHENA_INTERACTION_PERMUTATIONS")
        fi
        if [[ "$RUN_ATHENA_RIPLEY" == "1" ]]; then
            ATHENA_CMD+=(--run-ripley)
        fi

        printf '[INFO] ATHENA_CMD:' > "${CHUNK_LOGDIR}/athena_${job_tag}.cmd.txt"
        printf ' %q' "${ATHENA_CMD[@]}" >> "${CHUNK_LOGDIR}/athena_${job_tag}.cmd.txt"
        printf '\n' >> "${CHUNK_LOGDIR}/athena_${job_tag}.cmd.txt"

        "${ATHENA_CMD[@]}" \
            > "${CHUNK_LOGDIR}/athena_${job_tag}.out" \
            2> "${CHUNK_LOGDIR}/athena_${job_tag}.err"
        deactivate_env
        [[ -s "$ATHENA_OUT" ]] || { echo "[ERROR] ATHENA output missing/empty: $ATHENA_OUT" >&2; exit 1; }
    fi
fi

echo "[DONE] ${job_tag}"
