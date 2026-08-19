#!/bin/bash
#SBATCH --job-name=build_spatial_manifest
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/build_spatial_manifest_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/build_spatial_manifest_%j.err

set -euo pipefail

# Usage:
#   RUN_ROOT=/path/to/prep_root sbatch build_spatial_feature_manifest_v3.sh
#
# Expected layout:
#   RUN_ROOT/dataset/cohort/panel/chunk_xxxx/1NN_prep.tsv
#   RUN_ROOT/dataset/cohort/panel/chunk_xxxx/tissue_prep.tsv

RUN_ROOT="${RUN_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/spatial_feature_chunk_manifest.tsv}"
LOGDIR="${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs}"

mkdir -p "$LOGDIR"
mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"

if [[ ! -d "$RUN_ROOT" ]]; then
    echo "RUN_ROOT not found: $RUN_ROOT" >&2
    exit 1
fi

find "$RUN_ROOT" -type f -name "1NN_prep.tsv" | sort | while read -r prep; do
    chunk_dir="$(dirname "$prep")"

    # expected layout: RUN_ROOT/dataset/cohort/panel/chunk_xxxx
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
        "$dataset" "$cohort" "$panel" "$chunk" "$prep" "$tissue" >> "$MANIFEST"
done

echo "Manifest written to: $MANIFEST"
echo "Number of chunk jobs: $(wc -l < "$MANIFEST")"
