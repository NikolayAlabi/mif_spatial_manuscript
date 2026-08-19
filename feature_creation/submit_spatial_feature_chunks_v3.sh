#!/bin/bash
#SBATCH --job-name=submit_spatial_features
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/submit_spatial_features_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/submit_spatial_features_%j.err

set -euo pipefail

# Usage:
#   MANIFEST=/path/to/spatial_feature_chunk_manifest.tsv \
#   WORKER=/path/to/run_spatial_feature_chunk_v3.sh \
#   MAX_CONCURRENT=32 \
#   sbatch submit_spatial_feature_chunks_v3.sh

MANIFEST="${MANIFEST:-/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only/spatial_feature_chunk_manifest.tsv}"
WORKER="${WORKER:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation/run_spatial_feature_chunk_v3.sh}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$WORKER" ]]; then
    echo "Worker not found: $WORKER" >&2
    exit 1
fi

N=$(wc -l < "$MANIFEST")
if [[ "$N" -lt 1 ]]; then
    echo "Manifest is empty: $MANIFEST" >&2
    exit 1
fi

echo "Submitting $N chunk jobs from $MANIFEST"
echo "Worker: $WORKER"
echo "Max concurrent: $MAX_CONCURRENT"

sbatch --array=1-"$N"%"$MAX_CONCURRENT" \
    --export=ALL,MANIFEST="$MANIFEST" \
    "$WORKER"
