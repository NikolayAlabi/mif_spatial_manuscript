#!/bin/bash
set -euo pipefail

FEATURE_DIR=${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/univariate_screening}
ROOT=${ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v4/results}
OUTDIR=${OUTDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v4/combined}
LOGDIR=${LOGDIR:-/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v4/logs}
CONDA_SH=${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}
PY_ENV=${PY_ENV:-cuda6}

mkdir -p "$LOGDIR" "$OUTDIR"

sbatch \
  --job-name=combine_stage1_v4 \
  --cpus-per-task=1 \
  --mem=32G \
  --time=04:00:00 \
  -p ${PARTITION:-upgrade} \
  --output="$LOGDIR/combine_stage1_v4.out" \
  --error="$LOGDIR/combine_stage1_v4.err" \
  --wrap="set +u; source $CONDA_SH; conda activate $PY_ENV; set -u; python -u $FEATURE_DIR/combine_stage1_v4_results.py --root $ROOT --outdir $OUTDIR"
