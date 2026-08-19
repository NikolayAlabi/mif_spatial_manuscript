#!/usr/bin/env bash
#SBATCH --job-name=prep_sources_v6
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/prep_sources_v6_%j.out
#SBATCH --error=/projects/ovcare/users/nikolay_alabi/immuno/weibull/logs/prep_sources_v6_%j.err

# python -u generate_phenotype_abundance_consistency.py \
#   --tma_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_cohorts \
#   --whole_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_wholesections \
#   --blasst_metadata_csv /projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_BLASST.csv \
#   --out_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/phenotype_abundance_rebuild \
#   --existing_annotation_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments \
#   --qc_dir /projects/ovcare/users/nikolay_alabi/immuno/data \
#   --panels AR BT MY

# Build all five prep roots from canonical reviewed cell_df parquets.
# Compatible with optional KOLL cohort input.
#
# Feature sources produced:
#   1. phenotype_only       -> label_phenotype, AR + BT
#   2. AR_state             -> label_ar_state, AR only
#   3. AR_checkpoint_state  -> label_checkpoint_state, AR only
#   4. compartment          -> label_compartment, AR + BT
#   5. compartment_state    -> label_compartment_state, AR only
#
# Expected input parquets:
#   tma_cell_df.parquet
#   wholesection_cell_df.parquet
#   koll_cell_df.parquet                 optional, controlled by INCLUDE_KOLL
#
# Useful env overrides:
#   FEATURE_DIR=/projects/.../feature_creation
#   CELL_DIR=/projects/.../cell_df_rebuild
#   WEIBULL_ROOT=/projects/.../weibull
#   INCLUDE_KOLL=auto | 1 | 0             default: auto
#   CLEAN_EXISTING=1 | 0                  default: 1; archive old prep roots before writing
#   ARCHIVE_TAG=pre_v6_YYYYmmdd_HHMMSS    default auto timestamp
#   CHUNK_SIZE=100
#   BBOX_AREA_TO_MM2_FACTOR=1e-6
#   DRY_RUN=1                             print commands but do not run

set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
FEATURE_DIR="${FEATURE_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/manuscript/feature_creation}"
PREP_SCRIPT="${PREP_SCRIPT:-${FEATURE_DIR}/prep_inputs.py}"

CELL_DIR="${CELL_DIR:-/projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/cell_df_rebuild}"
TMA_CELL="${TMA_CELL:-${CELL_DIR}/tma_cell_df.parquet}"
WHOLE_CELL="${WHOLE_CELL:-${CELL_DIR}/wholesection_cell_df.parquet}"
KOLL_CELL="${KOLL_CELL:-${CELL_DIR}/koll_cell_df.parquet}"

WEIBULL_ROOT="${WEIBULL_ROOT:-/projects/ovcare/users/nikolay_alabi/immuno/weibull}"
LOGDIR="${LOGDIR:-${WEIBULL_ROOT}/logs}"

CONDA_SH="${CONDA_SH:-/home/nalabi/miniconda3/etc/profile.d/conda.sh}"
PY_ENV="${PY_ENV:-cuda6}"

INCLUDE_KOLL="${INCLUDE_KOLL:-auto}"     # auto, 1, or 0
CLEAN_EXISTING="${CLEAN_EXISTING:-1}"
ARCHIVE_TAG="${ARCHIVE_TAG:-pre_v6_$(date +%Y%m%d_%H%M%S)}"

CHUNK_SIZE="${CHUNK_SIZE:-100}"
BBOX_AREA_TO_MM2_FACTOR="${BBOX_AREA_TO_MM2_FACTOR:-1e-6}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOGDIR" "$WEIBULL_ROOT"

# -----------------------------
# Helpers
# -----------------------------
need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    exit 1
  fi
}

archive_dir_if_needed() {
  local d="$1"
  if [[ "$CLEAN_EXISTING" != "1" ]]; then
    return 0
  fi
  if [[ -d "$d" ]]; then
    local archived="${d}_${ARCHIVE_TAG}"
    echo "[INFO] Archiving existing prep root:"
    echo "       $d"
    echo "    -> $archived"
    if [[ "$DRY_RUN" != "1" ]]; then
      mv "$d" "$archived"
    fi
  fi
}

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_prep() {
  local source_name="$1"
  local outdir="$2"
  local label_col="$3"
  shift 3
  local include_panels=("$@")

  echo
  echo "============================================================"
  echo "[PREP] ${source_name}"
  echo "[PREP] outdir:    ${outdir}"
  echo "[PREP] label_col: ${label_col}"
  echo "[PREP] panels:    ${include_panels[*]}"
  echo "============================================================"

  archive_dir_if_needed "$outdir"

  run_cmd python -u "$PREP_SCRIPT" \
    --inputs "${INPUT_FILES[@]}" \
    --outdir "$outdir" \
    --label-col "$label_col" \
    --label-mode column \
    --include-panels "${include_panels[@]}" \
    --exclude-panels MY \
    --chunk-size "$CHUNK_SIZE" \
    --bbox-area-to-mm2-factor "$BBOX_AREA_TO_MM2_FACTOR"

  if [[ "$DRY_RUN" != "1" ]]; then
    echo
    echo "[CHECK] ${source_name} chunks: $(find "$outdir" -name '1NN_prep.tsv' | wc -l)"
    if [[ -f "$outdir/chunk_summary.tsv" ]]; then
      echo "[CHECK] chunk_summary: $outdir/chunk_summary.tsv"
      awk 'NR==1 || /KOLL|koll/' "$outdir/chunk_summary.tsv" | head -20 || true
    fi
  fi
}

# -----------------------------
# Environment and input checks
# -----------------------------
need_file "$PREP_SCRIPT"
need_file "$TMA_CELL"
need_file "$WHOLE_CELL"

INPUT_FILES=("$TMA_CELL" "$WHOLE_CELL")

case "$INCLUDE_KOLL" in
  auto)
    if [[ -f "$KOLL_CELL" ]]; then
      echo "[INFO] KOLL parquet found; including: $KOLL_CELL"
      INPUT_FILES+=("$KOLL_CELL")
    else
      echo "[WARN] KOLL parquet not found; continuing without KOLL: $KOLL_CELL"
    fi
    ;;
  1|true|TRUE|yes|YES)
    need_file "$KOLL_CELL"
    echo "[INFO] INCLUDE_KOLL=1; including: $KOLL_CELL"
    INPUT_FILES+=("$KOLL_CELL")
    ;;
  0|false|FALSE|no|NO)
    echo "[INFO] INCLUDE_KOLL=0; running without KOLL."
    ;;
  *)
    echo "[ERROR] INCLUDE_KOLL must be one of: auto, 1, 0. Got: $INCLUDE_KOLL" >&2
    exit 1
    ;;
esac

cat <<EOF_INFO

[INFO] Prep configuration
  FEATURE_DIR:              $FEATURE_DIR
  PREP_SCRIPT:              $PREP_SCRIPT
  CELL_DIR:                 $CELL_DIR
  WEIBULL_ROOT:             $WEIBULL_ROOT
  INCLUDE_KOLL:             $INCLUDE_KOLL
  CLEAN_EXISTING:           $CLEAN_EXISTING
  ARCHIVE_TAG:              $ARCHIVE_TAG
  CHUNK_SIZE:               $CHUNK_SIZE
  BBOX_AREA_TO_MM2_FACTOR:  $BBOX_AREA_TO_MM2_FACTOR
  DRY_RUN:                  $DRY_RUN

[INFO] Input parquets:
$(printf '  - %s\n' "${INPUT_FILES[@]}")
EOF_INFO

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] Conda setup file not found: $CONDA_SH" >&2
    exit 1
  fi
  set +u
  source "$CONDA_SH"
  conda activate "$PY_ENV"
  set -u
  echo "[INFO] python: $(command -v python)"
fi

# Optional quick schema check before running all prep roots.
if [[ "$DRY_RUN" != "1" ]]; then
  python - <<'PY_SCHEMA' "${INPUT_FILES[@]}"
import sys
import pyarrow.parquet as pq
required = {
    "sample_name", "coord", "x", "y", "tissue_region", "Panel", "cohort",
    "label_phenotype", "label_ar_state", "label_checkpoint_state",
    "label_checkpoint_binary", "label_compartment", "label_compartment_state",
}
for fp in sys.argv[1:]:
    cols = set(pq.read_schema(fp).names)
    missing = sorted(required - cols)
    print(f"[SCHEMA] {fp}")
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in {fp}: {missing}")
    print("         OK")
PY_SCHEMA
fi

# -----------------------------
# Build all five prep roots
# -----------------------------
run_prep \
  "phenotype_only" \
  "$WEIBULL_ROOT/run_reviewed_phenotype_only" \
  "label_phenotype" \
  AR BT

run_prep \
  "AR_state" \
  "$WEIBULL_ROOT/run_reviewed_AR_state" \
  "label_ar_state" \
  AR

run_prep \
  "AR_checkpoint_state" \
  "$WEIBULL_ROOT/run_reviewed_AR_checkpoint_state" \
  "label_checkpoint_state" \
  AR

run_prep \
  "compartment" \
  "$WEIBULL_ROOT/run_reviewed_compartment" \
  "label_compartment" \
  AR BT

run_prep \
  "compartment_state" \
  "$WEIBULL_ROOT/run_reviewed_compartment_state" \
  "label_compartment_state" \
  AR

# -----------------------------
# Final audit summary
# -----------------------------
if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "============================================================"
  echo "[DONE] Prep roots created"
  echo "============================================================"
  for d in \
    run_reviewed_phenotype_only \
    run_reviewed_AR_state \
    run_reviewed_AR_checkpoint_state \
    run_reviewed_compartment \
    run_reviewed_compartment_state
  do
    root="$WEIBULL_ROOT/$d"
    echo
    echo "$root"
    echo "  chunks:      $(find "$root" -name '1NN_prep.tsv' | wc -l)"
    echo "  KOLL chunks: $(find "$root" -path '*KOLL*' -name '1NN_prep.tsv' | wc -l)"
    if [[ -f "$root/chunk_summary.tsv" ]]; then
      echo "  summary:     $root/chunk_summary.tsv"
    fi
  done
fi
