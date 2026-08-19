# Stage 2B-1: Root-specific cohort matrices and consensus correlations

## Goal

Freeze endpoint-independent feature universes after Stage 2A-5, rebuild every surviving union feature in every discovery cohort where it can be measured, then create one signed-Spearman consensus matrix per panel × prep root.

This stage does **not** cluster features and does **not** choose K.

## Architecture

1. `setup`: reads `all_context_root_final_candidates.parquet` once, creates small root-universe CSVs and worker indices.
2. `matrices`: one CPU per cohort × panel × prep root. Rebuilds patient-level values from Stage 1 v6 using TURBT / all / median / QC threshold 0.05.
3. `consensus`: one CPU per panel × prep root. Computes cohort Spearman matrices, pairwise N, equal-weight consensus, pair support, sign consistency, support filtering, and diagnostic heatmaps.
4. `aggregate`: verifies all workers and writes Stage 2B-2-ready manifests and review summaries.

## Scientific defaults

- discovery cohorts: NAC2020, PURE01, BLASST, No-NAC
- NAC2015 excluded
- sample type: TURBT
- patient subset: all
- aggregation: median
- epithelial threshold: 0.05
- within-cohort correlation: signed Spearman
- feature nonmissing fraction for correlation: >= 0.20
- pairwise-complete patients: >= 20
- pair support: >= 2 cohorts
- feature support fraction: >= 0.10
- nomination support across cohorts is annotation only, not eligibility
- unsupported pair correlations are kept as NaN in an audit matrix and set to 0 only in the clustering-ready matrix

## Install

Put these in `modules2/`:

- `stage2b1_root_consensus_v1.py`
- `worker_stage2b1_root_matrix_v1.sh`
- `worker_stage2b1_root_consensus_v1.sh`
- `submit_stage2b1_root_consensus_v1.sh`

Put this in `modules2/configs/`:

- `stage2b1_root_consensus_v1.json`

## Run

Wait for each stage to finish before starting the next.

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2

MODE=validate bash "$MODULE_DIR/submit_stage2b1_root_consensus_v1.sh"
MODE=setup bash "$MODULE_DIR/submit_stage2b1_root_consensus_v1.sh"
MODE=matrices bash "$MODULE_DIR/submit_stage2b1_root_consensus_v1.sh"
MODE=consensus bash "$MODULE_DIR/submit_stage2b1_root_consensus_v1.sh"
MODE=aggregate bash "$MODULE_DIR/submit_stage2b1_root_consensus_v1.sh"
```

### After setup

```bash
ROOT=/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2b1_root_consensus_v1
awk 'END {print NR-1}' "$ROOT/stage2b1_matrix_worker_index.csv"
awk 'END {print NR-1}' "$ROOT/stage2b1_consensus_worker_index.csv"
```

With 4 discovery cohorts and the expected 7 panel/root combinations, this should normally be 28 matrix workers and 7 consensus workers.

### Check matrix completion

```bash
echo expected=$(awk 'END {print NR-1}' "$ROOT/stage2b1_matrix_worker_index.csv")
echo complete=$(find "$ROOT/cohort_root_matrices" -name .done | wc -l)
```

### Check consensus completion

```bash
echo expected=$(awk 'END {print NR-1}' "$ROOT/stage2b1_consensus_worker_index.csv")
echo complete=$(find "$ROOT/root_consensus" -name .done | wc -l)
```

## First outputs to inspect after aggregation

- `stage2b1_panel_root_review_summary.csv`
- `stage2b1_summary.txt`
- `all_cohort_root_matrix_summary.csv`
- `all_panel_root_feature_support_summary.csv`
- `stage2b1_root_consensus_manifest.csv`

Each root also gets:

- `consensus_signed_spearman_unfiltered.parquet`
- `pair_support_unfiltered.parquet`
- `sign_consistency_unfiltered.parquet`
- `consensus_signed_spearman_support_filtered_clustering.parquet`
- `consensus_signed_spearman_support_filtered_nan.parquet`
- `feature_support_summary.csv`
- `pairwise_consensus_audit.csv.gz`
- cohort-specific Spearman and pairwise-N matrices
- five diagnostic plots

## What to upload for review

The smallest useful files are:

- `stage2b1_panel_root_review_summary.csv`
- `stage2b1_summary.txt`
- `all_cohort_root_matrix_summary.csv`

Those are enough to decide whether support rules are behaving sensibly before Stage 2B-2 K selection.
