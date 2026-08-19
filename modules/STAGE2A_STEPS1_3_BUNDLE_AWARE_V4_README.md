# Stage 2A steps 1–3: bundle-aware context review v4

## Purpose

This version treats each Stage 1 run/chunk as a result bundle with a shared filename prefix:

- `__summary.csv`: one-row-per-feature backbone
- `__fullmodels.csv`: full-data biomarker, clinical, and combined models
- `__feature_filter.csv`: completeness and feature-QC information
- `__folds.csv`: fold/repeat-level predictive metrics
- `__oof.csv`: patient-level predictions; discovered only when enabled and not loaded by default

The bundle files are joined by `feature`. Different chunks, prep roots, feature groups, panels, cohorts, endpoints, and transforms remain separate until the context worker combines them.

## Parallel architecture

`bundle_processing_mode = "context_workers"` is the recommended mode.

1. The inventory job performs a lightweight path scan and writes one bundle manifest per context.
2. A Slurm array launches one task per context, requesting one CPU per task.
3. Each worker loads and joins all bundles for its own context, selects the best transform per underlying variable, and creates the complete review package.
4. A dependent aggregation job combines the context summaries.

This prevents the inventory job from serially reading every fold table across the entire project.

## Primary fields

The script uses:

- OOF metric: `biomarker_oof_auc` for response or `biomarker_oof_cindex` for survival
- clinical OOF metric: corresponding `clinical_oof_*`
- combined OOF metric: `clinical_plus_biomarker_oof_*`
- clinical delta: corresponding `delta_oof_*_vs_clinical`
- stability: `biomarker_oof_*_repeat_std`
- completeness: `nonmissing_frac` from `__feature_filter.csv`
- primary statistical annotation: `biomarker_only_wald_p` from `__fullmodels.csv`
- incremental statistical annotation: clinical-plus-biomarker LRT p-value
- fold diagnostic: sign consistency of fold-level clinical delta, not coefficient-direction consistency

P-values remain annotation-only and do not enter the default candidate evidence score.

## Installation

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules
mkdir -p "$MODULE_DIR/configs"

cp stage2a_steps1_3_context_review_v4.py "$MODULE_DIR/"
cp submit_stage2a_steps1_3_context_review_v4.sh "$MODULE_DIR/"
cp stage2a_steps1_3_context_review_config_v4_FULL.example.json \
   "$MODULE_DIR/configs/stage2a_steps1_3_context_review_v4.json"
```

## Syntax checks

```bash
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

python -m py_compile "$MODULE_DIR/stage2a_steps1_3_context_review_v4.py"
bash -n "$MODULE_DIR/submit_stage2a_steps1_3_context_review_v4.sh"
```

## Run the lightweight inventory first

```bash
python "$MODULE_DIR/stage2a_steps1_3_context_review_v4.py" inventory \
  --config "$MODULE_DIR/configs/stage2a_steps1_3_context_review_v4.json"
```

Inspect:

- `stage2a_bundle_index_all.csv`
- `stage2a_bundle_index_filtered.csv`
- `bundle_file_discovery_audit.csv`
- `stage2a_context_index.csv`
- `stage2a_inventory_summary.csv`

Confirm both transforms, expected prep roots, feature groups, cohorts, panels, and endpoints are present before submitting workers.

## Submit the context array

```bash
CONFIG_JSON="$MODULE_DIR/configs/stage2a_steps1_3_context_review_v4.json" \
MAX_CONCURRENT=16 \
WORKER_MEM=32G \
WORKER_TIME=08:00:00 \
bash "$MODULE_DIR/submit_stage2a_steps1_3_context_review_v4.sh"
```

Each context worker uses one CPU. `MAX_CONCURRENT` controls how many one-CPU tasks run at once.

## Important outputs

Root-level:

- `context_review_index.html`
- `context_quality_review.csv`
- `context_manual_review_template.csv`
- `all_context_best_transform_features.parquet` or `.csv.gz`
- `all_context_top_features_for_manual_review.csv.gz`
- `input_schema_audit.csv`

Per context:

- `all_transforms_merged.parquet` or `.csv.gz`
- `bundle_load_audit.csv`
- `best_transform_features.parquet` or `.csv.gz`
- `transform_selection_audit.csv`
- `context_quality_summary.csv`
- `top_features_for_manual_review.csv`
- `context_review_report.html`
- `plots/`
- `tables/`

## OOF files

Keep `"oof": false` for the main review. OOF files are much larger and are not needed for best-transform selection, ranking, P-value annotation, completeness, or fold stability. They can be loaded later for a small shortlist of candidates when patient-level ROC/calibration plots are needed.
