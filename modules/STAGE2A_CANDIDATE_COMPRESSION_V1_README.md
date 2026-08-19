# Stage 2A candidate ranking and redundancy compression v1

## Purpose

Convert completed Stage 1 feature-level univariate results into a redundancy-compressed candidate manifest for Stage 2B global-module discovery.

Primary discovery uses **TURBT only**. RC should be run separately as an exploratory sensitivity analysis after TURBT modules are frozen.

## Files

- `stage2a_rank_compress_candidates_v1.py`
- `stage2a_rank_compress_candidates_zscore_config.example.json`
- `submit_stage2a_rank_compress_candidates_v1.sh`

## Recommended first run: schema audit

Edit the config so `results_roots` points only to the completed **zscore** Stage 1 result roots.

```bash
python stage2a_rank_compress_candidates_v1.py \
  --config configs/stage2a_rank_compress_candidates_zscore.json \
  --audit-only
```

Inspect:

```text
<input_root>/input_schema_audit.csv
<input_root>/standardized_rows_preview.csv
<input_root>/context_input_summary.csv
<input_root>/all_scored_eligible_features.parquet (or .csv.gz)
<input_root>/preselected_features.parquet (or .csv.gz)
```

The script accepts only tables containing both a feature-name column and a recognizable OOF metric. It coalesces endpoint-specific columns such as `oof_auc` and `oof_cindex` row-wise.

## Full compression run

```bash
python stage2a_rank_compress_candidates_v1.py \
  --config configs/stage2a_rank_compress_candidates_zscore.json
```

or

```bash
sbatch submit_stage2a_rank_compress_candidates_v1.sh
```

## Main outputs

```text
run_config.resolved.json
input_schema_audit.csv
context_input_summary.csv
all_scored_eligible_features.parquet
provenance_duplicate_audit.csv
preselected_features.parquet
matrix_reconstruction/
redundancy_group_audit.csv
compressed_context_representatives.parquet
feature_family_selection_audit.csv
context_selection_summary.csv
candidate_context_recurrence.csv
final_candidate_composition.csv
global_module_candidate_manifest.csv
```

Use `global_module_candidate_manifest.csv` as the `candidate_manifest` in the existing Stage 2B configuration.

## Ranking

Within each cohort × panel × endpoint context:

```text
candidate_score =
    0.50 × percentile rank of OOF performance
  + 0.25 × percentile rank of improvement over the fixed clinical model
  + 0.15 × percentile rank favoring lower fold SD
  + 0.10 × percentile rank of nonmissingness
```

Weights are automatically rescaled per row when a secondary metric is unavailable.

## Compression

1. Collapse canonical duplicate feature definitions across prep roots.
2. Preselect the top 300 AR and 250 BT features per context.
3. Reconstruct patient vectors for only those features.
4. Build biological microfamilies from feature group, parsed cell set, tissue, metric family, and state class.
5. Within each microfamily, connect features with positive Spearman correlation above the panel threshold:
   - AR: 0.93
   - BT: 0.97
6. Keep the highest-ranked representative from each correlated group.
7. Nominate up to 60 AR and 75 BT representatives per context using soft feature-family caps.

## Later multi-transform run

When log1p-zscore results are complete, add them to the input roots and change:

```json
"transforms": ["zscore", "log1p_zscore"]
```

The script chooses one transform per underlying feature and defaults to the simpler transform when candidate scores differ by less than 0.01.
