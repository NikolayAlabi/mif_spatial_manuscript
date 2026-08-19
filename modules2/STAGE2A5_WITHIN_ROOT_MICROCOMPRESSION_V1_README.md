# Stage 2A-5 — within-root conservative microcompression v1

This is the finalized Stage 2A-5 immediately downstream of `stage2a4_root_nomination_v1`.

## Policy

Compression is performed **separately inside each prep root**. The script never compares or replaces a feature with a feature from another prep root.

Automatic compression is deliberately narrow:

1. **Exact semantic + patient-vector duplicate**
   - same fully parsed semantic measurement;
   - identical transformed patient vector (same missingness, tolerance `1e-12`);
   - retain the higher-ranked Stage 2A-4 candidate.

2. **Mean ↔ Median of the same measurement**
   - full cell/checkpoint-state identities are preserved;
   - measurement compartment is preserved;
   - only the summary label differs;
   - positive Spearman rho >= `0.95`;
   - at least 20 pairwise-complete patients;
   - retained higher-ranked candidate may lose at most `0.005` OOF performance relative to the removed feature.

3. **Residual exact-semantic near duplicate**
   - exact same fully parsed biological/measurement identity;
   - positive Spearman rho >= `0.98`;
   - at least 20 pairwise-complete patients;
   - OOF loss <= `0.005`.

The following are explicitly disabled:

- cross-root rescue/compression;
- state stripping or parent-phenotype simplification;
- tissue-compartment simplification;
- generic high-correlation pruning of biologically distinct features;
- rescue-only variables not nominated in Stage 2A-4.

This is important because moderate/high correlation among biologically distinct features is the signal that Stage 2B will use to discover root modules.

## Required files under `modules2`

```text
modules2/
    stage2a5_within_root_microcompression_v1.py
    submit_stage2a5_within_root_microcompression_v1.sh
    stage2_feature_parser_v8.py

modules2/configs/
    stage2a5_within_root_microcompression_v1.json
```

The parser bundled here is the grammar-aware parser already used in the Stage 2 review work. Do not substitute the older interpretability adapter, because that adapter contains state/compartment rescue concepts intentionally disabled in this root-aware workflow.

## Run order

Use explicit stages and wait for each to finish.

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2
```

### 1. Validate

```bash
MODE=validate \
bash "$MODULE_DIR/submit_stage2a5_within_root_microcompression_v1.sh"
```

### 2. Inventory

```bash
MODE=inventory \
bash "$MODULE_DIR/submit_stage2a5_within_root_microcompression_v1.sh"
```

Then check:

```bash
ROOT=/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2a5_within_root_microcompression_v1
awk 'END {print NR-1}' "$ROOT/stage2a5_context_index.csv"
```

### 3. Workers

```bash
MODE=workers \
bash "$MODULE_DIR/submit_stage2a5_within_root_microcompression_v1.sh"
```

One CPU is used per endpoint context; each worker loops over only the few prep roots present in that context.

After completion:

```bash
echo "Expected:"
awk 'END {print NR-1}' "$ROOT/stage2a5_context_index.csv"

echo "Completed:"
find "$ROOT/contexts" -name .done | wc -l
```

### 4. Aggregate

```bash
MODE=aggregate \
bash "$MODULE_DIR/submit_stage2a5_within_root_microcompression_v1.sh"
```

## Most useful outputs

Aggregate:

```text
all_context_stage2a5_summary.csv
all_context_root_compression_summary.csv
panel_root_microcompression_summary.csv
stage2a5_root_matrix_manifest.csv
all_context_root_final_candidates.parquet
stage2a5_final_feature_support.csv
all_context_compression_decision_audit.csv
all_context_seed_to_final_representative.csv
stage2a5_aggregate_summary.json
```

Within each context/root:

```text
compressed_root_candidates.csv
compressed_patient_feature_matrix.parquet
compression_decision_audit.csv
evaluated_compression_pairs.csv
seed_to_final_representative.csv
pairwise_correlations_before.csv
pairwise_correlations_after.csv
correlation_heatmap_before.png
correlation_heatmap_after.png
compression_flow.png
root_compression_summary.csv
```

## What to inspect before Stage 2B

Start with `panel_root_microcompression_summary.csv`. Compression should generally be modest. If a root loses a large fraction of its Stage 2A-4 seeds, inspect its `compression_decision_audit.csv` before proceeding.

The before/after correlation heatmaps are diagnostics only. Features with high correlation are **not** automatically pruned unless they satisfy one of the tightly defined same-measurement rules above.
