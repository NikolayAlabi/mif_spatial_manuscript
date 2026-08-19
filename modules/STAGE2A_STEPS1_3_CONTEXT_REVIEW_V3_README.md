# Stage 2A steps 1–3 v3: comprehensive context review

This version retains the best-transform selection and manual context-review workflow from v2, but adds a substantially expanded visual and tabular review package for every cohort × panel × endpoint context.

It still stops before biological microcompression and before final candidate nomination.

## Core workflow

1. Inventory and standardize completed z-score and log1p-z-score Stage 1 results.
2. Create one input shard per context.
3. Select one best transform per underlying variable.
4. Calculate the candidate evidence score.
5. Produce a comprehensive context-level review package.
6. Aggregate all contexts and create a cross-context review index.

## Candidate evidence score

P-values remain annotation-only. The default within-context percentile-rank score is:

- 55% OOF AUC or C-index
- 20% stability
- 15% improvement over the clinical model
- 10% nonmissingness

Available weights are renormalized row-wise when an optional metric is missing.

## Per-context directory structure

Each worker writes:

```text
contexts/<context_slug>/
    best_transform_features.parquet or .csv.gz
    transform_selection_audit.csv
    context_quality_summary.csv
    top_features_for_manual_review.csv
    context_review_report.html
    plots/
    tables/
```

`context_review_report.html` is the easiest starting point. It contains the context summary, the top 25 candidate table, links to every CSV, and a gallery of all available plots.

## Per-context figures

Figures are created only when the required metric is available.

- `01_oof_distribution.png`
- `02_clinical_delta_distribution.png`
- `03_clinical_delta_waterfall_top_candidates.png`
- `04_clinical_delta_top30_labeled.png`
- `05_fold_sd_distribution.png`
- `06_direction_consistency_distribution.png`
- `07_nominal_pvalue_distribution.png`
- `08_context_qvalue_distribution.png`
- `09_candidate_evidence_score_distribution.png`
- `10a_oof_vs_delta_colored_by_feature_group.png`
- `10b_oof_vs_delta_colored_by_prep_root.png`
- `10c_oof_vs_fold_sd_colored_by_feature_group.png`
- `10d_oof_vs_qvalue_colored_by_feature_group.png`
- `11_rank_profile_oof_and_evidence.png`
- `12_rank_profile_clinical_delta.png`
- `13_cumulative_topn_feature_group_composition.png`
- `14_cumulative_topn_prep_root_composition.png`
- `15_oof_boxplot_by_feature_group.png`
- `16_oof_boxplot_by_prep_root.png`
- `17_zscore_vs_log1p_oof_scatter.png`
- `18_transform_oof_difference_waterfall.png`
- `19_candidate_metric_correlation_heatmap.png`
- `20_evidence_tier_counts.png`
- `21_transform_selection_counts.png`

The top-candidate scatter plots use the top 100 candidates by default. This is controlled by `plot_top_n`.

## Per-context CSV tables

- `01_metric_distribution_summary.csv`: quantiles, means, missingness, and range for each review metric.
- `02_threshold_count_summary.csv`: counts passing configurable OOF, clinical-delta, stability, P-value, and q-value thresholds.
- `03_feature_group_summary.csv`: candidate quality summarized by NN, ATHENA, cell features, and triads.
- `04_prep_root_summary.csv`: candidate quality summarized by prep root / feature source.
- `05_top_candidates_overall.csv`: top candidates by evidence score.
- `06_top_candidates_by_feature_group.csv`: strongest candidates within every feature family.
- `07_top_candidates_by_prep_root.csv`: strongest candidates within every prep root.
- `08_candidate_metric_spearman_correlations.csv`: correlations among review metrics.
- `09_candidate_metric_pairwise_n.csv`: pairwise sample sizes for the correlation table.
- `10_topn_composition_by_feature_group.csv`: cumulative feature-family representation across top-N cutoffs.
- `11_topn_composition_by_prep_root.csv`: cumulative prep-root representation across top-N cutoffs.
- `12_transform_pair_comparison.csv`: z-score versus log1p-z-score performance for variables with both transforms.
- `13_pareto_front_candidates.csv`: descriptive non-dominated candidates with high OOF, high clinical delta, and low fold SD.
- `14_metric_availability.csv`: metric completeness audit.
- `15_evidence_tier_counts.csv`: descriptive evidence-tier counts.

## Cross-context outputs

After aggregation, review:

```text
context_review_index.html
context_quality_review.csv
context_manual_review_template.csv
context_review_output_inventory.csv
context_comparison_plots/
```

The cross-context plots compare:

- median top-10 OOF performance;
- maximum OOF performance;
- number of variables with OOF ≥0.60;
- number of stable variables;
- fraction selecting log1p-z-score;
- number of variables with q ≤0.20.

## Installation

```bash
cp stage2a_steps1_3_context_review_v3.py \
  /projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/

cp submit_stage2a_steps1_3_context_review_v3.sh \
  /projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/

cp stage2a_steps1_3_context_review_config_v3.example.json \
  /projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/configs/stage2a_steps1_3_context_review.json
```

Edit the two completed result roots in the JSON.

## Run

Schema/inventory check:

```bash
python stage2a_steps1_3_context_review_v3.py inventory \
  --config configs/stage2a_steps1_3_context_review.json
```

Submit one worker per context and the dependent aggregate job:

```bash
CONFIG_JSON=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/configs/stage2a_steps1_3_context_review.json \
  bash submit_stage2a_steps1_3_context_review_v3.sh
```

## Configurable review settings

```json
{
  "plot_top_n": 100,
  "top_n_per_category": 25,
  "scatter_annotate_top_n": 8,
  "hist_bins": 30,
  "composition_top_n_values": [10, 20, 30, 50, 75, 100],
  "oof_review_thresholds": [0.55, 0.60, 0.65],
  "delta_review_thresholds": [0.00, 0.02, 0.05],
  "fold_sd_review_thresholds": [0.05, 0.10, 0.15],
  "direction_review_thresholds": [0.60, 0.80, 1.00],
  "p_review_thresholds": [0.05, 0.01],
  "q_review_thresholds": [0.20, 0.10, 0.05]
}
```

## Compatibility and testing

The script was syntax-checked for Python 3.9 compatibility and tested end-to-end on synthetic response data containing both z-score and log1p-z-score results. The first real inventory run remains important because your Stage 1 outputs may require column-alias adjustments.
