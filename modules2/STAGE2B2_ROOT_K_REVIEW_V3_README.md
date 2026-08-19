# Stage 2B-2 Root K Review v3

This revision keeps row-Spearman as the primary clustering geometry and direct-signed distance as a sensitivity analysis, while making K selection explicitly parsimonious.

## Primary K-selection policy

- The full K grid (default 2-30) is still evaluated and plotted as an extended diagnostic.
- Manuscript-facing K selection is restricted to K <= 10 by default.
- The same composite components are re-ranked only within K=2..10:
  - 45% statistical silhouette
  - 30% primitive coherence enrichment
  - 10% tissue-aware coherence enrichment
  - 7.5% largest-cluster control
  - 7.5% singleton control
- Metric-aware coherence and all semantic silhouettes remain diagnostic only.
- The parsimonious recommendation is the smallest K whose primary-range score is >=90% of the best K<=10 score, preferentially satisfying the module-size health guardrails.
- K>10 remains visible so that over-fragmentation / continued metric inflation can be diagnosed, but it cannot be the primary recommendation unless the user deliberately changes PRIMARY_SELECTION_K_MAX.

## New annotated heatmap

`plot_annotated_k_heatmap()` creates a support-filtered signed-consensus heatmap for any manually chosen K:

- red = positive signed consensus Spearman rho
- blue = negative signed consensus Spearman rho
- white = approximately zero
- a categorical bar above the heatmap marks the selected modules
- modules are labelled M01, M02, ... in heatmap order
- black boundaries separate modules
- an ordered membership CSV can be saved alongside the heatmap

The notebook contains an additional cell where `HEATMAP_K`, panel, root and distance mode can be changed directly.

## Interpretation of plots 07 and 08

### 07_within_module_consensus_rho.png
For every candidate K, each proposed module is assessed using the signed cross-cohort consensus correlation among its member features. The plot shows the median module mean rho and the 25th percentile across modules. This is a direct-cohesion guardrail for row-Spearman clustering: high positive values mean modules are not only similar in network profile but also directly coordinated enough for downstream mean-z scoring.

### 08_cross_cohort_reproducibility.png
For every candidate K, within-module pairwise rho is recalculated separately in each discovery cohort. The plot summarizes the resulting module x cohort mean-rho values. A positive median and positive lower quartile indicate that the proposed modules recur within individual cohorts rather than being driven by an average consensus produced by incompatible cohort-specific structures.

## Files

- `stage2b2_root_k_review_v3.ipynb`
- `stage2b2_root_k_utils_v3.py`
