# Stage 2B-2 root-specific K review v2

Files:
- `stage2b2_root_k_review_v2.ipynb`: thin review notebook.
- `stage2b2_root_k_utils_v2.py`: clustering, semantic ontology, coherence, K scoring, diagnostics, and plotting.

## Main changes from v1

1. Row-Spearman is the primary clustering geometry; direct signed is sensitivity.
2. Composite K score:
   - 45% statistical silhouette
   - 30% primitive coherence enrichment
   - 10% tissue-aware coherence enrichment
   - 0% metric-aware coherence (diagnostic only)
   - 7.5% largest-cluster control
   - 7.5% singleton control
3. Primitive/tissue/metric semantic silhouettes remain diagnostic only.
4. Direct signed within-module cohesion is reported as a guardrail for row-Spearman modules.
5. Soft review warnings:
   - largest module fraction > 0.50
   - singleton-module fraction > 0.25
   - weak direct cohesion when the 25th percentile module mean signed rho < 0 or the median module fraction of positive pairs < 0.60.
6. Candidate-cap stability is not run here; it is deferred to a later sensitivity analysis.

## Install

Place both main files in:
`/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2`

The notebook also expects:
`stage2_feature_parser_v8.py`

## Inputs

Completed Stage 2B-1 output:
`/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2b1_root_consensus_v1`

## Outputs

`/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2b2_root_k_review_v2`

Run the notebook top to bottom after Stage 2B-1 has finished.
