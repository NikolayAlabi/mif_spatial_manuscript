# Stage 2A steps 1-3 root-aware v1

This is the first finalized stage of the root-module -> meta-module workflow.

## What is reused
- Existing Stage 1 univariate CV result bundles.
- Best-transform logic from Stage 2A v4.
- Existing evidence-score weights:
  - 0.55 OOF discrimination
  - 0.20 stability
  - 0.15 clinical-model delta
  - 0.10 completeness
- P-values remain annotation-only.

## What changes
1. Technical bookkeeping variables are excluded before transform selection/ranking:
   - `(All|Epi|Stroma)__n_cells`
   - `(All|Epi|Stroma)__n_resolved_for_ratio`
   - `n_cells_total_input`
2. Panel-specific prep-root membership is enforced.
3. Two evidence rankings are saved:
   - `candidate_evidence_score` / `candidate_review_rank`: whole-context, audit only.
   - `root_candidate_evidence_score` / `root_candidate_rank`: within prep root, primary for later nomination.
4. Per-context root summaries and candidate-depth profiles are generated so root-specific candidate limits can be chosen from the data.
5. No raw-feature rescue or microcompression occurs in this stage.

## Biological design
AR roots:
- phenotype_only
- AR_state
- AR_checkpoint_state
- compartment
- compartment_state

BT roots:
- phenotype_only
- compartment

These roots are intentionally separate biological spaces. Cross-root raw-feature rescue/microcompression is disabled in the finalized design. Cross-root integration will occur only after root-specific modules have been scored, at the meta-module stage.

See `stage2_root_ontology_v1.json`.

## Output root
`/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2a_steps1_3_rootaware_v1`

Key aggregate outputs:
- `all_context_best_transform_features.parquet`
- `context_quality_review.csv`
- `all_context_root_candidate_summary.csv`
- `all_context_root_candidate_depth_profile.csv`
- `transform_selection_summary_by_context_and_root.csv`

## Cluster run order
The submitter intentionally uses explicit sequencing and does not use dependency jobs.

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2

MODE=inventory bash "$MODULE_DIR/submit_stage2a_steps1_3_rootaware_v1.sh"
# wait for inventory to finish

MODE=workers bash "$MODULE_DIR/submit_stage2a_steps1_3_rootaware_v1.sh"
# wait for all workers to finish

MODE=aggregate bash "$MODULE_DIR/submit_stage2a_steps1_3_rootaware_v1.sh"
```

## After aggregation
Do NOT run Stage 2A-4 yet. Review:
- root candidate counts,
- quality at candidate depths 5/10/15/20/30/50,
- OOF/evidence-score dropoff by root and context.

Then freeze root-specific nomination limits and build the new within-root Stage 2A-4/5.
