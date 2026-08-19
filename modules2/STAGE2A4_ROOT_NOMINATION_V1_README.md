# Stage 2A-4 root nomination v1

This is the finalized root-aware nomination stage.

## Fixed caps

- AR / AR_checkpoint_state = 3
- AR / AR_state = 10
- AR / compartment = 5
- AR / compartment_state = 10
- AR / phenotype_only = 5
- BT / compartment = 5
- BT / phenotype_only = 10

Caps are maxima, not quotas. A context/root contributes fewer candidates when fewer pass the frozen context-level quality/stability criteria.

## Important design rules

- Candidates have already passed the frozen context thresholds in the cap-sensitivity setup.
- Selection is by `eligible_root_rank`, whose upstream ranking is based on `root_candidate_evidence_score` within each context/root.
- No cross-root rescue.
- No cross-root simplification.
- No correlation compression in Stage 2A-4.
- Root-specific transformed patient matrices are written for Stage 2A-5.
- The existing shared raw patient caches from the candidate-cap sensitivity run are reused, so Stage 1 feature sources do not need to be rebuilt again.

## Install

Copy:

- `stage2a4_root_nomination_v1.py` -> `modules2/`
- `submit_stage2a4_root_nomination_v1.sh` -> `modules2/`
- `stage2a_root_candidate_caps_v1.csv` -> `modules2/configs/`
- `stage2a4_root_nomination_v1.json` -> `modules2/configs/`

## Run

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2

MODE=validate bash "$MODULE_DIR/submit_stage2a4_root_nomination_v1.sh"
MODE=workers  bash "$MODULE_DIR/submit_stage2a4_root_nomination_v1.sh"
# wait until all workers finish
MODE=aggregate bash "$MODULE_DIR/submit_stage2a4_root_nomination_v1.sh"
```

## Key outputs

Output root:

`/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2a4_root_nomination_v1`

Aggregate outputs:

- `all_context_stage2a4_summary.csv`
- `all_context_root_seed_candidates.parquet`
- `all_context_root_nomination_summary.csv`
- `stage2a4_root_matrix_manifest.csv`
- `stage2a4_context_matrix_manifest.csv`
- `stage2a4_candidate_composition.csv`
- `stage2a4_feature_nomination_support.csv`
- `all_context_matrix_feature_build_audit.csv`
- `stage2a4_missing_context_outputs.csv`

Per context:

- `candidate_registry.csv`
- `root_nomination_summary.csv`
- `patient_feature_matrix.parquet`
- `root_matrix_manifest.csv`
- `roots/<prep_root>/patient_feature_matrix.parquet`
- `roots/<prep_root>/candidate_registry.csv`
- `roots/<prep_root>/matrix_feature_meta.csv`
