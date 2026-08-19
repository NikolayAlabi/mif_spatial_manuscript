# Stage 2A candidate-cap sensitivity v1

This workflow chooses candidate depth only after applying the frozen context-level quality/stability rules.
It then evaluates whether increasing candidate depth mainly adds new information or near-duplicate features.

## Architecture

1. `setup`: reads the huge best-transform parquet once, applies context rules, ranks within prep root, and writes compact shards.
2. `cache-worker`: one CPU per cohort/panel matrix context; builds the union raw patient matrix once across endpoints.
3. `worker`: one CPU per endpoint context/panel; computes quality/stability/redundancy curves and cap recommendations.
4. `aggregate`: combines all contexts and provides panel x prep-root consensus recommendations plus a manual review template.

Spearman correlations use raw patient-level feature values because z-score and log1p-z-score are monotonic transformations and therefore do not change rank correlation.

## Primary diagnostics

For every context x prep root and candidate depth:

- Nth-candidate OOF margin above the hard context threshold
- Nth-candidate fold-SD headroom below the hard threshold
- root evidence-rank profile
- median / 90th-percentile / maximum within-root |Spearman rho|
- fraction of candidate pairs with sufficient pairwise patient support
- fraction of candidates involved in a near-redundant pair
- greedy cumulative nonredundant candidate count
- nonredundant novelty yield = nonredundant count / nominated depth

Redundancy is evaluated at |rho| >= 0.85, 0.90, and 0.95; 0.90 is primary.

## Mathematical advisory cap

All candidates already pass the context quality/stability thresholds. Candidate-depth selection is therefore based on redundancy/diminishing returns.

Three constraint profiles are reported:

- strict: novelty yield >= 0.80
- balanced: novelty yield >= 0.70
- permissive: novelty yield >= 0.60

Within a profile, the algorithm chooses the depth that maximizes cumulative nonredundant candidates while satisfying the yield constraint; ties prefer the shallower depth.

A separate sustained-redundancy elbow is detected when two consecutive 5-candidate windows each add <=2 new nonredundant candidates. The advisory recommendation is the smaller of the balanced constraint cap and the redundancy elbow.

These numbers are deliberately exposed in the config and should be treated as a sensitivity framework, not as hidden tuning.

## Install

Copy these files to `/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2`:

- `stage2a_candidate_cap_sensitivity_v1.py`
- `worker_stage2a_candidate_cap_cache_v1.sh`
- `worker_stage2a_candidate_cap_context_v1.sh`
- `submit_stage2a_candidate_cap_sensitivity_v1.sh`

Copy the JSON into `modules2/configs/`.

## Run

Run each step explicitly and wait for completion before the next:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 \
MODE=setup bash $MODULE_DIR/submit_stage2a_candidate_cap_sensitivity_v1.sh
```

Then:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 \
MODE=cache bash $MODULE_DIR/submit_stage2a_candidate_cap_sensitivity_v1.sh
```

Then:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 \
MODE=workers bash $MODULE_DIR/submit_stage2a_candidate_cap_sensitivity_v1.sh
```

Finally:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2 \
MODE=aggregate bash $MODULE_DIR/submit_stage2a_candidate_cap_sensitivity_v1.sh
```

## Most useful outputs

Per context/root:

- `roots/<root>/plots/00_candidate_cap_dashboard.png`
- `roots/<root>/cap_recommendation.csv`
- `roots/<root>/cap_recommendation.txt`
- `roots/<root>/depth_diagnostics_all_integer_depths.csv`
- `roots/<root>/pairwise_correlations_top_max_depth.csv`

Aggregate:

- `cap_manual_review_template.csv`
- `all_context_root_cap_recommendations.csv`
- `panel_root_consensus_cap_recommendations.csv`
- `cap_recommendations_summary.txt`
- `aggregate_plots/AR/`
- `aggregate_plots/BT/`

The `cap_manual_review_template.csv` includes blank `manual_candidate_cap` and `manual_notes` columns. It also carries the context-specific advisory recommendation and the panel x prep-root consensus recommendation.
