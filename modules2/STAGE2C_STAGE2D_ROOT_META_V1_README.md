# Stage 2C + Stage 2D root → meta-module workflow v1

## Scope

This bundle performs the fast, outcome-blind handoff from the manually frozen
Stage 2B-2 root-module Ks to discovery-cohort meta-modules.

### Stage 2C — freeze and score root modules

Inputs:
- Stage 2B-1 consensus/matrix outputs
- Stage 2B-2 v3 `stage2b2_final_k_selections.csv`

Stage 2C:
1. freezes the selected row-Spearman K solution for every panel × prep root;
2. relabels root modules in dendrogram order (`M01`, `M02`, ...);
3. reuses the already-built Stage 2B-1 discovery TURBT patient matrices;
4. z-scores each feature within cohort;
5. scores each root module as the mean of its available feature z-scores;
6. writes canonical discovery root-module score tables.

No outcome is read or used.

This v1 intentionally scores only the discovery TURBT matrices already cached
by Stage 2B-1. That is sufficient for Stage 2D meta-module discovery. After the
meta-module definitions are frozen, a downstream evaluation scorer should apply
the frozen definitions to RC, NAC2015, and the future validation cohort.

Primary outputs:
- `final_root_module_membership.csv`
- `final_root_module_summary.csv`
- `all_discovery_root_module_scores_long.parquet`
- `all_discovery_root_module_scores_wide.parquet`
- `stage2c_root_module_manifest.csv`

## Stage 2C run order

Place:
- `stage2c_score_root_modules_v1.py`
- `submit_stage2c_score_root_modules_v1.sh`

in:
`/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2`

Place config:
`stage2c_score_root_modules_v1.json`

in:
`/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2/configs`

Then run, waiting for each stage:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2

MODE=validate bash "$MODULE_DIR/submit_stage2c_score_root_modules_v1.sh"
MODE=setup bash "$MODULE_DIR/submit_stage2c_score_root_modules_v1.sh"
MODE=workers bash "$MODULE_DIR/submit_stage2c_score_root_modules_v1.sh"
MODE=aggregate bash "$MODULE_DIR/submit_stage2c_score_root_modules_v1.sh"
```

Completion check after workers:

```bash
ROOT=/projects/ovcare/users/nikolay_alabi/immuno/stage2_root_meta_modules_v1/stage2c_root_module_scores_v1

echo Expected:
awk 'END {print NR-1}' "$ROOT/stage2c_worker_index.csv"

echo Completed:
find "$ROOT/workers" -name .done | wc -l
```

---

## Stage 2D — outcome-blind meta-module discovery

Stage 2D uses **root-module patient scores**, not outcomes.

Within each panel separately:
1. calculate root-module score Spearman correlations in each discovery cohort;
2. equal-weight those cohort correlations into a signed consensus;
3. require pair support and cross-cohort sign consistency;
4. cluster the reproducibility-filtered consensus using **direct signed**
   distance (`1-rho`) because meta-module members will ultimately be averaged;
5. explore rho cut thresholds 0.20–0.50;
6. use average linkage as primary and complete linkage as sensitivity;
7. call a cluster a meta-module only when it contains >=2 root modules from
   >=2 distinct prep roots;
8. leave all other root modules as standalone programs;
9. score the primary meta-modules as the mean of within-cohort standardized
   root-module scores.

The default primary rho threshold is **0.30** for AR and BT. This is a
provisional, transparent starting value, not an outcome-selected threshold.
The script writes threshold-sensitivity plots/tables so it can be changed in the
JSON and rerun in minutes if the 0.30 solution is too coarse or fragmented.

Primary outputs:
- `final_meta_module_membership.csv`
- `final_meta_module_summary.csv`
- `stage2d_primary_meta_summary.csv`
- `stage2d_meta_threshold_diagnostics.csv`
- `all_discovery_meta_module_scores_long.parquet`
- `all_discovery_integrated_program_scores_long.parquet`
- `stage2d_manual_meta_review_template.csv`
- `stage2d_summary.txt`

Plots for each panel include:
- root-module consensus heatmap (red/blue)
- meta-clustering dendrogram
- rho-threshold sensitivity
- primary meta-module heatmap with prep-root + integrated-program annotation bars

## Stage 2D run

Place:
- `stage2d_discover_meta_modules_v1.py`
- `submit_stage2d_discover_meta_modules_v1.sh`

in `modules2`, and place `stage2d_discover_meta_modules_v1.json` in
`modules2/configs`.

After Stage 2C aggregate finishes:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules2
bash "$MODULE_DIR/submit_stage2d_discover_meta_modules_v1.sh"
```

The first Stage 2D files to review are:

```text
stage2d_summary.txt
stage2d_primary_meta_summary.csv
stage2d_meta_threshold_diagnostics.csv
final_meta_module_summary.csv
final_meta_module_membership.csv
plots/AR/04_primary_meta_module_heatmap.png
plots/BT/04_primary_meta_module_heatmap.png
```

If the primary 0.30 cut is not satisfactory, only edit:

```json
"primary_rho_thresholds": {
  "AR": 0.30,
  "BT": 0.30
}
```

and rerun Stage 2D. No Stage 2C rerun is needed.
