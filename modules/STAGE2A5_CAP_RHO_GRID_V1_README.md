# Stage 2A-5 cap × semantic-rho sensitivity grid

This bundle uses the completed **Stage 2A-4 top-100 matrices as a read-only master superset**. It does not rerun Stage 1 or rebuild patient-level features.

## Grid evaluated

- Candidate cap per context: **5, 10, 15, …, 100**
- Semantic Spearman threshold: **0.85, 0.86, …, 0.95**
- Total: **220 grid cells per context**

The semantic threshold is applied identically to:

1. state simplification;
2. metric-summary simplification; and
3. compartment simplification.

The following are fixed so that the grid changes only one microcompression parameter:

- residual Spearman threshold: **0.95** for AR and BT;
- maximum OOF loss: **0.01**;
- minimum pairwise-complete patients: **20**.

The script submits one Slurm worker per context. Each worker evaluates all 220 combinations, avoiding thousands of small Slurm jobs.

## Files

- `stage2a5_cap_rho_grid_v1.py`
- `stage2a5_cap_rho_grid_config_v1.json`
- `submit_stage2a5_cap_rho_grid_v1.sh`
- `export_stage2b_manifest_from_grid_v1.sh`

## Install

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules
mkdir -p "$MODULE_DIR/configs"

cp stage2a5_cap_rho_grid_v1.py "$MODULE_DIR/"
cp submit_stage2a5_cap_rho_grid_v1.sh "$MODULE_DIR/"
cp export_stage2b_manifest_from_grid_v1.sh "$MODULE_DIR/"
cp stage2a5_cap_rho_grid_config_v1.json "$MODULE_DIR/configs/"

chmod +x \
  "$MODULE_DIR/stage2a5_cap_rho_grid_v1.py" \
  "$MODULE_DIR/submit_stage2a5_cap_rho_grid_v1.sh" \
  "$MODULE_DIR/export_stage2b_manifest_from_grid_v1.sh"
```

Confirm that the JSON points to the completed top-100 Stage 2A-4 output:

```json
"stage2a4_output_root": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/stage2a4_filtered_context_matrices"
```

The grid has its own output root and will not overwrite the prior Stage 2A-5 runs:

```json
"output_root": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/stage2a5_cap_rho_grid_v1"
```

## Validate

Run on a compute node:

```bash
srun --partition=upgrade --cpus-per-task=1 --mem=8G --time=01:00:00 --pty bash

export PS1="${PS1-}"
source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules
python "$MODULE_DIR/stage2a5_cap_rho_grid_v1.py" validate \
  --config "$MODULE_DIR/configs/stage2a5_cap_rho_grid_config_v1.json"
```

Expected:

```text
n_caps: 20
n_rhos: 11
n_grid_cells_per_context: 220
problems: []
```

## Submit

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules

CONFIG_JSON="$MODULE_DIR/configs/stage2a5_cap_rho_grid_config_v1.json" \
MAX_CONCURRENT=8 \
WORKER_MEM=16G \
WORKER_TIME=16:00:00 \
bash "$MODULE_DIR/submit_stage2a5_cap_rho_grid_v1.sh"
```

The submitter:

1. runs inventory on a compute node;
2. submits one array worker per context;
3. submits aggregation with an `afterok` dependency.

Completed contexts are skipped automatically on rerun. To deliberately rerun all workers:

```bash
FORCE_WORKERS=1 SKIP_INVENTORY=1 \
CONFIG_JSON="$MODULE_DIR/configs/stage2a5_cap_rho_grid_config_v1.json" \
bash "$MODULE_DIR/submit_stage2a5_cap_rho_grid_v1.sh"
```

## Support definition

Support is calculated **after microcompression and canonical duplicate collapsing**, separately for AR and BT.

For canonical feature \(f\):

```text
context support = number of distinct endpoint contexts selecting f
cohort support  = number of distinct cohorts selecting f
endpoint support = number of distinct endpoint names selecting f
```

Example:

```text
NAC2020 complete response + NAC2020 any response
context support = 2
cohort support = 1
```

```text
NAC2020 complete response + PURE01 complete response
context support = 2
cohort support = 2
```

The primary support definition is **cohort support**:

- `S1`: cohort support ≥1; the complete canonical candidate universe.
- `S2`: cohort support ≥2; independently recurrent candidates.
- `S2E`: `S2` plus a limited exceptional single-cohort set.

This is **candidate recurrence support**. It is different from Stage 2B pair support, which counts the cohorts in which a feature-pair correlation can be estimated.

## Exceptional single-cohort rule

A single-cohort candidate is eligible only when:

1. it is supported by a context classified as strong; and
2. it either recurs in at least two contexts within that cohort or falls in the top 5% of panel/grid evidence scores.

Selection is capped at:

- 10 exceptional candidates per panel;
- 3 from any one cohort.

These values are editable in the JSON and are applied identically to every grid cell.

## Main aggregate outputs

```text
stage2a5_cap_rho_grid_v1/
├── grid_context_counts.csv
├── grid_cohort_counts.csv
├── grid_panel_summary_metrics.csv
├── grid_joint_quantity_screen.csv
├── candidate_support_grid.parquet
├── exceptional_single_cohort_candidates_grid.csv
├── candidate_set_S2_plus_E_grid.csv
├── all_grid_context_manifest_raw.parquet
├── all_grid_context_manifest_deduplicated.parquet
├── grid_plot_index.csv
└── plots/
```

Important counts in `grid_panel_summary_metrics.csv`:

- `n_seed_records`: seeds submitted across contexts;
- `n_postcompression_records_raw`: context-level representatives after microcompression;
- `n_postcompression_records_after_within_context_dedup`: after removing duplicate canonical representations within the same context;
- `n_unique_canonical_features`: after collapsing across contexts and cohorts;
- `n_S1`;
- `n_S2`;
- `n_exceptional_single_cohort`;
- `n_S2_plus_E`;
- `proportion_support_ge2`;
- `max_single_cohort_owned_fraction`.

`grid_joint_quantity_screen.csv` places AR and BT side by side and ranks grid cells using **quantity guardrails only**. It does not declare a final scientific optimum. Module quality and module stability must be evaluated in Stage 2B for shortlisted cells.

## Plots

For each panel, the aggregation creates grid heatmaps for:

1. total seeds;
2. post-microcompression context records;
3. unique canonical candidates;
4. candidates with support in at least two cohorts;
5. `S2 + exceptional` count;
6. proportion with ≥2-cohort support;
7. duplicate-collapse fraction;
8. largest single-cohort-only contribution.

It also creates:

- candidate-count curves for every semantic threshold;
- seed-count heatmaps by context;
- postcompression heatmaps by context for every semantic threshold;
- cohort-specific contribution heatmaps.

## Export a shortlisted cell for Stage 2B

Example: cap 25, semantic rho 0.90, using `S2E`:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules
bash "$MODULE_DIR/export_stage2b_manifest_from_grid_v1.sh" 25 0.90 S2E
```

The default export path is:

```text
.../stage2a5_cap_rho_grid_v1/exports/cap025__rho0p9/S2E/
```

The Stage 2B-ready file is:

```text
global_module_candidate_manifest.csv
```

The export assigns one globally consistent `feature_uid` to each canonical feature while retaining each context's local source/group/feature fields for patient-matrix reconstruction.

## Decision sequence

1. Review the full count and support grid.
2. Exclude cells with implausibly large or tiny `S2E` candidate counts.
3. Exclude cells dominated by single-cohort-only candidates.
4. Select a small neighborhood of adjacent cap/rho cells.
5. Run Stage 2B consensus clustering for that neighborhood.
6. Choose the smallest cap and most conservative rho on the module-stability plateau.

The default quantity guardrails are deliberately configurable:

```json
"min_S2_plus_E_per_panel": 60,
"max_S2_plus_E_per_panel": 125,
"min_proportion_support_ge2": 0.20,
"max_single_cohort_owned_fraction": 0.40
```

They are screening criteria, not the final basis for choosing the global modules.
