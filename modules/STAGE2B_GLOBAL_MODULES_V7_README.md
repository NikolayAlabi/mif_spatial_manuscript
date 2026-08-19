# Stage 2B global module discovery v7

## Files

- `stage2_global_module_utils_v7.py`  
  Shared functions for matrix reconstruction, consensus correlations, support filtering, ontology/semantic scoring, k diagnostics, heatmaps, and final module export.

- `stage2b_prepare_global_module_inputs_v7.py`  
  Heavy-prep script. Run before the notebook. It builds transformed patient matrices, consensus matrices, support-filtered matrices, k diagnostics, provisional heatmaps, and all intermediate CSV/parquet outputs.

- `submit_stage2b_prepare_global_modules_v7.sh`  
  Slurm submitter for the Stage 2B heavy-prep script.

- `stage2c_finalize_global_modules_v7.py`  
  Optional command-line finalizer after choosing final k. The notebook can also do this.

- `submit_stage2c_finalize_global_modules_v7.sh`  
  Optional Slurm submitter for finalizing modules.

- `stage2_global_modules_review_v7.ipynb`  
  Compact review notebook. It loads prepared outputs, displays diagnostics/heatmaps, lets you choose final k, and saves frozen module memberships.

- `configs/*.json`  
  Preset configs for primary TURBT expanded discovery, phenotype-only reproducibility, and RC exploratory discovery.

## Recommended first run

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules

bash "$MODULE_DIR/submit_stage2b_prepare_global_modules_v7.sh" primary_turbt_expanded
```

## Dry run

```bash
DRY_RUN=1 bash "$MODULE_DIR/submit_stage2b_prepare_global_modules_v7.sh" primary_turbt_expanded
```

## After Stage 2B completes

Open:

```text
/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/stage2_global_modules_review_v7.ipynb
```

Set:

```python
FINAL_K = {"AR": <chosen_k>, "BT": <chosen_k>}
```

Then run the final cells to save frozen modules.

## Optional command-line finalize

```bash
FINAL_K="AR=24,BT=12" bash "$MODULE_DIR/submit_stage2c_finalize_global_modules_v7.sh" primary_turbt_expanded
```

## Main prepared outputs

```text
<output_root>/<PANEL>/<PANEL>_consensus_similarity.parquet
<output_root>/<PANEL>/<PANEL>_pair_support.parquet
<output_root>/<PANEL>/<PANEL>_consensus_similarity_support_filtered.parquet
<output_root>/<PANEL>/<PANEL>_feature_support_summary.csv
<output_root>/<PANEL>/<PANEL>_k_selection_diagnostics.csv
<output_root>/<PANEL>/<PANEL>_memberships_all_k.csv
<output_root>/<PANEL>/<PANEL>_feature_ontology.csv
<output_root>/plots/<PANEL>/*.png
```

## Final module outputs

```text
<output_root>/final_modules/<PANEL>/<PANEL>_global_module_memberships_k<K>.csv
<output_root>/final_modules/<PANEL>/<PANEL>_dendrogram_k_memberships.csv
<output_root>/final_modules/<PANEL>/<PANEL>_module_summary_k<K>.csv
<output_root>/final_modules/<PANEL>/<PANEL>_module_representatives_k<K>.csv
```
