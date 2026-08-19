# Stage 2B cap-sensitivity workflow v1

This workflow compares candidate caps 10, 15, and 20 at semantic rho 0.90 while reusing the existing Stage 2A-4 patient matrices wherever possible.

## Design

1. `stage2b_setup_cap_sensitivity_v1.py` creates cap-specific S1 manifests and expands each selected feature across all four discovery cohorts.
2. `stage2b_build_shared_matrix_cache_v1.py` builds one cap-20 union cache:
   - reuses matching Stage 2A-4 matrix columns;
   - reconstructs only missing feature/cohort combinations through Stage 1;
   - saves global-feature-UID matrices for 4 cohorts x 2 panels.
3. `stage2b_prepare_global_module_inputs_cached_v1.py` subsets the shared cache for cap 10, 15, or 20 and runs consensus correlation, pair-support filtering, k diagnostics, and heatmaps.
4. `stage2b_compare_cap_sensitivity_v1.ipynb` compares the three prepared runs and quantifies module stability.

No Slurm script in this bundle contains a partition directive.

## Installation

Copy these files into the module directory:

```bash
MODULE_DIR=/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules
mkdir -p "$MODULE_DIR/configs/stage2b_cap_sensitivity"

cp stage2b_setup_cap_sensitivity_v1.py "$MODULE_DIR/"
cp stage2b_build_shared_matrix_cache_v1.py "$MODULE_DIR/"
cp stage2b_prepare_global_module_inputs_cached_v1.py "$MODULE_DIR/"
cp submit_stage2b_shared_matrix_cache_v1.sh "$MODULE_DIR/"
cp submit_stage2b_cap_sensitivity_v1.sh "$MODULE_DIR/"
cp stage2b_compare_cap_sensitivity_v1.ipynb "$MODULE_DIR/"
cp stage2b_cap_sensitivity_paths_v1.example.json \
  "$MODULE_DIR/configs/stage2b_cap_sensitivity/paths.json"

chmod +x "$MODULE_DIR"/stage2b_*_v1.py
chmod +x "$MODULE_DIR"/submit_stage2b_*_v1.sh
```

Also ensure the existing `stage2_global_module_utils_v7.py` remains in `MODULE_DIR`.

## Step 1: edit paths

Edit:

```text
$MODULE_DIR/configs/stage2b_cap_sensitivity/paths.json
```

The most important paths are:

- `grid_root`
- `output_root`
- `stage2a4_root`
- `stage2_global_module_utils_path`
- `stage1_script_path`
- `harmonized_path`

## Step 2: lightweight setup

Run in a notebook kernel with pandas, or an interactive CPU shell:

```bash
python "$MODULE_DIR/stage2b_setup_cap_sensitivity_v1.py" \
  --config "$MODULE_DIR/configs/stage2b_cap_sensitivity/paths.json"
```

The generated JSON files will initially be written under:

```text
<output_root>/configs/
```

Copy or symlink them to the submitter's expected config directory:

```bash
cp <output_root>/configs/*.json \
  "$MODULE_DIR/configs/stage2b_cap_sensitivity/"
```

## Step 3: build the shared cache on CPUs

```bash
CONFIG_JSON="$MODULE_DIR/configs/stage2b_cap_sensitivity/stage2b_shared_matrix_cache.json" \
CPUS=8 MEM=64G TIME=12:00:00 \
bash "$MODULE_DIR/submit_stage2b_shared_matrix_cache_v1.sh"
```

Wait for this job to finish. Check:

```text
<output_root>/shared_matrix_cache/shared_cache_context_summary.csv
<output_root>/shared_matrix_cache/shared_cache_reuse_summary.csv
<output_root>/shared_matrix_cache/shared_cache_build_failures.csv
```

`shared_cache_build_failures.csv` should be empty.

## Step 4: submit the three independent Stage 2B jobs

```bash
CONFIG_DIR="$MODULE_DIR/configs/stage2b_cap_sensitivity" \
CPUS=8 MEM=64G TIME=12:00:00 \
bash "$MODULE_DIR/submit_stage2b_cap_sensitivity_v1.sh"
```

No dependency job is used. The three jobs are independent and can run simultaneously after the shared cache is complete.

## Step 5: compare in the notebook

Open:

```text
$MODULE_DIR/stage2b_compare_cap_sensitivity_v1.ipynb
```

The notebook compares:

- input and support-filtered feature counts;
- k-selection diagnostics;
- module-size distributions;
- singleton and largest-module fractions;
- ARI and NMI across shared features;
- pairwise co-clustering agreement;
- module-level best-Jaccard matching across caps.

Choose the smallest cap whose module structure is stable relative to the next larger cap and that retains interpretable biological programs.
