# Raw Preprocessing

This directory contains the earliest identified preprocessing scripts used to reconstruct multiplex immunofluorescence (mIF) datasets from raw inForm exports. These scripts represent the upstream portion of the biomarker discovery pipeline and are intended to document the provenance of the feature tables ultimately used for univariate and multivariable analyses in the MIBC spatial immune ecosystem study.

## Overall Pipeline (Current Understanding)

```text
Raw inForm exports
    (*_cell_seg_data.txt)
    (*tissue_seg_data_summary.txt)
            ↓

combine_cohort_from_raw.py
combine_wholesection_panel_batch.py

            ↓

Combined per-cell phenotype tables
(parquet format)

            ↓

Phenotype harmonization / collapse
(under investigation)

            ↓

Core-level feature generation
    • Cell densities
    • Cell ratios
    • Spatial NN features
    • ATHENA features

            ↓

stage1_univariate_cv_screen.py

            ↓

Feature selection
Module discovery
Predictive modeling
```

---

# Script Descriptions

## combine_cohort_from_raw.py

### Purpose

Combines raw TMA-level inForm exports from multiple marker algorithms into a single per-cell phenotype table.

Each marker algorithm generates a separate `*_cell_seg_data.txt` file containing phenotype assignments for the same cells. This script merges those marker calls into a unified phenotype representation for each cell.

### Inputs

Raw inForm exports:

```text
*_cell_seg_data.txt
```

Required fields:

```text
Sample Name
Tissue Category
Phenotype
Cell ID
Cell X Position
Cell Y Position
```

### Processing

For each TMA core:

1. Loads all marker-specific cell segmentation files.
2. Matches cells using:

```text
sample_name
tissue_category
cell_id
x
y
```

3. Combines marker calls into:

```text
phenotype_combined
```

4. Calculates:

```text
n_positive_markers
any_blank_call
```

5. Optionally removes cells with incomplete marker assignments.

### Outputs

Partitioned parquet files:

```text
part-00000.parquet
part-00001.parquet
...
```

Optional merged parquet:

```text
combined_cells.parquet
```

### Key Output Columns

```text
sample_name
tissue_category
cell_id
x
y
phenotype_combined
n_positive_markers
core_key
```

---

## combine_wholesection_panel_batch.py

### Purpose

Equivalent workflow for whole-section datasets.

Processes whole-section inForm exports organized by:

```text
Panel
Batch
Region
```

rather than TMA core.

### Supported Panels

```text
AR
B&T
Myeloid
```

### Processing

For each region:

1. Loads all marker algorithm outputs.
2. Merges marker calls.
3. Generates unified cell phenotypes.
4. Writes partitioned parquet outputs.

### Outputs

```text
WholeSections_<Panel>_Batch<N>/
    part-00000.parquet
    part-00001.parquet
```

Optional merged parquet.

### Key Output Columns

```text
sample_name
tissue_category
cell_id
x
y
phenotype_combined
n_positive_markers
region_id
panel
batch
```

---

# Outstanding Questions

The following downstream steps have not yet been fully reconstructed:

## Phenotype Harmonization

Unknown scripts that converted:

```text
phenotype_combined
```

into the final collapsed phenotype labels used throughout the project.

Examples include:

```text
t_cell
treg_cell
macrophage
b_cell
plasma_cell
nk_cell
tumour
ALL_NEG
```

---

## Feature Generation

Need to identify scripts responsible for generating:

### Density / Ratio Features

Output examples:

```text
cell_features_per_sample_core.csv
```

### Nearest-Neighbour Features

Output examples:

```text
NNstats.tsv
```

### ATHENA Features

Output examples:

```text
athena_features.csv
```

---

## Stage1 Input Assembly

Need to identify the exact scripts that merged:

```text
Clinical metadata
QC information
Tissue segmentation summaries
Density/ratio features
NN features
ATHENA features
```

into the final feature matrix used by:

```text
stage1_univariate_cv_screen.py
```

---

# Provenance Status

| Step                             | Status                              |
| -------------------------------- | ----------------------------------- |
| Raw inForm exports               | Identified                          |
| Cell-level phenotype combination | Identified                          |
| Tissue segmentation extraction   | Partially identified                |
| Phenotype harmonization          | Unknown                             |
| Density/ratio feature generation | Unknown                             |
| NN feature generation            | Identified outputs, scripts unknown |
| ATHENA feature generation        | Identified outputs, scripts unknown |
| Stage1 input assembly            | Partially identified                |
| Univariate screening             | Identified                          |
| Module discovery                 | Identified                          |
| Predictive modeling              | Identified                          |

```
```
