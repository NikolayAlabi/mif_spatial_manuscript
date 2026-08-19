#!/bin/bash
#SBATCH --job-name=core_qc
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=core_qc_%j.out
#SBATCH --error=core_qc_%j.err

source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6


# python -u generate_phenotype_abundance_consistency.py \
#   --tma_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_cohorts \
#   --whole_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_wholesections \
#   --blasst_metadata_csv /projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_BLASST.csv \
#   --out_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/phenotype_abundance_rebuild \
#   --existing_annotation_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments \
#   --qc_dir /projects/ovcare/users/nikolay_alabi/immuno/data \
#   --panels AR BT MY

# python -u build_cell_dfs_from_reviewed_phenotypes.py \
#   --tma_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_cohorts \
#   --whole_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_wholesections \
#   --phenotype_assignments_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/phenotype_abundance_rebuild \
#   --out_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/cell_df_rebuild \
#   --panels AR BT

python -u build_koll_cell_df.py \
  --ar_csv /projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/florestan_cell_seg_AR.csv \
  --bt_csv /projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/florestan_cell_seg_BT.csv \
  --out_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/cell_df_rebuild \
  --out_file koll_cell_df.parquet \
  --cohort KOLL \
  --other-collapse-label stroma \
  --state-panels AR BT