#!/bin/bash
#SBATCH --job-name=core_qc
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=core_qc_%j.out
#SBATCH --error=core_qc_%j.err

source /home/nalabi/miniconda3/etc/profile.d/conda.sh
conda activate cuda6

# python qc_verify_cores_and_split_by_cohort.py \
#   --tissue-area /projects/ovcare/users/nikolay_alabi/immuno/data/Data_NAC_NoNAC_PURE01_NAC2_TissueArea.csv \
#   --clinical    /projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv \
#   --review-dir  /projects/ovcare/users/nikolay_alabi/immuno/data \
#   --panels AR,BT,MY \
#   --chunksize 500000 \
#   --outdir /projects/ovcare/users/nikolay_alabi/immuno/data/qc_runs/phase1_core_qc

python qc_check.py \
  --inform_summary_dir /projects/ovcare/users/nikolay_alabi/immuno/data/inform_summary_rebuild \
  --tma_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_cohorts \
  --whole_parquet_dir /projects/ovcare/users/nikolay_alabi/immuno/data/raw_phenoptr/combined_wholesections \
  --data_dir /projects/ovcare/users/nikolay_alabi/immuno/data \
  --tma_clinical /projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv \
  --blasst_clinical /projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_BLASST.csv \
  --koll_dir /projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort \
  --out_dir /projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild
