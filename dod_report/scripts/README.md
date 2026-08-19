cd /projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report

python scripts/plot_tissue_composition_by_panel_cohort.py \
  --qc-dir /projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild \
  --out-dir /projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/figure_outputs

python scripts/build_dod_report_table1.py   
    --clinical /projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv   
    --qc-dir /projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild   
    --out-dir /projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/table_outputs

cd /projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report

python scripts/plot_cell_counts_by_qc.py \
  --qc-dir /projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild \
  --out-dir /projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/figure_outputs