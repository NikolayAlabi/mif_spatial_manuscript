#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build expanded DOD report Table 1 with clinical, survival, and mIF QC/core-section metrics.

Inputs
------
1) harmonized_modeling_dataframe.csv
   Patient-level clinical dataframe.

2) qc_check_rebuild directory
   Expected files:
     - source_presence_all.csv
     - coverage_summary_by_panel_cohort_tma_sample_type.csv
     - clinical_inventory_tma.csv
     - clinical_inventory_blasst.csv

Optional but recommended:
3) original core-level clinical CSVs used in qc_check.py
   - ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv
   - ClinicalData_Core_BLASST.csv

Why optional?
-------------
source_presence_all.csv is a coord/panel-level QC matrix. Depending on how
qc_check.py was run, it may or may not retain patient IDs for all TMA cohorts.
The original core-level clinical files usually provide the best coord -> patient
mapping. This script uses them when supplied, and falls back to the best
available identifiers in source_presence_all.csv otherwise.

Definitions
-----------
- mIF submitted patient: patient with >=1 physical core/section coord submitted
  for AR/BT in source_presence_all.csv.
- mIF analysis-ready patient after QC: patient with >=1 physical core/section
  coord that is source-complete and QC-acceptable.
- physical core/section: unique coord per patient/sample_type/cohort. For BLASST
  this refers to whole-section regions; for TMA cohorts, TMA cores.
- panel unit: coord x panel entity. Used only in supplemental QC table.
- QC-acceptable TMA unit: Acceptable or Borderline structural_acceptability.
  BLASST does not use TMA-style manual review, so BLASST units are retained if
  source-complete.

Outputs
-------
- table1_report_by_cohort_expanded_wide.csv
- table1_report_by_cohort_expanded_compact.csv
- supp_table_mif_patient_core_counts_by_cohort_sample_type.csv
- supp_table_mif_panel_unit_qc_by_cohort_sample_panel.csv
- source_data_table1_clinical_expanded.csv
- source_data_table1_mif_patient_unit_level.csv
- table1_expanded_data_dictionary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REPORT_COHORTS = ["NAC2020", "PURE01", "No-NAC", "BLASST"]
REPORT_PANELS = ["AR", "BT"]
SAMPLE_TYPES = ["TURBT", "RC"]
GOOD_QC = {"Acceptable", "Borderline"}

DEFAULT_CLINICAL = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv")
DEFAULT_QC_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild")
DEFAULT_TMA_CLINICAL = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv")
DEFAULT_BLASST_CLINICAL = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_BLASST.csv")
DEFAULT_OUT_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/table_outputs")

COORD_RE = re.compile(r"\[\s*(\d{3,})\s*,\s*(\d{3,})\s*\]")


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_cohort(x) -> str:
    s = norm_str(x)
    su = s.upper().replace("_", " ").replace("-", "")
    if su in {"NAC2020", "BCA2020", "BCA 2020"}:
        return "NAC2020"
    if su in {"PURE01"}:
        return "PURE01"
    if su in {"NONAC", "NO NAC"}:
        return "No-NAC"
    if su in {"BLASST"}:
        return "BLASST"
    if su in {"NAC2015"} or su.startswith("BLADDER"):
        return "NAC2015"
    if su in {"KOLL", "FLORESTAN"}:
        return "KOLL"
    return s


def normalize_panel(x) -> str:
    s = norm_str(x).upper()
    if s in {"AR", "ARP"}:
        return "AR"
    if s in {"BT", "B&T", "B+T", "B_T", "B T"}:
        return "BT"
    if s in {"MY", "M", "MYELOID"}:
        return "MY"
    return norm_str(x)


def normalize_sample_type(x) -> str:
    s = norm_str(x).upper()
    if "TURBT" in s or "TUR" in s:
        return "TURBT"
    if s == "RC" or "CYST" in s or "RADICAL" in s:
        return "RC"
    return norm_str(x) if norm_str(x) else "Unknown"


def normalize_qc(x) -> str:
    s = norm_str(x)
    if s == "" or s.lower() in {"nan", "none", "na", "<na>", "unknown"}:
        return "Missing QC"
    sl = s.lower()
    if sl in {"acceptable", "accepted", "accept", "good", "pass", "usable"}:
        return "Acceptable"
    if sl in {"borderline", "borderline usable", "borderline/usable"}:
        return "Borderline"
    if sl in {"unusable", "fail", "failed", "bad", "reject", "rejected"}:
        return "Unusable"
    return s


def extract_coord_token(x) -> Optional[str]:
    if pd.isna(x):
        return None
    matches = COORD_RE.findall(str(x))
    if not matches:
        return None
    a, b = matches[-1]
    return f"[{a},{b}]"


def clean_percent(n: int | float, denom: int | float) -> str:
    if denom is None or pd.isna(denom) or denom == 0:
        return f"{int(n) if pd.notna(n) else 0}"
    pct = 100.0 * float(n) / float(denom)
    return f"{int(n)} ({pct:.1f}%)"


def fmt_n(x) -> str:
    if pd.isna(x):
        return "0"
    return f"{int(round(float(x))):,}"


def fmt_float(x, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_median_iqr(series: pd.Series, digits: int = 1) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def normalize_binary(s: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        val = str(x).strip().lower()
        if val in {"1", "true", "yes", "y", "pos", "positive", "event", "responder", "response", "cr", "pr"}:
            return 1
        if val in {"0", "false", "no", "n", "neg", "negative", "none", "nonresponder", "non-responder", "stable", "sd", "pd"}:
            return 0
        try:
            return int(float(val))
        except Exception:
            return np.nan
    return s.map(f)


def parse_stage_ge3(s: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return np.nan
        val = str(x).upper().strip()
        nums = re.findall(r"\d+", val)
        if not nums:
            return np.nan
        return int(max(map(int, nums)) >= 3)
    return s.map(f)


def parse_node_positive(s: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return np.nan
        val = str(x).upper().strip()
        if val in {"N0", "PN0", "CN0", "0"} or re.search(r"N\s*0", val):
            return 0
        if re.search(r"N\s*[1-9]", val):
            return 1
        try:
            return int(float(val) > 0)
        except Exception:
            return np.nan
    return s.map(f)


def parse_pt0(s: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return np.nan
        val = str(x).upper().strip()
        if re.search(r"T\s*0", val) or val in {"0", "PT0", "YPT0"}:
            return 1
        if re.search(r"T\s*[1-9]", val):
            return 0
        try:
            return int(float(val) == 0)
        except Exception:
            return np.nan
    return s.map(f)


def km_median(time: pd.Series, event: pd.Series) -> Optional[float]:
    """Simple Kaplan-Meier median. Returns None if median not reached."""
    t = pd.to_numeric(time, errors="coerce")
    e = normalize_binary(event)
    d = pd.DataFrame({"time": t, "event": e}).dropna()
    d = d[d["time"] >= 0].sort_values("time")
    if d.empty:
        return None
    surv = 1.0
    for ti, g in d.groupby("time", sort=True):
        at_risk = int((d["time"] >= ti).sum())
        n_events = int(g["event"].sum())
        if at_risk > 0:
            surv *= (1.0 - n_events / at_risk)
        if surv <= 0.5:
            return float(ti)
    return None


def fmt_km_median(time: pd.Series, event: pd.Series, digits: int = 1) -> str:
    med = km_median(time, event)
    if med is None or pd.isna(med):
        return "NR"
    return f"{med:.{digits}f}"


def choose_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_clinical(clinical_path: Path, cohorts: list[str]) -> pd.DataFrame:
    df = pd.read_csv(clinical_path, low_memory=False)
    if "cohort" not in df.columns:
        raise ValueError(f"{clinical_path} must contain a 'cohort' column")
    df = df.copy()
    df["cohort"] = df["cohort"].map(normalize_cohort)
    df = df[df["cohort"].isin(cohorts)].copy()
    if "patient_id" not in df.columns:
        raise ValueError(f"{clinical_path} must contain a 'patient_id' column")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df[df["patient_id"].ne("")].copy()
    return df


def treatment_context(cohort: str, sub: pd.DataFrame) -> str:
    if "NAT" in sub.columns:
        vals = [norm_str(x) for x in sub["NAT"].dropna().unique() if norm_str(x)]
        if vals:
            return "; ".join(sorted(vals))
    mapping = {
        "NAC2020": "NAC",
        "PURE01": "IO",
        "No-NAC": "RC-only / no NAT",
        "BLASST": "NAC/IO",
    }
    return mapping.get(cohort, "")


def summarize_clinical(df: pd.DataFrame, cohorts: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        sub = df[df["cohort"].eq(cohort)].copy()
        n_pat = sub["patient_id"].nunique()
        row = {"cohort": cohort, "clinical_patients_n": n_pat, "treatment_context": treatment_context(cohort, sub)}

        if "Age" in sub.columns:
            row["age_median_iqr"] = fmt_median_iqr(sub.drop_duplicates("patient_id")["Age"], 1)
        if "Sex" in sub.columns:
            sex = sub.drop_duplicates("patient_id")["Sex"].astype(str).str.lower()
            male = sex.isin(["m", "male", "man", "1"]).sum()
            denom = sex.notna().sum()
            row["male_n_pct"] = clean_percent(male, denom)
        if "variant" in sub.columns:
            var = sub.drop_duplicates("patient_id")["variant"]
            has_var = (~var.isna()) & (~var.astype(str).str.strip().str.lower().isin(["", "nan", "none", "no", "0", "uc", "pure uc", "urothelial"]))
            row["variant_histology_n_pct"] = clean_percent(int(has_var.sum()), int(var.notna().sum()))
        if "cT" in sub.columns:
            b = parse_stage_ge3(sub.drop_duplicates("patient_id")["cT"])
            row["clinical_T_ge3_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "pT" in sub.columns:
            b = parse_pt0(sub.drop_duplicates("patient_id")["pT"])
            row["pathologic_T0_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "pN" in sub.columns:
            b = parse_node_positive(sub.drop_duplicates("patient_id")["pN"])
            row["pathologic_node_positive_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "any_response" in sub.columns:
            b = normalize_binary(sub.drop_duplicates("patient_id")["any_response"])
            row["any_response_evaluable_n"] = int(b.notna().sum())
            row["any_response_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "complete_response" in sub.columns:
            b = normalize_binary(sub.drop_duplicates("patient_id")["complete_response"])
            row["complete_response_evaluable_n"] = int(b.notna().sum())
            row["complete_response_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "adjuvant_chemo" in sub.columns:
            b = normalize_binary(sub.drop_duplicates("patient_id")["adjuvant_chemo"])
            row["adjuvant_chemo_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))
        if "adjuvant_IO" in sub.columns:
            b = normalize_binary(sub.drop_duplicates("patient_id")["adjuvant_IO"])
            row["adjuvant_IO_n_pct"] = clean_percent(int(b.sum(skipna=True)), int(b.notna().sum()))

        # Survival: prefer RC-origin time, fallback to TURBT-origin time.
        os_time_rc = sub["OS_months_RC"] if "OS_months_RC" in sub.columns else pd.Series(np.nan, index=sub.index)
        os_time_tur = sub["OS_months_TUR"] if "OS_months_TUR" in sub.columns else pd.Series(np.nan, index=sub.index)
        os_time = pd.to_numeric(os_time_rc, errors="coerce").combine_first(pd.to_numeric(os_time_tur, errors="coerce"))
        if "OS_event" in sub.columns:
            os_dat = sub[["patient_id", "OS_event"]].copy()
            os_dat["os_time"] = os_time
            os_dat = os_dat.drop_duplicates("patient_id")
            e = normalize_binary(os_dat["OS_event"])
            t = pd.to_numeric(os_dat["os_time"], errors="coerce")
            mask = t.notna() & e.notna()
            row["OS_evaluable_n"] = int(mask.sum())
            row["OS_events_n_pct"] = clean_percent(int(e[mask].sum(skipna=True)), int(mask.sum()))
            row["OS_time_median_iqr_months"] = fmt_median_iqr(t[mask], 1)
            row["KM_median_OS_months"] = fmt_km_median(t[mask], e[mask], 1)

        rfs_time_rc = sub["REC_months_RC"] if "REC_months_RC" in sub.columns else pd.Series(np.nan, index=sub.index)
        rfs_time_tur = sub["REC_months_TURBT"] if "REC_months_TURBT" in sub.columns else pd.Series(np.nan, index=sub.index)
        rfs_time = pd.to_numeric(rfs_time_rc, errors="coerce").combine_first(pd.to_numeric(rfs_time_tur, errors="coerce"))
        if "REC" in sub.columns:
            rfs_dat = sub[["patient_id", "REC"]].copy()
            rfs_dat["rfs_time"] = rfs_time
            rfs_dat = rfs_dat.drop_duplicates("patient_id")
            e = normalize_binary(rfs_dat["REC"])
            t = pd.to_numeric(rfs_dat["rfs_time"], errors="coerce")
            mask = t.notna() & e.notna()
            row["RFS_evaluable_n"] = int(mask.sum())
            row["RFS_events_n_pct"] = clean_percent(int(e[mask].sum(skipna=True)), int(mask.sum()))
            row["RFS_time_median_iqr_months"] = fmt_median_iqr(t[mask], 1)
            row["KM_median_RFS_months"] = fmt_km_median(t[mask], e[mask], 1)

        rows.append(row)
    return pd.DataFrame(rows)


def load_coord_patient_map(path: Optional[Path], source: str) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["coord", "patient_id_from_core", "sample_type_from_core", "source"])
    df = pd.read_csv(path, low_memory=False).copy()
    out = pd.DataFrame()
    # coord from Core or image/sample-like columns
    coord_source = choose_first_col(df, ["coord", "Coord", "Core", "image", "sample_name", "Sample Name"])
    if coord_source is None:
        return pd.DataFrame(columns=["coord", "patient_id_from_core", "sample_type_from_core", "source"])
    out["coord"] = df[coord_source].map(extract_coord_token)

    pid_col = choose_first_col(df, [
        "patient_id", "Patient_ID", "patient_ID", "Sample_ID_Adjusted", "Sample_ID", "sample_id",
        "case_id", "Case_ID", "ID", "patient", "Patient"
    ])
    if pid_col is not None:
        out["patient_id_from_core"] = df[pid_col].astype(str).str.strip()
    else:
        # Fallback: strip coordinate and core token from the Core/sample string. This is not perfect,
        # so the output also reports how many rows required fallback IDs.
        raw = df[coord_source].astype(str)
        tmp = raw.str.replace(r"_?\[\s*\d{3,}\s*,\s*\d{3,}\s*\].*$", "", regex=True)
        tmp = tmp.str.replace(r"_?Core\[.*$", "", regex=True)
        out["patient_id_from_core"] = tmp.str.strip()

    st_col = choose_first_col(df, ["sample_type", "TURBT_or_RC", "Specimen", "specimen", "timepoint"])
    if st_col is not None:
        out["sample_type_from_core"] = df[st_col].map(normalize_sample_type)
    else:
        out["sample_type_from_core"] = "Unknown"

    out["source"] = source
    out = out.dropna(subset=["coord"])
    out = out[out["patient_id_from_core"].astype(str).str.strip().ne("")].copy()
    out = out.drop_duplicates(["coord", "patient_id_from_core", "sample_type_from_core"])
    return out


def build_mif_unit_table(
    qc_dir: Path,
    cohorts: list[str],
    panels: list[str],
    tma_clinical: Optional[Path] = None,
    blasst_clinical: Optional[Path] = None,
) -> pd.DataFrame:
    fp = qc_dir / "source_presence_all.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    d = pd.read_csv(fp, low_memory=False)
    d = d.copy()
    for c in ["cohort_label", "panel", "sample_type", "branch", "coord"]:
        if c not in d.columns:
            d[c] = pd.NA
    d["cohort"] = d["cohort_label"].map(normalize_cohort)
    d["panel"] = d["panel"].map(normalize_panel)
    d["sample_type"] = d["sample_type"].map(normalize_sample_type)
    d = d[d["cohort"].isin(cohorts)].copy()
    d = d[d["panel"].isin(panels)].copy()
    d = d[d["coord"].notna()].copy()

    # Merge better coord->patient maps when available.
    maps = []
    maps.append(load_coord_patient_map(tma_clinical, "tma_clinical"))
    maps.append(load_coord_patient_map(blasst_clinical, "blasst_clinical"))
    cmap = pd.concat(maps, ignore_index=True, sort=False) if maps else pd.DataFrame()
    if not cmap.empty:
        # If duplicate coord has multiple patient IDs, keep first but flag duplicates.
        dup_counts = cmap.groupby("coord")["patient_id_from_core"].nunique().rename("n_patient_ids_for_coord").reset_index()
        cmap1 = cmap.sort_values(["coord", "source"]).drop_duplicates("coord")
        cmap1 = cmap1.merge(dup_counts, on="coord", how="left")
        d = d.merge(cmap1, on="coord", how="left")
    else:
        d["patient_id_from_core"] = pd.NA
        d["sample_type_from_core"] = pd.NA
        d["n_patient_ids_for_coord"] = np.nan

    # Patient ID priority.
    pid_candidates = [
        "patient_id", "patient_id_coord_any", "Sample_ID_Adjusted", "Sample_ID",
        "patient_id_from_core",
    ]
    d["mif_patient_id"] = pd.NA
    for col in pid_candidates:
        if col in d.columns:
            val = d[col].astype(str).str.strip()
            val = val.mask(val.isin(["", "nan", "None", "<NA>"]))
            d["mif_patient_id"] = d["mif_patient_id"].combine_first(val)

    # Last-resort fallback to coord, to avoid crashing, but mark it.
    d["patient_id_source_is_fallback_coord"] = d["mif_patient_id"].isna()
    d["mif_patient_id"] = d["mif_patient_id"].fillna("coord:" + d["coord"].astype(str))

    if "sample_type_from_core" in d.columns:
        st_core = d["sample_type_from_core"].map(normalize_sample_type)
        d["sample_type"] = d["sample_type"].where(d["sample_type"].isin(SAMPLE_TYPES), st_core)
    d["sample_type"] = d["sample_type"].map(normalize_sample_type)

    for flag in ["has_parquet", "has_tissue", "has_review", "has_clinical", "has_all_primary_inputs"]:
        if flag not in d.columns:
            d[flag] = False
        d[flag] = d[flag].fillna(False).astype(bool)
    if "structural_acceptability" in d.columns:
        d["qc_status"] = d["structural_acceptability"].map(normalize_qc)
    else:
        d["qc_status"] = "Missing QC"

    is_tma = d["branch"].astype(str).eq("TMA")
    d["passes_structural_qc"] = (~is_tma) | d["qc_status"].isin(GOOD_QC)
    d["analysis_ready_unit"] = d["has_all_primary_inputs"] & d["passes_structural_qc"]

    # Cell counts.
    if "n_parquet_cells" in d.columns:
        d["n_parquet_cells"] = pd.to_numeric(d["n_parquet_cells"], errors="coerce").fillna(0)
    else:
        d["n_parquet_cells"] = 0

    # Panel-unit level distinctness.
    d["panel_unit_id"] = d["cohort"].astype(str) + "|" + d["sample_type"].astype(str) + "|" + d["panel"].astype(str) + "|" + d["coord"].astype(str)
    d["physical_unit_id"] = d["cohort"].astype(str) + "|" + d["sample_type"].astype(str) + "|" + d["coord"].astype(str)
    return d


def summarize_mif(mif: pd.DataFrame, cohorts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_rows = []
    sample_rows = []

    # Physical-unit table collapses AR/BT duplicate coord rows.
    phys = (
        mif.groupby(["cohort", "mif_patient_id", "sample_type", "coord", "physical_unit_id"], dropna=False)
        .agg(
            branch=("branch", "first"),
            submitted=("physical_unit_id", lambda x: True),
            analysis_ready=("analysis_ready_unit", "any"),
            any_panel_cells=("has_parquet", "any"),
            total_cells_any_panel=("n_parquet_cells", "sum"),
            fallback_patient_id=("patient_id_source_is_fallback_coord", "max"),
        )
        .reset_index()
    )

    for cohort in cohorts:
        sub = phys[phys["cohort"].eq(cohort)].copy()
        submitted_patients = sub.loc[sub["submitted"], "mif_patient_id"].nunique()
        ready_patients = sub.loc[sub["analysis_ready"], "mif_patient_id"].nunique()
        lost_patients = max(submitted_patients - ready_patients, 0)
        submitted_units = sub["physical_unit_id"].nunique()
        ready_units = sub.loc[sub["analysis_ready"], "physical_unit_id"].nunique()
        fallback_ids = sub.loc[sub["fallback_patient_id"].astype(bool), "mif_patient_id"].nunique()

        row = {
            "cohort": cohort,
            "specimen_format": "Whole section" if cohort == "BLASST" else "TMA",
            "mif_patients_submitted_n": submitted_patients,
            "mif_patients_analysis_ready_n": ready_patients,
            "mif_patients_lost_at_qc_n_pct": clean_percent(lost_patients, submitted_patients),
            "physical_cores_sections_submitted_n": submitted_units,
            "physical_cores_sections_analysis_ready_n": ready_units,
            "physical_cores_sections_analysis_ready_pct": clean_percent(ready_units, submitted_units),
            "patient_ids_requiring_fallback_n": fallback_ids,
        }

        # sample type patient and unit counts, submitted and after QC.
        for st in SAMPLE_TYPES:
            st_sub = sub[sub["sample_type"].eq(st)].copy()
            st_ready = st_sub[st_sub["analysis_ready"]].copy()
            row[f"{st}_patients_submitted_n"] = st_sub["mif_patient_id"].nunique()
            row[f"{st}_patients_analysis_ready_n"] = st_ready["mif_patient_id"].nunique()
            row[f"{st}_cores_sections_submitted_n"] = st_sub["physical_unit_id"].nunique()
            row[f"{st}_cores_sections_analysis_ready_n"] = st_ready["physical_unit_id"].nunique()

            per_patient_sub = st_sub.groupby("mif_patient_id")["physical_unit_id"].nunique()
            per_patient_ready = st_ready.groupby("mif_patient_id")["physical_unit_id"].nunique()
            row[f"{st}_cores_sections_per_patient_submitted_mean"] = per_patient_sub.mean() if len(per_patient_sub) else np.nan
            row[f"{st}_cores_sections_per_patient_analysis_ready_mean"] = per_patient_ready.mean() if len(per_patient_ready) else np.nan
            row[f"{st}_cores_sections_per_patient_submitted_median_iqr"] = fmt_median_iqr(per_patient_sub, 1)
            row[f"{st}_cores_sections_per_patient_analysis_ready_median_iqr"] = fmt_median_iqr(per_patient_ready, 1)

            sample_rows.append({
                "cohort": cohort,
                "sample_type": st,
                "patients_submitted_n": row[f"{st}_patients_submitted_n"],
                "patients_analysis_ready_n": row[f"{st}_patients_analysis_ready_n"],
                "cores_sections_submitted_n": row[f"{st}_cores_sections_submitted_n"],
                "cores_sections_analysis_ready_n": row[f"{st}_cores_sections_analysis_ready_n"],
                "cores_sections_per_patient_submitted_mean": row[f"{st}_cores_sections_per_patient_submitted_mean"],
                "cores_sections_per_patient_analysis_ready_mean": row[f"{st}_cores_sections_per_patient_analysis_ready_mean"],
                "cores_sections_per_patient_submitted_median_iqr": row[f"{st}_cores_sections_per_patient_submitted_median_iqr"],
                "cores_sections_per_patient_analysis_ready_median_iqr": row[f"{st}_cores_sections_per_patient_analysis_ready_median_iqr"],
            })

        # matched TURBT-RC patients.
        pat_st_sub = sub.groupby("mif_patient_id")["sample_type"].apply(lambda x: set(x.dropna())).reset_index()
        matched_sub = pat_st_sub[pat_st_sub["sample_type"].map(lambda z: {"TURBT", "RC"}.issubset(z))]
        sub_ready = sub[sub["analysis_ready"]].copy()
        pat_st_ready = sub_ready.groupby("mif_patient_id")["sample_type"].apply(lambda x: set(x.dropna())).reset_index()
        matched_ready = pat_st_ready[pat_st_ready["sample_type"].map(lambda z: {"TURBT", "RC"}.issubset(z))]
        row["matched_TURBT_RC_patients_submitted_n"] = matched_sub["mif_patient_id"].nunique()
        row["matched_TURBT_RC_patients_analysis_ready_n"] = matched_ready["mif_patient_id"].nunique()

        # panel-specific counts.
        panel_sub = mif[mif["cohort"].eq(cohort)].copy()
        for panel in REPORT_PANELS:
            psub = panel_sub[panel_sub["panel"].eq(panel)]
            row[f"{panel}_panel_units_submitted_n"] = psub["panel_unit_id"].nunique()
            row[f"{panel}_panel_units_analysis_ready_n"] = psub.loc[psub["analysis_ready_unit"], "panel_unit_id"].nunique()
            row[f"{panel}_segmented_cells_n"] = int(psub["n_parquet_cells"].sum())

        row["total_AR_BT_segmented_cells_n"] = int(panel_sub["n_parquet_cells"].sum())
        cohort_rows.append(row)

    cohort_summary = pd.DataFrame(cohort_rows)
    sample_summary = pd.DataFrame(sample_rows)

    panel_summary = (
        mif.groupby(["cohort", "sample_type", "panel"], dropna=False)
        .agg(
            panel_units_submitted_n=("panel_unit_id", "nunique"),
            panel_units_analysis_ready_n=("analysis_ready_unit", lambda s: int(mif.loc[s.index, "panel_unit_id"][s].nunique())),
            patients_submitted_n=("mif_patient_id", "nunique"),
            segmented_cells_n=("n_parquet_cells", "sum"),
            n_acceptable=("qc_status", lambda s: int((s == "Acceptable").sum())),
            n_borderline=("qc_status", lambda s: int((s == "Borderline").sum())),
            n_unusable=("qc_status", lambda s: int((s == "Unusable").sum())),
            n_missing_qc=("qc_status", lambda s: int((s == "Missing QC").sum())),
        )
        .reset_index()
    )
    panel_summary["panel_units_analysis_ready_pct"] = panel_summary.apply(
        lambda r: clean_percent(r["panel_units_analysis_ready_n"], r["panel_units_submitted_n"]), axis=1
    )
    return cohort_summary, sample_summary, panel_summary


def make_compact_table(clin: pd.DataFrame, mif: pd.DataFrame, cohorts: list[str]) -> pd.DataFrame:
    merged = clin.merge(mif, on="cohort", how="outer")
    # Derived display columns.
    for st in SAMPLE_TYPES:
        for stage in ["submitted", "analysis_ready"]:
            mean_col = f"{st}_cores_sections_per_patient_{stage}_mean"
            display_col = f"{st}_cores_sections_per_patient_{stage}_mean_display"
            merged[display_col] = merged[mean_col].map(lambda x: fmt_float(x, 1)) if mean_col in merged.columns else "NA"

    # Keep useful order.
    keep = [
        "cohort", "treatment_context", "specimen_format", "clinical_patients_n",
        "mif_patients_submitted_n", "mif_patients_analysis_ready_n", "mif_patients_lost_at_qc_n_pct",
        "physical_cores_sections_submitted_n", "physical_cores_sections_analysis_ready_n", "physical_cores_sections_analysis_ready_pct",
        "TURBT_patients_submitted_n", "TURBT_patients_analysis_ready_n", "TURBT_cores_sections_submitted_n", "TURBT_cores_sections_analysis_ready_n",
        "TURBT_cores_sections_per_patient_submitted_mean_display", "TURBT_cores_sections_per_patient_analysis_ready_mean_display",
        "RC_patients_submitted_n", "RC_patients_analysis_ready_n", "RC_cores_sections_submitted_n", "RC_cores_sections_analysis_ready_n",
        "RC_cores_sections_per_patient_submitted_mean_display", "RC_cores_sections_per_patient_analysis_ready_mean_display",
        "matched_TURBT_RC_patients_submitted_n", "matched_TURBT_RC_patients_analysis_ready_n",
        "AR_panel_units_submitted_n", "AR_panel_units_analysis_ready_n", "BT_panel_units_submitted_n", "BT_panel_units_analysis_ready_n",
        "total_AR_BT_segmented_cells_n",
        "age_median_iqr", "male_n_pct", "variant_histology_n_pct", "clinical_T_ge3_n_pct", "pathologic_T0_n_pct", "pathologic_node_positive_n_pct",
        "any_response_evaluable_n", "any_response_n_pct", "complete_response_evaluable_n", "complete_response_n_pct",
        "OS_evaluable_n", "OS_events_n_pct", "OS_time_median_iqr_months", "KM_median_OS_months",
        "RFS_evaluable_n", "RFS_events_n_pct", "RFS_time_median_iqr_months", "KM_median_RFS_months",
        "adjuvant_chemo_n_pct", "adjuvant_IO_n_pct",
        "patient_ids_requiring_fallback_n",
    ]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].copy()
    out["cohort"] = pd.Categorical(out["cohort"], categories=cohorts, ordered=True)
    return out.sort_values("cohort").reset_index(drop=True)


def make_wide_table(compact: pd.DataFrame, cohorts: list[str]) -> pd.DataFrame:
    row_defs = [
        ("Treatment context", "treatment_context"),
        ("Specimen format", "specimen_format"),
        ("Clinical patients, n", "clinical_patients_n"),
        ("Patients with mIF cores/sections submitted, n", "mif_patients_submitted_n"),
        ("Patients analysis-ready after QC, n", "mif_patients_analysis_ready_n"),
        ("Patients lost at mIF QC/source reconciliation, n (%)", "mif_patients_lost_at_qc_n_pct"),
        ("Physical TMA cores/whole-section regions submitted, n", "physical_cores_sections_submitted_n"),
        ("Physical TMA cores/whole-section regions analysis-ready, n (%)", "physical_cores_sections_analysis_ready_pct"),
        ("TURBT patients submitted, n", "TURBT_patients_submitted_n"),
        ("TURBT patients analysis-ready, n", "TURBT_patients_analysis_ready_n"),
        ("TURBT cores/sections submitted, n", "TURBT_cores_sections_submitted_n"),
        ("TURBT cores/sections analysis-ready, n", "TURBT_cores_sections_analysis_ready_n"),
        ("TURBT cores/sections per patient submitted, mean", "TURBT_cores_sections_per_patient_submitted_mean_display"),
        ("TURBT cores/sections per patient analysis-ready, mean", "TURBT_cores_sections_per_patient_analysis_ready_mean_display"),
        ("RC patients submitted, n", "RC_patients_submitted_n"),
        ("RC patients analysis-ready, n", "RC_patients_analysis_ready_n"),
        ("RC cores/sections submitted, n", "RC_cores_sections_submitted_n"),
        ("RC cores/sections analysis-ready, n", "RC_cores_sections_analysis_ready_n"),
        ("RC cores/sections per patient submitted, mean", "RC_cores_sections_per_patient_submitted_mean_display"),
        ("RC cores/sections per patient analysis-ready, mean", "RC_cores_sections_per_patient_analysis_ready_mean_display"),
        ("Matched TURBT-RC patients submitted, n", "matched_TURBT_RC_patients_submitted_n"),
        ("Matched TURBT-RC patients analysis-ready, n", "matched_TURBT_RC_patients_analysis_ready_n"),
        ("AR panel-units submitted, n", "AR_panel_units_submitted_n"),
        ("AR panel-units analysis-ready, n", "AR_panel_units_analysis_ready_n"),
        ("BT panel-units submitted, n", "BT_panel_units_submitted_n"),
        ("BT panel-units analysis-ready, n", "BT_panel_units_analysis_ready_n"),
        ("Total segmented cells across AR/BT, n", "total_AR_BT_segmented_cells_n"),
        ("Age, median (IQR), years", "age_median_iqr"),
        ("Male sex, n (%)", "male_n_pct"),
        ("Variant histology recorded, n (%)", "variant_histology_n_pct"),
        ("Clinical T stage ≥3, n (%)", "clinical_T_ge3_n_pct"),
        ("Pathologic T0, n (%)", "pathologic_T0_n_pct"),
        ("Pathologic node-positive, n (%)", "pathologic_node_positive_n_pct"),
        ("Any response evaluable, n", "any_response_evaluable_n"),
        ("Any response, n (%)", "any_response_n_pct"),
        ("Complete response evaluable, n", "complete_response_evaluable_n"),
        ("Complete response, n (%)", "complete_response_n_pct"),
        ("OS evaluable, n", "OS_evaluable_n"),
        ("OS events, n (%)", "OS_events_n_pct"),
        ("OS time, median (IQR), months", "OS_time_median_iqr_months"),
        ("KM median OS, months", "KM_median_OS_months"),
        ("RFS evaluable, n", "RFS_evaluable_n"),
        ("RFS events, n (%)", "RFS_events_n_pct"),
        ("RFS time, median (IQR), months", "RFS_time_median_iqr_months"),
        ("KM median RFS, months", "KM_median_RFS_months"),
        ("Adjuvant chemotherapy, n (%)", "adjuvant_chemo_n_pct"),
        ("Adjuvant IO, n (%)", "adjuvant_IO_n_pct"),
        ("Patient IDs inferred/fallback from coord, n", "patient_ids_requiring_fallback_n"),
    ]
    rows = []
    idx = compact.set_index("cohort")
    for label, col in row_defs:
        row = {"Characteristic": label}
        for cohort in cohorts:
            if cohort in idx.index and col in compact.columns:
                val = idx.loc[cohort, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.isna(val):
                    row[cohort] = "NA"
                elif isinstance(val, (int, np.integer)):
                    row[cohort] = f"{int(val):,}"
                elif isinstance(val, (float, np.floating)) and float(val).is_integer():
                    row[cohort] = f"{int(val):,}"
                elif isinstance(val, (float, np.floating)):
                    row[cohort] = f"{float(val):.1f}"
                else:
                    row[cohort] = str(val)
            else:
                row[cohort] = "NA"
        rows.append(row)
    return pd.DataFrame(rows)


def write_dictionary(out_dir: Path) -> pd.DataFrame:
    rows = [
        ("mif_patients_submitted_n", "Unique patients with ≥1 report-eligible AR/BT physical mIF core/whole-section region submitted."),
        ("mif_patients_analysis_ready_n", "Unique patients with ≥1 AR/BT physical mIF core/whole-section region that is source-complete and passes QC."),
        ("mif_patients_lost_at_qc_n_pct", "Submitted mIF patients with no remaining analysis-ready AR/BT physical core/whole-section region; percent denominator is submitted mIF patients."),
        ("physical_cores_sections_submitted_n", "Unique physical TMA cores or BLASST whole-section regions submitted, deduplicated across AR/BT panels."),
        ("physical_cores_sections_analysis_ready_n", "Unique physical TMA cores or BLASST whole-section regions with ≥1 analysis-ready AR/BT panel unit."),
        ("panel_units_submitted_n", "Panel-specific units, i.e., coordinate × panel."),
        ("analysis_ready_unit", "has_all_primary_inputs and passes structural QC for TMA; for BLASST, manual TMA structural QC is not required."),
        ("KM_median_OS_months", "Kaplan-Meier median OS; NR indicates median was not reached."),
        ("KM_median_RFS_months", "Kaplan-Meier median RFS; NR indicates median was not reached."),
        ("OS_time_median_iqr_months", "Observed OS time/follow-up distribution among patients with OS time and event indicator."),
        ("RFS_time_median_iqr_months", "Observed RFS time/follow-up distribution among patients with RFS time and event indicator."),
        ("patient_ids_requiring_fallback_n", "Patients whose mIF patient ID could not be read from core-level metadata and was therefore inferred from coordinate. Ideally this should be zero or reviewed."),
    ]
    df = pd.DataFrame(rows, columns=["field", "definition"])
    df.to_csv(out_dir / "table1_expanded_data_dictionary.csv", index=False)
    return df


def build_tables(
    clinical_path: Path = DEFAULT_CLINICAL,
    qc_dir: Path = DEFAULT_QC_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    cohorts: list[str] = REPORT_COHORTS,
    panels: list[str] = REPORT_PANELS,
    tma_clinical: Optional[Path] = DEFAULT_TMA_CLINICAL,
    blasst_clinical: Optional[Path] = DEFAULT_BLASST_CLINICAL,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical(clinical_path, cohorts)
    clinical_summary = summarize_clinical(clinical, cohorts)

    mif_units = build_mif_unit_table(
        qc_dir=qc_dir,
        cohorts=cohorts,
        panels=panels,
        tma_clinical=tma_clinical if tma_clinical and Path(tma_clinical).exists() else None,
        blasst_clinical=blasst_clinical if blasst_clinical and Path(blasst_clinical).exists() else None,
    )
    mif_cohort_summary, mif_sample_summary, mif_panel_summary = summarize_mif(mif_units, cohorts)

    compact = make_compact_table(clinical_summary, mif_cohort_summary, cohorts)
    wide = make_wide_table(compact, cohorts)
    dictionary = write_dictionary(out_dir)

    # Save.
    wide.to_csv(out_dir / "table1_report_by_cohort_expanded_wide.csv", index=False)
    compact.to_csv(out_dir / "table1_report_by_cohort_expanded_compact.csv", index=False)
    mif_sample_summary.to_csv(out_dir / "supp_table_mif_patient_core_counts_by_cohort_sample_type.csv", index=False)
    mif_panel_summary.to_csv(out_dir / "supp_table_mif_panel_unit_qc_by_cohort_sample_panel.csv", index=False)
    clinical_summary.to_csv(out_dir / "source_data_table1_clinical_expanded.csv", index=False)
    mif_units.to_csv(out_dir / "source_data_table1_mif_patient_unit_level.csv", index=False)

    # Optional warning table for ID fallback.
    warnings = []
    if "patient_ids_requiring_fallback_n" in compact.columns:
        for _, r in compact.iterrows():
            n = r.get("patient_ids_requiring_fallback_n", 0)
            if pd.notna(n) and int(n) > 0:
                warnings.append({
                    "cohort": r["cohort"],
                    "warning": f"{int(n)} mIF patient IDs required fallback inference. Review source_data_table1_mif_patient_unit_level.csv before final reporting.",
                })
    warn_df = pd.DataFrame(warnings)
    warn_df.to_csv(out_dir / "table1_patient_id_mapping_warnings.csv", index=False)

    return {
        "wide": wide,
        "compact": compact,
        "clinical_summary": clinical_summary,
        "mif_cohort_summary": mif_cohort_summary,
        "mif_sample_summary": mif_sample_summary,
        "mif_panel_summary": mif_panel_summary,
        "mif_units": mif_units,
        "dictionary": dictionary,
        "warnings": warn_df,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", default=str(DEFAULT_CLINICAL))
    ap.add_argument("--qc-dir", default=str(DEFAULT_QC_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tma-clinical", default=str(DEFAULT_TMA_CLINICAL))
    ap.add_argument("--blasst-clinical", default=str(DEFAULT_BLASST_CLINICAL))
    ap.add_argument("--cohorts", nargs="+", default=REPORT_COHORTS)
    ap.add_argument("--panels", nargs="+", default=REPORT_PANELS)
    args = ap.parse_args()

    outputs = build_tables(
        clinical_path=Path(args.clinical),
        qc_dir=Path(args.qc_dir),
        out_dir=Path(args.out_dir),
        cohorts=[normalize_cohort(x) for x in args.cohorts],
        panels=[normalize_panel(x) for x in args.panels],
        tma_clinical=Path(args.tma_clinical) if args.tma_clinical else None,
        blasst_clinical=Path(args.blasst_clinical) if args.blasst_clinical else None,
    )
    print("\nWrote expanded Table 1 outputs to:", Path(args.out_dir))
    print("\nPreview:")
    print(outputs["wide"].to_string(index=False))
    if not outputs["warnings"].empty:
        print("\nWARNINGS:")
        print(outputs["warnings"].to_string(index=False))


if __name__ == "__main__":
    main()
