#!/usr/bin/env python3
"""
build_dod_report_table1.py

Build report-restricted Table 1 summaries for the DOD/annual report.

Inputs
------
1) Harmonized clinical dataframe:
   /projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv

2) QC rebuild directory from qc_check.py:
   /projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild

Outputs
-------
- table1_report_by_cohort_wide.csv
- table1_report_by_cohort_compact.csv
- supp_table_mif_core_qc_by_cohort_sample_panel.csv
- source_data_table1_clinical_summary.csv
- source_data_table1_mif_summary.csv
- table1_report_data_dictionary.csv

Notes
-----
The mIF unit is a core for TMA cohorts and a whole-section region for BLASST.
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
COHORT_ORDER = {c: i for i, c in enumerate(REPORT_COHORTS)}
PANEL_ORDER = {p: i for i, p in enumerate(REPORT_PANELS)}

DEFAULT_CLINICAL = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv")
DEFAULT_QC_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild")
DEFAULT_OUT_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/table_outputs")


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def clean_cohort(x) -> str:
    s = norm_str(x)
    mapping = {
        "BCA2020": "NAC2020",
        "BCA 2020": "NAC2020",
        "BCA_2020": "NAC2020",
        "NO-NAC": "No-NAC",
        "NONAC": "No-NAC",
        "NO NAT": "No-NAC",
        "PURE01": "PURE01",
        "BLASST": "BLASST",
    }
    return mapping.get(s.upper(), s)


def clean_sample_type(x) -> str:
    s = norm_str(x).upper().replace(" ", "")
    if s in {"TURBT", "TUR", "TURB", "PRE", "PRETREATMENT"}:
        return "TURBT"
    if s in {"RC", "CYSTECTOMY", "RADICALCYSTECTOMY", "POST", "POSTTREATMENT"}:
        return "RC"
    if not s or s in {"NAN", "NA", "NONE", "UNKNOWN"}:
        return "Unknown"
    return norm_str(x)


def clean_panel(x) -> str:
    s = norm_str(x).upper()
    if s in {"AR", "ARP"} or "ARP" in s:
        return "AR"
    if s in {"BT", "B&T", "B+T", "B_T"} or "B&T" in s or "B+T" in s:
        return "BT"
    if s in {"MY", "M", "MYELOID"} or "MYELOID" in s:
        return "MY"
    if not s:
        return "Unknown"
    return norm_str(x)


def n_pct(n: int | float, denom: int | float) -> str:
    if denom is None or pd.isna(denom) or denom == 0:
        return "0/0"
    n = int(n) if not pd.isna(n) else 0
    denom = int(denom)
    return f"{n}/{denom} ({100*n/denom:.0f}%)"


def median_iqr(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "NA"
    q1, med, q3 = x.quantile([0.25, 0.50, 0.75])
    return f"{med:.0f} ({q1:.0f}-{q3:.0f})"


def count_nonmissing(s: pd.Series) -> int:
    return int(s.notna().sum())


def is_positive_binary(x) -> bool:
    if pd.isna(x):
        return False
    s = norm_str(x).lower()
    if s in {"1", "1.0", "yes", "y", "true", "positive", "pos", "pcr", "cr", "response", "responder"}:
        return True
    try:
        return float(s) == 1.0
    except Exception:
        return False


def is_variant_positive(x) -> bool:
    if pd.isna(x):
        return False
    s = norm_str(x).lower()
    if s in {"", "0", "0.0", "no", "none", "nan", "na", "unknown", "uc", "urothelial", "pure uc", "pure urothelial carcinoma"}:
        return False
    return True


def stage_number(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = norm_str(x).upper()
    # Exclude things like Tis/Ta from numeric parsing.
    if "TIS" in s or re.search(r"\bIS\b", s):
        return 0
    if re.search(r"\bTA\b", s):
        return 0
    m = re.search(r"[CTPN]*\s*([0-4])", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_t_ge3(x) -> bool:
    n = stage_number(x)
    return bool(n is not None and n >= 3)


def is_pt0(x) -> bool:
    if pd.isna(x):
        return False
    s = norm_str(x).upper()
    return bool(re.search(r"\bP?T?0\b", s) or s == "0" or s == "0.0" or s.startswith("PT0"))


def is_node_positive(x) -> bool:
    if pd.isna(x):
        return False
    s = norm_str(x).upper()
    if "N+" in s:
        return True
    if re.search(r"N[1-3]", s):
        return True
    try:
        return float(s) > 0
    except Exception:
        return False


def endpoint_count(df: pd.DataFrame, col: str) -> tuple[int, int, str]:
    if col not in df.columns:
        return 0, 0, "NA"
    sub = df[df[col].notna()]
    n_eval = int(len(sub))
    n_pos = int(sub[col].map(is_positive_binary).sum())
    return n_eval, n_pos, n_pct(n_pos, n_eval)


def survival_count(df: pd.DataFrame, event_col: str, time_cols: Iterable[str]) -> tuple[int, int, str]:
    if event_col not in df.columns:
        return 0, 0, "NA"
    time_available = pd.Series(False, index=df.index)
    for c in time_cols:
        if c in df.columns:
            time_available = time_available | pd.to_numeric(df[c], errors="coerce").notna()
    sub = df[df[event_col].notna() & time_available]
    n_eval = int(len(sub))
    n_event = int(sub[event_col].map(is_positive_binary).sum())
    return n_eval, n_event, n_pct(n_event, n_eval)


def load_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "cohort" not in df.columns:
        raise ValueError(f"Clinical file must have a 'cohort' column: {path}")
    df = df.copy()
    df["cohort"] = df["cohort"].map(clean_cohort)
    return df


def clinical_summary_by_cohort(clin: pd.DataFrame, cohorts: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        d = clin[clin["cohort"].eq(cohort)].copy()
        if d.empty:
            rows.append({"cohort": cohort})
            continue
        patient_col = "patient_id" if "patient_id" in d.columns else None
        n_patients = int(d[patient_col].nunique()) if patient_col else int(len(d))
        nat_vals = sorted([norm_str(v) for v in d.get("NAT", pd.Series(dtype=object)).dropna().unique() if norm_str(v)])
        if cohort == "No-NAC" and not nat_vals:
            treatment = "No NAT / RC-only"
        elif nat_vals:
            treatment = "; ".join(nat_vals).replace("NAC\\IO", "NAC/IO")
        else:
            treatment = "Not recorded"
        sex = d.get("Sex", pd.Series(index=d.index, dtype=object)).astype(str).str.lower()
        n_male = int(sex.str.startswith("male").sum())
        age = median_iqr(d["Age"]) if "Age" in d.columns else "NA"
        variant_n = int(d.get("variant", pd.Series(index=d.index, dtype=object)).map(is_variant_positive).sum()) if "variant" in d.columns else 0
        any_eval, any_resp, any_str = endpoint_count(d, "any_response")
        cr_eval, cr_resp, cr_str = endpoint_count(d, "complete_response")
        os_eval, os_events, os_str = survival_count(d, "OS_event", ["OS_months_RC", "OS_months_TUR", "OS_months_TURBT"])
        rfs_eval, rfs_events, rfs_str = survival_count(d, "REC", ["REC_months_RC", "REC_months_TURBT", "RFS_months_RC", "RFS_months_TURBT"])
        ct_eval = int(d["cT"].notna().sum()) if "cT" in d.columns else 0
        ct_ge3 = int(d["cT"].map(is_t_ge3).sum()) if "cT" in d.columns else 0
        pt_eval = int(d["pT"].notna().sum()) if "pT" in d.columns else 0
        pt0 = int(d["pT"].map(is_pt0).sum()) if "pT" in d.columns else 0
        pn_eval = int(d["pN"].notna().sum()) if "pN" in d.columns else 0
        pn_pos = int(d["pN"].map(is_node_positive).sum()) if "pN" in d.columns else 0
        adj_chemo_eval = int(d["adjuvant_chemo"].notna().sum()) if "adjuvant_chemo" in d.columns else 0
        adj_chemo_pos = int(d["adjuvant_chemo"].map(is_positive_binary).sum()) if "adjuvant_chemo" in d.columns else 0
        adj_io_eval = int(d["adjuvant_IO"].notna().sum()) if "adjuvant_IO" in d.columns else 0
        adj_io_pos = int(d["adjuvant_IO"].map(is_positive_binary).sum()) if "adjuvant_IO" in d.columns else 0
        rows.append({
            "cohort": cohort,
            "treatment_context": treatment,
            "n_clinical_patients": n_patients,
            "age_median_iqr": age,
            "male_n_pct": n_pct(n_male, n_patients),
            "variant_histology_n_pct": n_pct(variant_n, n_patients),
            "cT_ge3_n_pct": n_pct(ct_ge3, ct_eval),
            "pT0_n_pct": n_pct(pt0, pt_eval),
            "pN_positive_n_pct": n_pct(pn_pos, pn_eval),
            "any_response_n_pct": any_str,
            "complete_response_n_pct": cr_str,
            "os_events_n_pct": os_str,
            "rfs_events_n_pct": rfs_str,
            "adjuvant_chemo_n_pct": n_pct(adj_chemo_pos, adj_chemo_eval),
            "adjuvant_io_n_pct": n_pct(adj_io_pos, adj_io_eval),
            "any_response_evaluable_n": any_eval,
            "complete_response_evaluable_n": cr_eval,
            "os_evaluable_n": os_eval,
            "rfs_evaluable_n": rfs_eval,
        })
    out = pd.DataFrame(rows)
    out["cohort_order"] = out["cohort"].map(COHORT_ORDER).fillna(999).astype(int)
    return out.sort_values("cohort_order").drop(columns="cohort_order")


def load_qc_presence(qc_dir: Path) -> pd.DataFrame:
    fp = qc_dir / "source_presence_all.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing expected QC file: {fp}")
    df = pd.read_csv(fp, low_memory=False)
    if "cohort_label" not in df.columns:
        raise ValueError(f"source_presence_all.csv must contain cohort_label: {fp}")
    df = df.copy()
    df["cohort_label"] = df["cohort_label"].map(clean_cohort)
    df["panel"] = df.get("panel", pd.Series(index=df.index, dtype=object)).map(clean_panel)
    df["sample_type"] = df.get("sample_type", pd.Series(index=df.index, dtype=object)).map(clean_sample_type)
    for c in ["has_parquet", "has_tissue", "has_review", "has_clinical", "has_all_primary_inputs"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
        else:
            df[c] = False
    for c in ["n_parquet_cells", "n_parquet_epi_cells", "n_parquet_str_cells", "n_parquet_other_cells"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    return df


def mif_summary_by_cohort(presence: pd.DataFrame, cohorts: list[str], panels: list[str]) -> pd.DataFrame:
    d = presence[presence["cohort_label"].isin(cohorts) & presence["panel"].isin(panels)].copy()
    rows = []
    for cohort in cohorts:
        sub = d[d["cohort_label"].eq(cohort)].copy()
        if sub.empty:
            rows.append({"cohort": cohort})
            continue
        specimen_format = "Whole-section regions" if cohort == "BLASST" or sub.get("branch", pd.Series(dtype=object)).astype(str).str.contains("BLASST", case=False, na=False).any() else "TMA cores"
        sample_types = "; ".join([x for x in ["TURBT", "RC"] if x in set(sub["sample_type"])])
        if not sample_types:
            sample_types = "; ".join(sorted(sub["sample_type"].dropna().unique())) or "Unknown"
        # unique physical units across panels
        submitted_any = int(sub["coord"].nunique()) if "coord" in sub.columns else int(sub["entity_id"].nunique())
        ready_units = sub[sub["has_all_primary_inputs"]]
        ready_any = int(ready_units["coord"].nunique()) if "coord" in ready_units.columns else int(ready_units["entity_id"].nunique())
        row = {
            "cohort": cohort,
            "specimen_format": specimen_format,
            "mif_sample_types": sample_types,
            "mif_units_submitted_any_panel": submitted_any,
            "mif_units_ready_any_panel": ready_any,
            "mif_units_ready_any_panel_n_pct": n_pct(ready_any, submitted_any),
            "total_ar_bt_cells": int(sub["n_parquet_cells"].sum()),
        }
        for panel in panels:
            p = sub[sub["panel"].eq(panel)].copy()
            row[f"{panel}_units_submitted"] = int(p["coord"].nunique()) if "coord" in p.columns else int(p["entity_id"].nunique())
            pr = p[p["has_all_primary_inputs"]]
            row[f"{panel}_units_ready"] = int(pr["coord"].nunique()) if "coord" in pr.columns else int(pr["entity_id"].nunique())
            row[f"{panel}_units_ready_n_pct"] = n_pct(row[f"{panel}_units_ready"], row[f"{panel}_units_submitted"])
            row[f"{panel}_cells_total"] = int(p["n_parquet_cells"].sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["cohort_order"] = out["cohort"].map(COHORT_ORDER).fillna(999).astype(int)
    return out.sort_values("cohort_order").drop(columns="cohort_order")


def mif_summary_by_cohort_sample_panel(presence: pd.DataFrame, cohorts: list[str], panels: list[str]) -> pd.DataFrame:
    d = presence[presence["cohort_label"].isin(cohorts) & presence["panel"].isin(panels)].copy()
    if d.empty:
        return pd.DataFrame()
    group_cols = ["cohort_label", "branch", "sample_type", "panel"]
    for c in group_cols:
        if c not in d.columns:
            d[c] = "Unknown"
    rows = []
    for keys, sub in d.groupby(group_cols, dropna=False):
        cohort, branch, sample_type, panel = keys
        submitted = int(sub["coord"].nunique()) if "coord" in sub.columns else int(sub["entity_id"].nunique())
        ready_sub = sub[sub["has_all_primary_inputs"]]
        ready = int(ready_sub["coord"].nunique()) if "coord" in ready_sub.columns else int(ready_sub["entity_id"].nunique())
        rows.append({
            "cohort": cohort,
            "branch": branch,
            "specimen_format": "Whole-section regions" if cohort == "BLASST" or str(branch).upper() == "BLASST" else "TMA cores",
            "sample_type": sample_type,
            "panel": panel,
            "units_submitted": submitted,
            "units_with_cells": int(sub[sub["has_parquet"]]["coord"].nunique()) if "coord" in sub.columns else int(sub[sub["has_parquet"]]["entity_id"].nunique()),
            "units_with_tissue": int(sub[sub["has_tissue"]]["coord"].nunique()) if "coord" in sub.columns else int(sub[sub["has_tissue"]]["entity_id"].nunique()),
            "units_with_review": int(sub[sub["has_review"]]["coord"].nunique()) if "coord" in sub.columns else int(sub[sub["has_review"]]["entity_id"].nunique()),
            "units_with_clinical": int(sub[sub["has_clinical"]]["coord"].nunique()) if "coord" in sub.columns else int(sub[sub["has_clinical"]]["entity_id"].nunique()),
            "units_analysis_ready": ready,
            "units_analysis_ready_n_pct": n_pct(ready, submitted),
            "n_cells_total": int(sub["n_parquet_cells"].sum()),
            "n_epi_cells_total": int(sub["n_parquet_epi_cells"].sum()),
            "n_str_cells_total": int(sub["n_parquet_str_cells"].sum()),
        })
    out = pd.DataFrame(rows)
    out["cohort_order"] = out["cohort"].map(COHORT_ORDER).fillna(999).astype(int)
    out["panel_order"] = out["panel"].map(PANEL_ORDER).fillna(999).astype(int)
    sample_order = {"TURBT": 0, "RC": 1, "Unknown": 2}
    out["sample_order"] = out["sample_type"].map(sample_order).fillna(99).astype(int)
    return out.sort_values(["cohort_order", "sample_order", "panel_order"]).drop(columns=["cohort_order", "sample_order", "panel_order"])


def combine_table1(clin_sum: pd.DataFrame, mif_sum: pd.DataFrame) -> pd.DataFrame:
    out = clin_sum.merge(mif_sum, on="cohort", how="outer")
    out["cohort_order"] = out["cohort"].map(COHORT_ORDER).fillna(999).astype(int)
    return out.sort_values("cohort_order").drop(columns="cohort_order")


def make_wide_table(compact: pd.DataFrame) -> pd.DataFrame:
    row_defs = [
        ("Treatment context", "treatment_context"),
        ("Specimen format", "specimen_format"),
        ("mIF sample types", "mif_sample_types"),
        ("Clinical patients, n", "n_clinical_patients"),
        ("Age, median (IQR)", "age_median_iqr"),
        ("Male sex, n/N (%)", "male_n_pct"),
        ("Variant histology recorded, n/N (%)", "variant_histology_n_pct"),
        ("Clinical T stage ≥3, n/N (%)", "cT_ge3_n_pct"),
        ("Pathologic T0, n/N (%)", "pT0_n_pct"),
        ("Pathologic node-positive, n/N (%)", "pN_positive_n_pct"),
        ("Any pathologic response, n/N (%)", "any_response_n_pct"),
        ("Complete pathologic response, n/N (%)", "complete_response_n_pct"),
        ("OS deaths, n/N (%)", "os_events_n_pct"),
        ("RFS recurrences, n/N (%)", "rfs_events_n_pct"),
        ("Adjuvant chemotherapy, n/N (%)", "adjuvant_chemo_n_pct"),
        ("Adjuvant IO, n/N (%)", "adjuvant_io_n_pct"),
        ("mIF units submitted, n", "mif_units_submitted_any_panel"),
        ("mIF units QC/source-complete, n/N (%)", "mif_units_ready_any_panel_n_pct"),
        ("AR units submitted, n", "AR_units_submitted"),
        ("AR units QC/source-complete, n/N (%)", "AR_units_ready_n_pct"),
        ("BT units submitted, n", "BT_units_submitted"),
        ("BT units QC/source-complete, n/N (%)", "BT_units_ready_n_pct"),
        ("Total AR+BT segmented cells, n", "total_ar_bt_cells"),
    ]
    rows = []
    for label, col in row_defs:
        row = {"Characteristic": label}
        for cohort in REPORT_COHORTS:
            sub = compact[compact["cohort"].eq(cohort)]
            if sub.empty or col not in sub.columns:
                val = "NA"
            else:
                val = sub.iloc[0][col]
                if isinstance(val, (int, np.integer)):
                    val = f"{val:,}"
                elif isinstance(val, (float, np.floating)) and not pd.isna(val):
                    if float(val).is_integer():
                        val = f"{int(val):,}"
                elif pd.isna(val):
                    val = "NA"
            row[cohort] = val
        rows.append(row)
    return pd.DataFrame(rows)


def make_dictionary() -> pd.DataFrame:
    rows = [
        ("Clinical patients", "Unique patient_id rows in harmonized_modeling_dataframe.csv after report cohort filtering."),
        ("mIF units", "Unique coord-level units in source_presence_all.csv. These are TMA cores for NAC2020/PURE01/No-NAC and whole-section regions for BLASST."),
        ("Submitted", "Unit appears in the QC source-presence universe for AR/BT panels."),
        ("QC/source-complete", "Unit has all primary inputs according to qc_check.py: cells/parquet, tissue, clinical, and for TMA cohorts manual review; BLASST does not require review."),
        ("Any pathologic response", "any_response == 1 among non-missing any_response values."),
        ("Complete pathologic response", "complete_response == 1 among non-missing complete_response values."),
        ("OS deaths", "OS_event == 1 among patients with OS_event and at least one OS time column."),
        ("RFS recurrences", "REC == 1 among patients with REC and at least one recurrence time column."),
        ("Clinical T stage ≥3", "cT parsed as T3/T4 or numeric 3/4 among evaluable cT values."),
        ("Pathologic T0", "pT parsed as pT0/0 among evaluable pT values."),
        ("Pathologic node-positive", "pN parsed as N+, N1-N3, or numeric >0 among evaluable pN values."),
    ]
    return pd.DataFrame(rows, columns=["term", "definition"])


def build_tables(clinical_path: Path, qc_dir: Path, out_dir: Path, cohorts: list[str] = REPORT_COHORTS) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clin = load_clinical(clinical_path)
    clin = clin[clin["cohort"].isin(cohorts)].copy()
    presence = load_qc_presence(qc_dir)
    presence = presence[presence["cohort_label"].isin(cohorts) & presence["panel"].isin(REPORT_PANELS)].copy()
    clin_sum = clinical_summary_by_cohort(clin, cohorts)
    mif_sum = mif_summary_by_cohort(presence, cohorts, REPORT_PANELS)
    compact = combine_table1(clin_sum, mif_sum)
    wide = make_wide_table(compact)
    supp = mif_summary_by_cohort_sample_panel(presence, cohorts, REPORT_PANELS)
    dictionary = make_dictionary()
    outputs = {
        "table1_report_by_cohort_compact": compact,
        "table1_report_by_cohort_wide": wide,
        "supp_table_mif_core_qc_by_cohort_sample_panel": supp,
        "source_data_table1_clinical_summary": clin_sum,
        "source_data_table1_mif_summary": mif_sum,
        "table1_report_data_dictionary": dictionary,
    }
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build DOD report Table 1 from harmonized clinical + QC source-presence files.")
    ap.add_argument("--clinical", default=str(DEFAULT_CLINICAL), help="Path to harmonized_modeling_dataframe.csv")
    ap.add_argument("--qc-dir", default=str(DEFAULT_QC_DIR), help="Path to qc_check_rebuild directory")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    ap.add_argument("--cohorts", nargs="+", default=REPORT_COHORTS, help="Cohorts to include")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_tables(Path(args.clinical), Path(args.qc_dir), Path(args.out_dir), cohorts=args.cohorts)
    print("\nWrote DOD report Table 1 outputs:")
    for name, df in outputs.items():
        print(f"  {name}.csv: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"\nOutput directory: {args.out_dir}")


if __name__ == "__main__":
    main()
