#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_check.py

Compact, documented QC reconciliation script for the mIF preprocessing rebuild.

This script is intended to replace the scattered matching/QC portions of:
  - qc_checks2.ipynb
  - qc_checks3.ipynb
  - qc_checks_wholesections.ipynb

It does NOT regenerate feature tables. Instead, it reconciles the upstream
preprocessing products:

  1. Tissue/cell-summary rebuild outputs
     /projects/.../immuno/data/inform_summary_rebuild

  2. Combined per-cell parquet outputs
     /projects/.../immuno/data/raw_phenoptr/combined_cohorts
     /projects/.../immuno/data/raw_phenoptr/combined_wholesections

  3. Manual QC review files
     /projects/.../immuno/data/*review.csv

  4. Clinical metadata
     ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv
     ClinicalData_Core_BLASST.csv

  5. Optional KOLL/Florestan outputs
     /projects/.../immuno/data/KOLL_cohort

The output is a descriptive audit folder containing:
  - source inventories
  - merged coord/panel-level presence matrix
  - summaries by panel/cohort/TMA/sample type
  - mismatch files with identities
  - KOLL-specific sample-level QC
  - README and output manifest

The helper functions are written so they can also be imported into later
plotting notebooks or feature-generation scripts.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data")
DEFAULT_INFORM_SUMMARY_DIR = DEFAULT_DATA_DIR / "inform_summary_rebuild"
DEFAULT_TMA_PARQUET_DIR = DEFAULT_DATA_DIR / "raw_phenoptr" / "combined_cohorts"
DEFAULT_WHOLE_PARQUET_DIR = DEFAULT_DATA_DIR / "raw_phenoptr" / "combined_wholesections"
DEFAULT_TMA_CLINICAL = DEFAULT_DATA_DIR / "ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv"
DEFAULT_BLASST_CLINICAL = DEFAULT_DATA_DIR / "ClinicalData_Core_BLASST.csv"
DEFAULT_KOLL_DIR = DEFAULT_DATA_DIR / "KOLL_cohort"
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "qc_check_rebuild"

COORD_RE = re.compile(r"\[\s*(\d{3,})\s*,\s*(\d{3,})\s*\]")
COHORT_CODE_MAP = {
    0: "NAC2020",
    1: "No-NAC",
    2: "PURE01",
    4: "NAC2015",
}
COHORT_NAME_MAP = {
    "BCA2020": "NAC2020",
    "BCA_2020": "NAC2020",
    "BCA 2020": "NAC2020",
    "NO-NAC": "No-NAC",
    "NONAC": "No-NAC",
    "PURE01": "PURE01",
    "BLADDER": "NAC2015",
    "BLASST": "BLASST",
    "FLORESTAN": "KOLL",
    "KOLL": "KOLL",
}


# -----------------------------------------------------------------------------
# Generic cleaning / parsing helpers
# -----------------------------------------------------------------------------

def norm_ws(x: object) -> str:
    """Collapse repeated whitespace and strip."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def safe_token(x: object) -> str:
    """Filesystem/column-safe token."""
    x = norm_ws(x)
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "NA"


def normalize_panel(x: object) -> str:
    """Normalize panel labels used across notebooks and files."""
    s = norm_ws(x).upper()
    if s in {"", "NAN", "NONE", "UNKNOWN", "UNK", "<NA>"}:
        return "Unknown"
    if "B&T" in s or "B+T" in s or s in {"BT", "B_T", "B T"}:
        return "BT"
    if "MYELOID" in s or s in {"MY", "M"}:
        return "MY"
    if "ARP" in s or s == "AR" or re.search(r"(^|[^A-Z0-9])AR($|[^A-Z0-9])", s):
        return "AR"
    return norm_ws(x)


def infer_panel_from_path(path: Path) -> str:
    """
    Infer panel from file/path tokens.

    Important historical NAC2015 rule:
      Bladder_19_M.parquet / Bladder_26_M.parquet are the MY panel,
      even though the token is a single letter M rather than MY/Myeloid.
    """
    full = str(path).upper()
    name = path.name.upper()
    stem = path.stem.upper()
    joined = " ".join([p.upper() for p in path.parts])

    # BT first because B&T/B+T are distinctive.
    if "B&T" in joined or "B+T" in joined or "B_T" in joined:
        return "BT"
    if re.search(r"(^|[^A-Z0-9])BT($|[^A-Z0-9])", joined):
        return "BT"

    # Myeloid. Explicitly support historical NAC2015 Bladder_19_M / Bladder_26_M files.
    if "MYELOID" in joined:
        return "MY"
    if re.search(r"(^|[^A-Z0-9])MY($|[^A-Z0-9])", joined):
        return "MY"
    if re.search(r"(^|[_\-\s])M($|[_\-\s\.])", stem) or re.search(r"BLADDER[_\s]+(?:19|26)[_\s]+M($|[_\-\s\.])", name):
        return "MY"

    # AR / ARP.
    if "ARP" in joined:
        return "AR"
    if "AR INFORM" in joined or re.search(r"(^|[^A-Z0-9])AR($|[^A-Z0-9])", joined):
        return "AR"
    return "Unknown"


def infer_panel_from_sample_name(x: object) -> str:
    """
    Infer panel from a raw sample/core name.

    Handles examples like:
      Bladder 19_M_Core[1,9,B]_[33899,18633].im3  -> MY
      Bladder 19_AR_Core[...]                     -> AR
      Bladder 19_BT_Core[...]                     -> BT
    """
    s = norm_ws(x).upper()
    s2 = s.replace("+", "&")
    if "B&T" in s2 or re.search(r"(^|[^A-Z0-9])BT($|[^A-Z0-9])", s2):
        return "BT"
    if "MYELOID" in s2 or re.search(r"(^|[^A-Z0-9])MY($|[^A-Z0-9])", s2):
        return "MY"
    if re.search(r"BLADDER\s+(?:19|26)[_\s]+M[_\s]+CORE", s2) or re.search(r"(^|[_\-\s])M[_\s]+CORE", s2):
        return "MY"
    if "ARP" in s2 or re.search(r"(^|[^A-Z0-9])AR($|[^A-Z0-9])", s2):
        return "AR"
    return "Unknown"

def extract_coord_token_from_string(x: object) -> Optional[str]:
    """
    Extract bracket coordinate as '[12345,6789]'.
    Keeps the historical bracket format used by qc_checks2/qc_checks3.
    """
    if pd.isna(x):
        return None
    s = str(x)
    matches = COORD_RE.findall(s)
    if not matches:
        return None
    # Prefer the last bracket token because TMA sample names often contain Core[...] then coord[...].
    a, b = matches[-1]
    return f"[{a},{b}]"


def extract_coord_key_from_string(x: object) -> Optional[str]:
    """Extract coordinate as '12345_6789', used in the old whole-section notebook."""
    coord = extract_coord_token_from_string(x)
    if coord is None:
        return None
    m = COORD_RE.search(coord)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def extract_coord_token(series: pd.Series) -> pd.Series:
    return series.map(extract_coord_token_from_string).astype("object")


def extract_coord_key(series: pd.Series) -> pd.Series:
    return series.map(extract_coord_key_from_string).astype("object")


def parse_core_idx(core_idx_val: object) -> tuple[object, object]:
    """
    Parse historical core_idx strings like '(0, 1, ...)' into (cohort_code, tma).
    Adapted from qc_checks2/qc_checks3.
    """
    if pd.isna(core_idx_val):
        return (pd.NA, pd.NA)
    s = str(core_idx_val)
    m = re.search(r"\(\s*'?\s*([0-9]+)\s*'?\s*,\s*'?\s*([0-9]+)\s*'?\s*,", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    nums = re.findall(r"[0-9]+", s)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return (pd.NA, pd.NA)


def infer_cohort_from_sample_name(x: object) -> str:
    """
    Infer cohort label from sample_name. Adapted from qc_checks3:
      BCA2020 -> NAC2020
      No-NAC -> No-NAC
      PURE01 -> PURE01
      Bladder -> NAC2015
    """
    s = norm_ws(x).upper().replace("_", " ")
    raw = None
    if re.match(r"^BCA\s*2020\b", s):
        raw = "BCA2020"
    elif re.match(r"^NO-?NAC\b", s):
        raw = "NO-NAC"
    elif re.match(r"^PURE01\b", s):
        raw = "PURE01"
    elif re.match(r"^BLADDER\b", s):
        raw = "BLADDER"
    elif "BLASST" in s:
        raw = "BLASST"
    elif "FLORESTAN" in s or "KOLL" in s:
        raw = "KOLL"
    return COHORT_NAME_MAP.get(raw, raw or "Unknown")


def infer_tma_from_sample_name(x: object) -> object:
    """
    Infer TMA number from sample name.

    Historical NAC2015 rule:
      Bladder 19_* -> TMA 1
      Bladder 26_* -> TMA 2
    """
    s = norm_ws(x).upper()
    m = re.search(r"TMA\s*([0-9]+)", s)
    if m:
        return int(m.group(1))
    if re.search(r"^BLADDER\s*19\b|^BLADDER[_\s]+19[_\s]", s):
        return 1
    if re.search(r"^BLADDER\s*26\b|^BLADDER[_\s]+26[_\s]", s):
        return 2
    return pd.NA

def is_usable_qc_label(x: object) -> bool:
    return norm_ws(x).lower() not in {"unusable", "fail", "failed", "bad"}


def infer_review_panel_from_filename(filename: object) -> str:
    """Infer panel from manual review filename, e.g. NAC2015_AR_review.csv."""
    return infer_panel_from_path(Path(str(filename)))


def infer_review_cohort_from_filename(filename: object) -> str:
    """Infer cohort from manual review filename when possible."""
    s = str(filename).upper().replace("_", " ").replace("-", " ")
    if "NAC2015" in s:
        return "NAC2015"
    if "NAC2020" in s or "BCA2020" in s or "BCA 2020" in s:
        return "NAC2020"
    if "PURE01" in s:
        return "PURE01"
    if "NO NAC" in s or "NONAC" in s:
        return "No-NAC"
    return "Unknown"



def clean_pheno_combo(x: object) -> str:
    """
    Keep only positive marker calls from a semicolon-delimited combined phenotype.
    Historical rule from qc_checks3 and whole-section notebook:
      'CD3+;CD68+;PD1-' -> 'CD3+;CD68+'
      'PD1-;PDL1-'      -> 'ALL_NEG'
    """
    if pd.isna(x):
        return "ALL_NEG"
    s = norm_ws(x)
    if not s or s.lower() == "nan":
        return "ALL_NEG"
    parts = [p.strip() for p in s.split(";") if p.strip()]
    pos = sorted(set([p for p in parts if p.endswith("+")]))
    return ";".join(pos) if pos else "ALL_NEG"


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name}: rows={len(df):,} cols={df.shape[1]:,}")


# -----------------------------------------------------------------------------
# Load clinical and review metadata
# -----------------------------------------------------------------------------

def load_tma_clinical(path: Path) -> pd.DataFrame:
    """Load TMA clinical metadata and standardize coord/panel/cohort/tma/sample_type."""
    if not path.exists():
        print(f"[warn] TMA clinical file not found: {path}")
        return pd.DataFrame(columns=["branch", "coord", "coord_key", "panel", "cohort_label", "tma", "sample_type"])

    clin = pd.read_csv(path, low_memory=False).copy()
    if "Core" not in clin.columns:
        raise ValueError(f"Expected column 'Core' in TMA clinical CSV: {path}")

    clin["coord"] = extract_coord_token(clin["Core"].astype(str))
    clin["coord_key"] = extract_coord_key(clin["Core"].astype(str))

    if "Panel" in clin.columns:
        clin["panel"] = clin["Panel"].map(normalize_panel)
    else:
        clin["panel"] = "Unknown"

    if "core_idx" in clin.columns:
        parsed = clin["core_idx"].apply(parse_core_idx)
        clin["cohort_code"] = [x[0] for x in parsed]
        clin["tma"] = [x[1] for x in parsed]
        clin["cohort_label"] = clin["cohort_code"].map(COHORT_CODE_MAP).fillna("Unknown")
    else:
        clin["cohort_code"] = pd.NA
        clin["tma"] = clin["Core"].map(infer_tma_from_sample_name)
        clin["cohort_label"] = clin["Core"].map(infer_cohort_from_sample_name)

    if "TURBT_or_RC" in clin.columns:
        clin["sample_type"] = clin["TURBT_or_RC"].map(norm_ws).replace({"": "Unknown"})
    else:
        clin["sample_type"] = "Unknown"

    keep = [
        "coord", "coord_key", "panel", "cohort_code", "cohort_label", "tma", "sample_type",
        "Core", "TURBT_or_RC", "RESPONSE_CR", "RESPONSE_PR", "DEATH", "RECURRENCE",
        "TIME_ELAPSED_DEATH_EVENT", "TIME_ELAPSED_RECURRENCE_EVENT",
    ]
    keep = [c for c in keep if c in clin.columns]
    out = clin[keep].dropna(subset=["coord"]).drop_duplicates(["coord", "panel"]).copy()
    out["branch"] = "TMA"
    return out


def load_blasst_clinical(path: Path) -> pd.DataFrame:
    """Load BLASST/whole-section clinical metadata and standardize coord/panel/cohort/sample_type."""
    if not path.exists():
        print(f"[warn] BLASST clinical file not found: {path}")
        return pd.DataFrame(columns=["branch", "coord", "coord_key", "panel", "cohort_label", "sample_type"])

    meta = pd.read_csv(path, low_memory=False).copy()
    if "Core" not in meta.columns:
        raise ValueError(f"Expected column 'Core' in BLASST clinical CSV: {path}")

    meta["coord"] = extract_coord_token(meta["Core"].astype(str))
    meta["coord_key"] = extract_coord_key(meta["Core"].astype(str))
    meta["panel"] = "Unknown"  # BLASST clinical is coord-level; panel comes from parquet path.
    meta["cohort_label"] = "BLASST"

    if "TURBT_or_RC" in meta.columns:
        meta["sample_type"] = meta["TURBT_or_RC"].map(norm_ws).replace({"": "Unknown"})
    else:
        meta["sample_type"] = "Unknown"

    if "Sample_ID_Adjusted" in meta.columns:
        meta["patient_id"] = meta["Sample_ID_Adjusted"]
    elif "Sample_ID" in meta.columns:
        meta["patient_id"] = meta["Sample_ID"]
    else:
        meta["patient_id"] = pd.NA

    keep = [
        "coord", "coord_key", "panel", "cohort_label", "sample_type", "patient_id",
        "Core", "Sample_ID", "Sample_ID_Adjusted", "TURBT_or_RC",
    ]
    keep = [c for c in keep if c in meta.columns]
    out = meta[keep].dropna(subset=["coord"]).drop_duplicates("coord").copy()
    out["branch"] = "BLASST"
    return out


def load_qc_reviews(qc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all *review.csv files and deduplicate by coord.
    Historical rule: extract coord from image column and prefer rows with non-null structural_acceptability.
    """
    files = sorted(qc_dir.glob("*review.csv"))
    rows = []
    parts = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            df["__qc_file"] = fp.name
            parts.append(df)
            rows.append({"file": str(fp), "filename": fp.name, "status": "OK", "n_rows": len(df), "columns": ";".join(map(str, df.columns))})
        except Exception as e:
            rows.append({"file": str(fp), "filename": fp.name, "status": "FAIL_READ", "n_rows": 0, "error": repr(e)})

    file_inv = pd.DataFrame(rows)
    if not parts:
        return pd.DataFrame(columns=["coord", "structural_acceptability", "segmentation_comments", "__qc_file"]), file_inv

    qc = pd.concat(parts, ignore_index=True)
    if "image" not in qc.columns:
        raise ValueError("QC review files must contain an 'image' column.")

    keep = [c for c in ["image", "structural_acceptability", "segmentation_comments", "__qc_file"] if c in qc.columns]
    qc = qc[keep].copy()
    qc["coord"] = extract_coord_token(qc["image"].astype(str))
    qc["coord_key"] = extract_coord_key(qc["image"].astype(str))
    qc["panel"] = qc["__qc_file"].map(infer_review_panel_from_filename)
    qc["cohort_label_from_review_file"] = qc["__qc_file"].map(infer_review_cohort_from_filename)
    qc["branch"] = "TMA"
    qc = qc[qc["coord"].notna()].copy()

    if "structural_acceptability" not in qc.columns:
        qc["structural_acceptability"] = pd.NA
    if "segmentation_comments" not in qc.columns:
        qc["segmentation_comments"] = pd.NA

    qc["__has_sa"] = qc["structural_acceptability"].notna().astype(int)
    qc = qc.sort_values(["panel", "coord", "__has_sa"], ascending=[True, True, False]).drop_duplicates(["panel", "coord"])
    qc = qc.drop(columns=["__has_sa"])
    qc["qc_is_usable"] = qc["structural_acceptability"].map(is_usable_qc_label)
    return qc, file_inv


# -----------------------------------------------------------------------------
# Load tissue-summary rebuild
# -----------------------------------------------------------------------------

def find_first_existing(directory: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def load_tissue_inventory(summary_dir: Path) -> pd.DataFrame:
    """
    Load compact/wide tissue summary rebuild output and standardize to one row per sample/panel/coord.
    Priority:
      1. tissue_region_summary_compact.csv
      2. tissue_seg_areas_wide.csv
      3. tissue_seg_areas_merged_dedup.csv
    """
    path = find_first_existing(summary_dir, [
        "tissue_region_summary_compact.csv",
        "tissue_seg_areas_wide.csv",
        "tissue_seg_areas_merged_dedup.csv",
    ])

    if path is None:
        print(f"[warn] No tissue summary file found in: {summary_dir}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False).copy()
    df["__source_tissue_file"] = path.name

    # If long-format dedup was selected, pivot enough for QC.
    if {"sample_name", "tissue_category"}.issubset(df.columns):
        val_cols = [c for c in ["region_area_percent", "region_area_sq_microns", "region_area_pixels"] if c in df.columns]
        id_cols = [c for c in ["sample_name", "panel", "source_root", "cohort_prefix", "core_token", "coord_token"] if c in df.columns]
        if "panel" not in id_cols:
            id_cols.append("panel")
            df["panel"] = "Unknown"
        wide_pieces = []
        for val in val_cols:
            tmp = df[id_cols + ["tissue_category", val]].copy()
            tmp["metric"] = tmp["tissue_category"].map(lambda x: f"{safe_token(x)}_{val}")
            wide = tmp.pivot_table(index=id_cols, columns="metric", values=val, aggfunc="first").reset_index()
            wide.columns.name = None
            wide_pieces.append(wide)
        out = wide_pieces[0]
        for w in wide_pieces[1:]:
            out = out.merge(w, on=id_cols, how="outer")
        df = out

    if "sample_name" not in df.columns:
        raise ValueError(f"Tissue summary file lacks sample_name: {path}")

    if "panel" in df.columns:
        df["panel"] = df["panel"].map(normalize_panel)
        inferred_from_sample = df["sample_name"].map(infer_panel_from_sample_name)
        df["panel"] = df["panel"].where(df["panel"].ne("Unknown"), inferred_from_sample)
        if "source_file" in df.columns:
            inferred_from_source = df["source_file"].map(lambda x: infer_panel_from_path(Path(str(x))))
            df["panel"] = df["panel"].where(df["panel"].ne("Unknown"), inferred_from_source)
    else:
        df["panel"] = df["sample_name"].map(infer_panel_from_sample_name)

    # Extract coord. Prefer coord_token if supplied by rebuild script.
    if "coord_token" in df.columns:
        coord_from_coord_col = extract_coord_token(df["coord_token"].astype(str))
        coord_from_sample = extract_coord_token(df["sample_name"].astype(str))
        df["coord"] = coord_from_coord_col.where(coord_from_coord_col.notna(), coord_from_sample)
    else:
        df["coord"] = extract_coord_token(df["sample_name"].astype(str))

    df["coord_key"] = extract_coord_key(df["coord"].fillna(df["sample_name"]).astype(str))
    df["cohort_label_inferred"] = df["sample_name"].map(infer_cohort_from_sample_name)
    df["tma_inferred"] = df["sample_name"].map(infer_tma_from_sample_name)

    # Standardize common area column names.
    rename_candidates = {
        "Epi_region_area_percent": ["Epi_region_area_percent", "Epi_region_area_percent", "Epi_region_area_percent"],
        "Str_region_area_percent": ["Str_region_area_percent"],
        "Other_region_area_percent": ["Other_region_area_percent"],
        "Epi_region_area_sq_microns": ["Epi_region_area_sq_microns"],
        "Str_region_area_sq_microns": ["Str_region_area_sq_microns"],
        "Other_region_area_sq_microns": ["Other_region_area_sq_microns"],
    }
    # Older wide can use category tokens with original metric names already. Keep as-is when present.
    for col in [c for vals in rename_candidates.values() for c in vals]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    group_cols = ["panel", "coord", "coord_key", "sample_name"]
    # For rows with no coord (KOLL-like), sample_name remains the entity key.
    # Keep coord_key in the grouping keys so it survives the aggregation and can be used downstream.
    agg = {}
    for c in df.columns:
        if c in group_cols:
            continue
        if c.endswith("_percent") or c.endswith("_sq_microns") or c.endswith("_pixels"):
            agg[c] = "first"
    for c in ["source_root", "cohort_prefix", "core_token", "coord_token", "__source_tissue_file", "cohort_label_inferred", "tma_inferred"]:
        if c in df.columns:
            agg[c] = "first"

    if agg:
        df = df.groupby(group_cols, dropna=False, as_index=False).agg(agg)

    df["has_tissue"] = True
    return df


# -----------------------------------------------------------------------------
# Scan parquet outputs
# -----------------------------------------------------------------------------

def iter_parquet_files(root: Path) -> list[Path]:
    if not root.exists():
        print(f"[warn] parquet root does not exist: {root}")
        return []
    files = []
    for fp in sorted(root.rglob("*.parquet")):
        # Avoid reading previous QC outputs if accidentally placed under the same tree.
        if any(part.startswith("_") for part in fp.relative_to(root).parts):
            continue
        files.append(fp)
    return files


def read_parquet_minimal(fp: Path, wanted: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(fp, columns=wanted)
    except Exception:
        df = pd.read_parquet(fp)
        keep = [c for c in wanted if c in df.columns]
        return df[keep].copy()


def scan_parquet_core_inventory(root: Path, branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one row per branch/panel/coord from combined per-cell parquet outputs.
    """
    files = iter_parquet_files(root)
    file_rows = []
    parts = []

    wanted = ["sample_name", "phenotype_combined", "tissue_category"]
    for fp in files:
        panel_from_path = infer_panel_from_path(fp)
        try:
            df = read_parquet_minimal(fp, wanted)
            status = "OK"
            error = ""
        except Exception as e:
            file_rows.append({"branch": branch, "parquet": str(fp), "panel_from_path": panel_from_path, "status": "FAIL_READ", "error": repr(e)})
            continue

        if "sample_name" not in df.columns:
            file_rows.append({"branch": branch, "parquet": str(fp), "panel_from_path": panel_from_path, "status": "MISSING_SAMPLE_NAME", "n_rows": len(df)})
            continue

        df["coord"] = extract_coord_token(df["sample_name"].astype(str))
        df["coord_key"] = extract_coord_key(df["sample_name"].astype(str))
        df = df[df["coord"].notna()].copy()
        if df.empty:
            file_rows.append({"branch": branch, "parquet": str(fp), "panel_from_path": panel_from_path, "status": "NO_COORDS", "n_rows": 0})
            continue

        if panel_from_path == "Unknown":
            sample_panels = df["sample_name"].map(infer_panel_from_sample_name)
            non_unknown = sample_panels[sample_panels.ne("Unknown")]
            panel_from_path = non_unknown.mode().iat[0] if not non_unknown.empty else "Unknown"
        df["panel"] = panel_from_path
        if "phenotype_combined" not in df.columns:
            df["phenotype_combined"] = pd.NA
        if "tissue_category" not in df.columns:
            df["tissue_category"] = pd.NA

        df["phenotype_clean"] = df["phenotype_combined"].map(clean_pheno_combo)
        df["cohort_label_inferred"] = df["sample_name"].map(infer_cohort_from_sample_name)
        df["tma_inferred"] = df["sample_name"].map(infer_tma_from_sample_name)

        # Core-level summaries from this parquet part.
        g = (
            df.groupby(["panel", "coord", "coord_key"], dropna=False)
              .agg(
                  n_parquet_cells=("sample_name", "size"),
                  n_unique_sample_names=("sample_name", "nunique"),
                  example_sample_name=("sample_name", "first"),
                  n_unique_phenotype_combined=("phenotype_combined", "nunique"),
                  n_unique_phenotype_clean=("phenotype_clean", "nunique"),
                  cohort_label_inferred=("cohort_label_inferred", "first"),
                  tma_inferred=("tma_inferred", "first"),
              )
              .reset_index()
        )

        # Tissue-category counts.
        tc = (
            df.groupby(["panel", "coord", "coord_key", "tissue_category"], dropna=False)
              .size()
              .reset_index(name="n_cells")
        )
        if not tc.empty:
            tc["tissue_token"] = tc["tissue_category"].map(safe_token)
            tc_w = tc.pivot_table(
                index=["panel", "coord", "coord_key"],
                columns="tissue_token",
                values="n_cells",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()
            tc_w.columns.name = None
            for col in ["Epi", "Str", "Other"]:
                if col not in tc_w.columns:
                    tc_w[col] = 0
            tc_w = tc_w.rename(columns={"Epi": "n_parquet_epi_cells", "Str": "n_parquet_str_cells", "Other": "n_parquet_other_cells"})
            g = g.merge(tc_w[["panel", "coord", "coord_key", "n_parquet_epi_cells", "n_parquet_str_cells", "n_parquet_other_cells"]],
                        on=["panel", "coord", "coord_key"], how="left")

        g["branch"] = branch
        g["source_parquet_count"] = 1
        g["source_parquets"] = str(fp)
        parts.append(g)

        file_rows.append({
            "branch": branch,
            "parquet": str(fp),
            "relative_parquet": str(fp.relative_to(root)),
            "panel_from_path": panel_from_path,
            "status": status,
            "n_rows": int(len(df)),
            "n_coords": int(df["coord"].nunique()),
            "error": error,
        })

    if not parts:
        return pd.DataFrame(), pd.DataFrame(file_rows)

    raw = pd.concat(parts, ignore_index=True)

    sum_cols = [
        "n_parquet_cells",
        "n_parquet_epi_cells",
        "n_parquet_str_cells",
        "n_parquet_other_cells",
        "source_parquet_count",
    ]
    for c in sum_cols:
        if c not in raw.columns:
            raw[c] = 0

    out = (
        raw.groupby(["branch", "panel", "coord", "coord_key"], dropna=False)
           .agg(
               n_parquet_cells=("n_parquet_cells", "sum"),
               n_parquet_epi_cells=("n_parquet_epi_cells", "sum"),
               n_parquet_str_cells=("n_parquet_str_cells", "sum"),
               n_parquet_other_cells=("n_parquet_other_cells", "sum"),
               n_unique_sample_names=("n_unique_sample_names", "max"),
               example_sample_name=("example_sample_name", "first"),
               n_unique_phenotype_combined=("n_unique_phenotype_combined", "max"),
               n_unique_phenotype_clean=("n_unique_phenotype_clean", "max"),
               cohort_label_inferred=("cohort_label_inferred", "first"),
               tma_inferred=("tma_inferred", "first"),
               source_parquet_count=("source_parquet_count", "sum"),
               source_parquets=("source_parquets", lambda s: ";".join(sorted(set(map(str, s))))),
           )
           .reset_index()
    )
    out["has_parquet"] = True
    return out, pd.DataFrame(file_rows)


# -----------------------------------------------------------------------------
# Merge / coverage functions
# -----------------------------------------------------------------------------


def ensure_identity_columns(df: pd.DataFrame, branch: Optional[str] = None) -> pd.DataFrame:
    """
    Defensive harmonization for source-inventory tables before coverage joins.

    Guarantees the columns used for matching exist:
      branch, panel, coord, coord_key

    This is intentionally permissive because different inputs identify samples
    using different columns (coord, coord_token, sample_name, Core, image).
    """
    out = df.copy()
    if out.empty:
        for c in ["branch", "panel", "coord", "coord_key"]:
            if c not in out.columns:
                out[c] = pd.Series(dtype="object")
        return out

    if branch is not None and "branch" not in out.columns:
        out["branch"] = branch

    if "panel" not in out.columns:
        out["panel"] = "Unknown"
    out["panel"] = out["panel"].map(normalize_panel)

    if "coord" not in out.columns:
        source_col = None
        for cand in ["coord_token", "sample_name", "Core", "image", "core_key", "region_id"]:
            if cand in out.columns:
                source_col = cand
                break
        if source_col is None:
            out["coord"] = pd.NA
        else:
            out["coord"] = extract_coord_token(out[source_col].astype(str))

    if "coord_key" not in out.columns:
        fallback = out["coord"].astype(str)
        for cand in ["coord_token", "sample_name", "Core", "image", "core_key", "region_id"]:
            if cand in out.columns:
                fallback = out["coord"].where(out["coord"].notna(), out[cand].astype(str))
                break
        out["coord_key"] = extract_coord_key(fallback.astype(str))

    return out

def prepare_tissue_for_branch(tissue: pd.DataFrame, branch: str) -> pd.DataFrame:
    """Subset tissue summary to inferred branch using cohort/sample naming."""
    if tissue.empty:
        return tissue.copy()

    out = tissue.copy()
    if "cohort_label_inferred" not in out.columns:
        out["cohort_label_inferred"] = out["sample_name"].map(infer_cohort_from_sample_name)

    if branch == "TMA":
        # Exclude BLASST/KOLL when recognizable. Also exclude rows whose raw root
        # clearly came from Whole Sections, since those are the BLASST branch.
        mask = ~out["cohort_label_inferred"].isin(["BLASST", "KOLL"])
        if "source_root" in out.columns:
            mask = mask & ~out["source_root"].astype(str).str.contains("whole|blasst", case=False, regex=True, na=False)
        out = out[mask].copy()
    elif branch == "BLASST":
        # Whole-section tissue usually lives under a "Whole Sections" root and/or
        # uses BLASST-style sample IDs. Do NOT match all bracket coordinates here,
        # because TMA samples also contain bracket coordinates.
        mask = out["cohort_label_inferred"].eq("BLASST")
        if "source_root" in out.columns:
            mask = mask | out["source_root"].astype(str).str.contains("whole|blasst", case=False, regex=True, na=False)
        if "sample_name" in out.columns:
            mask = mask | out["sample_name"].astype(str).str.contains("BLASST|UU-|SF-", case=False, regex=True, na=False)
        out = out[mask].copy()
    out["branch"] = branch
    return out


def build_branch_presence(
    *,
    branch: str,
    parquet_inv: pd.DataFrame,
    tissue_inv: pd.DataFrame,
    clinical_inv: pd.DataFrame,
    qc_inv: pd.DataFrame,
    allow_coord_only_tissue_match: bool = True,
    allow_coord_only_clinical_match: bool = True,
    require_review: bool = True,
) -> pd.DataFrame:
    """Build a coord/panel-level source-presence matrix for TMA or BLASST."""
    parquet_inv = ensure_identity_columns(parquet_inv, branch=branch)
    tissue_inv = ensure_identity_columns(tissue_inv, branch=branch)
    clinical_inv = ensure_identity_columns(clinical_inv, branch=branch)
    qc_inv = ensure_identity_columns(qc_inv, branch=branch)

    frames = []

    if not parquet_inv.empty:
        frames.append(parquet_inv[["branch", "panel", "coord", "coord_key"]].drop_duplicates())

    if not tissue_inv.empty:
        t = tissue_inv.copy()
        frames.append(t[["branch", "panel", "coord", "coord_key"]].drop_duplicates())

    if not clinical_inv.empty:
        c = clinical_inv.copy()
        frames.append(c[["branch", "panel", "coord", "coord_key"]].drop_duplicates())

    # QC/review rows do not define the base universe. This avoids artificial
    # Unknown-panel review-only rows. Review is merged onto parquet/tissue/clinical
    # entities below.

    if not frames:
        return pd.DataFrame()

    base = pd.concat(frames, ignore_index=True).dropna(subset=["coord"]).drop_duplicates(["branch", "panel", "coord"])

    # Merge parquet exactly by branch/panel/coord.
    if not parquet_inv.empty:
        base = base.merge(parquet_inv, on=["branch", "panel", "coord", "coord_key"], how="left")
    else:
        base["has_parquet"] = False

    # Merge tissue exactly.
    tissue_cols = []
    if not tissue_inv.empty:
        t = tissue_inv.copy()
        if "branch" not in t.columns:
            t["branch"] = branch
        t["panel"] = t.get("panel", "Unknown").map(normalize_panel)
        t["has_tissue_exact"] = True
        tissue_cols = [c for c in t.columns if c not in {"coord_key"}]
        base = base.merge(
            t[tissue_cols + ["coord_key"]].drop_duplicates(["branch", "panel", "coord"]),
            on=["branch", "panel", "coord", "coord_key"],
            how="left",
            suffixes=("", "_tissue"),
        )
    else:
        base["has_tissue_exact"] = False

    if "has_tissue_exact" not in base.columns:
        base["has_tissue_exact"] = False
    base["has_tissue_exact"] = base["has_tissue_exact"].fillna(False).astype(bool)

    # Add coord-only tissue fallback when exact panel is absent.
    if allow_coord_only_tissue_match and not tissue_inv.empty:
        t_any = (
            tissue_inv.dropna(subset=["coord"]).groupby("coord", as_index=False)
            .agg(has_tissue_coord_any=("sample_name", lambda s: True),
                 tissue_sample_name_any=("sample_name", "first"))
        )
        base = base.merge(t_any, on="coord", how="left")
    else:
        base["has_tissue_coord_any"] = False
    base["has_tissue_coord_any"] = base["has_tissue_coord_any"].fillna(False).astype(bool)
    base["has_tissue"] = base["has_tissue_exact"] | base["has_tissue_coord_any"]

    # Merge clinical exact where possible.
    if not clinical_inv.empty:
        c = clinical_inv.copy()
        if "branch" not in c.columns:
            c["branch"] = branch
        c["panel"] = c.get("panel", "Unknown").map(normalize_panel)
        c["has_clinical_exact"] = True
        base = base.merge(
            c.drop_duplicates(["branch", "panel", "coord"]),
            on=["branch", "panel", "coord", "coord_key"],
            how="left",
            suffixes=("", "_clinical"),
        )
    else:
        base["has_clinical_exact"] = False

    if "has_clinical_exact" not in base.columns:
        base["has_clinical_exact"] = False
    base["has_clinical_exact"] = base["has_clinical_exact"].fillna(False).astype(bool)

    # Clinical coord-only fallback (important for BLASST and for coord-level metadata).
    if allow_coord_only_clinical_match and not clinical_inv.empty:
        keep_cols = ["coord", "cohort_label", "tma", "sample_type", "patient_id", "Core", "TURBT_or_RC"]
        keep_cols = [c for c in keep_cols if c in clinical_inv.columns]
        c_any = clinical_inv[keep_cols].dropna(subset=["coord"]).drop_duplicates("coord").copy()
        rename = {c: f"{c}_coord_any" for c in keep_cols if c != "coord"}
        c_any = c_any.rename(columns=rename)
        c_any["has_clinical_coord_any"] = True
        base = base.merge(c_any, on="coord", how="left")
    else:
        base["has_clinical_coord_any"] = False

    base["has_clinical_coord_any"] = base["has_clinical_coord_any"].fillna(False).astype(bool)
    base["has_clinical"] = base["has_clinical_exact"] | base["has_clinical_coord_any"]

    # Fill key clinical columns from exact then coord-any.
    for col in ["cohort_label", "tma", "sample_type", "patient_id"]:
        coord_col = f"{col}_coord_any"
        if col not in base.columns:
            base[col] = pd.NA
        if coord_col in base.columns:
            base[col] = base[col].where(base[col].notna() & (base[col].astype(str) != ""), base[coord_col])

    # Fall back to sample-name inferred cohort/TMA from parquet/tissue.
    if "cohort_label_inferred" in base.columns:
        base["cohort_label"] = base["cohort_label"].where(base["cohort_label"].notna(), base["cohort_label_inferred"])
    if "tma_inferred" in base.columns:
        base["tma"] = base["tma"].where(base["tma"].notna(), base["tma_inferred"])
    base["cohort_label"] = base["cohort_label"].fillna("Unknown")
    base["sample_type"] = base["sample_type"].fillna("Unknown")
    base["tma"] = base["tma"].fillna("Unknown")

    # Merge QC/review. Prefer exact panel+coord when review files have panel labels;
    # fall back to coord-only only for review rows with Unknown panel.
    if not qc_inv.empty:
        q_cols = ["branch", "panel", "coord", "coord_key", "structural_acceptability", "segmentation_comments", "__qc_file", "qc_is_usable"]
        q_cols = [c for c in q_cols if c in qc_inv.columns]
        q_exact = qc_inv[q_cols].copy()
        q_exact = q_exact[q_exact["panel"].ne("Unknown")].drop_duplicates(["branch", "panel", "coord"])
        q_exact["has_review"] = True
        base = base.merge(q_exact, on=["branch", "panel", "coord", "coord_key"], how="left", suffixes=("", "_review"))

        if "has_review" not in base.columns:
            base["has_review"] = False
        base["has_review"] = base["has_review"].fillna(False).astype(bool)

        unmatched = base[~base["has_review"]].copy()
        q_any = qc_inv[qc_inv["panel"].eq("Unknown")][["coord", "structural_acceptability", "segmentation_comments", "__qc_file", "qc_is_usable"]].drop_duplicates("coord") if "panel" in qc_inv.columns else pd.DataFrame()
        if not q_any.empty and not unmatched.empty:
            q_any["has_review_coord_any"] = True
            base = base.merge(q_any.rename(columns={
                "structural_acceptability": "structural_acceptability_coord_any",
                "segmentation_comments": "segmentation_comments_coord_any",
                "__qc_file": "__qc_file_coord_any",
                "qc_is_usable": "qc_is_usable_coord_any",
            }), on="coord", how="left")
            base["has_review"] = base["has_review"] | base.get("has_review_coord_any", False).fillna(False).astype(bool)
    else:
        base["has_review"] = False
    base["has_review"] = base["has_review"].fillna(False).astype(bool)

    # Flags.
    if "has_parquet" not in base.columns:
        base["has_parquet"] = False
    base["has_parquet"] = base["has_parquet"].fillna(False).astype(bool)

    base["review_expected"] = bool(require_review)
    if require_review:
        base["has_all_primary_inputs"] = base["has_parquet"] & base["has_tissue"] & base["has_review"] & base["has_clinical"]
    else:
        base["has_all_primary_inputs"] = base["has_parquet"] & base["has_tissue"] & base["has_clinical"]
    base["entity_id"] = base.apply(
        lambda r: f"{r['branch']}|{r['panel']}|{r['coord']}" if pd.notna(r["coord"]) else f"{r['branch']}|{r['panel']}|{r.get('sample_name', '')}",
        axis=1,
    )

    # Sort useful columns first.
    first_cols = [
        "entity_id", "branch", "panel", "coord", "coord_key", "cohort_label", "tma", "sample_type",
        "has_parquet", "has_tissue", "has_tissue_exact", "has_tissue_coord_any",
        "has_review", "has_clinical", "has_clinical_exact", "has_clinical_coord_any",
        "has_all_primary_inputs", "structural_acceptability", "qc_is_usable",
        "n_parquet_cells", "n_parquet_epi_cells", "n_parquet_str_cells", "n_parquet_other_cells",
        "example_sample_name", "sample_name",
    ]
    first_cols = [c for c in first_cols if c in base.columns]
    rest = [c for c in base.columns if c not in first_cols]
    return base[first_cols + rest].sort_values(["branch", "panel", "cohort_label", "tma", "sample_type", "coord"]).reset_index(drop=True)


def summarize_presence(presence: pd.DataFrame) -> pd.DataFrame:
    if presence.empty:
        return pd.DataFrame()

    group_cols = ["branch", "panel", "cohort_label", "tma", "sample_type"]
    for c in group_cols:
        if c not in presence.columns:
            presence[c] = "Unknown"

    def ntrue(s: pd.Series) -> int:
        return int(s.fillna(False).astype(bool).sum())

    out = (
        presence.groupby(group_cols, dropna=False)
        .agg(
            n_entities=("entity_id", "nunique"),
            n_with_parquet=("has_parquet", ntrue),
            n_with_tissue=("has_tissue", ntrue),
            n_with_review=("has_review", ntrue),
            n_with_clinical=("has_clinical", ntrue),
            n_with_all_primary_inputs=("has_all_primary_inputs", ntrue),
            n_parquet_cells_total=("n_parquet_cells", "sum"),
        )
        .reset_index()
    )

    for col in ["parquet", "tissue", "review", "clinical", "all_primary_inputs"]:
        n_col = f"n_with_{col}"
        frac_col = f"frac_with_{col}"
        if n_col in out.columns:
            out[frac_col] = out[n_col] / out["n_entities"].replace(0, np.nan)

    return out.sort_values(group_cols).reset_index(drop=True)


def make_mismatch_tables(presence: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if presence.empty:
        return {}

    tables = {
        "parquet_missing_tissue.csv": presence[presence["has_parquet"] & ~presence["has_tissue"]].copy(),
        "tissue_missing_parquet.csv": presence[presence["has_tissue"] & ~presence["has_parquet"]].copy(),
        "parquet_missing_review.csv": presence[presence["has_parquet"] & ~presence["has_review"]].copy(),
        "review_missing_parquet.csv": presence[presence["has_review"] & ~presence["has_parquet"]].copy(),
        "parquet_missing_clinical.csv": presence[presence["has_parquet"] & ~presence["has_clinical"]].copy(),
        "clinical_missing_parquet.csv": presence[presence["has_clinical"] & ~presence["has_parquet"]].copy(),
        "primary_input_incomplete.csv": presence[~presence["has_all_primary_inputs"]].copy(),
    }
    return tables


def duplicate_diagnostics(*, parquet_inv: pd.DataFrame, tissue_inv: pd.DataFrame, clinical_inv: pd.DataFrame, qc_inv: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_dup(df: pd.DataFrame, name: str, keys: list[str]):
        if df.empty or any(k not in df.columns for k in keys):
            rows.append({"source": name, "keys": ",".join(keys), "n_duplicate_key_rows": 0, "n_duplicate_keys": 0})
            return
        dup = df[df.duplicated(keys, keep=False)].copy()
        rows.append({
            "source": name,
            "keys": ",".join(keys),
            "n_rows": len(df),
            "n_duplicate_key_rows": len(dup),
            "n_duplicate_keys": dup[keys].drop_duplicates().shape[0] if not dup.empty else 0,
        })

    add_dup(parquet_inv, "parquet_inventory", ["branch", "panel", "coord"])
    add_dup(tissue_inv, "tissue_inventory", ["branch", "panel", "coord"])
    add_dup(clinical_inv, "clinical_inventory", ["branch", "panel", "coord"])
    add_dup(qc_inv, "qc_reviews", ["coord"])

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# KOLL-specific QC
# -----------------------------------------------------------------------------

def load_koll_qc(koll_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KOLL/Florestan-specific sample-level QC.
    Expected files from build_koll_florestan_summary_tables.py:
      - florestan_tissue_region_summary_compact.csv
      - florestan_cell_seg_BT.csv
      - florestan_cell_seg_AR.csv
    """
    if not koll_dir.exists():
        print(f"[warn] KOLL directory not found: {koll_dir}")
        return pd.DataFrame(), pd.DataFrame()

    tissue_candidates = [
        "florestan_tissue_region_summary_compact.csv",
        "tissue_region_summary_compact.csv",
    ]
    tissue_path = find_first_existing(koll_dir, tissue_candidates)
    tissue = pd.DataFrame()
    if tissue_path is not None:
        tissue = pd.read_csv(tissue_path, low_memory=False).copy()
        tissue["has_tissue"] = True
        tissue["tissue_source_file"] = tissue_path.name
    else:
        print(f"[warn] No KOLL tissue compact file found in {koll_dir}")

    cell_files = []
    for pat in ["*cell_seg_BT.csv", "*cell_seg_AR.csv", "florestan_cell_seg_*.csv"]:
        cell_files.extend(sorted(koll_dir.glob(pat)))
    cell_files = sorted(set(cell_files), key=lambda p: p.name.lower())

    cell_parts = []
    file_rows = []
    for fp in cell_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            file_rows.append({"file": str(fp), "filename": fp.name, "status": "FAIL_READ", "error": repr(e)})
            continue

        if "sample_name" not in df.columns:
            file_rows.append({"file": str(fp), "filename": fp.name, "status": "MISSING_SAMPLE_NAME", "n_rows": len(df)})
            continue

        if "Panel" in df.columns:
            df["panel"] = df["Panel"].map(normalize_panel)
        else:
            df["panel"] = infer_panel_from_path(fp)

        if "phenotype" not in df.columns:
            df["phenotype"] = pd.NA
        if "tissue_region" not in df.columns:
            df["tissue_region"] = pd.NA

        g = (
            df.groupby(["panel", "sample_name"], dropna=False)
            .agg(
                n_cell_rows=("sample_name", "size"),
                n_unique_phenotypes=("phenotype", "nunique"),
                n_tissue_regions=("tissue_region", "nunique"),
            )
            .reset_index()
        )
        tr = (
            df.groupby(["panel", "sample_name", "tissue_region"], dropna=False)
            .size()
            .reset_index(name="n_cells")
        )
        if not tr.empty:
            tr["tissue_token"] = tr["tissue_region"].map(safe_token)
            trw = tr.pivot_table(index=["panel", "sample_name"], columns="tissue_token", values="n_cells", fill_value=0, aggfunc="sum").reset_index()
            trw.columns.name = None
            trw = trw.rename(columns={"Epi": "n_epi_cells", "Str": "n_str_cells", "Other": "n_other_cells"})
            g = g.merge(trw, on=["panel", "sample_name"], how="left")
        g["has_cell_csv"] = True
        g["cell_source_file"] = fp.name
        cell_parts.append(g)
        file_rows.append({"file": str(fp), "filename": fp.name, "status": "OK", "n_rows": len(df), "n_samples": df["sample_name"].nunique()})

    cells = pd.concat(cell_parts, ignore_index=True) if cell_parts else pd.DataFrame()
    file_inv = pd.DataFrame(file_rows)

    # Build sample/panel presence. Tissue is sample-level, so duplicate tissue onto each observed panel.
    frames = []
    if not cells.empty:
        frames.append(cells[["panel", "sample_name"]].drop_duplicates())
    if not tissue.empty and "sample_name" in tissue.columns:
        # Add Unknown rows for tissue-only samples; panel-specific rows will be filled by merge below.
        tmp = tissue[["sample_name"]].drop_duplicates().copy()
        tmp["panel"] = "Unknown"
        frames.append(tmp[["panel", "sample_name"]])

    if not frames:
        return pd.DataFrame(), file_inv

    base = pd.concat(frames, ignore_index=True).drop_duplicates(["panel", "sample_name"])
    if not cells.empty:
        base = base.merge(cells, on=["panel", "sample_name"], how="left")
    else:
        base["has_cell_csv"] = False

    if not tissue.empty and "sample_name" in tissue.columns:
        t = tissue.drop_duplicates("sample_name").copy()
        t_any = t.copy()
        t_any["has_tissue"] = True
        base = base.merge(t_any, on="sample_name", how="left", suffixes=("", "_tissue"))
    else:
        base["has_tissue"] = False

    base["has_cell_csv"] = base["has_cell_csv"].fillna(False).astype(bool)
    base["has_tissue"] = base["has_tissue"].fillna(False).astype(bool)
    base["has_both_tissue_and_cells"] = base["has_cell_csv"] & base["has_tissue"]
    base["branch"] = "KOLL"
    base["cohort_label"] = "KOLL"
    return base.sort_values(["panel", "sample_name"]).reset_index(drop=True), file_inv


# -----------------------------------------------------------------------------
# Output documentation
# -----------------------------------------------------------------------------

def write_output_manifest(out_dir: Path, generated: dict[str, str]) -> None:
    rows = [{"file": k, "description": v} for k, v in generated.items()]
    desc = pd.DataFrame(rows)
    write_csv(desc, out_dir / "output_file_descriptions.csv")

    lines = [
        "# QC Check Outputs",
        "",
        "Generated by `qc_check.py`.",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    for k, v in generated.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    (out_dir / "README_outputs.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] README_outputs.md")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile tissue summaries, parquet cell-level outputs, clinical metadata, and review files.")
    ap.add_argument("--inform_summary_dir", default=str(DEFAULT_INFORM_SUMMARY_DIR))
    ap.add_argument("--tma_parquet_dir", default=str(DEFAULT_TMA_PARQUET_DIR))
    ap.add_argument("--whole_parquet_dir", default=str(DEFAULT_WHOLE_PARQUET_DIR))
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Directory containing *review.csv files.")
    ap.add_argument("--tma_clinical", default=str(DEFAULT_TMA_CLINICAL))
    ap.add_argument("--blasst_clinical", default=str(DEFAULT_BLASST_CLINICAL))
    ap.add_argument("--koll_dir", default=str(DEFAULT_KOLL_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--skip_koll", action="store_true")
    ap.add_argument("--skip_tma", action="store_true")
    ap.add_argument("--skip_blasst", action="store_true")
    ap.add_argument("--no_coord_only_fallback", action="store_true", help="Require exact panel+coord matches for tissue/clinical rather than allowing coord-only fallback.")
    args = ap.parse_args()

    inform_summary_dir = Path(args.inform_summary_dir)
    tma_parquet_dir = Path(args.tma_parquet_dir)
    whole_parquet_dir = Path(args.whole_parquet_dir)
    data_dir = Path(args.data_dir)
    tma_clinical_path = Path(args.tma_clinical)
    blasst_clinical_path = Path(args.blasst_clinical)
    koll_dir = Path(args.koll_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "script": "qc_check_v3.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inform_summary_dir": str(inform_summary_dir),
        "tma_parquet_dir": str(tma_parquet_dir),
        "whole_parquet_dir": str(whole_parquet_dir),
        "data_dir": str(data_dir),
        "tma_clinical": str(tma_clinical_path),
        "blasst_clinical": str(blasst_clinical_path),
        "koll_dir": str(koll_dir),
        "out_dir": str(out_dir),
        "coord_only_fallback": not args.no_coord_only_fallback,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[write] run_metadata.json")

    generated: dict[str, str] = {
        "run_metadata.json": "Command-level provenance for this QC run.",
    }

    # Load shared review files and tissue summary.
    qc, qc_file_inv = load_qc_reviews(data_dir)
    write_csv(qc_file_inv, out_dir / "review_file_inventory.csv")
    write_csv(qc, out_dir / "qc_by_coord.csv")
    generated["review_file_inventory.csv"] = "One row per *review.csv file read from data_dir."
    generated["qc_by_coord.csv"] = "Deduplicated manual review table, one row per coordinate, using structural_acceptability when available."

    tissue_all = load_tissue_inventory(inform_summary_dir)
    if not tissue_all.empty:
        write_csv(tissue_all, out_dir / "tissue_inventory_all.csv")
        generated["tissue_inventory_all.csv"] = "Standardized tissue-area inventory from inform_summary_rebuild, one row per sample/panel/coord where possible."

    all_presence = []
    all_parquet_files = []
    all_clinical = []

    if not args.skip_tma:
        print("\n[section] TMA")
        tma_clin = load_tma_clinical(tma_clinical_path)
        tma_parq, tma_parq_files = scan_parquet_core_inventory(tma_parquet_dir, branch="TMA")
        tma_tissue = prepare_tissue_for_branch(tissue_all, "TMA") if not tissue_all.empty else pd.DataFrame()

        write_csv(tma_clin, out_dir / "clinical_inventory_tma.csv")
        write_csv(tma_parq, out_dir / "parquet_core_inventory_tma.csv")
        write_csv(tma_parq_files, out_dir / "parquet_file_inventory_tma.csv")
        generated["clinical_inventory_tma.csv"] = "TMA clinical metadata standardized by coord/panel, including cohort/tma/sample_type when available."
        generated["parquet_core_inventory_tma.csv"] = "Per-core/per-panel inventory from combined_cohorts parquet files with cell counts and phenotype richness."
        generated["parquet_file_inventory_tma.csv"] = "Per-parquet file read status and row/coord counts for combined_cohorts."

        tma_presence = build_branch_presence(
            branch="TMA",
            parquet_inv=tma_parq,
            tissue_inv=tma_tissue,
            clinical_inv=tma_clin,
            qc_inv=qc,
            allow_coord_only_tissue_match=not args.no_coord_only_fallback,
            allow_coord_only_clinical_match=not args.no_coord_only_fallback,
            require_review=True,
        )
        write_csv(tma_presence, out_dir / "source_presence_tma.csv")
        generated["source_presence_tma.csv"] = "Coord/panel-level TMA source-presence matrix across parquet, tissue, review, and clinical inputs."
        all_presence.append(tma_presence)
        all_parquet_files.append(tma_parq_files)
        all_clinical.append(tma_clin)

    if not args.skip_blasst:
        print("\n[section] BLASST / whole sections")
        blasst_clin = load_blasst_clinical(blasst_clinical_path)
        whole_parq, whole_parq_files = scan_parquet_core_inventory(whole_parquet_dir, branch="BLASST")
        blasst_tissue = prepare_tissue_for_branch(tissue_all, "BLASST") if not tissue_all.empty else pd.DataFrame()

        write_csv(blasst_clin, out_dir / "clinical_inventory_blasst.csv")
        write_csv(whole_parq, out_dir / "parquet_core_inventory_blasst.csv")
        write_csv(whole_parq_files, out_dir / "parquet_file_inventory_blasst.csv")
        generated["clinical_inventory_blasst.csv"] = "BLASST clinical metadata standardized by coord, patient, and TURBT/RC sample type."
        generated["parquet_core_inventory_blasst.csv"] = "Per-core/per-panel inventory from combined_wholesections parquet files with cell counts and phenotype richness."
        generated["parquet_file_inventory_blasst.csv"] = "Per-parquet file read status and row/coord counts for combined_wholesections."

        blasst_presence = build_branch_presence(
            branch="BLASST",
            parquet_inv=whole_parq,
            tissue_inv=blasst_tissue,
            clinical_inv=blasst_clin,
            qc_inv=pd.DataFrame(),
            allow_coord_only_tissue_match=not args.no_coord_only_fallback,
            allow_coord_only_clinical_match=not args.no_coord_only_fallback,
            require_review=False,
        )
        write_csv(blasst_presence, out_dir / "source_presence_blasst.csv")
        generated["source_presence_blasst.csv"] = "Coord/panel-level BLASST source-presence matrix across parquet, tissue, review, and clinical inputs."
        all_presence.append(blasst_presence)
        all_parquet_files.append(whole_parq_files)
        all_clinical.append(blasst_clin)

    if all_presence:
        presence = pd.concat(all_presence, ignore_index=True, sort=False)
        write_csv(presence, out_dir / "source_presence_all.csv")
        generated["source_presence_all.csv"] = "Combined TMA + BLASST coord/panel source-presence matrix."

        summary = summarize_presence(presence)
        write_csv(summary, out_dir / "coverage_summary_by_panel_cohort_tma_sample_type.csv")
        generated["coverage_summary_by_panel_cohort_tma_sample_type.csv"] = "Counts and fractions of entities represented in parquet, tissue, review, clinical, and all primary inputs by branch/panel/cohort/TMA/sample_type."

        mismatch_tables = make_mismatch_tables(presence)
        mismatch_dir = out_dir / "mismatches"
        mismatch_dir.mkdir(parents=True, exist_ok=True)
        for fname, df in mismatch_tables.items():
            write_csv(df, mismatch_dir / fname)
            generated[f"mismatches/{fname}"] = f"Detailed source mismatch list: {fname.replace('.csv', '').replace('_', ' ')}."

        parquet_all = pd.concat(all_parquet_files, ignore_index=True, sort=False) if all_parquet_files else pd.DataFrame()
        clinical_all = pd.concat(all_clinical, ignore_index=True, sort=False) if all_clinical else pd.DataFrame()
        dup = duplicate_diagnostics(
            parquet_inv=presence[presence.get("has_parquet", False)].copy() if "has_parquet" in presence.columns else pd.DataFrame(),
            tissue_inv=tissue_all.assign(branch="ALL") if not tissue_all.empty else pd.DataFrame(),
            clinical_inv=clinical_all,
            qc_inv=qc,
        )
        write_csv(dup, out_dir / "duplicate_diagnostics.csv")
        generated["duplicate_diagnostics.csv"] = "High-level duplicate-key diagnostics for parquet, tissue, clinical, and review sources."

    if not args.skip_koll:
        print("\n[section] KOLL / Florestan")
        koll_presence, koll_file_inv = load_koll_qc(koll_dir)
        write_csv(koll_file_inv, out_dir / "koll_file_inventory.csv")
        generated["koll_file_inventory.csv"] = "Read status and basic counts for KOLL/Florestan cell CSV files."

        if not koll_presence.empty:
            write_csv(koll_presence, out_dir / "source_presence_koll.csv")
            generated["source_presence_koll.csv"] = "KOLL sample/panel-level source-presence matrix between tissue compact file and AR/BT cell CSVs."

            koll_summary = (
                koll_presence.groupby(["branch", "panel", "cohort_label"], dropna=False)
                .agg(
                    n_samples=("sample_name", "nunique"),
                    n_with_cell_csv=("has_cell_csv", lambda s: int(s.fillna(False).astype(bool).sum())),
                    n_with_tissue=("has_tissue", lambda s: int(s.fillna(False).astype(bool).sum())),
                    n_with_both_tissue_and_cells=("has_both_tissue_and_cells", lambda s: int(s.fillna(False).astype(bool).sum())),
                    n_cell_rows_total=("n_cell_rows", "sum"),
                )
                .reset_index()
            )
            write_csv(koll_summary, out_dir / "coverage_summary_koll.csv")
            generated["coverage_summary_koll.csv"] = "KOLL sample/panel-level coverage summary for tissue and cell CSV outputs."

            koll_mismatch_dir = out_dir / "mismatches"
            koll_mismatch_dir.mkdir(parents=True, exist_ok=True)
            write_csv(koll_presence[koll_presence["has_cell_csv"] & ~koll_presence["has_tissue"]], koll_mismatch_dir / "koll_cells_missing_tissue.csv")
            write_csv(koll_presence[koll_presence["has_tissue"] & ~koll_presence["has_cell_csv"]], koll_mismatch_dir / "koll_tissue_missing_cells.csv")
            generated["mismatches/koll_cells_missing_tissue.csv"] = "KOLL samples with AR/BT cell rows but no tissue compact row."
            generated["mismatches/koll_tissue_missing_cells.csv"] = "KOLL tissue compact samples with no AR/BT cell rows."

    write_output_manifest(out_dir, generated)

    print("\nDone.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
