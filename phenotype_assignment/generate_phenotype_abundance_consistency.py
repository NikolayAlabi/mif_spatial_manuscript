#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_phenotype_abundance_consistency_v2.py

Rebuild panel-specific phenotype abundance/consistency tables from combined
cell-level inForm parquet files.

This version keeps phenotype curation metrics restricted to development TMA
cohorts by default (NAC2020, No-NAC, PURE01), while the holdout audit can include
both the held-out NAC2015 TMA cohort and BLASST whole-section samples.

Key design choices
------------------
1. Marker combinations are canonicalized before summarization:
   - split marker strings on ';', '/', or '|'
   - keep positive marker calls ending in '+'
   - drop negative marker calls ending in '-'
   - sort positive markers alphabetically
   - if no positive markers remain, label as 'ALL_NEG'

   Examples:
       'CD68+;PD1-;PanCK+' -> 'CD68+;PanCK+'
       'CD68+/PD1-'        -> 'CD68+'
       'PD1-;PDL1-'        -> 'ALL_NEG'

2. Default curation cohorts:
       NAC2020, No-NAC, PURE01

3. Default holdout audit cohorts:
       NAC2015, BLASST

4. BLASST whole-section data are optional. If --whole_parquet_dir is supplied,
   BLASST counts are included in all_panels_core_counts.csv and in each panel's
   holdout audit table. If --blasst_metadata_csv is supplied, BLASST TURBT/RC
   sample type is merged from metadata by coordinate.

Outputs
-------
For each panel, e.g. AR/BT/MY:
    <PANEL>_phenotype_abundance_consistency_normalized.csv
    <PANEL>_phenotype_abundance_consistency_core_counts.csv
    <PANEL>_phenotype_abundance_consistency_holdout_audit.csv

Also writes:
    all_panels_core_counts.csv
    run_metadata.json
    README_phenotype_abundance_rebuild.md
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


COORD_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")

ANNOTATION_COLS = [
    "assigned_label",
    "artifact_flag",
    "simple_spatial",
    "state_spatial",
    "state",
    "exploratory",
    "lineage",
    "collapse_label",
    "artifact_reason",
    "review_notes",
]

BOOL_ANNOTATION_COLS = [
    "artifact_flag",
    "simple_spatial",
    "state_spatial",
    "exploratory",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def safe_read_csv(fp: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV robustly across common encodings."""
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else ValueError(f"Could not read {fp}")


def parse_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def extract_coord_token_series(series: pd.Series) -> pd.Series:
    """Extract bracket coordinate token '[x,y]' from strings."""
    m = series.astype(str).str.extract(COORD_RE)
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    ok = m[0].notna() & m[1].notna()
    out.loc[ok] = "[" + m.loc[ok, 0].astype(str) + "," + m.loc[ok, 1].astype(str) + "]"
    return out


def coord_to_key(x) -> object:
    """Convert '[x,y]' or 'x_y'/'x,y' to 'x_y' for robust metadata joins."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    m = COORD_RE.search(s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    s = s.replace(" ", "").replace(",", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    if re.match(r"^\d+_\d+$", s):
        return s
    return pd.NA


def key_to_coord(x) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    m = re.match(r"^(\d+)_(\d+)$", s)
    if m:
        return f"[{m.group(1)},{m.group(2)}]"
    return x


def canonicalize_marker_combo(x) -> str:
    """
    Canonicalize marker combination by ignoring negative marker calls.

    Handles semicolon-, slash-, and pipe-separated strings.
    """
    if pd.isna(x):
        return "ALL_NEG"

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return "ALL_NEG"

    parts = re.split(r"[;/|]+", s)
    parts = [p.strip() for p in parts if str(p).strip() != ""]

    pos = []
    for p in parts:
        if p.lower() in {"marker-", "all_neg"}:
            continue
        if p.endswith("+"):
            pos.append(p)
        elif p.endswith("-"):
            continue
        else:
            # Unknown un-signed tokens are ignored; audit can catch these by comparing raw tables.
            continue

    if len(pos) == 0:
        return "ALL_NEG"
    return ";".join(sorted(set(pos)))


def normalize_panel_value(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().upper()
    if s in {"AR", "ARP"} or "ARP" in s:
        return "AR"
    if s in {"BT", "B&T", "B+T", "B_T"} or "B&T" in s or "B+T" in s:
        return "BT"
    if s in {"MY", "M", "MYELOID"} or "MYELOID" in s:
        return "MY"
    return "Unknown"


def infer_panel_from_path(fp: Path) -> str:
    b = fp.name.upper()
    s = str(fp).upper()

    if "ARP" in b or re.search(r"(^|[_\-\s])AR([_\-\s.]|$)", b):
        return "AR"
    if "B&T" in b or "B+T" in b or re.search(r"(^|[_\-\s])BT([_\-\s.]|$)", b):
        return "BT"
    if "MYELOID" in b or re.search(r"(^|[_\-\s])MY([_\-\s.]|$)", b):
        return "MY"
    # Historical NAC2015 myeloid files: Bladder_19_M.parquet / Bladder_26_M.parquet
    if re.search(r"BLADDER[_\s]*(19|26)[_\s]*M\.PARQUET$", b):
        return "MY"
    if re.search(r"(^|[_\-\s])M([_\-\s.]|$)", b) and "BLADDER" in b:
        return "MY"

    if "MYELOID" in s:
        return "MY"
    if "WHOLESECTIONS_AR" in s or "WHOLESECTIONS_ARP" in s:
        return "AR"
    if "WHOLESECTIONS_B&T" in s or "WHOLESECTIONS_BT" in s or "WHOLESECTIONS_B_T" in s:
        return "BT"
    if "WHOLESECTIONS_MYELOID" in s or "WHOLESECTIONS_MY" in s:
        return "MY"
    return "Unknown"


def infer_panel_from_sample(sample: pd.Series) -> pd.Series:
    s = sample.astype(str).str.upper()
    out = pd.Series("Unknown", index=sample.index, dtype="object")
    out.loc[s.str.contains("ARP", na=False)] = "AR"
    out.loc[s.str.contains("B&T|B\+T|BT", regex=True, na=False)] = "BT"
    out.loc[s.str.contains("MYELOID", na=False)] = "MY"
    out.loc[s.str.contains(r"BLADDER\s*(19|26)_M", regex=True, na=False)] = "MY"
    out.loc[s.str.contains(r"BLADDER\s*(19|26)_AR", regex=True, na=False)] = "AR"
    out.loc[s.str.contains(r"BLADDER\s*(19|26)_BT", regex=True, na=False)] = "BT"
    return out


def infer_cohort_from_sample(sample: pd.Series) -> pd.Series:
    s = sample.astype(str).str.upper()
    out = pd.Series("Unknown", index=sample.index, dtype="object")
    out.loc[s.str.contains(r"^BCA\s*2020|^BCA2020|BCA_2020", regex=True, na=False)] = "NAC2020"
    out.loc[s.str.contains(r"^NO[-_\s]?NAC|NO-NAC", regex=True, na=False)] = "No-NAC"
    out.loc[s.str.contains(r"^PURE01|PURE01", regex=True, na=False)] = "PURE01"
    out.loc[s.str.contains(r"^BLADDER\s*19|^BLADDER\s*26|^BLADDER_19|^BLADDER_26", regex=True, na=False)] = "NAC2015"
    return out


def infer_sample_type_from_sample(sample: pd.Series) -> pd.Series:
    s = sample.astype(str).str.upper()
    out = pd.Series("Unknown", index=sample.index, dtype="object")
    out.loc[s.str.contains("TURBT|PRE NAC", regex=True, na=False)] = "TURBT"
    out.loc[s.str.contains(r"\bRC\b|POST NAC|POST", regex=True, na=False)] = "RC"
    both = s.str.contains("PRE", na=False) & s.str.contains("POST", na=False)
    out.loc[both] = "Unknown"
    return out


def infer_tma_from_sample(sample: pd.Series) -> pd.Series:
    s = sample.astype(str).str.upper()
    out = pd.Series(pd.NA, index=sample.index, dtype="object")
    m = s.str.extract(r"TMA\s*([0-9]+)", expand=False)
    out.loc[m.notna()] = m.loc[m.notna()].astype(str)
    out.loc[s.str.contains(r"BLADDER\s*19|BLADDER_19", regex=True, na=False)] = "1"
    out.loc[s.str.contains(r"BLADDER\s*26|BLADDER_26", regex=True, na=False)] = "2"
    return out


def normalize_annotation_string(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    return s


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def discover_parquet_files(root: Path, recursive: bool = False) -> list[Path]:
    files = sorted(root.rglob("*.parquet") if recursive else root.glob("*.parquet"))
    bad_tokens = [
        "tma_cell_df",
        "wholesection_cell_df",
        "marker_df",
        "patient",
        "cache",
    ]
    files = [p for p in files if not any(tok in p.name.lower() for tok in bad_tokens)]
    return files


def load_qc_review(qc_dir: Optional[Path]) -> pd.DataFrame:
    if qc_dir is None:
        return pd.DataFrame(columns=["coord", "structural_acceptability"])

    files = sorted(qc_dir.glob("*review.csv"))
    if not files:
        return pd.DataFrame(columns=["coord", "structural_acceptability"])

    parts = []
    for fp in files:
        try:
            df = safe_read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read QC file {fp}: {e}")
            continue
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
        if "image" not in df.columns:
            continue
        df["coord"] = extract_coord_token_series(df["image"])
        df["__qc_file"] = fp.name
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["coord", "structural_acceptability"])

    qc = pd.concat(parts, ignore_index=True, sort=False)
    qc = qc[qc["coord"].notna()].copy()
    if "structural_acceptability" not in qc.columns:
        qc["structural_acceptability"] = pd.NA

    qc["__has_sa"] = qc["structural_acceptability"].notna().astype(int)
    qc = (
        qc.sort_values(["coord", "__has_sa"], ascending=[True, False])
          .drop_duplicates("coord")
          .drop(columns=["__has_sa"])
    )
    return qc


def load_existing_annotations(existing_dir: Optional[Path], panels: Iterable[str]) -> pd.DataFrame:
    if existing_dir is None:
        return pd.DataFrame(columns=["Panel", "phenotype"] + ANNOTATION_COLS)

    parts = []
    panels = {p.upper() for p in panels}
    for fp in sorted(existing_dir.glob("*_phenotype_abundance_consistency_normalized.csv")):
        panel = fp.name.split("_")[0].upper()
        if panel not in panels:
            continue
        try:
            df = safe_read_csv(fp)
        except Exception as e:
            print(f"[WARN] Could not read existing annotation file {fp}: {e}")
            continue
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
        if "phenotype" not in df.columns:
            continue
        df["Panel"] = panel
        for c in ANNOTATION_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        keep = ["Panel", "phenotype"] + ANNOTATION_COLS
        parts.append(df[keep].copy())

    if not parts:
        return pd.DataFrame(columns=["Panel", "phenotype"] + ANNOTATION_COLS)

    ann = pd.concat(parts, ignore_index=True, sort=False)
    ann["phenotype"] = ann["phenotype"].apply(canonicalize_marker_combo)
    ann = ann.drop_duplicates(subset=["Panel", "phenotype"], keep="first")
    return ann


def load_blasst_metadata(metadata_csv: Optional[Path]) -> pd.DataFrame:
    """Load BLASST metadata and extract coord/sample_type for whole-section audit."""
    if metadata_csv is None:
        return pd.DataFrame(columns=["coord", "coord_key", "sample_type"])

    meta = safe_read_csv(metadata_csv)
    meta = meta.loc[:, ~meta.columns.astype(str).str.match(r"^Unnamed")].copy()

    core_col = None
    for c in ["Core", "core", "region_id", "sample_name", "Sample_ID_Adjusted", "Sample_ID"]:
        if c in meta.columns:
            core_col = c
            break

    if core_col is None:
        print(f"[WARN] BLASST metadata has no recognizable coordinate column: {metadata_csv}")
        return pd.DataFrame(columns=["coord", "coord_key", "sample_type"])

    meta["coord"] = extract_coord_token_series(meta[core_col])
    meta["coord_key"] = meta["coord"].map(coord_to_key)

    sample_type_col = None
    for c in ["TURBT_or_RC", "sample_type", "specimen_type", "tma", "Specimen_Type"]:
        if c in meta.columns:
            sample_type_col = c
            break

    if sample_type_col is None:
        meta["sample_type"] = "Unknown"
    else:
        meta["sample_type"] = meta[sample_type_col]

    meta["sample_type"] = (
        meta["sample_type"]
        .astype(str)
        .str.strip()
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )
    # Canonicalize common values.
    st_upper = meta["sample_type"].str.upper()
    meta.loc[st_upper.str.contains("TURBT|PRE", regex=True, na=False), "sample_type"] = "TURBT"
    meta.loc[st_upper.str.contains(r"\bRC\b|POST", regex=True, na=False), "sample_type"] = "RC"

    out = meta[["coord", "coord_key", "sample_type"]].dropna(subset=["coord_key"]).drop_duplicates("coord_key")
    print(f"[blasst metadata] rows={len(out):,} unique_coords={out['coord_key'].nunique():,}")
    return out


def read_parquet_minimal(fp: Path) -> pd.DataFrame:
    required = ["sample_name", "phenotype_combined"]
    optional = ["core_key", "region_id", "panel", "Panel", "batch"]
    try:
        df0 = pd.read_parquet(fp, columns=required + optional)
    except Exception:
        df0 = pd.read_parquet(fp)
        keep = [c for c in required + optional if c in df0.columns]
        df0 = df0[keep].copy()

    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {fp}")

    return df0


def build_core_counts(
    parquet_dir: Path,
    panels: Iterable[str],
    *,
    recursive: bool = False,
    source: str = "TMA",
    cohort_override: Optional[str] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    panels = {p.upper() for p in panels}
    files = discover_parquet_files(parquet_dir, recursive=recursive)
    if not files:
        raise ValueError(f"No parquet files found under {parquet_dir}")

    all_counts = []
    all_core_totals = []

    print(f"[discover:{source}] parquet files: {len(files)}")

    meta_small = None
    if metadata_df is not None and not metadata_df.empty and "coord_key" in metadata_df.columns:
        meta_small = metadata_df[["coord_key", "sample_type"]].drop_duplicates("coord_key").copy()

    for i, fp in enumerate(files, 1):
        panel_from_path = infer_panel_from_path(fp)
        if panel_from_path not in panels and panel_from_path != "Unknown":
            continue

        print(f"[{source} {i:03d}/{len(files):03d}] {fp.name} | panel_from_path={panel_from_path}")
        df = read_parquet_minimal(fp)

        df["coord"] = extract_coord_token_series(df["sample_name"])
        for c in ["core_key", "region_id"]:
            if c in df.columns:
                coord_from_col = extract_coord_token_series(df[c])
                df["coord"] = df["coord"].where(df["coord"].notna(), coord_from_col)
        df = df[df["coord"].notna()].copy()
        if df.empty:
            continue

        df["coord_key"] = df["coord"].map(coord_to_key)

        # Panel: prefer explicit parquet column, else path, else sample name.
        if "Panel" in df.columns:
            df["Panel"] = df["Panel"].map(normalize_panel_value)
        elif "panel" in df.columns:
            df["Panel"] = df["panel"].map(normalize_panel_value)
        else:
            df["Panel"] = panel_from_path

        unknown_panel = df["Panel"].eq("Unknown")
        if unknown_panel.any():
            df.loc[unknown_panel, "Panel"] = infer_panel_from_sample(df.loc[unknown_panel, "sample_name"])
        df = df[df["Panel"].isin(panels)].copy()
        if df.empty:
            continue

        if cohort_override is not None:
            df["cohort"] = cohort_override
        else:
            df["cohort"] = infer_cohort_from_sample(df["sample_name"])

        df["sample_type"] = infer_sample_type_from_sample(df["sample_name"])
        if meta_small is not None:
            df = df.merge(meta_small.rename(columns={"sample_type": "sample_type_meta"}), on="coord_key", how="left")
            df["sample_type"] = df["sample_type"].where(df["sample_type"].ne("Unknown"), df["sample_type_meta"])
            df["sample_type"] = df["sample_type"].fillna("Unknown")
            df = df.drop(columns=["sample_type_meta"])

        if source.upper() == "BLASST":
            df["tma"] = "WholeSection"
        else:
            df["tma"] = infer_tma_from_sample(df["sample_name"])

        df["source"] = source
        df["phenotype"] = df["phenotype_combined"].apply(canonicalize_marker_combo)

        core_cols = ["source", "Panel", "cohort", "sample_type", "tma", "coord", "sample_name"]

        core_totals = (
            df.groupby(core_cols, dropna=False)
              .size()
              .reset_index(name="core_total")
        )

        counts = (
            df.groupby(core_cols + ["phenotype"], dropna=False)
              .size()
              .reset_index(name="n_cells")
        )

        all_core_totals.append(core_totals)
        all_counts.append(counts)

        print(
            f"    rows={len(df):,} cores={core_totals['sample_name'].nunique():,} "
            f"phenotypes={counts['phenotype'].nunique():,}"
        )

    if not all_counts:
        raise ValueError(f"No valid cell counts were built for {source}. Check parquet paths/panels.")

    counts = pd.concat(all_counts, ignore_index=True, sort=False)
    core_totals = pd.concat(all_core_totals, ignore_index=True, sort=False)

    core_cols = ["source", "Panel", "cohort", "sample_type", "tma", "coord", "sample_name"]
    counts = (
        counts.groupby(core_cols + ["phenotype"], dropna=False)["n_cells"]
              .sum()
              .reset_index()
    )
    core_totals = (
        core_totals.groupby(core_cols, dropna=False)["core_total"]
                   .sum()
                   .reset_index()
    )

    counts = counts.merge(core_totals, on=core_cols, how="left")
    counts["frac"] = counts["n_cells"] / counts["core_total"]
    return counts


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def add_zero_rows_for_absent_phenotypes(counts: pd.DataFrame, phenotypes: list[str]) -> pd.DataFrame:
    core_cols = ["source", "Panel", "cohort", "sample_type", "tma", "coord", "sample_name", "core_total"]
    cores = counts[core_cols].drop_duplicates().copy()

    ph = pd.DataFrame({"phenotype": phenotypes})
    cores["__key"] = 1
    ph["__key"] = 1
    grid = cores.merge(ph, on="__key", how="outer").drop(columns="__key")

    observed = counts[core_cols + ["phenotype", "n_cells", "frac"]].copy()
    out = grid.merge(observed, on=core_cols + ["phenotype"], how="left")
    out["n_cells"] = out["n_cells"].fillna(0).astype(int)
    out["frac"] = out["frac"].fillna(0.0)
    return out


def summarize_panel(
    counts: pd.DataFrame,
    panel: str,
    curation_cohorts: list[str],
    qc_df: pd.DataFrame,
    frac_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_counts = counts[
        counts["Panel"].eq(panel) &
        counts["cohort"].isin(curation_cohorts)
    ].copy()

    if panel_counts.empty:
        return pd.DataFrame(), pd.DataFrame()

    phenotypes = sorted(panel_counts["phenotype"].dropna().astype(str).unique().tolist())
    frac_df = add_zero_rows_for_absent_phenotypes(panel_counts, phenotypes)

    if qc_df is not None and not qc_df.empty and "coord" in qc_df.columns:
        q = qc_df[["coord", "structural_acceptability"]].drop_duplicates("coord").copy()
        frac_df = frac_df.merge(q, on="coord", how="left")
    else:
        frac_df["structural_acceptability"] = pd.NA

    total_cores = (
        frac_df[["source", "Panel", "cohort", "sample_type", "tma", "coord", "sample_name"]]
        .drop_duplicates()
        .shape[0]
    )

    core_totals_unique = frac_df[["cohort", "sample_type", "sample_name", "core_total"]].drop_duplicates()
    total_cells_bg = core_totals_unique["core_total"].sum()
    cohort_bg = core_totals_unique.groupby("cohort")["core_total"].sum() / total_cells_bg
    st_bg = core_totals_unique.groupby("sample_type")["core_total"].sum() / total_cells_bg

    rows = []
    for phenotype, sub in frac_df.groupby("phenotype", dropna=False):
        total_cells = int(sub["n_cells"].sum())
        n_cores_present = int((sub["n_cells"] > 0).sum())
        n_cores_frac_ge = int((sub["frac"] >= frac_threshold).sum()) if frac_threshold > 0 else n_cores_present

        row = {
            "phenotype": phenotype,
            "total_cells": total_cells,
            "median_frac": float(sub["frac"].median()),
            "mean_frac": float(sub["frac"].mean()),
            "n_cores_present": n_cores_present,
            "n_cores_frac_ge": n_cores_frac_ge,
            "n_total_cores": int(total_cores),
            "core_presence_frac": float(n_cores_present / total_cores) if total_cores else np.nan,
        }

        for cohort in curation_cohorts:
            n = int(sub.loc[sub["cohort"].eq(cohort), "n_cells"].sum())
            contrib = n / total_cells if total_cells else 0.0
            bg = float(cohort_bg.get(cohort, np.nan))
            row[f"{cohort}_contribution"] = contrib
            row[f"{cohort}_enrichment"] = contrib / bg if bg and not np.isnan(bg) else np.nan

        for st in ["TURBT", "RC"]:
            n = int(sub.loc[sub["sample_type"].eq(st), "n_cells"].sum())
            contrib = n / total_cells if total_cells else 0.0
            bg = float(st_bg.get(st, np.nan))
            row[f"{st}_contribution"] = contrib
            row[f"{st}_enrichment"] = contrib / bg if bg and not np.isnan(bg) else np.nan

        present = sub[sub["n_cells"] > 0].copy()
        if present.empty or "structural_acceptability" not in present.columns:
            row["pct_acceptable_cores"] = np.nan
            row["pct_borderline_cores"] = np.nan
        else:
            sa = present["structural_acceptability"].astype(str).str.strip().str.lower()
            denom = len(present)
            row["pct_acceptable_cores"] = float((sa == "acceptable").sum() / denom) if denom else np.nan
            row["pct_borderline_cores"] = float((sa == "borderline").sum() / denom) if denom else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("total_cells", ascending=False).reset_index(drop=True)
    return summary, frac_df


def summarize_holdout(
    counts: pd.DataFrame,
    panel: str,
    holdout_cohorts: list[str],
) -> pd.DataFrame:
    hold = counts[
        counts["Panel"].eq(panel) &
        counts["cohort"].isin(holdout_cohorts)
    ].copy()
    if hold.empty:
        return pd.DataFrame()

    rows = []
    for phenotype, sub in hold.groupby("phenotype", dropna=False):
        total_cells = int(sub["n_cells"].sum())
        total_cores = int(sub.loc[sub["n_cells"] > 0, "sample_name"].nunique())

        row = {
            "Panel": panel,
            "phenotype": phenotype,
            "holdout_total_cells": total_cells,
            "holdout_n_cores_present": total_cores,
        }

        for cohort in holdout_cohorts:
            csub = sub[sub["cohort"].eq(cohort)].copy()
            c_cells = int(csub["n_cells"].sum())
            c_cores = int(csub.loc[csub["n_cells"] > 0, "sample_name"].nunique())
            row[f"{cohort}_total_cells"] = c_cells
            row[f"{cohort}_n_cores_present"] = c_cores
            row[f"{cohort}_cell_fraction_of_holdout"] = c_cells / total_cells if total_cells else 0.0

            for st in ["TURBT", "RC", "Unknown"]:
                st_cells = int(csub.loc[csub["sample_type"].eq(st), "n_cells"].sum())
                row[f"{cohort}_{st}_total_cells"] = st_cells

        rows.append(row)

    return pd.DataFrame(rows).sort_values("holdout_total_cells", ascending=False).reset_index(drop=True)


def merge_annotations(summary: pd.DataFrame, ann: pd.DataFrame, panel: str) -> pd.DataFrame:
    out = summary.copy()
    for c in ANNOTATION_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    if ann is not None and not ann.empty:
        sub_ann = ann[ann["Panel"].eq(panel)].copy()
        if not sub_ann.empty:
            out = out.drop(columns=ANNOTATION_COLS, errors="ignore")
            out = out.merge(sub_ann[["phenotype"] + ANNOTATION_COLS], on="phenotype", how="left")

    for c in ANNOTATION_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    for c in BOOL_ANNOTATION_COLS:
        out[c] = out[c].map(parse_bool)

    out["assigned_label"] = out["assigned_label"].map(normalize_annotation_string)
    out["state"] = out["state"].map(normalize_annotation_string).fillna("None")
    out["lineage"] = out["lineage"].map(normalize_annotation_string)
    out["collapse_label"] = out["collapse_label"].map(normalize_annotation_string)
    out["artifact_reason"] = out["artifact_reason"].map(normalize_annotation_string).fillna("None")
    out["review_notes"] = out["review_notes"].map(normalize_annotation_string)

    metric_cols = [c for c in out.columns if c not in ANNOTATION_COLS]
    out = out[metric_cols + ANNOTATION_COLS]
    return out


def write_readme(out_dir: Path, args, curation_cohorts: list[str], holdout_cohorts: list[str]) -> None:
    txt = f"""# Phenotype abundance/consistency rebuild

Generated: {datetime.now().isoformat(timespec='seconds')}

## Purpose

This folder contains regenerated phenotype abundance/consistency tables derived
from combined cell-level parquet files.

## Input parquet directories

TMA: `{args.tma_parquet_dir}`

Whole section / BLASST: `{args.whole_parquet_dir if args.whole_parquet_dir else 'not supplied'}`

## Curation cohorts

{', '.join(curation_cohorts)}

These cohorts are used to compute the primary abundance, contribution,
enrichment, and core-presence statistics.

## Holdout cohorts

{', '.join(holdout_cohorts) if holdout_cohorts else 'None'}

Holdout cohorts are not used to define the primary phenotype statistics. They
are summarized separately for transferability/audit purposes. If BLASST whole-
section parquets were supplied, BLASST is included here as a specimen-format
transferability audit.

## Phenotype canonicalization

Marker combinations are canonicalized by splitting on semicolon, slash, or pipe,
keeping positive marker calls ending in `+`, dropping negative marker calls
ending in `-`, sorting positive markers, and assigning `ALL_NEG` when no
positive marker remains.

Examples:

- `CD68+;PD1-;PanCK+` -> `CD68+;PanCK+`
- `CD68+/PD1-` -> `CD68+`
- `PD1-;PDL1-` -> `ALL_NEG`

## Main output files

- `<PANEL>_phenotype_abundance_consistency_normalized.csv`: panel-specific
  review-ready phenotype table based only on curation cohorts.
- `<PANEL>_phenotype_abundance_consistency_core_counts.csv`: explicit per-core
  phenotype counts and fractions for curation cohorts, including zero rows for
  absent phenotypes.
- `<PANEL>_phenotype_abundance_consistency_holdout_audit.csv`: summary of
  holdout-cohort phenotype counts, including NAC2015 and BLASST when present.
- `all_panels_core_counts.csv`: observed per-core phenotype counts across all
  loaded panels/cohorts/sources.
- `run_metadata.json`: run parameters and timestamp.
"""
    (out_dir / "README_phenotype_abundance_rebuild.md").write_text(txt)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate phenotype abundance/consistency CSVs from combined parquet files.")
    ap.add_argument("--tma_parquet_dir", required=True, help="Directory containing combined TMA cell-level parquet files.")
    ap.add_argument("--whole_parquet_dir", default="", help="Optional: directory containing BLASST/whole-section parquet files.")
    ap.add_argument("--blasst_metadata_csv", default="", help="Optional: BLASST metadata CSV to assign TURBT/RC sample type by coordinate.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--existing_annotation_dir", default="", help="Optional: carry forward manual annotation columns from existing normalized CSVs.")
    ap.add_argument("--qc_dir", default="", help="Optional: directory containing *review.csv files for acceptable/borderline percentages.")
    ap.add_argument("--panels", nargs="+", default=["AR", "BT", "MY"], help="Panels to process.")
    ap.add_argument("--curation_cohorts", nargs="+", default=["NAC2020", "No-NAC", "PURE01"], help="Cohorts used to compute primary curation metrics.")
    ap.add_argument("--holdout_cohorts", nargs="+", default=["NAC2015", "BLASST"], help="Holdout cohorts for separate audit only.")
    ap.add_argument("--frac_threshold", type=float, default=0.0, help="Threshold for n_cores_frac_ge. Default 0 means same as n_cores_present.")
    ap.add_argument("--recursive", action="store_true", help="Search TMA parquet files recursively instead of only top-level.")
    ap.add_argument("--whole_recursive", action="store_true", default=True, help="Search whole-section parquet files recursively. Default True.")
    args = ap.parse_args()

    tma_parquet_dir = Path(args.tma_parquet_dir)
    whole_parquet_dir = Path(args.whole_parquet_dir) if args.whole_parquet_dir else None
    blasst_metadata_csv = Path(args.blasst_metadata_csv) if args.blasst_metadata_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [p.upper() for p in args.panels]
    curation_cohorts = list(args.curation_cohorts)
    holdout_cohorts = list(args.holdout_cohorts)

    existing_dir = Path(args.existing_annotation_dir) if args.existing_annotation_dir else None
    qc_dir = Path(args.qc_dir) if args.qc_dir else None

    print("=" * 80)
    print("Phenotype abundance/consistency rebuild v2")
    print("=" * 80)
    print("tma_parquet_dir:   ", tma_parquet_dir)
    print("whole_parquet_dir: ", whole_parquet_dir)
    print("blasst_metadata:   ", blasst_metadata_csv)
    print("out_dir:           ", out_dir)
    print("panels:            ", panels)
    print("curation:          ", curation_cohorts)
    print("holdout:           ", holdout_cohorts)
    print("existing_ann:      ", existing_dir)
    print("qc_dir:            ", qc_dir)

    qc_df = load_qc_review(qc_dir)
    print(f"[qc] rows={len(qc_df):,} unique_coords={qc_df['coord'].nunique() if 'coord' in qc_df.columns else 0:,}")

    ann = load_existing_annotations(existing_dir, panels=panels)
    print(f"[annotations] rows={len(ann):,}")

    counts_parts = []
    counts_tma = build_core_counts(tma_parquet_dir, panels=panels, recursive=args.recursive, source="TMA")
    counts_parts.append(counts_tma)

    blasst_meta = pd.DataFrame()
    if whole_parquet_dir is not None:
        blasst_meta = load_blasst_metadata(blasst_metadata_csv)
        counts_blasst = build_core_counts(
            whole_parquet_dir,
            panels=panels,
            recursive=True,
            source="BLASST",
            cohort_override="BLASST",
            metadata_df=blasst_meta,
        )
        counts_parts.append(counts_blasst)

    counts = pd.concat(counts_parts, ignore_index=True, sort=False)
    counts.to_csv(out_dir / "all_panels_core_counts.csv", index=False)
    print(f"[write] all_panels_core_counts.csv rows={len(counts):,}")

    for panel in panels:
        print("\n" + "-" * 80)
        print(f"Panel: {panel}")
        print("-" * 80)

        summary, frac_df = summarize_panel(
            counts,
            panel=panel,
            curation_cohorts=curation_cohorts,
            qc_df=qc_df,
            frac_threshold=args.frac_threshold,
        )

        if summary.empty:
            print(f"[WARN] no curation rows for panel {panel}")
            continue

        summary = merge_annotations(summary, ann=ann, panel=panel)

        summary_path = out_dir / f"{panel}_phenotype_abundance_consistency_normalized.csv"
        core_path = out_dir / f"{panel}_phenotype_abundance_consistency_core_counts.csv"
        hold_path = out_dir / f"{panel}_phenotype_abundance_consistency_holdout_audit.csv"

        summary.to_csv(summary_path, index=False)
        frac_df.to_csv(core_path, index=False)

        print(f"[write] {summary_path.name} rows={len(summary):,}")
        print(f"[write] {core_path.name} rows={len(frac_df):,}")

        hold = summarize_holdout(counts, panel=panel, holdout_cohorts=holdout_cohorts)
        if not hold.empty:
            hold.to_csv(hold_path, index=False)
            print(f"[write] {hold_path.name} rows={len(hold):,}")
        else:
            print(f"[WARN] no holdout rows for panel {panel}")

        print("Top phenotypes:")
        print(summary[["phenotype", "total_cells", "core_presence_frac"]].head(10).to_string(index=False))

    metadata = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tma_parquet_dir": str(tma_parquet_dir),
        "whole_parquet_dir": str(whole_parquet_dir) if whole_parquet_dir else None,
        "blasst_metadata_csv": str(blasst_metadata_csv) if blasst_metadata_csv else None,
        "out_dir": str(out_dir),
        "existing_annotation_dir": str(existing_dir) if existing_dir else None,
        "qc_dir": str(qc_dir) if qc_dir else None,
        "panels": panels,
        "curation_cohorts": curation_cohorts,
        "holdout_cohorts": holdout_cohorts,
        "frac_threshold": args.frac_threshold,
        "recursive": bool(args.recursive),
        "whole_recursive": True,
        "canonicalization": "split on ; / |, keep tokens ending '+', drop tokens ending '-', sort positives, ALL_NEG if none",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_readme(out_dir, args, curation_cohorts, holdout_cohorts)

    print("\nDONE")
    print("Output directory:", out_dir)


if __name__ == "__main__":
    main()
