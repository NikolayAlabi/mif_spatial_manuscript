#!/usr/bin/env python3
"""
Build reviewed TMA and whole-section cell-level analysis parquet files from raw combined mIF parquet files.

Purpose
-------
This script rebuilds manuscript-ready cell-level parquet tables after manual
phenotype review. It uses the reviewed phenotype abundance/consistency CSVs as
the authoritative marker-combination dictionary, maps raw combined cell parquet
files to final `collapse_label` values, filters artifact/exploratory labels for
primary analyses, and writes separate TMA and BLASST/whole-section cell tables.

Inputs
------
1. Raw combined TMA cell parquet files produced from inForm cell_seg_data files.
2. Raw combined BLASST / whole-section cell parquet files.
3. Manually reviewed phenotype assignment CSVs:
       AR_phenotype_abundance_consistency_normalized.csv
       BT_phenotype_abundance_consistency_normalized.csv
       MY_phenotype_abundance_consistency_normalized.csv

Expected phenotype assignment columns
-------------------------------------
The script primarily uses:
    phenotype       : marker-combination key
    collapse_label  : final reviewed phenotype label
    state           : AR checkpoint state, e.g. PD1, PDL1, PD1_PDL1, None
    artifact_flag   : if TRUE, excluded from primary feature tables
    exploratory     : if TRUE, excluded from primary feature tables by default

Derived label columns written to the output parquets
----------------------------------------------------
The final cell_df parquet also contains reusable label views for downstream
feature generation:

    label_phenotype          : fine phenotype-only label; currently collapse_label
    label_ar_state           : AR-only collapse_label__state label; non-AR rows are NA
    label_checkpoint_state   : AR-only checkpoint state label: PD1, PDL1, PD1_PDL1, checkpoint_neg
    label_checkpoint_binary  : AR-only checkpoint_pos/checkpoint_neg
    label_compartment        : coarse tumor/immune/stroma label
    label_compartment_state  : AR-only compartment__state label

These columns allow the same prep/feature scripts to create multiple feature
sources without duplicating raw cell parquets.

Other annotation columns are carried into map/audit outputs when present.

Canonicalization
----------------
Marker combinations are canonicalized by dropping negative marker calls and
sorting positive marker calls. This means:
    CD68+;PD1-       -> CD68+
    CD68+/PD1-       -> CD68+
    PD1-;PDL1-       -> ALL_NEG
    PDL1+;PanCK+     -> PDL1+;PanCK+

Primary outputs
---------------
    tma_cell_df.parquet
    wholesection_cell_df.parquet
        Standard primary reviewed cell tables using collapse_label.

    Optional, if --write-ar-state-parquets is used:
    tma_cell_df_AR_state.parquet
    wholesection_cell_df_AR_state.parquet
        AR-only tables with the same columns, but with collapse_label replaced
        by phenotype+state labels such as macrophage__PDL1 or t_cell__PD1.
        These are not required if using prep_weibull_inputs_v3.py with
        --label-mode ar_state.

By default, artifact_flag==TRUE and exploratory==TRUE rows are excluded from
primary outputs, but retained in summary diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

COORD_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")

DERIVED_LABEL_COLS = [
    "label_phenotype",
    "label_ar_state",
    "label_checkpoint_state",
    "label_checkpoint_binary",
    "label_compartment",
    "label_compartment_state",
]

FINAL_COLS = [
    "sample_name",
    "coord",
    "x",
    "y",
    "tissue_region",
    "marker_combination",
    "phenotype",
    "Panel",
    "cohort",
    "phenotype_canonical",
    "collapse_label",
    "state",
    *DERIVED_LABEL_COLS,
]

ANNOTATION_OPTIONAL_COLS = [
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
    "total_cells",
    "median_frac",
    "mean_frac",
    "n_cores_present",
    "core_presence_frac",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build reviewed TMA and whole-section cell_df parquet files."
    )
    ap.add_argument("--tma_parquet_dir", required=True)
    ap.add_argument("--whole_parquet_dir", required=True)
    ap.add_argument("--phenotype_assignments_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--panels", nargs="+", default=["AR", "BT", "MY"])

    ap.add_argument(
        "--exclude-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows mapping to artifact_flag==TRUE from primary outputs.",
    )
    ap.add_argument(
        "--exclude-exploratory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows mapping to exploratory==TRUE from primary outputs.",
    )
    ap.add_argument(
        "--exclude-collapse-labels",
        nargs="+",
        default=["artifact", "unresolved", "mixed_lineage"],
        help="collapse_label values to exclude from primary outputs.",
    )
    ap.add_argument(
        "--state-none-label",
        default="checkpoint_neg",
        help="State suffix used for AR cells where state is missing/None.",
    )
    ap.add_argument(
        "--state-separator",
        default="__",
        help="Separator between collapse_label and state in AR-state outputs.",
    )
    ap.add_argument(
        "--state-label-all-neg-mode",
        choices=["keep_all_neg", "append"],
        default="append",
        help=(
            "How to label ALL_NEG cells in optional AR-state output. "
            "Default is append so STROMA/ALL_NEG-like cells can become "
            "STROMA__checkpoint_neg, STROMA__PD1, etc."
        ),
    )
    ap.add_argument(
        "--write-ar-state-parquets",
        action="store_true",
        help=(
            "Optionally write AR-only duplicate parquets where collapse_label is replaced "
            "by collapse_label__state. Usually unnecessary if using prep_weibull_inputs_v3.py."
        ),
    )
    ap.add_argument(
        "--write-audit-parquets",
        action="store_true",
        help="Also write all mapped cells with audit columns before primary filtering. Large output.",
    )
    ap.add_argument(
        "--parquet-discovery-mode",
        choices=["auto", "recursive_all", "top_level_only"],
        default="auto",
        help="How to discover raw parquet inputs.",
    )
    return ap.parse_args()


def safe_read_csv(fp: str | Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err


def parse_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_none_string(x) -> str:
    if pd.isna(x):
        return "None"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "<na>", "null"}:
        return "None"
    return s


def extract_coord_token(series: pd.Series) -> pd.Series:
    m = series.astype(str).str.extract(COORD_RE)
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    ok = m[0].notna() & m[1].notna()
    out.loc[ok] = "[" + m.loc[ok, 0].astype(str) + "," + m.loc[ok, 1].astype(str) + "]"
    return out


def canonicalize_marker_combo(x) -> str:
    """Drop negative marker calls and sort positive marker calls."""
    if pd.isna(x):
        return "ALL_NEG"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return "ALL_NEG"

    # Historical files are mostly semicolon-delimited, but some cells use slash.
    raw_parts = re.split(r"[;/]", s)
    parts = []
    for p in raw_parts:
        p = p.strip()
        if not p:
            continue
        # normalize common lower-case marker token generated by inForm artifacts
        if p.lower() in {"marker-", "marker+"}:
            # marker+ is not a biological marker; marker- is negative.
            continue
        # Drop negative calls. Examples: PD1-, PDL1-, PanCK-
        if p.endswith("-"):
            continue
        parts.append(p)

    parts = sorted(set(parts))
    return ";".join(parts) if parts else "ALL_NEG"


def infer_panel_from_path(fp: Path) -> str:
    s = str(fp).upper()
    b = fp.name.upper()
    if "ARP" in s or re.search(r"(^|[^A-Z0-9])AR($|[^A-Z0-9])", s) or re.search(r"(^|_)AR(\.|_|$)", b):
        return "AR"
    if "B&T" in s or "B+T" in s or re.search(r"(^|[^A-Z0-9])BT($|[^A-Z0-9])", s) or re.search(r"(^|_)BT(\.|_|$)", b):
        return "BT"
    if "MYELOID" in s or re.search(r"(^|_)M(\.|_|$)", b) or re.search(r"(_M($|[^A-Z0-9]))", s):
        return "MY"
    return "Unknown"


def infer_cohort_from_sample(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper()
    raw = s.str.extract(r"^(BCA(?:_| )?2020|NO-NAC|PURE01|BLADDER|BLASST)\b", expand=False)
    raw = raw.str.replace("BCA_2020", "BCA2020", regex=False)
    raw = raw.str.replace("BCA 2020", "BCA2020", regex=False)
    out = raw.map({
        "BCA2020": "NAC2020",
        "NO-NAC": "No-NAC",
        "PURE01": "PURE01",
        "BLADDER": "NAC2015",
        "BLASST": "BLASST",
    }).fillna(raw)
    return out


def infer_cohort_from_path_or_sample(fp: Path, sample_series: pd.Series, dataset_kind: str) -> pd.Series:
    if dataset_kind == "whole":
        return pd.Series("BLASST", index=sample_series.index, dtype="object")
    return infer_cohort_from_sample(sample_series)


def discover_parquets(root: str | Path, mode: str = "auto") -> list[Path]:
    root = Path(root)
    if mode == "top_level_only":
        files = sorted(root.glob("*.parquet"))
    elif mode == "recursive_all":
        files = sorted(root.rglob("*.parquet"))
    else:
        all_files = sorted(root.rglob("*.parquet"))
        top = sorted(root.glob("*.parquet"))
        # Prefer finalized top-level parquet files when present to avoid duplicating part files.
        files = top if top else all_files

    # Avoid accidentally reading outputs from prior rebuilds if they are under the same tree.
    banned_stems = {
        "tma_cell_df",
        "wholesection_cell_df",
        "tma_cell_df_ar_state",
        "wholesection_cell_df_ar_state",
    }
    out = [p for p in files if p.stem.lower() not in banned_stems]
    return out


def load_phenotype_map(phenotype_assignments_dir: str | Path, panels: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(phenotype_assignments_dir)
    panels = {str(p).upper() for p in panels}
    files = sorted(root.glob("*_phenotype_abundance_consistency_normalized.csv"))
    if not files:
        raise ValueError(f"No *_phenotype_abundance_consistency_normalized.csv files found under {root}")

    parts = []
    for fp in files:
        panel = fp.name.split("_")[0].upper()
        if panel not in panels:
            continue
        df = safe_read_csv(fp)
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
        df["Panel"] = panel
        df["source_file"] = fp.name

        # The marker-combination key should normally be 'phenotype'. Fallbacks are for safety only.
        key_col = None
        for cand in ["phenotype", "assigned_label", "marker_combination", "phenotype_combined"]:
            if cand in df.columns:
                key_col = cand
                break
        if key_col is None:
            raise ValueError(f"{fp} has no phenotype-like key column.")
        if key_col != "phenotype":
            df["phenotype"] = df[key_col]

        for c in ANNOTATION_OPTIONAL_COLS:
            if c not in df.columns:
                df[c] = pd.NA

        for c in ["artifact_flag", "simple_spatial", "state_spatial", "exploratory"]:
            df[c] = df[c].map(parse_bool)

        df["phenotype_canonical"] = df["phenotype"].apply(canonicalize_marker_combo)
        parts.append(df)

    if not parts:
        raise ValueError(f"No normalized phenotype files loaded for panels={sorted(panels)}")

    raw_map = pd.concat(parts, ignore_index=True, sort=False)

    # Sort so that non-artifact/non-exploratory, higher-abundance definitions win if duplicate canonical keys exist.
    if "total_cells" in raw_map.columns:
        raw_map["__total_cells_num"] = pd.to_numeric(raw_map["total_cells"], errors="coerce").fillna(0)
    else:
        raw_map["__total_cells_num"] = 0

    raw_map["__artifact_sort"] = raw_map["artifact_flag"].astype(bool).astype(int)
    raw_map["__exploratory_sort"] = raw_map["exploratory"].astype(bool).astype(int)

    raw_map = raw_map.sort_values(
        ["Panel", "phenotype_canonical", "__artifact_sort", "__exploratory_sort", "__total_cells_num"],
        ascending=[True, True, True, True, False],
        kind="stable",
    )

    # Duplicate/conflict report before deduplication.
    conflict_cols = ["collapse_label", "state", "artifact_flag", "exploratory", "lineage"]
    dup_groups = raw_map[raw_map.duplicated(["Panel", "phenotype_canonical"], keep=False)].copy()
    conflict_rows = []
    if not dup_groups.empty:
        for (panel, canon), sub in dup_groups.groupby(["Panel", "phenotype_canonical"], dropna=False):
            row = {
                "Panel": panel,
                "phenotype_canonical": canon,
                "n_rows": len(sub),
                "phenotype_examples": "; ".join(sub["phenotype"].astype(str).head(10).tolist()),
            }
            for c in conflict_cols:
                if c in sub.columns:
                    row[f"n_unique_{c}"] = sub[c].astype(str).nunique(dropna=False)
                    row[f"values_{c}"] = "; ".join(sorted(sub[c].astype(str).unique().tolist()))
            conflict_rows.append(row)
    conflicts = pd.DataFrame(conflict_rows)

    keep_cols = [
        "Panel",
        "phenotype_canonical",
        "phenotype",
        "assigned_label",
        "collapse_label",
        "state",
        "lineage",
        "artifact_flag",
        "simple_spatial",
        "state_spatial",
        "exploratory",
        "artifact_reason",
        "review_notes",
        "source_file",
    ]
    keep_cols = [c for c in keep_cols if c in raw_map.columns]
    canonical_map = raw_map.drop_duplicates(["Panel", "phenotype_canonical"], keep="first")[keep_cols].copy()

    # Clean state/collapse strings.
    canonical_map["state"] = canonical_map["state"].map(normalize_none_string)
    canonical_map["collapse_label"] = canonical_map["collapse_label"].where(
        canonical_map["collapse_label"].notna(), pd.NA
    )

    # Guarantee ALL_NEG maps to ALL_NEG when no explicit manual map exists.
    existing_all_neg = set(canonical_map.loc[canonical_map["phenotype_canonical"].eq("ALL_NEG"), "Panel"].dropna())
    add_rows = []
    for panel in panels:
        if panel not in existing_all_neg:
            add_rows.append({
                "Panel": panel,
                "phenotype_canonical": "ALL_NEG",
                "phenotype": "ALL_NEG",
                "assigned_label": "ALL_NEG",
                "collapse_label": "ALL_NEG",
                "state": "None",
                "lineage": "ALL_NEG",
                "artifact_flag": False,
                "simple_spatial": True,
                "state_spatial": False,
                "exploratory": False,
                "artifact_reason": "",
                "review_notes": "auto-added all-negative phenotype",
                "source_file": "auto",
            })
    if add_rows:
        canonical_map = pd.concat([canonical_map, pd.DataFrame(add_rows)], ignore_index=True, sort=False)

    return raw_map.drop(columns=[c for c in raw_map.columns if c.startswith("__")], errors="ignore"), canonical_map, conflicts


def standardize_raw_cell_df(df: pd.DataFrame, fp: Path, dataset_kind: str) -> pd.DataFrame:
    out = df.copy()

    rename = {}
    if "phenotype_combined" in out.columns:
        rename["phenotype_combined"] = "marker_combination"
    if "tissue_category" in out.columns:
        rename["tissue_category"] = "tissue_region"
    if "Cell X Position" in out.columns:
        rename["Cell X Position"] = "x"
    if "Cell Y Position" in out.columns:
        rename["Cell Y Position"] = "y"
    if "panel" in out.columns and "Panel" not in out.columns:
        rename["panel"] = "Panel"
    out = out.rename(columns=rename)

    required = ["sample_name", "marker_combination", "x", "y"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns after standardization: {missing}")

    if "tissue_region" not in out.columns:
        out["tissue_region"] = pd.NA

    if "coord" not in out.columns:
        coord = extract_coord_token(out["sample_name"].astype(str))
        if coord.notna().sum() == 0 and "region_id" in out.columns:
            coord = extract_coord_token(out["region_id"].astype(str))
        if coord.notna().sum() == 0 and "core_key" in out.columns:
            coord = extract_coord_token(out["core_key"].astype(str))
        out["coord"] = coord
    else:
        out["coord"] = extract_coord_token(out["coord"].astype(str))

    out = out[out["coord"].notna()].copy()
    if out.empty:
        return out

    out["sample_name"] = out["sample_name"].astype(str)
    out["marker_combination"] = out["marker_combination"].astype(str)
    out["phenotype"] = out["marker_combination"].astype(str)
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["x", "y"])

    if "Panel" not in out.columns:
        out["Panel"] = infer_panel_from_path(fp)
    else:
        out["Panel"] = out["Panel"].astype(str).replace({"B&T": "BT", "B+T": "BT", "Myeloid": "MY"})
        bad = out["Panel"].isna() | out["Panel"].astype(str).str.lower().isin(["nan", "none", "", "unknown", "unk"])
        if bad.any():
            out.loc[bad, "Panel"] = infer_panel_from_path(fp)

    out["Panel"] = out["Panel"].astype(str).replace({"ARP": "AR", "B&T": "BT", "B+T": "BT", "Myeloid": "MY"})

    if "cohort" not in out.columns:
        out["cohort"] = infer_cohort_from_path_or_sample(fp, out["sample_name"], dataset_kind)
    else:
        out["cohort"] = out["cohort"].fillna(infer_cohort_from_path_or_sample(fp, out["sample_name"], dataset_kind))

    if dataset_kind == "whole":
        out["cohort"] = "BLASST"

    out["phenotype_canonical"] = out["marker_combination"].apply(canonicalize_marker_combo)

    return out


def build_cell_df_from_parquets(
    parquet_dir: str | Path,
    canonical_map: pd.DataFrame,
    *,
    dataset_kind: str,
    panels: Iterable[str],
    discovery_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels = {p.upper() for p in panels}
    files = discover_parquets(parquet_dir, mode=discovery_mode)
    if not files:
        raise ValueError(f"No parquet files found under {parquet_dir}")

    parts = []
    inventory_rows = []

    map_cols = [
        "Panel",
        "phenotype_canonical",
        "collapse_label",
        "state",
        "lineage",
        "artifact_flag",
        "exploratory",
        "assigned_label",
        "artifact_reason",
        "review_notes",
    ]
    map_cols = [c for c in map_cols if c in canonical_map.columns]
    cmap = canonical_map[map_cols].drop_duplicates(["Panel", "phenotype_canonical"]).copy()

    for fp in files:
        print(f"[READ:{dataset_kind}] {fp}")
        raw = pd.read_parquet(fp)
        std = standardize_raw_cell_df(raw, fp, dataset_kind=dataset_kind)
        if std.empty:
            inventory_rows.append({"file": str(fp), "rows_raw": len(raw), "rows_kept": 0, "status": "empty_after_standardization"})
            continue

        std = std[std["Panel"].astype(str).str.upper().isin(panels)].copy()
        if std.empty:
            inventory_rows.append({"file": str(fp), "rows_raw": len(raw), "rows_kept": 0, "status": "no_requested_panels"})
            continue

        merged = std.merge(cmap, on=["Panel", "phenotype_canonical"], how="left")

        # Force ALL_NEG to map when needed.
        allneg = merged["phenotype_canonical"].eq("ALL_NEG")
        merged.loc[allneg & merged["collapse_label"].isna(), "collapse_label"] = "ALL_NEG"
        merged.loc[allneg & merged["state"].isna(), "state"] = "None"
        merged.loc[allneg & merged["artifact_flag"].isna(), "artifact_flag"] = False
        merged.loc[allneg & merged["exploratory"].isna(), "exploratory"] = False

        # Ensure expected audit columns exist.
        for c in ["artifact_flag", "exploratory"]:
            if c not in merged.columns:
                merged[c] = False
            merged[c] = merged[c].map(parse_bool)
        if "state" not in merged.columns:
            merged["state"] = "None"
        merged["state"] = merged["state"].map(normalize_none_string)

        parts.append(merged)

        inventory_rows.append({
            "file": str(fp),
            "rows_raw": len(raw),
            "rows_standardized": len(std),
            "rows_mapped": int(merged["collapse_label"].notna().sum()),
            "rows_unmapped": int(merged["collapse_label"].isna().sum()),
            "panels": ";".join(sorted(merged["Panel"].dropna().astype(str).unique())),
            "cohorts": ";".join(sorted(merged["cohort"].dropna().astype(str).unique())),
            "status": "ok",
        })

    if not parts:
        raise ValueError(f"No usable cell rows built from {parquet_dir}")

    cell = pd.concat(parts, ignore_index=True, sort=False)
    inv = pd.DataFrame(inventory_rows)
    return cell, inv


def apply_primary_filter(
    cell: pd.DataFrame,
    *,
    exclude_artifacts: bool,
    exclude_exploratory: bool,
    exclude_collapse_labels: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cell.copy()

    exclude_labels = {str(x).strip().lower() for x in exclude_collapse_labels}
    df["exclude_reason"] = pd.NA

    if exclude_artifacts and "artifact_flag" in df.columns:
        m = df["artifact_flag"].map(parse_bool)
        df.loc[m, "exclude_reason"] = "artifact_flag"

    if exclude_exploratory and "exploratory" in df.columns:
        m = df["exclude_reason"].isna() & df["exploratory"].map(parse_bool)
        df.loc[m, "exclude_reason"] = "exploratory"

    m = df["exclude_reason"].isna() & df["collapse_label"].isna()
    df.loc[m, "exclude_reason"] = "unmapped_collapse_label"

    if exclude_labels:
        cl = df["collapse_label"].astype(str).str.strip().str.lower()
        m = df["exclude_reason"].isna() & cl.isin(exclude_labels)
        df.loc[m, "exclude_reason"] = "excluded_collapse_label"

    excluded = df[df["exclude_reason"].notna()].copy()
    primary = df[df["exclude_reason"].isna()].copy()

    return primary, excluded


def sanitize_label_token(x) -> str:
    s = normalize_none_string(x)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("+", "pos").replace("-", "neg")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "None"


def make_ar_state_df(
    primary: pd.DataFrame,
    *,
    state_none_label: str,
    state_separator: str,
    all_neg_mode: str,
) -> pd.DataFrame:
    ar = primary[primary["Panel"].astype(str).eq("AR")].copy()
    if ar.empty:
        return ar

    state = ar["state"].map(normalize_none_string)
    state = state.replace({"None": state_none_label})
    state = state.map(sanitize_label_token)

    base = ar["collapse_label"].astype(str).str.strip()
    label = base + state_separator + state

    if all_neg_mode == "keep_all_neg":
        label = label.where(~base.eq("ALL_NEG"), "ALL_NEG")

    ar["collapse_label"] = label
    return ar



# -----------------------------------------------------------------------------
# Derived label views for downstream feature sources
# -----------------------------------------------------------------------------

def _norm_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_state_label(x, none_label: str = "checkpoint_neg") -> str:
    """Normalize AR checkpoint state to a compact label token."""
    s = normalize_none_string(x)
    if s == "None":
        s = none_label
    return sanitize_label_token(s)


def make_checkpoint_binary(state_label: str) -> str:
    s = str(state_label).strip()
    if s in {"", "None", "checkpoint_neg", "nan", "NA", "<NA>"}:
        return "checkpoint_neg"
    return "checkpoint_pos"


def map_compartment_label(collapse_label) -> str:
    """Map fine phenotype labels to tumor / immune / stroma.

    STROMA is intentionally a broad marker-negative / lineage-unassigned
    stromal-background compartment. Anything not tumor or stroma and not already
    excluded by primary filtering is treated as immune.
    """
    s = _norm_label(collapse_label)
    sl = s.lower()

    tumor_labels = {
        "tumor", "tumour", "cancer", "panck", "panckpos", "panck_pos",
        "epithelial", "epithelial_tumor", "neoplastic",
    }
    stroma_labels = {
        "stroma", "stroma_unassigned", "stromal_background", "background",
        "all_neg", "allneg", "marker_neg", "marker_negative",
    }

    if sl in tumor_labels:
        return "tumor"
    if sl in stroma_labels:
        return "stroma"
    if sl in {"", "none", "nan", "<na>"}:
        return pd.NA
    return "immune"


def add_derived_label_columns(
    cell: pd.DataFrame,
    *,
    state_none_label: str = "checkpoint_neg",
    state_separator: str = "__",
) -> pd.DataFrame:
    """Add reusable label columns for downstream feature generation.

    Outputs support these feature sources:
      phenotype_only       -> label_phenotype, panels AR/BT
      AR_state             -> label_ar_state, panel AR
      AR_checkpoint_state  -> label_checkpoint_state, panel AR
      compartment          -> label_compartment, panels AR/BT
      compartment_state    -> label_compartment_state, panel AR
    """
    out = cell.copy()

    panel = out["Panel"].astype(str).str.strip().str.upper()
    is_ar = panel.eq("AR")

    base = out["collapse_label"].astype(str).str.strip()
    state_clean = out["state"].map(lambda x: clean_state_label(x, state_none_label))

    # 1) Normal phenotype-only label.
    out["label_phenotype"] = base

    # 2) AR lineage-aware state label. Apply state suffix to all AR base labels,
    # including STROMA, so STROMA__PD1/STROMA__PDL1 are retained.
    out["label_ar_state"] = pd.NA
    out.loc[is_ar, "label_ar_state"] = (
        base.loc[is_ar].astype(str) + state_separator + state_clean.loc[is_ar].astype(str)
    )

    # 3) AR checkpoint-state-only label.
    out["label_checkpoint_state"] = pd.NA
    out.loc[is_ar, "label_checkpoint_state"] = state_clean.loc[is_ar]

    # 4) AR checkpoint-positive binary label.
    out["label_checkpoint_binary"] = pd.NA
    out.loc[is_ar, "label_checkpoint_binary"] = out.loc[is_ar, "label_checkpoint_state"].map(make_checkpoint_binary)

    # 5) Coarse tumor / immune / stroma label.
    out["label_compartment"] = out["collapse_label"].map(map_compartment_label)

    # 6) AR coarse compartment + checkpoint state label.
    out["label_compartment_state"] = pd.NA
    comp = out["label_compartment"].astype(str).str.strip()
    out.loc[is_ar, "label_compartment_state"] = (
        comp.loc[is_ar].astype(str) + state_separator + state_clean.loc[is_ar].astype(str)
    )

    # Clean pseudo-missing strings in label columns.
    for c in DERIVED_LABEL_COLS:
        out[c] = out[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    return out


def finalize_header(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FINAL_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[FINAL_COLS].copy()
    # Keep old-like dtypes / strings.
    for c in ["sample_name", "coord", "tissue_region", "marker_combination", "phenotype", "Panel", "cohort", "phenotype_canonical", "collapse_label", "state", *DERIVED_LABEL_COLS]:
        out[c] = out[c].astype("object")
    return out


def summarize_cells(label: str, df: pd.DataFrame) -> dict:
    return {
        "dataset": label,
        "n_cells": len(df),
        "n_samples": df["sample_name"].nunique() if "sample_name" in df.columns else np.nan,
        "n_coords": df["coord"].nunique() if "coord" in df.columns else np.nan,
        "panels": ";".join(sorted(df["Panel"].dropna().astype(str).unique())) if "Panel" in df.columns else "",
        "cohorts": ";".join(sorted(df["cohort"].dropna().astype(str).unique())) if "cohort" in df.columns else "",
        "n_labels": df["collapse_label"].nunique(dropna=True) if "collapse_label" in df.columns else np.nan,
    }


def write_readme(out_dir: Path) -> None:
    text = f"""# Reviewed cell-level dataframe rebuild

Generated by `build_reviewed_cell_level_dfs.py` on {datetime.now().isoformat(timespec='seconds')}.

## Main outputs

- `tma_cell_df.parquet`: primary reviewed TMA cell-level table with `collapse_label` and derived label columns.
- `wholesection_cell_df.parquet`: primary reviewed BLASST / whole-section cell-level table with `collapse_label` and derived label columns.
Optional if `--write-ar-state-parquets` is used:

- `tma_cell_df_AR_state.parquet`: AR-only TMA table with the same columns as `tma_cell_df.parquet`, but `collapse_label` contains phenotype+state labels.
- `wholesection_cell_df_AR_state.parquet`: AR-only whole-section table with phenotype+state labels.

In the current recommended workflow, AR state labels can instead be created at prep time using `prep_weibull_inputs_v3.py --label-mode ar_state`, so these duplicate AR-state parquets are optional.

## Header

All primary parquet outputs use the same column order:

```text
{', '.join(FINAL_COLS)}
```

## Filtering

Primary outputs exclude rows according to the command-line options. By default:

- `artifact_flag == TRUE` is excluded.
- `exploratory == TRUE` is excluded.
- `collapse_label in {{artifact, unresolved, mixed_lineage}}` is excluded.
- unmapped marker combinations are excluded.

Excluded and unmapped rows are summarized in CSV diagnostics.

## AR state labels

For AR-only state outputs, `collapse_label` is replaced by labels such as:

```text
macrophage__PDL1
t_cell__PD1
tumor__checkpoint_neg
```

The remaining columns are intentionally identical to the standard cell_df so that downstream scripts that use `--label-col collapse_label` can be run unchanged.
"""
    (out_dir / "README_reviewed_cell_df_outputs.md").write_text(text)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_map, canonical_map, conflicts = load_phenotype_map(args.phenotype_assignments_dir, args.panels)
    raw_map.to_csv(out_dir / "phenotype_assignment_map_raw.csv", index=False)
    canonical_map.to_csv(out_dir / "phenotype_assignment_map_canonical_used.csv", index=False)
    conflicts.to_csv(out_dir / "phenotype_assignment_canonical_duplicate_conflicts.csv", index=False)

    tma_all, tma_inv = build_cell_df_from_parquets(
        args.tma_parquet_dir,
        canonical_map,
        dataset_kind="tma",
        panels=args.panels,
        discovery_mode=args.parquet_discovery_mode,
    )
    whole_all, whole_inv = build_cell_df_from_parquets(
        args.whole_parquet_dir,
        canonical_map,
        dataset_kind="whole",
        panels=args.panels,
        discovery_mode=args.parquet_discovery_mode,
    )

    tma_inv.to_csv(out_dir / "tma_input_parquet_inventory.csv", index=False)
    whole_inv.to_csv(out_dir / "wholesection_input_parquet_inventory.csv", index=False)

    tma_primary, tma_excluded = apply_primary_filter(
        tma_all,
        exclude_artifacts=args.exclude_artifacts,
        exclude_exploratory=args.exclude_exploratory,
        exclude_collapse_labels=args.exclude_collapse_labels,
    )
    whole_primary, whole_excluded = apply_primary_filter(
        whole_all,
        exclude_artifacts=args.exclude_artifacts,
        exclude_exploratory=args.exclude_exploratory,
        exclude_collapse_labels=args.exclude_collapse_labels,
    )

    tma_primary = add_derived_label_columns(
        tma_primary,
        state_none_label=args.state_none_label,
        state_separator=args.state_separator,
    )
    whole_primary = add_derived_label_columns(
        whole_primary,
        state_none_label=args.state_none_label,
        state_separator=args.state_separator,
    )

    tma_final = finalize_header(tma_primary)
    whole_final = finalize_header(whole_primary)

    tma_ar_state = finalize_header(make_ar_state_df(
        tma_primary,
        state_none_label=args.state_none_label,
        state_separator=args.state_separator,
        all_neg_mode=args.state_label_all_neg_mode,
    ))
    whole_ar_state = finalize_header(make_ar_state_df(
        whole_primary,
        state_none_label=args.state_none_label,
        state_separator=args.state_separator,
        all_neg_mode=args.state_label_all_neg_mode,
    ))

    # Write requested primary parquet files.
    tma_final.to_parquet(out_dir / "tma_cell_df.parquet", index=False)
    whole_final.to_parquet(out_dir / "wholesection_cell_df.parquet", index=False)

    if args.write_ar_state_parquets:
        tma_ar_state.to_parquet(out_dir / "tma_cell_df_AR_state.parquet", index=False)
        whole_ar_state.to_parquet(out_dir / "wholesection_cell_df_AR_state.parquet", index=False)

    if args.write_audit_parquets:
        tma_all.to_parquet(out_dir / "tma_cell_df_all_mapped_audit.parquet", index=False)
        whole_all.to_parquet(out_dir / "wholesection_cell_df_all_mapped_audit.parquet", index=False)

    # Diagnostics.
    tma_excluded.groupby(["Panel", "cohort", "exclude_reason"], dropna=False).size().reset_index(name="n_cells").to_csv(
        out_dir / "tma_excluded_cell_summary.csv", index=False
    )
    whole_excluded.groupby(["Panel", "cohort", "exclude_reason"], dropna=False).size().reset_index(name="n_cells").to_csv(
        out_dir / "wholesection_excluded_cell_summary.csv", index=False
    )

    for name, df in [("tma", tma_all), ("wholesection", whole_all)]:
        unmapped = df[df["collapse_label"].isna()].copy()
        if unmapped.empty:
            pd.DataFrame(columns=["Panel", "cohort", "phenotype_canonical", "n_cells"]).to_csv(out_dir / f"{name}_unmapped_phenotype_summary.csv", index=False)
        else:
            (unmapped.groupby(["Panel", "cohort", "phenotype_canonical"], dropna=False)
             .size()
             .reset_index(name="n_cells")
             .sort_values("n_cells", ascending=False)
             .to_csv(out_dir / f"{name}_unmapped_phenotype_summary.csv", index=False))

    summary_rows = [
        summarize_cells("tma_cell_df", tma_final),
        summarize_cells("wholesection_cell_df", whole_final),
    ]
    if args.write_ar_state_parquets:
        summary_rows.extend([
            summarize_cells("tma_cell_df_AR_state", tma_ar_state),
            summarize_cells("wholesection_cell_df_AR_state", whole_ar_state),
        ])
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "cell_df_output_summary.csv", index=False)

    # Derived-label audit tables.
    label_audit_parts = []
    for dataset_name, df_lab in [("tma", tma_final), ("wholesection", whole_final)]:
        for label_col in DERIVED_LABEL_COLS:
            tmp = (
                df_lab.groupby(["Panel", "cohort", label_col], dropna=False)
                .size()
                .reset_index(name="n_cells")
                .rename(columns={label_col: "label"})
            )
            tmp["dataset"] = dataset_name
            tmp["label_col"] = label_col
            label_audit_parts.append(tmp)
    if label_audit_parts:
        label_audit = pd.concat(label_audit_parts, ignore_index=True, sort=False)
        label_audit = label_audit[["dataset", "Panel", "cohort", "label_col", "label", "n_cells"]]
        label_audit.to_csv(out_dir / "derived_label_cell_summary.csv", index=False)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tma_parquet_dir": str(args.tma_parquet_dir),
        "whole_parquet_dir": str(args.whole_parquet_dir),
        "phenotype_assignments_dir": str(args.phenotype_assignments_dir),
        "out_dir": str(out_dir),
        "panels": args.panels,
        "exclude_artifacts": args.exclude_artifacts,
        "exclude_exploratory": args.exclude_exploratory,
        "exclude_collapse_labels": args.exclude_collapse_labels,
        "state_none_label": args.state_none_label,
        "state_separator": args.state_separator,
        "state_label_all_neg_mode": args.state_label_all_neg_mode,
        "derived_label_cols": DERIVED_LABEL_COLS,
        "write_ar_state_parquets": args.write_ar_state_parquets,
        "parquet_discovery_mode": args.parquet_discovery_mode,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_readme(out_dir)

    print("\nDONE. Output summary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
