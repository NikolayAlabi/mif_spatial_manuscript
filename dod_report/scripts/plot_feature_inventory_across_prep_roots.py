#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build feature-inventory plots/tables across all reviewed prep roots for the DOD report.

Scans spatial chunk outputs and aggregated cell/triad feature tables:
  - WEIBULL_ROOT/run_reviewed_*/.../chunk_*/NNstats.tsv
  - WEIBULL_ROOT/run_reviewed_*/.../chunk_*/athena_features.csv
  - CELL_FEATURE_ROOT/<source>/cell_features_*.csv
  - CELL_FEATURE_ROOT/triads_<source>/triad_features_*.csv

Default cohorts: NAC2020, PURE01, No-NAC, BLASST. NAC2015 is excluded.

Main plots:
  1. Stacked bar: unique generated features by cohort/panel/family, amalgamated across prep roots.
  2. Stacked bar: mean non-missing features per core/section by cohort/panel/family.
  3. Heatmap: unique generated features by prep root/family/cohort/panel.

Each feature is namespaced as prep_root::feature_family::feature_name, so the same nominal
feature in different prep roots is treated as a distinct generated feature.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_WEIBULL_ROOT = Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull")
DEFAULT_CELL_FEATURE_ROOT = Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed")
DEFAULT_OUT_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/figure_outputs/feature_inventory")

REPORT_COHORTS = ["NAC2020", "PURE01", "No-NAC", "BLASST"]
PANEL_ORDER = ["AR", "BT"]

PREP_ROOTS = {
    "phenotype_only": {
        "run_suffix": "run_reviewed_phenotype_only",
        "cell_subdir": "phenotype_only",
        "triad_subdir": "triads_phenotype_only",
    },
    "AR_state": {
        "run_suffix": "run_reviewed_AR_state",
        "cell_subdir": "AR_state",
        "triad_subdir": "triads_AR_state",
    },
    "AR_checkpoint_state": {
        "run_suffix": "run_reviewed_AR_checkpoint_state",
        "cell_subdir": "AR_checkpoint_state",
        "triad_subdir": "triads_AR_checkpoint_state",
    },
    "compartment": {
        "run_suffix": "run_reviewed_compartment",
        "cell_subdir": "compartment",
        "triad_subdir": "triads_compartment",
    },
    "compartment_state": {
        "run_suffix": "run_reviewed_compartment_state",
        "cell_subdir": "compartment_state",
        "triad_subdir": "triads_compartment_state",
    },
}

FEATURE_FAMILY_ORDER = ["cell_features", "nearest_neighbor", "ATHENA", "triads"]

KNOWN_METADATA_COLS = {
    "sample", "sample_id", "sample_name", "tnumber", "coord", "core", "core_id",
    "entity_id", "patient_id", "mif_patient_id", "dataset", "cohort", "cohort_label",
    "Panel", "panel", "chunk", "path", "source", "prep_root", "feature_family",
    "ClusterID", "analysisregion", "assigned_loc", "tumor_stroma", "tissue_region",
    "region", "Region", "Xcenter", "Ycenter", "x", "y", "X", "Y",
    "feature", "feature_name", "metric", "metric_name", "variable", "value",
    "count", "n", "index",
}


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_cohort(x) -> str:
    s = norm_str(x)
    su = s.upper().replace("_", " ").replace("-", "")
    if su in {"BCA2020", "BCA 2020", "NAC2020"}:
        return "NAC2020"
    if su in {"NONAC", "NO NAC", "NO-NAC"}:
        return "No-NAC"
    if su == "PURE01":
        return "PURE01"
    if su == "BLASST":
        return "BLASST"
    if su == "NAC2015" or su.startswith("BLADDER"):
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


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def get_col_case_insensitive(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def get_feature_cols_wide(df: pd.DataFrame) -> list[str]:
    meta_lower = {x.lower() for x in KNOWN_METADATA_COLS}
    feature_cols = []
    for c in df.columns:
        if str(c).lower() in meta_lower:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            feature_cols.append(c)
    return feature_cols


def infer_metadata_from_path(path: Path, prep_root: str, feature_family: str, weibull_root: Path) -> dict:
    out = {
        "prep_root": prep_root,
        "feature_family": feature_family,
        "dataset_from_path": "",
        "cohort_from_path": "",
        "panel_from_path": "",
        "chunk_from_path": "",
    }
    try:
        run_root = (weibull_root / PREP_ROOTS[prep_root]["run_suffix"]).resolve()
        rel = path.resolve().relative_to(run_root)
        parts = list(rel.parts)
        if len(parts) >= 5:
            out["dataset_from_path"] = parts[0]
            out["cohort_from_path"] = normalize_cohort(parts[1])
            out["panel_from_path"] = normalize_panel(parts[2])
            out["chunk_from_path"] = parts[3]
    except Exception:
        pass
    return out


def discover_feature_tables(weibull_root: Path, cell_feature_root: Path, prep_roots: Iterable[str]) -> pd.DataFrame:
    rows = []
    for prep_root in prep_roots:
        if prep_root not in PREP_ROOTS:
            continue
        cfg = PREP_ROOTS[prep_root]
        spatial_root = weibull_root / cfg["run_suffix"]
        if spatial_root.exists():
            for fp in sorted(spatial_root.rglob("NNstats.tsv")):
                rows.append({"path": str(fp), "exists": True, **infer_metadata_from_path(fp, prep_root, "nearest_neighbor", weibull_root)})
            for fp in sorted(spatial_root.rglob("athena_features.csv")):
                rows.append({"path": str(fp), "exists": True, **infer_metadata_from_path(fp, prep_root, "ATHENA", weibull_root)})

        cell_dir = cell_feature_root / cfg["cell_subdir"]
        if cell_dir.exists():
            for fp in sorted(cell_dir.glob("cell_features*.csv")):
                rows.append({"path": str(fp), "exists": True, **infer_metadata_from_path(fp, prep_root, "cell_features", weibull_root)})

        triad_dir = cell_feature_root / cfg["triad_subdir"]
        if triad_dir.exists():
            for fp in sorted(triad_dir.glob("triad_features*.csv")):
                rows.append({"path": str(fp), "exists": True, **infer_metadata_from_path(fp, prep_root, "triads", weibull_root)})

    if not rows:
        return pd.DataFrame(columns=["path", "exists", "prep_root", "feature_family", "dataset_from_path", "cohort_from_path", "panel_from_path", "chunk_from_path"])
    return pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)


def summarize_wide(df: pd.DataFrame, rec: pd.Series, unit_col: str, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    cohort_col = get_col_case_insensitive(d, ["cohort", "cohort_label"])
    panel_col = get_col_case_insensitive(d, ["Panel", "panel"])
    d["cohort_norm"] = d[cohort_col].map(normalize_cohort) if cohort_col else normalize_cohort(rec.get("cohort_from_path", ""))
    d["panel_norm"] = d[panel_col].map(normalize_panel) if panel_col else normalize_panel(rec.get("panel_from_path", ""))
    d["unit_id"] = d[unit_col].astype(str)

    rows = []
    for (cohort, panel), sub in d.groupby(["cohort_norm", "panel_norm"], dropna=False):
        n_units = sub["unit_id"].nunique()
        if n_units == 0:
            continue
        feat_numeric = sub[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=sub.index)
        nonmissing_by_unit = feat_numeric.notna().sum(axis=1) if feature_cols else pd.Series(0, index=sub.index)
        for feat in feature_cols:
            vals = pd.to_numeric(sub[feat], errors="coerce")
            units_nonmissing = sub.loc[vals.notna(), "unit_id"].nunique()
            if units_nonmissing == 0:
                continue
            rows.append({
                "prep_root": rec["prep_root"],
                "feature_family": rec["feature_family"],
                "cohort": normalize_cohort(cohort),
                "panel": normalize_panel(panel),
                "feature_name": str(feat),
                "feature_uid": f"{rec['prep_root']}::{rec['feature_family']}::{feat}",
                "n_units_total": n_units,
                "n_units_nonmissing": units_nonmissing,
                "fraction_units_nonmissing": units_nonmissing / n_units if n_units else np.nan,
                "format_detected": "wide",
                "source_file": rec["path"],
            })
        unit_rows = (
            pd.DataFrame({
                "cohort": normalize_cohort(cohort),
                "panel": normalize_panel(panel),
                "unit_id": sub["unit_id"].values,
                "n_nonmissing_features_for_unit": nonmissing_by_unit.values,
            })
            .groupby(["cohort", "panel", "unit_id"], dropna=False)
            .agg(n_nonmissing_features_for_unit=("n_nonmissing_features_for_unit", "sum"))
            .reset_index()
        )
        unit_rows["prep_root"] = rec["prep_root"]
        unit_rows["feature_family"] = rec["feature_family"]
        unit_rows["source_file"] = rec["path"]
        yield pd.DataFrame(rows), unit_rows
        rows = []


def summarize_long(df: pd.DataFrame, rec: pd.Series, unit_col: str, feature_col: str, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    cohort_col = get_col_case_insensitive(d, ["cohort", "cohort_label"])
    panel_col = get_col_case_insensitive(d, ["Panel", "panel"])
    d["cohort_norm"] = d[cohort_col].map(normalize_cohort) if cohort_col else normalize_cohort(rec.get("cohort_from_path", ""))
    d["panel_norm"] = d[panel_col].map(normalize_panel) if panel_col else normalize_panel(rec.get("panel_from_path", ""))
    d["unit_id"] = d[unit_col].astype(str)
    d["feature_name"] = d[feature_col].astype(str)
    d["value_numeric"] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d["feature_name"].notna() & d["unit_id"].notna()].copy()

    rows = []
    for (cohort, panel, feat), sub in d.groupby(["cohort_norm", "panel_norm", "feature_name"], dropna=False):
        n_units = sub["unit_id"].nunique()
        units_nonmissing = sub.loc[sub["value_numeric"].notna(), "unit_id"].nunique()
        if units_nonmissing == 0:
            continue
        rows.append({
            "prep_root": rec["prep_root"],
            "feature_family": rec["feature_family"],
            "cohort": normalize_cohort(cohort),
            "panel": normalize_panel(panel),
            "feature_name": str(feat),
            "feature_uid": f"{rec['prep_root']}::{rec['feature_family']}::{feat}",
            "n_units_total": n_units,
            "n_units_nonmissing": units_nonmissing,
            "fraction_units_nonmissing": units_nonmissing / n_units if n_units else np.nan,
            "format_detected": "long",
            "source_file": rec["path"],
        })

    nn = (
        d[d["value_numeric"].notna()]
        .drop_duplicates(["cohort_norm", "panel_norm", "unit_id", "feature_name"])
        .groupby(["cohort_norm", "panel_norm", "unit_id"], dropna=False)
        .size()
        .reset_index(name="n_nonmissing_features_for_unit")
    )
    unit_rows = nn.rename(columns={"cohort_norm": "cohort", "panel_norm": "panel"})
    unit_rows["cohort"] = unit_rows["cohort"].map(normalize_cohort)
    unit_rows["panel"] = unit_rows["panel"].map(normalize_panel)
    unit_rows["prep_root"] = rec["prep_root"]
    unit_rows["feature_family"] = rec["feature_family"]
    unit_rows["source_file"] = rec["path"]
    return pd.DataFrame(rows), unit_rows


def summarize_one_file(rec: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(rec["path"])
    try:
        df = read_table(path)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"path": str(path), "error": repr(e)}])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"path": str(path), "error": "empty table"}])

    unit_col = get_col_case_insensitive(df, ["sample_id", "sample", "sample_name", "tnumber", "coord"])
    if unit_col is None:
        df = df.copy()
        df["sample_id"] = path.parent.name
        unit_col = "sample_id"

    feature_col = get_col_case_insensitive(df, ["feature", "feature_name", "metric", "metric_name", "variable"])
    value_col = get_col_case_insensitive(df, ["value", "feature_value", "metric_value"])
    if feature_col is not None and value_col is not None:
        feat, unit = summarize_long(df, rec, unit_col, feature_col, value_col)
        return feat, unit, pd.DataFrame()

    feature_cols = get_feature_cols_wide(df)
    feat_parts, unit_parts = [], []
    for feat, unit in summarize_wide(df, rec, unit_col, feature_cols):
        if not feat.empty:
            feat_parts.append(feat)
        if not unit.empty:
            unit_parts.append(unit)
    feat = pd.concat(feat_parts, ignore_index=True) if feat_parts else pd.DataFrame()
    unit = pd.concat(unit_parts, ignore_index=True) if unit_parts else pd.DataFrame()
    return feat, unit, pd.DataFrame()


def build_feature_inventory(weibull_root: Path, cell_feature_root: Path, cohorts: list[str], panels: list[str], prep_roots: list[str]) -> dict[str, pd.DataFrame]:
    manifest = discover_feature_tables(weibull_root, cell_feature_root, prep_roots)
    feature_parts, unit_parts, error_parts = [], [], []

    for _, rec in manifest.iterrows():
        feat, unit, err = summarize_one_file(rec)
        if not feat.empty:
            feature_parts.append(feat)
        if not unit.empty:
            unit_parts.append(unit)
        if not err.empty:
            error_parts.append(err.assign(source_file=rec["path"], prep_root=rec["prep_root"], feature_family=rec["feature_family"]))

    feature_long = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    unit_long = pd.concat(unit_parts, ignore_index=True) if unit_parts else pd.DataFrame()
    errors = pd.concat(error_parts, ignore_index=True) if error_parts else pd.DataFrame()

    if feature_long.empty:
        return {"manifest": manifest, "feature_long": feature_long, "unit_long": unit_long, "errors": errors,
                "summary_family": pd.DataFrame(), "summary_prep_root": pd.DataFrame(), "summary_prep_root_family": pd.DataFrame()}

    feature_long["cohort"] = feature_long["cohort"].map(normalize_cohort)
    feature_long["panel"] = feature_long["panel"].map(normalize_panel)
    feature_long = feature_long[feature_long["cohort"].isin(cohorts) & feature_long["panel"].isin(panels)].copy()

    if not unit_long.empty:
        unit_long["cohort"] = unit_long["cohort"].map(normalize_cohort)
        unit_long["panel"] = unit_long["panel"].map(normalize_panel)
        unit_long = unit_long[unit_long["cohort"].isin(cohorts) & unit_long["panel"].isin(panels)].copy()

    available = (
        feature_long.groupby(["cohort", "panel", "prep_root", "feature_family", "feature_uid"], dropna=False)
        .agg(
            feature_name=("feature_name", "first"),
            n_units_total=("n_units_total", "max"),
            n_units_nonmissing=("n_units_nonmissing", "max"),
            fraction_units_nonmissing=("fraction_units_nonmissing", "max"),
            n_source_files=("source_file", "nunique"),
        )
        .reset_index()
    )

    if not unit_long.empty:
        unit_summary = (
            unit_long.groupby(["cohort", "panel", "prep_root", "feature_family"], dropna=False)
            .agg(
                n_units_observed=("unit_id", "nunique"),
                mean_nonmissing_features_per_unit=("n_nonmissing_features_for_unit", "mean"),
                median_nonmissing_features_per_unit=("n_nonmissing_features_for_unit", "median"),
            )
            .reset_index()
        )
    else:
        unit_summary = pd.DataFrame()

    summary_prep_root_family = (
        available.groupby(["cohort", "panel", "prep_root", "feature_family"], dropna=False)
        .agg(
            n_unique_features=("feature_uid", "nunique"),
            n_features_present_in_all_units=("fraction_units_nonmissing", lambda s: int((s >= 1.0).sum())),
            median_fraction_units_nonmissing=("fraction_units_nonmissing", "median"),
            n_source_files=("n_source_files", "sum"),
        )
        .reset_index()
    )
    if not unit_summary.empty:
        summary_prep_root_family = summary_prep_root_family.merge(unit_summary, on=["cohort", "panel", "prep_root", "feature_family"], how="left")

    summary_family = (
        summary_prep_root_family.groupby(["cohort", "panel", "feature_family"], dropna=False)
        .agg(
            n_unique_features=("n_unique_features", "sum"),
            n_features_present_in_all_units=("n_features_present_in_all_units", "sum"),
            median_fraction_units_nonmissing=("median_fraction_units_nonmissing", "median"),
            mean_nonmissing_features_per_unit=("mean_nonmissing_features_per_unit", "sum"),
            median_nonmissing_features_per_unit=("median_nonmissing_features_per_unit", "sum"),
            n_units_observed=("n_units_observed", "sum"),
        )
        .reset_index()
    )

    summary_prep_root = (
        summary_prep_root_family.groupby(["cohort", "panel", "prep_root"], dropna=False)
        .agg(
            n_unique_features=("n_unique_features", "sum"),
            n_features_present_in_all_units=("n_features_present_in_all_units", "sum"),
            median_fraction_units_nonmissing=("median_fraction_units_nonmissing", "median"),
            mean_nonmissing_features_per_unit=("mean_nonmissing_features_per_unit", "sum"),
            median_nonmissing_features_per_unit=("median_nonmissing_features_per_unit", "sum"),
            n_units_observed=("n_units_observed", "sum"),
        )
        .reset_index()
    )

    return {"manifest": manifest, "feature_long": available, "unit_long": unit_long, "errors": errors,
            "summary_family": summary_family, "summary_prep_root": summary_prep_root, "summary_prep_root_family": summary_prep_root_family}


def plot_stacked_by_family(summary: pd.DataFrame, value_col: str, panels: list[str], cohorts: list[str], title: str, ylabel: str):
    d = summary[summary["panel"].isin(panels) & summary["cohort"].isin(cohorts)].copy()
    fig, axes = plt.subplots(1, len(panels), figsize=(12.8, 4.6), sharey=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        sub = d[d["panel"].eq(panel)].copy()
        pivot = sub.pivot_table(index="cohort", columns="feature_family", values=value_col, aggfunc="sum", fill_value=0).reindex(cohorts).fillna(0)
        x = np.arange(len(pivot.index))
        bottom = np.zeros(len(pivot.index))
        for fam in FEATURE_FAMILY_ORDER:
            vals = pivot[fam].to_numpy() if fam in pivot.columns else np.zeros(len(pivot.index))
            ax.bar(x, vals, bottom=bottom, label=fam)
            bottom += vals
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
        ax.set_ylabel(ylabel if panel == panels[0] else "")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.02, 1.0), title="Feature family")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    return fig, axes


def plot_heatmap_prep_root_family(summary: pd.DataFrame, panels: list[str], cohorts: list[str], value_col: str = "n_unique_features"):
    d = summary[summary["panel"].isin(panels) & summary["cohort"].isin(cohorts)].copy()
    d["row_label"] = d["prep_root"].astype(str) + " | " + d["feature_family"].astype(str)
    d["col_label"] = d["panel"].astype(str) + "\n" + d["cohort"].astype(str)
    row_order = [f"{root} | {fam}" for root in PREP_ROOTS for fam in FEATURE_FAMILY_ORDER if f"{root} | {fam}" in set(d["row_label"])]
    col_order = [f"{panel}\n{cohort}" for panel in panels for cohort in cohorts]
    mat = d.pivot_table(index="row_label", columns="col_label", values=value_col, aggfunc="sum", fill_value=0).reindex(index=row_order, columns=col_order).fillna(0)
    fig_h = max(5.5, 0.35 * len(mat.index) + 1.5)
    fig_w = max(8.5, 0.8 * len(mat.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title("Unique generated features by prep root and feature family", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(value_col.replace("_", " "))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7)
    return fig, ax


def save_outputs(outputs: dict[str, pd.DataFrame], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    table_files = {
        "manifest": "feature_inventory_file_manifest.csv",
        "feature_long": "source_data_feature_inventory_long.csv",
        "unit_long": "source_data_mean_usable_features_by_file.csv",
        "errors": "feature_inventory_read_errors.csv",
        "summary_family": "table_feature_inventory_by_cohort_panel_family.csv",
        "summary_prep_root": "table_feature_inventory_by_prep_root.csv",
        "summary_prep_root_family": "table_feature_inventory_by_cohort_panel_prep_root_family.csv",
    }
    table_paths = {}
    for key, fname in table_files.items():
        p = table_dir / fname
        outputs.get(key, pd.DataFrame()).to_csv(p, index=False)
        table_paths[key] = p

    dictionary = pd.DataFrame([
        {"file": "table_feature_inventory_by_cohort_panel_family.csv", "description": "Main summary: generated feature counts by cohort, panel, and feature family, amalgamated across prep roots."},
        {"file": "table_feature_inventory_by_prep_root.csv", "description": "Summary by cohort, panel, and prep root."},
        {"file": "table_feature_inventory_by_cohort_panel_prep_root_family.csv", "description": "Detailed summary by cohort, panel, prep root, and feature family."},
        {"file": "source_data_feature_inventory_long.csv", "description": "Feature-level source data with prep-root namespacing."},
        {"file": "source_data_mean_usable_features_by_file.csv", "description": "Unit-level mean non-missing feature source data."},
    ])
    dict_path = table_dir / "table_feature_inventory_data_dictionary.csv"
    dictionary.to_csv(dict_path, index=False)
    table_paths["dictionary"] = dict_path
    return {"table_paths": table_paths, "plot_dir": plot_dir}


def build_all_outputs(weibull_root=DEFAULT_WEIBULL_ROOT, cell_feature_root=DEFAULT_CELL_FEATURE_ROOT, out_dir=DEFAULT_OUT_DIR, cohorts=None, panels=None, prep_roots=None, save=True):
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER
    prep_roots = prep_roots or list(PREP_ROOTS.keys())
    outputs = build_feature_inventory(Path(weibull_root), Path(cell_feature_root), cohorts, panels, prep_roots)
    paths = save_outputs(outputs, Path(out_dir)) if save else {"table_paths": {}, "plot_dir": Path(out_dir) / "plots"}
    plot_paths = []
    if not outputs["summary_family"].empty:
        fig1, _ = plot_stacked_by_family(outputs["summary_family"], "n_unique_features", panels, cohorts, "Unique generated features across all prep roots", "Number of unique features")
        fig2, _ = plot_stacked_by_family(outputs["summary_family"], "mean_nonmissing_features_per_unit", panels, cohorts, "Mean usable features per core/section across all prep roots", "Mean non-missing features per unit")
        for fig, prefix in [(fig1, "fig_feature_inventory_unique_features_by_cohort_panel"), (fig2, "fig_feature_inventory_mean_usable_features_by_cohort_panel")]:
            if save:
                pdf = paths["plot_dir"] / f"{prefix}.pdf"
                png = paths["plot_dir"] / f"{prefix}.png"
                fig.savefig(pdf, bbox_inches="tight")
                fig.savefig(png, dpi=300, bbox_inches="tight")
                plot_paths.append({"pdf": pdf, "png": png})
    if not outputs["summary_prep_root_family"].empty:
        fig3, _ = plot_heatmap_prep_root_family(outputs["summary_prep_root_family"], panels, cohorts)
        if save:
            pdf = paths["plot_dir"] / "fig_feature_inventory_heatmap_by_prep_root_family.pdf"
            png = paths["plot_dir"] / "fig_feature_inventory_heatmap_by_prep_root_family.png"
            fig3.savefig(pdf, bbox_inches="tight")
            fig3.savefig(png, dpi=300, bbox_inches="tight")
            plot_paths.append({"pdf": pdf, "png": png})
    outputs.update(paths)
    outputs["plot_paths"] = plot_paths
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weibull-root", default=str(DEFAULT_WEIBULL_ROOT))
    ap.add_argument("--cell-feature-root", default=str(DEFAULT_CELL_FEATURE_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cohorts", nargs="+", default=REPORT_COHORTS)
    ap.add_argument("--panels", nargs="+", default=PANEL_ORDER)
    ap.add_argument("--prep-roots", nargs="+", default=list(PREP_ROOTS.keys()))
    args = ap.parse_args()
    outputs = build_all_outputs(
        weibull_root=Path(args.weibull_root),
        cell_feature_root=Path(args.cell_feature_root),
        out_dir=Path(args.out_dir),
        cohorts=args.cohorts,
        panels=args.panels,
        prep_roots=args.prep_roots,
        save=True,
    )
    print("\nGenerated tables:")
    for k, v in outputs.get("table_paths", {}).items():
        print(f"  {k}: {v}")
    print("\nGenerated plots:")
    for d in outputs.get("plot_paths", []):
        for k, v in d.items():
            print(f"  {k}: {v}")
    if outputs["summary_family"].empty:
        print("\n[WARN] No feature summaries generated. Check whether feature jobs are complete and paths are correct.")
    else:
        print("\nFeature summary:")
        print(outputs["summary_family"].to_string(index=False))


if __name__ == "__main__":
    main()
