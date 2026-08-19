#!/usr/bin/env python3
"""
Build per-sample/core cell count, proportion, and ratio features directly from
Weibull/ATHENA prep outputs (1NN_prep.tsv files).

Why this exists
---------------
This script mirrors make_cell_feature_table_v2.py, but uses the already-prepared
1NN_prep.tsv files as input instead of the canonical cell_df parquet files.
That means the labels are exactly the labels sent into Weibull/ATHENA:

  - phenotype-only prep root: phenotype == collapse_label
  - AR-state prep root:      phenotype == collapse_label__state, except exempt labels such as ALL_NEG

Densities are intentionally omitted by default and are not computed here.

Expected 1NN_prep.tsv columns
-----------------------------
Required:
  sample_id, analysisregion, phenotype
Optional but strongly preferred:
  sample_name, coord, Panel, cohort, Xcenter, Ycenter, tumor_stroma

Input modes
-----------
1. Pass one or more prep roots with --prep-roots. The script recursively finds
   all 1NN_prep.tsv files under each root.
2. Or pass explicit 1NN files with --prep-files.

The expected prep layout is:
  <prep_root>/<dataset>/<cohort>/<panel>/<chunk>/1NN_prep.tsv

Outputs
-------
  <outfile>                  Wide feature matrix, one row per sample/core/panel/cohort/dataset.
  cell_feature_label_map.csv Label and sanitization map.
  input_manifest.csv         Input 1NN files discovered/used.
  run_metadata.json          Provenance/parameters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


DEFAULT_BAD_LABELS = {"artifact", "unresolved", "mixed_lineage", "unmapped", "nan", "none", ""}
BASE_KEY_COLS = ["prep_run", "dataset", "sample_id", "sample_name", "coord", "Panel", "cohort"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute count/proportion/ratio features directly from 1NN_prep.tsv outputs."
    )
    ap.add_argument("--prep-roots", nargs="*", default=[], help="Prep output roots to scan recursively for 1NN_prep.tsv.")
    ap.add_argument("--prep-files", nargs="*", default=[], help="Explicit 1NN_prep.tsv files.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--outfile", default="cell_features_from_prep.csv", help="Output CSV filename.")

    ap.add_argument("--include-panels", nargs="+", default=None, help="Optional panel allow-list, e.g. AR BT.")
    ap.add_argument("--exclude-panels", nargs="+", default=[], help="Panels to exclude.")
    ap.add_argument("--include-cohorts", nargs="+", default=None, help="Optional cohort allow-list.")
    ap.add_argument("--exclude-cohorts", nargs="+", default=[], help="Cohorts to exclude.")

    ap.add_argument(
        "--bad-labels",
        nargs="+",
        default=sorted(DEFAULT_BAD_LABELS),
        help="Labels treated as non-primary. Dropped unless --keep-bad-labels is set; excluded from ratios either way.",
    )
    ap.add_argument("--keep-bad-labels", action="store_true", help="Keep bad/non-primary labels in counts/proportions.")
    ap.add_argument("--exclude-ratio-labels", nargs="+", default=[], help="Additional labels to exclude from ratios.")
    ap.add_argument("--exclude-all-neg-from-ratios", action="store_true", help="Exclude ALL_NEG from ratio features.")

    ap.add_argument("--min-cells-per-sample", type=int, default=1, help="Drop sample/panel groups with fewer cells.")
    ap.add_argument("--chunksize", type=int, default=750_000, help="Rows per chunk when reading TSVs.")
    ap.add_argument(
        "--aggregate-duplicate-samples",
        action="store_true",
        help="Aggregate duplicate sample_id rows across multiple prep roots/runs. Usually leave off when phenotype-only and AR-state are separate runs.",
    )
    return ap.parse_args()


def sanitize_label(s: object) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s if s else "EMPTY"


def normalize_panel(x: object) -> str:
    s = str(x).strip().upper()
    if s in {"ARP", "AR"}:
        return "AR"
    if s in {"BT", "B&T", "BTP", "T CELL", "TCELL"}:
        return "BT"
    if s in {"MY", "M", "MYELOID"}:
        return "MY"
    return s


def normalize_region(x: object) -> str:
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()
    if s in {"tumor", "tumour", "epi", "epithelial", "epithelium", "cancer", "neoplastic"}:
        return "Epi"
    if s in {"stroma", "stromal", "str"}:
        return "Stroma"
    if "tumor" in s or "tumour" in s or "epi" in s or "panck" in s:
        return "Epi"
    if "strom" in s:
        return "Stroma"
    return "Other"


def discover_prep_files(prep_roots: Sequence[str], prep_files: Sequence[str]) -> pd.DataFrame:
    rows = []

    for root_str in prep_roots:
        root = Path(root_str).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Prep root does not exist: {root}")
        for fp in sorted(root.rglob("1NN_prep.tsv")):
            rows.append(_manifest_row_from_file(fp.resolve(), root))

    for fp_str in prep_files:
        fp = Path(fp_str).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Prep file does not exist: {fp}")
        # No reliable root, infer from parent layout.
        rows.append(_manifest_row_from_file(fp, None))

    manifest = pd.DataFrame(rows).drop_duplicates(subset=["prep_file"]).reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No 1NN_prep.tsv files found. Provide --prep-roots and/or --prep-files.")
    return manifest


def _manifest_row_from_file(fp: Path, root: Optional[Path]) -> dict:
    chunk_dir = fp.parent
    chunk = chunk_dir.name
    panel = chunk_dir.parent.name if chunk_dir.parent else "Unknown"
    cohort = chunk_dir.parent.parent.name if chunk_dir.parent and chunk_dir.parent.parent else "Unknown"
    dataset = chunk_dir.parent.parent.parent.name if chunk_dir.parent and chunk_dir.parent.parent and chunk_dir.parent.parent.parent else "Unknown"

    if root is not None:
        prep_run = root.name
        prep_root = str(root)
    else:
        # Try to define prep_run as the directory above dataset.
        prep_run = chunk_dir.parent.parent.parent.parent.name if chunk_dir.parent.parent.parent.parent else "manual"
        prep_root = ""

    return {
        "prep_run": prep_run,
        "prep_root": prep_root,
        "dataset": dataset,
        "cohort": cohort,
        "Panel": normalize_panel(panel),
        "chunk": chunk,
        "prep_file": str(fp),
    }


def _standardize_prep_chunk(chunk: pd.DataFrame, meta: dict) -> pd.DataFrame:
    x = chunk.copy()
    x.columns = [str(c).strip() for c in x.columns]

    required = ["sample_id", "analysisregion", "phenotype"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"{meta['prep_file']} is missing required columns: {missing}")

    # Add/backfill metadata from path when not present.
    if "sample_name" not in x.columns:
        x["sample_name"] = x["sample_id"]
    if "coord" not in x.columns:
        x["coord"] = "Unknown"
    if "Panel" not in x.columns:
        x["Panel"] = meta["Panel"]
    if "cohort" not in x.columns:
        x["cohort"] = meta["cohort"]

    x["prep_run"] = meta["prep_run"]
    x["dataset"] = meta["dataset"]
    x["chunk"] = meta["chunk"]
    x["source_1nn_prep"] = meta["prep_file"]

    x["Panel"] = x["Panel"].map(normalize_panel)
    x["cohort"] = x["cohort"].astype(str).str.strip()
    x["sample_id"] = x["sample_id"].astype(str).str.strip()
    x["sample_name"] = x["sample_name"].astype(str).str.strip()
    x["coord"] = x["coord"].astype(str).str.strip()
    x["phenotype"] = x["phenotype"].astype(str).str.strip()

    # Prefer tumor_stroma if present, otherwise analysisregion.
    region_source = "tumor_stroma" if "tumor_stroma" in x.columns else "analysisregion"
    x["region_class"] = x[region_source].map(normalize_region)

    keep_cols = BASE_KEY_COLS + ["chunk", "phenotype", "region_class", "source_1nn_prep"]
    return x[keep_cols].copy()


def read_prep_files(manifest: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for i, meta in manifest.iterrows():
        fp = meta["prep_file"]
        print(f"[INFO] Reading {i + 1}/{len(manifest)}: {fp}")
        for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, low_memory=False):
            frames.append(_standardize_prep_chunk(chunk, meta.to_dict()))
    if not frames:
        raise ValueError("No rows were read from prep files.")
    return pd.concat(frames, ignore_index=True)


def compute_counts(df: pd.DataFrame, label_list: Sequence[str]) -> pd.Series:
    counts = df["phenotype"].value_counts(dropna=False)
    return counts.reindex(label_list, fill_value=0).astype(float)


def compute_props(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total <= 0:
        return pd.Series(np.nan, index=counts.index, dtype=float)
    return counts / total


def compute_all_ordered_ratios(counts: pd.Series) -> dict:
    out = {}
    labels = list(counts.index)
    for num in labels:
        num_val = counts[num]
        num_clean = sanitize_label(num)
        for den in labels:
            if den == num:
                continue
            den_val = counts[den]
            den_clean = sanitize_label(den)
            out[f"{num_clean}__over__{den_clean}"] = np.nan if den_val == 0 else float(num_val / den_val)
    return out


def summarize_region(df_region: pd.DataFrame, region_name: str, all_labels: Sequence[str], ratio_labels: Sequence[str]) -> dict:
    out = {}
    counts_all = compute_counts(df_region, all_labels)
    props_all = compute_props(counts_all)

    out[f"{region_name}__n_cells"] = float(counts_all.sum())
    for label in all_labels:
        clean = sanitize_label(label)
        out[f"{region_name}__count__{clean}"] = counts_all[label]
        out[f"{region_name}__prop__{clean}"] = props_all[label]

    df_ratio = df_region[df_region["phenotype"].isin(ratio_labels)].copy()
    ratio_counts = compute_counts(df_ratio, ratio_labels)
    out[f"{region_name}__n_resolved_for_ratio"] = float(ratio_counts.sum())
    for k, v in compute_all_ordered_ratios(ratio_counts).items():
        out[f"{region_name}__ratio__{k}"] = v

    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = discover_prep_files(args.prep_roots, args.prep_files)
    manifest.to_csv(outdir / "input_manifest.csv", index=False)
    print(f"[INFO] Found {len(manifest):,} 1NN_prep.tsv files")

    cell_df = read_prep_files(manifest, chunksize=args.chunksize)
    print(f"[INFO] Raw rows from prep files: {len(cell_df):,}")

    include_panels = {normalize_panel(x) for x in args.include_panels} if args.include_panels else None
    exclude_panels = {normalize_panel(x) for x in args.exclude_panels}
    include_cohorts = {str(x).strip() for x in args.include_cohorts} if args.include_cohorts else None
    exclude_cohorts = {str(x).strip() for x in args.exclude_cohorts}

    if include_panels is not None:
        cell_df = cell_df[cell_df["Panel"].isin(include_panels)].copy()
    if exclude_panels:
        cell_df = cell_df[~cell_df["Panel"].isin(exclude_panels)].copy()
    if include_cohorts is not None:
        cell_df = cell_df[cell_df["cohort"].isin(include_cohorts)].copy()
    if exclude_cohorts:
        cell_df = cell_df[~cell_df["cohort"].isin(exclude_cohorts)].copy()

    bad_labels_lower = {str(x).strip().lower() for x in args.bad_labels}
    before_bad = len(cell_df)
    if not args.keep_bad_labels:
        cell_df = cell_df[~cell_df["phenotype"].str.lower().isin(bad_labels_lower)].copy()
    print(f"[INFO] Dropped bad-label rows: {before_bad - len(cell_df):,}")

    if cell_df.empty:
        raise ValueError("No rows remain after filtering.")

    all_labels = sorted(cell_df["phenotype"].dropna().astype(str).unique().tolist())
    exclude_ratio = {str(x).strip() for x in args.exclude_ratio_labels} | {str(x).strip() for x in args.bad_labels}
    if args.exclude_all_neg_from_ratios:
        exclude_ratio.add("ALL_NEG")
    ratio_labels = sorted([x for x in all_labels if x not in exclude_ratio and x.lower() not in {z.lower() for z in exclude_ratio}])

    print(f"[INFO] Rows after filtering: {len(cell_df):,}")
    print(f"[INFO] Panels: {sorted(cell_df['Panel'].dropna().unique().tolist())}")
    print(f"[INFO] Cohorts: {sorted(cell_df['cohort'].dropna().unique().tolist())}")
    print(f"[INFO] Labels: {len(all_labels):,}; ratio labels: {len(ratio_labels):,}")

    label_map = pd.DataFrame({
        "label": all_labels,
        "sanitized_label": [sanitize_label(x) for x in all_labels],
        "used_in_ratios": [x in ratio_labels for x in all_labels],
    })
    label_map.to_csv(outdir / "cell_feature_label_map.csv", index=False)

    key_cols = BASE_KEY_COLS.copy()
    if args.aggregate_duplicate_samples:
        # Drop prep_run from grouping if intentionally combining multiple roots.
        key_cols = [c for c in key_cols if c != "prep_run"]

    rows = []
    grouped = cell_df.groupby(key_cols, dropna=False, sort=False)
    for i, (group_key, gdf) in enumerate(grouped, start=1):
        if i % 100 == 0:
            print(f"[INFO] Processed {i:,} groups")
        if len(gdf) < args.min_cells_per_sample:
            continue

        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(key_cols, group_key))
        row.update(summarize_region(gdf, "All", all_labels, ratio_labels))
        row.update(summarize_region(gdf[gdf["region_class"].eq("Epi")].copy(), "Epi", all_labels, ratio_labels))
        row.update(summarize_region(gdf[gdf["region_class"].eq("Stroma")].copy(), "Stroma", all_labels, ratio_labels))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    meta_cols = [c for c in key_cols if c in out_df.columns]
    feat_cols = [c for c in out_df.columns if c not in meta_cols]
    out_df = out_df[meta_cols + sorted(feat_cols)]

    out_csv = outdir / args.outfile
    out_df.to_csv(out_csv, index=False)

    metadata = {
        "prep_roots": args.prep_roots,
        "prep_files": args.prep_files,
        "out_csv": str(out_csv),
        "n_input_files": int(len(manifest)),
        "n_cells_after_filtering": int(len(cell_df)),
        "n_rows": int(out_df.shape[0]),
        "n_columns": int(out_df.shape[1]),
        "n_labels": int(len(all_labels)),
        "n_ratio_labels": int(len(ratio_labels)),
        "include_panels": sorted(include_panels) if include_panels is not None else None,
        "exclude_panels": sorted(exclude_panels),
        "include_cohorts": sorted(include_cohorts) if include_cohorts is not None else None,
        "exclude_cohorts": sorted(exclude_cohorts),
        "bad_labels": sorted(args.bad_labels),
        "exclude_ratio_labels": sorted(exclude_ratio),
        "aggregate_duplicate_samples": bool(args.aggregate_duplicate_samples),
        "densities_emitted": False,
    }
    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Shape: {out_df.shape[0]:,} rows x {out_df.shape[1]:,} columns")
    print(f"[DONE] Manifest: {outdir / 'input_manifest.csv'}")
    print(f"[DONE] Label map: {outdir / 'cell_feature_label_map.csv'}")


if __name__ == "__main__":
    main()
