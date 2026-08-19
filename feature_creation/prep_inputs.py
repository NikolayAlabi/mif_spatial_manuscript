#!/usr/bin/env python3
"""
Build ATHENA prep chunks directly from cell_df parquet files.

What this script does
---------------------
- Reads one or more cell_df parquet files.
- Uses a chosen label column, usually `collapse_label`, as the final phenotype label.
- Optionally builds AR checkpoint-state labels on the fly, e.g. `macrophage__PDL1`.
- Keeps extra metadata columns in 1NN_prep.tsv:
    sample_name, coord, Panel, cohort
- Excludes selected panels (default: MY).
- Splits outputs by dataset -> cohort -> panel -> chunk.
- Chunks by whole samples/cores, so all cells from a given sample stay together.
- Writes:
    1NN_prep.tsv
    tissue_prep.tsv
    sample_manifest.tsv
  into each chunk directory.

Notes
-----
- This is for ATHENA-style prep, not the old Weibull QC/type-mode prep.
- `tissue_prep.tsv` contains `total_area` in mm^2, because Step1 Weibull
  normalizes distance-bin counts as N / total_area and ATHENA converts mm^2
  back to um^2 for Ripley window construction.
- By default, `total_area` is estimated from each sample's bounding box using
  `--bbox-area-to-mm2-factor` (default 1e-6, appropriate when x/y are um).
- If a true tissue-area lookup is supplied with `--tissue-area-file`, that area
  is used preferentially.
- No QC directories, no type_simple/type_specific logic, no surgery splits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build chunked ATHENA prep files from cell_df parquet files."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more cell_df parquet files.",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output root directory.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of unique samples/cores per chunk.",
    )
    ap.add_argument(
        "--exclude-panels",
        nargs="+",
        default=["MY"],
        help="Panels to exclude entirely.",
    )
    ap.add_argument(
        "--label-col",
        default="collapse_label",
        help="Column to use as the base/final phenotype label.",
    )
    ap.add_argument(
        "--label-mode",
        choices=["column", "ar_state"],
        default="column",
        help=(
            "How to build the phenotype label written to 1NN_prep.tsv. "
            "'column' uses --label-col directly. "
            "'ar_state' appends --state-col to --label-col for AR panel cells only."
        ),
    )
    ap.add_argument(
        "--state-col",
        default="state",
        help="State column used when --label-mode ar_state, usually 'state'.",
    )
    ap.add_argument(
        "--state-panels",
        nargs="+",
        default=["AR"],
        help="Panels for which --label-mode ar_state should append state to the label.",
    )
    ap.add_argument(
        "--state-separator",
        default="__",
        help="Separator between base phenotype and state in ar_state mode.",
    )
    ap.add_argument(
        "--state-negative-label",
        default="checkpoint_neg",
        help="State label used for missing/None state values in ar_state mode.",
    )
    ap.add_argument(
        "--state-exempt-labels",
        nargs="+",
        default=["ALL_NEG"],
        help=(
            "Base labels that should not receive a state suffix in ar_state mode. "
            "Default keeps ALL_NEG as ALL_NEG instead of ALL_NEG__checkpoint_neg."
        ),
    )
    ap.add_argument(
        "--bbox-area-to-mm2-factor",
        type=float,
        default=1e-6,
        help=(
            "Conversion factor from bounding-box coordinate area to mm^2. "
            "Use 1e-6 if X/Y are microns. Use 1.0 if X/Y are already millimetres."
        ),
    )
    ap.add_argument(
        "--tissue-area-file",
        default=None,
        help=(
            "Optional CSV/TSV with true per-sample tissue area. If supplied, "
            "these values are preferred over bounding-box estimates."
        ),
    )
    ap.add_argument(
        "--tissue-area-sample-col",
        default="sample_id",
        help="Sample column in --tissue-area-file. Must match the prep sample_id after renaming.",
    )
    ap.add_argument(
        "--tissue-area-col",
        default="total_area",
        help="Area column in --tissue-area-file.",
    )
    ap.add_argument(
        "--tissue-area-units",
        choices=["mm2", "um2", "raw"],
        default="mm2",
        help=(
            "Units for --tissue-area-col. 'mm2' is used unchanged; "
            "'um2' is divided by 1e6; 'raw' is used unchanged."
        ),
    )
    ap.add_argument(
        "--include-panels",
        nargs="+",
        default=None,
        help="Optional panels to include. If supplied, applied before --exclude-panels.",
    )
    ap.add_argument(
        "--sample-col",
        default="sample_name",
        help="Column to use as sample_id.",
    )
    ap.add_argument(
        "--coord-col",
        default="coord",
        help="Coordinate token column to retain in outputs.",
    )
    ap.add_argument(
        "--x-col",
        default="x",
        help="X coordinate column.",
    )
    ap.add_argument(
        "--y-col",
        default="y",
        help="Y coordinate column.",
    )
    ap.add_argument(
        "--region-col",
        default="tissue_region",
        help="Region column used to derive tumor/stroma labels.",
    )
    ap.add_argument(
        "--panel-col",
        default="Panel",
        help="Panel column.",
    )
    ap.add_argument(
        "--cohort-col",
        default="cohort",
        help="Cohort column.",
    )
    return ap.parse_args()



def normalize_region(value: object) -> str:
    if pd.isna(value):
        return "Other"
    s = str(value).strip().lower()

    # stroma-like
    if s in {"str", "stroma", "stroma"}:
        return "Stroma"

    # tumor-like / epithelial-like
    if s in {"epi", "ep", "tum", "tumor", "tumour", "cancer", "neoplastic"}:
        return "Tumor"

    # other / unknown
    if s in {"other", "bg", "background", "unknown", "unk"}:
        return "Other"

    # conservative fallback: preserve the original string if not recognized
    return str(value)



def assigned_loc_from_region(norm_region: str) -> str:
    s = str(norm_region).strip().lower()
    if s == "tumor":
        return "epithelial"
    if s == "stroma":
        return "stroma"
    return "other"



def clean_state_value(value: object, negative_label: str = "checkpoint_neg") -> str:
    """Normalize AR checkpoint-state values for feature labels."""
    if pd.isna(value):
        return negative_label
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan", "na", "<na>", "null"}:
        return negative_label
    s = s.replace("+", "pos")
    s = s.replace("-", "neg")
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    s = s.replace(";", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or negative_label


def build_final_label(
    df: pd.DataFrame,
    *,
    label_col: str,
    label_mode: str,
    state_col: str,
    state_panels: Iterable[str],
    state_separator: str,
    state_negative_label: str,
    state_exempt_labels: Iterable[str],
) -> pd.Series:
    """Return the phenotype label to write into ATHENA/1NN prep.

    - column mode: returns df[label_col].
    - ar_state mode: for state_panels, returns label_col + separator + state_col;
      for all other panels, returns label_col.
    """
    base = df[label_col].astype(str).str.strip()

    if label_mode == "column":
        return base

    if label_mode != "ar_state":
        raise ValueError(f"Unknown label_mode: {label_mode}")

    state_panels = {str(x).strip().upper() for x in state_panels}
    is_state_panel = df["Panel"].astype(str).str.strip().str.upper().isin(state_panels)

    exempt = {str(x).strip() for x in state_exempt_labels}
    is_exempt_label = base.isin(exempt)

    if state_col not in df.columns:
        raise ValueError(
            f"--label-mode ar_state requires state column '{state_col}', but it is missing."
        )

    state_clean = df[state_col].map(lambda x: clean_state_value(x, state_negative_label))

    out = base.copy()
    use_state_suffix = is_state_panel & ~is_exempt_label

    out.loc[use_state_suffix] = (
        base.loc[use_state_suffix].astype(str)
        + state_separator
        + state_clean.loc[use_state_suffix].astype(str)
    )

    return out



def read_and_standardize(
    parquet_path: str,
    sample_col: str,
    coord_col: str,
    x_col: str,
    y_col: str,
    region_col: str,
    panel_col: str,
    cohort_col: str,
    label_col: str,
    exclude_panels: Iterable[str],
    include_panels: Optional[Iterable[str]],
    label_mode: str,
    state_col: str,
    state_panels: Iterable[str],
    state_separator: str,
    state_negative_label: str,
    state_exempt_labels: Iterable[str],
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path).copy()

    required = [sample_col, coord_col, x_col, y_col, region_col, panel_col, cohort_col, label_col]
    if label_mode == "ar_state":
        required.append(state_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {missing}")

    dataset_name = Path(parquet_path).stem.replace("_cell_df", "")
    df["dataset"] = dataset_name

    df = df.rename(
        columns={
            sample_col: "sample_id",
            coord_col: "coord",
            x_col: "Xcenter",
            y_col: "Ycenter",
            region_col: "tissue_region",
            panel_col: "Panel",
            cohort_col: "cohort",
        }
    )

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["coord"] = df["coord"].astype(str).str.strip()
    df["Panel"] = df["Panel"].astype(str).str.strip()
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df["Xcenter"] = pd.to_numeric(df["Xcenter"], errors="coerce")
    df["Ycenter"] = pd.to_numeric(df["Ycenter"], errors="coerce")

    include_panels_set = None
    if include_panels is not None:
        include_panels_set = {str(x).strip() for x in include_panels}
        df = df[df["Panel"].isin(include_panels_set)].copy()

    exclude_panels = {str(x).strip() for x in exclude_panels}
    df = df[~df["Panel"].isin(exclude_panels)].copy()

    df["phenotype"] = build_final_label(
        df,
        label_col=label_col,
        label_mode=label_mode,
        state_col=state_col,
        state_panels=state_panels,
        state_separator=state_separator,
        state_negative_label=state_negative_label,
        state_exempt_labels=state_exempt_labels,
    )

    df = df.dropna(subset=["sample_id", "Xcenter", "Ycenter", "Panel", "cohort", "phenotype"])
    df["phenotype"] = df["phenotype"].astype(str).str.strip()
    df = df[df["phenotype"] != ""].copy()

    df["tumor_stroma"] = df["tissue_region"].map(normalize_region)
    df["analysisregion"] = df["tumor_stroma"]
    df["assigned_loc"] = df["tumor_stroma"].map(assigned_loc_from_region)

    df["sample_name"] = df["sample_id"]
    return df



def build_1nn_prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ClusterID"] = out.groupby("sample_id").cumcount() + 1
    out["tnumber"] = out["sample_id"]
    out["sample"] = out["sample_id"]

    cols = [
        "sample_id",
        "analysisregion",
        "Xcenter",
        "Ycenter",
        "phenotype",
        "tnumber",
        "ClusterID",
        "sample",
        "assigned_loc",
        "tumor_stroma",
        "sample_name",
        "coord",
        "Panel",
        "cohort",
    ]
    return out[cols].copy()




def load_tissue_area_lookup(
    tissue_area_file: Optional[str],
    sample_col: str,
    area_col: str,
    area_units: str,
) -> dict:
    """Load optional true tissue areas and return sample_id -> area_mm2."""
    if tissue_area_file is None:
        return {}

    fp = Path(tissue_area_file)
    if not fp.exists():
        raise FileNotFoundError(f"Tissue area file not found: {fp}")

    sep = "\t" if fp.suffix.lower() in {".tsv", ".txt"} else ","
    area_df = pd.read_csv(fp, sep=sep, low_memory=False)

    missing = [c for c in [sample_col, area_col] if c not in area_df.columns]
    if missing:
        raise ValueError(f"{fp} is missing required tissue-area columns: {missing}")

    tmp = area_df[[sample_col, area_col]].copy()
    tmp[sample_col] = tmp[sample_col].astype(str).str.strip()
    tmp[area_col] = pd.to_numeric(tmp[area_col], errors="coerce")
    tmp = tmp.dropna(subset=[sample_col, area_col])
    tmp = tmp[tmp[area_col] > 0].copy()

    if area_units == "um2":
        tmp["area_mm2"] = tmp[area_col] / 1_000_000.0
    elif area_units in {"mm2", "raw"}:
        tmp["area_mm2"] = tmp[area_col]
    else:
        raise ValueError(f"Unknown tissue area units: {area_units}")

    # If duplicate samples exist, keep the first positive value.
    lookup = tmp.groupby(sample_col)["area_mm2"].first().to_dict()
    print(f"[INFO] Loaded true tissue areas for {len(lookup):,} samples from {fp}")
    return lookup


def build_tissue_prep(
    df: pd.DataFrame,
    *,
    area_lookup_mm2: Optional[dict] = None,
    bbox_area_to_mm2_factor: float = 1e-6,
) -> pd.DataFrame:
    """Build tissue_prep.tsv.

    The downstream R Step1 code expects `total_area` as the denominator for
    N.per.mm2 = N / total_area. ATHENA's Ripley code then multiplies total_area
    by 1e6 to recover um^2. Therefore this function writes total_area in mm^2.

    If `area_lookup_mm2` is supplied, it is preferred; otherwise we estimate area
    from the sample bounding box: (max_x - min_x) * (max_y - min_y) multiplied by
    `bbox_area_to_mm2_factor`.
    """
    if area_lookup_mm2 is None:
        area_lookup_mm2 = {}

    bbox = (
        df.groupby("sample_id")
        .agg(
            min_x=("Xcenter", "min"),
            max_x=("Xcenter", "max"),
            min_y=("Ycenter", "min"),
            max_y=("Ycenter", "max"),
            n_cells=("sample_id", "size"),
            sample_name=("sample_name", "first"),
            coord=("coord", "first"),
            Panel=("Panel", "first"),
            cohort=("cohort", "first"),
        )
        .reset_index()
    )

    bbox["bbox_width"] = (bbox["max_x"] - bbox["min_x"]).clip(lower=0)
    bbox["bbox_height"] = (bbox["max_y"] - bbox["min_y"]).clip(lower=0)
    bbox["bbox_area_raw"] = bbox["bbox_width"] * bbox["bbox_height"]
    bbox["bbox_area_mm2"] = bbox["bbox_area_raw"] * float(bbox_area_to_mm2_factor)

    bbox["true_area_mm2"] = bbox["sample_id"].map(area_lookup_mm2)
    bbox["area_source"] = np.where(bbox["true_area_mm2"].notna(), "tissue_area_lookup", "bbox_estimate")
    bbox["total_area"] = bbox["true_area_mm2"].combine_first(bbox["bbox_area_mm2"])

    # Safeguard for degenerate cases with one cell or flat coordinates.
    bbox["total_area"] = bbox["total_area"].mask(bbox["total_area"] <= 0, np.nan)
    bbox["total_area"] = bbox["total_area"].fillna(1e-12)

    out = bbox[[
        "sample_id",
        "total_area",
        "area_source",
        "bbox_area_raw",
        "bbox_area_mm2",
        "bbox_width",
        "bbox_height",
        "sample_name",
        "coord",
        "Panel",
        "cohort",
        "n_cells",
    ]].copy()
    return out


def chunk_sample_ids(sample_ids: List[str], chunk_size: int) -> List[List[str]]:
    return [sample_ids[i:i + chunk_size] for i in range(0, len(sample_ids), chunk_size)]



def write_chunk(
    df_chunk: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    cohort: str,
    panel: str,
    chunk_index: int,
    area_lookup_mm2: Optional[dict] = None,
    bbox_area_to_mm2_factor: float = 1e-6,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    nn = build_1nn_prep(df_chunk)
    tissue = build_tissue_prep(
        df_chunk,
        area_lookup_mm2=area_lookup_mm2,
        bbox_area_to_mm2_factor=bbox_area_to_mm2_factor,
    )

    nn.to_csv(out_dir / "1NN_prep.tsv", sep="\t", index=False)
    tissue.to_csv(out_dir / "tissue_prep.tsv", sep="\t", index=False)

    manifest = (
        df_chunk[["sample_id", "sample_name", "coord", "Panel", "cohort"]]
        .drop_duplicates()
        .sort_values(["sample_id", "coord"])
        .reset_index(drop=True)
    )
    manifest.to_csv(out_dir / "sample_manifest.tsv", sep="\t", index=False)

    return {
        "dataset": dataset,
        "cohort": cohort,
        "panel": panel,
        "chunk": f"chunk_{chunk_index:04d}",
        "n_samples": manifest["sample_id"].nunique(),
        "n_cells": len(df_chunk),
        "path": str(out_dir),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    area_lookup_mm2 = load_tissue_area_lookup(
        tissue_area_file=args.tissue_area_file,
        sample_col=args.tissue_area_sample_col,
        area_col=args.tissue_area_col,
        area_units=args.tissue_area_units,
    )

    all_frames = []
    for fp in args.inputs:
        all_frames.append(
            read_and_standardize(
                parquet_path=fp,
                sample_col=args.sample_col,
                coord_col=args.coord_col,
                x_col=args.x_col,
                y_col=args.y_col,
                region_col=args.region_col,
                panel_col=args.panel_col,
                cohort_col=args.cohort_col,
                label_col=args.label_col,
                exclude_panels=args.exclude_panels,
                include_panels=args.include_panels,
                label_mode=args.label_mode,
                state_col=args.state_col,
                state_panels=args.state_panels,
                state_separator=args.state_separator,
                state_negative_label=args.state_negative_label,
                state_exempt_labels=args.state_exempt_labels,
            )
        )

    df = pd.concat(all_frames, ignore_index=True)

    if df.empty:
        raise ValueError("No rows remain after filtering. Check inputs and excluded panels.")

    summary_rows = []

    # keep TMA and BLASST separate by dataset, then cohort, then panel
    for (dataset, cohort, panel), sub in df.groupby(["dataset", "cohort", "Panel"], sort=True):
        sample_ids = sorted(sub["sample_id"].dropna().astype(str).unique().tolist())
        if not sample_ids:
            continue

        chunks = chunk_sample_ids(sample_ids, args.chunk_size)

        for i, sample_chunk in enumerate(chunks, start=1):
            df_chunk = sub[sub["sample_id"].isin(sample_chunk)].copy()
            out_dir = out_root / dataset / cohort / panel / f"chunk_{i:04d}"
            row = write_chunk(
                df_chunk=df_chunk,
                out_dir=out_dir,
                dataset=dataset,
                cohort=cohort,
                panel=panel,
                chunk_index=i,
                area_lookup_mm2=area_lookup_mm2,
                bbox_area_to_mm2_factor=args.bbox_area_to_mm2_factor,
            )
            summary_rows.append(row)
            print(
                f"[WROTE] {out_dir} | samples={row['n_samples']} | cells={row['n_cells']}"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "chunk_summary.tsv", sep="\t", index=False)

    print("\nDone.")
    print(f"Chunk summary: {out_root / 'chunk_summary.tsv'}")


if __name__ == "__main__":
    main()
