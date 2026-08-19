#!/usr/bin/env python3
"""
Build KOLL / Florestan cell_df parquet with the same derived label columns used
by the main mIF feature-generation pipeline.

This is an adapter for KOLL-style already-collapsed CSV files. It does not use
the reviewed AR/BT phenotype-abundance dictionaries because KOLL phenotypes are
already classifier-assigned labels.

Expected input columns, by default:
    sample_name, tissue_region, phenotype, x, y, Panel, source_file

Typical use:
    python -u build_koll_cell_df.py \
      --ar_csv /projects/.../KOLL_AR_cells.csv \
      --bt_csv /projects/.../KOLL_BT_cells.csv \
      --out_dir /projects/ovcare/users/nikolay_alabi/immuno/phenotype_assignments/cell_df_rebuild

Output:
    koll_cell_df.parquet
    koll_input_phenotype_counts.csv
    koll_label_summary.csv
    koll_excluded_cell_summary.csv
    koll_run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build standardized KOLL cell_df parquet with derived label columns."
    )
    ap.add_argument("--ar_csv", default=None, help="KOLL AR-panel cell CSV.")
    ap.add_argument("--bt_csv", default=None, help="KOLL BT-panel cell CSV.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--out_file", default="koll_cell_df.parquet")
    ap.add_argument("--cohort", default="KOLL")
    ap.add_argument("--dataset_name", default="koll")

    ap.add_argument("--sample-col", default="sample_name")
    ap.add_argument("--tissue-region-col", default="tissue_region")
    ap.add_argument("--phenotype-col", default="phenotype")
    ap.add_argument("--x-col", default="x")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--panel-col", default="Panel")
    ap.add_argument("--source-file-col", default="source_file")
    ap.add_argument("--coord-col", default=None, help="Optional coord column. If absent, coord=sample_name.")

    ap.add_argument(
        "--drop-na-phenotype",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with missing/blank phenotype. Recommended for KOLL.",
    )
    ap.add_argument(
        "--other-collapse-label",
        default="stroma",
        help="Collapse label for KOLL 'other' cells. Use 'stroma' for pipeline compatibility.",
    )
    ap.add_argument(
        "--state-none-label",
        default="checkpoint_neg",
        help="State label for checkpoint-negative cells.",
    )
    ap.add_argument(
        "--state-separator",
        default="__",
        help="Separator for state labels.",
    )
    ap.add_argument(
        "--state-panels",
        nargs="+",
        default=["AR", "BT"],
        help=(
            "Panels for which label_checkpoint_state and label_compartment_state are populated. "
            "Default includes BT because KOLL BT phenotypes include PD1+ T-cell labels."
        ),
    )
    return ap.parse_args()


def read_table(fp: str | Path) -> pd.DataFrame:
    fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(fp)

    if fp.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(fp, sep="\t", low_memory=False)

    # sep=None lets pandas sniff comma vs tab for .csv-like files.
    return pd.read_csv(fp, sep=None, engine="python")


def clean_string(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def phenotype_key(x) -> str:
    s = clean_string(x).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def sanitize_label_token(x, none_label: str = "checkpoint_neg") -> str:
    if pd.isna(x):
        s = none_label
    else:
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "nan", "na", "<na>", "null"}:
            s = none_label
    s = s.replace("+", "pos").replace("-", "neg")
    s = s.replace("/", "_").replace(";", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or none_label


def make_checkpoint_binary(state_label: str) -> str:
    s = str(state_label).strip()
    if s in {"", "None", "checkpoint_neg", "nan", "NA", "<NA>"}:
        return "checkpoint_neg"
    return "checkpoint_pos"


def map_compartment(collapse_label: object) -> object:
    s = clean_string(collapse_label).lower()
    if s in {"tumor", "tumour", "cancer", "epithelial", "neoplastic"}:
        return "tumor"
    if s in {"stroma", "background", "other", "all_neg", "allneg", "fibro", "muscle"}:
        return "stroma"
    if s in {"", "nan", "none", "<na>", "unresolved", "artifact"}:
        return pd.NA
    return "immune"


def map_koll_ar(raw_pheno: object, other_label: str, neg_state: str) -> dict:
    """Map KOLL AR-style phenotypes.

    Important: proliferative labels are NOT encoded into `state`, because the
    downstream checkpoint-state feature source treats any non-negative state as
    checkpoint-positive. Proliferative tumor/immune cells are therefore collapsed
    to tumor/immune with checkpoint_neg state.
    """
    k = phenotype_key(raw_pheno)

    mapping = {
        "tumor": ("tumor", neg_state),
        "pdl1 tumor": ("tumor", "PDL1"),
        "proli tumor": ("tumor", neg_state),

        "immune": ("immune", neg_state),
        "pdl1 immune": ("immune", "PDL1"),
        "proli immune": ("immune", neg_state),

        "fibro": ("stroma", neg_state),
        "pdl1 fibro": ("stroma", "PDL1"),
        "muscle": ("stroma", neg_state),
        "other": (other_label, neg_state),
    }

    if k in mapping:
        collapse, state = mapping[k]
        return {
            "collapse_label": collapse,
            "state": state,
            "exclude_reason": pd.NA,
            "mapping_status": "mapped",
        }

    # Missing/unknown phenotypes should not silently become biology.
    if k in {"", "nan", "none", "<na>", "null"}:
        return {
            "collapse_label": pd.NA,
            "state": pd.NA,
            "exclude_reason": "missing_phenotype",
            "mapping_status": "excluded_missing_phenotype",
        }

    return {
        "collapse_label": pd.NA,
        "state": pd.NA,
        "exclude_reason": "unmapped_phenotype",
        "mapping_status": "unmapped",
    }


def map_koll_bt(raw_pheno: object, other_label: str, neg_state: str) -> dict:
    """Map KOLL BT-style phenotypes.

    The BT file does not appear to contain a tumor-cell marker. Therefore
    'other' is treated as stroma/background, and no tumor-cell labels are
    created from BT phenotypes.
    """
    k = phenotype_key(raw_pheno)

    mapping = {
        "cd8 t cells": ("cd8_t_cell", neg_state),
        "pd1+ cd8+ t cells": ("cd8_t_cell", "PD1"),

        "cd4 t cells": ("cd4_t_cell", neg_state),
        "pd1+ cd4+ t cells": ("cd4_t_cell", "PD1"),

        "dn t cells": ("t_cell", neg_state),
        "treg": ("treg_cell", neg_state),
        "macrophages": ("macrophage", neg_state),

        "other": (other_label, neg_state),
    }

    if k in mapping:
        collapse, state = mapping[k]
        return {
            "collapse_label": collapse,
            "state": state,
            "exclude_reason": pd.NA,
            "mapping_status": "mapped",
        }

    if k in {"", "nan", "none", "<na>", "null"}:
        return {
            "collapse_label": pd.NA,
            "state": pd.NA,
            "exclude_reason": "missing_phenotype",
            "mapping_status": "excluded_missing_phenotype",
        }

    return {
        "collapse_label": pd.NA,
        "state": pd.NA,
        "exclude_reason": "unmapped_phenotype",
        "mapping_status": "unmapped",
    }


def standardize_one(
    fp: str | Path,
    *,
    panel: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    df = read_table(fp).copy()
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()

    required = [
        args.sample_col,
        args.tissue_region_col,
        args.phenotype_col,
        args.x_col,
        args.y_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} is missing required columns: {missing}")

    out = pd.DataFrame()
    out["sample_name"] = df[args.sample_col].astype(str).str.strip()

    if args.coord_col and args.coord_col in df.columns:
        out["coord"] = df[args.coord_col].astype(str).str.strip()
    else:
        # KOLL does not necessarily have TMA-style [row,col] coordinates.
        # For feature generation, sample-level coord is sufficient.
        out["coord"] = out["sample_name"]

    out["x"] = pd.to_numeric(df[args.x_col], errors="coerce")
    out["y"] = pd.to_numeric(df[args.y_col], errors="coerce")
    out["tissue_region"] = df[args.tissue_region_col].astype("object")
    out["phenotype"] = df[args.phenotype_col].astype("object")
    out["marker_combination"] = out["phenotype"]
    out["phenotype_canonical"] = out["phenotype"].map(lambda x: clean_string(x))
    out["Panel"] = panel.upper()
    out["cohort"] = args.cohort

    if args.source_file_col in df.columns:
        out["source_file"] = df[args.source_file_col].astype("object")
    else:
        out["source_file"] = str(fp)

    if panel.upper() == "AR":
        mapped = out["phenotype"].map(
            lambda x: map_koll_ar(x, args.other_collapse_label, args.state_none_label)
        )
    elif panel.upper() == "BT":
        mapped = out["phenotype"].map(
            lambda x: map_koll_bt(x, args.other_collapse_label, args.state_none_label)
        )
    else:
        raise ValueError(f"Unsupported KOLL panel: {panel}")

    map_df = pd.DataFrame(list(mapped))
    out = pd.concat([out.reset_index(drop=True), map_df.reset_index(drop=True)], axis=1)

    # Remove invalid coordinates.
    bad_xy = out["x"].isna() | out["y"].isna()
    out.loc[bad_xy & out["exclude_reason"].isna(), "exclude_reason"] = "missing_xy"
    out.loc[bad_xy & out["mapping_status"].eq("mapped"), "mapping_status"] = "excluded_missing_xy"

    if args.drop_na_phenotype:
        bad_pheno = out["phenotype"].isna() | out["phenotype"].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        out.loc[bad_pheno & out["exclude_reason"].isna(), "exclude_reason"] = "missing_phenotype"
        out.loc[bad_pheno & out["mapping_status"].eq("mapped"), "mapping_status"] = "excluded_missing_phenotype"

    return out


def add_derived_label_columns(
    cell: pd.DataFrame,
    *,
    state_none_label: str,
    state_separator: str,
    state_panels: list[str],
) -> pd.DataFrame:
    out = cell.copy()

    panel = out["Panel"].astype(str).str.strip().str.upper()
    state_panels = {str(p).strip().upper() for p in state_panels}
    is_state_panel = panel.isin(state_panels)
    is_ar = panel.eq("AR")

    base = out["collapse_label"].astype(str).str.strip()
    state_clean = out["state"].map(lambda x: sanitize_label_token(x, state_none_label))

    out["label_phenotype"] = base

    # Keep this AR-only for compatibility with the existing AR_state source.
    out["label_ar_state"] = pd.NA
    out.loc[is_ar, "label_ar_state"] = (
        base.loc[is_ar].astype(str) + state_separator + state_clean.loc[is_ar].astype(str)
    )

    # KOLL BT has PD1+ CD4/CD8 labels, so checkpoint-state labels can be populated
    # for BT as well when --state-panels includes BT.
    out["label_checkpoint_state"] = pd.NA
    out.loc[is_state_panel, "label_checkpoint_state"] = state_clean.loc[is_state_panel]

    out["label_checkpoint_binary"] = pd.NA
    out.loc[is_state_panel, "label_checkpoint_binary"] = (
        out.loc[is_state_panel, "label_checkpoint_state"].map(make_checkpoint_binary)
    )

    out["label_compartment"] = out["collapse_label"].map(map_compartment)

    out["label_compartment_state"] = pd.NA
    comp = out["label_compartment"].astype(str).str.strip()
    out.loc[is_state_panel, "label_compartment_state"] = (
        comp.loc[is_state_panel].astype(str) + state_separator + state_clean.loc[is_state_panel].astype(str)
    )

    for c in DERIVED_LABEL_COLS:
        out[c] = out[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    return out


def finalize_header(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FINAL_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[FINAL_COLS].copy()
    for c in [
        "sample_name", "coord", "tissue_region", "marker_combination", "phenotype",
        "Panel", "cohort", "phenotype_canonical", "collapse_label", "state",
        *DERIVED_LABEL_COLS,
    ]:
        out[c] = out[c].astype("object")
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    input_files = {}

    if args.ar_csv:
        parts.append(standardize_one(args.ar_csv, panel="AR", args=args))
        input_files["AR"] = str(args.ar_csv)

    if args.bt_csv:
        parts.append(standardize_one(args.bt_csv, panel="BT", args=args))
        input_files["BT"] = str(args.bt_csv)

    if not parts:
        raise ValueError("At least one of --ar_csv or --bt_csv is required.")

    all_cells = pd.concat(parts, ignore_index=True, sort=False)

    # Input phenotype counts before exclusion.
    counts = (
        all_cells.groupby(["Panel", "phenotype"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["Panel", "n_cells"], ascending=[True, False])
    )
    counts.to_csv(out_dir / "koll_input_phenotype_counts.csv", index=False)

    excluded = all_cells[all_cells["exclude_reason"].notna()].copy()
    primary = all_cells[all_cells["exclude_reason"].isna()].copy()

    primary = add_derived_label_columns(
        primary,
        state_none_label=args.state_none_label,
        state_separator=args.state_separator,
        state_panels=args.state_panels,
    )

    final = finalize_header(primary)
    final.to_parquet(out_dir / args.out_file, index=False)

    if excluded.empty:
        pd.DataFrame(columns=["Panel", "phenotype", "exclude_reason", "n_cells"]).to_csv(
            out_dir / "koll_excluded_cell_summary.csv", index=False
        )
    else:
        (
            excluded.groupby(["Panel", "phenotype", "exclude_reason"], dropna=False)
            .size()
            .reset_index(name="n_cells")
            .sort_values(["Panel", "n_cells"], ascending=[True, False])
            .to_csv(out_dir / "koll_excluded_cell_summary.csv", index=False)
        )

    label_parts = []
    for label_col in ["collapse_label", "state", *DERIVED_LABEL_COLS]:
        tmp = (
            final.groupby(["Panel", label_col], dropna=False)
            .size()
            .reset_index(name="n_cells")
            .rename(columns={label_col: "label"})
        )
        tmp["label_col"] = label_col
        label_parts.append(tmp)
    label_summary = pd.concat(label_parts, ignore_index=True, sort=False)
    label_summary = label_summary[["Panel", "label_col", "label", "n_cells"]]
    label_summary.to_csv(out_dir / "koll_label_summary.csv", index=False)

    sample_summary = (
        final.groupby(["Panel", "sample_name"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["Panel", "sample_name"])
    )
    sample_summary.to_csv(out_dir / "koll_sample_cell_counts.csv", index=False)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_files": input_files,
        "out_dir": str(out_dir),
        "out_file": args.out_file,
        "cohort": args.cohort,
        "dataset_name": args.dataset_name,
        "other_collapse_label": args.other_collapse_label,
        "state_none_label": args.state_none_label,
        "state_separator": args.state_separator,
        "state_panels": args.state_panels,
        "n_cells_input": int(len(all_cells)),
        "n_cells_primary": int(len(final)),
        "n_cells_excluded": int(len(excluded)),
    }
    (out_dir / "koll_run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nDONE: KOLL cell_df built")
    print(f"Input cells:    {len(all_cells):,}")
    print(f"Primary cells:  {len(final):,}")
    print(f"Excluded cells: {len(excluded):,}")
    print(f"Output parquet: {out_dir / args.out_file}")
    print(f"Label summary:  {out_dir / 'koll_label_summary.csv'}")


if __name__ == "__main__":
    main()
