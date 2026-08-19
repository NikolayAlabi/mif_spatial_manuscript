from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns


COORD_RE = re.compile(r"\[\s*(\d{1,})\s*,\s*(\d{1,})\s*\]")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
import re
import pandas as pd
import numpy as np


def sanitize_coord_value(x):
    """
    Normalize coord values to the format 'x_y'.

    Examples:
        '[32148,9283]'  -> '32148_9283'
        '[32148, 9283]' -> '32148_9283'
        '32148,9283'    -> '32148_9283'
        '32148_9283'    -> '32148_9283'
        '(32148,9283)'  -> '32148_9283'
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s == "":
        return np.nan

    # remove brackets / parentheses / spaces
    s = s.replace("[", "").replace("]", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace(" ", "")

    # unify separators
    s = s.replace(",", "_")

    # collapse repeated underscores
    s = re.sub(r"_+", "_", s)

    # remove leading/trailing underscores
    s = s.strip("_")

    return s

def drop_old_collapse_cols(obj):
    out = dict(obj)
    cols_to_drop = ["collapse_label", "phenotype_canonical", "state"]

    for key in ["cell_df", "marker_df"]:
        if key in out and out[key] is not None:
            out[key] = out[key].drop(columns=cols_to_drop, errors="ignore")

    return out
    
def sanitize_coord_in_df(df, coord_col="coord", verbose=True, name=None):
    """
    Return a copy of df with sanitized coord column, if present.
    """
    out = df.copy()

    if coord_col in out.columns:
        before_non_null = out[coord_col].notna().sum()
        before_unique = out[coord_col].astype(str).nunique(dropna=True)

        out[coord_col] = out[coord_col].map(sanitize_coord_value)

        after_non_null = out[coord_col].notna().sum()
        after_unique = out[coord_col].astype(str).nunique(dropna=True)

        if verbose:
            label = name if name is not None else "df"
            print(
                f"[sanitize_coord] {label}: "
                f"non-null {before_non_null}->{after_non_null}, "
                f"unique {before_unique}->{after_unique}"
            )

    return out

def add_meta_to_tissue_seg(dataset_dict):
    dataset_dict["tissue_seg_df"] = dataset_dict["tissue_seg_df"].copy()
    
    meta_add = (
        dataset_dict["meta_df"][["coord", "Panel", "cohort", "TURBT_or_RC"]]
        .drop_duplicates()
        .copy()
    )
    
    # drop old versions first if they exist
    dataset_dict["tissue_seg_df"] = dataset_dict["tissue_seg_df"].drop(
        columns=["Panel", "cohort", "TURBT_or_RC"],
        errors="ignore"
    )
    
    dataset_dict["tissue_seg_df"] = dataset_dict["tissue_seg_df"].merge(
        meta_add,
        on="coord",
        how="left"
    )
    
    return dataset_dict

def sanitize_coord_in_dataset_dict(dataset_dict, coord_col="coord", verbose=True, prefix="dataset"):
    """
    Apply coord sanitization to every dataframe value inside a dict like tma or blasst.
    Non-dataframe values are passed through unchanged.
    """
    out = {}

    for key, value in dataset_dict.items():
        if isinstance(value, pd.DataFrame):
            out[key] = sanitize_coord_in_df(
                value,
                coord_col=coord_col,
                verbose=verbose,
                name=f"{prefix}['{key}']"
            )
        else:
            out[key] = value

    return out

def extract_coord_token(series: pd.Series) -> pd.Series:
    """Extract standardized bracket coordinates like '[12345,6789]' from a Series."""
    m = series.astype(str).str.extract(COORD_RE)
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    ok = m[0].notna() & m[1].notna()
    out.loc[ok] = "[" + m.loc[ok, 0].astype(str) + "," + m.loc[ok, 1].astype(str) + "]"
    return out


def extract_coords(name: object) -> Optional[str]:
    """Extract coordinates as '12345_6789' from strings containing '[12345,6789]'."""
    if pd.isna(name):
        return None
    s = str(name)
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m2 = re.search(r"[Xx]?(\d+)[_\-][Yy]?(\d+)", s)
    if m2:
        return f"{m2.group(1)}_{m2.group(2)}"
    return None


def parse_core_idx(core_idx_val: object) -> tuple[object, object]:
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


def infer_coord_column(df: pd.DataFrame) -> pd.Series:
    if "coord" in df.columns:
        return extract_coord_token(df["coord"])
    for col in ["sample_name", "image", "Core", "core", "file", "filename", "Slide", "slide", "path", "filepath"]:
        if col in df.columns:
            c = extract_coord_token(df[col])
            if c.notna().any():
                return c
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    combo = df[obj_cols].astype(str).agg(" | ".join, axis=1)
    return extract_coord_token(combo)


def _find_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_coord_from_sample(df: pd.DataFrame, sample_col: str = "sample_name") -> pd.DataFrame:
    out = df.copy()
    if "coord" not in out.columns:
        if sample_col in out.columns:
            out["coord"] = extract_coord_token(out[sample_col])
        else:
            out["coord"] = infer_coord_column(out)
    else:
        out["coord"] = extract_coord_token(out["coord"])
    return out


def _standardize_cell_columns(
    df: pd.DataFrame,
    *,
    sample_col: str = "sample_name",
    marker_col: str = "phenotype_combined",
    tissue_col: Optional[str] = None,
) -> pd.DataFrame:
    """Rename likely raw parquet columns into stable analysis names where possible."""
    out = df.copy()

    x_col = _find_first_existing(out, ["Cell X Position", "Cell_X_Position", "Xcenter", "x", "X", "CellXPosition"])
    y_col = _find_first_existing(out, ["Cell Y Position", "Cell_Y_Position", "Ycenter", "y", "Y", "CellYPosition"])
    tissue_guess = tissue_col or _find_first_existing(out, ["tissue_category", "tissue_region", "Tissue Category", "analysisregion", "region", "Region"])
    phenotype_guess = _find_first_existing(out, ["phenotype", "phenotype_combined", "celltype", "cell_type"])

    rename_map = {}
    if sample_col in out.columns and sample_col != "sample_name":
        rename_map[sample_col] = "sample_name"
    if marker_col in out.columns and marker_col != "marker_combination":
        rename_map[marker_col] = "marker_combination"
    if phenotype_guess and phenotype_guess != "phenotype":
        rename_map[phenotype_guess] = "phenotype"
    if x_col and x_col != "x":
        rename_map[x_col] = "x"
    if y_col and y_col != "y":
        rename_map[y_col] = "y"
    if tissue_guess and tissue_guess != "tissue_region":
        rename_map[tissue_guess] = "tissue_region"

    out = out.rename(columns=rename_map)

    if "sample_name" not in out.columns and sample_col in df.columns:
        out["sample_name"] = df[sample_col]
    if "marker_combination" not in out.columns and marker_col in df.columns:
        out["marker_combination"] = df[marker_col]
    if "phenotype" not in out.columns and "marker_combination" in out.columns:
        out["phenotype"] = out["marker_combination"].apply(clean_pheno_combo)
    if "tissue_region" not in out.columns:
        out["tissue_region"] = pd.NA

    out = _ensure_coord_from_sample(out, sample_col="sample_name")
    return out


# -----------------------------------------------------------------------------
# Phenotype / panel / cohort helpers
# -----------------------------------------------------------------------------

def infer_panel_from_path(fp: Path) -> Optional[str]:
    s = str(fp).upper()
    if "ARP" in s or re.search(r"(^|[^A-Z0-9])AR($|[^A-Z0-9])", s):
        return "AR"
    if "B&T" in s or "B+T" in s or re.search(r"(^|[^A-Z0-9])BT($|[^A-Z0-9])", s):
        return "BT"
    if "MYELOID" in s or re.search(r"(_M($|[^A-Z0-9]))", s):
        return "MY"
    return None


def infer_panel_from_filename(path: str | os.PathLike[str]) -> str:
    b = os.path.basename(str(path))
    if re.search(r"(^|_)AR(_|\.parquet$)", b, flags=re.IGNORECASE):
        return "AR"
    if re.search(r"(^|_)B&T(_|\.parquet$)", b, flags=re.IGNORECASE):
        return "BT"
    if re.search(r"(^|_)Myeloid(_|\.parquet$)", b, flags=re.IGNORECASE):
        return "MY"
    if "myeloid" in b.lower():
        return "MY"
    if "b&t" in b.lower() or "b_t" in b.lower() or "bt" in b.lower():
        return "BT"
    if re.search(r"(^|[_\-\s])ar([_\-\s]|$)", b.lower()):
        return "AR"
    return "UNK"


COHORT_MAP = {
    "BCA2020": "NAC2020",
    "NO-NAC": "No-NAC",
    "PURE01": "PURE01",
    "BLADDER": "NAC2015",
}


def infer_cohort_from_sample(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper()
    raw = s.str.extract(r"^(BCA(?:_| )?2020|NO-NAC|PURE01|BLADDER)\b", expand=False)
    raw = raw.str.replace("BCA_2020", "BCA2020", regex=False)
    raw = raw.str.replace("BCA 2020", "BCA2020", regex=False)
    return raw.map(COHORT_MAP).fillna(raw)


def clean_pheno_combo(x: object) -> str:
    if pd.isna(x):
        return "ALL_NEG"
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return "ALL_NEG"
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    pos = [p for p in parts if p.endswith("+")]
    if len(pos) == 0:
        return "ALL_NEG"
    return ";".join(sorted(set(pos)))


# -----------------------------------------------------------------------------
# TMA cohorts
# -----------------------------------------------------------------------------

def load_tma_metadata(
    clinical_csv: str | os.PathLike[str],
    *,
    core_col: str = "Core",
    core_idx_col: str = "core_idx",
) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv, low_memory=False).copy()
    if core_col not in clin.columns:
        raise ValueError(f"Expected column '{core_col}' in clinical CSV: {clinical_csv}")
    clin["coord"] = extract_coord_token(clin[core_col].astype(str))
    if core_idx_col in clin.columns:
        ct = clin[core_idx_col].apply(parse_core_idx)
        if "cohort" not in clin.columns:
            clin["cohort"] = [x[0] for x in ct]
        if "tma" not in clin.columns:
            clin["tma"] = [x[1] for x in ct]
    return clin


def load_tma_qc(
    qc_dir: str | os.PathLike[str],
    *,
    pattern: str = "*review.csv",
    image_col: str = "image",
) -> pd.DataFrame:
    qc_files = sorted(Path(qc_dir).glob(pattern))
    qc_parts: list[pd.DataFrame] = []
    for fp in qc_files:
        df = pd.read_csv(fp, low_memory=False)
        df["__qc_file"] = fp.name
        qc_parts.append(df)
    if not qc_parts:
        return pd.DataFrame(columns=["coord", "structural_acceptability", "segmentation_comments", "__qc_file"])
    qc = pd.concat(qc_parts, ignore_index=True)
    if image_col not in qc.columns:
        raise ValueError(f"QC review CSVs must contain '{image_col}' to extract coords.")
    qc["coord"] = extract_coord_token(qc[image_col].astype(str))
    qc = qc[qc["coord"].notna()].copy()
    has_sa = qc["structural_acceptability"].notna().astype(int) if "structural_acceptability" in qc.columns else 0
    qc["__has_sa"] = has_sa
    qc = qc.sort_values(["coord", "__has_sa"], ascending=[True, False]).drop_duplicates("coord")
    return qc.drop(columns=["__has_sa"])


def build_tma_qc_metadata_df(meta_df: pd.DataFrame, qc_df: pd.DataFrame) -> pd.DataFrame:
    return meta_df.merge(qc_df, on="coord", how="outer", suffixes=("", "_qc")).drop_duplicates()


def load_tma_tissue_segmentation(tissue_seg_csv: str | os.PathLike[str]) -> pd.DataFrame:
    seg = pd.read_csv(tissue_seg_csv, low_memory=False).copy()
    if "sample_name" in seg.columns:
        seg["coord"] = extract_coord_token(seg["sample_name"].astype(str))
    else:
        seg["coord"] = infer_coord_column(seg)
    return seg


def build_tma_cell_dataframe(
    combined_root: str | os.PathLike[str],
    meta_df: Optional[pd.DataFrame] = None,
    qc_df: Optional[pd.DataFrame] = None,
    *,
    sample_col: str = "sample_name",
    marker_col: str = "phenotype_combined",
    tissue_col: Optional[str] = None,
    exclude_structural_acceptability: Optional[Iterable[str]] = ("unusable",),
    include_all_neg: bool = True,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Build a lean TMA cell-level dataframe.

    Final columns kept:
        ["sample_name", "coord", "x", "y", "tissue_region",
         "marker_combination", "phenotype", "Panel", "cohort"]

    Notes
    -----
    - qc_df is only used to determine coords to exclude
    - meta_df is only used to backfill missing Panel / cohort if needed
    - no metadata / QC columns are merged into the final cell_df
    """
    root = Path(combined_root)
    parq_files = sorted(root.rglob("*.parquet") if recursive else root.glob("*.parquet"))
    if not parq_files:
        raise ValueError(f"No parquet files found under {combined_root}")

    # ------------------------------------------------------------
    # 1. Build exclusion set from QC
    # ------------------------------------------------------------
    exclude_coords: set[str] = set()
    if (
        qc_df is not None
        and exclude_structural_acceptability is not None
        and "structural_acceptability" in qc_df.columns
        and "coord" in qc_df.columns
    ):
        exclude_vals = {str(x).strip().lower() for x in exclude_structural_acceptability}
        exclude_coords = set(
            qc_df.loc[
                qc_df["structural_acceptability"].astype(str).str.strip().str.lower().isin(exclude_vals),
                "coord",
            ]
            .dropna()
            .astype(str)
            .unique()
        )

    # ------------------------------------------------------------
    # 2. Minimal metadata only for Panel / cohort backfill
    # ------------------------------------------------------------
    meta_small = None
    if meta_df is not None and "coord" in meta_df.columns:
        meta_keep = [c for c in ["coord", "Panel", "cohort"] if c in meta_df.columns]
        if meta_keep:
            meta_small = meta_df[meta_keep].drop_duplicates("coord").copy()

    # ------------------------------------------------------------
    # 3. Read parquet files
    # ------------------------------------------------------------
    parts: list[pd.DataFrame] = []
    for fp in parq_files:
        df = pd.read_parquet(fp)

        if sample_col not in df.columns or marker_col not in df.columns:
            continue

        df = _standardize_cell_columns(
            df,
            sample_col=sample_col,
            marker_col=marker_col,
            tissue_col=tissue_col,
        )

        df = df[df["coord"].notna()].copy()

        # exclude unusable etc.
        if exclude_coords:
            df = df[~df["coord"].astype(str).isin(exclude_coords)].copy()

        if df.empty:
            continue

        # infer or backfill Panel / cohort
        if "Panel" not in df.columns:
            df["Panel"] = infer_panel_from_path(fp)
        if "cohort" not in df.columns:
            df["cohort"] = infer_cohort_from_sample(df["sample_name"])

        if meta_small is not None:
            df = df.merge(meta_small, on="coord", how="left", suffixes=("", "_meta"))

            if "Panel_meta" in df.columns:
                df["Panel"] = df["Panel"].fillna(df["Panel_meta"])
                df = df.drop(columns=["Panel_meta"])

            if "cohort_meta" in df.columns:
                df["cohort"] = df["cohort"].fillna(df["cohort_meta"])
                df = df.drop(columns=["cohort_meta"])

        # harmonize marker / phenotype
        if "marker_combination" in df.columns:
            df["marker_combination"] = df["marker_combination"].astype(str)

        if "phenotype" not in df.columns and "marker_combination" in df.columns:
            df["phenotype"] = df["marker_combination"].apply(clean_pheno_combo)

        if not include_all_neg and "phenotype" in df.columns:
            df = df[df["phenotype"] != "ALL_NEG"].copy()

        if df.empty:
            continue

        # keep only lean core columns
        keep_cols = [
            c for c in [
                "sample_name",
                "coord",
                "x",
                "y",
                "tissue_region",
                "marker_combination",
                "phenotype",
                "Panel",
                "cohort",
            ]
            if c in df.columns
        ]

        df = df[keep_cols].copy()
        parts.append(df)

    # ------------------------------------------------------------
    # 4. Return empty template if nothing found
    # ------------------------------------------------------------
    final_cols = [
        "sample_name",
        "coord",
        "x",
        "y",
        "tissue_region",
        "marker_combination",
        "phenotype",
        "Panel",
        "cohort",
    ]

    if not parts:
        return pd.DataFrame(columns=final_cols)

    cell_df = pd.concat(parts, ignore_index=True)

    # ensure consistent column order
    ordered = [c for c in final_cols if c in cell_df.columns] + [c for c in cell_df.columns if c not in final_cols]
    return cell_df[ordered]


def build_marker_dataframe(
    cell_df: pd.DataFrame,
    *,
    group_col: str = "phenotype",
    tissue_col: str = "tissue_region",
    add_all_region: bool = True,
) -> pd.DataFrame:
    """
    Aggregate cell_df to coord-level marker/phenotype summaries.

    By default, this counts cells separately within each tissue_region
    and also adds a synthetic region called 'All' representing the total
    across all regions for that core.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Cell-level dataframe.
    group_col : str
        Column to summarize by, usually 'phenotype' or 'marker_combination'.
    tissue_col : str
        Tissue-region column, usually 'tissue_region'.
    add_all_region : bool
        If True, add an additional region called 'All' containing the
        total counts across all tissue regions.

    Returns
    -------
    pd.DataFrame
        Aggregated marker dataframe with:
          - n_cells
          - core_total
          - prop
    """
    if "coord" not in cell_df.columns:
        raise ValueError("cell_df must contain 'coord'.")
    if group_col not in cell_df.columns:
        raise ValueError(f"cell_df must contain '{group_col}'.")

    df = cell_df.copy()

    # base grouping columns
    base_group = [c for c in ["Panel", "cohort", "coord"] if c in df.columns]
    extra_group = [c for c in ["TURBT_or_RC", "tma", "structural_acceptability", "sample_name"] if c in df.columns]

    marker_parts = []

    # ------------------------------------------------------------
    # 1. Region-specific counts
    # ------------------------------------------------------------
    if tissue_col in df.columns:
        group_cols_region = base_group + extra_group + [tissue_col, group_col]
        marker_region = (
            df.groupby(group_cols_region, dropna=False)
              .size()
              .reset_index(name="n_cells")
        )
        marker_parts.append(marker_region)

        # --------------------------------------------------------
        # 2. Whole-core counts across all tissue regions
        # --------------------------------------------------------
        if add_all_region:
            marker_all = df.copy()
            marker_all[tissue_col] = "All"

            group_cols_all = base_group + extra_group + [tissue_col, group_col]
            marker_all = (
                marker_all.groupby(group_cols_all, dropna=False)
                          .size()
                          .reset_index(name="n_cells")
            )
            marker_parts.append(marker_all)

    else:
        # fallback: no tissue column available
        group_cols = base_group + extra_group + [group_col]
        marker_df = (
            df.groupby(group_cols, dropna=False)
              .size()
              .reset_index(name="n_cells")
        )
        marker_parts.append(marker_df)

    marker_df = pd.concat(marker_parts, ignore_index=True, sort=False)

    # ------------------------------------------------------------
    # 3. Compute totals and proportions within each region
    # ------------------------------------------------------------
    total_group = [c for c in ["Panel", "coord", tissue_col] if c in marker_df.columns]
    if not total_group:
        total_group = ["coord"]

    marker_df["core_total"] = (
        marker_df.groupby(total_group, dropna=False)["n_cells"]
                 .transform("sum")
    )
    marker_df["prop"] = marker_df["n_cells"] / marker_df["core_total"]

    return marker_df


# -----------------------------------------------------------------------------
# BLASST / whole sections
# -----------------------------------------------------------------------------

def load_blasst_metadata(metadata_csv: str | os.PathLike[str], *, core_col: str = "Core") -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, low_memory=False).copy()
    if core_col not in meta.columns:
        raise ValueError(f"Metadata missing required column: {core_col}")
    meta["coord"] = meta[core_col].apply(extract_coords)
    return meta


def load_blasst_tissue_segmentation(tissue_seg_csv: str | os.PathLike[str]) -> pd.DataFrame:
    seg = pd.read_csv(tissue_seg_csv, low_memory=False).copy()
    if "sample_name" in seg.columns:
        seg["coord"] = seg["sample_name"].apply(extract_coords)
    else:
        seg["coord"] = pd.NA
    return seg


def build_blasst_cell_dataframe(
    parquet_dir: str | os.PathLike[str],
    meta_df: Optional[pd.DataFrame] = None,
    *,
    sample_col: str = "sample_name",
    marker_col: str = "phenotype_combined",
    tissue_col: str = "tissue_category",
    recursive: bool = False,
    include_all_neg: bool = True,
) -> pd.DataFrame:
    root = Path(parquet_dir)
    parquet_files = sorted(root.rglob("*.parquet") if recursive else root.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    meta_small = None
    if meta_df is not None:
        meta_keep = [c for c in meta_df.columns if c in ["coord", "Core", "Sample_ID_Adjusted", "Sample_ID", "TURBT_or_RC", "patient_id", "case_id"]]
        meta_small = meta_df[meta_keep].drop_duplicates("coord") if "coord" in meta_keep else None

    parts: list[pd.DataFrame] = []
    for fp in parquet_files:
        df = pd.read_parquet(fp)
        if sample_col not in df.columns or marker_col not in df.columns:
            continue
        df = _standardize_cell_columns(df, sample_col=sample_col, marker_col=marker_col, tissue_col=tissue_col)
        df = df[df["coord"].notna()].copy()
        if df.empty:
            continue
        df["Panel"] = infer_panel_from_filename(fp)
        if "cohort" not in df.columns:
            df["cohort"] = "BLASST"
        if not include_all_neg and "phenotype" in df.columns:
            df = df[df["phenotype"] != "ALL_NEG"].copy()
        if df.empty:
            continue
        if meta_small is not None:
            df = df.merge(meta_small, on="coord", how="left")
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["sample_name", "coord", "x", "y", "tissue_region", "marker_combination", "phenotype", "Panel", "cohort"])

    cell_df = pd.concat(parts, ignore_index=True)
    preferred = ["sample_name", "coord", "x", "y", "tissue_region", "marker_combination", "phenotype", "Panel", "cohort", "Sample_ID_Adjusted", "Sample_ID", "TURBT_or_RC"]
    ordered = [c for c in preferred if c in cell_df.columns] + [c for c in cell_df.columns if c not in preferred]
    return cell_df[ordered]


# -----------------------------------------------------------------------------
# Alignment report
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np


def report_alignment(
    cell_df: pd.DataFrame,
    marker_df: pd.DataFrame | None = None,
    meta_df: pd.DataFrame | None = None,
    qc_df: pd.DataFrame | None = None,
    tissue_seg_df: pd.DataFrame | None = None,
    name: str = "dataset",
    panel_col: str = "Panel",
    cohort_col: str = "cohort",
    coord_col: str = "coord",
    show_full: bool = True,
) -> pd.DataFrame:
    """
    Print a compact alignment report by Panel x cohort.

    For each Panel/cohort group in cell_df, reports:
      - unique coords in cell_df
      - how many of those coords are present in meta_df
      - how many are present in qc_df
      - how many are present in tissue_seg_df

    Works for:
      - TMA: pass cell_df, meta_df, qc_df, tissue_seg_df
      - BLASST: pass cell_df, meta_df, tissue_seg_df (qc_df can be None)

    Returns
    -------
    pd.DataFrame
        Summary table used for printing.
    """

    def _require_col(df: pd.DataFrame | None, df_name: str, col: str) -> None:
        if df is not None and col not in df.columns:
            raise ValueError(f"{df_name} must contain column '{col}'. Found: {list(df.columns)}")

    def _norm_coord_set(df: pd.DataFrame | None, col: str) -> set[str]:
        if df is None or col not in df.columns:
            return set()
        s = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        s = s[s != ""]
        return set(s.unique())

    def _norm_group_val(series: pd.Series) -> pd.Series:
        # avoid mixed dtype sort issues
        out = series.copy()
        out = out.where(~out.isna(), "NA")
        out = out.astype(str).str.strip()
        out = out.replace({"": "NA", "nan": "NA", "None": "NA", "<NA>": "NA"})
        return out

    def _fmt_count_pct(n: int, d: int) -> str:
        if d == 0:
            return "0 (0.0%)"
        return f"{n} ({100.0*n/d:.1f}%)"

    _require_col(cell_df, "cell_df", coord_col)
    _require_col(cell_df, "cell_df", cohort_col)
    _require_col(cell_df, "cell_df", panel_col)
    _require_col(meta_df, "meta_df", coord_col)
    _require_col(qc_df, "qc_df", coord_col)
    _require_col(tissue_seg_df, "tissue_seg_df", coord_col)

    df = cell_df.copy()
    df[panel_col] = _norm_group_val(df[panel_col])
    df[cohort_col] = _norm_group_val(df[cohort_col])
    df[coord_col] = _norm_group_val(df[coord_col])

    meta_coords = _norm_coord_set(meta_df, coord_col)
    qc_coords = _norm_coord_set(qc_df, coord_col)
    tissue_coords = _norm_coord_set(tissue_seg_df, coord_col)

    rows = []

    grouped = (
        df[[panel_col, cohort_col, coord_col]]
        .drop_duplicates()
        .groupby([panel_col, cohort_col], dropna=False)
    )

    for (panel, cohort), sub in grouped:
        coords = set(sub[coord_col].dropna().astype(str))
        n_cell = len(coords)

        n_meta = len(coords & meta_coords) if meta_df is not None else 0
        n_qc = len(coords & qc_coords) if qc_df is not None else 0
        n_tissue = len(coords & tissue_coords) if tissue_seg_df is not None else 0

        rows.append(
            {
                "Panel": panel,
                "cohort": cohort,
                "cell_df_unique_coords": n_cell,
                "meta_match": _fmt_count_pct(n_meta, n_cell) if meta_df is not None else "NA",
                "qc_match": _fmt_count_pct(n_qc, n_cell) if qc_df is not None else "NA",
                "tissue_seg_match": _fmt_count_pct(n_tissue, n_cell) if tissue_seg_df is not None else "NA",
                "_meta_n": n_meta,
                "_qc_n": n_qc,
                "_tissue_n": n_tissue,
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        print(f"\n=== ALIGNMENT REPORT: {name} ===")
        print("No Panel/cohort groups found in cell_df.")
        return summary

    summary = summary.sort_values(["Panel", "cohort"], kind="stable").reset_index(drop=True)

    print(f"\n=== ALIGNMENT REPORT: {name} ===")
    print("Per Panel x cohort, how many unique coords in cell_df are found in each source:\n")
    print(summary[["Panel", "cohort", "cell_df_unique_coords", "meta_match", "qc_match", "tissue_seg_match"]].to_string(index=False))

    # optional quick pivot view
    if show_full:
        print("\n--- Meta match (%) quick view ---")
        meta_pivot = summary.pivot(index="cohort", columns="Panel", values="meta_match")
        print(meta_pivot.fillna("NA").to_string())

        if qc_df is not None:
            print("\n--- QC match (%) quick view ---")
            qc_pivot = summary.pivot(index="cohort", columns="Panel", values="qc_match")
            print(qc_pivot.fillna("NA").to_string())

        if tissue_seg_df is not None:
            print("\n--- Tissue segmentation match (%) quick view ---")
            tissue_pivot = summary.pivot(index="cohort", columns="Panel", values="tissue_seg_match")
            print(tissue_pivot.fillna("NA").to_string())

    # global totals
    all_cell_coords = _norm_coord_set(cell_df, coord_col)
    print("\n--- Overall unique coord coverage ---")
    print(f"cell_df:      {len(all_cell_coords):,}")
    if meta_df is not None:
        print(f"meta_df:      {len(meta_coords):,} | matched to cell_df: {_fmt_count_pct(len(all_cell_coords & meta_coords), len(all_cell_coords))}")
    if qc_df is not None:
        print(f"qc_df:        {len(qc_coords):,} | matched to cell_df: {_fmt_count_pct(len(all_cell_coords & qc_coords), len(all_cell_coords))}")
    if tissue_seg_df is not None:
        print(f"tissue_seg_df:{len(tissue_coords):,} | matched to cell_df: {_fmt_count_pct(len(all_cell_coords & tissue_coords), len(all_cell_coords))}")

    return summary

# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

def build_tma_analysis_inputs(
    *,
    combined_root: str | os.PathLike[str],
    clinical_csv: str | os.PathLike[str],
    qc_dir: str | os.PathLike[str],
    tissue_seg_csv: Optional[str | os.PathLike[str]] = None,
    include_all_neg: bool = True,
    sample_col: str = "sample_name",
    marker_col: str = "phenotype_combined",
    tissue_col: Optional[str] = None,
) -> dict[str, pd.DataFrame | None]:
    meta_df = load_tma_metadata(clinical_csv)
    qc_df = load_tma_qc(qc_dir)
    qc_meta_df = build_tma_qc_metadata_df(meta_df, qc_df)
    cell_df = build_tma_cell_dataframe(
        combined_root=combined_root,
        meta_df=meta_df,
        qc_df=qc_df,
        sample_col=sample_col,
        marker_col=marker_col,
        tissue_col=tissue_col,
        include_all_neg=include_all_neg,
    )
    marker_df = build_marker_dataframe(cell_df, group_col="phenotype")
    tissue_seg_df = load_tma_tissue_segmentation(tissue_seg_csv) if tissue_seg_csv is not None else None
    return {
        "cell_df": cell_df,
        "marker_df": marker_df,
        "meta_df": meta_df,
        "qc_df": qc_df,
        "qc_meta_df": qc_meta_df,
        "tissue_seg_df": tissue_seg_df,
    }


def build_blasst_analysis_inputs(
    *,
    parquet_dir: str | os.PathLike[str],
    metadata_csv: str | os.PathLike[str],
    tissue_seg_csv: Optional[str | os.PathLike[str]] = None,
    include_all_neg: bool = True,
    sample_col: str = "sample_name",
    marker_col: str = "phenotype_combined",
    tissue_col: str = "tissue_category",
) -> dict[str, pd.DataFrame | None]:
    meta_df = load_blasst_metadata(metadata_csv)
    cell_df = build_blasst_cell_dataframe(
        parquet_dir=parquet_dir,
        meta_df=meta_df,
        sample_col=sample_col,
        marker_col=marker_col,
        tissue_col=tissue_col,
        include_all_neg=include_all_neg,
    )
    marker_df = build_marker_dataframe(cell_df, group_col="phenotype")
    tissue_seg_df = load_blasst_tissue_segmentation(tissue_seg_csv) if tissue_seg_csv is not None else None
    blasst_out = {
        "cell_df": cell_df,
        "marker_df": marker_df,
        "meta_df": meta_df,
        "qc_df": None,
        "tissue_seg_df": tissue_seg_df,
    }

    blasst_out = add_blasst_sample_type_to_dataframes(blasst_out)
    return blasst_out

import pandas as pd


def normalize_coord_format(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        return s
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2:
            return f"[{parts[0]},{parts[1]}]"
    return s


def add_blasst_sample_type_to_dataframes(
    blasst_dict: dict,
    sample_type_candidates=("TURBT_or_RC", "tma", "sample_type", "specimen_type"),
) -> dict:
    """
    Standardize BLASST sample type as TURBT_or_RC across all returned dataframes.

    Expected blasst_dict keys:
      - meta_df
      - optionally cell_df, marker_df, tissue_seg_df

    Returns
    -------
    dict
        Updated dictionary with TURBT_or_RC harmonized across dataframes.
    """
    out = dict(blasst_dict)

    if "meta_df" not in out or out["meta_df"] is None:
        raise ValueError("blasst_dict must contain 'meta_df'.")

    meta_df = out["meta_df"].copy()

    if "coord" not in meta_df.columns:
        raise ValueError("blasst['meta_df'] must contain 'coord'.")

    # find source sample-type column
    sample_type_col = None
    for c in sample_type_candidates:
        if c in meta_df.columns:
            sample_type_col = c
            break

    if sample_type_col is None:
        raise ValueError(
            "Could not find a BLASST sample-type column in meta_df. "
            f"Tried: {sample_type_candidates}"
        )

    meta_df["coord_norm"] = meta_df["coord"].apply(normalize_coord_format)

    # standardize sample type name
    if sample_type_col != "TURBT_or_RC":
        meta_df["TURBT_or_RC"] = meta_df[sample_type_col]
    else:
        meta_df["TURBT_or_RC"] = meta_df["TURBT_or_RC"]

    meta_df_small = (
        meta_df[["coord_norm", "TURBT_or_RC"]]
        .drop_duplicates("coord_norm")
        .copy()
    )

    def _merge_sample_type(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        cur = df.copy()
        if "coord" not in cur.columns:
            return cur

        cur["coord_norm"] = cur["coord"].apply(normalize_coord_format)
        cur = cur.merge(
            meta_df_small.rename(columns={"TURBT_or_RC": "TURBT_or_RC_meta"}),
            on="coord_norm",
            how="left",
        )

        if "TURBT_or_RC" in cur.columns:
            cur["TURBT_or_RC"] = cur["TURBT_or_RC"].fillna(cur["TURBT_or_RC_meta"])
        else:
            cur["TURBT_or_RC"] = cur["TURBT_or_RC_meta"]

        cur = cur.drop(columns=["TURBT_or_RC_meta"])
        return cur

    # update all relevant dataframes
    out["meta_df"] = _merge_sample_type(meta_df)
    for key in ["cell_df", "marker_df", "tissue_seg_df", "ph_long", "core_qc_df"]:
        if key in out:
            out[key] = _merge_sample_type(out[key])

    return out

from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict
import pandas as pd


def safe_read_csv(fp):
    """
    Read CSV robustly across common encodings.
    """
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read file with known encodings: {fp}")


def canonicalize_marker_combo(x):
    """
    Convert a marker combination string into a canonical form by:
      - splitting on ';'
      - removing negative markers ending in '-'
      - sorting remaining markers alphabetically
      - joining back with ';'

    IMPORTANT:
    If all markers are negative and get removed, return 'ALL_NEG'
    rather than NA.

    Examples
    --------
    'PD1+;PanCK+;PDL1-' -> 'PD1+;PanCK+'
    'CD8+;CD3+;PD1-'    -> 'CD3+;CD8+'
    'PD1-'              -> 'ALL_NEG'
    'PD1-;PDL1-'        -> 'ALL_NEG'
    """
    if pd.isna(x):
        return "ALL_NEG"

    parts = [p.strip() for p in str(x).split(";") if str(p).strip() != ""]
    parts = [p for p in parts if not p.endswith("-")]
    parts = sorted(parts)

    if len(parts) == 0:
        return "ALL_NEG"

    return ";".join(parts)


def load_normalized_collapse_map(
    phenotype_assignments_dir: str | Path,
    panels: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all normalized phenotype CSVs and build a canonical annotation map.

    Returns
    -------
    norm_map_df : pd.DataFrame
        Raw combined normalized map with columns including:
          Panel, phenotype, collapse_label, state, phenotype_canonical
    norm_map_canonical : pd.DataFrame
        Deduplicated canonical map with columns:
          Panel, phenotype_canonical, collapse_label, state
    """
    root = Path(phenotype_assignments_dir)
    norm_files = sorted(root.glob("*_phenotype_abundance_consistency_normalized.csv"))
    if not norm_files:
        raise ValueError(f"No normalized phenotype CSVs found under: {root}")

    if panels is not None:
        panels = {str(p).upper() for p in panels}

    norm_parts = []
    for fp in norm_files:
        panel = fp.name.split("_")[0].upper()

        if panels is not None and panel not in panels:
            continue

        df = safe_read_csv(fp)
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

        required = {"phenotype", "collapse_label"}
        if not required.issubset(df.columns):
            continue

        keep_cols = [c for c in ["phenotype", "collapse_label", "state"] if c in df.columns]
        df = df[keep_cols].copy()
        df["Panel"] = panel
        norm_parts.append(df)

    if not norm_parts:
        raise ValueError("No normalized phenotype files with phenotype/collapse_label were loaded.")

    norm_map_df = pd.concat(norm_parts, ignore_index=True, sort=False)
    norm_map_df = norm_map_df.drop_duplicates(subset=["Panel", "phenotype"]).copy()
    norm_map_df["phenotype_canonical"] = norm_map_df["phenotype"].apply(canonicalize_marker_combo)

    dups = norm_map_df.duplicated(subset=["Panel", "phenotype_canonical"], keep=False)
    if verbose and dups.any():
        print("\nWARNING: duplicate canonical phenotypes in normalized map:")
        print(
            norm_map_df.loc[dups, ["Panel", "phenotype", "phenotype_canonical", "collapse_label"] + ([ "state"] if "state" in norm_map_df.columns else [])]
            .sort_values(["Panel", "phenotype_canonical", "phenotype"])
            .to_string(index=False)
        )

    keep_cols = ["Panel", "phenotype_canonical", "collapse_label"]
    if "state" in norm_map_df.columns:
        keep_cols.append("state")

    norm_map_canonical = (
        norm_map_df
        .dropna(subset=["phenotype_canonical"])
        .drop_duplicates(subset=["Panel", "phenotype_canonical"])
        [keep_cols]
        .copy()
    )

    return norm_map_df, norm_map_canonical


def add_collapse_labels_to_df(
    df: pd.DataFrame,
    norm_map_canonical: pd.DataFrame,
    *,
    phenotype_col: str = "phenotype",
    panel_col: str = "Panel",
    unmapped_to_all_neg: bool = False,
    coerce_special_to_all_neg: bool = False,
) -> pd.DataFrame:
    """
    Add phenotype_canonical, collapse_label, and state to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Panel and phenotype-like column.
    norm_map_canonical : pd.DataFrame
        Output from load_normalized_collapse_map().
    phenotype_col : str
        Column to canonicalize and map. Usually 'phenotype'.
    panel_col : str
        Panel column. Usually 'Panel'.
    unmapped_to_all_neg : bool
        If True, set unmapped collapse_label values to 'ALL_NEG'.
    coerce_special_to_all_neg : bool
        If True, convert 'mixed_lineage', 'unresolved', and 'artifact'
        collapse_label values to 'ALL_NEG'.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with:
          - phenotype_canonical
          - collapse_label
          - state   (if present in normalized map)
    """
    if panel_col not in df.columns:
        raise ValueError(f"Input dataframe must contain '{panel_col}'.")
    if phenotype_col not in df.columns:
        raise ValueError(f"Input dataframe must contain '{phenotype_col}'.")

    out = df.copy()
    out["phenotype_canonical"] = out[phenotype_col].apply(canonicalize_marker_combo)

    merge_cols = ["Panel", "phenotype_canonical", "collapse_label"]
    if "state" in norm_map_canonical.columns:
        merge_cols.append("state")

    out = out.merge(
        norm_map_canonical[merge_cols],
        on=[panel_col, "phenotype_canonical"],
        how="left",
    )

    if unmapped_to_all_neg:
        out["collapse_label"] = out["collapse_label"].fillna("ALL_NEG")

    if coerce_special_to_all_neg:
        out["collapse_label"] = out["collapse_label"].replace({
            "mixed_lineage": "ALL_NEG",
            "unresolved": "ALL_NEG",
            "artifact": "ALL_NEG",
        })

    # For all-negative canonical phenotype, force collapse_label/state if missing
    out.loc[out["phenotype_canonical"] == "ALL_NEG", "collapse_label"] = (
        out.loc[out["phenotype_canonical"] == "ALL_NEG", "collapse_label"].fillna("ALL_NEG")
    )

    return out


def add_collapse_labels_to_object(
    obj: Dict[str, pd.DataFrame],
    phenotype_assignments_dir: str | Path,
    *,
    panels: Optional[Iterable[str]] = ("AR", "BT"),
    cell_df_key: str = "cell_df",
    marker_df_key: str = "marker_df",
    cell_phenotype_col: str = "phenotype",
    marker_phenotype_col: str = "phenotype",
    unmapped_to_all_neg: bool = False,
    coerce_special_to_all_neg: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add collapse-label annotations to cell_df and marker_df inside a built object
    such as tma or blasst.

    This adds:
      - phenotype_canonical
      - collapse_label

    It also stores:
      - normalized_map_df
      - normalized_map_canonical

    Parameters
    ----------
    obj : dict
        Dictionary like tma or blasst containing at least cell_df / marker_df.
    phenotype_assignments_dir : str or Path
        Directory containing normalized phenotype CSVs.
    panels : iterable
        Panels to use for mapping, e.g. ('AR', 'BT').
    unmapped_to_all_neg : bool
        If True, assign ALL_NEG to unmapped phenotypes.
    coerce_special_to_all_neg : bool
        If True, force mixed_lineage / unresolved / artifact to ALL_NEG.

    Returns
    -------
    dict
        Updated copy of obj.
    """
    out = dict(obj)

    norm_map_df, norm_map_canonical = load_normalized_collapse_map(
        phenotype_assignments_dir=phenotype_assignments_dir,
        panels=panels,
        verbose=verbose,
    )

    if cell_df_key in out and out[cell_df_key] is not None:
        out[cell_df_key] = add_collapse_labels_to_df(
            out[cell_df_key],
            norm_map_canonical=norm_map_canonical,
            phenotype_col=cell_phenotype_col,
            panel_col="Panel",
            unmapped_to_all_neg=unmapped_to_all_neg,
            coerce_special_to_all_neg=coerce_special_to_all_neg,
        )

    if marker_df_key in out and out[marker_df_key] is not None:
        out[marker_df_key] = add_collapse_labels_to_df(
            out[marker_df_key],
            norm_map_canonical=norm_map_canonical,
            phenotype_col=marker_phenotype_col,
            panel_col="Panel",
            unmapped_to_all_neg=unmapped_to_all_neg,
            coerce_special_to_all_neg=coerce_special_to_all_neg,
        )

    out["normalized_map_df"] = norm_map_df
    out["normalized_map_canonical"] = norm_map_canonical

    return out