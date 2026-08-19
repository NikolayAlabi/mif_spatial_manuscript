#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Cohorts (canonical list)
# ----------------------------
# COHORTS = [
#   "BCA 2020 RC ARP",
#   "BCA2020 TURBT NAC ARP",
#   "No-NAC TURBT TMA1 ARP",
#   "No-NAC TURBT TMA2 ARP",
#   "PURE01 TMA1 Pre NAC ARP",
#   "PURE01 TMA2 Pre NAC ARP",
#   "PURE01 TMA3 Pre + Post NAC ARP",

#   "BCA 2020 RC B&T",
#   "BCA2020 TURBT NAC B&T",
#   "No-NAC TURBT TMA1  B+T",
#   "No-NAC TURBT TMA2  B+T",
#   "PURE01 TMA1 Pre NAC  B+T",
#   "PURE01 TMA2 Pre NAC  B+T",
#   "PURE01 TMA3 Pre + Post NAC  B+T",

#   "BCA 2020 RC Myeloid",
#   "BCA2020 TURBT NAC  Myeloid",
#   "No-NAC TURBT TMA1  Myeloid",
#   "No-NAC TURBT TMA2  Myeloid",
#   "PURE01 TMA1 Pre NAC  Myeloid",
#   "PURE01 TMA2 Pre NAC  Myeloid",
#   "PURE01 TMA3 Pre + Post NAC  Myeloid",

#   "Bladder 19_AR",
#   "Bladder 26_AR",
#   "Bladder 19_BT",
#   "Bladder 26_BT",
#   "Bladder 19_M",
#   "Bladder 26_M",
# ]
COHORTS = [
  "No-NAC TURBT TMA1  B+T",
  "No-NAC TURBT TMA2  B+T",
  "PURE01 TMA1 Pre NAC  B+T",
  "PURE01 TMA2 Pre NAC  B+T",
  "PURE01 TMA3 Pre + Post NAC  B+T"
]

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# Canonical normalized cohort names for matching
CANON = [_norm_ws(c) for c in COHORTS]


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w\+\-& ]+", "_", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s


def parse_prefix_from_filename(fp: Path) -> Optional[str]:
    """
    Infer cohort prefix from filename only (no file reading).

    Example:
      "Bladder 19_AR_Core[2,1,A]_[... ]_cell_seg_data.txt" -> "Bladder 19_AR"
      "BCA 2020 RC ARP (2)_Core[1,1,1]_[...]_cell_seg_data.txt" -> "BCA 2020 RC ARP"
    """
    name = fp.name
    if not name.endswith("_cell_seg_data.txt"):
        return None
    if "_Core[" not in name:
        return None

    prefix = name.split("_Core[", 1)[0]
    prefix = _norm_ws(prefix)

    # Strip trailing run tags like " (2)" or "(3)"
    prefix = re.sub(r"\s*\(\d+\)\s*$", "", prefix)
    prefix = _norm_ws(prefix)

    # Match against canonical cohort list by startswith
    # (handles extra suffixes occasionally)
    for c in CANON:
        if prefix.startswith(c):
            return c

    return None


def core_key_from_filename(fp: Path) -> Optional[str]:
    """
    Core identifier for grouping marker files of the same core.
    Use the full base name without the trailing _cell_seg_data.txt.
    """
    name = fp.name
    if not name.endswith("_cell_seg_data.txt"):
        return None
    return name[:-len("_cell_seg_data.txt")]


# ----------------------------
# TSV reading + minimal load
# ----------------------------
def _standardize_cols(cols):
    out = []
    for c in cols:
        c2 = re.sub(r"\s+", " ", str(c)).strip()
        c2 = c2.replace("\ufeff", "")
        out.append(c2)
    return out


def _find_col(df_cols, targets):
    norm = {c.lower(): c for c in df_cols}
    for t in targets:
        key = _norm_ws(t).lower()
        if key in norm:
            return norm[key]
    return None


def _read_tsv_like(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        raise ValueError("Empty file")
    return pd.read_csv(path, sep="\t", engine="python", dtype=str)


def extract_algorithm_id(fp: Path) -> str:
    return fp.parent.name


def load_cell_seg_minimal(fp: Path) -> Optional[pd.DataFrame]:
    """
    Load minimal fields required for per-cell merging.
    Keeps x/y as-is (no rounding).
    """
    try:
        df = _read_tsv_like(fp)
        if df is None or df.shape[0] == 0:
            return None

        df.columns = _standardize_cols(df.columns)

        c_sample = _find_col(df.columns, ["Sample Name", "SampleName"])
        c_tissue = _find_col(df.columns, ["Tissue Category", "TissueCategory"])
        c_pheno  = _find_col(df.columns, ["Phenotype"])
        c_id     = _find_col(df.columns, ["Cell ID", "CellID"])
        c_x      = _find_col(df.columns, ["Cell X Position", "Cell X Position ", "Cell X"])
        c_y      = _find_col(df.columns, ["Cell Y Position", "Cell Y Position ", "Cell Y"])

        if any(v is None for v in [c_sample, c_tissue, c_pheno, c_id, c_x, c_y]):
            return None

        out = df[[c_sample, c_tissue, c_pheno, c_id, c_x, c_y]].copy()
        out = out.rename(columns={
            c_sample: "sample_name",
            c_tissue: "tissue_category",
            c_pheno: "phenotype",
            c_id: "cell_id",
            c_x: "x",
            c_y: "y",
        })

        out["sample_name"] = out["sample_name"].astype(str)
        out["tissue_category"] = out["tissue_category"].astype(str)
        out["cell_id"] = out["cell_id"].astype(str).str.strip()

        out["x"] = pd.to_numeric(out["x"], errors="coerce")
        out["y"] = pd.to_numeric(out["y"], errors="coerce")

        out["phenotype"] = out["phenotype"].fillna("").astype(str).str.strip()

        out["algorithm_id"] = extract_algorithm_id(fp)
        out["source_file"] = str(fp)

        out = out.dropna(subset=["x", "y"])
        out = out[out["cell_id"].notna() & (out["cell_id"] != "")]
        out = out[out["sample_name"].notna() & (_norm_ws(str(out["sample_name"].iloc[0])) != "")]

        return out if len(out) else None

    except Exception:
        return None


# ----------------------------
# Combining logic (fast path with verification, else safe path)
# ----------------------------
KEY_COLS = ["sample_name", "tissue_category", "cell_id", "x", "y"]
ORDER_COLS = ["cell_id", "x", "y", "tissue_category"]


def can_rowwise_merge(dfs: List[pd.DataFrame]) -> bool:
    if len(dfs) <= 1:
        return True
    n0 = len(dfs[0])
    for d in dfs[1:]:
        if len(d) != n0:
            return False

    base = dfs[0][ORDER_COLS].reset_index(drop=True)
    for d in dfs[1:]:
        other = d[ORDER_COLS].reset_index(drop=True)
        if not base.equals(other):
            return False
    return True


def _combine_tokens(values: List[str], all_negative_label: str) -> Tuple[Optional[str], bool, int]:
    """
    returns (phenotype_combined or None if drop, any_blank, n_positive_markers)
    """
    vals = [(v if v is not None else "") for v in values]
    vals = [str(v).strip() for v in vals]

    any_blank = any(v == "" for v in vals)

    toks: List[str] = []
    for v in vals:
        if not v:
            continue
        toks.extend([t.strip() for t in v.split(";") if t.strip()])

    toks = [t for t in toks if t != "Marker-"]
    if len(toks) == 0:
        return all_negative_label, any_blank, 0

    toks = sorted(set(toks))
    return ";".join(toks), any_blank, len(toks)


def rowwise_combine(dfs: List[pd.DataFrame], drop_blank: bool, all_negative_label: str) -> pd.DataFrame:
    base = dfs[0][KEY_COLS].copy()

    ph = pd.concat([d["phenotype"].fillna("").astype(str).str.strip() for d in dfs], axis=1)

    combined = []
    any_blank_list = []
    npos_list = []

    for row in ph.itertuples(index=False):
        out, any_blank, npos = _combine_tokens(list(row), all_negative_label)
        combined.append(out)
        any_blank_list.append(any_blank)
        npos_list.append(npos)

    base["any_blank_call"] = any_blank_list
    base["phenotype_combined"] = combined
    base["n_positive_markers"] = npos_list

    if drop_blank:
        base = base[~base["any_blank_call"]].copy()

    return base


def safe_key_combine(df_long: pd.DataFrame, drop_blank: bool, all_negative_label: str) -> pd.DataFrame:
    d = df_long.copy()
    d["phenotype"] = d["phenotype"].fillna("").astype(str).str.strip()

    # any blank per key
    blank = (
        d.assign(_blank=(d["phenotype"] == ""))
         .groupby(KEY_COLS, as_index=False)["_blank"]
         .any()
         .rename(columns={"_blank": "any_blank_call"})
    )

    tok = d.loc[d["phenotype"] != "", KEY_COLS + ["phenotype"]].copy()
    if not tok.empty:
        tok["phenotype"] = tok["phenotype"].str.split(";")
        tok = tok.explode("phenotype")
        tok["phenotype"] = tok["phenotype"].astype(str).str.strip()
        tok = tok[(tok["phenotype"] != "") & (tok["phenotype"] != "Marker-")]

    if tok.empty:
        out = blank.copy()
        out["phenotype_combined"] = all_negative_label
        out["n_positive_markers"] = 0
    else:
        markers = tok.groupby(KEY_COLS)["phenotype"].apply(lambda s: sorted(set(s.tolist()))).reset_index()
        markers["phenotype_combined"] = markers["phenotype"].apply(
            lambda lst: ";".join(lst) if len(lst) else all_negative_label
        )
        markers["n_positive_markers"] = markers["phenotype"].apply(lambda lst: len(lst))
        markers = markers.drop(columns=["phenotype"])

        out = blank.merge(markers, on=KEY_COLS, how="left")
        out["n_positive_markers"] = out["n_positive_markers"].fillna(0).astype(int)
        out["phenotype_combined"] = out["phenotype_combined"].fillna(all_negative_label)

    if drop_blank:
        out = out[out["any_blank_call"] == False].copy()

    return out


# ----------------------------
# IO: discover + process core-by-core + write partitions
# ----------------------------
def discover_files_for_cohort(root_dir: Path, cohort_prefix: str) -> Dict[str, List[Path]]:
    """
    Returns dict: core_key -> list of file paths for that core, for this cohort.
    Uses filename prefix only (no TSV peek).
    """
    cohort_norm = _norm_ws(cohort_prefix)
    core_map: Dict[str, List[Path]] = {}

    for fp in root_dir.rglob("*cell_seg_data*.txt"):
        name = fp.name.lower()
        if "summary" in name or "rejected" in name:
            continue
        if not name.endswith("cell_seg_data.txt"):
            continue

        inferred = parse_prefix_from_filename(fp)
        if inferred != cohort_norm:
            continue

        ck = core_key_from_filename(fp)
        if ck is None:
            continue
        core_map.setdefault(ck, []).append(fp)

    # deterministic order within core
    for ck in list(core_map.keys()):
        core_map[ck] = sorted(core_map[ck], key=lambda p: str(p).lower())

    return core_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--cohort", required=True, help="One cohort string (exact from the list)")

    ap.add_argument("--out_dir", required=True, help="Output directory (will write partitioned parquet here)")
    ap.add_argument("--part_size_cores", type=int, default=50, help="How many cores per output parquet part")

    ap.add_argument("--drop_blank_cells", action="store_true")
    ap.add_argument("--keep_blank_cells", action="store_true")
    ap.add_argument("--all_negative_label", default="ALL_NEG")

    ap.add_argument("--final_parquet", default="", help="Optional: merge parts into a single parquet file at this path")
    ap.add_argument("--skip_if_done", action="store_true", help="Skip if out_dir already has parts")

    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    cohort_norm = _norm_ws(args.cohort)

    if cohort_norm not in CANON:
        raise SystemExit(f"Unknown cohort: {args.cohort!r}. Must be one of the COHORTS list.")

    out_dir = Path(args.out_dir) / sanitize_name(cohort_norm)
    out_dir.mkdir(parents=True, exist_ok=True)

    # skip if already done
    existing_parts = sorted(out_dir.glob("part-*.parquet"))
    if args.skip_if_done and len(existing_parts) > 0:
        print(f"[skip] found {len(existing_parts)} existing parts in {out_dir}")
        return

    drop_blank = args.drop_blank_cells and not args.keep_blank_cells
    if (not args.drop_blank_cells) and (not args.keep_blank_cells):
        drop_blank = True

    core_map = discover_files_for_cohort(root_dir, cohort_norm)
    core_keys = sorted(core_map.keys())

    print("--------------------------------------------------")
    print("root_dir:   ", root_dir)
    print("cohort:     ", cohort_norm)
    print("n_cores:    ", len(core_keys))
    print("out_dir:    ", out_dir)
    print("drop_blank: ", drop_blank)
    print("part_size:  ", args.part_size_cores, "cores/part")
    print("--------------------------------------------------")

    part_idx = 0
    buffer: List[pd.DataFrame] = []

    n_rowwise = 0
    n_keywise = 0
    n_cores_ok = 0
    n_cores_fail = 0

    for i, ck in enumerate(core_keys, 1):
        fps = core_map[ck]

        dfs = []
        for fp in fps:
            d = load_cell_seg_minimal(fp)
            if d is not None and len(d) > 0:
                dfs.append(d)

        if len(dfs) == 0:
            n_cores_fail += 1
            continue

        n_cores_ok += 1

        if can_rowwise_merge(dfs):
            out = rowwise_combine(dfs, drop_blank=drop_blank, all_negative_label=args.all_negative_label)
            n_rowwise += 1
        else:
            df_long = pd.concat(dfs, ignore_index=True)
            out = safe_key_combine(df_long, drop_blank=drop_blank, all_negative_label=args.all_negative_label)
            n_keywise += 1

        # Add core identifier for sanity/debug
        out["core_key"] = ck
        buffer.append(out)

        # Flush per part_size_cores
        if (len(buffer) >= args.part_size_cores) or (i == len(core_keys)):
            part_path = out_dir / f"part-{part_idx:05d}.parquet"
            df_part = pd.concat(buffer, ignore_index=True)
            df_part.to_parquet(part_path, index=False)

            print(f"[write] part={part_idx:05d} cores_in_part={len(buffer)} rows={len(df_part):,} -> {part_path}")
            buffer = []
            part_idx += 1

    print("--------------------------------------------------")
    print("DONE cohort:", cohort_norm)
    print("cores_ok:   ", n_cores_ok)
    print("cores_fail: ", n_cores_fail)
    print("rowwise:    ", n_rowwise)
    print("keywise:    ", n_keywise)
    print("parts:      ", part_idx)
    print("out_dir:    ", out_dir)
    print("--------------------------------------------------")

    # Optional: merge parts -> single parquet
    if args.final_parquet:
        final_path = Path(args.final_parquet)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        parts = sorted(out_dir.glob("part-*.parquet"))
        if len(parts) == 0:
            raise SystemExit("No parts found to merge.")

        df_all = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df_all.to_parquet(final_path, index=False)
        print(f"[merge] wrote final parquet rows={len(df_all):,} -> {final_path}")


if __name__ == "__main__":
    main()