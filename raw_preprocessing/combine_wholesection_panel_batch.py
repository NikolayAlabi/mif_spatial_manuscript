#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

KEY_COLS = ["sample_name", "tissue_category", "cell_id", "x", "y"]
ORDER_COLS = ["cell_id", "x", "y", "tissue_category"]


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w\+\-& ]+", "_", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s


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
        out = out[out["sample_name"].notna() & (out["sample_name"].astype(str).str.strip() != "")]

        return out if len(out) else None
    except Exception:
        return None


# ----------------------------
# Combine logic
# ----------------------------
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


def _combine_tokens(values: List[str], all_negative_label: str):
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
# Whole section file discovery
# ----------------------------
# def discover_files_panel_batch(root_whole: Path, panel: str, batch_num: str) -> List[Path]:
#     """
#     Find all *_cell_seg_data.txt under Whole Sections for a given panel+batch.

#     Expected structure:
#       Whole Sections/<Panel inForm on whole sections>/Batch N/<algo folder>/*_cell_seg_data.txt
#     """
#     panel_l = panel.lower()
#     batch_tag = f"batch {batch_num}".lower()

#     out = []
#     for fp in root_whole.rglob("*cell_seg_data.txt"):
#         nm = fp.name.lower()
#         if "summary" in nm or "rejected" in nm:
#             continue

#         p = str(fp).lower()

#         # batch filter
#         if f"/batch {batch_num}/" not in p and f"\\batch {batch_num}\\" not in p:
#             continue

#         # panel folder filter (robust-ish)
#         if panel_l == "myeloid":
#             if "myeloid inform on whole sections" not in p:
#                 continue
#         elif panel_l == "ar":
#             if "ar inform on whole sections" not in p:
#                 continue
#         elif panel_l in ["b&t", "bt", "b_t"]:
#             if "b&t inform on whole sections" not in p and "b&amp;t inform on whole sections" not in p:
#                 continue
#         else:
#             continue

#         out.append(fp)

#     return sorted(out, key=lambda x: str(x).lower())

def discover_files_panel_batch(root_whole: Path, panel: str, batch_num: str) -> List[Path]:
    """
    Find all *_cell_seg_data.txt under Whole Sections for a given panel+batch.

    Accepts batch folder variants:
      Batch1, Batch 1, Batch_1 (case-insensitive)
    """
    panel_l = panel.lower()
    batch_num = str(batch_num).strip()

    # accept Batch1 / Batch 1 / Batch_1 (and case variants)
    batch_patterns = [
        f"/batch{batch_num}/",
        f"/batch {batch_num}/",
        f"/batch_{batch_num}/",
        f"\\batch{batch_num}\\",
        f"\\batch {batch_num}\\",
        f"\\batch_{batch_num}\\",
    ]

    out = []
    for fp in root_whole.rglob("*_cell_seg_data.txt"):
        nm = fp.name.lower()
        if "summary" in nm or "rejected" in nm:
            continue

        p = str(fp).lower()

        # batch filter
        if not any(bpat in p for bpat in batch_patterns):
            continue

        # panel filter (robust substring match)
        if panel_l == "myeloid":
            if "myeloid inform on whole sections" not in p:
                continue
        elif panel_l == "ar":
            if "ar inform on whole sections" not in p:
                continue
        elif panel_l in ["b&t", "bt", "b_t"]:
            # handle literal "b&t" in paths
            if "b&t inform on whole sections" not in p:
                continue
        else:
            continue

        out.append(fp)

    return sorted(out, key=lambda x: str(x).lower())

def region_id_from_filename(fp: Path) -> str:
    """
    Whole sections "region id" is filename base without _cell_seg_data.txt
    e.g. "18SF-5218-19_[38165,7440]"
    """
    return fp.name[:-len("_cell_seg_data.txt")]


def build_region_map(files: List[Path]) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for fp in files:
        rid = region_id_from_filename(fp)
        m.setdefault(rid, []).append(fp)
    for rid in list(m.keys()):
        m[rid] = sorted(m[rid], key=lambda p: str(p).lower())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_whole", required=True)
    ap.add_argument("--panel", required=True, choices=["AR", "B&T", "Myeloid"])
    ap.add_argument("--batch", required=True, choices=["1", "2", "3"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--part_size_regions", type=int, default=25)
    ap.add_argument("--all_negative_label", default="ALL_NEG")

    ap.add_argument("--drop_blank_cells", action="store_true")
    ap.add_argument("--keep_blank_cells", action="store_true")
    ap.add_argument("--skip_if_done", action="store_true")

    ap.add_argument("--final_parquet", default="", help="Optional: merge parts into one parquet at this path")


    args = ap.parse_args()

    drop_blank = args.drop_blank_cells and not args.keep_blank_cells
    if (not args.drop_blank_cells) and (not args.keep_blank_cells):
        drop_blank = True

    root_whole = Path(args.root_whole)
    panel = args.panel
    batch = args.batch

    out_dir = Path(args.out_dir) / f"WholeSections_{sanitize_name(panel)}_Batch{batch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("part-*.parquet"))
    if args.skip_if_done and len(existing) > 0:
        print(f"[skip] found {len(existing)} parquet parts in {out_dir}")
        return

    files = discover_files_panel_batch(root_whole, panel=panel, batch_num=batch)
    print(f"[discover] panel={panel} batch={batch} files={len(files)}")

    reg_map = build_region_map(files)
    reg_ids = sorted(reg_map.keys())
    print(f"[group] panel={panel} batch={batch} regions={len(reg_ids)} out_dir={out_dir}")

    part_idx = 0
    buffer: List[pd.DataFrame] = []

    n_rowwise = 0
    n_keywise = 0
    n_ok = 0
    n_fail = 0

    for i, rid in enumerate(reg_ids, 1):
        fps = reg_map[rid]

        dfs = []
        for fp in fps:
            d = load_cell_seg_minimal(fp)
            if d is not None and len(d) > 0:
                dfs.append(d)

        if len(dfs) == 0:
            n_fail += 1
            continue
        n_ok += 1

        if can_rowwise_merge(dfs):
            out = rowwise_combine(dfs, drop_blank=drop_blank, all_negative_label=args.all_negative_label)
            n_rowwise += 1
        else:
            out = safe_key_combine(pd.concat(dfs, ignore_index=True), drop_blank=drop_blank, all_negative_label=args.all_negative_label)
            n_keywise += 1

        out["region_id"] = rid
        out["panel"] = panel
        out["batch"] = f"Batch {batch}"
        buffer.append(out)

        if (len(buffer) >= args.part_size_regions) or (i == len(reg_ids)):
            part_path = out_dir / f"part-{part_idx:05d}.parquet"
            df_part = pd.concat(buffer, ignore_index=True)
            df_part.to_parquet(part_path, index=False)
            print(f"[write] part={part_idx:05d} regions_in_part={len(buffer)} rows={len(df_part):,} -> {part_path}")
            buffer = []
            part_idx += 1

    print("--------------------------------------------------")
    print(f"DONE panel={panel} batch={batch}")
    print("regions_ok:  ", n_ok)
    print("regions_fail:", n_fail)
    print("rowwise:     ", n_rowwise)
    print("keywise:     ", n_keywise)
    print("parts:       ", part_idx)
    print("out_dir:     ", out_dir)
    print("--------------------------------------------------")

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