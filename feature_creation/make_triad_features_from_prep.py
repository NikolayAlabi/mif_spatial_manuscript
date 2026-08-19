#!/usr/bin/env python3
"""
Compute triad / motif features directly from 1NN_prep.tsv files.

This script is intended to sit beside the Weibull/ATHENA prep pipeline:

    prep_weibull_inputs_v3_patched.py
        -> run_reviewed_phenotype_only/**/1NN_prep.tsv
        -> run_reviewed_AR_state/**/1NN_prep.tsv

It scans one or more prep roots, reads every 1NN_prep.tsv, and writes a
sample/core-level feature table. It does not use tissue area and does not
compute densities.

Default motif definition
------------------------
For a centre phenotype C and two neighbour phenotypes A/B, the centered triad
count is the number of C cells that have at least one A neighbour and at least
one B neighbour within the radius threshold. A/B are distinct by default.

This is deliberately separate from the simple abundance/ratio feature script.

Examples
--------
Phenotype-only run:

python -u make_triad_features_from_prep.py \
  --prep-roots /path/to/run_reviewed_phenotype_only \
  --outdir /path/to/triad_features/phenotype_only \
  --outfile triad_features_phenotype_only.csv \
  --include-panels AR BT \
  --center-regex macrophage \
  --threshold 100 \
  --regions All Tumor Stroma

AR-state run:

python -u make_triad_features_from_prep.py \
  --prep-roots /path/to/run_reviewed_AR_state \
  --outdir /path/to/triad_features/AR_state \
  --outfile triad_features_AR_state.csv \
  --include-panels AR \
  --center-regex macrophage \
  --threshold 100 \
  --regions All Tumor Stroma
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_EXCLUDE_LABELS = [
    "ALL_NEG",
    "Marker-",
    "Unknown",
    "Other",
    "StromaCell",
    "artifact",
    "unresolved",
    "mixed_lineage",
]

META_COLS = [
    "dataset",
    "cohort",
    "Panel",
    "chunk",
    "sample_id",
    "sample_name",
    "coord",
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute centered triad/motif features from 1NN_prep.tsv files."
    )
    ap.add_argument(
        "--prep-roots",
        nargs="+",
        required=True,
        help="One or more roots containing chunked 1NN_prep.tsv files.",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )
    ap.add_argument(
        "--outfile",
        default="triad_features_from_prep.csv",
        help="Output CSV filename.",
    )
    ap.add_argument(
        "--include-panels",
        nargs="*",
        default=None,
        help="Panels to include. If omitted, all panels are included.",
    )
    ap.add_argument(
        "--exclude-panels",
        nargs="*",
        default=None,
        help="Panels to exclude.",
    )
    ap.add_argument(
        "--regions",
        nargs="+",
        default=["All", "Tumor", "Stroma"],
        help="Regions to analyze. Typical: All Tumor Stroma.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Radius threshold in coordinate units, usually microns.",
    )
    ap.add_argument(
        "--center-labels",
        nargs="*",
        default=None,
        help="Explicit centre phenotypes. If omitted, --center-regex is used; if both omitted, all labels are possible centres.",
    )
    ap.add_argument(
        "--center-regex",
        default=None,
        help="Regex used to select centre phenotypes, e.g. 'macrophage'. Case-insensitive.",
    )
    ap.add_argument(
        "--neighbor-labels",
        nargs="*",
        default=None,
        help="Explicit neighbour phenotype universe. If omitted, all non-excluded labels are used.",
    )
    ap.add_argument(
        "--exclude-labels",
        nargs="*",
        default=DEFAULT_EXCLUDE_LABELS,
        help="Labels excluded from centre/neighbor motif definitions.",
    )
    ap.add_argument(
        "--allow-center-as-neighbor",
        action="store_true",
        help="Allow centre phenotype to also appear as one of the neighbour phenotypes.",
    )
    ap.add_argument(
        "--allow-same-neighbor-type",
        action="store_true",
        help="Also compute motifs like C with A/A neighbours. Default uses distinct A/B neighbour types only.",
    )
    ap.add_argument(
        "--min-cells-sample-region",
        type=int,
        default=20,
        help="Skip sample-region slices with fewer cells than this.",
    )
    ap.add_argument(
        "--min-center-cells",
        type=int,
        default=1,
        help="Do not compute centre-normalized fractions if a centre has fewer cells than this.",
    )
    ap.add_argument(
        "--write-long",
        action="store_true",
        help="Also write a long-format motif table.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Debug option: process at most this many 1NN_prep.tsv files.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sanitize_label(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "EMPTY"


def norm_panel(x: object) -> str:
    s = str(x).strip()
    su = s.upper()
    if su in {"ARP", "AR"}:
        return "AR"
    if su in {"BT", "B&T", "BTP"}:
        return "BT"
    if su in {"MY", "M", "MYELOID"}:
        return "MY"
    return s


def normalize_region(x: object) -> str:
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()
    if s in {"tumor", "tumour", "epi", "epithelial", "epithelium", "cancer", "neoplastic"}:
        return "Tumor"
    if s in {"stroma", "stromal", "str"}:
        return "Stroma"
    return "Other"


def region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    r = str(region).strip().lower()
    if r == "all":
        return df[df["analysisregion_norm"].isin(["Tumor", "Stroma"])].copy()
    if r in {"tumor", "epi", "epithelial"}:
        return df[df["analysisregion_norm"].eq("Tumor")].copy()
    if r in {"stroma", "stromal", "str"}:
        return df[df["analysisregion_norm"].eq("Stroma")].copy()
    return df[df["analysisregion_norm"].eq(region)].copy()


def infer_path_metadata(prep_file: Path) -> Dict[str, str]:
    """
    Expected layout from prep script:
      root / dataset / cohort / panel / chunk_xxxx / 1NN_prep.tsv
    """
    chunk = prep_file.parent.name
    panel = prep_file.parent.parent.name if len(prep_file.parents) >= 2 else "Unknown"
    cohort = prep_file.parent.parent.parent.name if len(prep_file.parents) >= 3 else "Unknown"
    dataset = prep_file.parent.parent.parent.parent.name if len(prep_file.parents) >= 4 else "Unknown"
    return {
        "dataset": dataset,
        "cohort": cohort,
        "Panel": norm_panel(panel),
        "chunk": chunk,
    }


def find_prep_files(roots: Sequence[str], max_files: Optional[int] = None) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.name == "1NN_prep.tsv":
            files.append(root_path)
        else:
            files.extend(sorted(root_path.rglob("1NN_prep.tsv")))
    files = sorted(set(files))
    if max_files is not None:
        files = files[:max_files]
    return files


def label_is_excluded(label: str, exclude_labels: Iterable[str]) -> bool:
    lab = str(label).strip().lower()
    excluded = {str(x).strip().lower() for x in exclude_labels}
    return lab in excluded


def choose_centers_and_neighbors(
    labels: Sequence[str],
    center_labels: Optional[Sequence[str]],
    center_regex: Optional[str],
    neighbor_labels: Optional[Sequence[str]],
    exclude_labels: Sequence[str],
) -> Tuple[List[str], List[str]]:
    labels = sorted({str(x).strip() for x in labels if str(x).strip()})
    labels = [x for x in labels if not label_is_excluded(x, exclude_labels)]

    if neighbor_labels:
        neighbors = [str(x).strip() for x in neighbor_labels]
        neighbors = [x for x in neighbors if x in labels and not label_is_excluded(x, exclude_labels)]
    else:
        neighbors = labels.copy()

    if center_labels:
        centers = [str(x).strip() for x in center_labels]
        centers = [x for x in centers if x in labels and not label_is_excluded(x, exclude_labels)]
    elif center_regex:
        pat = re.compile(center_regex, flags=re.IGNORECASE)
        centers = [x for x in labels if pat.search(x)]
    else:
        centers = labels.copy()

    return sorted(set(centers)), sorted(set(neighbors))


def neighbor_pairs_for_center(
    center: str,
    neighbors: Sequence[str],
    allow_center_as_neighbor: bool,
    allow_same_neighbor_type: bool,
) -> List[Tuple[str, str]]:
    nbs = list(neighbors)
    if not allow_center_as_neighbor:
        nbs = [x for x in nbs if x != center]

    if allow_same_neighbor_type:
        return list(itertools.combinations_with_replacement(nbs, 2))
    return list(itertools.combinations(nbs, 2))


def build_radius_neighbor_label_counts(df: pd.DataFrame, threshold: float) -> List[Dict[str, int]]:
    """
    Return per-cell dictionaries of neighbour phenotype counts within threshold.
    """
    coords = df[["Xcenter", "Ycenter"]].to_numpy(dtype=float)
    phenos = df["phenotype"].astype(str).to_numpy()
    n = len(df)
    counts: List[Dict[str, int]] = [dict() for _ in range(n)]

    if n < 2:
        return counts

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(threshold), output_type="ndarray")

    for i, j in pairs:
        pi = phenos[i]
        pj = phenos[j]
        counts[i][pj] = counts[i].get(pj, 0) + 1
        counts[j][pi] = counts[j].get(pi, 0) + 1

    return counts


def compute_centered_triads_for_region(
    df_region: pd.DataFrame,
    region_name: str,
    threshold: float,
    center_labels: Optional[Sequence[str]],
    center_regex: Optional[str],
    neighbor_labels: Optional[Sequence[str]],
    exclude_labels: Sequence[str],
    allow_center_as_neighbor: bool,
    allow_same_neighbor_type: bool,
    min_center_cells: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Compute centered triad features for one sample-region.
    """
    out: Dict[str, float] = {}
    long_rows: List[Dict[str, object]] = []

    if df_region.empty:
        return out, long_rows

    labels_present = sorted(df_region["phenotype"].dropna().astype(str).unique().tolist())
    centers, neighbors = choose_centers_and_neighbors(
        labels=labels_present,
        center_labels=center_labels,
        center_regex=center_regex,
        neighbor_labels=neighbor_labels,
        exclude_labels=exclude_labels,
    )

    out[f"triad__{region_name}__n_cells"] = float(len(df_region))
    out[f"triad__{region_name}__n_labels"] = float(len(labels_present))
    out[f"triad__{region_name}__n_centers_considered"] = float(len(centers))
    out[f"triad__{region_name}__n_neighbors_considered"] = float(len(neighbors))

    if not centers or len(neighbors) < 2:
        return out, long_rows

    df_region = df_region.reset_index(drop=True).copy()
    neighbor_count_dicts = build_radius_neighbor_label_counts(df_region, threshold=threshold)
    phenos = df_region["phenotype"].astype(str).to_numpy()

    # Precompute center indices and denominators
    center_to_indices: Dict[str, np.ndarray] = {
        c: np.flatnonzero(phenos == c) for c in centers
    }

    for center in centers:
        center_indices = center_to_indices[center]
        n_center = int(len(center_indices))
        center_clean = sanitize_label(center)
        out[f"triad__{region_name}__center__{center_clean}__n_cells"] = float(n_center)

        pairs = neighbor_pairs_for_center(
            center=center,
            neighbors=neighbors,
            allow_center_as_neighbor=allow_center_as_neighbor,
            allow_same_neighbor_type=allow_same_neighbor_type,
        )

        for n1, n2 in pairs:
            n1_clean = sanitize_label(n1)
            n2_clean = sanitize_label(n2)
            feature_base = f"triad_centered__{center_clean}__{n1_clean}__{n2_clean}__{region_name}"

            if n_center < min_center_cells:
                count = np.nan
                frac_center = np.nan
            else:
                motif_count = 0
                for idx in center_indices:
                    d = neighbor_count_dicts[int(idx)]
                    if n1 == n2:
                        ok = d.get(n1, 0) >= 2
                    else:
                        ok = (d.get(n1, 0) >= 1) and (d.get(n2, 0) >= 1)
                    if ok:
                        motif_count += 1

                count = float(motif_count)
                frac_center = float(motif_count / n_center) if n_center > 0 else np.nan

            out[f"{feature_base}__count"] = count
            out[f"{feature_base}__frac_center"] = frac_center

            long_rows.append({
                "region": region_name,
                "center": center,
                "neighbor_1": n1,
                "neighbor_2": n2,
                "count": count,
                "n_center": float(n_center),
                "frac_center": frac_center,
            })

    return out, long_rows


def read_prep_file(prep_file: Path) -> pd.DataFrame:
    df = pd.read_csv(prep_file, sep="\t", low_memory=False)
    required = ["sample_id", "analysisregion", "Xcenter", "Ycenter", "phenotype"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{prep_file} is missing required columns: {missing}")

    meta = infer_path_metadata(prep_file)

    # Prefer file columns where present, but fall back to path metadata.
    if "Panel" not in df.columns:
        df["Panel"] = meta["Panel"]
    if "cohort" not in df.columns:
        df["cohort"] = meta["cohort"]
    if "sample_name" not in df.columns:
        df["sample_name"] = df["sample_id"]
    if "coord" not in df.columns:
        df["coord"] = df["sample_id"]

    df["dataset"] = meta["dataset"]
    df["chunk"] = meta["chunk"]
    df["Panel"] = df["Panel"].map(norm_panel)
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["sample_name"] = df["sample_name"].astype(str).str.strip()
    df["coord"] = df["coord"].astype(str).str.strip()
    df["phenotype"] = df["phenotype"].astype(str).str.strip()
    df["Xcenter"] = pd.to_numeric(df["Xcenter"], errors="coerce")
    df["Ycenter"] = pd.to_numeric(df["Ycenter"], errors="coerce")
    df["analysisregion_norm"] = df["analysisregion"].map(normalize_region)

    df = df.dropna(subset=["sample_id", "Xcenter", "Ycenter", "phenotype"])
    return df


def process_prep_file(
    prep_file: Path,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    df = read_prep_file(prep_file)

    if args.include_panels:
        include = {norm_panel(x) for x in args.include_panels}
        df = df[df["Panel"].isin(include)].copy()

    if args.exclude_panels:
        exclude = {norm_panel(x) for x in args.exclude_panels}
        df = df[~df["Panel"].isin(exclude)].copy()

    if df.empty:
        return [], []

    sample_rows: List[Dict[str, object]] = []
    long_rows_all: List[Dict[str, object]] = []

    for sample_id, sdf in df.groupby("sample_id", sort=False):
        meta = {c: sdf[c].iloc[0] for c in META_COLS if c in sdf.columns}
        row: Dict[str, object] = dict(meta)
        row["source_prep_file"] = str(prep_file)
        row["n_cells_total_input"] = int(len(sdf))

        sample_long_rows: List[Dict[str, object]] = []

        for region in args.regions:
            rdf = region_filter(sdf, region=region)
            region_name = "Tumor" if str(region).lower() in {"epi", "epithelial"} else str(region)

            if len(rdf) < args.min_cells_sample_region:
                row[f"triad__{region_name}__n_cells"] = float(len(rdf))
                continue

            feat, long_rows = compute_centered_triads_for_region(
                df_region=rdf,
                region_name=region_name,
                threshold=args.threshold,
                center_labels=args.center_labels,
                center_regex=args.center_regex,
                neighbor_labels=args.neighbor_labels,
                exclude_labels=args.exclude_labels,
                allow_center_as_neighbor=args.allow_center_as_neighbor,
                allow_same_neighbor_type=args.allow_same_neighbor_type,
                min_center_cells=args.min_center_cells,
            )
            row.update(feat)

            for lr in long_rows:
                lr2 = dict(meta)
                lr2.update(lr)
                lr2["source_prep_file"] = str(prep_file)
                sample_long_rows.append(lr2)

        sample_rows.append(row)
        long_rows_all.extend(sample_long_rows)

    return sample_rows, long_rows_all


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep_files = find_prep_files(args.prep_roots, max_files=args.max_files)
    if not prep_files:
        raise FileNotFoundError(f"No 1NN_prep.tsv files found under: {args.prep_roots}")

    print(f"[INFO] Found {len(prep_files)} 1NN_prep.tsv files")
    print(f"[INFO] threshold = {args.threshold}")
    print(f"[INFO] regions   = {args.regions}")
    print(f"[INFO] centers   = {args.center_labels if args.center_labels else args.center_regex if args.center_regex else 'all non-excluded labels'}")
    print(f"[INFO] excluded labels = {args.exclude_labels}")

    rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for i, prep_file in enumerate(prep_files, start=1):
        print(f"[INFO] ({i}/{len(prep_files)}) {prep_file}")
        sample_rows, sample_long = process_prep_file(prep_file, args)
        rows.extend(sample_rows)
        if args.write_long:
            long_rows.extend(sample_long)

    if not rows:
        raise ValueError("No feature rows were produced. Check panel filters / prep roots.")

    out_df = pd.DataFrame(rows)

    # Stable metadata-first ordering
    meta_cols = [c for c in META_COLS + ["source_prep_file", "n_cells_total_input"] if c in out_df.columns]
    feature_cols = [c for c in out_df.columns if c not in meta_cols]
    count_cols = [c for c in feature_cols if c.endswith("__count") or c.endswith("__n_cells") or c.endswith("__n_labels") or c.endswith("__n_centers_considered") or c.endswith("__n_neighbors_considered")]
    frac_cols = [c for c in feature_cols if c.endswith("__frac_center")]
    other_cols = [c for c in feature_cols if c not in count_cols and c not in frac_cols]

    # Missing count-like features mean absent motif/feature in that sample/chunk.
    for c in count_cols:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0.0)
    for c in frac_cols + other_cols:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    out_df = out_df[meta_cols + sorted(count_cols) + sorted(frac_cols) + sorted(other_cols)]

    out_csv = outdir / args.outfile
    out_df.to_csv(out_csv, index=False)

    manifest_rows = []
    for c in sorted(count_cols + frac_cols + other_cols):
        if c.startswith("triad_centered__"):
            parts = c.split("__")
            # triad_centered, center, n1, n2, region, metric
            manifest_rows.append({
                "feature": c,
                "feature_family": "centered_triad",
                "center": parts[1] if len(parts) > 1 else None,
                "neighbor_1": parts[2] if len(parts) > 2 else None,
                "neighbor_2": parts[3] if len(parts) > 3 else None,
                "region": parts[4] if len(parts) > 4 else None,
                "metric": parts[5] if len(parts) > 5 else None,
                "threshold": args.threshold,
            })
        elif c.startswith("triad__"):
            parts = c.split("__")
            manifest_rows.append({
                "feature": c,
                "feature_family": "triad_qc",
                "center": None,
                "neighbor_1": None,
                "neighbor_2": None,
                "region": parts[1] if len(parts) > 1 else None,
                "metric": "__".join(parts[2:]) if len(parts) > 2 else None,
                "threshold": args.threshold,
            })

    pd.DataFrame(manifest_rows).to_csv(outdir / "triad_feature_manifest.csv", index=False)

    if args.write_long:
        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(outdir / args.outfile.replace(".csv", "_long.csv"), index=False)

    print(f"[DONE] Wrote: {out_csv}")
    print(f"[DONE] Shape: {out_df.shape[0]} rows x {out_df.shape[1]} columns")
    print(f"[DONE] Feature manifest: {outdir / 'triad_feature_manifest.csv'}")


if __name__ == "__main__":
    main()
