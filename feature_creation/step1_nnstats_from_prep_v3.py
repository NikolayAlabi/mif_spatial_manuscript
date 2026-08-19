#!/usr/bin/env python3
"""
Compute 1-nearest-neighbor summary statistics directly from 1NN_prep.tsv.

This is a lightweight Python replacement for the NNstats-producing part of
cmdline_step1_weibull.R. It does NOT fit Weibull models and does NOT create
area-normalized smoothed histograms; it writes NNstats.tsv-style summaries.

Inputs
------
1NN_prep.tsv with columns:
    sample_id, Xcenter, Ycenter, phenotype

Outputs
-------
NNstats.tsv with columns similar to the R script:
    sample_id, phenotype_combo,
    Distance_Mean, Distance_SD, Distance_Max, Distance_Min,
    Distance_Median, Distance_Q1, Distance_Q3
plus QC/support columns:
    phenotype_from, phenotype_to, n_from, n_to, n_distances

Nearest-neighbor definition
---------------------------
For from != to: nearest target cell of phenotype_to for each source cell
of phenotype_from.

For from == to: second nearest same-phenotype cell for each source cell,
so that a cell is not its own nearest neighbor. If fewer than two cells of
that phenotype exist in the sample, distance summaries are NA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_EXCLUDE_LABELS = [
    "Unknown", "Other", "Immune", "artifact", "unresolved", "mixed_lineage",
]


STAT_COLS = [
    "Distance_Mean", "Distance_SD", "Distance_Max", "Distance_Min",
    "Distance_Median", "Distance_Q1", "Distance_Q3",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute NNstats.tsv-style 1NN summaries from 1NN_prep.tsv."
    )

    # Direct mode
    ap.add_argument("--prep-file", type=Path, default=None, help="Path to 1NN_prep.tsv.")
    ap.add_argument("--out-file", type=Path, default=None, help="Output NNstats.tsv path.")

    # Legacy wrapper mode, matching old R arguments
    ap.add_argument("--root", type=Path, default=None, help="Legacy root directory.")
    ap.add_argument("--filter", type=str, default="all", help="Legacy filter directory.")
    ap.add_argument("--typemode", type=str, default="canonical", help="Legacy typemode directory.")
    ap.add_argument("--panel", type=str, default=None, help="Legacy panel directory.")
    ap.add_argument("--surgery", type=str, default="all", help="Legacy surgery directory.")

    ap.add_argument("--sample-col", default="sample_id")
    ap.add_argument("--x-col", default="Xcenter")
    ap.add_argument("--y-col", default="Ycenter")
    ap.add_argument("--phenotype-col", default="phenotype")
    ap.add_argument("--exclude-labels", nargs="*", default=DEFAULT_EXCLUDE_LABELS)
    ap.add_argument("--include-labels", nargs="*", default=None,
                    help="Optional explicit phenotype list/order. Otherwise inferred from data after exclusions.")
    ap.add_argument("--min-from", type=int, default=1,
                    help="Minimum number of source cells required to compute a pair.")
    ap.add_argument("--min-to", type=int, default=1,
                    help="Minimum number of target cells required for different-phenotype pairs.")
    ap.add_argument("--min-same", type=int, default=2,
                    help="Minimum cells required for same-phenotype second-nearest calculation.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel processes across samples. Use 1 inside single-core array tasks.")
    ap.add_argument("--write-long", action="store_true",
                    help="Also write per-source-cell long distances next to the summary output. Can be huge.")
    ap.add_argument("--metadata-json", type=Path, default=None,
                    help="Optional path to write run metadata JSON.")
    return ap.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.prep_file is not None:
        prep = args.prep_file
        out = args.out_file if args.out_file is not None else prep.parent / "NNstats.tsv"
        return prep, out

    if args.root is None or args.panel is None:
        raise ValueError("Provide either --prep-file or legacy --root plus --panel.")

    leaf = Path(args.root) / args.filter / args.typemode / args.panel / args.surgery
    return leaf / "1NN_prep.tsv", leaf / "NNstats.tsv"


def clean_labels(s: pd.Series) -> pd.Series:
    return s.fillna("Marker-").astype(str).str.strip()


def summarize_distances(d: np.ndarray) -> Dict[str, float]:
    if d is None or len(d) == 0:
        return {c: np.nan for c in STAT_COLS}
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {c: np.nan for c in STAT_COLS}
    return {
        "Distance_Mean": float(np.mean(d)),
        "Distance_SD": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
        "Distance_Max": float(np.max(d)),
        "Distance_Min": float(np.min(d)),
        "Distance_Median": float(np.median(d)),
        "Distance_Q1": float(np.quantile(d, 0.25)),
        "Distance_Q3": float(np.quantile(d, 0.75)),
    }


def compute_one_sample(
    payload: Tuple[str, pd.DataFrame, List[str], str, str, str, str, int, int, int, bool]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, object]]:
    (
        sample_id, g, labels, sample_col, x_col, y_col, pheno_col,
        min_from, min_to, min_same, write_long,
    ) = payload

    g = g[[x_col, y_col, pheno_col]].copy()
    g[x_col] = pd.to_numeric(g[x_col], errors="coerce")
    g[y_col] = pd.to_numeric(g[y_col], errors="coerce")
    g = g.dropna(subset=[x_col, y_col, pheno_col])

    coords_by_label: Dict[str, np.ndarray] = {}
    trees: Dict[str, cKDTree] = {}
    n_by_label: Dict[str, int] = {}

    for lbl in labels:
        arr = g.loc[g[pheno_col].eq(lbl), [x_col, y_col]].to_numpy(dtype=float)
        coords_by_label[lbl] = arr
        n_by_label[lbl] = int(arr.shape[0])
        if arr.shape[0] > 0:
            trees[lbl] = cKDTree(arr)

    summary_rows = []
    long_rows = [] if write_long else None

    for p_from in labels:
        src = coords_by_label[p_from]
        n_from = n_by_label[p_from]

        for p_to in labels:
            tgt = coords_by_label[p_to]
            n_to = n_by_label[p_to]
            distances = np.array([], dtype=float)

            if p_from == p_to:
                if n_from >= max(min_same, 2):
                    d, _idx = trees[p_to].query(src, k=2)
                    distances = np.asarray(d)[:, 1]
            else:
                if n_from >= min_from and n_to >= min_to and n_to > 0:
                    d, _idx = trees[p_to].query(src, k=1)
                    distances = np.asarray(d)

            stats = summarize_distances(distances)
            row = {
                "sample_id": sample_id,
                "phenotype_from": p_from,
                "phenotype_to": p_to,
                "phenotype_combo": f"{p_from}_to_{p_to}",
                "n_from": n_from,
                "n_to": n_to,
                "n_distances": int(len(distances)),
            }
            row.update(stats)
            summary_rows.append(row)

            if write_long and len(distances) > 0:
                for val in distances:
                    long_rows.append({
                        "sample_id": sample_id,
                        "phenotype_from": p_from,
                        "phenotype_to": p_to,
                        "phenotype_combo": f"{p_from}_to_{p_to}",
                        "Distance": float(val),
                    })

    status = {
        "sample_id": sample_id,
        "n_cells": int(g.shape[0]),
        "n_labels_present": int(sum(v > 0 for v in n_by_label.values())),
        "n_labels_total": int(len(labels)),
    }

    summary = pd.DataFrame(summary_rows)
    long = pd.DataFrame(long_rows) if write_long else None
    return summary, long, status


def main() -> None:
    args = parse_args()
    prep_file, out_file = resolve_paths(args)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not prep_file.exists():
        raise FileNotFoundError(f"Missing prep file: {prep_file}")

    print(f"[INFO] Reading {prep_file}", flush=True)
    df = pd.read_csv(prep_file, sep="\t", low_memory=False)

    required = [args.sample_col, args.x_col, args.y_col, args.phenotype_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{prep_file} is missing required columns: {missing}")

    df = df.copy()
    df[args.sample_col] = df[args.sample_col].astype(str).str.strip()
    df[args.phenotype_col] = clean_labels(df[args.phenotype_col])

    exclude = {str(x).strip() for x in args.exclude_labels}
    df = df.loc[~df[args.phenotype_col].isin(exclude)].copy()

    if args.include_labels:
        labels = [str(x).strip() for x in args.include_labels]
        df = df.loc[df[args.phenotype_col].isin(labels)].copy()
    else:
        labels = sorted(df[args.phenotype_col].dropna().unique().tolist())

    if not labels:
        raise ValueError("No phenotypes remain after filtering.")

    samples = df[args.sample_col].dropna().unique().tolist()
    print(f"[INFO] Samples: {len(samples)} | labels: {len(labels)} | cells: {len(df):,}", flush=True)

    payloads = [
        (
            str(s),
            df.loc[df[args.sample_col].eq(s), [args.sample_col, args.x_col, args.y_col, args.phenotype_col]].copy(),
            labels,
            args.sample_col, args.x_col, args.y_col, args.phenotype_col,
            args.min_from, args.min_to, args.min_same, args.write_long,
        )
        for s in samples
    ]

    summaries = []
    longs = []
    status_rows = []

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for k, p in enumerate(payloads, start=1):
            if k % 25 == 0:
                print(f"[INFO] Processed {k}/{len(payloads)} samples", flush=True)
            summary, long, status = compute_one_sample(p)
            summaries.append(summary)
            if long is not None:
                longs.append(long)
            status_rows.append(status)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(compute_one_sample, p) for p in payloads]
            for k, fut in enumerate(as_completed(futs), start=1):
                if k % 25 == 0:
                    print(f"[INFO] Processed {k}/{len(payloads)} samples", flush=True)
                summary, long, status = fut.result()
                summaries.append(summary)
                if long is not None:
                    longs.append(long)
                status_rows.append(status)

    out = pd.concat(summaries, ignore_index=True)
    col_order = [
        "sample_id", "phenotype_combo", "phenotype_from", "phenotype_to",
        "n_from", "n_to", "n_distances",
    ] + STAT_COLS
    out = out[col_order]
    out.to_csv(out_file, sep="\t", index=False)
    print(f"[DONE] Wrote {out_file} with shape {out.shape}", flush=True)

    status = pd.DataFrame(status_rows)
    status_file = out_file.with_name("NNstats_sample_status.tsv")
    status.to_csv(status_file, sep="\t", index=False)

    label_file = out_file.with_name("NNstats_label_manifest.tsv")
    pd.DataFrame({"phenotype": labels}).to_csv(label_file, sep="\t", index=False)

    if args.write_long and longs:
        long_file = out_file.with_name("NNstats_long_distances.tsv")
        pd.concat(longs, ignore_index=True).to_csv(long_file, sep="\t", index=False)
        print(f"[DONE] Wrote long distances: {long_file}", flush=True)

    metadata = {
        "script": Path(__file__).name,
        "prep_file": str(prep_file),
        "out_file": str(out_file),
        "n_samples": len(samples),
        "n_cells": int(len(df)),
        "labels": labels,
        "exclude_labels": list(exclude),
        "jobs": jobs,
        "note": "NNstats-only Python replacement; no area-normalized Weibull histogram generated.",
    }
    metadata_file = args.metadata_json if args.metadata_json is not None else out_file.with_name("NNstats_run_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
