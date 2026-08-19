#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.sum() == 0:
        return out
    vals = p.loc[ok].astype(float)
    order = np.argsort(vals.values)
    ranked = vals.values[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out.loc[vals.index[order]] = q
    return out


def read_many(root: Path, suffix: str) -> pd.DataFrame:
    files = sorted(root.rglob(f"*__{suffix}.csv"))
    if not files:
        return pd.DataFrame()
    parts = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            df["source_file"] = str(fp)
            parts.append(df)
        except Exception as e:
            parts.append(pd.DataFrame({"source_file": [str(fp)], "read_error": [f"{type(e).__name__}: {e}"]}))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def add_fullmodel_fdr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    base_group = [
        "feature_source", "cohort", "panel", "feature_group", "endpoint",
        "sample_type", "patient_subset", "qc_acceptability", "min_epi_fraction",
        "agg", "transform_mode", "model_name",
    ]
    group_cols = [c for c in base_group if c in out.columns]
    if "wald_p_value" in out.columns:
        out["wald_fdr_bh"] = out.groupby(group_cols, dropna=False)["wald_p_value"].transform(bh_fdr) if group_cols else bh_fdr(out["wald_p_value"])
    if "lrt_p_value" in out.columns:
        lrt_group = [c for c in group_cols if c != "model_name"]
        out["lrt_fdr_bh"] = out.groupby(lrt_group, dropna=False)["lrt_p_value"].transform(bh_fdr) if lrt_group else bh_fdr(out["lrt_p_value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="Stage-1 result root to scan recursively.")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for suffix in ["summary", "folds", "oof", "failures", "fullmodels", "feature_filter"]:
        df = read_many(args.root, suffix)
        if suffix == "fullmodels":
            df = add_fullmodel_fdr(df)
        outputs[suffix] = df
        fp = args.outdir / f"stage1_v6_combined__{suffix}.csv"
        df.to_csv(fp, index=False)
        print(f"[DONE] {suffix}: {df.shape} -> {fp}")

    # Convenience top-hit tables.
    summ = outputs.get("summary", pd.DataFrame())
    if not summ.empty:
        sort_cols = []
        for c in ["delta_oof_auc_vs_clinical", "delta_oof_cindex_vs_clinical", "biomarker_oof_auc", "biomarker_oof_cindex"]:
            if c in summ.columns:
                sort_cols.append(c)
        if sort_cols:
            top = summ.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            fp = args.outdir / "stage1_v6_top_summary_sorted.csv"
            top.to_csv(fp, index=False)
            print(f"[DONE] top summary -> {fp}")


if __name__ == "__main__":
    main()
