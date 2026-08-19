#!/usr/bin/env python3
"""
stage2b_v9_setup.py

Prepare corrected v9 Stage 2B cap sensitivity inputs.

Inputs
------
S1 exports created by stage2a5_cap_rho_grid_v2.py for:
  cap 10, 15, 20
  rho 0.90

Outputs
-------
1. One normalized cap manifest per cap.
2. One union feature manifest across all caps.
3. One cross-cohort-expanded union manifest used to build the shared matrix cache.
4. An 8-row cache worker index: 4 discovery cohorts x 2 panels.

Important
---------
The Stage 2A export rewrites feature_uid to the global representative but may retain
local feature_source/feature_group/feature columns. This setup script therefore
reconstructs source/group/feature directly from the final feature_uid and never
trusts the local columns for matrix reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def split_uid(uid: str):
    parts = str(uid).split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid feature_uid (expected source|group|feature): {uid}")
    return parts


def normalize_manifest(df: pd.DataFrame, cap: int, rho: float) -> pd.DataFrame:
    out = df.copy()
    if "feature_uid" not in out.columns:
        raise ValueError("Manifest lacks feature_uid")

    parsed = out["feature_uid"].astype(str).apply(split_uid)
    out["feature_source"] = parsed.str[0]
    out["feature_group"] = parsed.str[1]
    out["feature"] = parsed.str[2]

    out["candidate_cap"] = int(cap)
    out["semantic_rho"] = float(rho)

    # Spearman correlation is invariant to affine z-scoring, and zscore is safe
    # for both signed and nonnegative variables. The shared cache therefore uses
    # one consistent reconstruction mode for every feature/cohort.
    out["selected_transform_mode"] = "zscore"

    if "canonical_feature_id" not in out.columns:
        out["canonical_feature_id"] = (
            out["panel"].astype(str) + "||" + out["feature_uid"].astype(str)
        )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_json(args.config)
    root = ensure_dir(cfg["stage2b_root"])
    manifests_root = ensure_dir(root / "manifests")
    setup_root = ensure_dir(root / "setup")

    caps = [int(x) for x in cfg.get("caps", [10, 15, 20])]
    rho = float(cfg.get("rho", 0.90))
    cohorts = [str(x) for x in cfg["discovery_cohorts"]]
    panels = [str(x) for x in cfg["panels"]]

    union_parts: List[pd.DataFrame] = []
    cap_rows: List[dict] = []

    for cap in caps:
        src = (
            manifests_root
            / f"cap{cap:03d}__rho0p9"
            / "S1"
            / "global_module_candidate_manifest.csv"
        )
        if not src.exists():
            raise FileNotFoundError(
                f"Missing S1 export for cap {cap}: {src}\n"
                "Run the export section of stage2_global_modules_review_v9.ipynb first."
            )

        raw = pd.read_csv(src)
        norm = normalize_manifest(raw, cap=cap, rho=rho)

        # Keep context-level rows for the cap-specific consensus manifest.
        norm_out = setup_root / f"cap{cap:03d}__rho0p9__S1_normalized_manifest.csv"
        norm.to_csv(norm_out, index=False)

        unique = (
            norm.sort_values(
                [c for c in ["candidate_score", "primary_oof_metric"] if c in norm.columns],
                ascending=False,
                na_position="last",
            )
            .drop_duplicates(["panel", "feature_uid"], keep="first")
            .copy()
        )
        union_parts.append(unique)

        cap_rows.append({
            "cap": cap,
            "manifest_path": str(norm_out),
            "n_rows": len(norm),
            "n_unique_features": norm["feature_uid"].nunique(),
            "AR_unique": norm.loc[norm["panel"].astype(str).eq("AR"), "feature_uid"].nunique(),
            "BT_unique": norm.loc[norm["panel"].astype(str).eq("BT"), "feature_uid"].nunique(),
        })

    union = pd.concat(union_parts, ignore_index=True, sort=False)
    union = (
        union.sort_values(
            [c for c in ["candidate_score", "primary_oof_metric"] if c in union.columns],
            ascending=False,
            na_position="last",
        )
        .drop_duplicates(["panel", "feature_uid"], keep="first")
        .reset_index(drop=True)
    )

    union_path = setup_root / "shared_union_feature_manifest.csv"
    union.to_csv(union_path, index=False)

    # Cross-cohort expansion: every selected global candidate is requested in
    # every discovery cohort. Availability is determined by the actual raw data.
    expanded_parts = []
    for cohort in cohorts:
        x = union.copy()
        x["cohort"] = cohort
        x["sample_type"] = str(cfg.get("sample_type", "TURBT"))
        x["patient_subset"] = str(cfg.get("patient_subset", "all"))
        x["agg"] = str(cfg.get("agg", "median"))
        x["selected_transform_mode"] = "zscore"
        expanded_parts.append(x)

    expanded = pd.concat(expanded_parts, ignore_index=True, sort=False)
    expanded_path = setup_root / "shared_union_cross_cohort_expanded_manifest.csv"
    expanded.to_csv(expanded_path, index=False)

    work_rows = []
    wid = 0
    for cohort in cohorts:
        for panel in panels:
            required = expanded[
                expanded["cohort"].astype(str).eq(cohort)
                & expanded["panel"].astype(str).eq(panel)
            ]
            work_rows.append({
                "cache_array_id": wid,
                "cohort": cohort,
                "panel": panel,
                "sample_type": str(cfg.get("sample_type", "TURBT")),
                "patient_subset": str(cfg.get("patient_subset", "all")),
                "agg": str(cfg.get("agg", "median")),
                "n_requested_features": required["feature_uid"].nunique(),
            })
            wid += 1

    work = pd.DataFrame(work_rows)
    work_path = setup_root / "shared_cache_worker_index.csv"
    work.to_csv(work_path, index=False)

    pd.DataFrame(cap_rows).to_csv(setup_root / "cap_manifest_summary.csv", index=False)

    resolved = dict(cfg)
    resolved.update({
        "union_feature_manifest": str(union_path),
        "expanded_union_manifest": str(expanded_path),
        "cache_worker_index": str(work_path),
    })
    with open(setup_root / "stage2b_v9_setup_resolved.json", "w") as f:
        json.dump(resolved, f, indent=2)

    print("[DONE] Stage 2B v9 setup")
    print(pd.DataFrame(cap_rows).to_string(index=False))
    print(f"[SAVE] {union_path}")
    print(f"[SAVE] {expanded_path}")
    print(f"[SAVE] {work_path}")
    print(f"[INFO] cache workers={len(work)}")


if __name__ == "__main__":
    main()
