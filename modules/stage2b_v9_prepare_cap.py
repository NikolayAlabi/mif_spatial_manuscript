#!/usr/bin/env python3
"""
stage2b_v9_prepare_cap.py

Statistical Stage 2B preparation for one corrected v9 candidate cap.

This script deliberately DOES NOT use the legacy semantic parser or legacy
composite k score. It performs only:
  - shared-cache loading/subsetting
  - signed Spearman consensus across cohorts
  - pair-support filtering
  - row-Spearman clustering distance
  - average-linkage hierarchical clustering
  - memberships for every requested k
  - statistical silhouette / cluster-size diagnostics

Primitive/tissue-aware/measure-aware semantic silhouettes are recalculated later
in stage2_global_modules_review_v9.ipynb using stage2_feature_parser_v8_2.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def import_module_from_path(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_silhouette(D: pd.DataFrame, labels) -> float:
    labels = np.asarray(labels)
    if len(labels) < 3 or len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return np.nan
    arr = D.to_numpy(dtype=float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    try:
        return float(silhouette_score(arr, labels, metric="precomputed"))
    except Exception:
        return np.nan


def cluster_stats(labels):
    sizes = pd.Series(labels).value_counts()
    n = len(labels)
    n_singletons = int((sizes == 1).sum())
    return {
        "n_clusters_observed": int(len(sizes)),
        "median_cluster_size": float(sizes.median()),
        "mean_cluster_size": float(sizes.mean()),
        "max_cluster_size": int(sizes.max()),
        "max_cluster_fraction": float(sizes.max() / n) if n else np.nan,
        "singleton_fraction": float(n_singletons / len(sizes)) if len(sizes) else np.nan,
        "singleton_feature_fraction": float(n_singletons / n) if n else np.nan,
        "n_singletons": n_singletons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cap", type=int, required=True)
    args = ap.parse_args()

    cfg = read_json(args.config)
    cap = int(args.cap)

    stage2b_root = Path(cfg["stage2b_root"])
    setup_root = stage2b_root / "setup"
    cache_root = stage2b_root / "shared_matrix_cache"

    manifest_path = setup_root / f"cap{cap:03d}__rho0p9__S1_normalized_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)
    output_root = stage2b_root / "prepared" / f"cap{cap:03d}__rho0p9"
    output_root.mkdir(parents=True, exist_ok=True)

    gm = import_module_from_path(
        "stage2_global_module_utils_v7_stat_only",
        cfg["stage2_global_module_utils_path"],
    )

    # Load the same eight shared-cache matrices for every cap, then subset each
    # matrix to the cap-specific panel candidate list.
    matrices: Dict[str, pd.DataFrame] = {}
    cache_index = pd.read_csv(setup_root / "shared_cache_worker_index.csv")

    for _, r in cache_index.iterrows():
        cohort = str(r["cohort"])
        panel = str(r["panel"])
        sample_type = str(r["sample_type"])
        subset = str(r["patient_subset"])
        agg = str(r["agg"])

        matrix_path = (
            cache_root / "patient_matrices" / panel
            / f"{cohort}__{sample_type}__{subset}__agg-{agg}.parquet"
        )
        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Shared-cache matrix missing: {matrix_path}"
            )

        M = pd.read_parquet(matrix_path)
        cap_uids = set(
            manifest.loc[
                manifest["panel"].astype(str).eq(panel),
                "feature_uid"
            ].astype(str)
        )
        keep = ["patient_id"] + [
            u for u in M.columns
            if u != "patient_id" and u in cap_uids
        ]
        M = M[keep].copy()

        ctx_id = "__".join([cohort, panel, sample_type, subset, agg])
        matrices[ctx_id] = M

    all_stat = []

    resolved = dict(cfg)
    resolved.update({
        "cap": cap,
        "candidate_manifest": str(manifest_path),
        "output_root": str(output_root),
    })
    with open(output_root / "stage2b_prepare_config_v9.resolved.json", "w") as f:
        json.dump(resolved, f, indent=2)

    manifest.to_csv(output_root / "candidate_manifest_used.csv", index=False)

    for panel in cfg["panels"]:
        panel = str(panel)
        panel_manifest = manifest[
            manifest["panel"].astype(str).eq(panel)
        ].copy()
        if panel_manifest.empty:
            print(f"[WARN] no manifest rows for {panel}")
            continue

        print("=" * 90)
        print(f"[PANEL] cap={cap} {panel}")

        consensus_res = gm.build_consensus_for_panel(
            panel=panel,
            matrices=matrices,
            context_manifest=manifest,
            min_nonmissing_frac=float(
                cfg.get("min_nonmissing_frac_for_corr", 0.20)
            ),
            consensus_level=str(cfg.get("consensus_level", "cohort")),
        )
        consensus = consensus_res["consensus"]
        pair_support = consensus_res["pair_support"]
        matrix_qc = consensus_res["matrix_qc"]

        support_res = gm.apply_support_filter(
            consensus,
            pair_support,
            min_pair_support=int(cfg.get("min_pair_support", 2)),
            min_feature_support_frac=float(
                cfg.get("min_feature_support_frac", 0.10)
            ),
        )
        consensus_filtered = support_res["consensus_filtered"]
        pair_support_filtered = support_res["pair_support_filtered"]
        feature_support_summary = support_res["feature_support_summary"]

        print(f"[INFO] consensus={consensus.shape}")
        print(f"[INFO] support-filtered={consensus_filtered.shape}")

        pdir = output_root / panel
        pdir.mkdir(parents=True, exist_ok=True)

        consensus.to_parquet(
            pdir / f"{panel}_consensus_similarity.parquet"
        )
        pair_support.to_parquet(
            pdir / f"{panel}_pair_support.parquet"
        )
        consensus_filtered.to_parquet(
            pdir / f"{panel}_consensus_similarity_support_filtered.parquet"
        )
        pair_support_filtered.to_parquet(
            pdir / f"{panel}_pair_support_filtered.parquet"
        )
        feature_support_summary.to_csv(
            pdir / f"{panel}_feature_support_summary.csv",
            index=False,
        )
        panel_manifest.to_csv(
            pdir / f"{panel}_manifest_used.csv",
            index=False,
        )
        matrix_qc.to_csv(
            pdir / f"{panel}_matrix_qc.csv",
            index=False,
        )

        if consensus_filtered.shape[0] < 3:
            pd.DataFrame().to_csv(
                pdir / f"{panel}_memberships_all_k.csv",
                index=False,
            )
            pd.DataFrame().to_csv(
                pdir / f"{panel}_k_selection_diagnostics.csv",
                index=False,
            )
            continue

        D = gm.build_distance_matrix(
            consensus_filtered,
            mode=str(cfg.get("distance_mode", "row_spearman")),
        )
        D.to_parquet(
            pdir / f"{panel}_clustering_distance.parquet"
        )

        Z = gm.linkage_from_distance(
            D,
            method=str(cfg.get("linkage_method", "average")),
        )

        features = consensus_filtered.index.astype(str).tolist()
        k_min = int(cfg.get("k_min", 4))
        k_max = int(
            (cfg.get("k_max_by_panel", {}) or {}).get(
                panel,
                25 if panel == "AR" else 20,
            )
        )
        k_max = min(k_max, len(features) - 1)

        diag_rows: List[dict] = []
        membership_rows: List[dict] = []

        for k in range(k_min, k_max + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            diag_rows.append({
                "cap": cap,
                "panel": panel,
                "requested_k": int(k),
                "n_features": len(features),
                "stat_silhouette": safe_silhouette(D, labels),
                **cluster_stats(labels),
            })

            for uid, lab in zip(features, labels):
                membership_rows.append({
                    "requested_k": int(k),
                    "feature_uid": uid,
                    "raw_cluster_id": int(lab),
                })

        diag = pd.DataFrame(diag_rows)
        memberships = pd.DataFrame(membership_rows)

        diag.to_csv(
            pdir / f"{panel}_k_selection_diagnostics.csv",
            index=False,
        )
        memberships.to_csv(
            pdir / f"{panel}_memberships_all_k.csv",
            index=False,
        )

        all_stat.append(diag)

    if all_stat:
        pd.concat(
            all_stat,
            ignore_index=True,
            sort=False,
        ).to_csv(
            output_root / "all_panel_statistical_k_diagnostics.csv",
            index=False,
        )

    print("=" * 90)
    print(f"[DONE] cap={cap} Stage 2B statistical preparation")
    print(f"[SAVE] {output_root}")


if __name__ == "__main__":
    main()
