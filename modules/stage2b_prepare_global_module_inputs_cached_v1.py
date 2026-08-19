#!/usr/bin/env python3
"""
stage2b_prepare_global_module_inputs_cached_v1.py

Stage 2B global-module preparation using a prebuilt shared patient-matrix cache.
No Stage 1 reconstruction occurs in this script. It subsets the cap-20 union cache
to the current cap manifest, then performs consensus correlations, support
filtering, k diagnostics, and provisional heatmaps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


def import_module(name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: str | Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(obj: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def load_config(args) -> dict:
    cfg = read_json(args.config)
    if args.candidate_manifest is not None:
        cfg["candidate_manifest"] = args.candidate_manifest
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.shared_matrix_cache_root is not None:
        cfg["shared_matrix_cache_root"] = args.shared_matrix_cache_root
    return cfg


def save_inputs_qc(gm, manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    outroot = gm.ensure_dir(cfg["output_root"])
    qc = gm.validate_candidate_manifest(manifest, cfg)
    qc["manifest_used"].to_csv(outroot / "candidate_manifest_used.csv", index=False)
    qc["summary"].to_csv(outroot / "candidate_manifest_summary.csv", index=False)
    qc["counts_by_context"].to_csv(outroot / "candidate_counts_by_context.csv", index=False)
    qc["counts_by_source_group"].to_csv(outroot / "candidate_counts_by_source_group.csv", index=False)
    qc["warnings"].to_csv(outroot / "candidate_manifest_warnings.csv", index=False)
    comp = gm.summarize_candidate_composition(qc["manifest_used"])
    for name, df in comp.items():
        df.to_csv(outroot / f"candidate_composition_{name}.csv", index=False)
    gm.plot_candidate_composition(qc["manifest_used"], outroot / "plots" / "candidate_composition")
    return qc["manifest_used"]


def load_shared_matrices(gm, manifest: pd.DataFrame, cfg: Mapping) -> Dict[str, object]:
    cache_root = Path(cfg["shared_matrix_cache_root"]) / "patient_matrices"
    matrices = {}
    context_rows = []
    meta_parts = []
    missing_rows = []

    group_cols = [c for c in gm.CONTEXT_MATRIX_COLS if c in manifest.columns]
    for _, ctx in manifest.groupby(group_cols, dropna=False):
        first = ctx.iloc[0]
        cohort = str(first["cohort"])
        panel = str(first["panel"])
        sample_type = str(first.get("sample_type", "TURBT"))
        patient_subset = str(first.get("patient_subset", "all"))
        agg = str(first.get("agg", "median"))
        ctx_id = gm.context_key_from_row(first)
        path = cache_root / panel / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.parquet"
        metapath = cache_root / panel / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__feature_meta.csv"
        if not path.exists() or not metapath.exists():
            raise FileNotFoundError(f"Missing shared cache for {ctx_id}: {path}")

        matrix = pd.read_parquet(path)
        meta = pd.read_csv(metapath)
        requested = list(dict.fromkeys(ctx["feature_uid"].dropna().astype(str).tolist()))
        missing = sorted(set(requested) - set(matrix.columns))
        for uid in missing:
            missing_rows.append({"context_matrix_id": ctx_id, "feature_uid": uid, "reason": "missing_from_shared_cache"})
        if missing and bool(cfg.get("fail_on_missing_cache_features", True)):
            pd.DataFrame(missing_rows).to_csv(Path(cfg["output_root"]) / "shared_cache_missing_features.csv", index=False)
            raise RuntimeError(f"{ctx_id}: {len(missing)} requested features missing from shared cache")

        keep = [uid for uid in requested if uid in matrix.columns]
        sub = matrix[["patient_id", *keep]].copy()
        matrices[ctx_id] = sub
        meta_sub = meta[meta["feature_uid"].astype(str).isin(keep)].copy()
        meta_parts.append(meta_sub)
        context_rows.append({
            "context_matrix_id": ctx_id,
            "cohort": cohort,
            "panel": panel,
            "sample_type": sample_type,
            "patient_subset": patient_subset,
            "agg": agg,
            "path": str(path),
            "feature_meta_path": str(metapath),
            "n_patients": int(sub.shape[0]),
            "n_feature_uid_columns": int(max(sub.shape[1] - 1, 0)),
        })

    output_root = Path(cfg["output_root"])
    matrix_audit_root = output_root / "patient_matrices"
    matrix_audit_root.mkdir(parents=True, exist_ok=True)
    context_df = pd.DataFrame(context_rows)
    feature_meta = pd.concat(meta_parts, ignore_index=True, sort=False) if meta_parts else pd.DataFrame()
    context_df.to_csv(matrix_audit_root / "context_matrix_manifest.csv", index=False)
    feature_meta.to_csv(matrix_audit_root / "context_feature_meta.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(output_root / "shared_cache_missing_features.csv", index=False)
    return {"matrices": matrices, "context_matrix_manifest": context_df, "context_feature_meta": feature_meta}


def run(args) -> None:
    cfg = load_config(args)
    gm = import_module("stage2_global_module_utils_cached_v1", cfg["stage2_global_module_utils_path"])
    output_root = gm.ensure_dir(cfg["output_root"])
    write_json(cfg, output_root / "stage2b_prepare_config.resolved.json")

    gm.log("=" * 80)
    gm.log("[START] Stage 2B cached global-module preparation")
    gm.log(f"[INFO] output_root={output_root}")
    gm.log(f"[INFO] candidate_manifest={cfg['candidate_manifest']}")
    gm.log(f"[INFO] shared_matrix_cache_root={cfg['shared_matrix_cache_root']}")

    manifest0 = gm.load_candidate_manifest(cfg["candidate_manifest"])
    manifest = save_inputs_qc(gm, manifest0, cfg)
    if manifest.empty:
        raise RuntimeError("No candidate rows remain after filtering")

    matrix_result = load_shared_matrices(gm, manifest, cfg)
    matrices = matrix_result["matrices"]
    context_feature_meta = matrix_result["context_feature_meta"]

    plots_root = gm.ensure_dir(output_root / "plots")
    all_kdiag = []
    for panel in cfg.get("panels", ["AR", "BT"]):
        panel = str(panel)
        gm.log("=" * 80)
        gm.log(f"[PANEL] {panel}")
        panel_manifest = manifest[manifest["panel"].astype(str).eq(panel)].copy()
        if panel_manifest.empty:
            gm.log(f"[WARN] No candidates for panel={panel}; skipping")
            continue

        consensus_res = gm.build_consensus_for_panel(
            panel=panel,
            matrices=matrices,
            context_manifest=manifest,
            min_nonmissing_frac=float(cfg.get("min_nonmissing_frac_for_corr", 0.20)),
            consensus_level=str(cfg.get("consensus_level", "cohort")),
        )
        consensus = consensus_res["consensus"]
        pair_support = consensus_res["pair_support"]
        matrix_qc = consensus_res["matrix_qc"]
        if consensus.empty:
            gm.log(f"[WARN] Empty consensus for panel={panel}; skipping")
            continue

        support_res = gm.apply_support_filter(
            consensus,
            pair_support,
            min_pair_support=int(cfg.get("min_pair_support", 2)),
            min_feature_support_frac=float(cfg.get("min_feature_support_frac", 0.10)),
        )
        consensus_filtered = support_res["consensus_filtered"]
        pair_support_filtered = support_res["pair_support_filtered"]
        feature_support_summary = support_res["feature_support_summary"]
        gm.log(f"[INFO] {panel} consensus shape={consensus.shape}")
        gm.log(f"[INFO] {panel} support-filtered shape={consensus_filtered.shape}")

        panel_feature_meta = (
            context_feature_meta[context_feature_meta["panel"].astype(str).eq(panel)].copy()
            if not context_feature_meta.empty
            else pd.DataFrame()
        )
        panel_plot_dir = gm.ensure_dir(plots_root / panel)
        gm.plot_support_diagnostics(feature_support_summary, panel_plot_dir, panel)
        gm.plot_consensus_heatmap(
            consensus,
            panel_plot_dir / f"{panel}_consensus_heatmap_unfiltered.png",
            title=f"{panel}: consensus similarity unfiltered",
            feature_meta=panel_feature_meta,
        )
        gm.plot_consensus_heatmap(
            consensus_filtered,
            panel_plot_dir / f"{panel}_consensus_heatmap_support_filtered.png",
            title=f"{panel}: consensus similarity support-filtered",
            feature_meta=panel_feature_meta,
        )

        k_min = int(cfg.get("k_min", 4))
        if consensus_filtered.shape[0] < max(2, k_min):
            gm.log(f"[WARN] {panel}: only {consensus_filtered.shape[0]} support-filtered features; skipping k grid")
            empty = {"k_diagnostics": pd.DataFrame(), "memberships_all_k": pd.DataFrame(), "linkage": None}
            gm.save_panel_prepared_outputs(
                panel=panel,
                output_root=output_root,
                consensus=consensus,
                pair_support=pair_support,
                consensus_filtered=consensus_filtered,
                pair_support_filtered=pair_support_filtered,
                feature_support_summary=feature_support_summary,
                k_results=empty,
                manifest_panel=panel_manifest,
                matrix_qc=matrix_qc,
            )
            continue

        k_max_by_panel = cfg.get("k_max_by_panel", {}) or {}
        k_max = int(k_max_by_panel.get(panel, 25 if panel == "AR" else 20))
        k_max = min(k_max, max(k_min, consensus_filtered.shape[0] - 1))
        k_results = gm.evaluate_k_grid(
            consensus_filtered=consensus_filtered,
            feature_meta=panel_feature_meta,
            k_min=k_min,
            k_max=k_max,
            linkage_method=str(cfg.get("linkage_method", "average")),
            distance_mode=str(cfg.get("distance_mode", "row_spearman")),
        )
        gm.save_panel_prepared_outputs(
            panel=panel,
            output_root=output_root,
            consensus=consensus,
            pair_support=pair_support,
            consensus_filtered=consensus_filtered,
            pair_support_filtered=pair_support_filtered,
            feature_support_summary=feature_support_summary,
            k_results=k_results,
            manifest_panel=panel_manifest,
            matrix_qc=matrix_qc,
        )
        gm.plot_k_diagnostics(k_results["k_diagnostics"], panel_plot_dir, panel)
        kd = k_results["k_diagnostics"].copy()
        kd["panel"] = panel
        kd["candidate_cap"] = cfg.get("candidate_cap")
        all_kdiag.append(kd)

        shortlist = gm.make_k_shortlist(kd, top_n=4)
        for k in shortlist.get("requested_k", pd.Series(dtype=float)).dropna().astype(int).tolist():
            try:
                mem = gm.remap_modules_by_size(k_results["memberships_all_k"], int(k))
                gm.plot_consensus_heatmap(
                    consensus_filtered,
                    panel_plot_dir / f"{panel}_consensus_heatmap_k{k:02d}.png",
                    title=f"{panel}: support-filtered consensus | k={k}",
                    membership=mem,
                    feature_meta=panel_feature_meta,
                )
            except Exception as exc:
                gm.log(f"[WARN] provisional heatmap failed for {panel} k={k}: {type(exc).__name__}: {exc}")

    if all_kdiag:
        pd.concat(all_kdiag, ignore_index=True, sort=False).to_csv(output_root / "all_panel_k_selection_diagnostics.csv", index=False)
    gm.log(f"[DONE] Stage 2B cached preparation complete: {output_root}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidate-manifest", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--shared-matrix-cache-root", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
