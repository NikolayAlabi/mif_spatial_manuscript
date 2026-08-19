#!/usr/bin/env python3
"""
stage2b_prepare_global_module_inputs_v7.py

Heavy-prep script for streamlined global module discovery.

This script is intended to be run before opening the review notebook. It reads the
Stage 2A candidate manifest, reconstructs transformed patient-level feature_uid
matrices, builds cross-cohort consensus correlation matrices, applies support
filtering, evaluates k-selection diagnostics, and writes heatmaps/diagnostic plots.

After this finishes, the notebook can be small and fast: it only loads prepared
outputs, reviews k diagnostics/heatmaps, chooses final k, and saves final module
memberships.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

import stage2_global_module_utils_v7 as gm


DEFAULT_CONFIG = {
    "candidate_manifest": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v7/discovery_primary_median_best_transform/global_module_candidate_manifest.csv",
    "output_root": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v7/global_module_discovery_primary_turbt_expanded",
    "stage1_script_path": "/projects/ovcare/users/nikolay_alabi/immuno/manuscript/univariate_screening/stage1_univariate_cv_screen_v6.py",
    "harmonized_path": "/projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv",
    "koll_metadata_csv": "/projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/KOLL_core_metadata.csv",
    "discovery_cohorts": ["NAC2020", "PURE01", "BLASST", "No-NAC"],
    "panels": ["AR", "BT"],
    "sample_types": ["TURBT"],
    "patient_subsets": ["all"],
    "aggs": ["median"],
    "feature_sources": None,
    "feature_groups": ["NN", "athena", "cell_features", "triads"],
    "qc_acceptability": "acceptable_or_borderline",
    "min_epi_fraction": 0.05,
    "min_nonmissing_frac_for_corr": 0.20,
    "min_pair_support": 2,
    "min_feature_support_frac": 0.10,
    "consensus_level": "cohort",
    "distance_mode": "row_spearman",
    "linkage_method": "average",
    "k_min": 5,
    "k_max_by_panel": {"AR": 35, "BT": 25},
}


def parse_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    if str(x).strip() == "":
        return None
    return [v.strip() for v in str(x).split(",") if v.strip()]


def load_config(args) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if args.config is not None:
        user_cfg = gm.read_json(args.config)
        cfg.update(user_cfg)

    # CLI overrides
    for name in [
        "candidate_manifest", "output_root", "stage1_script_path", "harmonized_path",
        "koll_metadata_csv", "qc_acceptability", "distance_mode", "linkage_method",
    ]:
        val = getattr(args, name, None)
        if val is not None:
            cfg[name] = val

    if args.discovery_cohorts is not None:
        cfg["discovery_cohorts"] = parse_list(args.discovery_cohorts)
    if args.panels is not None:
        cfg["panels"] = parse_list(args.panels)
    if args.sample_types is not None:
        cfg["sample_types"] = parse_list(args.sample_types)
    if args.patient_subsets is not None:
        cfg["patient_subsets"] = parse_list(args.patient_subsets)
    if args.aggs is not None:
        cfg["aggs"] = parse_list(args.aggs)
    if args.feature_sources is not None:
        cfg["feature_sources"] = parse_list(args.feature_sources)
    if args.feature_groups is not None:
        cfg["feature_groups"] = parse_list(args.feature_groups)
    if args.min_epi_fraction is not None:
        cfg["min_epi_fraction"] = args.min_epi_fraction
    if args.min_pair_support is not None:
        cfg["min_pair_support"] = int(args.min_pair_support)
    if args.min_feature_support_frac is not None:
        cfg["min_feature_support_frac"] = float(args.min_feature_support_frac)
    if args.k_min is not None:
        cfg["k_min"] = int(args.k_min)
    if args.k_max is not None:
        cfg["k_max"] = int(args.k_max)
    return cfg


def save_inputs_qc(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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


def run(args) -> None:
    cfg = load_config(args)
    output_root = gm.ensure_dir(cfg["output_root"])
    gm.write_json(cfg, output_root / "stage2b_prepare_config.resolved.json")

    gm.log("=" * 80)
    gm.log("[START] Stage 2B global module input preparation")
    gm.log(f"[INFO] output_root={output_root}")
    gm.log(f"[INFO] candidate_manifest={cfg['candidate_manifest']}")
    gm.log(f"[INFO] stage1_script_path={cfg['stage1_script_path']}")

    manifest0 = gm.load_candidate_manifest(cfg["candidate_manifest"])
    manifest = save_inputs_qc(manifest0, cfg)
    if manifest.empty:
        raise RuntimeError("No candidate rows remain after filtering. Check config and manifest.")

    # Import Stage 1 runtime only after confirming candidate manifest is valid.
    stage1_mod = gm.import_module_from_path("stage1_runtime_v6_for_stage2b", cfg["stage1_script_path"])

    # Build or load transformed patient matrices.
    matrix_result = gm.build_all_context_matrices(
        manifest=manifest,
        stage1_mod=stage1_mod,
        config=cfg,
        force=bool(args.force_rebuild_matrices),
    )
    matrices = matrix_result["matrices"]
    context_feature_meta = matrix_result["context_feature_meta"]
    context_matrix_manifest = matrix_result["context_matrix_manifest"]

    plots_root = gm.ensure_dir(output_root / "plots")

    # Build panel-level consensus matrices and k diagnostics.
    all_kdiag = []
    for panel in cfg.get("panels", ["AR", "BT"]):
        panel = str(panel)
        gm.log("=" * 80)
        gm.log(f"[PANEL] {panel}")
        panel_manifest = manifest[manifest["panel"].astype(str) == panel].copy()
        if panel_manifest.empty:
            gm.log(f"[WARN] No candidate rows for panel={panel}; skipping")
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

        panel_feature_meta = context_feature_meta[context_feature_meta["panel"].astype(str) == panel].copy() if not context_feature_meta.empty else pd.DataFrame()
        panel_plot_dir = gm.ensure_dir(plots_root / panel)

        # Always save support-filter outputs/diagnostic plots, even if the panel is too sparse for k selection.
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

        k_min = int(cfg.get("k_min", 5))
        if consensus_filtered.shape[0] < max(2, k_min):
            gm.log(
                f"[WARN] {panel} has only {consensus_filtered.shape[0]} support-filtered features; "
                f"skipping k selection. Relax the manifest/support thresholds for this panel."
            )
            empty_k_results = {
                "k_diagnostics": pd.DataFrame(),
                "memberships_all_k": pd.DataFrame(),
                "linkage": None,
            }
            gm.save_panel_prepared_outputs(
                panel=panel,
                output_root=output_root,
                consensus=consensus,
                pair_support=pair_support,
                consensus_filtered=consensus_filtered,
                pair_support_filtered=pair_support_filtered,
                feature_support_summary=feature_support_summary,
                k_results=empty_k_results,
                manifest_panel=panel_manifest,
                matrix_qc=matrix_qc,
            )
            continue

        k_max_by_panel = cfg.get("k_max_by_panel", {}) or {}
        k_max = int(cfg.get("k_max", k_max_by_panel.get(panel, 35 if panel == "AR" else 25)))
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

        # K-selection diagnostics plots.
        gm.plot_k_diagnostics(k_results["k_diagnostics"], panel_plot_dir, panel)
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

        kd = k_results["k_diagnostics"].copy()
        kd["panel"] = panel
        all_kdiag.append(kd)

        # Make a few provisional heatmaps for top k choices so the notebook can load images quickly.
        shortlist = gm.make_k_shortlist(kd, top_n=4)
        for k in shortlist["requested_k"].dropna().astype(int).tolist():
            try:
                mem = gm.remap_modules_by_size(k_results["memberships_all_k"], int(k))
                gm.plot_consensus_heatmap(
                    consensus_filtered,
                    panel_plot_dir / f"{panel}_consensus_heatmap_k{k:02d}.png",
                    title=f"{panel}: support-filtered consensus | k={k}",
                    membership=mem,
                    feature_meta=panel_feature_meta,
                )
            except Exception as e:
                gm.log(f"[WARN] provisional heatmap failed for {panel} k={k}: {type(e).__name__}: {e}")

    if all_kdiag:
        pd.concat(all_kdiag, ignore_index=True, sort=False).to_csv(output_root / "all_panel_k_selection_diagnostics.csv", index=False)

    gm.log("=" * 80)
    gm.log(f"[DONE] Stage 2B preparation complete: {output_root}")
    gm.log("Open stage2_global_modules_review_v7.ipynb to choose final k and save final modules.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON config file. CLI arguments override values in the config.")
    ap.add_argument("--candidate-manifest", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--stage1-script-path", default=None)
    ap.add_argument("--harmonized-path", default=None)
    ap.add_argument("--koll-metadata-csv", default=None)
    ap.add_argument("--discovery-cohorts", default=None, help="Comma-separated cohort list.")
    ap.add_argument("--panels", default=None, help="Comma-separated panel list.")
    ap.add_argument("--sample-types", default=None, help="Comma-separated sample types.")
    ap.add_argument("--patient-subsets", default=None, help="Comma-separated patient subsets.")
    ap.add_argument("--aggs", default=None, help="Comma-separated aggregations.")
    ap.add_argument("--feature-sources", default=None, help="Comma-separated feature sources or omit to use manifest.")
    ap.add_argument("--feature-groups", default=None, help="Comma-separated feature groups.")
    ap.add_argument("--qc-acceptability", default=None)
    ap.add_argument("--min-epi-fraction", type=float, default=None)
    ap.add_argument("--min-pair-support", type=int, default=None)
    ap.add_argument("--min-feature-support-frac", type=float, default=None)
    ap.add_argument("--distance-mode", default=None, choices=[None, "row_spearman", "direct_signed", "direct_abs"])
    ap.add_argument("--linkage-method", default=None)
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--force-rebuild-matrices", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
