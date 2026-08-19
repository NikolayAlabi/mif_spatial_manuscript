#!/usr/bin/env python3
"""
stage2b_setup_cap_sensitivity_v1.py

Lightweight setup for cap-sensitivity global-module discovery.

Reads the finished Stage 2A-5 cap x semantic-rho grid aggregation and creates:
  1) nominated S1 manifests for cap 10/15/20;
  2) cross-cohort-expanded manifests, where every selected canonical feature is
     requested in every discovery cohort;
  3) an all-cap union manifest used to build one shared patient-matrix cache;
  4) three Stage 2B config JSON files that reuse the shared cache.

This script is lightweight and can be run in a notebook or an interactive CPU job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


def read_json(path: str | Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(obj: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def rho_token(rho: float) -> str:
    return str(float(rho)).replace(".", "p")


def choose_support_rows(
    support: pd.DataFrame,
    cap: int,
    rho: float,
) -> pd.DataFrame:
    out = support[
        support["candidate_cap"].astype(int).eq(int(cap))
        & np.isclose(support["semantic_rho"].astype(float), float(rho))
    ].copy()
    if "support_set_S1" in out.columns:
        mask = out["support_set_S1"].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
        # Some aggregations store booleans rather than strings.
        if out["support_set_S1"].dtype == bool:
            mask = out["support_set_S1"]
        out = out[mask].copy()
    if out.empty:
        raise RuntimeError(f"No S1 support rows for cap={cap}, rho={rho}")
    return out


def make_nominated_manifest(
    dedup: pd.DataFrame,
    support_cell: pd.DataFrame,
    cap: int,
    rho: float,
) -> pd.DataFrame:
    cell = dedup[
        dedup["candidate_cap"].astype(int).eq(int(cap))
        & np.isclose(dedup["semantic_rho"].astype(float), float(rho))
    ].copy()
    selected = set(support_cell["canonical_feature_id"].astype(str))
    cell = cell[cell["canonical_feature_id"].astype(str).isin(selected)].copy()
    if cell.empty:
        raise RuntimeError(f"No nominated manifest rows for cap={cap}, rho={rho}")

    support_cols = [
        "panel",
        "canonical_feature_id",
        "n_contexts",
        "n_cohorts",
        "n_endpoints",
        "cohorts",
        "contexts",
        "support_set_S1",
        "support_set_S2",
        "exceptional_selected",
        "support_set_S2_plus_E",
        "global_representative_feature_uid",
        "global_representative_feature_source",
        "global_representative_feature_group",
        "global_representative_feature",
    ]
    support_cols = [c for c in support_cols if c in support_cell.columns]
    cell = cell.merge(
        support_cell[support_cols],
        on=["panel", "canonical_feature_id"],
        how="left",
        suffixes=("", "_support"),
    )
    cell["local_feature_uid"] = cell["feature_uid"].astype(str)
    cell["feature_uid"] = cell["global_representative_feature_uid"].astype(str)
    cell["grid_candidate_cap"] = int(cap)
    cell["grid_semantic_rho"] = float(rho)
    cell["grid_support_set"] = "S1"

    sort_cols = [c for c in ["context_id", "feature_uid", "candidate_score", "primary_oof_metric"] if c in cell.columns]
    ascending = [True, True, False, False][: len(sort_cols)]
    cell = cell.sort_values(sort_cols, ascending=ascending, na_position="last")
    dedup_cols = [c for c in ["context_id", "feature_uid"] if c in cell.columns]
    if dedup_cols:
        cell = cell.drop_duplicates(dedup_cols, keep="first")
    return cell.reset_index(drop=True)


def make_feature_catalog(
    nominated: pd.DataFrame,
    support_cell: pd.DataFrame,
) -> pd.DataFrame:
    """One row per panel/canonical feature using the grid-selected global representative."""
    score_summary = (
        nominated.groupby(["panel", "canonical_feature_id"], dropna=False)
        .agg(
            candidate_score=("candidate_score", "max") if "candidate_score" in nominated.columns else ("feature_uid", "size"),
            primary_oof_metric=("primary_oof_metric", "max") if "primary_oof_metric" in nominated.columns else ("feature_uid", "size"),
            primary_delta_metric=("primary_delta_metric", "max") if "primary_delta_metric" in nominated.columns else ("feature_uid", "size"),
            nominated_contexts=("context_id", lambda x: ";".join(sorted(set(x.dropna().astype(str))))) if "context_id" in nominated.columns else ("feature_uid", "size"),
            nominated_cohorts=("cohort", lambda x: ";".join(sorted(set(x.dropna().astype(str))))) if "cohort" in nominated.columns else ("feature_uid", "size"),
            nominated_endpoints=("endpoint", lambda x: ";".join(sorted(set(x.dropna().astype(str))))) if "endpoint" in nominated.columns else ("feature_uid", "size"),
        )
        .reset_index()
    )

    cols = [
        "panel",
        "canonical_feature_id",
        "global_representative_feature_uid",
        "global_representative_feature_source",
        "global_representative_feature_group",
        "global_representative_feature",
        "n_contexts",
        "n_cohorts",
        "n_endpoints",
        "cohorts",
        "contexts",
        "support_set_S2",
        "exceptional_selected",
        "support_set_S2_plus_E",
    ]
    cols = [c for c in cols if c in support_cell.columns]
    cat = support_cell[cols].drop_duplicates(["panel", "canonical_feature_id"]).copy()
    cat = cat.merge(score_summary, on=["panel", "canonical_feature_id"], how="left")
    cat = cat.rename(
        columns={
            "global_representative_feature_uid": "feature_uid",
            "global_representative_feature_source": "feature_source",
            "global_representative_feature_group": "feature_group",
            "global_representative_feature": "feature",
        }
    )
    required = ["feature_uid", "feature_source", "feature_group", "feature"]
    missing = [c for c in required if c not in cat.columns]
    if missing:
        raise ValueError(f"Support table lacks representative metadata: {missing}")
    return cat


def expand_across_cohorts(
    catalog: pd.DataFrame,
    discovery_cohorts: Iterable[str],
    cap: int,
    rho: float,
    sample_type: str,
    patient_subset: str,
    agg: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, feat in catalog.iterrows():
        nominated_cohorts = set(str(feat.get("nominated_cohorts", "")).split(";")) - {""}
        for cohort in discovery_cohorts:
            row = feat.to_dict()
            row.update(
                {
                    "cohort": str(cohort),
                    "endpoint": "global_candidate_union",
                    "sample_type": str(sample_type),
                    "patient_subset": str(patient_subset),
                    "agg": str(agg),
                    "context_id": "__".join(
                        [str(cohort), str(feat["panel"]), str(sample_type), str(patient_subset), str(agg)]
                    ),
                    # Spearman correlation is invariant to z-scoring and to monotone
                    # log1p transforms. zscore is therefore the safest common mode,
                    # including for features that may take negative values.
                    "selected_transform_mode": "zscore",
                    "nominated_in_this_cohort": str(cohort) in nominated_cohorts,
                    "grid_candidate_cap": int(cap),
                    "grid_semantic_rho": float(rho),
                    "grid_support_set": "S1",
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    preferred = [
        "cohort",
        "panel",
        "endpoint",
        "sample_type",
        "patient_subset",
        "agg",
        "context_id",
        "feature_source",
        "feature_group",
        "feature",
        "feature_uid",
        "selected_transform_mode",
        "candidate_score",
        "primary_oof_metric",
        "primary_delta_metric",
        "canonical_feature_id",
        "n_contexts",
        "n_cohorts",
        "n_endpoints",
        "cohorts",
        "contexts",
        "nominated_contexts",
        "nominated_cohorts",
        "nominated_endpoints",
        "nominated_in_this_cohort",
        "support_set_S2",
        "exceptional_selected",
        "support_set_S2_plus_E",
        "grid_candidate_cap",
        "grid_semantic_rho",
        "grid_support_set",
    ]
    return out[[c for c in preferred if c in out.columns]].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = read_json(args.config)

    grid_root = Path(cfg["grid_root"])
    output_root = Path(cfg["output_root"])
    manifests_root = output_root / "manifests"
    configs_root = output_root / "configs"
    manifests_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    caps = [int(x) for x in cfg.get("candidate_caps", [10, 15, 20])]
    rho = float(cfg.get("semantic_rho", 0.90))
    cohorts = [str(x) for x in cfg.get("discovery_cohorts", ["NAC2020", "PURE01", "BLASST", "No-NAC"])]
    sample_type = str(cfg.get("sample_type", "TURBT"))
    patient_subset = str(cfg.get("patient_subset", "all"))
    agg = str(cfg.get("agg", "median"))

    dedup = read_table(grid_root / "all_grid_context_manifest_deduplicated.parquet")
    support = read_table(grid_root / "candidate_support_grid.parquet")

    cap_outputs: Dict[int, dict] = {}
    catalogs: Dict[int, pd.DataFrame] = {}
    for cap in caps:
        support_cell = choose_support_rows(support, cap, rho)
        nominated = make_nominated_manifest(dedup, support_cell, cap, rho)
        catalog = make_feature_catalog(nominated, support_cell)
        expanded = expand_across_cohorts(
            catalog,
            discovery_cohorts=cohorts,
            cap=cap,
            rho=rho,
            sample_type=sample_type,
            patient_subset=patient_subset,
            agg=agg,
        )

        cap_dir = manifests_root / f"cap{cap:03d}__rho{rho_token(rho)}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        nominated.to_csv(cap_dir / "nominated_S1_manifest.csv", index=False)
        catalog.to_csv(cap_dir / "canonical_feature_catalog.csv", index=False)
        expanded.to_csv(cap_dir / "cross_cohort_expanded_manifest.csv", index=False)
        catalogs[cap] = catalog

        stage2b_root = output_root / "prepared" / f"cap{cap:03d}__rho{rho_token(rho)}"
        stage2b_cfg = {
            "candidate_manifest": str(cap_dir / "cross_cohort_expanded_manifest.csv"),
            "output_root": str(stage2b_root),
            "shared_matrix_cache_root": str(output_root / "shared_matrix_cache"),
            "stage2_global_module_utils_path": cfg["stage2_global_module_utils_path"],
            "discovery_cohorts": cohorts,
            "panels": cfg.get("panels", ["AR", "BT"]),
            "sample_types": [sample_type],
            "patient_subsets": [patient_subset],
            "aggs": [agg],
            "feature_sources": None,
            "feature_groups": cfg.get("feature_groups", ["NN", "athena", "cell_features", "triads"]),
            "min_nonmissing_frac_for_corr": cfg.get("min_nonmissing_frac_for_corr", 0.20),
            "min_pair_support": cfg.get("min_pair_support", 2),
            "min_feature_support_frac": cfg.get("min_feature_support_frac", 0.10),
            "consensus_level": "cohort",
            "distance_mode": cfg.get("distance_mode", "row_spearman"),
            "linkage_method": cfg.get("linkage_method", "average"),
            "k_min": cfg.get("k_min", 4),
            "k_max_by_panel": cfg.get("k_max_by_panel", {"AR": 25, "BT": 20}),
            "fail_on_missing_cache_features": True,
            "cap_label": f"cap{cap}",
            "candidate_cap": cap,
            "semantic_rho": rho,
        }
        stage2b_cfg_path = configs_root / f"stage2b_cap{cap:03d}_rho{rho_token(rho)}.json"
        write_json(stage2b_cfg, stage2b_cfg_path)
        cap_outputs[cap] = {
            "nominated_manifest": str(cap_dir / "nominated_S1_manifest.csv"),
            "expanded_manifest": str(cap_dir / "cross_cohort_expanded_manifest.csv"),
            "stage2b_config": str(stage2b_cfg_path),
            "stage2b_output_root": str(stage2b_root),
            "n_features_by_panel": catalog.groupby("panel")["feature_uid"].nunique().to_dict(),
        }

    # IMPORTANT: microcompression representatives are not guaranteed to be nested
    # across candidate caps. A canonical feature retained at cap 10 or 15 can be
    # represented by a different feature_uid than the same canonical feature at
    # cap 20. Therefore the shared cache must contain the UNION of all cap-specific
    # expanded manifests, not merely the largest-cap manifest.
    expanded_parts = []
    for cap in caps:
        cap_manifest = manifests_root / f"cap{cap:03d}__rho{rho_token(rho)}" / "cross_cohort_expanded_manifest.csv"
        expanded_parts.append(pd.read_csv(cap_manifest))
    union_df = pd.concat(expanded_parts, ignore_index=True, sort=False)
    union_key_cols = [c for c in ["cohort", "panel", "sample_type", "patient_subset", "agg", "feature_uid"] if c in union_df.columns]
    union_df = union_df.drop_duplicates(union_key_cols, keep="first").reset_index(drop=True)
    cap_token = "_".join(f"{c:03d}" for c in caps)
    union_manifest = manifests_root / f"shared_union_caps_{cap_token}__rho{rho_token(rho)}.csv"
    union_df.to_csv(union_manifest, index=False)
    print(f"[UNION] all-cap shared-cache manifest: {union_manifest} rows={len(union_df)}")

    cache_cfg = {
        "union_manifest": str(union_manifest),
        "output_root": str(output_root / "shared_matrix_cache"),
        "stage2a4_root": cfg["stage2a4_root"],
        "stage2_global_module_utils_path": cfg["stage2_global_module_utils_path"],
        "stage1_script_path": cfg["stage1_script_path"],
        "harmonized_path": cfg["harmonized_path"],
        "koll_metadata_csv": cfg.get("koll_metadata_csv"),
        "spatial_root": cfg.get("spatial_root"),
        "cell_features_path": cfg.get("cell_features_path"),
        "triads_path": cfg.get("triads_path"),
        "qc_acceptability": cfg.get("qc_acceptability", "acceptable_or_borderline"),
        "min_epi_fraction": cfg.get("min_epi_fraction", 0.05),
        "prefer_reuse": True,
        "rebuild_missing": True,
        "fail_if_any_missing": True,
    }
    cache_cfg_path = configs_root / "stage2b_shared_matrix_cache.json"
    write_json(cache_cfg, cache_cfg_path)

    setup_summary = {
        "grid_root": str(grid_root),
        "output_root": str(output_root),
        "candidate_caps": caps,
        "semantic_rho": rho,
        "discovery_cohorts": cohorts,
        "shared_cache_config": str(cache_cfg_path),
        "cap_outputs": cap_outputs,
    }
    write_json(setup_summary, output_root / "stage2b_cap_sensitivity_setup_summary.json")

    print(f"[DONE] Wrote manifests/configs under {output_root}")
    print(f"[NEXT] Build shared cache using {cache_cfg_path}")
    for cap in caps:
        print(f"[CONFIG cap={cap}] {cap_outputs[cap]['stage2b_config']}")


if __name__ == "__main__":
    main()