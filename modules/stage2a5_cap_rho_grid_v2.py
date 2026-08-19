#!/usr/bin/env python3
"""
stage2a5_cap_rho_grid_v2.py

Dense Stage 2A-5 sensitivity grid over:
  1) the number of threshold-passing seed candidates retained per context; and
  2) one shared semantic-correlation threshold used for state, metric-summary,
     and compartment simplification.

The existing Stage 2A-4 top-100 outputs are treated as a read-only master
superset. For every cap, only the top-ranked seeds and rescue features linked to
those seeds are retained. The fixed residual-correlation and OOF-loss safeguards
are then applied.

Commands
--------
validate   Validate paths/config and report the full grid size.
inventory  Build a one-row-per-context worker index from Stage 2A-4.
worker     Evaluate all cap x semantic-rho combinations for one context.
aggregate Aggregate contexts, canonicalize duplicates, calculate cohort support,
          select exceptional single-cohort candidates, and create plots.
export     Export one grid cell and support set as a Stage 2B candidate manifest.

Support definition
------------------
For a canonical biological feature, cohort support is the number of distinct
cohorts in which that feature survives context nomination and microcompression.
Multiple endpoints from one cohort increase context support but not cohort
support. Support is calculated separately for AR and BT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_COLS = ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"]
GRID_COLS = ["candidate_cap", "semantic_rho"]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(obj: Mapping, path: Union[str, Path]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def import_module_from_path(module_name: str, path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError("Could not import {}".format(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        alternatives = [p.with_suffix(".csv.gz"), p.with_suffix(".csv")]
        for alt in alternatives:
            if alt.exists():
                p = alt
                break
    if not p.exists():
        raise FileNotFoundError(path)
    if p.name.lower().endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_table(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return p
        except (ImportError, ModuleNotFoundError):
            p = p.with_suffix(".csv.gz")
            df.to_csv(p, index=False, compression="gzip")
            return p
    df.to_csv(p, index=False)
    return p


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def numeric(value: object, default: float = np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def panel_value(cfg: Mapping, key: str, panel: str, default: float) -> float:
    value = cfg.get(key, default)
    if isinstance(value, Mapping):
        return float(value.get(panel, default))
    return float(value)


def rho_token(value: float) -> str:
    return ("{:.3f}".format(float(value))).rstrip("0").rstrip(".").replace(".", "p")


def grid_id(cap: int, rho: float) -> str:
    return "cap{:03d}__rho{}".format(int(cap), rho_token(rho))


def safe_representative_sort(df: pd.DataFrame, priority_col: Optional[str] = None) -> pd.DataFrame:
    """Deterministic sort that avoids feature_uid index/column ambiguity."""
    out = df.copy().reset_index(drop=True)
    sort_cols: List[str] = []
    ascending: List[bool] = []
    if priority_col is not None and priority_col in out.columns:
        sort_cols.append(priority_col)
        ascending.append(True)
    for col, asc in [
        ("source_priority", True),
        ("state_complexity", True),
        ("summary_priority", True),
        ("compartment_priority", True),
        ("candidate_evidence_score", False),
        ("oof_metric", False),
        ("fold_sd", True),
        ("nonmissing_fraction", False),
        ("feature_uid", True),
    ]:
        if col in out.columns and col not in sort_cols:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return out
    return out.sort_values(sort_cols, ascending=ascending, na_position="last")


def get_modules(cfg: Mapping):
    iu = import_module_from_path(
        "stage2a_grid_interpretability_utils",
        cfg["interpretability_utils_path"],
    )
    s2a5 = import_module_from_path(
        "stage2a_grid_base_microcompression",
        cfg["base_stage2a5_script_path"],
    )
    # Patch the known pandas ambiguity safely at runtime.
    s2a5.representative_sort = safe_representative_sort
    return iu, s2a5


def validate_grid_values(cfg: Mapping) -> Tuple[List[int], List[float]]:
    caps = sorted(set(int(x) for x in cfg.get("candidate_caps", [])))
    rhos = sorted(set(round(float(x), 6) for x in cfg.get("semantic_rhos", [])))
    if not caps:
        raise ValueError("candidate_caps is empty")
    if min(caps) < 1:
        raise ValueError("candidate_caps must all be >=1")
    if max(caps) > int(cfg.get("master_candidate_cap", 100)):
        raise ValueError(
            "Grid cap {} exceeds master_candidate_cap={}; rebuild Stage 2A-4 or lower the grid.".format(
                max(caps), cfg.get("master_candidate_cap", 100)
            )
        )
    if not rhos:
        raise ValueError("semantic_rhos is empty")
    if min(rhos) < -1 or max(rhos) > 1:
        raise ValueError("semantic_rhos must be between -1 and 1")
    return caps, rhos


def config_paths(cfg: Mapping) -> Dict[str, Path]:
    root = Path(cfg["stage2a4_output_root"])
    return {
        "stage2a4_root": root,
        "stage2a4_manifest": root / "stage2a4_matrix_manifest.csv",
        "stage2a4_summary": root / "all_context_stage2a4_summary.csv",
        "output_root": Path(cfg["output_root"]),
    }


def command_validate(cfg: Mapping) -> None:
    caps, rhos = validate_grid_values(cfg)
    paths = config_paths(cfg)
    problems: List[str] = []
    for key in ["stage2a4_manifest", "stage2a4_summary"]:
        if not paths[key].exists():
            problems.append("missing {}: {}".format(key, paths[key]))
    for key in ["interpretability_utils_path", "base_stage2a5_script_path"]:
        p = Path(cfg[key])
        if not p.exists():
            problems.append("missing {}: {}".format(key, p))
    report = {
        "candidate_caps": caps,
        "semantic_rhos": rhos,
        "n_caps": len(caps),
        "n_rhos": len(rhos),
        "n_grid_cells_per_context": len(caps) * len(rhos),
        "fixed_residual_corr_threshold_by_panel": cfg.get("residual_corr_threshold_by_panel"),
        "fixed_max_oof_loss": cfg.get("max_oof_loss", 0.01),
        "problems": problems,
    }
    ensure_dir(paths["output_root"])
    write_json(report, paths["output_root"] / "grid_validation_report.json")
    print(json.dumps(report, indent=2))
    if problems:
        raise RuntimeError("Validation failed with {} problem(s)".format(len(problems)))


def command_inventory(cfg: Mapping) -> None:
    command_validate(cfg)
    paths = config_paths(cfg)
    outroot = ensure_dir(paths["output_root"])
    manifest = pd.read_csv(paths["stage2a4_manifest"])
    if manifest.empty:
        raise RuntimeError("Stage 2A-4 matrix manifest is empty")

    summary = pd.read_csv(paths["stage2a4_summary"]) if paths["stage2a4_summary"].exists() else pd.DataFrame()
    if not summary.empty and "context_strength" in summary.columns:
        strength_map = summary.drop_duplicates("context_id").set_index("context_id")["context_strength"].to_dict()
        manifest["context_strength"] = manifest["context_id"].map(strength_map)
    elif "context_strength" not in manifest.columns:
        manifest["context_strength"] = ""

    required = ["context_id", "context_slug", "matrix_path", "candidate_registry_path", *CONTEXT_COLS]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError("Stage 2A-4 manifest missing columns: {}".format(missing))

    keep_rows = []
    missing_rows = []
    for _, row in manifest.iterrows():
        matrix_path = Path(str(row["matrix_path"]))
        registry_path = Path(str(row["candidate_registry_path"]))
        if not matrix_path.exists() or not registry_path.exists():
            missing_rows.append({
                "context_id": row["context_id"],
                "matrix_path": str(matrix_path),
                "candidate_registry_path": str(registry_path),
                "matrix_exists": matrix_path.exists(),
                "registry_exists": registry_path.exists(),
            })
            continue
        keep_rows.append(row)

    if not keep_rows:
        raise RuntimeError("No complete Stage 2A-4 context matrices were found")
    index = pd.DataFrame(keep_rows).sort_values([c for c in CONTEXT_COLS if c in manifest.columns]).reset_index(drop=True)
    index.insert(0, "grid_array_id", np.arange(len(index), dtype=int))
    index.to_csv(outroot / "stage2a5_grid_context_index.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(outroot / "stage2a5_grid_missing_source_contexts.csv", index=False)
    write_json(cfg, outroot / "stage2a5_grid_config.resolved.json")
    log("[SAVE] {} contexts={}".format(outroot / "stage2a5_grid_context_index.csv", len(index)))


def resolve_array_id(value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or use SLURM_ARRAY_TASK_ID")
    return int(env)


def get_context_row(cfg: Mapping, array_id: int) -> pd.Series:
    path = Path(cfg["output_root"]) / "stage2a5_grid_context_index.csv"
    if not path.exists():
        raise FileNotFoundError("Run inventory first: {}".format(path))
    index = pd.read_csv(path)
    match = index[index["grid_array_id"].astype(int) == int(array_id)]
    if match.empty:
        raise IndexError("grid_array_id={} not found".format(array_id))
    return match.iloc[0]


def context_output_dir(cfg: Mapping, row: pd.Series) -> Path:
    return Path(cfg["output_root"]) / "contexts" / str(row["context_slug"])


def sort_master_seeds(registry: pd.DataFrame) -> pd.DataFrame:
    seeds = registry[registry["seed_pass_thresholds"].map(parse_bool)].copy()
    if "seed_rank" in seeds.columns and pd.to_numeric(seeds["seed_rank"], errors="coerce").notna().any():
        seeds["_seed_rank_num"] = pd.to_numeric(seeds["seed_rank"], errors="coerce")
        seeds = seeds.sort_values(
            ["_seed_rank_num", "candidate_evidence_score", "oof_metric"],
            ascending=[True, False, False],
            na_position="last",
        )
    else:
        seeds = seeds.sort_values(
            ["candidate_evidence_score", "oof_metric", "fold_sd", "nonmissing_fraction"],
            ascending=[False, False, True, False],
            na_position="last",
        )
    seeds = seeds.drop_duplicates("feature_uid", keep="first").reset_index(drop=True)
    seeds["grid_master_seed_rank"] = np.arange(1, len(seeds) + 1)
    return seeds


def load_rescue_links(registry_path: Path, registry: pd.DataFrame) -> pd.DataFrame:
    link_path = registry_path.parent / "rescue_candidate_links.csv"
    if link_path.exists():
        links = pd.read_csv(link_path)
        needed = {"seed_feature_uid", "rescue_feature_uid"}
        if needed.issubset(links.columns):
            return links

    # Fallback to semicolon-delimited linkage embedded in the registry.
    rows: List[dict] = []
    if "rescues_seed_feature_uids" in registry.columns:
        for _, row in registry.iterrows():
            rescue_uid = str(row.get("feature_uid"))
            raw = row.get("rescues_seed_feature_uids")
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            for seed_uid in str(raw).split(";"):
                seed_uid = seed_uid.strip()
                if seed_uid:
                    rows.append({"seed_feature_uid": seed_uid, "rescue_feature_uid": rescue_uid, "rescue_rule": "registry_fallback"})
    return pd.DataFrame(rows)


def derive_cap_registry(
    master_registry: pd.DataFrame,
    master_seeds: pd.DataFrame,
    rescue_links: pd.DataFrame,
    matrix: pd.DataFrame,
    cap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str], Set[str]]:
    selected_seeds = master_seeds.head(int(cap)).copy()
    selected_seed_uids = set(selected_seeds["feature_uid"].astype(str))

    selected_rescue_uids: Set[str] = set()
    if not rescue_links.empty:
        links = rescue_links[rescue_links["seed_feature_uid"].astype(str).isin(selected_seed_uids)].copy()
        selected_rescue_uids = set(links["rescue_feature_uid"].dropna().astype(str))

    selected_uids = selected_seed_uids | selected_rescue_uids
    selected_uids = {uid for uid in selected_uids if uid in matrix.columns}
    selected_seed_uids &= selected_uids
    selected_rescue_uids &= selected_uids

    registry = master_registry[master_registry["feature_uid"].astype(str).isin(selected_uids)].copy()
    registry = registry.drop_duplicates("feature_uid", keep="first").reset_index(drop=True)
    registry["seed_pass_thresholds"] = registry["feature_uid"].astype(str).isin(selected_seed_uids)
    registry["included_as_rescue"] = registry["feature_uid"].astype(str).isin(selected_rescue_uids)
    registry["candidate_role"] = np.where(
        registry["seed_pass_thresholds"] & registry["included_as_rescue"],
        "seed_and_rescue",
        np.where(registry["seed_pass_thresholds"], "seed", "rescue_only"),
    )
    registry["grid_candidate_cap"] = int(cap)

    keep_cols = ["patient_id"] + [uid for uid in registry["feature_uid"].astype(str) if uid in matrix.columns]
    cap_matrix = matrix[keep_cols].copy()
    return registry, cap_matrix, selected_seed_uids, selected_rescue_uids


def compress_one_grid_cell(
    *,
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    panel: str,
    semantic_rho: float,
    cfg: Mapping,
    iu,
    s2a5,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    registry = iu.add_interpretability_columns(registry)

    parser_problem = registry[
        registry["parser_status"].astype(str).ne("ok")
    ].copy()
    if not parser_problem.empty:
        raise RuntimeError(
            "Grid cell contains {} feature(s) that failed corrected parser".format(
                len(parser_problem)
            )
        )

    registry = registry[registry["feature_uid"].astype(str).isin(matrix.columns)].drop_duplicates("feature_uid").copy()
    active: Set[str] = set(registry["feature_uid"].astype(str))
    replacement: Dict[str, str] = {}
    all_audit: List[dict] = []
    all_pairs: List[dict] = []
    min_pairwise_n = int(cfg.get("min_pairwise_n", 20))
    max_oof_loss = float(cfg.get("max_oof_loss", 0.01))

    n_registry = len(active)
    audit, pairs = s2a5.exact_duplicate_pass(registry, matrix, active, replacement, iu, cfg)
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_exact = len(active)

    for rule_name, group_key, priority_col, require_location in [
        ("state_simplification", "state_simplification_key", "state_complexity", False),
        ("metric_summary_simplification", "metric_simplification_key", "summary_priority", True),
        ("compartment_simplification", "compartment_simplification_key", "compartment_priority", False),
    ]:
        audit, pairs = s2a5.simplification_pass(
            rule_name=rule_name,
            registry=registry,
            matrix=matrix,
            active=active,
            replacement=replacement,
            group_key=group_key,
            priority_col=priority_col,
            corr_threshold=float(semantic_rho),
            max_oof_loss=max_oof_loss,
            min_pairwise_n=min_pairwise_n,
            iu=iu,
            require_location_summary=require_location,
        )
        all_audit.extend(audit); all_pairs.extend(pairs)
        if rule_name == "state_simplification":
            n_after_state = len(active)
        elif rule_name == "metric_summary_simplification":
            n_after_metric = len(active)
        else:
            n_after_compartment = len(active)

    residual_rho = panel_value(cfg, "residual_corr_threshold_by_panel", panel, 0.95)
    residual_loss = panel_value(cfg, "residual_max_oof_loss_by_panel", panel, max_oof_loss)
    audit, pairs = s2a5.residual_greedy_pass(
        registry=registry,
        matrix=matrix,
        active=active,
        replacement=replacement,
        corr_threshold=residual_rho,
        max_oof_loss=residual_loss,
        min_pairwise_n=min_pairwise_n,
        iu=iu,
    )
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_residual = len(active)

    seed_uids = set(registry.loc[registry["seed_pass_thresholds"].map(parse_bool), "feature_uid"].astype(str))
    represented_by_final: Dict[str, List[str]] = defaultdict(list)
    represented_seeds_by_final: Dict[str, List[str]] = defaultdict(list)
    seed_to_final_rows: List[dict] = []

    for uid in registry["feature_uid"].astype(str):
        final_uid = s2a5.resolve_rep(uid, replacement)
        represented_by_final[final_uid].append(uid)
    for seed_uid in sorted(seed_uids):
        final_uid = s2a5.resolve_rep(seed_uid, replacement)
        represented_seeds_by_final[final_uid].append(seed_uid)
        seed_to_final_rows.append({
            "seed_feature_uid": seed_uid,
            "final_representative_uid": final_uid,
            "seed_changed_representative": seed_uid != final_uid,
        })

    final_uids = sorted(represented_seeds_by_final.keys())
    unused_rescues = sorted(active - set(final_uids))
    for uid in unused_rescues:
        all_audit.append({
            "removed_feature_uid": uid,
            "representative_feature_uid": "",
            "compression_rule": "unused_rescue_pruned",
            "decision_reason": "rescue-only feature did not represent a retained seed",
        })

    final = registry[registry["feature_uid"].astype(str).isin(final_uids)].copy()
    final["represented_feature_uids"] = final["feature_uid"].map(
        lambda uid: ";".join(sorted(set(represented_by_final.get(str(uid), [str(uid)]))))
    )
    final["represented_seed_feature_uids"] = final["feature_uid"].map(
        lambda uid: ";".join(sorted(set(represented_seeds_by_final.get(str(uid), []))))
    )
    final["n_represented_features"] = final["feature_uid"].map(
        lambda uid: len(set(represented_by_final.get(str(uid), [str(uid)])))
    )
    final["n_represented_seeds"] = final["feature_uid"].map(
        lambda uid: len(set(represented_seeds_by_final.get(str(uid), [])))
    )
    final = safe_representative_sort(final).reset_index(drop=True)
    final["compressed_context_rank"] = np.arange(1, len(final) + 1)
    final["candidate_score"] = final.get("candidate_evidence_score", np.nan)
    final["primary_oof_metric"] = final.get("oof_metric", np.nan)
    final["primary_delta_metric"] = final.get("delta_clinical", np.nan)
    if "selected_transform_mode" not in final.columns and "transform_mode" in final.columns:
        final["selected_transform_mode"] = final["transform_mode"]

    audit_df = pd.DataFrame(all_audit)
    seed_map = pd.DataFrame(seed_to_final_rows)
    summary = {
        "n_registry_features": n_registry,
        "n_seed_features": len(seed_uids),
        "n_rescue_only_features": int((~registry["seed_pass_thresholds"].map(parse_bool)).sum()),
        "n_after_exact": n_after_exact,
        "n_after_state": n_after_state,
        "n_after_metric": n_after_metric,
        "n_after_compartment": n_after_compartment,
        "n_after_residual": n_after_residual,
        "n_unused_rescues_pruned": len(unused_rescues),
        "n_final_representatives": len(final),
        "compression_fraction_seed_to_final": len(final) / len(seed_uids) if seed_uids else np.nan,
        "n_state_removed": int((audit_df.get("compression_rule", pd.Series(dtype=str)) == "state_simplification").sum()),
        "n_metric_removed": int((audit_df.get("compression_rule", pd.Series(dtype=str)) == "metric_summary_simplification").sum()),
        "n_compartment_removed": int((audit_df.get("compression_rule", pd.Series(dtype=str)) == "compartment_simplification").sum()),
        "n_residual_removed": int((audit_df.get("compression_rule", pd.Series(dtype=str)) == "residual_redundancy").sum()),
        "residual_rho": residual_rho,
        "max_oof_loss": max_oof_loss,
    }
    return final, summary, audit_df, seed_map


def command_worker(cfg: Mapping, array_id: int, force: bool = False) -> None:
    caps, rhos = validate_grid_values(cfg)
    row = get_context_row(cfg, array_id)
    cdir = ensure_dir(context_output_dir(cfg, row))
    done_path = cdir / ".done"
    if done_path.exists() and not force:
        log("[SKIP] already complete: {}".format(row["context_id"]))
        return

    log("=" * 88)
    log("[GRID WORKER {}] {}".format(array_id, row["context_id"]))
    registry_path = Path(str(row["candidate_registry_path"]))
    master_registry = pd.read_csv(registry_path)
    matrix = read_table(row["matrix_path"])
    if master_registry.empty or matrix.empty or matrix.shape[1] <= 1:
        raise RuntimeError("Empty registry or matrix for {}".format(row["context_id"]))

    iu, s2a5 = get_modules(cfg)
    master_registry = iu.add_interpretability_columns(master_registry)
    built_uids = [uid for uid in master_registry["feature_uid"].astype(str) if uid in matrix.columns]
    master_registry = master_registry[master_registry["feature_uid"].astype(str).isin(built_uids)].drop_duplicates("feature_uid").copy()
    master_seeds = sort_master_seeds(master_registry)
    rescue_links = load_rescue_links(registry_path, master_registry)

    max_requested = max(caps)
    if len(master_seeds) < max_requested:
        log("[INFO] only {} master seeds are available; higher caps will saturate naturally".format(len(master_seeds)))

    summary_rows: List[dict] = []
    manifest_parts: List[pd.DataFrame] = []
    audit_parts: List[pd.DataFrame] = []
    seedmap_parts: List[pd.DataFrame] = []
    seed_selection_rows: List[dict] = []

    for cap in caps:
        cap_registry, cap_matrix, seed_uids, rescue_uids = derive_cap_registry(
            master_registry, master_seeds, rescue_links, matrix, cap
        )
        for rank, uid in enumerate(master_seeds.head(cap)["feature_uid"].astype(str), start=1):
            if uid in seed_uids:
                seed_selection_rows.append({
                    "context_id": row["context_id"],
                    "candidate_cap": cap,
                    "seed_rank_within_context": rank,
                    "feature_uid": uid,
                })

        for rho in rhos:
            gid = grid_id(cap, rho)
            final, comp_summary, audit_df, seed_map = compress_one_grid_cell(
                registry=cap_registry,
                matrix=cap_matrix,
                panel=str(row["panel"]),
                semantic_rho=rho,
                cfg=cfg,
                iu=iu,
                s2a5=s2a5,
            )

            base = {
                "grid_id": gid,
                "candidate_cap": int(cap),
                "semantic_rho": float(rho),
                "grid_array_id": int(array_id),
                "context_id": row["context_id"],
                "context_slug": row["context_slug"],
                "context_strength": row.get("context_strength", ""),
                **{c: row[c] for c in CONTEXT_COLS},
                "n_master_seed_features": len(master_seeds),
                "n_selected_seed_features": len(seed_uids),
                "n_linked_rescue_features": len(rescue_uids),
            }
            summary_rows.append({**base, **comp_summary})

            if not final.empty:
                final = final.copy()
                for key, value in base.items():
                    final[key] = value
                # Canonical biological identity intentionally excludes cohort,
                # endpoint, transform, and prep-root duplication.
                final["canonical_feature_id"] = (
                    final["panel"].astype(str) + "||" + final["full_semantic_key"].astype(str)
                )
                manifest_parts.append(final)

            if bool(cfg.get("save_compression_decision_audits", True)) and not audit_df.empty:
                audit_df = audit_df.copy()
                for key, value in base.items():
                    audit_df[key] = value
                audit_parts.append(audit_df)

            if bool(cfg.get("save_seed_maps", True)) and not seed_map.empty:
                seed_map = seed_map.copy()
                for key, value in base.items():
                    seed_map[key] = value
                seedmap_parts.append(seed_map)

    pd.DataFrame(summary_rows).to_csv(cdir / "context_grid_summary.csv", index=False)
    save_table(
        pd.concat(manifest_parts, ignore_index=True, sort=False) if manifest_parts else pd.DataFrame(),
        cdir / "context_grid_manifest.parquet",
    )
    save_table(
        pd.concat(audit_parts, ignore_index=True, sort=False) if audit_parts else pd.DataFrame(),
        cdir / "context_grid_compression_audit.parquet",
    )
    save_table(
        pd.concat(seedmap_parts, ignore_index=True, sort=False) if seedmap_parts else pd.DataFrame(),
        cdir / "context_grid_seed_to_final.parquet",
    )
    pd.DataFrame(seed_selection_rows).to_csv(cdir / "context_cap_seed_selection.csv", index=False)
    done_path.write_text("complete\n")
    log("[DONE] grid cells={} summary_rows={} manifest_rows={}".format(
        len(caps) * len(rhos), len(summary_rows), sum(len(x) for x in manifest_parts)
    ))


def strength_is_strong(value: object) -> bool:
    text = str(value).strip().lower().replace("_", "-")
    return "strong" in text and "weak" not in text


def context_canonical_deduplicate(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    work = manifest.copy()
    for col, default in [
        ("source_priority", 99),
        ("state_complexity", 99),
        ("summary_priority", 99),
        ("compartment_priority", 99),
        ("candidate_score", -np.inf),
        ("primary_oof_metric", -np.inf),
        ("fold_sd", np.inf),
    ]:
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    group_cols = [*GRID_COLS, "panel", "context_id", "canonical_feature_id"]
    work = work.sort_values(
        group_cols + [
            "source_priority", "state_complexity", "summary_priority", "compartment_priority",
            "candidate_score", "primary_oof_metric", "fold_sd", "feature_uid",
        ],
        ascending=[True] * len(group_cols) + [True, True, True, True, False, False, True, True],
        na_position="last",
    )
    dedup = work.drop_duplicates(group_cols, keep="first").reset_index(drop=True)
    return dedup


def choose_global_representation(group: pd.DataFrame) -> pd.Series:
    work = safe_representative_sort(group)
    return work.iloc[0]


def calculate_candidate_support(dedup_manifest: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    if dedup_manifest.empty:
        return pd.DataFrame()

    group_cols = [*GRID_COLS, "panel", "canonical_feature_id"]
    rows: List[dict] = []
    for keys, group in dedup_manifest.groupby(group_cols, dropna=False):
        cap, rho, panel, canonical_id = keys
        global_rep = choose_global_representation(group)
        contexts = sorted(set(group["context_id"].astype(str)))
        cohorts = sorted(set(group["cohort"].astype(str)))
        endpoints = sorted(set(group["endpoint"].astype(str)))
        strengths = sorted(set(group.get("context_strength", pd.Series(dtype=str)).dropna().astype(str)))
        scores = pd.to_numeric(group.get("candidate_score"), errors="coerce")
        oofs = pd.to_numeric(group.get("primary_oof_metric"), errors="coerce")
        rows.append({
            "candidate_cap": int(cap),
            "semantic_rho": float(rho),
            "grid_id": grid_id(int(cap), float(rho)),
            "panel": str(panel),
            "canonical_feature_id": str(canonical_id),
            "global_representative_feature_uid": str(global_rep.get("feature_uid")),
            "global_representative_feature_source": global_rep.get("feature_source"),
            "global_representative_feature_group": global_rep.get("feature_group"),
            "global_representative_feature": global_rep.get("feature"),
            "feature_kind": global_rep.get("feature_kind"),
            "metric_kind": global_rep.get("metric_kind"),
            "full_semantic_key": global_rep.get("full_semantic_key"),
            "n_contexts": len(contexts),
            "n_cohorts": len(cohorts),
            "n_endpoints": len(endpoints),
            "contexts": ";".join(contexts),
            "cohorts": ";".join(cohorts),
            "endpoints": ";".join(endpoints),
            "context_strengths": ";".join(strengths),
            "any_strong_context": any(strength_is_strong(x) for x in strengths),
            "max_candidate_score": float(scores.max()) if scores.notna().any() else np.nan,
            "median_candidate_score": float(scores.median()) if scores.notna().any() else np.nan,
            "max_oof_metric": float(oofs.max()) if oofs.notna().any() else np.nan,
            "median_oof_metric": float(oofs.median()) if oofs.notna().any() else np.nan,
        })
    support = pd.DataFrame(rows)
    support["support_set_S1"] = support["n_cohorts"] >= 1
    support["support_set_S2"] = support["n_cohorts"] >= 2

    # Evidence percentile is calculated within each panel/grid cell.
    support["panel_grid_evidence_percentile"] = (
        support.groupby(GRID_COLS + ["panel"])["median_candidate_score"]
        .rank(pct=True, ascending=True, method="average")
    )

    exc_cfg = cfg.get("exceptional_single_cohort", {}) or {}
    require_strong = bool(exc_cfg.get("require_strong_context", True))
    min_contexts = int(exc_cfg.get("min_contexts_within_cohort", 2))
    evidence_q = float(exc_cfg.get("panel_evidence_quantile", 0.95))
    support["exceptional_eligible"] = (
        (support["n_cohorts"] == 1)
        & ((support["any_strong_context"]) if require_strong else True)
        & (
            (support["n_contexts"] >= min_contexts)
            | (support["panel_grid_evidence_percentile"] >= evidence_q)
        )
    )
    support["exceptional_selected"] = False
    support["exceptional_rank"] = np.nan
    support["exceptional_selection_reason"] = ""

    if bool(exc_cfg.get("enabled", True)):
        max_per_panel = int(exc_cfg.get("max_per_panel", 10))
        max_per_cohort = int(exc_cfg.get("max_per_cohort", 3))
        for keys, group in support.groupby(GRID_COLS + ["panel"], dropna=False):
            candidates = group[group["exceptional_eligible"]].copy()
            if candidates.empty:
                continue
            candidates["single_cohort"] = candidates["cohorts"].astype(str).str.split(";").str[0]
            candidates = candidates.sort_values(
                ["n_contexts", "panel_grid_evidence_percentile", "median_candidate_score", "max_oof_metric", "canonical_feature_id"],
                ascending=[False, False, False, False, True],
                na_position="last",
            )
            selected_indices: List[int] = []
            cohort_counts: Dict[str, int] = defaultdict(int)
            for idx, candidate in candidates.iterrows():
                cohort = str(candidate["single_cohort"])
                if len(selected_indices) >= max_per_panel:
                    break
                if cohort_counts[cohort] >= max_per_cohort:
                    continue
                selected_indices.append(idx)
                cohort_counts[cohort] += 1
            for rank, idx in enumerate(selected_indices, start=1):
                support.loc[idx, "exceptional_selected"] = True
                support.loc[idx, "exceptional_rank"] = rank
                support.loc[idx, "exceptional_selection_reason"] = (
                    "single-cohort; strong-context requirement met; recurring within cohort or top evidence quantile; "
                    "selected under panel/cohort quotas"
                )

    support["support_set_S2_plus_E"] = support["support_set_S2"] | support["exceptional_selected"]
    return support


def aggregate_per_cohort(dedup_manifest: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    if dedup_manifest.empty:
        return pd.DataFrame()
    support_key = support[[*GRID_COLS, "panel", "canonical_feature_id", "n_cohorts"]].copy()
    merged = dedup_manifest.merge(
        support_key,
        on=[*GRID_COLS, "panel", "canonical_feature_id"],
        how="left",
    )
    rows: List[dict] = []
    group_cols = [*GRID_COLS, "panel", "cohort"]
    for keys, group in merged.groupby(group_cols, dropna=False):
        cap, rho, panel, cohort = keys
        unique = group.drop_duplicates("canonical_feature_id")
        rows.append({
            "candidate_cap": int(cap),
            "semantic_rho": float(rho),
            "grid_id": grid_id(int(cap), float(rho)),
            "panel": str(panel),
            "cohort": str(cohort),
            "n_context_level_records": int(len(group)),
            "n_unique_canonical_features": int(unique["canonical_feature_id"].nunique()),
            "n_unique_features_supported_elsewhere": int(unique.loc[unique["n_cohorts"] >= 2, "canonical_feature_id"].nunique()),
            "n_unique_single_cohort_features": int(unique.loc[unique["n_cohorts"] == 1, "canonical_feature_id"].nunique()),
            "n_contexts": int(group["context_id"].nunique()),
        })
    return pd.DataFrame(rows)


def aggregate_grid_metrics(
    context_summary: pd.DataFrame,
    raw_manifest: pd.DataFrame,
    dedup_manifest: pd.DataFrame,
    support: pd.DataFrame,
    per_cohort: pd.DataFrame,
    cfg: Mapping,
) -> pd.DataFrame:
    rows: List[dict] = []
    for keys, group in context_summary.groupby([*GRID_COLS, "panel"], dropna=False):
        cap, rho, panel = keys
        raw_g = raw_manifest[
            (raw_manifest["candidate_cap"].astype(int) == int(cap))
            & np.isclose(raw_manifest["semantic_rho"].astype(float), float(rho))
            & (raw_manifest["panel"].astype(str) == str(panel))
        ]
        dedup_g = dedup_manifest[
            (dedup_manifest["candidate_cap"].astype(int) == int(cap))
            & np.isclose(dedup_manifest["semantic_rho"].astype(float), float(rho))
            & (dedup_manifest["panel"].astype(str) == str(panel))
        ]
        supp_g = support[
            (support["candidate_cap"].astype(int) == int(cap))
            & np.isclose(support["semantic_rho"].astype(float), float(rho))
            & (support["panel"].astype(str) == str(panel))
        ]
        cohort_g = per_cohort[
            (per_cohort["candidate_cap"].astype(int) == int(cap))
            & np.isclose(per_cohort["semantic_rho"].astype(float), float(rho))
            & (per_cohort["panel"].astype(str) == str(panel))
        ]

        n_raw = int(len(raw_g))
        n_context_dedup = int(len(dedup_g))
        n_unique = int(supp_g["canonical_feature_id"].nunique())
        n_s2 = int(supp_g["support_set_S2"].sum())
        n_exc = int(supp_g["exceptional_selected"].sum())
        n_s2e = int(supp_g["support_set_S2_plus_E"].sum())
        max_coverage = (
            float(cohort_g["n_unique_canonical_features"].max() / n_unique)
            if n_unique > 0 and not cohort_g.empty else np.nan
        )
        max_single_owned = (
            float(cohort_g["n_unique_single_cohort_features"].max() / n_unique)
            if n_unique > 0 and not cohort_g.empty else np.nan
        )

        support_counts = supp_g["n_cohorts"].value_counts().to_dict()
        rows.append({
            "candidate_cap": int(cap),
            "semantic_rho": float(rho),
            "grid_id": grid_id(int(cap), float(rho)),
            "panel": str(panel),
            "n_contexts": int(group["context_id"].nunique()),
            "n_seed_records": int(group["n_selected_seed_features"].sum()),
            "n_precompression_registry_records": int(group["n_registry_features"].sum()),
            "n_postcompression_records_raw": n_raw,
            "n_postcompression_records_after_within_context_dedup": n_context_dedup,
            "n_within_context_canonical_duplicates_removed": n_raw - n_context_dedup,
            "n_unique_canonical_features": n_unique,
            "n_duplicates_removed_across_contexts": n_context_dedup - n_unique,
            "duplicate_collapse_fraction_raw_to_unique": (n_raw - n_unique) / n_raw if n_raw else np.nan,
            "n_support_exactly_1_cohort": int(support_counts.get(1, 0)),
            "n_support_exactly_2_cohorts": int(support_counts.get(2, 0)),
            "n_support_exactly_3_cohorts": int(support_counts.get(3, 0)),
            "n_support_exactly_4_cohorts": int(support_counts.get(4, 0)),
            "n_S1": n_unique,
            "n_S2": n_s2,
            "n_exceptional_single_cohort": n_exc,
            "n_S2_plus_E": n_s2e,
            "proportion_support_ge2": n_s2 / n_unique if n_unique else np.nan,
            "max_cohort_coverage_fraction": max_coverage,
            "max_single_cohort_owned_fraction": max_single_owned,
            "median_context_compression_fraction": float(pd.to_numeric(group["compression_fraction_seed_to_final"], errors="coerce").median()),
            "state_removed": int(group["n_state_removed"].sum()),
            "metric_removed": int(group["n_metric_removed"].sum()),
            "compartment_removed": int(group["n_compartment_removed"].sum()),
            "residual_removed": int(group["n_residual_removed"].sum()),
        })
    metrics = pd.DataFrame(rows)

    guard = cfg.get("quantity_guardrails", {}) or {}
    min_n = int(guard.get("min_S2_plus_E_per_panel", 60))
    max_n = int(guard.get("max_S2_plus_E_per_panel", 125))
    min_prop = float(guard.get("min_proportion_support_ge2", 0.20))
    max_single = float(guard.get("max_single_cohort_owned_fraction", 0.40))
    midpoint = (min_n + max_n) / 2.0
    metrics["passes_candidate_count_guardrail"] = metrics["n_S2_plus_E"].between(min_n, max_n)
    metrics["passes_support_guardrail"] = metrics["proportion_support_ge2"] >= min_prop
    metrics["passes_cohort_balance_guardrail"] = metrics["max_single_cohort_owned_fraction"] <= max_single
    metrics["passes_all_quantity_guardrails"] = (
        metrics["passes_candidate_count_guardrail"]
        & metrics["passes_support_guardrail"]
        & metrics["passes_cohort_balance_guardrail"]
    )
    metrics["distance_from_candidate_target_midpoint"] = (metrics["n_S2_plus_E"] - midpoint).abs()
    return metrics


def make_joint_grid_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    value_cols = [
        "n_seed_records", "n_postcompression_records_raw", "n_unique_canonical_features",
        "n_S2", "n_exceptional_single_cohort", "n_S2_plus_E", "proportion_support_ge2",
        "max_single_cohort_owned_fraction", "passes_all_quantity_guardrails",
        "distance_from_candidate_target_midpoint",
    ]
    wide = metrics.pivot_table(index=GRID_COLS + ["grid_id"], columns="panel", values=value_cols, aggfunc="first")
    wide.columns = ["{}_{}".format(metric, panel) for metric, panel in wide.columns]
    wide = wide.reset_index()
    panel_pass_cols = [c for c in wide.columns if c.startswith("passes_all_quantity_guardrails_")]
    if panel_pass_cols:
        wide["both_panels_pass_quantity_guardrails"] = wide[panel_pass_cols].fillna(False).all(axis=1)
    distance_cols = [c for c in wide.columns if c.startswith("distance_from_candidate_target_midpoint_")]
    wide["joint_distance_from_target"] = wide[distance_cols].sum(axis=1) if distance_cols else np.nan
    support_cols = [c for c in wide.columns if c.startswith("proportion_support_ge2_")]
    wide["mean_proportion_support_ge2"] = wide[support_cols].mean(axis=1) if support_cols else np.nan
    wide = wide.sort_values(
        ["both_panels_pass_quantity_guardrails", "joint_distance_from_target", "mean_proportion_support_ge2", "candidate_cap", "semantic_rho"],
        ascending=[False, True, False, True, False],
        na_position="last",
    )
    wide["quantity_screen_rank"] = np.arange(1, len(wide) + 1)
    return wide


def heatmap_plot(
    data: pd.DataFrame,
    value_col: str,
    path: Path,
    title: str,
    fmt: str = "int",
) -> None:
    if data.empty or value_col not in data.columns:
        return
    pivot = data.pivot_table(index="semantic_rho", columns="candidate_cap", values=value_col, aggfunc="first")
    pivot = pivot.sort_index(ascending=False)
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig_w = max(12, 0.72 * len(pivot.columns) + 4)
    fig_h = max(7, 0.58 * len(pivot.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(masked, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(["{:.2f}".format(float(x)) for x in pivot.index])
    ax.set_xlabel("Maximum seed candidates per context")
    ax.set_ylabel("Semantic Spearman threshold")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            if fmt == "int":
                text = "{:,.0f}".format(v)
            elif fmt == "pct":
                text = "{:.2f}".format(v)
            else:
                text = "{:.3f}".format(v)
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def line_plot(data: pd.DataFrame, value_col: str, path: Path, title: str, ylabel: str) -> None:
    if data.empty or value_col not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for rho, group in data.groupby("semantic_rho"):
        group = group.sort_values("candidate_cap")
        ax.plot(group["candidate_cap"], group[value_col], marker="o", markersize=3, label="rho={:.2f}".format(float(rho)))
    ax.set_xlabel("Maximum seed candidates per context")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def context_seed_heatmap(summary: pd.DataFrame, panel: str, path: Path) -> None:
    sub = summary[summary["panel"].astype(str) == panel].copy()
    if sub.empty:
        return
    # Seed counts do not depend on rho; take one row per cap/context.
    sub = sub.sort_values("semantic_rho").drop_duplicates(["context_id", "candidate_cap"], keep="first")
    pivot = sub.pivot(index="context_id", columns="candidate_cap", values="n_selected_seed_features")
    fig_w = max(12, 0.72 * len(pivot.columns) + 5)
    fig_h = max(7, 0.45 * len(pivot.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vals = pivot.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(vals), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=7)
    ax.set_xlabel("Maximum seed candidates per context")
    ax.set_ylabel("Context")
    ax.set_title("{}: seed candidates submitted by each context".format(panel))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, "{:.0f}".format(vals[i, j]), ha="center", va="center", fontsize=5.5)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def context_postcompression_heatmaps(summary: pd.DataFrame, panel: str, root: Path) -> None:
    sub = summary[summary["panel"].astype(str) == panel].copy()
    if sub.empty:
        return
    for rho, group in sub.groupby("semantic_rho"):
        pivot = group.pivot(index="context_id", columns="candidate_cap", values="n_final_representatives")
        fig_w = max(12, 0.72 * len(pivot.columns) + 5)
        fig_h = max(7, 0.45 * len(pivot.index) + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vals = pivot.to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(vals), aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str), fontsize=7)
        ax.set_xlabel("Maximum seed candidates per context")
        ax.set_ylabel("Context")
        ax.set_title("{}: post-microcompression representatives by context | semantic rho={:.2f}".format(panel, float(rho)))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, "{:.0f}".format(vals[i, j]), ha="center", va="center", fontsize=5.5)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = root / "context_postcompression_rho_{}.png".format(rho_token(float(rho)))
        ensure_dir(path.parent)
        fig.savefig(path, dpi=190)
        plt.close(fig)


def create_all_plots(
    metrics: pd.DataFrame,
    context_summary: pd.DataFrame,
    per_cohort: pd.DataFrame,
    cfg: Mapping,
) -> pd.DataFrame:
    plot_rows: List[dict] = []
    plot_root = ensure_dir(Path(cfg["output_root"]) / "plots")
    metric_specs = [
        ("n_seed_records", "01_seed_records", "Total seed records", "int"),
        ("n_postcompression_records_raw", "02_postcompression_records", "Post-microcompression context records", "int"),
        ("n_unique_canonical_features", "03_unique_canonical", "Unique canonical features after cross-context collapse", "int"),
        ("n_S2", "04_support_ge2", "Candidates supported in at least two cohorts (S2)", "int"),
        ("n_S2_plus_E", "05_S2_plus_exceptional", "S2 plus exceptional single-cohort candidates", "int"),
        ("proportion_support_ge2", "06_prop_support_ge2", "Proportion of canonical candidates with >=2-cohort support", "pct"),
        ("duplicate_collapse_fraction_raw_to_unique", "07_duplicate_collapse_fraction", "Fraction removed by canonical collapse", "pct"),
        ("max_single_cohort_owned_fraction", "08_max_single_cohort_owned_fraction", "Largest single-cohort-only contribution fraction", "pct"),
    ]
    curve_specs = [
        ("n_postcompression_records_raw", "postcompression_records", "Post-microcompression records"),
        ("n_unique_canonical_features", "unique_canonical", "Unique canonical features"),
        ("n_S2", "support_ge2", "Candidates with >=2-cohort support"),
        ("n_S2_plus_E", "S2_plus_E", "S2 plus exceptional candidates"),
        ("proportion_support_ge2", "prop_support_ge2", "Proportion with >=2-cohort support"),
    ]

    for panel in sorted(metrics["panel"].astype(str).unique()):
        pdir = ensure_dir(plot_root / panel)
        panel_metrics = metrics[metrics["panel"].astype(str) == panel].copy()
        for col, stem, title, fmt in metric_specs:
            path = pdir / "{}_{}.png".format(stem, panel)
            heatmap_plot(panel_metrics, col, path, "{}: {}".format(panel, title), fmt=fmt)
            plot_rows.append({"panel": panel, "plot_type": "grid_heatmap", "metric": col, "path": str(path)})
        for col, stem, ylabel in curve_specs:
            path = pdir / "curve_{}_{}.png".format(stem, panel)
            line_plot(panel_metrics, col, path, "{}: {} across the full grid".format(panel, ylabel), ylabel)
            plot_rows.append({"panel": panel, "plot_type": "curve", "metric": col, "path": str(path)})

        path = pdir / "context_seed_counts_by_cap_{}.png".format(panel)
        context_seed_heatmap(context_summary, panel, path)
        plot_rows.append({"panel": panel, "plot_type": "context_seed_heatmap", "metric": "n_selected_seed_features", "path": str(path)})

        context_root = ensure_dir(pdir / "context_postcompression")
        context_postcompression_heatmaps(context_summary, panel, context_root)
        for rho in sorted(context_summary.loc[context_summary["panel"].astype(str) == panel, "semantic_rho"].unique()):
            path = context_root / "context_postcompression_rho_{}.png".format(rho_token(float(rho)))
            plot_rows.append({"panel": panel, "plot_type": "context_postcompression_heatmap", "metric": "n_final_representatives", "semantic_rho": rho, "path": str(path)})

        cohort_dir = ensure_dir(pdir / "cohort_contributions")
        for cohort in sorted(per_cohort.loc[per_cohort["panel"].astype(str) == panel, "cohort"].astype(str).unique()):
            cohort_data = per_cohort[(per_cohort["panel"].astype(str) == panel) & (per_cohort["cohort"].astype(str) == cohort)]
            for col, suffix, title in [
                ("n_unique_canonical_features", "unique", "unique canonical features contributed"),
                ("n_unique_single_cohort_features", "single_cohort_only", "single-cohort-only canonical features"),
            ]:
                path = cohort_dir / "{}_{}_{}.png".format(cohort, suffix, panel)
                heatmap_plot(cohort_data, col, path, "{} / {}: {}".format(panel, cohort, title), fmt="int")
                plot_rows.append({"panel": panel, "cohort": cohort, "plot_type": "cohort_heatmap", "metric": col, "path": str(path)})

    return pd.DataFrame(plot_rows)


def command_aggregate(cfg: Mapping) -> None:
    outroot = ensure_dir(cfg["output_root"])
    index_path = outroot / "stage2a5_grid_context_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("Run inventory first")
    index = pd.read_csv(index_path)

    summaries: List[pd.DataFrame] = []
    manifests: List[pd.DataFrame] = []
    audits: List[pd.DataFrame] = []
    seedmaps: List[pd.DataFrame] = []
    missing: List[dict] = []

    for _, row in index.iterrows():
        cdir = context_output_dir(cfg, row)
        summary_path = cdir / "context_grid_summary.csv"
        manifest_path = cdir / "context_grid_manifest.parquet"
        if not summary_path.exists():
            missing.append({"context_id": row["context_id"], "reason": "missing_context_grid_summary"})
            continue
        summaries.append(pd.read_csv(summary_path))
        try:
            manifests.append(read_table(manifest_path))
        except FileNotFoundError:
            missing.append({"context_id": row["context_id"], "reason": "missing_context_grid_manifest"})
        audit_path = cdir / "context_grid_compression_audit.parquet"
        seedmap_path = cdir / "context_grid_seed_to_final.parquet"
        try:
            audit = read_table(audit_path)
            if not audit.empty:
                audits.append(audit)
        except FileNotFoundError:
            pass
        try:
            seedmap = read_table(seedmap_path)
            if not seedmap.empty:
                seedmaps.append(seedmap)
        except FileNotFoundError:
            pass

    context_summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    raw_manifest = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()
    if context_summary.empty or raw_manifest.empty:
        raise RuntimeError("No grid worker outputs were available for aggregation")

    dedup_manifest = context_canonical_deduplicate(raw_manifest)
    support = calculate_candidate_support(dedup_manifest, cfg)
    per_cohort = aggregate_per_cohort(dedup_manifest, support)
    metrics = aggregate_grid_metrics(context_summary, raw_manifest, dedup_manifest, support, per_cohort, cfg)
    joint = make_joint_grid_summary(metrics)

    context_summary.to_csv(outroot / "grid_context_counts.csv", index=False)
    save_table(raw_manifest, outroot / "all_grid_context_manifest_raw.parquet")
    save_table(dedup_manifest, outroot / "all_grid_context_manifest_deduplicated.parquet")
    save_table(support, outroot / "candidate_support_grid.parquet")
    per_cohort.to_csv(outroot / "grid_cohort_counts.csv", index=False)
    metrics.to_csv(outroot / "grid_panel_summary_metrics.csv", index=False)
    joint.to_csv(outroot / "grid_joint_quantity_screen.csv", index=False)
    pd.DataFrame(missing).to_csv(outroot / "stage2a5_grid_missing_context_outputs.csv", index=False)
    save_table(
        pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame(),
        outroot / "all_grid_compression_decision_audit.parquet",
    )
    save_table(
        pd.concat(seedmaps, ignore_index=True, sort=False) if seedmaps else pd.DataFrame(),
        outroot / "all_grid_seed_to_final_representative.parquet",
    )

    exceptional = support[support["exceptional_selected"]].copy()
    exceptional.to_csv(outroot / "exceptional_single_cohort_candidates_grid.csv", index=False)
    support[support["support_set_S2_plus_E"]].to_csv(outroot / "candidate_set_S2_plus_E_grid.csv", index=False)

    plot_index = create_all_plots(metrics, context_summary, per_cohort, cfg)
    plot_index.to_csv(outroot / "grid_plot_index.csv", index=False)

    log("[DONE] grid aggregation")
    log("[SAVE] {}".format(outroot / "grid_panel_summary_metrics.csv"))
    log("[SAVE] {}".format(outroot / "grid_joint_quantity_screen.csv"))
    log("[INFO] raw_manifest_rows={} context_dedup_rows={} canonical_support_rows={}".format(
        len(raw_manifest), len(dedup_manifest), len(support)
    ))


def select_support_column(name: str) -> str:
    normalized = str(name).strip().upper().replace("+", "PLUS").replace("_", "")
    mapping = {
        "S1": "support_set_S1",
        "S2": "support_set_S2",
        "S2E": "support_set_S2_plus_E",
        "S2PLUSE": "support_set_S2_plus_E",
    }
    if normalized not in mapping:
        raise ValueError("support-set must be S1, S2, or S2E")
    return mapping[normalized]


def command_export(
    cfg: Mapping,
    candidate_cap: int,
    semantic_rho: float,
    support_set: str,
    output_dir: Optional[str],
) -> None:
    outroot = Path(cfg["output_root"])
    dedup = read_table(outroot / "all_grid_context_manifest_deduplicated.parquet")
    support = read_table(outroot / "candidate_support_grid.parquet")
    support_col = select_support_column(support_set)
    rho = float(semantic_rho)
    cap = int(candidate_cap)

    dedup_cell = dedup[
        (dedup["candidate_cap"].astype(int) == cap)
        & np.isclose(dedup["semantic_rho"].astype(float), rho)
    ].copy()
    support_cell = support[
        (support["candidate_cap"].astype(int) == cap)
        & np.isclose(support["semantic_rho"].astype(float), rho)
        & support[support_col].map(parse_bool)
    ].copy()
    if dedup_cell.empty or support_cell.empty:
        raise RuntimeError("No candidates found for cap={} rho={} set={}".format(cap, rho, support_set))

    selected_ids = set(support_cell["canonical_feature_id"].astype(str))
    manifest = dedup_cell[dedup_cell["canonical_feature_id"].astype(str).isin(selected_ids)].copy()
    support_cols = [
        "panel", "canonical_feature_id", "n_contexts", "n_cohorts", "n_endpoints", "cohorts", "contexts",
        "support_set_S1", "support_set_S2", "exceptional_selected", "support_set_S2_plus_E",
        "global_representative_feature_uid", "global_representative_feature_source",
        "global_representative_feature_group", "global_representative_feature",
    ]
    manifest = manifest.merge(
        support_cell[support_cols],
        on=["panel", "canonical_feature_id"],
        how="left",
        suffixes=("", "_support"),
    )
    manifest["local_feature_uid"] = manifest["feature_uid"].astype(str)
    manifest["feature_uid"] = manifest["global_representative_feature_uid"].astype(str)
    manifest["candidate_score"] = pd.to_numeric(manifest.get("candidate_score"), errors="coerce")
    manifest["primary_oof_metric"] = pd.to_numeric(manifest.get("primary_oof_metric"), errors="coerce")
    manifest["primary_delta_metric"] = pd.to_numeric(manifest.get("primary_delta_metric"), errors="coerce")
    manifest["grid_candidate_cap"] = cap
    manifest["grid_semantic_rho"] = rho
    manifest["grid_support_set"] = support_set.upper()

    # One row per context/global feature UID. This is the form expected by Stage 2B.
    manifest = manifest.sort_values(
        ["context_id", "feature_uid", "candidate_score", "primary_oof_metric"],
        ascending=[True, True, False, False],
        na_position="last",
    ).drop_duplicates(["context_id", "feature_uid"], keep="first")

    export_root = Path(output_dir) if output_dir else (
        outroot / "exports" / grid_id(cap, rho) / support_set.upper()
    )
    ensure_dir(export_root)
    preferred = [
        *CONTEXT_COLS, "context_id", "context_strength", "feature_source", "feature_group", "feature",
        "feature_uid", "local_feature_uid", "selected_transform_mode", "candidate_score", "primary_oof_metric",
        "primary_delta_metric", "fold_sd", "direction_consistency", "nonmissing_fraction", "valid_folds",
        "canonical_feature_id", "n_contexts", "n_cohorts", "n_endpoints", "cohorts", "contexts",
        "support_set_S1", "support_set_S2", "exceptional_selected", "support_set_S2_plus_E",
        "grid_candidate_cap", "grid_semantic_rho", "grid_support_set", "source_file",
    ]
    manifest_out = manifest[[c for c in preferred if c in manifest.columns]].copy()
    manifest_out.to_csv(export_root / "global_module_candidate_manifest.csv", index=False)
    support_cell.to_csv(export_root / "canonical_candidate_support.csv", index=False)
    summary = (
        manifest_out.groupby("panel", dropna=False)
        .agg(
            n_manifest_rows=("feature_uid", "size"),
            n_global_feature_uids=("feature_uid", "nunique"),
            n_canonical_features=("canonical_feature_id", "nunique"),
            n_contexts=("context_id", "nunique"),
            n_cohorts=("cohort", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(export_root / "export_summary.csv", index=False)
    write_json({
        "candidate_cap": cap,
        "semantic_rho": rho,
        "support_set": support_set.upper(),
        "source_grid_root": str(outroot),
        "candidate_manifest": str(export_root / "global_module_candidate_manifest.csv"),
    }, export_root / "export_config.json")
    log("[SAVE] {}".format(export_root / "global_module_candidate_manifest.csv"))
    log("[INFO] rows={} canonical_features={}".format(len(manifest_out), manifest_out["canonical_feature_id"].nunique()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["validate", "inventory", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)

    worker = sub.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--array-id", type=int, default=None)
    worker.add_argument("--force", action="store_true")

    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--candidate-cap", type=int, required=True)
    export.add_argument("--semantic-rho", type=float, required=True)
    export.add_argument("--support-set", default="S2E", choices=["S1", "S2", "S2E"])
    export.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    ensure_dir(cfg["output_root"])
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "inventory":
        command_inventory(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id), force=bool(args.force))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    elif args.command == "export":
        command_export(cfg, args.candidate_cap, args.semantic_rho, args.support_set, args.output_dir)


if __name__ == "__main__":
    main()
