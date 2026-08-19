#!/usr/bin/env python3
"""
stage2a5_interpretable_microcompression_v1.py

Stage 2A-5: interpretable, correlation-gated microfamily compression.

The script consumes Stage 2A-4 patient matrices and candidate registries. It
applies sequential, fully audited simplification passes:
  1. exact patient-vector/provenance duplicates;
  2. checkpoint-state simplification (prefer the fundamental parent phenotype);
  3. distribution-summary simplification (prefer median, then mean, then quantiles);
  4. tissue-compartment simplification (prefer All, then Tumor/Epi, then Stroma);
  5. residual within-family redundancy compression.

A simpler feature replaces a more complex feature only when:
  * the relevant semantic key matches;
  * patient-level positive Spearman correlation exceeds the configured threshold;
  * enough pairwise-complete patients are available; and
  * the simpler feature's OOF metric is within the configured tolerance.

Unused rescue-only variables are removed after compression. Every final variable
must either be an original threshold-passing seed or represent at least one seed.

Commands
--------
inventory   Build a sequential Stage 2A-5 context index from Stage 2A-4 matrices.
worker      Compress one context (one CPU; supports SLURM_ARRAY_TASK_ID).
aggregate   Aggregate context manifests and audits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_COLS = ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"]


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


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        alternatives = [p.with_suffix(".csv.gz"), p.with_suffix(".csv")]
        for alt in alternatives:
            if alt.exists():
                p = alt
                break
    if not p.exists():
        raise FileNotFoundError(str(path))
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


def get_utils(cfg: Mapping):
    path = cfg.get("interpretability_utils_path")
    if path is None:
        path = Path(__file__).with_name("stage2a_interpretability_utils_v1.py")
    return import_module_from_path("stage2a_interpretability_utils_for_stage2a5", path)



def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def panel_value(cfg: Mapping, key: str, panel: str, default: float) -> float:
    value = cfg.get(key, default)
    if isinstance(value, Mapping):
        return float(value.get(panel, default))
    return float(value)


def stage2a4_root(cfg: Mapping) -> Path:
    return Path(cfg["stage2a4_output_root"])


def command_inventory(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    source_manifest = stage2a4_root(cfg) / "stage2a4_matrix_manifest.csv"
    if not source_manifest.exists():
        raise FileNotFoundError("Run Stage 2A-4 aggregate first: {}".format(source_manifest))
    source = pd.read_csv(source_manifest)
    if source.empty:
        raise RuntimeError("Stage 2A-4 matrix manifest is empty")
    source = source.sort_values([c for c in CONTEXT_COLS if c in source.columns]).reset_index(drop=True)
    source.insert(0, "stage2a5_array_id", np.arange(len(source), dtype=int))
    source.to_csv(output_root / "stage2a5_context_index.csv", index=False)
    write_json(cfg, output_root / "stage2a5_config.resolved.json")
    log("[SAVE] {} contexts={}".format(output_root / "stage2a5_context_index.csv", len(source)))


def resolve_array_id(value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or use a Slurm array")
    return int(env)


def context_output_dir(cfg: Mapping, row: pd.Series) -> Path:
    return Path(cfg["output_root"]) / "contexts" / str(row["context_slug"])


def get_context_row(cfg: Mapping, array_id: int) -> pd.Series:
    path = Path(cfg["output_root"]) / "stage2a5_context_index.csv"
    if not path.exists():
        raise FileNotFoundError("Run inventory first: {}".format(path))
    index = pd.read_csv(path)
    match = index[index["stage2a5_array_id"].astype(int) == int(array_id)]
    if match.empty:
        raise IndexError("stage2a5_array_id={} not found".format(array_id))
    return match.iloc[0]


def numeric_value(row: pd.Series, column: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(column, default))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def representative_sort(df: pd.DataFrame, priority_col: Optional[str] = None) -> pd.DataFrame:
    # Prevent ambiguity when feature_uid exists as both an index and a column.
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
    return out.sort_values(sort_cols, ascending=ascending, na_position="last")


def vector_hash(series: pd.Series, decimals: int = 12) -> str:
    x = pd.to_numeric(series, errors="coerce")
    tokens = ["NA" if pd.isna(v) else ("{:.{}f}".format(float(v), decimals)) for v in x]
    return hashlib.sha1("|".join(tokens).encode("utf-8")).hexdigest()


def resolve_rep(uid: str, replacement: Dict[str, str]) -> str:
    seen: Set[str] = set()
    current = uid
    while current in replacement and replacement[current] != current:
        if current in seen:
            raise RuntimeError("Replacement cycle involving {}".format(uid))
        seen.add(current)
        current = replacement[current]
    return current


def exact_duplicate_pass(
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    active: Set[str],
    replacement: Dict[str, str],
    iu,
    cfg: Mapping,
) -> Tuple[List[dict], List[dict]]:
    audit: List[dict] = []
    pair_rows: List[dict] = []
    groups: Dict[str, List[str]] = {}
    for uid in active:
        if uid not in matrix.columns:
            continue
        groups.setdefault(vector_hash(matrix[uid]), []).append(uid)

    meta = registry.set_index("feature_uid", drop=False)
    atol = float(cfg.get("exact_duplicate_atol", 1e-12))
    for hash_value, uids in groups.items():
        if len(uids) < 2:
            continue
        candidates = representative_sort(meta.loc[uids].copy())
        representative = str(candidates.iloc[0]["feature_uid"])
        for uid in uids:
            if uid == representative or uid not in active:
                continue
            exact, n = iu.is_exact_vector_duplicate(matrix[representative], matrix[uid], atol=atol)
            pair_rows.append({
                "compression_rule": "exact_duplicate",
                "feature_uid_a": representative,
                "feature_uid_b": uid,
                "spearman_rho": 1.0 if exact else np.nan,
                "pairwise_n": n,
                "passes_rule": exact,
            })
            if not exact:
                continue
            active.remove(uid)
            replacement[uid] = representative
            audit.append({
                "removed_feature_uid": uid,
                "representative_feature_uid": representative,
                "compression_rule": "exact_duplicate",
                "spearman_rho": 1.0,
                "pairwise_n": n,
                "removed_oof_metric": numeric_value(meta.loc[uid], "oof_metric"),
                "representative_oof_metric": numeric_value(meta.loc[representative], "oof_metric"),
                "oof_loss": numeric_value(meta.loc[uid], "oof_metric") - numeric_value(meta.loc[representative], "oof_metric"),
                "decision_reason": "identical transformed patient vector; simplest/highest-evidence representation retained",
            })
    return audit, pair_rows


def simplification_pass(
    *,
    rule_name: str,
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    active: Set[str],
    replacement: Dict[str, str],
    group_key: str,
    priority_col: str,
    corr_threshold: float,
    max_oof_loss: float,
    min_pairwise_n: int,
    iu,
    require_location_summary: bool = False,
) -> Tuple[List[dict], List[dict]]:
    audit: List[dict] = []
    pair_rows: List[dict] = []
    meta = registry.set_index("feature_uid", drop=False)
    active_df = registry[registry["feature_uid"].astype(str).isin(active)].copy()
    if require_location_summary:
        active_df = active_df[active_df["summary_class"].astype(str) == "location"]

    for _, group in active_df.groupby(group_key, dropna=False):
        if len(group) < 2:
            continue
        # Process more complex variables first. Candidate representatives must be
        # strictly simpler on the dimension controlled by this pass.
        complex_first = group.sort_values(
            [priority_col, "candidate_evidence_score", "oof_metric"],
            ascending=[False, False, False],
            na_position="last",
        )
        for _, removed_row in complex_first.iterrows():
            removed_uid = str(removed_row["feature_uid"])
            if removed_uid not in active or removed_uid not in matrix.columns:
                continue
            removed_priority = numeric_value(removed_row, priority_col, default=999)
            possible = group[
                (pd.to_numeric(group[priority_col], errors="coerce") < removed_priority)
                & (group["feature_uid"].astype(str).isin(active))
            ].copy()
            possible = possible[possible["feature_uid"].astype(str).isin(matrix.columns)]
            if possible.empty:
                continue
            possible = representative_sort(possible, priority_col=priority_col)

            accepted: List[dict] = []
            for _, rep_row in possible.iterrows():
                rep_uid = str(rep_row["feature_uid"])
                rho, n = iu.safe_spearman(matrix[removed_uid], matrix[rep_uid], min_n=min_pairwise_n)
                removed_oof = numeric_value(removed_row, "oof_metric")
                rep_oof = numeric_value(rep_row, "oof_metric")
                oof_loss = removed_oof - rep_oof
                passes = bool(np.isfinite(rho) and rho >= corr_threshold and np.isfinite(oof_loss) and oof_loss <= max_oof_loss)
                pair_rows.append({
                    "compression_rule": rule_name,
                    "feature_uid_a": removed_uid,
                    "feature_uid_b": rep_uid,
                    "spearman_rho": rho,
                    "pairwise_n": n,
                    "removed_priority": removed_priority,
                    "representative_priority": numeric_value(rep_row, priority_col),
                    "oof_loss": oof_loss,
                    "passes_rule": passes,
                })
                if passes:
                    accepted.append({
                        "rep_uid": rep_uid,
                        "rho": rho,
                        "n": n,
                        "oof_loss": oof_loss,
                        "rep_priority": numeric_value(rep_row, priority_col),
                        "rep_score": numeric_value(rep_row, "candidate_evidence_score", default=-np.inf),
                    })
            if not accepted:
                continue
            accepted = sorted(accepted, key=lambda z: (z["rep_priority"], -z["rep_score"], -z["rho"], z["rep_uid"]))
            chosen = accepted[0]
            rep_uid = chosen["rep_uid"]
            active.remove(removed_uid)
            replacement[removed_uid] = rep_uid
            audit.append({
                "removed_feature_uid": removed_uid,
                "representative_feature_uid": rep_uid,
                "compression_rule": rule_name,
                "spearman_rho": chosen["rho"],
                "pairwise_n": chosen["n"],
                "removed_priority": removed_priority,
                "representative_priority": chosen["rep_priority"],
                "removed_oof_metric": numeric_value(removed_row, "oof_metric"),
                "representative_oof_metric": numeric_value(meta.loc[rep_uid], "oof_metric"),
                "oof_loss": chosen["oof_loss"],
                "decision_reason": "simpler interpretation retained after correlation and OOF-tolerance checks",
            })
    return audit, pair_rows


def residual_greedy_pass(
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    active: Set[str],
    replacement: Dict[str, str],
    corr_threshold: float,
    max_oof_loss: float,
    min_pairwise_n: int,
    iu,
) -> Tuple[List[dict], List[dict]]:
    audit: List[dict] = []
    pair_rows: List[dict] = []
    active_df = registry[registry["feature_uid"].astype(str).isin(active)].copy()
    meta = registry.set_index("feature_uid", drop=False)

    for _, group in active_df.groupby("residual_microfamily_key", dropna=False):
        group = group[group["feature_uid"].astype(str).isin(active)]
        if len(group) < 2:
            continue
        ordered = representative_sort(group)
        representatives: List[str] = []
        for _, row in ordered.iterrows():
            uid = str(row["feature_uid"])
            if uid not in active or uid not in matrix.columns:
                continue
            assigned = False
            for rep_uid in representatives:
                if rep_uid not in active:
                    continue
                rho, n = iu.safe_spearman(matrix[uid], matrix[rep_uid], min_n=min_pairwise_n)
                uid_oof = numeric_value(row, "oof_metric")
                rep_oof = numeric_value(meta.loc[rep_uid], "oof_metric")
                oof_loss = uid_oof - rep_oof
                passes = bool(np.isfinite(rho) and rho >= corr_threshold and np.isfinite(oof_loss) and oof_loss <= max_oof_loss)
                pair_rows.append({
                    "compression_rule": "residual_redundancy",
                    "feature_uid_a": uid,
                    "feature_uid_b": rep_uid,
                    "spearman_rho": rho,
                    "pairwise_n": n,
                    "oof_loss": oof_loss,
                    "passes_rule": passes,
                })
                if passes:
                    active.remove(uid)
                    replacement[uid] = rep_uid
                    audit.append({
                        "removed_feature_uid": uid,
                        "representative_feature_uid": rep_uid,
                        "compression_rule": "residual_redundancy",
                        "spearman_rho": rho,
                        "pairwise_n": n,
                        "removed_oof_metric": uid_oof,
                        "representative_oof_metric": rep_oof,
                        "oof_loss": oof_loss,
                        "decision_reason": "residual semantically matched feature compressed to representative",
                    })
                    assigned = True
                    break
            if not assigned:
                representatives.append(uid)
    return audit, pair_rows


def make_stage2b_manifest(final: pd.DataFrame) -> pd.DataFrame:
    out = final.copy()
    out["candidate_score"] = out.get("candidate_evidence_score", np.nan)
    out["primary_oof_metric"] = out.get("oof_metric", np.nan)
    out["primary_delta_metric"] = out.get("delta_clinical", np.nan)
    if "selected_transform_mode" not in out.columns and "transform_mode" in out.columns:
        out["selected_transform_mode"] = out["transform_mode"]
    out = out.sort_values(
        ["candidate_score", "primary_oof_metric"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    out["context_candidate_rank"] = np.arange(1, len(out) + 1)
    preferred = [
        *CONTEXT_COLS,
        "context_id",
        "feature_source",
        "feature_group",
        "feature",
        "feature_uid",
        "selected_transform_mode",
        "candidate_score",
        "primary_oof_metric",
        "primary_delta_metric",
        "fold_sd",
        "direction_consistency",
        "nonmissing_fraction",
        "p_value",
        "context_q_value",
        "n",
        "n_events",
        "n_positive",
        "n_negative",
        "valid_folds",
        "context_candidate_rank",
        "seed_pass_thresholds",
        "candidate_role",
        "represented_seed_feature_uids",
        "represented_feature_uids",
        "n_represented_seeds",
        "n_represented_features",
        "feature_kind",
        "metric_kind",
        "state_complexity",
        "summary_stat",
        "compartment",
        "source_file",
    ]
    return out[[c for c in preferred if c in out.columns]].copy()


def save_compression_plot(summary: pd.DataFrame, path: Path, title: str) -> None:
    labels = [
        "Registry",
        "After exact",
        "After state",
        "After metric",
        "After compartment",
        "After residual",
        "Final seed representatives",
    ]
    cols = [
        "n_registry_features",
        "n_after_exact",
        "n_after_state",
        "n_after_metric",
        "n_after_compartment",
        "n_after_residual",
        "n_final_representatives",
    ]
    values = [float(summary.iloc[0].get(c, np.nan)) for c in cols]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def command_worker(cfg: Mapping, array_id: int) -> None:
    row = get_context_row(cfg, array_id)
    context_dir = ensure_dir(context_output_dir(cfg, row))
    log("=" * 80)
    log("[STAGE2A-5 context {}] {}".format(array_id, row["context_id"]))

    registry = pd.read_csv(row["candidate_registry_path"])
    matrix = read_table(row["matrix_path"])
    if registry.empty or matrix.empty or matrix.shape[1] <= 1:
        raise RuntimeError("Empty registry or patient matrix for {}".format(row["context_id"]))

    iu = get_utils(cfg)
    registry = iu.add_interpretability_columns(registry)
    built_uids = [uid for uid in registry["feature_uid"].astype(str) if uid in matrix.columns]
    registry = registry[registry["feature_uid"].astype(str).isin(built_uids)].drop_duplicates("feature_uid").copy()
    if registry.empty:
        raise RuntimeError("No registry features were found as matrix columns")

    panel = str(row["panel"])
    min_pairwise_n = int(cfg.get("min_pairwise_n", 20))
    active: Set[str] = set(registry["feature_uid"].astype(str))
    replacement: Dict[str, str] = {}
    all_audit: List[dict] = []
    all_pairs: List[dict] = []

    n_registry = len(active)

    audit, pairs = exact_duplicate_pass(registry, matrix, active, replacement, iu, cfg)
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_exact = len(active)

    audit, pairs = simplification_pass(
        rule_name="state_simplification",
        registry=registry, matrix=matrix, active=active, replacement=replacement,
        group_key="state_simplification_key", priority_col="state_complexity",
        corr_threshold=panel_value(cfg, "state_corr_threshold_by_panel", panel, 0.90),
        max_oof_loss=panel_value(cfg, "state_max_oof_loss_by_panel", panel, 0.01),
        min_pairwise_n=min_pairwise_n, iu=iu,
    )
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_state = len(active)

    audit, pairs = simplification_pass(
        rule_name="metric_summary_simplification",
        registry=registry, matrix=matrix, active=active, replacement=replacement,
        group_key="metric_simplification_key", priority_col="summary_priority",
        corr_threshold=panel_value(cfg, "metric_corr_threshold_by_panel", panel, 0.93),
        max_oof_loss=panel_value(cfg, "metric_max_oof_loss_by_panel", panel, 0.01),
        min_pairwise_n=min_pairwise_n, iu=iu, require_location_summary=True,
    )
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_metric = len(active)

    audit, pairs = simplification_pass(
        rule_name="compartment_simplification",
        registry=registry, matrix=matrix, active=active, replacement=replacement,
        group_key="compartment_simplification_key", priority_col="compartment_priority",
        corr_threshold=panel_value(cfg, "compartment_corr_threshold_by_panel", panel, 0.90),
        max_oof_loss=panel_value(cfg, "compartment_max_oof_loss_by_panel", panel, 0.01),
        min_pairwise_n=min_pairwise_n, iu=iu,
    )
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_compartment = len(active)

    audit, pairs = residual_greedy_pass(
        registry=registry, matrix=matrix, active=active, replacement=replacement,
        corr_threshold=panel_value(cfg, "residual_corr_threshold_by_panel", panel, 0.95),
        max_oof_loss=panel_value(cfg, "residual_max_oof_loss_by_panel", panel, 0.01),
        min_pairwise_n=min_pairwise_n, iu=iu,
    )
    all_audit.extend(audit); all_pairs.extend(pairs)
    n_after_residual = len(active)

    seed_uids = set(registry.loc[registry["seed_pass_thresholds"].map(parse_bool), "feature_uid"].astype(str))
    seed_to_final_rows = []
    represented_by_final: Dict[str, List[str]] = {}
    represented_seeds_by_final: Dict[str, List[str]] = {}
    for uid in registry["feature_uid"].astype(str):
        final_uid = resolve_rep(uid, replacement)
        represented_by_final.setdefault(final_uid, []).append(uid)
    for seed_uid in sorted(seed_uids):
        final_uid = resolve_rep(seed_uid, replacement)
        represented_seeds_by_final.setdefault(final_uid, []).append(seed_uid)
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
            "spearman_rho": np.nan,
            "pairwise_n": np.nan,
            "oof_loss": np.nan,
            "decision_reason": "rescue-only feature did not become the representative of any threshold-passing seed",
        })
    active = set(final_uids)

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
    final = representative_sort(final).reset_index(drop=True)
    final["compressed_context_rank"] = np.arange(1, len(final) + 1)

    audit_df = pd.DataFrame(all_audit)
    if not audit_df.empty:
        meta_cols = ["feature_uid", "feature", "feature_source", "feature_group"]
        removed_meta = registry[meta_cols].rename(columns={c: "removed_" + c for c in meta_cols})
        rep_meta = registry[meta_cols].rename(columns={c: "representative_" + c for c in meta_cols})
        audit_df = audit_df.merge(removed_meta, left_on="removed_feature_uid", right_on="removed_feature_uid", how="left")
        audit_df = audit_df.merge(rep_meta, left_on="representative_feature_uid", right_on="representative_feature_uid", how="left")
    pair_df = pd.DataFrame(all_pairs)
    seed_to_final = pd.DataFrame(seed_to_final_rows)

    manifest = make_stage2b_manifest(final)
    final_matrix = matrix[["patient_id"] + [uid for uid in final["feature_uid"].astype(str) if uid in matrix.columns]].copy()

    final.to_csv(context_dir / "compressed_context_candidates.csv", index=False)
    manifest.to_csv(context_dir / "stage2b_candidate_manifest.csv", index=False)
    audit_df.to_csv(context_dir / "compression_decision_audit.csv", index=False)
    pair_df.to_csv(context_dir / "evaluated_pair_correlations.csv", index=False)
    seed_to_final.to_csv(context_dir / "seed_to_final_representative.csv", index=False)
    save_table(final_matrix, context_dir / "compressed_patient_feature_matrix.parquet")

    summary = pd.DataFrame([{
        "stage2a5_array_id": array_id,
        "context_id": row["context_id"],
        **{c: row[c] for c in CONTEXT_COLS},
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
    }])
    summary.to_csv(context_dir / "context_compression_summary.csv", index=False)
    save_compression_plot(summary, context_dir / "compression_flow.png", "{} | {} | {}".format(row["cohort"], row["panel"], row["endpoint"]))
    (context_dir / ".done").write_text("complete\n")
    log("[DONE] seeds={} final_representatives={}".format(len(seed_uids), len(final)))


def command_aggregate(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    index_path = output_root / "stage2a5_context_index.csv"
    if not index_path.exists():
        raise FileNotFoundError("Run inventory first")
    index = pd.read_csv(index_path)
    summaries: List[pd.DataFrame] = []
    manifests: List[pd.DataFrame] = []
    audits: List[pd.DataFrame] = []
    seeds: List[pd.DataFrame] = []
    matrix_rows: List[dict] = []
    missing: List[dict] = []

    for _, row in index.iterrows():
        cdir = context_output_dir(cfg, row)
        required = {
            "summary": cdir / "context_compression_summary.csv",
            "manifest": cdir / "stage2b_candidate_manifest.csv",
            "audit": cdir / "compression_decision_audit.csv",
            "seedmap": cdir / "seed_to_final_representative.csv",
        }
        if not required["summary"].exists():
            missing.append({"context_id": row["context_id"], "reason": "missing_context_output"})
            continue
        summaries.append(pd.read_csv(required["summary"]))
        manifest = pd.read_csv(required["manifest"])
        manifest["stage2a5_array_id"] = row["stage2a5_array_id"]
        manifests.append(manifest)
        if required["audit"].exists():
            audit = pd.read_csv(required["audit"])
            audit["context_id"] = row["context_id"]
            audits.append(audit)
        if required["seedmap"].exists():
            seedmap = pd.read_csv(required["seedmap"])
            seedmap["context_id"] = row["context_id"]
            seeds.append(seedmap)
        matrix_path = cdir / "compressed_patient_feature_matrix.parquet"
        if not matrix_path.exists() and matrix_path.with_suffix(".csv.gz").exists():
            matrix_path = matrix_path.with_suffix(".csv.gz")
        matrix_rows.append({
            "stage2a5_array_id": row["stage2a5_array_id"],
            "context_id": row["context_id"],
            "context_slug": row["context_slug"],
            **{c: row[c] for c in CONTEXT_COLS},
            "compressed_matrix_path": str(matrix_path),
            "context_manifest_path": str(required["manifest"]),
        })

    summary_df = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    manifest_df = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()
    audit_df = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    seed_df = pd.concat(seeds, ignore_index=True, sort=False) if seeds else pd.DataFrame()

    summary_df.to_csv(output_root / "all_context_compression_summary.csv", index=False)
    manifest_df.to_csv(output_root / "global_module_candidate_manifest_after_microcompression.csv", index=False)
    save_table(audit_df, output_root / "all_context_compression_decision_audit.parquet")
    seed_df.to_csv(output_root / "all_context_seed_to_final_representative.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(output_root / "stage2a5_compressed_matrix_manifest.csv", index=False)
    pd.DataFrame(missing).to_csv(output_root / "stage2a5_missing_context_outputs.csv", index=False)

    if not manifest_df.empty:
        recurrence = (
            manifest_df.groupby(["panel", "feature_uid", "feature_source", "feature_group", "feature"], dropna=False)
            .agg(
                n_contexts=("context_id", "nunique"),
                n_cohorts=("cohort", "nunique"),
                n_endpoints=("endpoint", "nunique"),
                contexts=("context_id", lambda x: ";".join(sorted(set(map(str, x))))),
                max_candidate_score=("candidate_score", "max"),
                median_candidate_score=("candidate_score", "median"),
                max_oof_metric=("primary_oof_metric", "max"),
                median_oof_metric=("primary_oof_metric", "median"),
            )
            .reset_index()
            .sort_values(["panel", "n_cohorts", "n_contexts", "median_candidate_score"], ascending=[True, False, False, False])
        )
        recurrence.to_csv(output_root / "candidate_context_recurrence_after_microcompression.csv", index=False)

    log("[DONE] context_manifests={} total_manifest_rows={}".format(len(manifests), len(manifest_df)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["inventory", "worker", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "worker":
            p.add_argument("--array-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    ensure_dir(cfg["output_root"])
    if args.command == "inventory":
        command_inventory(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)


if __name__ == "__main__":
    main()
