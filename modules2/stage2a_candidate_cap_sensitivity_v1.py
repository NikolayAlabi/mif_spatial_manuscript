#!/usr/bin/env python3
"""
stage2a_candidate_cap_sensitivity_v1.py

Root-aware candidate-cap sensitivity analysis for Stage 2A.

Design
------
1. setup
   * Reads the large Stage 2A best-transform parquet ONCE.
   * Applies the manually frozen context-specific hard quality/stability rules.
   * Ranks passing candidates within context x prep-root.
   * Writes compact per-context shards (top max_depth only).
   * Writes shared cache manifests for each cohort/panel/sample_type/subset/agg,
     taking the union of candidates required across endpoints.

2. cache-worker
   * One CPU per shared matrix context (normally cohort x panel).
   * Reconstructs raw patient-level values once for the union of required features
     using the tested Stage-1 loading/aggregation functions.
   * Raw values are sufficient because Spearman correlation is invariant to the
     monotonic z-score and log1p-z-score transforms used in Stage 1.

3. worker
   * One CPU per endpoint context/panel.
   * Reads only its compact candidate shard plus its shared patient matrix.
   * Evaluates candidate depths 1..max_depth for quality, CV stability and
     redundancy at configurable |rho| thresholds.
   * Produces root-specific plots, pairwise-correlation audits and transparent
     mathematical cap recommendations.

4. aggregate
   * Combines all context/root results.
   * Produces panel x prep-root consensus recommendations, aggregate sensitivity
     plots, a manual review template and a text summary.

The cap recommendation is advisory. All candidates first pass the user-defined
context-specific quality/stability thresholds. Candidate depth is then chosen
primarily from redundancy/diminishing-return behavior rather than by arbitrary
quota.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_COLS = [
    "cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"
]
MATRIX_COLS = [
    "cohort", "panel", "sample_type", "patient_subset", "agg"
]
ROOT_COL = "feature_source"
FEATURE_ID_COLS = ["feature_source", "feature_group", "feature"]


# =============================================================================
# Generic helpers
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path | str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(obj: Mapping, path: Path | str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def read_table(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        for alt in [p.with_suffix(".csv.gz"), p.with_suffix(".csv")]:
            if alt.exists():
                p = alt
                break
    if not p.exists():
        raise FileNotFoundError(p)
    low = p.name.lower()
    if low.endswith(".parquet"):
        return pd.read_parquet(p)
    if low.endswith(".tsv") or low.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def save_table(df: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return p
        except (ImportError, ModuleNotFoundError, ValueError):
            p = p.with_suffix(".csv.gz")
            df.to_csv(p, index=False, compression="gzip")
            return p
    if p.name.lower().endswith(".csv.gz"):
        df.to_csv(p, index=False, compression="gzip")
    else:
        df.to_csv(p, index=False)
    return p


def import_module_from_path(module_name: str, path: Path | str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {
        "1", "true", "t", "yes", "y", "include", "included"
    }


def optional_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def make_feature_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "|" + df["feature_group"].astype(str)
        + "|" + df["feature"].astype(str)
    )


def slugify(values: Sequence[object]) -> str:
    parts = []
    for value in values:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
        parts.append(s or "NA")
    return "__".join(parts)


def quantile_or_nan(s: pd.Series, q: float) -> float:
    x = safe_numeric(s).dropna()
    return float(x.quantile(q)) if not x.empty else np.nan


def median_or_nan(s: pd.Series) -> float:
    x = safe_numeric(s).dropna()
    return float(x.median()) if not x.empty else np.nan


def min_or_nan(s: pd.Series) -> float:
    x = safe_numeric(s).dropna()
    return float(x.min()) if not x.empty else np.nan


def max_or_nan(s: pd.Series) -> float:
    x = safe_numeric(s).dropna()
    return float(x.max()) if not x.empty else np.nan


# =============================================================================
# Configuration
# =============================================================================


def load_config(path: str) -> dict:
    cfg = read_json(path)
    required = [
        "candidate_parquet", "context_rules_csv", "output_root",
        "stage1_script_path", "harmonized_path"
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")
    cfg.setdefault("max_depth", 50)
    cfg.setdefault("plot_depths", [1,2,3,4,5,7,10,12,15,20,25,30,40,50])
    cfg.setdefault("redundancy_rhos", [0.85, 0.90, 0.95])
    cfg.setdefault("primary_redundancy_rho", 0.90)
    cfg.setdefault("min_pairwise_n", 20)
    cfg.setdefault("novelty_yield_profiles", {
        "strict": 0.80, "balanced": 0.70, "permissive": 0.60
    })
    cfg.setdefault("diminishing_window", 5)
    cfg.setdefault("max_new_novel_in_window", 2)
    cfg.setdefault("qc_acceptability", "acceptable_or_borderline")
    cfg.setdefault("min_epi_fraction", 0.05)
    cfg.setdefault("spatial_root", None)
    cfg.setdefault("cell_features_path", None)
    cfg.setdefault("triads_path", None)
    cfg.setdefault("koll_metadata_csv", None)
    return cfg


# =============================================================================
# Hard context rules and root-wise ranking
# =============================================================================


def merge_rules_and_apply_thresholds(
    features: pd.DataFrame,
    rules: pd.DataFrame,
) -> pd.DataFrame:
    df = features.copy()
    if "feature_uid" not in df.columns:
        df["feature_uid"] = make_feature_uid(df)

    rule_keep = CONTEXT_COLS + [
        "include_context", "context_strength",
        "min_oof_metric", "min_delta_clinical", "max_fold_sd",
        "min_direction_consistency", "min_nonmissing_fraction",
        "min_candidate_evidence_score", "min_valid_folds",
        "max_nominal_p", "max_context_q", "manual_notes"
    ]
    rule_keep = [c for c in rule_keep if c in rules.columns]
    rules2 = rules[rule_keep].drop_duplicates(CONTEXT_COLS)
    if rules2.duplicated(CONTEXT_COLS).any():
        raise ValueError("Context rules are not unique by context.")

    df = df.merge(
        rules2,
        on=CONTEXT_COLS,
        how="left",
        validate="many_to_one",
        suffixes=("", "_rule"),
    )

    df["include_context_flag"] = df.get("include_context", 0).map(parse_bool)
    passed = df["include_context_flag"].copy()

    specs = [
        ("oof_metric", "min_oof_metric", "min"),
        ("delta_clinical", "min_delta_clinical", "min"),
        ("fold_sd", "max_fold_sd", "max"),
        ("direction_consistency", "min_direction_consistency", "min"),
        ("nonmissing_fraction", "min_nonmissing_fraction", "min"),
        ("candidate_evidence_score", "min_candidate_evidence_score", "min"),
        ("valid_folds", "min_valid_folds", "min"),
        ("p_value", "max_nominal_p", "max"),
        ("context_q_value", "max_context_q", "max"),
    ]

    failure_cols: List[str] = []
    for value_col, threshold_col, direction in specs:
        flag_col = f"pass_{value_col}"
        failure_cols.append(flag_col)
        if threshold_col not in df.columns:
            df[flag_col] = True
            continue
        threshold = safe_numeric(df[threshold_col])
        active = threshold.notna()
        if value_col not in df.columns:
            df[flag_col] = ~active
            passed &= df[flag_col]
            continue
        value = safe_numeric(df[value_col])
        if direction == "min":
            flag = (~active) | (value.notna() & (value >= threshold))
        else:
            flag = (~active) | (value.notna() & (value <= threshold))
        df[flag_col] = flag
        passed &= flag

    df["passes_context_rules"] = passed

    def failure_reason(row: pd.Series) -> str:
        reasons = []
        if not bool(row.get("include_context_flag", False)):
            reasons.append("context_excluded")
        for c in failure_cols:
            if c in row.index and not bool(row[c]):
                reasons.append(c.replace("pass_", ""))
        return ";".join(reasons)

    df["threshold_failure_reasons"] = df.apply(failure_reason, axis=1)

    # Threshold-relative margins. Positive values mean comfortably inside the
    # prespecified acceptable region; zero is exactly on the hard threshold.
    if "min_oof_metric" in df.columns:
        df["oof_margin"] = safe_numeric(df["oof_metric"]) - safe_numeric(df["min_oof_metric"])
    else:
        df["oof_margin"] = np.nan

    if "max_fold_sd" in df.columns:
        df["fold_sd_headroom"] = safe_numeric(df["max_fold_sd"]) - safe_numeric(df["fold_sd"])
    else:
        df["fold_sd_headroom"] = np.nan

    if "min_direction_consistency" in df.columns:
        df["direction_margin"] = safe_numeric(df["direction_consistency"]) - safe_numeric(df["min_direction_consistency"])
    else:
        df["direction_margin"] = np.nan

    if "min_delta_clinical" in df.columns:
        df["delta_margin"] = safe_numeric(df["delta_clinical"]) - safe_numeric(df["min_delta_clinical"])
    else:
        df["delta_margin"] = np.nan

    if "min_nonmissing_fraction" in df.columns:
        df["nonmissing_margin"] = safe_numeric(df["nonmissing_fraction"]) - safe_numeric(df["min_nonmissing_fraction"])
    else:
        df["nonmissing_margin"] = np.nan

    return df


def rank_passing_within_roots(df: pd.DataFrame) -> pd.DataFrame:
    passing = df[df["passes_context_rules"]].copy()
    if passing.empty:
        passing["eligible_root_rank"] = pd.Series(dtype=int)
        return passing

    if "root_candidate_evidence_score" not in passing.columns:
        passing["root_candidate_evidence_score"] = passing.get(
            "candidate_evidence_score", np.nan
        )

    sort_cols = CONTEXT_COLS + [ROOT_COL,
        "root_candidate_evidence_score", "oof_metric", "fold_sd", "nonmissing_fraction"
    ]
    ascending = [True] * (len(CONTEXT_COLS) + 1) + [False, False, True, False]
    passing = passing.sort_values(sort_cols, ascending=ascending, na_position="last")
    passing["eligible_root_rank"] = (
        passing.groupby(CONTEXT_COLS + [ROOT_COL], dropna=False).cumcount() + 1
    )
    return passing


# =============================================================================
# SETUP: one read of the large candidate parquet
# =============================================================================


def command_setup(cfg: Mapping) -> None:
    outroot = ensure_dir(cfg["output_root"])
    shard_dir = ensure_dir(outroot / "candidate_shards")
    cache_manifest_dir = ensure_dir(outroot / "cache_manifests")

    log("[SETUP] Reading large best-transform parquet ONCE")
    features = read_table(cfg["candidate_parquet"])
    rules = pd.read_csv(cfg["context_rules_csv"])
    log(f"[SETUP] feature rows={len(features):,} rules={len(rules):,}")

    merged = merge_rules_and_apply_thresholds(features, rules)
    passing = rank_passing_within_roots(merged)

    max_depth = int(cfg["max_depth"])
    passing_top = passing[passing["eligible_root_rank"] <= max_depth].copy()

    # Compact global audits, not the huge all-feature table.
    save_table(passing_top, outroot / "all_passing_top_depth_candidates.parquet")

    full_counts = (
        passing.groupby(CONTEXT_COLS + [ROOT_COL], dropna=False)
        .agg(
            n_passing_context_rules=("feature_uid", "size"),
            best_oof=("oof_metric", "max"),
            worst_oof=("oof_metric", "min"),
            median_fold_sd=("fold_sd", "median"),
        )
        .reset_index()
    )
    full_counts.to_csv(outroot / "passing_candidate_counts_by_context_root.csv", index=False)

    # Context shards. Derive contexts from included rules so contexts with zero
    # passing candidates remain explicit in the review index.
    included_rules = rules[rules["include_context"].map(parse_bool)].copy()
    context_rows: List[dict] = []
    for array_id, (_, rr) in enumerate(included_rules.sort_values(CONTEXT_COLS).iterrows()):
        mask = pd.Series(True, index=passing_top.index)
        for c in CONTEXT_COLS:
            mask &= passing_top[c].astype(str).eq(str(rr[c]))
        sub = passing_top.loc[mask].copy()
        slug = slugify([rr[c] for c in CONTEXT_COLS])
        shard_path = save_table(sub, shard_dir / f"context_{array_id:03d}__{slug}.parquet")
        context_rows.append({
            "array_id": int(array_id),
            **{c: rr[c] for c in CONTEXT_COLS},
            "context_slug": slug,
            "candidate_shard": str(shard_path),
            "n_passing_top_depth_rows": int(len(sub)),
            "n_roots_with_signal": int(sub[ROOT_COL].nunique()) if not sub.empty else 0,
        })

    context_index = pd.DataFrame(context_rows)

    # Shared raw patient matrix manifests. One cache normally serves all endpoint
    # contexts for the same cohort x panel. Thus feature source files are loaded
    # only once per cohort/panel instead of once per endpoint worker.
    cache_rows: List[dict] = []
    for cache_id, (keys, gctx) in enumerate(context_index.groupby(MATRIX_COLS, dropna=False, sort=True)):
        key_dict = dict(zip(MATRIX_COLS, keys if isinstance(keys, tuple) else (keys,)))
        mask = pd.Series(True, index=passing_top.index)
        for c in MATRIX_COLS:
            mask &= passing_top[c].astype(str).eq(str(key_dict[c]))
        union = passing_top.loc[mask].copy()
        if not union.empty:
            # Same feature_uid can arise in several endpoints. Raw patient values
            # are endpoint-invariant, so keep one provenance row.
            union = union.sort_values(
                ["eligible_root_rank", "root_candidate_evidence_score", "oof_metric"],
                ascending=[True, False, False],
                na_position="last",
            ).drop_duplicates("feature_uid", keep="first")
        cache_slug = slugify([key_dict[c] for c in MATRIX_COLS])
        manifest_path = cache_manifest_dir / f"cache_{cache_id:03d}__{cache_slug}.csv"
        union.to_csv(manifest_path, index=False)
        cache_rows.append({
            "cache_id": int(cache_id),
            **key_dict,
            "cache_slug": cache_slug,
            "cache_manifest": str(manifest_path),
            "n_union_features": int(union["feature_uid"].nunique()) if not union.empty else 0,
        })

    cache_index = pd.DataFrame(cache_rows)

    # Link each endpoint context to its cache.
    context_index = context_index.merge(
        cache_index[["cache_id", "cache_slug"] + MATRIX_COLS],
        on=MATRIX_COLS,
        how="left",
        validate="many_to_one",
    )

    context_index.to_csv(outroot / "context_index.csv", index=False)
    cache_index.to_csv(outroot / "cache_index.csv", index=False)
    rules.to_csv(outroot / "context_rules_snapshot.csv", index=False)
    write_json(dict(cfg), outroot / "config.resolved.json")

    log(f"[SAVE] contexts={len(context_index)} -> {outroot / 'context_index.csv'}")
    log(f"[SAVE] shared caches={len(cache_index)} -> {outroot / 'cache_index.csv'}")
    log(f"[INFO] large parquet will NOT be read by cache/context workers")


# =============================================================================
# Shared raw patient cache construction
# =============================================================================


def build_patient_raw_matrix_for_source_group(
    *,
    stage1_mod,
    harm_df: pd.DataFrame,
    cohort: str,
    panel: str,
    feature_source: str,
    feature_group: str,
    features: Sequence[str],
    sample_type: str,
    patient_subset: str,
    agg: str,
    cfg: Mapping,
) -> pd.DataFrame:
    features = list(dict.fromkeys(str(f) for f in features))
    data_dict = stage1_mod.load_data_dict(
        feature_group=feature_group,
        feature_source=feature_source,
        panels=[panel],
        cohorts=[cohort],
        spatial_root=cfg.get("spatial_root"),
        cell_features_path=cfg.get("cell_features_path"),
        triads_path=cfg.get("triads_path"),
    )
    kwargs = dict(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=str(cfg.get("qc_acceptability", "acceptable_or_borderline")),
        min_epi_fraction=cfg.get("min_epi_fraction", 0.05),
        sample_type=sample_type,
    )
    if cfg.get("koll_metadata_csv") is not None:
        kwargs["koll_metadata_csv"] = cfg.get("koll_metadata_csv")

    core_df = stage1_mod.prepare_core_level_feature_table(**kwargs)
    if core_df.empty:
        raise ValueError("No cores remain after requested filters")
    core_df = stage1_mod.merge_harmonized_to_core_df(core_df, harm_df)
    core_df = stage1_mod.replace_with_harmonized_columns(core_df)
    core_df = stage1_mod.simplify_clinical_vars(core_df)
    core_df = stage1_mod.ensure_patient_id_column(core_df)

    present = [f for f in features if f in core_df.columns]
    if not present:
        raise ValueError("None of the requested features were found in core_df")

    patient_df = stage1_mod.aggregate_core_to_patient(
        core_df, feature_cols=present, agg=agg
    )
    if "cohort" in patient_df.columns:
        patient_df = patient_df[patient_df["cohort"].astype(str) == str(cohort)].copy()
    if cohort in {"No-NAC", "KOLL"} and patient_subset in {"no_adj_chemo", "adj_chemo"}:
        patient_df = stage1_mod.apply_patient_subset(patient_df, patient_subset=patient_subset)
    if patient_df.empty:
        raise ValueError("No patients remain after aggregation/subsetting")
    return patient_df[["patient_id"] + present].copy()


def command_cache_worker(cfg: Mapping, cache_id: int) -> None:
    outroot = Path(cfg["output_root"])
    cache_index = pd.read_csv(outroot / "cache_index.csv")
    match = cache_index[cache_index["cache_id"].astype(int) == int(cache_id)]
    if match.empty:
        raise IndexError(f"cache_id={cache_id} not found")
    row = match.iloc[0]
    cdir = ensure_dir(outroot / "shared_cache" / str(row["cache_slug"]))

    manifest = pd.read_csv(row["cache_manifest"])
    if manifest.empty:
        pd.DataFrame(columns=["patient_id"]).to_csv(cdir / "patient_feature_matrix.csv", index=False)
        pd.DataFrame([{"cache_id": cache_id, "status": "zero_features"}]).to_csv(cdir / "cache_summary.csv", index=False)
        return

    stage1_mod = import_module_from_path(
        f"stage1_for_cap_cache_{cache_id}", cfg["stage1_script_path"]
    )
    # Harmonized table is read once per cohort/panel cache worker, not once per
    # feature source/group.
    harm_df = stage1_mod.load_harmonized_df(cfg["harmonized_path"])

    cohort = str(row["cohort"])
    panel = str(row["panel"])
    sample_type = str(row["sample_type"])
    patient_subset = str(row["patient_subset"])
    agg = str(row["agg"])

    merged: Optional[pd.DataFrame] = None
    meta_rows: List[dict] = []
    failures: List[dict] = []

    for (feature_source, feature_group), g in manifest.groupby(
        ["feature_source", "feature_group"], dropna=False, sort=False
    ):
        features = g["feature"].dropna().astype(str).unique().tolist()
        log(
            f"[CACHE {cache_id}] {cohort}/{panel} source={feature_source} "
            f"group={feature_group} requested={len(features)}"
        )
        try:
            pdf = build_patient_raw_matrix_for_source_group(
                stage1_mod=stage1_mod,
                harm_df=harm_df,
                cohort=cohort,
                panel=panel,
                feature_source=str(feature_source),
                feature_group=str(feature_group),
                features=features,
                sample_type=sample_type,
                patient_subset=patient_subset,
                agg=agg,
                cfg=cfg,
            )
        except Exception as exc:
            failures.append({
                "feature_source": feature_source,
                "feature_group": feature_group,
                "reason": f"{type(exc).__name__}: {exc}",
                "n_requested_features": len(features),
            })
            continue

        tmp = pdf[["patient_id"]].copy()
        for _, fr in g.iterrows():
            feat = str(fr["feature"])
            uid = str(fr["feature_uid"])
            if feat not in pdf.columns:
                failures.append({
                    "feature_source": feature_source,
                    "feature_group": feature_group,
                    "feature": feat,
                    "feature_uid": uid,
                    "reason": "feature_missing_from_patient_matrix",
                })
                continue
            tmp[uid] = safe_numeric(pdf[feat])
            meta_rows.append({
                "feature_uid": uid,
                "feature": feat,
                "feature_source": feature_source,
                "feature_group": feature_group,
                "n_patients": int(len(pdf)),
                "nonmissing_fraction_raw": float(safe_numeric(pdf[feat]).notna().mean()),
                "n_unique_raw": int(safe_numeric(pdf[feat]).dropna().nunique()),
            })

        merged = tmp if merged is None else merged.merge(tmp, on="patient_id", how="outer")

    matrix = merged if merged is not None else pd.DataFrame(columns=["patient_id"])
    matrix_path = save_table(matrix, cdir / "patient_feature_matrix.parquet")
    pd.DataFrame(meta_rows).drop_duplicates("feature_uid").to_csv(cdir / "matrix_feature_meta.csv", index=False)
    pd.DataFrame(failures).to_csv(cdir / "matrix_build_failures.csv", index=False)

    summary = pd.DataFrame([{
        "cache_id": int(cache_id),
        **{c: row[c] for c in MATRIX_COLS},
        "status": "complete" if matrix.shape[1] > 1 else "zero_matrix_features",
        "n_manifest_features": int(manifest["feature_uid"].nunique()),
        "n_matrix_patients": int(matrix.shape[0]),
        "n_matrix_features": int(max(matrix.shape[1] - 1, 0)),
        "matrix_path": str(matrix_path),
        "n_failures": int(len(failures)),
    }])
    summary.to_csv(cdir / "cache_summary.csv", index=False)
    (cdir / ".done").write_text("complete\n")
    log(f"[CACHE {cache_id} DONE] patients={matrix.shape[0]} features={max(matrix.shape[1]-1,0)}")


# =============================================================================
# Redundancy calculations
# =============================================================================


def pairwise_n_matrix(df: pd.DataFrame) -> pd.DataFrame:
    obs = df.notna().astype(np.int32)
    n = obs.T.dot(obs)
    return n.astype(int)


def upper_triangle_values(matrix: pd.DataFrame) -> pd.Series:
    if matrix.shape[0] < 2:
        return pd.Series(dtype=float)
    arr = matrix.to_numpy(dtype=float)
    tri = np.triu_indices(arr.shape[0], 1)
    return pd.Series(arr[tri])


def graph_component_count(nodes: Sequence[str], corr_abs: pd.DataFrame, rho: float) -> int:
    nodes = list(nodes)
    if not nodes:
        return 0
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            val = corr_abs.loc[a, b] if a in corr_abs.index and b in corr_abs.columns else np.nan
            if pd.notna(val) and float(val) >= float(rho):
                union(a, b)
    return len({find(n) for n in nodes})


def greedy_novelty_series(
    ordered_uids: Sequence[str],
    corr_abs: pd.DataFrame,
    rho: float,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Process candidates in evidence-rank order. A candidate is novel if it is not
    near-redundant (|rho| >= threshold) with any previously accepted novel
    representative. Missing correlations are treated as unknown/not-proven-
    redundant and therefore retained; valid-correlation coverage is reported
    separately in diagnostics.
    """
    representatives: List[str] = []
    cumulative: List[int] = []
    is_novel: List[int] = []

    for uid in ordered_uids:
        redundant = False
        for rep in representatives:
            if uid not in corr_abs.index or rep not in corr_abs.columns:
                continue
            v = corr_abs.loc[uid, rep]
            if pd.notna(v) and float(v) >= float(rho):
                redundant = True
                break
        if redundant:
            is_novel.append(0)
        else:
            representatives.append(uid)
            is_novel.append(1)
        cumulative.append(len(representatives))
    return cumulative, is_novel, representatives


def compute_root_depth_diagnostics(
    candidates: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: Mapping,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    candidates = candidates.sort_values("eligible_root_rank").copy()
    ordered_uids = [u for u in candidates["feature_uid"].astype(str) if u in matrix.columns]
    candidates = candidates[candidates["feature_uid"].astype(str).isin(ordered_uids)].copy()
    candidates = candidates.drop_duplicates("feature_uid", keep="first").sort_values("eligible_root_rank")
    ordered_uids = candidates["feature_uid"].astype(str).tolist()
    if not ordered_uids:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    maxn = min(int(cfg["max_depth"]), len(ordered_uids))
    ordered_uids = ordered_uids[:maxn]
    candidates = candidates[candidates["feature_uid"].astype(str).isin(ordered_uids)].copy()
    candidates = candidates.set_index("feature_uid", drop=False).loc[ordered_uids].reset_index(drop=True)

    values = matrix[ordered_uids].apply(safe_numeric)
    min_pairwise_n = int(cfg["min_pairwise_n"])
    corr = values.corr(method="spearman", min_periods=min_pairwise_n)
    corr_abs = corr.abs()
    nmat = pairwise_n_matrix(values)

    # Long-form pairwise audit once at maximum diagnostic depth.
    pair_rows: List[dict] = []
    for i in range(len(ordered_uids)):
        for j in range(i + 1, len(ordered_uids)):
            a, b = ordered_uids[i], ordered_uids[j]
            pair_rows.append({
                "feature_uid_1": a,
                "feature_uid_2": b,
                "rank_1": int(candidates.loc[i, "eligible_root_rank"]),
                "rank_2": int(candidates.loc[j, "eligible_root_rank"]),
                "spearman_rho": corr.loc[a, b],
                "abs_spearman_rho": corr_abs.loc[a, b],
                "pairwise_n": int(nmat.loc[a, b]),
                "passes_min_pairwise_n": bool(int(nmat.loc[a, b]) >= min_pairwise_n),
            })
    pair_df = pd.DataFrame(pair_rows)

    # Precompute greedy novelty for every rho at every integer depth.
    novelty = {}
    for rho in [float(x) for x in cfg["redundancy_rhos"]]:
        cumulative, flags, _ = greedy_novelty_series(ordered_uids, corr_abs, rho)
        novelty[rho] = {"cumulative": cumulative, "flags": flags}

    rows: List[dict] = []
    for depth in range(1, maxn + 1):
        top = candidates.head(depth).copy()
        uids = top["feature_uid"].astype(str).tolist()
        nth = top.iloc[-1]

        csub = corr_abs.loc[uids, uids]
        nsub = nmat.loc[uids, uids]
        vals = upper_triangle_values(csub)
        nvals = upper_triangle_values(nsub)
        valid_corr = vals.notna()
        n_pairs_total = int(depth * (depth - 1) / 2)
        n_pairs_valid = int(valid_corr.sum())

        base = {
            "depth": int(depth),
            "n_candidates_available": int(len(candidates)),
            "n_pairs_total": n_pairs_total,
            "n_pairs_valid_corr": n_pairs_valid,
            "frac_pairs_valid_corr": float(n_pairs_valid / n_pairs_total) if n_pairs_total else np.nan,
            "median_abs_rho": float(vals[valid_corr].median()) if n_pairs_valid else np.nan,
            "q90_abs_rho": float(vals[valid_corr].quantile(.90)) if n_pairs_valid else np.nan,
            "max_abs_rho": float(vals[valid_corr].max()) if n_pairs_valid else np.nan,
            "median_pairwise_n": float(nvals.median()) if len(nvals) else np.nan,

            "nth_oof_metric": nth.get("oof_metric", np.nan),
            "nth_oof_margin": nth.get("oof_margin", np.nan),
            "nth_fold_sd": nth.get("fold_sd", np.nan),
            "nth_fold_sd_headroom": nth.get("fold_sd_headroom", np.nan),
            "nth_direction_consistency": nth.get("direction_consistency", np.nan),
            "nth_direction_margin": nth.get("direction_margin", np.nan),
            "nth_delta_clinical": nth.get("delta_clinical", np.nan),
            "nth_delta_margin": nth.get("delta_margin", np.nan),
            "nth_nonmissing_fraction": nth.get("nonmissing_fraction", np.nan),
            "nth_root_evidence_score": nth.get("root_candidate_evidence_score", np.nan),

            "median_topn_oof": median_or_nan(top["oof_metric"]),
            "q25_topn_oof_margin": quantile_or_nan(top["oof_margin"], .25),
            "median_topn_fold_sd": median_or_nan(top["fold_sd"]),
            "q25_topn_fold_sd_headroom": quantile_or_nan(top["fold_sd_headroom"], .25),
            "median_topn_root_evidence_score": median_or_nan(top["root_candidate_evidence_score"]),
        }

        for rho in [float(x) for x in cfg["redundancy_rhos"]]:
            tag = f"rho{int(round(rho*100)):02d}"
            if n_pairs_valid:
                high = vals[valid_corr] >= rho
                n_high = int(high.sum())
            else:
                n_high = 0
            involved = set()
            for i in range(depth):
                for j in range(i + 1, depth):
                    a, b = uids[i], uids[j]
                    v = csub.loc[a, b]
                    if pd.notna(v) and float(v) >= rho:
                        involved.add(a); involved.add(b)
            greedy_n = novelty[rho]["cumulative"][depth - 1]
            novelty_flag = novelty[rho]["flags"][depth - 1]
            window = int(cfg.get("diminishing_window", 5))
            start = max(0, depth - window)
            recent_new = int(sum(novelty[rho]["flags"][start:depth]))
            base.update({
                f"n_pairs_ge_{tag}": n_high,
                f"frac_valid_pairs_ge_{tag}": float(n_high / n_pairs_valid) if n_pairs_valid else np.nan,
                f"n_features_in_redundant_pairs_{tag}": int(len(involved)),
                f"frac_features_in_redundant_pairs_{tag}": float(len(involved) / depth),
                f"graph_components_{tag}": int(graph_component_count(uids, corr_abs, rho)),
                f"greedy_nonredundant_n_{tag}": int(greedy_n),
                f"novelty_yield_{tag}": float(greedy_n / depth),
                f"nth_candidate_is_novel_{tag}": int(novelty_flag),
                f"new_novel_in_last_{window}_{tag}": recent_new,
            })
        rows.append(base)

    diag = pd.DataFrame(rows)
    return diag, pair_df, candidates


# =============================================================================
# Transparent cap recommendation
# =============================================================================


def recommend_caps(diag: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()

    rho = float(cfg["primary_redundancy_rho"])
    tag = f"rho{int(round(rho*100)):02d}"
    yield_col = f"novelty_yield_{tag}"
    novel_col = f"greedy_nonredundant_n_{tag}"
    recent_col = f"new_novel_in_last_{int(cfg.get('diminishing_window',5))}_{tag}"

    rows: List[dict] = []
    max_depth_observed = int(diag["depth"].max())

    profile_caps: Dict[str, int] = {}
    for profile, min_yield in cfg["novelty_yield_profiles"].items():
        feasible = diag[
            safe_numeric(diag[yield_col]).notna()
            & (safe_numeric(diag[yield_col]) >= float(min_yield))
        ].copy()
        if feasible.empty:
            cap = 1
        else:
            # Maximize cumulative nonredundant candidates; when tied use the
            # shallower depth. This is transparent and avoids rewarding redundant
            # depth for its own sake.
            best_novel = feasible[novel_col].max()
            cap = int(feasible[feasible[novel_col] == best_novel]["depth"].min())
        profile_caps[str(profile)] = cap

    # Diminishing-return elbow: first depth where two consecutive windows each
    # add no more than the configured number of new nonredundant candidates.
    window = int(cfg.get("diminishing_window", 5))
    max_new = int(cfg.get("max_new_novel_in_window", 2))
    flags = diag[f"nth_candidate_is_novel_{tag}"].astype(int).tolist()
    elbow = None
    for end1 in range(window, len(flags) - window + 1):
        first_gain = sum(flags[end1-window:end1])
        second_gain = sum(flags[end1:end1+window])
        if first_gain <= max_new and second_gain <= max_new:
            elbow = max(1, end1 - window + 1)
            break

    balanced = int(profile_caps.get("balanced", profile_caps[list(profile_caps)[0]]))
    if elbow is not None:
        recommended = int(min(balanced, elbow))
        rationale = (
            f"min(balanced novelty-yield cap={balanced}, sustained redundancy elbow={elbow})"
        )
    else:
        recommended = balanced
        rationale = f"balanced novelty-yield cap={balanced}; no sustained redundancy elbow detected"

    # If recommendation is at the maximum explored depth, flag the boundary so
    # the user knows the cap is not fully identified by the current search range.
    boundary_hit = bool(recommended >= max_depth_observed)

    row = {
        "primary_redundancy_rho": rho,
        "max_depth_observed": max_depth_observed,
        "n_candidates_available": int(diag["n_candidates_available"].max()),
        "redundancy_elbow": elbow if elbow is not None else np.nan,
        "recommended_cap": int(recommended),
        "recommendation_rationale": rationale,
        "recommendation_hits_max_depth_boundary": boundary_hit,
    }
    for profile, cap in profile_caps.items():
        row[f"recommended_cap_{profile}"] = int(cap)
        row[f"min_novelty_yield_{profile}"] = float(cfg["novelty_yield_profiles"][profile])

    at = diag[diag["depth"] == recommended].iloc[0]
    for c in [
        "nth_oof_metric", "nth_oof_margin", "nth_fold_sd", "nth_fold_sd_headroom",
        "nth_root_evidence_score", "q25_topn_oof_margin", "q25_topn_fold_sd_headroom",
        "median_abs_rho", "q90_abs_rho", "max_abs_rho",
        f"frac_valid_pairs_ge_{tag}", f"frac_features_in_redundant_pairs_{tag}",
        f"greedy_nonredundant_n_{tag}", f"novelty_yield_{tag}", "frac_pairs_valid_corr"
    ]:
        if c in at.index:
            row[f"at_recommended__{c}"] = at[c]

    rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================


def save_line_plot(
    diag: pd.DataFrame,
    ycols: Sequence[Tuple[str, str]],
    ylabel: str,
    title: str,
    path: Path,
    hline: Optional[float] = None,
    recommended_cap: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for col, label in ycols:
        if col not in diag.columns or diag[col].notna().sum() == 0:
            continue
        ax.plot(diag["depth"], diag[col], marker="o", markersize=3, linewidth=1.2, label=label)
    if hline is not None:
        ax.axhline(float(hline), linestyle="--", linewidth=1)
    if recommended_cap is not None:
        ax.axvline(int(recommended_cap), linestyle=":", linewidth=1.2, label=f"recommended cap={recommended_cap}")
    ax.set_xlabel("Candidate depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(ax.lines) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_root_plots(
    diag: pd.DataFrame,
    recommendation: pd.DataFrame,
    root: str,
    plot_dir: Path,
    cfg: Mapping,
) -> None:
    ensure_dir(plot_dir)
    if diag.empty:
        return
    cap = int(recommendation.iloc[0]["recommended_cap"]) if not recommendation.empty else None
    primary_rho = float(cfg["primary_redundancy_rho"])
    primary_tag = f"rho{int(round(primary_rho*100)):02d}"

    save_line_plot(
        diag,
        [("nth_oof_margin", "Nth candidate OOF margin"),
         ("q25_topn_oof_margin", "25th percentile top-N OOF margin")],
        "OOF margin above context threshold",
        f"{root}: predictive quality versus candidate depth",
        plot_dir / "01_oof_margin_by_depth.png",
        hline=0,
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        [("nth_fold_sd_headroom", "Nth candidate SD headroom"),
         ("q25_topn_fold_sd_headroom", "25th percentile top-N SD headroom")],
        "Fold-SD headroom below maximum allowed",
        f"{root}: CV stability versus candidate depth",
        plot_dir / "02_fold_sd_headroom_by_depth.png",
        hline=0,
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        [("nth_root_evidence_score", "Nth candidate root evidence score"),
         ("median_topn_root_evidence_score", "Median top-N root evidence score")],
        "Root-specific evidence score",
        f"{root}: root evidence rank profile",
        plot_dir / "03_root_evidence_by_depth.png",
        recommended_cap=cap,
    )

    redundancy_lines = []
    novelty_lines = []
    nonred_lines = []
    for rho in [float(x) for x in cfg["redundancy_rhos"]]:
        tag = f"rho{int(round(rho*100)):02d}"
        redundancy_lines.append((f"frac_features_in_redundant_pairs_{tag}", f"|rho|≥{rho:.2f}"))
        novelty_lines.append((f"novelty_yield_{tag}", f"|rho|≥{rho:.2f}"))
        nonred_lines.append((f"greedy_nonredundant_n_{tag}", f"|rho|≥{rho:.2f}"))

    save_line_plot(
        diag,
        redundancy_lines,
        "Fraction of features in ≥1 near-redundant pair",
        f"{root}: redundancy burden versus candidate depth",
        plot_dir / "04_redundant_feature_fraction_by_depth.png",
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        novelty_lines,
        "Greedy nonredundant yield / nominated depth",
        f"{root}: nonredundant yield versus candidate depth",
        plot_dir / "05_novelty_yield_by_depth.png",
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        nonred_lines,
        "Cumulative nonredundant candidates",
        f"{root}: effective nonredundant candidates",
        plot_dir / "06_nonredundant_count_by_depth.png",
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        [("median_abs_rho", "Median |rho|"), ("q90_abs_rho", "90th percentile |rho|"), ("max_abs_rho", "Maximum |rho|")],
        "Absolute within-root Spearman correlation",
        f"{root}: correlation structure versus depth",
        plot_dir / "07_abs_rho_summary_by_depth.png",
        recommended_cap=cap,
    )
    save_line_plot(
        diag,
        [("frac_pairs_valid_corr", "Fraction with pairwise N sufficient")],
        "Correlation-estimation coverage",
        f"{root}: pairwise correlation coverage",
        plot_dir / "08_pairwise_coverage_by_depth.png",
        recommended_cap=cap,
    )

    # Compact multi-panel dashboard for rapid manual review.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        ("nth_oof_margin", "Nth OOF margin"),
        ("nth_fold_sd_headroom", "Nth fold-SD headroom"),
        ("nth_root_evidence_score", "Nth root evidence score"),
        (f"frac_features_in_redundant_pairs_{primary_tag}", f"Redundant feature fraction (|rho|≥{primary_rho:.2f})"),
        (f"novelty_yield_{primary_tag}", f"Novelty yield (|rho|≥{primary_rho:.2f})"),
        (f"greedy_nonredundant_n_{primary_tag}", f"Nonredundant count (|rho|≥{primary_rho:.2f})"),
    ]
    for ax, (col, label) in zip(axes.ravel(), panels):
        ax.plot(diag["depth"], diag[col], marker="o", markersize=3)
        if col in {"nth_oof_margin", "nth_fold_sd_headroom"}:
            ax.axhline(0, linestyle="--", linewidth=1)
        if cap is not None:
            ax.axvline(cap, linestyle=":", linewidth=1)
        ax.set_xlabel("Depth")
        ax.set_ylabel(label)
        ax.set_title(label)
    fig.suptitle(f"{root}: candidate-cap sensitivity dashboard", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(plot_dir / "00_candidate_cap_dashboard.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Context worker
# =============================================================================


def command_worker(cfg: Mapping, array_id: int) -> None:
    outroot = Path(cfg["output_root"])
    index = pd.read_csv(outroot / "context_index.csv")
    match = index[index["array_id"].astype(int) == int(array_id)]
    if match.empty:
        raise IndexError(f"array_id={array_id} not found")
    row = match.iloc[0]
    cdir = ensure_dir(outroot / "contexts" / str(row["context_slug"]))

    candidates = read_table(row["candidate_shard"])
    cache_dir = outroot / "shared_cache" / str(row["cache_slug"])
    matrix_path = cache_dir / "patient_feature_matrix.parquet"
    matrix = read_table(matrix_path)

    summary_rows: List[dict] = []
    diag_parts: List[pd.DataFrame] = []
    pair_parts: List[pd.DataFrame] = []
    candidate_parts: List[pd.DataFrame] = []
    rec_parts: List[pd.DataFrame] = []

    if candidates.empty:
        pd.DataFrame([{
            "array_id": array_id,
            **{c: row[c] for c in CONTEXT_COLS},
            "status": "included_context_zero_passing_candidates",
        }]).to_csv(cdir / "context_summary.csv", index=False)
        (cdir / ".done").write_text("zero_candidates\n")
        return

    for root, g in candidates.groupby(ROOT_COL, dropna=False, sort=True):
        root = str(root)
        rdir = ensure_dir(cdir / "roots" / slugify([root]))
        diag, pair_df, used_candidates = compute_root_depth_diagnostics(g, matrix, cfg)

        if diag.empty:
            summary_rows.append({
                "array_id": array_id,
                **{c: row[c] for c in CONTEXT_COLS},
                "feature_source": root,
                "status": "zero_matrix_features_for_root",
                "n_passing_candidates": int(len(g)),
            })
            continue

        for c in CONTEXT_COLS:
            diag[c] = row[c]
        diag["feature_source"] = root
        diag["array_id"] = int(array_id)
        for c in CONTEXT_COLS:
            pair_df[c] = row[c]
        pair_df["feature_source"] = root
        pair_df["array_id"] = int(array_id)
        used_candidates["array_id"] = int(array_id)

        rec = recommend_caps(diag, cfg)
        for c in CONTEXT_COLS:
            rec[c] = row[c]
        rec["feature_source"] = root
        rec["array_id"] = int(array_id)

        diag.to_csv(rdir / "depth_diagnostics_all_integer_depths.csv", index=False)
        pair_df.to_csv(rdir / "pairwise_correlations_top_max_depth.csv", index=False)
        used_candidates.to_csv(rdir / "candidate_rank_membership.csv", index=False)
        rec.to_csv(rdir / "cap_recommendation.csv", index=False)
        save_root_plots(diag, rec, root, rdir / "plots", cfg)

        # Concise human-readable root report.
        rr = rec.iloc[0]
        lines = [
            f"Context: {row['cohort']} | {row['panel']} | {row['endpoint']}",
            f"Prep root: {root}",
            f"Passing candidates available (within max-depth shard): {len(g)}",
            f"Recommended cap: {int(rr['recommended_cap'])}",
            f"Strict / balanced / permissive: {int(rr.get('recommended_cap_strict', np.nan))} / {int(rr.get('recommended_cap_balanced', np.nan))} / {int(rr.get('recommended_cap_permissive', np.nan))}",
            f"Redundancy elbow: {rr.get('redundancy_elbow', np.nan)}",
            f"Rationale: {rr['recommendation_rationale']}",
            f"Boundary hit: {bool(rr['recommendation_hits_max_depth_boundary'])}",
        ]
        (rdir / "cap_recommendation.txt").write_text("\n".join(lines) + "\n")

        summary_rows.append({
            "array_id": array_id,
            **{c: row[c] for c in CONTEXT_COLS},
            "feature_source": root,
            "status": "complete",
            "n_passing_candidates_in_shard": int(len(g)),
            "n_candidates_with_matrix": int(len(used_candidates)),
            "recommended_cap": int(rr["recommended_cap"]),
            "recommended_cap_strict": int(rr.get("recommended_cap_strict", rr["recommended_cap"])),
            "recommended_cap_balanced": int(rr.get("recommended_cap_balanced", rr["recommended_cap"])),
            "recommended_cap_permissive": int(rr.get("recommended_cap_permissive", rr["recommended_cap"])),
            "redundancy_elbow": rr.get("redundancy_elbow", np.nan),
            "boundary_hit": bool(rr["recommendation_hits_max_depth_boundary"]),
        })
        diag_parts.append(diag)
        pair_parts.append(pair_df)
        candidate_parts.append(used_candidates)
        rec_parts.append(rec)

    pd.DataFrame(summary_rows).to_csv(cdir / "context_summary.csv", index=False)
    if diag_parts:
        pd.concat(diag_parts, ignore_index=True).to_csv(cdir / "all_root_depth_diagnostics.csv", index=False)
    if pair_parts:
        pd.concat(pair_parts, ignore_index=True).to_csv(cdir / "all_root_pairwise_correlations.csv.gz", index=False, compression="gzip")
    if candidate_parts:
        pd.concat(candidate_parts, ignore_index=True).to_csv(cdir / "all_root_candidate_membership.csv", index=False)
    if rec_parts:
        pd.concat(rec_parts, ignore_index=True).to_csv(cdir / "all_root_cap_recommendations.csv", index=False)

    (cdir / ".done").write_text("complete\n")
    log(f"[CONTEXT {array_id} DONE] roots={len(summary_rows)}")


# =============================================================================
# Aggregation and cross-context visualizations
# =============================================================================


def aggregate_panel_root_diagnostics(all_diag: pd.DataFrame) -> pd.DataFrame:
    if all_diag.empty:
        return pd.DataFrame()
    numeric_cols = [
        c for c in all_diag.columns
        if c not in ["array_id"] + CONTEXT_COLS + [ROOT_COL]
        and pd.api.types.is_numeric_dtype(all_diag[c])
    ]
    rows = []
    for (panel, root, depth), g in all_diag.groupby(["panel", ROOT_COL, "depth"], dropna=False):
        row = {
            "panel": panel,
            ROOT_COL: root,
            "depth": int(depth),
            "n_contexts_with_depth": int(len(g)),
        }
        for c in numeric_cols:
            x = safe_numeric(g[c]).dropna()
            if x.empty:
                continue
            row[f"median__{c}"] = float(x.median())
            row[f"q25__{c}"] = float(x.quantile(.25))
            row[f"q75__{c}"] = float(x.quantile(.75))
        rows.append(row)
    return pd.DataFrame(rows)


def save_aggregate_panel_plots(panel_diag: pd.DataFrame, panel: str, outdir: Path, cfg: Mapping) -> None:
    d = panel_diag[panel_diag["panel"] == panel].copy()
    if d.empty:
        return
    ensure_dir(outdir)
    rho = float(cfg["primary_redundancy_rho"])
    tag = f"rho{int(round(rho*100)):02d}"
    metrics = [
        ("median__nth_oof_margin", "Median Nth-candidate OOF margin", "01_median_nth_oof_margin.png", True),
        ("median__nth_fold_sd_headroom", "Median Nth-candidate fold-SD headroom", "02_median_nth_fold_sd_headroom.png", True),
        (f"median__novelty_yield_{tag}", f"Median novelty yield (|rho|≥{rho:.2f})", "03_median_novelty_yield.png", False),
        (f"median__greedy_nonredundant_n_{tag}", f"Median nonredundant candidate count (|rho|≥{rho:.2f})", "04_median_nonredundant_count.png", False),
        (f"median__frac_features_in_redundant_pairs_{tag}", f"Median redundant-feature fraction (|rho|≥{rho:.2f})", "05_median_redundant_fraction.png", False),
    ]
    for col, ylabel, filename, zero in metrics:
        if col not in d.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for root, g in d.groupby(ROOT_COL, sort=True):
            g = g.sort_values("depth")
            ax.plot(g["depth"], g[col], marker="o", markersize=3, linewidth=1.2, label=str(root))
        if zero:
            ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Candidate depth")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel}: {ylabel}")
        ax.legend(title="Prep root", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)


def command_aggregate(cfg: Mapping) -> None:
    outroot = Path(cfg["output_root"])
    index = pd.read_csv(outroot / "context_index.csv")

    summaries = []
    recs = []
    diags = []
    missing = []
    for _, row in index.iterrows():
        cdir = outroot / "contexts" / str(row["context_slug"])
        if not (cdir / "context_summary.csv").exists():
            missing.append({"array_id": row["array_id"], **{c: row[c] for c in CONTEXT_COLS}})
            continue
        summaries.append(pd.read_csv(cdir / "context_summary.csv"))
        if (cdir / "all_root_cap_recommendations.csv").exists():
            recs.append(pd.read_csv(cdir / "all_root_cap_recommendations.csv"))
        if (cdir / "all_root_depth_diagnostics.csv").exists():
            diags.append(pd.read_csv(cdir / "all_root_depth_diagnostics.csv"))

    if missing:
        pd.DataFrame(missing).to_csv(outroot / "missing_context_workers.csv", index=False)
        raise RuntimeError(f"{len(missing)} context workers missing; see missing_context_workers.csv")

    all_summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    all_rec = pd.concat(recs, ignore_index=True, sort=False) if recs else pd.DataFrame()
    all_diag = pd.concat(diags, ignore_index=True, sort=False) if diags else pd.DataFrame()

    all_summary.to_csv(outroot / "all_context_root_cap_summary.csv", index=False)
    all_rec.to_csv(outroot / "all_context_root_cap_recommendations.csv", index=False)
    all_diag.to_csv(outroot / "all_context_root_depth_diagnostics.csv.gz", index=False, compression="gzip")

    panel_diag = aggregate_panel_root_diagnostics(all_diag)
    panel_diag.to_csv(outroot / "panel_root_depth_diagnostics.csv", index=False)

    # Robust panel x root consensus cap: median context-level advisory cap.
    consensus_rows = []
    if not all_rec.empty:
        for (panel, root), g in all_rec.groupby(["panel", ROOT_COL], dropna=False):
            caps = safe_numeric(g["recommended_cap"]).dropna()
            strict = safe_numeric(g.get("recommended_cap_strict", pd.Series(dtype=float))).dropna()
            balanced = safe_numeric(g.get("recommended_cap_balanced", pd.Series(dtype=float))).dropna()
            permissive = safe_numeric(g.get("recommended_cap_permissive", pd.Series(dtype=float))).dropna()
            boundary = g.get("recommendation_hits_max_depth_boundary", pd.Series(False, index=g.index)).map(parse_bool)
            consensus_rows.append({
                "panel": panel,
                ROOT_COL: root,
                "n_contexts": int(len(g)),
                "consensus_cap_median": int(round(float(caps.median()))) if not caps.empty else np.nan,
                "cap_q25": float(caps.quantile(.25)) if not caps.empty else np.nan,
                "cap_q75": float(caps.quantile(.75)) if not caps.empty else np.nan,
                "cap_min": float(caps.min()) if not caps.empty else np.nan,
                "cap_max": float(caps.max()) if not caps.empty else np.nan,
                "median_strict_cap": float(strict.median()) if not strict.empty else np.nan,
                "median_balanced_cap": float(balanced.median()) if not balanced.empty else np.nan,
                "median_permissive_cap": float(permissive.median()) if not permissive.empty else np.nan,
                "fraction_contexts_hitting_max_depth_boundary": float(boundary.mean()) if len(boundary) else np.nan,
            })
    consensus = pd.DataFrame(consensus_rows)
    consensus.to_csv(outroot / "panel_root_consensus_cap_recommendations.csv", index=False)

    # Manual review table preserves both context-specific advisory output and the
    # more defensible fixed panel x root consensus option.
    review = all_rec.copy()
    if not review.empty and not consensus.empty:
        review = review.merge(
            consensus[["panel", ROOT_COL, "consensus_cap_median", "cap_q25", "cap_q75"]],
            on=["panel", ROOT_COL], how="left", validate="many_to_one"
        )
    if not review.empty:
        review["manual_include_root"] = ""
        review["manual_candidate_cap"] = ""
        review["manual_notes"] = ""
    review.to_csv(outroot / "cap_manual_review_template.csv", index=False)

    plotroot = ensure_dir(outroot / "aggregate_plots")
    for panel in sorted(panel_diag["panel"].dropna().astype(str).unique()) if not panel_diag.empty else []:
        save_aggregate_panel_plots(panel_diag, panel, plotroot / panel, cfg)

    # Distribution of context recommendations by prep root.
    if not all_rec.empty:
        for panel, gpanel in all_rec.groupby("panel"):
            roots = sorted(gpanel[ROOT_COL].dropna().astype(str).unique())
            data = [safe_numeric(gpanel[gpanel[ROOT_COL].astype(str) == r]["recommended_cap"]).dropna().to_numpy() for r in roots]
            if any(len(x) for x in data):
                fig, ax = plt.subplots(figsize=(max(8, 1.5*len(roots)), 5.5))
                ax.boxplot(data, labels=roots, showfliers=False)
                ax.set_ylabel("Context-level recommended cap")
                ax.set_title(f"{panel}: context-level cap recommendations by prep root")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                fig.savefig(plotroot / f"{panel}_context_cap_distribution.png", dpi=250, bbox_inches="tight")
                plt.close(fig)

    lines = [
        "STAGE 2A CANDIDATE-CAP SENSITIVITY: ADVISORY MATHEMATICAL SUMMARY",
        "=" * 78,
        "",
        "All candidate pools were first restricted by the frozen context-specific",
        "quality/stability eligibility rules. Cap selection therefore addresses the",
        "additional question of diminishing nonredundant information with increasing",
        "candidate depth, rather than replacing the quality thresholds.",
        "",
        f"Primary near-redundancy threshold: |Spearman rho| >= {float(cfg['primary_redundancy_rho']):.2f}",
        f"Minimum pairwise-complete patients: {int(cfg['min_pairwise_n'])}",
        f"Maximum depth explored: {int(cfg['max_depth'])}",
        "",
        "Panel x prep-root consensus recommendations (median across contexts):",
    ]
    if not consensus.empty:
        for _, r in consensus.sort_values(["panel", ROOT_COL]).iterrows():
            lines.append(
                f"  {r['panel']:>2} | {str(r[ROOT_COL]):<22} -> "
                f"cap {r['consensus_cap_median']} "
                f"(context IQR {r['cap_q25']:.1f}-{r['cap_q75']:.1f}; "
                f"boundary-hit fraction {r['fraction_contexts_hitting_max_depth_boundary']:.2f})"
            )
    lines.extend([
        "",
        "Interpretation:",
        "  * A boundary-hit fraction >0 suggests increasing max_depth before freezing that root.",
        "  * Large context IQR indicates heterogeneity; inspect context dashboards before using a fixed root cap.",
        "  * For manuscript-facing reproducibility, a fixed panel x root cap is generally preferable",
        "    to hand-tuning a separate cap for every cohort/endpoint, while contexts with fewer eligible",
        "    candidates simply contribute fewer than the cap.",
        "",
    ])
    (outroot / "cap_recommendations_summary.txt").write_text("\n".join(lines))

    log(f"[DONE] aggregate outputs -> {outroot}")
    log(f"[REVIEW] {outroot / 'cap_manual_review_template.csv'}")
    log(f"[REVIEW] {outroot / 'panel_root_consensus_cap_recommendations.csv'}")
    log(f"[REVIEW] {outroot / 'cap_recommendations_summary.txt'}")


# =============================================================================
# CLI
# =============================================================================


def resolve_id(value: Optional[int], env_name: str) -> int:
    if value is not None:
        return int(value)
    env = os.environ.get(env_name)
    if env is None:
        raise ValueError(f"Provide ID argument or set {env_name}")
    return int(env)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup")
    p.add_argument("--config", required=True)

    p = sub.add_parser("cache-worker")
    p.add_argument("--config", required=True)
    p.add_argument("--cache-id", type=int, default=None)

    p = sub.add_parser("worker")
    p.add_argument("--config", required=True)
    p.add_argument("--array-id", type=int, default=None)

    p = sub.add_parser("aggregate")
    p.add_argument("--config", required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "setup":
        command_setup(cfg)
    elif args.command == "cache-worker":
        command_cache_worker(cfg, resolve_id(args.cache_id, "SLURM_ARRAY_TASK_ID"))
    elif args.command == "worker":
        command_worker(cfg, resolve_id(args.array_id, "SLURM_ARRAY_TASK_ID"))
    elif args.command == "aggregate":
        command_aggregate(cfg)


if __name__ == "__main__":
    main()
