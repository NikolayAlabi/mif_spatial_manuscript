#!/usr/bin/env python3
"""
stage2a_steps1_3_context_review_v3.py

Stage 2A, steps 1-3 with comprehensive context review outputs:
  1. Inventory and standardize completed Stage 1 univariate results.
  2. Choose one best transform per underlying variable within each context.
  3. Produce context-level evidence summaries for MANUAL review.

This script deliberately stops before biological microcompression and before final
candidate nomination. It does not decide which contexts enter global module
discovery. Instead, it creates a review package containing all best-transform
variables, context-strength summaries, transform-selection audits, comprehensive CSV summaries, per-context figures, and HTML review reports.

Parallel design
---------------
The inventory command writes one input shard per context. The worker command reads
one shard and uses one CPU. A Slurm array can therefore run one task per context.

Candidate evidence score
------------------------
P-values are NOT included in the primary candidate evidence score by default.
With cohorts of roughly 40-50 patients, nominal P-values and FDR can be unstable
and highly dependent on event count. They are retained as corroborating review
columns only.

Default score (percentile ranks within context):
    0.55 * OOF discrimination
  + 0.20 * stability
  + 0.15 * improvement versus clinical model
  + 0.10 * completeness
Available weights are renormalized row-wise when optional metrics are missing.

Usage
-----
1) Build inventory and context shards:
   python stage2a_steps1_3_context_review_v3.py inventory --config CONFIG.json

2) Run one context (or use SLURM_ARRAY_TASK_ID):
   python stage2a_steps1_3_context_review_v3.py worker --config CONFIG.json --array-id 0

3) Aggregate completed context outputs:
   python stage2a_steps1_3_context_review_v3.py aggregate --config CONFIG.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Constants and aliases
# =============================================================================

CONTEXT_COLS = [
    "cohort",
    "panel",
    "endpoint",
    "sample_type",
    "patient_subset",
    "agg",
]
FEATURE_ID_COLS = ["feature_source", "feature_group", "feature"]

DEFAULT_DISCOVERY_COHORTS = ["NAC2020", "PURE01", "BLASST", "No-NAC"]
DEFAULT_PANELS = ["AR", "BT"]
DEFAULT_ENDPOINTS = ["complete_response", "any_response", "OS", "RFS"]
DEFAULT_SAMPLE_TYPES = ["TURBT"]
DEFAULT_FEATURE_GROUPS = ["NN", "athena", "cell_features", "triads"]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "cohort": ["cohort", "dataset", "study"],
    "panel": ["panel", "mif_panel"],
    "endpoint": ["endpoint", "outcome", "target", "clinical_endpoint"],
    "sample_type": ["sample_type", "specimen_type", "tissue_timepoint"],
    "patient_subset": ["patient_subset", "subset", "analysis_subset"],
    "agg": ["agg", "core_agg", "aggregation"],
    "feature_source": ["feature_source", "source", "prep_root", "phenotype_source"],
    "feature_group": ["feature_group", "group", "feature_family"],
    "feature": ["feature", "feature_name", "biomarker", "predictor", "variable"],
    "transform_mode": ["transform_mode", "selected_transform_mode", "transform"],
    "status": ["status", "fit_status", "analysis_status"],
    "n": ["n", "n_patients", "n_samples", "sample_size"],
    "n_events": ["n_events", "events", "event_n"],
    "n_positive": ["n_positive", "n_pos", "n_cases", "n_responders", "events_positive"],
    "n_negative": ["n_negative", "n_neg", "n_controls", "n_nonresponders"],
    "valid_folds": ["valid_folds", "n_valid_folds", "folds_valid", "n_folds_successful"],
    "nonmissing_fraction": [
        "nonmissing_fraction",
        "non_missing_fraction",
        "feature_nonmissing_fraction",
        "patient_nonmissing_fraction",
        "coverage",
    ],
    "oof_metric": [
        "primary_oof_metric",
        "biomarker_oof_auc",
        "biomarker_oof_cindex",
        "oof_auc",
        "oof_cindex",
        "pooled_oof_auc",
        "pooled_oof_cindex",
        "auc_oof",
        "cindex_oof",
        "mean_oof_auc",
        "mean_oof_cindex",
        "oof_metric",
    ],
    "clinical_oof_metric": [
        "clinical_oof_metric",
        "clinical_oof_auc",
        "clinical_oof_cindex",
        "clinical_auc_oof",
        "clinical_cindex_oof",
    ],
    "combined_oof_metric": [
        "combined_oof_metric",
        "combined_oof_auc",
        "combined_oof_cindex",
        "combined_auc_oof",
        "combined_cindex_oof",
    ],
    "delta_clinical": [
        "delta_clinical",
        "primary_delta_metric",
        "delta_oof_auc_vs_clinical",
        "delta_oof_cindex_vs_clinical",
        "delta_auc_vs_clinical",
        "delta_cindex_vs_clinical",
        "combined_minus_clinical_oof",
        "delta_combined_vs_clinical",
    ],
    "fold_sd": [
        "fold_sd",
        "primary_oof_sd",
        "biomarker_fold_auc_sd",
        "biomarker_fold_cindex_sd",
        "fold_auc_sd",
        "fold_cindex_sd",
        "oof_sd",
        "cv_sd",
    ],
    "direction_consistency": [
        "direction_consistency",
        "fold_direction_consistency",
        "effect_direction_consistency",
        "fraction_folds_same_direction",
        "prop_folds_same_direction",
    ],
    "effect": [
        "effect",
        "coef",
        "coefficient",
        "beta",
        "log_odds",
        "log_hr",
        "full_coef",
    ],
    "p_value": [
        "p_value",
        "p",
        "biomarker_p",
        "full_p_value",
        "wald_p",
        "coef_p",
        "p_value_y",
        "p_value_x",
    ],
}

COALESCE_TARGETS = {
    "oof_metric",
    "clinical_oof_metric",
    "combined_oof_metric",
    "delta_clinical",
    "fold_sd",
    "direction_consistency",
    "effect",
    "p_value",
}

KNOWN_COHORTS = ["NAC2020", "NAC2015", "PURE01", "BLASST", "No-NAC", "NoNAC", "KOLL"]
KNOWN_SAMPLE_TYPES = ["TURBT", "RC"]
KNOWN_FEATURE_GROUPS = {
    "nn": "NN",
    "nearest_neighbor": "NN",
    "nearest-neighbor": "NN",
    "athena": "athena",
    "cell_features": "cell_features",
    "cell-feature": "cell_features",
    "triads": "triads",
    "triad": "triads",
    "ratios": "cell_features",
}

# =============================================================================
# Generic helpers
# =============================================================================


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
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def percentile_rank(series: pd.Series, higher_better: bool = True) -> pd.Series:
    x = safe_numeric(series)
    return x.rank(pct=True, ascending=not higher_better, method="average")


def first_existing_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, preferred_path: Union[str, Path]) -> Path:
    preferred_path = Path(preferred_path)
    ensure_dir(preferred_path.parent)
    try:
        df.to_parquet(preferred_path, index=False)
        return preferred_path
    except (ImportError, ModuleNotFoundError, ValueError):
        fallback = preferred_path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False, compression="gzip")
        log(f"[WARN] Could not save parquet; used {fallback}")
        return fallback


def load_saved_table(preferred_path: Union[str, Path]) -> pd.DataFrame:
    preferred_path = Path(preferred_path)
    if preferred_path.exists():
        return read_table(preferred_path)
    fallback = preferred_path.with_suffix(".csv.gz")
    if fallback.exists():
        return read_table(fallback)
    raise FileNotFoundError(f"Neither {preferred_path} nor {fallback} exists")


def bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving missing values."""
    p = safe_numeric(p_values)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().clip(0, 1)
    if valid.empty:
        return out
    order = valid.sort_values().index
    m = len(order)
    ranked = valid.loc[order].to_numpy(dtype=float)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out.loc[order] = q
    return out


def normalize_cohort(value: object) -> str:
    text = str(value).strip()
    if text.lower().replace("-", "") == "nonac":
        return "No-NAC"
    return text


def normalize_panel(value: object) -> str:
    return str(value).strip().upper()


def normalize_endpoint(value: object) -> str:
    text = str(value).strip()
    low = text.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "complete_response": "complete_response",
        "cr": "complete_response",
        "any_response": "any_response",
        "response": "any_response",
        "overall_survival": "OS",
        "os": "OS",
        "recurrence_free_survival": "RFS",
        "rfs": "RFS",
    }
    return mapping.get(low, text)


def normalize_transform(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    mapping = {
        "z": "zscore",
        "z_score": "zscore",
        "standardized": "zscore",
        "log1pzscore": "log1p_zscore",
        "log1p_z": "log1p_zscore",
        "log1p": "log1p_zscore",
    }
    return mapping.get(text, text)


def normalize_feature_group(value: object) -> str:
    text = str(value).strip()
    key = text.lower().replace(" ", "_")
    return KNOWN_FEATURE_GROUPS.get(key, text)


def safe_context_slug(row: pd.Series, array_id: int) -> str:
    parts = []
    for col in CONTEXT_COLS:
        text = str(row.get(col, "NA"))
        text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
        parts.append(text or "NA")
    return f"context_{array_id:03d}__" + "__".join(parts)


def make_context_id(df: pd.DataFrame) -> pd.Series:
    return df[CONTEXT_COLS].astype(str).agg("__".join, axis=1)


def make_feature_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "|"
        + df["feature_group"].astype(str)
        + "|"
        + df["feature"].astype(str)
    )

# =============================================================================
# Path inference and schema standardization
# =============================================================================


def infer_from_path(path: Union[str, Path]) -> Dict[str, str]:
    text = str(path)
    low = text.lower()
    out: Dict[str, str] = {}

    for cohort in KNOWN_COHORTS:
        if cohort.lower() in low or cohort.lower().replace("-", "") in low.replace("-", ""):
            out["cohort"] = normalize_cohort(cohort)
            break

    for panel in ["AR", "BT"]:
        if re.search(rf"(^|[/_\-.]){panel.lower()}($|[/_\-.])", low):
            out["panel"] = panel
            break

    endpoint_patterns = {
        "complete_response": ["complete_response", "complete-response"],
        "any_response": ["any_response", "any-response"],
        "RFS": ["/rfs/", "_rfs_", "endpoint=rfs", "endpoint-rfs"],
        "OS": ["/os/", "_os_", "endpoint=os", "endpoint-os"],
    }
    for endpoint, patterns in endpoint_patterns.items():
        if any(pattern in low for pattern in patterns):
            out["endpoint"] = endpoint
            break

    for sample_type in KNOWN_SAMPLE_TYPES:
        if re.search(rf"(^|[/_\-.]){sample_type.lower()}($|[/_\-.])", low):
            out["sample_type"] = sample_type
            break

    if "no_adj_chemo" in low:
        out["patient_subset"] = "no_adj_chemo"
    elif "adj_chemo" in low:
        out["patient_subset"] = "adj_chemo"
    else:
        out["patient_subset"] = "all"

    agg_match = re.search(r"(?:agg=|coreagg=|core_agg=|agg-)(mean|median|max|min)", low)
    if agg_match:
        out["agg"] = agg_match.group(1)

    for token, normalized in KNOWN_FEATURE_GROUPS.items():
        if re.search(rf"(^|[/_\-.]){re.escape(token)}($|[/_\-.])", low):
            out["feature_group"] = normalized
            break

    for feature_source in [
        "phenotype_only",
        "AR_checkpoint_state",
        "AR_state",
        "compartment_state",
        "compartment",
    ]:
        if feature_source.lower() in low:
            out["feature_source"] = feature_source
            break

    if "log1p_zscore" in low or "log1pzscore" in low:
        out["transform_mode"] = "log1p_zscore"
    elif "zscore" in low or "z_score" in low:
        out["transform_mode"] = "zscore"
    elif re.search(r"(^|[/_\-.])raw($|[/_\-.])", low):
        out["transform_mode"] = "raw"

    return out


def standardize_one_table(
    df: pd.DataFrame,
    source_path: Path,
    explicit_map: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    explicit_map = dict(explicit_map or {})
    inferred = infer_from_path(source_path)
    out = pd.DataFrame(index=df.index)
    audit: dict = {
        "source_file": str(source_path),
        "n_rows": int(df.shape[0]),
        "accepted": False,
    }

    for target, aliases in COLUMN_ALIASES.items():
        explicit_col = explicit_map.get(target)
        if explicit_col is not None:
            if explicit_col not in df.columns:
                raise ValueError(f"Explicit mapping {target}->{explicit_col} missing in {source_path}")
            out[target] = df[explicit_col]
            audit[f"col_{target}"] = explicit_col
            continue

        if target in COALESCE_TARGETS:
            lower_map = {str(c).lower(): c for c in df.columns}
            matched: List[str] = []
            for alias in aliases:
                if alias in df.columns:
                    matched.append(alias)
                elif alias.lower() in lower_map:
                    matched.append(lower_map[alias.lower()])
            matched = list(dict.fromkeys(matched))
            if matched:
                out[target] = df[matched].bfill(axis=1).iloc[:, 0]
                audit[f"col_{target}"] = ";".join(map(str, matched))
            elif target in inferred:
                out[target] = inferred[target]
                audit[f"col_{target}"] = "<path-inferred>"
            else:
                out[target] = np.nan
                audit[f"col_{target}"] = "<missing>"
            continue

        source_col = first_existing_column(df, aliases)
        if source_col is not None:
            out[target] = df[source_col]
            audit[f"col_{target}"] = str(source_col)
        elif target in inferred:
            out[target] = inferred[target]
            audit[f"col_{target}"] = "<path-inferred>"
        else:
            out[target] = np.nan
            audit[f"col_{target}"] = "<missing>"

    if out["feature"].isna().all() or out["oof_metric"].isna().all():
        audit["reason"] = "missing_feature_or_oof_metric"
        return pd.DataFrame(), audit

    out["source_file"] = str(source_path)
    out["source_row"] = np.arange(len(out), dtype=int)

    out["sample_type"] = out["sample_type"].fillna("TURBT")
    out["patient_subset"] = out["patient_subset"].fillna("all")
    out["agg"] = out["agg"].fillna("median")
    out["transform_mode"] = out["transform_mode"].fillna("zscore")
    out["feature_source"] = out["feature_source"].fillna("phenotype_only")

    out["cohort"] = out["cohort"].map(normalize_cohort)
    out["panel"] = out["panel"].map(normalize_panel)
    out["endpoint"] = out["endpoint"].map(normalize_endpoint)
    out["feature_group"] = out["feature_group"].map(normalize_feature_group)
    out["transform_mode"] = out["transform_mode"].map(normalize_transform)
    out["feature"] = out["feature"].astype(str)
    out["feature_source"] = out["feature_source"].astype(str)

    numeric_cols = [
        "n",
        "n_events",
        "n_positive",
        "n_negative",
        "valid_folds",
        "nonmissing_fraction",
        "oof_metric",
        "clinical_oof_metric",
        "combined_oof_metric",
        "delta_clinical",
        "fold_sd",
        "direction_consistency",
        "effect",
        "p_value",
    ]
    for col in numeric_cols:
        out[col] = safe_numeric(out[col])

    # Derive delta when explicit delta is absent but both metrics are available.
    derived_delta = out["combined_oof_metric"] - out["clinical_oof_metric"]
    out["delta_clinical"] = out["delta_clinical"].combine_first(derived_delta)

    out["feature_uid"] = make_feature_uid(out)
    out["context_id"] = make_context_id(out)

    critical = ["cohort", "panel", "endpoint", "feature_group", "feature"]
    missing_critical = out[critical].isna().any(axis=1)
    out = out.loc[~missing_critical].copy()

    if out.empty:
        audit["reason"] = "all_rows_missing_critical_context_fields"
        return pd.DataFrame(), audit

    audit["accepted"] = True
    audit["reason"] = "ok"
    audit["n_standardized_rows"] = int(out.shape[0])
    return out, audit


def collect_inputs(cfg: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths: List[Path] = []
    for item in cfg.get("input_tables", []):
        paths.append(Path(item))

    for root_text in cfg.get("results_roots", []):
        root = Path(root_text)
        if not root.exists():
            log(f"[WARN] Missing results root: {root}")
            continue
        for pattern in cfg.get("include_globs", []):
            paths.extend(root.glob(pattern))

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)

    explicit_map = None
    if cfg.get("column_map_json"):
        explicit_map = read_json(cfg["column_map_json"])

    rows: List[pd.DataFrame] = []
    audits: List[dict] = []
    for index, path in enumerate(unique_paths, start=1):
        try:
            table = read_table(path)
            standardized, audit = standardize_one_table(table, path, explicit_map=explicit_map)
            audits.append(audit)
            if not standardized.empty:
                rows.append(standardized)
        except Exception as exc:
            audits.append({
                "source_file": str(path),
                "accepted": False,
                "reason": f"{type(exc).__name__}: {exc}",
            })
        if index % 250 == 0:
            log(f"[INFO] scanned {index}/{len(unique_paths)} files")

    master = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return master, pd.DataFrame(audits)

# =============================================================================
# Filtering and eligibility
# =============================================================================


def status_is_ok(value: object) -> bool:
    if pd.isna(value) or str(value).strip() == "":
        return True
    text = str(value).strip().lower()
    bad_tokens = ["fail", "error", "skip", "insufficient", "empty", "constant"]
    return not any(token in text for token in bad_tokens)


def apply_project_filters(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    out = df.copy()
    filters = {
        "cohort": cfg.get("discovery_cohorts"),
        "panel": cfg.get("panels"),
        "endpoint": cfg.get("endpoints"),
        "sample_type": cfg.get("sample_types"),
        "patient_subset": cfg.get("patient_subsets"),
        "agg": cfg.get("aggs"),
        "feature_group": cfg.get("feature_groups"),
        "transform_mode": cfg.get("transforms"),
    }
    for col, values in filters.items():
        if values is not None:
            out = out[out[col].astype(str).isin([str(v) for v in values])].copy()
    return out


def add_eligibility_flags(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    out = df.copy()
    reasons: List[str] = []
    all_reasons: List[str] = []

    for _, row in out.iterrows():
        reasons = []
        if not status_is_ok(row.get("status")):
            reasons.append("status_not_ok")
        if pd.isna(row.get("oof_metric")):
            reasons.append("missing_oof_metric")

        n = row.get("n")
        if pd.notna(n) and float(n) < float(cfg.get("min_n", 20)):
            reasons.append("n_below_min")

        valid_folds = row.get("valid_folds")
        if pd.notna(valid_folds) and float(valid_folds) < float(cfg.get("min_valid_folds", 4)):
            reasons.append("valid_folds_below_min")

        nonmissing = row.get("nonmissing_fraction")
        if pd.notna(nonmissing) and float(nonmissing) < float(cfg.get("min_nonmissing_fraction", 0.50)):
            reasons.append("nonmissing_fraction_below_min")

        endpoint = str(row.get("endpoint"))
        if endpoint in {"OS", "RFS"}:
            n_events = row.get("n_events")
            if pd.notna(n_events) and float(n_events) < float(cfg.get("min_events", 5)):
                reasons.append("events_below_min")
        elif endpoint in {"complete_response", "any_response"}:
            n_positive = row.get("n_positive")
            n_negative = row.get("n_negative")
            if pd.notna(n_positive) and float(n_positive) < float(cfg.get("min_class_n", 5)):
                reasons.append("positive_class_below_min")
            if pd.notna(n_negative) and float(n_negative) < float(cfg.get("min_class_n", 5)):
                reasons.append("negative_class_below_min")

        all_reasons.append(";".join(reasons))

    out["eligibility_reasons"] = all_reasons
    out["eligibility_pass"] = out["eligibility_reasons"].eq("")
    return out

# =============================================================================
# Best-transform selection and scoring
# =============================================================================


def transform_complexity(transform: str, preference_order: Sequence[str]) -> int:
    mapping = {name: idx for idx, name in enumerate(preference_order)}
    return mapping.get(str(transform), len(mapping) + 10)


def collapse_duplicate_rows_within_transform(group: pd.DataFrame) -> pd.DataFrame:
    """Keep one result row per transform if the same output was discovered more than once."""
    parts = []
    for _, sub in group.groupby("transform_mode", dropna=False):
        sub = sub.sort_values(
            ["eligibility_pass", "oof_metric", "fold_sd", "nonmissing_fraction", "p_value"],
            ascending=[False, False, True, False, True],
            na_position="last",
        )
        rep = sub.iloc[0].copy()
        rep["duplicate_rows_same_transform"] = int(sub.shape[0])
        parts.append(rep)
    return pd.DataFrame(parts)


def choose_best_transform_for_context(context_df: pd.DataFrame, cfg: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = FEATURE_ID_COLS
    selected_rows: List[pd.Series] = []
    audit_rows: List[dict] = []

    preference_order = [normalize_transform(x) for x in cfg.get("transform_preference_order", ["zscore", "log1p_zscore"])]
    material_oof_gain = float(cfg.get("transform_oof_material_gain", 0.01))
    material_sd_gain = float(cfg.get("transform_fold_sd_material_gain", 0.02))

    for _, raw_group in context_df.groupby(group_cols, dropna=False):
        group = collapse_duplicate_rows_within_transform(raw_group)
        valid = group[group["eligibility_pass"]].copy()

        if valid.empty:
            for _, row in group.iterrows():
                audit_rows.append({
                    **{col: row.get(col) for col in CONTEXT_COLS + FEATURE_ID_COLS},
                    "transform_mode": row.get("transform_mode"),
                    "selected": False,
                    "selection_reason": "no_eligible_transform",
                    "eligibility_reasons": row.get("eligibility_reasons"),
                    "oof_metric": row.get("oof_metric"),
                    "fold_sd": row.get("fold_sd"),
                    "p_value": row.get("p_value"),
                    "source_file": row.get("source_file"),
                })
            continue

        valid["transform_complexity"] = valid["transform_mode"].map(
            lambda x: transform_complexity(str(x), preference_order)
        )
        valid = valid.sort_values(
            ["oof_metric", "fold_sd", "nonmissing_fraction", "transform_complexity"],
            ascending=[False, True, False, True],
            na_position="last",
        ).copy()

        top = valid.iloc[0]
        reason = "only_eligible_transform"

        if valid.shape[0] > 1:
            second = valid.iloc[1]
            top_oof = float(top["oof_metric"])
            second_oof = float(second["oof_metric"])
            gap = top_oof - second_oof

            if gap >= material_oof_gain:
                chosen = top
                reason = f"material_oof_gain_ge_{material_oof_gain:g}"
            else:
                # If OOF performance is practically tied, allow a clearly more stable
                # transform to win. Otherwise retain the simpler transform.
                candidates = valid[valid["oof_metric"] >= top_oof - material_oof_gain].copy()
                sd_available = candidates["fold_sd"].notna().sum() >= 2
                if sd_available:
                    best_sd_row = candidates.sort_values(
                        ["fold_sd", "transform_complexity", "nonmissing_fraction"],
                        ascending=[True, True, False],
                        na_position="last",
                    ).iloc[0]
                    worst_competing_sd = candidates.loc[candidates.index != best_sd_row.name, "fold_sd"].min()
                    if pd.notna(worst_competing_sd) and (
                        float(worst_competing_sd) - float(best_sd_row["fold_sd"])
                    ) >= material_sd_gain:
                        chosen = best_sd_row
                        reason = f"oof_tied_stability_gain_ge_{material_sd_gain:g}"
                    else:
                        chosen = candidates.sort_values(
                            ["transform_complexity", "fold_sd", "nonmissing_fraction", "oof_metric"],
                            ascending=[True, True, False, False],
                            na_position="last",
                        ).iloc[0]
                        reason = "oof_practically_tied_prefer_simpler_transform"
                else:
                    chosen = candidates.sort_values(
                        ["transform_complexity", "nonmissing_fraction", "oof_metric"],
                        ascending=[True, False, False],
                        na_position="last",
                    ).iloc[0]
                    reason = "oof_practically_tied_prefer_simpler_transform"
        else:
            chosen = top

        chosen = chosen.copy()
        chosen["selected_transform_mode"] = chosen["transform_mode"]
        chosen["transform_selection_reason"] = reason
        chosen["n_valid_transforms"] = int(valid.shape[0])
        chosen["n_transforms_seen"] = int(group["transform_mode"].nunique(dropna=True))
        if valid.shape[0] > 1:
            sorted_oof = valid["oof_metric"].dropna().sort_values(ascending=False)
            chosen["oof_gap_best_vs_second"] = (
                float(sorted_oof.iloc[0] - sorted_oof.iloc[1]) if len(sorted_oof) > 1 else np.nan
            )
        else:
            chosen["oof_gap_best_vs_second"] = np.nan
        selected_rows.append(chosen)

        selected_index = chosen.name
        for row_index, row in group.iterrows():
            is_selected = bool(row_index == selected_index)
            audit_rows.append({
                **{col: row.get(col) for col in CONTEXT_COLS + FEATURE_ID_COLS},
                "transform_mode": row.get("transform_mode"),
                "selected_transform_mode": chosen.get("transform_mode"),
                "selected": is_selected,
                "selection_reason": reason if is_selected else "not_selected",
                "eligibility_pass": row.get("eligibility_pass"),
                "eligibility_reasons": row.get("eligibility_reasons"),
                "oof_metric": row.get("oof_metric"),
                "delta_clinical": row.get("delta_clinical"),
                "fold_sd": row.get("fold_sd"),
                "direction_consistency": row.get("direction_consistency"),
                "nonmissing_fraction": row.get("nonmissing_fraction"),
                "p_value": row.get("p_value"),
                "source_file": row.get("source_file"),
                "duplicate_rows_same_transform": row.get("duplicate_rows_same_transform", 1),
            })

    selected = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)

    if selected.empty:
        return selected, audit

    # Correct the selected P-value for having considered multiple valid transforms,
    # then apply BH across underlying variables within the context. These values are
    # annotations for manual review; they do not enter the default evidence score.
    selected["transform_adjusted_p"] = (
        selected["p_value"] * selected["n_valid_transforms"].clip(lower=1)
    ).clip(upper=1.0)
    selected["context_q_value"] = bh_adjust(selected["transform_adjusted_p"])
    selected["feature_uid"] = make_feature_uid(selected)
    return selected, audit


def add_candidate_evidence_score(selected: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    out = selected.copy()
    if out.empty:
        return out

    out["rank_oof"] = percentile_rank(out["oof_metric"], higher_better=True)
    out["rank_delta_clinical"] = percentile_rank(out["delta_clinical"], higher_better=True)
    out["rank_fold_sd"] = percentile_rank(out["fold_sd"], higher_better=False)
    out["rank_direction_consistency"] = percentile_rank(
        out["direction_consistency"], higher_better=True
    )
    out["rank_completeness"] = percentile_rank(out["nonmissing_fraction"], higher_better=True)

    stability_parts = pd.concat(
        [out["rank_fold_sd"], out["rank_direction_consistency"]], axis=1
    )
    out["rank_stability"] = stability_parts.mean(axis=1, skipna=True)

    weights = cfg.get("score_weights", {})
    components = {
        "rank_oof": float(weights.get("oof", 0.55)),
        "rank_stability": float(weights.get("stability", 0.20)),
        "rank_delta_clinical": float(weights.get("delta_clinical", 0.15)),
        "rank_completeness": float(weights.get("completeness", 0.10)),
    }

    numerator = pd.Series(0.0, index=out.index)
    denominator = pd.Series(0.0, index=out.index)
    for column, weight in components.items():
        valid = out[column].notna()
        numerator.loc[valid] += weight * out.loc[valid, column]
        denominator.loc[valid] += weight

    out["candidate_evidence_score"] = numerator / denominator.replace(0, np.nan)
    out["evidence_score_weight_available"] = denominator
    out = out.sort_values(
        ["candidate_evidence_score", "oof_metric", "fold_sd", "nonmissing_fraction"],
        ascending=[False, False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    out["candidate_review_rank"] = np.arange(1, len(out) + 1)

    stable_direction_cutoff = float(cfg.get("stable_direction_cutoff", 0.80))
    stable_fold_sd_cutoff = float(cfg.get("stable_fold_sd_cutoff", 0.10))
    out["stable_direction_flag"] = out["direction_consistency"] >= stable_direction_cutoff
    out["low_fold_sd_flag"] = out["fold_sd"] <= stable_fold_sd_cutoff
    out["stable_flag"] = (
        out["stable_direction_flag"].fillna(False) | out["low_fold_sd_flag"].fillna(False)
    )

    # Descriptive tiers only; they do not filter features or contexts.
    out["review_evidence_tier"] = "D_low"
    out.loc[out["oof_metric"] >= 0.55, "review_evidence_tier"] = "C_exploratory"
    out.loc[
        (out["oof_metric"] >= 0.58)
        & out["stable_flag"]
        & ((out["p_value"] <= 0.05) | (out["delta_clinical"] > 0)),
        "review_evidence_tier",
    ] = "B_moderate"
    out.loc[
        (out["oof_metric"] >= 0.60)
        & out["stable_flag"]
        & (out["context_q_value"] <= 0.20),
        "review_evidence_tier",
    ] = "A_strong"
    return out

# =============================================================================
# Context summaries, tables, plots, and HTML reports
# =============================================================================


def top_median(series: pd.Series, n_top: int) -> float:
    values = safe_numeric(series).dropna().sort_values(ascending=False).head(n_top)
    return float(values.median()) if not values.empty else np.nan


def context_metric_label(selected: pd.DataFrame) -> str:
    if selected.empty or "endpoint" not in selected.columns:
        return "OOF AUC or C-index"
    endpoint = str(selected["endpoint"].iloc[0])
    return "OOF C-index" if endpoint in {"OS", "RFS"} else "OOF AUC"


def save_figure(fig, path: Union[str, Path], dpi: int = 250) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_neglog10(series: pd.Series) -> pd.Series:
    x = safe_numeric(series)
    return -np.log10(x.clip(lower=1e-300))


def build_context_quality_summary(
    raw_context: pd.DataFrame,
    selected: pd.DataFrame,
    array_id: int,
    cfg: Mapping,
) -> pd.DataFrame:
    first = raw_context.iloc[0]
    summary: Dict[str, object] = {col: first.get(col) for col in CONTEXT_COLS}
    summary["array_id"] = int(array_id)
    summary["context_id"] = first.get("context_id")
    summary["n_result_rows_all_transforms"] = int(raw_context.shape[0])
    summary["n_unique_underlying_variables"] = int(
        raw_context[FEATURE_ID_COLS].drop_duplicates().shape[0]
    )
    summary["n_eligible_result_rows"] = int(raw_context["eligibility_pass"].sum())
    summary["n_best_transform_variables"] = int(selected.shape[0])

    for col in ["n", "n_events", "n_positive", "n_negative"]:
        values = safe_numeric(raw_context[col]).dropna()
        summary[col] = float(values.median()) if not values.empty else np.nan

    if selected.empty:
        return pd.DataFrame([summary])

    transform_counts = selected["selected_transform_mode"].value_counts(dropna=False)
    summary["n_zscore_selected"] = int(transform_counts.get("zscore", 0))
    summary["n_log1p_zscore_selected"] = int(transform_counts.get("log1p_zscore", 0))
    summary["fraction_log1p_selected"] = float(
        summary["n_log1p_zscore_selected"] / max(int(selected.shape[0]), 1)
    )

    summary["max_oof"] = float(selected["oof_metric"].max())
    summary["median_oof_all"] = float(selected["oof_metric"].median())
    summary["median_top5_oof"] = top_median(selected["oof_metric"], 5)
    summary["median_top10_oof"] = top_median(selected["oof_metric"], 10)
    summary["median_top20_oof"] = top_median(selected["oof_metric"], 20)
    summary["max_delta_clinical"] = float(selected["delta_clinical"].max()) if selected["delta_clinical"].notna().any() else np.nan
    summary["median_delta_clinical"] = float(selected["delta_clinical"].median()) if selected["delta_clinical"].notna().any() else np.nan
    summary["n_positive_delta_clinical"] = int((selected["delta_clinical"] > 0).sum())

    for threshold in cfg.get("oof_review_thresholds", [0.55, 0.60, 0.65]):
        label = "{:03d}".format(int(round(float(threshold) * 100)))
        summary[f"n_oof_ge_{label}"] = int((selected["oof_metric"] >= float(threshold)).sum())

    summary["n_nominal_p_le_005"] = int((selected["p_value"] <= 0.05).sum())
    summary["n_nominal_p_le_001"] = int((selected["p_value"] <= 0.01).sum())
    summary["n_context_q_le_020"] = int((selected["context_q_value"] <= 0.20).sum())
    summary["n_context_q_le_010"] = int((selected["context_q_value"] <= 0.10).sum())
    summary["n_stable"] = int(selected["stable_flag"].sum())
    summary["n_tier_A"] = int((selected["review_evidence_tier"] == "A_strong").sum())
    summary["n_tier_B"] = int((selected["review_evidence_tier"] == "B_moderate").sum())
    summary["n_tier_C"] = int((selected["review_evidence_tier"] == "C_exploratory").sum())
    summary["best_candidate_evidence_score"] = float(selected["candidate_evidence_score"].max())
    summary["median_top10_candidate_evidence_score"] = top_median(
        selected["candidate_evidence_score"], 10
    )
    return pd.DataFrame([summary])


def metric_distribution_summary(selected: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "oof_metric": "OOF discrimination",
        "delta_clinical": "Clinical-model delta",
        "fold_sd": "Fold-level SD",
        "direction_consistency": "Direction consistency",
        "nonmissing_fraction": "Nonmissing fraction",
        "p_value": "Nominal p-value",
        "transform_adjusted_p": "Transform-adjusted p-value",
        "context_q_value": "Context q-value",
        "candidate_evidence_score": "Candidate evidence score",
    }
    rows: List[dict] = []
    n_total = int(selected.shape[0])
    for column, label in metrics.items():
        if column not in selected.columns:
            continue
        x = safe_numeric(selected[column]).dropna()
        rows.append({
            "metric": column,
            "metric_label": label,
            "n_total_variables": n_total,
            "n_available": int(x.shape[0]),
            "missing_fraction": float(1 - x.shape[0] / max(n_total, 1)),
            "min": float(x.min()) if not x.empty else np.nan,
            "p05": float(x.quantile(0.05)) if not x.empty else np.nan,
            "p10": float(x.quantile(0.10)) if not x.empty else np.nan,
            "p25": float(x.quantile(0.25)) if not x.empty else np.nan,
            "median": float(x.median()) if not x.empty else np.nan,
            "mean": float(x.mean()) if not x.empty else np.nan,
            "p75": float(x.quantile(0.75)) if not x.empty else np.nan,
            "p90": float(x.quantile(0.90)) if not x.empty else np.nan,
            "p95": float(x.quantile(0.95)) if not x.empty else np.nan,
            "max": float(x.max()) if not x.empty else np.nan,
        })
    return pd.DataFrame(rows)


def threshold_count_summary(selected: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    checks: List[Tuple[str, str, float, str]] = []
    for threshold in cfg.get("oof_review_thresholds", [0.55, 0.60, 0.65]):
        checks.append(("oof_metric", ">=", float(threshold), f"OOF >= {threshold:g}"))
    for threshold in cfg.get("delta_review_thresholds", [0.0, 0.02, 0.05]):
        checks.append(("delta_clinical", ">=", float(threshold), f"Clinical delta >= {threshold:g}"))
    for threshold in cfg.get("fold_sd_review_thresholds", [0.05, 0.10, 0.15]):
        checks.append(("fold_sd", "<=", float(threshold), f"Fold SD <= {threshold:g}"))
    for threshold in cfg.get("direction_review_thresholds", [0.60, 0.80, 1.00]):
        checks.append(("direction_consistency", ">=", float(threshold), f"Direction consistency >= {threshold:g}"))
    for threshold in cfg.get("p_review_thresholds", [0.05, 0.01]):
        checks.append(("p_value", "<=", float(threshold), f"Nominal p <= {threshold:g}"))
    for threshold in cfg.get("q_review_thresholds", [0.20, 0.10, 0.05]):
        checks.append(("context_q_value", "<=", float(threshold), f"Context q <= {threshold:g}"))

    rows: List[dict] = []
    n_total = int(selected.shape[0])
    for column, operator, threshold, label in checks:
        if column not in selected.columns:
            continue
        x = safe_numeric(selected[column])
        available = x.notna()
        if operator == ">=":
            passed = available & (x >= threshold)
        else:
            passed = available & (x <= threshold)
        rows.append({
            "metric": column,
            "rule": label,
            "operator": operator,
            "threshold": threshold,
            "n_total_variables": n_total,
            "n_available": int(available.sum()),
            "n_passing": int(passed.sum()),
            "fraction_of_available_passing": float(passed.sum() / max(int(available.sum()), 1)),
            "fraction_of_all_passing": float(passed.sum() / max(n_total, 1)),
        })
    return pd.DataFrame(rows)


def category_summary(selected: pd.DataFrame, category: str) -> pd.DataFrame:
    if selected.empty or category not in selected.columns:
        return pd.DataFrame()
    rows: List[dict] = []
    for value, g in selected.groupby(category, dropna=False):
        oof = safe_numeric(g["oof_metric"])
        delta = safe_numeric(g["delta_clinical"])
        fold_sd = safe_numeric(g["fold_sd"])
        direction = safe_numeric(g["direction_consistency"])
        score = safe_numeric(g["candidate_evidence_score"])
        rows.append({
            category: value,
            "n_variables": int(g["feature_uid"].nunique()) if "feature_uid" in g.columns else int(g.shape[0]),
            "fraction_of_context": float(g.shape[0] / max(selected.shape[0], 1)),
            "max_oof": float(oof.max()) if oof.notna().any() else np.nan,
            "median_oof": float(oof.median()) if oof.notna().any() else np.nan,
            "median_delta_clinical": float(delta.median()) if delta.notna().any() else np.nan,
            "fraction_positive_delta": float((delta > 0).sum() / max(int(delta.notna().sum()), 1)),
            "median_fold_sd": float(fold_sd.median()) if fold_sd.notna().any() else np.nan,
            "median_direction_consistency": float(direction.median()) if direction.notna().any() else np.nan,
            "median_evidence_score": float(score.median()) if score.notna().any() else np.nan,
            "n_oof_ge_055": int((oof >= 0.55).sum()),
            "n_oof_ge_060": int((oof >= 0.60).sum()),
            "n_nominal_p_le_005": int((safe_numeric(g["p_value"]) <= 0.05).sum()),
            "n_q_le_020": int((safe_numeric(g["context_q_value"]) <= 0.20).sum()),
            "n_stable": int(g["stable_flag"].fillna(False).sum()) if "stable_flag" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values(["n_variables", "max_oof"], ascending=[False, False])


def top_candidates_by_category(selected: pd.DataFrame, category: str, n_per_category: int) -> pd.DataFrame:
    if selected.empty or category not in selected.columns:
        return pd.DataFrame()
    parts = []
    for _, g in selected.groupby(category, dropna=False):
        parts.append(g.sort_values("candidate_review_rank").head(n_per_category))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def candidate_metric_correlations(selected: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "oof_metric",
        "delta_clinical",
        "fold_sd",
        "direction_consistency",
        "nonmissing_fraction",
        "p_value",
        "context_q_value",
        "candidate_evidence_score",
    ]
    columns = [c for c in columns if c in selected.columns]
    numeric = selected[columns].apply(safe_numeric)
    correlations = numeric.corr(method="spearman")
    pairwise_n = pd.DataFrame(index=columns, columns=columns, dtype=int)
    for c1 in columns:
        for c2 in columns:
            pairwise_n.loc[c1, c2] = int(numeric[[c1, c2]].dropna().shape[0])
    return correlations, pairwise_n


def cumulative_topn_composition(selected: pd.DataFrame, category: str, top_n_values: Sequence[int]) -> pd.DataFrame:
    if selected.empty or category not in selected.columns:
        return pd.DataFrame()
    rows: List[dict] = []
    ordered = selected.sort_values("candidate_review_rank")
    for n_top in sorted(set(int(x) for x in top_n_values if int(x) > 0)):
        sub = ordered.head(min(n_top, len(ordered)))
        counts = sub[category].fillna("<missing>").astype(str).value_counts()
        for value, count in counts.items():
            rows.append({
                "top_n_requested": int(n_top),
                "top_n_available": int(sub.shape[0]),
                category: value,
                "n_variables": int(count),
                "fraction": float(count / max(sub.shape[0], 1)),
            })
    return pd.DataFrame(rows)


def transform_pair_comparison(transform_audit: pd.DataFrame) -> pd.DataFrame:
    if transform_audit.empty:
        return pd.DataFrame()
    id_cols = CONTEXT_COLS + FEATURE_ID_COLS
    metrics = ["oof_metric", "delta_clinical", "fold_sd", "direction_consistency", "nonmissing_fraction", "p_value"]
    use = transform_audit[id_cols + ["transform_mode", "selected_transform_mode"] + [m for m in metrics if m in transform_audit.columns]].copy()
    use = use.drop_duplicates(id_cols + ["transform_mode"], keep="first")
    wide_parts = []
    for metric in metrics:
        if metric not in use.columns:
            continue
        wide = use.pivot_table(index=id_cols, columns="transform_mode", values=metric, aggfunc="first")
        wide.columns = [f"{metric}__{str(c)}" for c in wide.columns]
        wide_parts.append(wide)
    if not wide_parts:
        return pd.DataFrame()
    out = pd.concat(wide_parts, axis=1).reset_index()
    selected_map = use.groupby(id_cols, dropna=False)["selected_transform_mode"].first().reset_index()
    out = out.merge(selected_map, on=id_cols, how="left")
    zcol = "oof_metric__zscore"
    lcol = "oof_metric__log1p_zscore"
    if zcol in out.columns and lcol in out.columns:
        out["log1p_minus_zscore_oof"] = out[lcol] - out[zcol]
        out["absolute_oof_difference"] = out["log1p_minus_zscore_oof"].abs()
    return out


def pareto_front_candidates(selected: pd.DataFrame) -> pd.DataFrame:
    """Descriptive non-dominated set using OOF high, delta high, and fold SD low."""
    required = ["oof_metric", "delta_clinical", "fold_sd"]
    if selected.empty or any(c not in selected.columns for c in required):
        return pd.DataFrame()
    work = selected.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()
    values = work[required].to_numpy(dtype=float)
    keep = np.ones(len(work), dtype=bool)
    for i in range(len(work)):
        if not keep[i]:
            continue
        oof_i, delta_i, sd_i = values[i]
        dominated = (
            (values[:, 0] >= oof_i)
            & (values[:, 1] >= delta_i)
            & (values[:, 2] <= sd_i)
            & (
                (values[:, 0] > oof_i)
                | (values[:, 1] > delta_i)
                | (values[:, 2] < sd_i)
            )
        )
        if dominated.any():
            keep[i] = False
    out = work.loc[keep].sort_values(
        ["candidate_evidence_score", "oof_metric"], ascending=[False, False]
    )
    return out.reset_index(drop=True)


def build_context_review_tables(
    selected: pd.DataFrame,
    transform_audit: pd.DataFrame,
    tables_dir: Path,
    cfg: Mapping,
) -> Dict[str, Path]:
    ensure_dir(tables_dir)
    outputs: Dict[str, Path] = {}

    def save_csv(name: str, df: pd.DataFrame) -> None:
        path = tables_dir / name
        df.to_csv(path, index=False)
        outputs[name] = path

    save_csv("01_metric_distribution_summary.csv", metric_distribution_summary(selected))
    save_csv("02_threshold_count_summary.csv", threshold_count_summary(selected, cfg))
    save_csv("03_feature_group_summary.csv", category_summary(selected, "feature_group"))
    save_csv("04_prep_root_summary.csv", category_summary(selected, "feature_source"))

    top_n = int(cfg.get("plot_top_n", 100))
    save_csv("05_top_candidates_overall.csv", selected.sort_values("candidate_review_rank").head(top_n))
    save_csv(
        "06_top_candidates_by_feature_group.csv",
        top_candidates_by_category(selected, "feature_group", int(cfg.get("top_n_per_category", 25))),
    )
    save_csv(
        "07_top_candidates_by_prep_root.csv",
        top_candidates_by_category(selected, "feature_source", int(cfg.get("top_n_per_category", 25))),
    )

    corr, pairwise_n = candidate_metric_correlations(selected)
    corr.to_csv(tables_dir / "08_candidate_metric_spearman_correlations.csv")
    pairwise_n.to_csv(tables_dir / "09_candidate_metric_pairwise_n.csv")
    outputs["08_candidate_metric_spearman_correlations.csv"] = tables_dir / "08_candidate_metric_spearman_correlations.csv"
    outputs["09_candidate_metric_pairwise_n.csv"] = tables_dir / "09_candidate_metric_pairwise_n.csv"

    top_n_values = cfg.get("composition_top_n_values", [10, 20, 30, 50, 75, 100])
    save_csv(
        "10_topn_composition_by_feature_group.csv",
        cumulative_topn_composition(selected, "feature_group", top_n_values),
    )
    save_csv(
        "11_topn_composition_by_prep_root.csv",
        cumulative_topn_composition(selected, "feature_source", top_n_values),
    )
    save_csv("12_transform_pair_comparison.csv", transform_pair_comparison(transform_audit))
    save_csv("13_pareto_front_candidates.csv", pareto_front_candidates(selected))

    availability_rows = []
    for col in [
        "oof_metric", "delta_clinical", "fold_sd", "direction_consistency",
        "nonmissing_fraction", "p_value", "context_q_value", "candidate_evidence_score"
    ]:
        if col in selected.columns:
            availability_rows.append({
                "metric": col,
                "n_available": int(selected[col].notna().sum()),
                "n_missing": int(selected[col].isna().sum()),
                "fraction_available": float(selected[col].notna().mean()) if len(selected) else np.nan,
            })
    save_csv("14_metric_availability.csv", pd.DataFrame(availability_rows))
    save_csv(
        "15_evidence_tier_counts.csv",
        selected["review_evidence_tier"].value_counts(dropna=False).rename_axis("review_evidence_tier").reset_index(name="n_variables")
        if "review_evidence_tier" in selected.columns else pd.DataFrame(),
    )
    return outputs


def category_color_map(values: pd.Series) -> Dict[str, tuple]:
    categories = sorted(values.fillna("<missing>").astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20")
    return {category: cmap(i % cmap.N) for i, category in enumerate(categories)}


def annotate_top_points(ax, df: pd.DataFrame, x: str, y: str, n: int) -> None:
    if n <= 0 or df.empty:
        return
    label_df = df.sort_values("candidate_review_rank").head(n)
    for _, row in label_df.iterrows():
        if pd.isna(row.get(x)) or pd.isna(row.get(y)):
            continue
        label = str(row.get("feature", ""))
        if len(label) > 45:
            label = label[:42] + "..."
        ax.annotate(
            label,
            (float(row[x]), float(row[y])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            alpha=0.85,
        )


def plot_histogram(
    series: pd.Series,
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 30,
    reference_lines: Optional[Sequence[float]] = None,
) -> None:
    x = safe_numeric(series).dropna()
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(x, bins=min(bins, max(10, int(np.sqrt(len(x)) * 2))))
    for value in reference_lines or []:
        ax.axvline(float(value), linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of variables")
    ax.set_title(title)
    save_figure(fig, path)


def plot_delta_waterfall(selected: pd.DataFrame, plots_dir: Path, title_prefix: str, top_n: int) -> None:
    plot_df = selected.sort_values("candidate_review_rank").head(top_n).dropna(subset=["delta_clinical"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("delta_clinical").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [plt.get_cmap("coolwarm")(0.2 if x < 0 else 0.8) for x in plot_df["delta_clinical"]]
    ax.bar(np.arange(len(plot_df)), plot_df["delta_clinical"], color=colors, width=0.9)
    ax.axhline(0, linewidth=1, color="black")
    ax.set_xlabel(f"Top {len(plot_df)} candidates, sorted by clinical delta")
    ax.set_ylabel("Combined minus clinical OOF metric")
    ax.set_title(f"{title_prefix}: clinical-delta waterfall")
    ax.set_xticks([])
    save_figure(fig, plots_dir / "03_clinical_delta_waterfall_top_candidates.png")

    labeled = plot_df.sort_values("delta_clinical", ascending=False).head(min(30, len(plot_df))).sort_values("delta_clinical")
    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.26 * len(labeled))))
    colors = [plt.get_cmap("coolwarm")(0.2 if x < 0 else 0.8) for x in labeled["delta_clinical"]]
    ax.barh(np.arange(len(labeled)), labeled["delta_clinical"], color=colors)
    labels = [str(x)[:70] for x in labeled["feature"]]
    ax.set_yticks(np.arange(len(labeled)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Combined minus clinical OOF metric")
    ax.set_title(f"{title_prefix}: strongest positive clinical deltas")
    save_figure(fig, plots_dir / "04_clinical_delta_top30_labeled.png")


def plot_categorical_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    annotate_n: int = 0,
    hline_zero: bool = False,
    vlines: Optional[Sequence[float]] = None,
) -> None:
    needed = [x, y, category]
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty or any(c not in plot_df.columns for c in needed):
        return
    colors = category_color_map(plot_df[category])
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for value, group in plot_df.groupby(category, dropna=False):
        label = str(value)
        ax.scatter(
            group[x], group[y], s=38, alpha=0.72,
            label=label, color=colors.get(label), edgecolors="none"
        )
    if hline_zero:
        ax.axhline(0, color="black", linewidth=1)
    for value in vlines or []:
        ax.axvline(float(value), linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=category, fontsize=7, title_fontsize=8, loc="best")
    annotate_top_points(ax, plot_df, x, y, annotate_n)
    save_figure(fig, path)


def plot_rank_profile(selected: pd.DataFrame, plots_dir: Path, title_prefix: str, top_n: int) -> None:
    plot_df = selected.sort_values("candidate_review_rank").head(top_n).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_df["candidate_review_rank"], plot_df["oof_metric"], marker="o", markersize=2.5, linewidth=1, label="OOF metric")
    ax.plot(plot_df["candidate_review_rank"], plot_df["candidate_evidence_score"], linewidth=1.2, label="Evidence score")
    ax.set_xlabel("Candidate review rank")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{title_prefix}: top-candidate rank profile")
    ax.legend()
    save_figure(fig, plots_dir / "11_rank_profile_oof_and_evidence.png")

    delta_df = plot_df.dropna(subset=["delta_clinical"])
    if not delta_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(delta_df["candidate_review_rank"], delta_df["delta_clinical"], marker="o", markersize=2.5, linewidth=1)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Candidate review rank")
        ax.set_ylabel("Combined minus clinical OOF metric")
        ax.set_title(f"{title_prefix}: clinical delta across candidate rank")
        save_figure(fig, plots_dir / "12_rank_profile_clinical_delta.png")


def plot_cumulative_composition(
    composition: pd.DataFrame,
    category: str,
    title: str,
    path: Path,
) -> None:
    if composition.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = category_color_map(composition[category])
    for value, group in composition.groupby(category, dropna=False):
        group = group.sort_values("top_n_requested")
        label = str(value)
        ax.plot(group["top_n_requested"], group["fraction"], marker="o", label=label, color=colors.get(label))
    ax.set_xlabel("Top N candidates")
    ax.set_ylabel("Fraction of top-N candidates")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(title=category, fontsize=7, title_fontsize=8, loc="best")
    save_figure(fig, path)


def plot_box_by_category(selected: pd.DataFrame, category: str, y: str, ylabel: str, title: str, path: Path) -> None:
    if selected.empty or category not in selected.columns or y not in selected.columns:
        return
    groups = []
    labels = []
    for value, g in selected.groupby(category, dropna=False):
        x = safe_numeric(g[y]).dropna()
        if not x.empty:
            groups.append(x.to_numpy())
            labels.append(str(value))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(groups)), 5.5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, path)


def plot_transform_comparison(pair_df: pd.DataFrame, plots_dir: Path, title_prefix: str) -> None:
    if pair_df.empty:
        return
    zcol = "oof_metric__zscore"
    lcol = "oof_metric__log1p_zscore"
    if zcol in pair_df.columns and lcol in pair_df.columns:
        plot_df = pair_df.dropna(subset=[zcol, lcol]).copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(6.3, 6.0))
            ax.scatter(plot_df[zcol], plot_df[lcol], alpha=0.65, s=30)
            lo = float(min(plot_df[zcol].min(), plot_df[lcol].min()))
            hi = float(max(plot_df[zcol].max(), plot_df[lcol].max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")
            ax.set_xlabel("Z-score OOF metric")
            ax.set_ylabel("Log1p-z-score OOF metric")
            ax.set_title(f"{title_prefix}: transform performance comparison")
            save_figure(fig, plots_dir / "17_zscore_vs_log1p_oof_scatter.png")

            diff_df = plot_df.sort_values("log1p_minus_zscore_oof").reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            colors = [plt.get_cmap("coolwarm")(0.2 if x < 0 else 0.8) for x in diff_df["log1p_minus_zscore_oof"]]
            ax.bar(np.arange(len(diff_df)), diff_df["log1p_minus_zscore_oof"], color=colors, width=0.9)
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(0.01, linestyle="--", linewidth=1)
            ax.axhline(-0.01, linestyle="--", linewidth=1)
            ax.set_xlabel("Variables with both transforms, sorted by OOF difference")
            ax.set_ylabel("Log1p minus z-score OOF metric")
            ax.set_title(f"{title_prefix}: transform advantage waterfall")
            ax.set_xticks([])
            save_figure(fig, plots_dir / "18_transform_oof_difference_waterfall.png")


def plot_metric_correlation_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title(title)
    save_figure(fig, path)


def save_context_plots(
    selected: pd.DataFrame,
    transform_audit: pd.DataFrame,
    plots_dir: Path,
    title_prefix: str,
    cfg: Mapping,
) -> List[Path]:
    if selected.empty:
        return []
    ensure_dir(plots_dir)
    metric_label = context_metric_label(selected)
    top_n = int(cfg.get("plot_top_n", 100))
    annotate_n = int(cfg.get("scatter_annotate_top_n", 8))
    top_df = selected.sort_values("candidate_review_rank").head(top_n).copy()
    outputs: List[Path] = []

    def track(path: Path) -> None:
        if path.exists():
            outputs.append(path)

    plot_histogram(
        selected["oof_metric"], metric_label,
        f"{title_prefix}: best-transform {metric_label} distribution",
        plots_dir / "01_oof_distribution.png",
        bins=int(cfg.get("hist_bins", 30)), reference_lines=cfg.get("oof_review_thresholds", [0.55, 0.60, 0.65])
    )
    track(plots_dir / "01_oof_distribution.png")

    plot_histogram(
        selected["delta_clinical"], "Combined minus clinical OOF metric",
        f"{title_prefix}: clinical-model delta distribution",
        plots_dir / "02_clinical_delta_distribution.png",
        bins=int(cfg.get("hist_bins", 30)), reference_lines=[0]
    )
    track(plots_dir / "02_clinical_delta_distribution.png")
    plot_delta_waterfall(selected, plots_dir, title_prefix, top_n)
    track(plots_dir / "03_clinical_delta_waterfall_top_candidates.png")
    track(plots_dir / "04_clinical_delta_top30_labeled.png")

    plot_histogram(
        selected["fold_sd"], "Fold-level SD",
        f"{title_prefix}: fold-level stability distribution",
        plots_dir / "05_fold_sd_distribution.png",
        bins=int(cfg.get("hist_bins", 30)), reference_lines=cfg.get("fold_sd_review_thresholds", [0.05, 0.10, 0.15])
    )
    track(plots_dir / "05_fold_sd_distribution.png")
    plot_histogram(
        selected["direction_consistency"], "Fraction of folds with consistent effect direction",
        f"{title_prefix}: direction-consistency distribution",
        plots_dir / "06_direction_consistency_distribution.png",
        bins=int(cfg.get("hist_bins", 30)), reference_lines=cfg.get("direction_review_thresholds", [0.60, 0.80, 1.00])
    )
    track(plots_dir / "06_direction_consistency_distribution.png")

    if selected["p_value"].notna().any():
        plot_histogram(
            safe_neglog10(selected["p_value"]), "−log10(nominal p-value)",
            f"{title_prefix}: nominal statistical support",
            plots_dir / "07_nominal_pvalue_distribution.png",
            bins=int(cfg.get("hist_bins", 30)), reference_lines=[-math.log10(0.05), -math.log10(0.01)]
        )
        track(plots_dir / "07_nominal_pvalue_distribution.png")
    if selected["context_q_value"].notna().any():
        plot_histogram(
            safe_neglog10(selected["context_q_value"]), "−log10(context q-value)",
            f"{title_prefix}: multiple-testing-adjusted support",
            plots_dir / "08_context_qvalue_distribution.png",
            bins=int(cfg.get("hist_bins", 30)), reference_lines=[-math.log10(0.20), -math.log10(0.10), -math.log10(0.05)]
        )
        track(plots_dir / "08_context_qvalue_distribution.png")

    plot_histogram(
        selected["candidate_evidence_score"], "Candidate evidence score",
        f"{title_prefix}: candidate evidence-score distribution",
        plots_dir / "09_candidate_evidence_score_distribution.png",
        bins=int(cfg.get("hist_bins", 30))
    )
    track(plots_dir / "09_candidate_evidence_score_distribution.png")

    plot_categorical_scatter(
        top_df, "oof_metric", "delta_clinical", "feature_group",
        metric_label, "Combined minus clinical OOF metric",
        f"{title_prefix}: top candidates by feature family",
        plots_dir / "10a_oof_vs_delta_colored_by_feature_group.png",
        annotate_n=annotate_n, hline_zero=True, vlines=[0.55, 0.60]
    )
    track(plots_dir / "10a_oof_vs_delta_colored_by_feature_group.png")
    plot_categorical_scatter(
        top_df, "oof_metric", "delta_clinical", "feature_source",
        metric_label, "Combined minus clinical OOF metric",
        f"{title_prefix}: top candidates by prep root",
        plots_dir / "10b_oof_vs_delta_colored_by_prep_root.png",
        annotate_n=annotate_n, hline_zero=True, vlines=[0.55, 0.60]
    )
    track(plots_dir / "10b_oof_vs_delta_colored_by_prep_root.png")

    plot_categorical_scatter(
        top_df, "oof_metric", "fold_sd", "feature_group",
        metric_label, "Fold-level SD",
        f"{title_prefix}: performance versus fold variability",
        plots_dir / "10c_oof_vs_fold_sd_colored_by_feature_group.png",
        annotate_n=annotate_n, vlines=[0.55, 0.60]
    )
    track(plots_dir / "10c_oof_vs_fold_sd_colored_by_feature_group.png")

    q_df = top_df.copy()
    q_df["neglog10_q"] = safe_neglog10(q_df["context_q_value"])
    plot_categorical_scatter(
        q_df, "oof_metric", "neglog10_q", "feature_group",
        metric_label, "−log10(context q-value)",
        f"{title_prefix}: performance versus adjusted statistical support",
        plots_dir / "10d_oof_vs_qvalue_colored_by_feature_group.png",
        annotate_n=annotate_n, vlines=[0.55, 0.60]
    )
    track(plots_dir / "10d_oof_vs_qvalue_colored_by_feature_group.png")

    plot_rank_profile(selected, plots_dir, title_prefix, top_n)
    track(plots_dir / "11_rank_profile_oof_and_evidence.png")
    track(plots_dir / "12_rank_profile_clinical_delta.png")

    top_n_values = cfg.get("composition_top_n_values", [10, 20, 30, 50, 75, 100])
    group_comp = cumulative_topn_composition(selected, "feature_group", top_n_values)
    source_comp = cumulative_topn_composition(selected, "feature_source", top_n_values)
    plot_cumulative_composition(
        group_comp, "feature_group",
        f"{title_prefix}: feature-family representation across candidate depth",
        plots_dir / "13_cumulative_topn_feature_group_composition.png",
    )
    track(plots_dir / "13_cumulative_topn_feature_group_composition.png")
    plot_cumulative_composition(
        source_comp, "feature_source",
        f"{title_prefix}: prep-root representation across candidate depth",
        plots_dir / "14_cumulative_topn_prep_root_composition.png",
    )
    track(plots_dir / "14_cumulative_topn_prep_root_composition.png")

    plot_box_by_category(
        selected, "feature_group", "oof_metric", metric_label,
        f"{title_prefix}: {metric_label} by feature family",
        plots_dir / "15_oof_boxplot_by_feature_group.png",
    )
    track(plots_dir / "15_oof_boxplot_by_feature_group.png")
    plot_box_by_category(
        selected, "feature_source", "oof_metric", metric_label,
        f"{title_prefix}: {metric_label} by prep root",
        plots_dir / "16_oof_boxplot_by_prep_root.png",
    )
    track(plots_dir / "16_oof_boxplot_by_prep_root.png")

    pair_df = transform_pair_comparison(transform_audit)
    plot_transform_comparison(pair_df, plots_dir, title_prefix)
    track(plots_dir / "17_zscore_vs_log1p_oof_scatter.png")
    track(plots_dir / "18_transform_oof_difference_waterfall.png")

    corr, _ = candidate_metric_correlations(selected)
    plot_metric_correlation_heatmap(
        corr,
        plots_dir / "19_candidate_metric_correlation_heatmap.png",
        f"{title_prefix}: correlations among candidate-review metrics",
    )
    track(plots_dir / "19_candidate_metric_correlation_heatmap.png")

    tier_counts = selected["review_evidence_tier"].value_counts().sort_index()
    if not tier_counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(tier_counts.index.astype(str), tier_counts.values)
        ax.set_xlabel("Descriptive evidence tier")
        ax.set_ylabel("Number of variables")
        ax.set_title(f"{title_prefix}: evidence-tier counts")
        ax.tick_params(axis="x", rotation=25)
        save_figure(fig, plots_dir / "20_evidence_tier_counts.png")
        track(plots_dir / "20_evidence_tier_counts.png")

    transform_counts = selected["selected_transform_mode"].value_counts().sort_index()
    if not transform_counts.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(transform_counts.index.astype(str), transform_counts.values)
        ax.set_xlabel("Selected transform")
        ax.set_ylabel("Number of variables")
        ax.set_title(f"{title_prefix}: selected-transform counts")
        save_figure(fig, plots_dir / "21_transform_selection_counts.png")
        track(plots_dir / "21_transform_selection_counts.png")

    return outputs


def write_context_report(
    context_output: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    plot_paths: Sequence[Path],
    table_paths: Mapping[str, Path],
    title_prefix: str,
) -> Path:
    report_path = context_output / "context_review_report.html"
    summary_html = summary.T.reset_index()
    summary_html.columns = ["Metric", "Value"]
    summary_table = summary_html.to_html(index=False, border=0, classes="summary-table", escape=True)

    top_cols = [
        "candidate_review_rank", "feature_source", "feature_group", "feature",
        "selected_transform_mode", "oof_metric", "delta_clinical", "fold_sd",
        "direction_consistency", "p_value", "context_q_value", "candidate_evidence_score"
    ]
    top_cols = [c for c in top_cols if c in selected.columns]
    top_table = selected.sort_values("candidate_review_rank").head(25)[top_cols].to_html(
        index=False, border=0, classes="candidate-table", escape=True, float_format=lambda x: f"{x:.4g}"
    )

    plot_cards = []
    for path in plot_paths:
        relative = path.relative_to(context_output)
        plot_cards.append(
            f'<figure><img src="{html.escape(str(relative))}" alt="{html.escape(path.stem)}">'
            f'<figcaption>{html.escape(path.stem.replace("_", " "))}</figcaption></figure>'
        )
    table_links = "\n".join(
        f'<li><a href="{html.escape(str(path.relative_to(context_output)))}">{html.escape(name)}</a></li>'
        for name, path in sorted(table_paths.items())
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title_prefix)} context review</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 1.4em; }}
.summary-table, .candidate-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.summary-table td, .summary-table th, .candidate-table td, .candidate-table th {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
figure {{ margin: 0; border: 1px solid #ddd; padding: 10px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ margin-top: 6px; font-size: 12px; color: #555; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>{html.escape(title_prefix)}: Stage 2A context review</h1>
<p>This report summarizes the best-transform variables before biological microcompression or final candidate nomination.</p>
<h2>Context summary</h2>
{summary_table}
<h2>Top 25 candidates</h2>
{top_table}
<h2>CSV review tables</h2>
<ul>{table_links}</ul>
<h2>Figures</h2>
<div class="gallery">{''.join(plot_cards)}</div>
</body>
</html>
"""
    report_path.write_text(document)
    return report_path


def context_label_from_summary(row: pd.Series) -> str:
    return f"{row.get('cohort')} | {row.get('panel')} | {row.get('endpoint')}"


def plot_cross_context_summaries(all_summary: pd.DataFrame, output_root: Path) -> List[Path]:
    if all_summary.empty:
        return []
    plots_dir = ensure_dir(output_root / "context_comparison_plots")
    work = all_summary.copy()
    work["context_label"] = work.apply(context_label_from_summary, axis=1)
    outputs: List[Path] = []

    def horizontal_metric_plot(column: str, title: str, xlabel: str, filename: str) -> None:
        if column not in work.columns or work[column].notna().sum() == 0:
            return
        d = work.sort_values(column, ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(d))))
        ax.barh(np.arange(len(d)), d[column])
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["context_label"], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        path = plots_dir / filename
        save_figure(fig, path)
        outputs.append(path)

    horizontal_metric_plot(
        "median_top10_oof", "Median OOF performance among each context's top 10 variables",
        "Median top-10 OOF metric", "01_median_top10_oof_by_context.png"
    )
    horizontal_metric_plot(
        "max_oof", "Maximum best-transform OOF performance by context",
        "Maximum OOF metric", "02_max_oof_by_context.png"
    )
    horizontal_metric_plot(
        "n_oof_ge_060", "Number of variables with OOF performance at least 0.60",
        "Number of variables", "03_n_oof_ge_060_by_context.png"
    )
    horizontal_metric_plot(
        "n_stable", "Number of descriptively stable variables by context",
        "Number of variables", "04_n_stable_by_context.png"
    )
    horizontal_metric_plot(
        "fraction_log1p_selected", "Fraction of variables selecting log1p-z-score by context",
        "Fraction selecting log1p-z-score", "05_fraction_log1p_by_context.png"
    )
    horizontal_metric_plot(
        "n_context_q_le_020", "Number of variables with context q-value at most 0.20",
        "Number of variables", "06_n_q_le_020_by_context.png"
    )
    return outputs


def write_aggregate_review_index(output_root: Path, context_index: pd.DataFrame, all_summary: pd.DataFrame) -> Path:
    index_path = output_root / "context_review_index.html"
    summary_lookup = all_summary.set_index("array_id") if "array_id" in all_summary.columns else pd.DataFrame()
    rows = []
    for _, row in context_index.iterrows():
        array_id = int(row["array_id"])
        report_rel = Path("contexts") / str(row["context_slug"]) / "context_review_report.html"
        metrics = {}
        if not summary_lookup.empty and array_id in summary_lookup.index:
            summary_row = summary_lookup.loc[array_id]
            metrics = {
                "median_top10_oof": summary_row.get("median_top10_oof", np.nan),
                "max_oof": summary_row.get("max_oof", np.nan),
                "n_oof_ge_060": summary_row.get("n_oof_ge_060", np.nan),
                "n_stable": summary_row.get("n_stable", np.nan),
            }
        rows.append({
            "array_id": array_id,
            "context": f"{row['cohort']} | {row['panel']} | {row['endpoint']}",
            "report": str(report_rel),
            **metrics,
        })
    def fmt_float(value: object) -> str:
        return "" if pd.isna(value) else "{:.3f}".format(float(value))

    def fmt_int(value: object) -> str:
        return "" if pd.isna(value) else str(int(value))

    table_rows = "".join(
        "<tr>"
        + "<td>{}</td>".format(r["array_id"])
        + "<td><a href=\"{}\">{}</a></td>".format(
            html.escape(r["report"]), html.escape(r["context"])
        )
        + "<td>{}</td>".format(fmt_float(r.get("median_top10_oof")))
        + "<td>{}</td>".format(fmt_float(r.get("max_oof")))
        + "<td>{}</td>".format(fmt_int(r.get("n_oof_ge_060")))
        + "<td>{}</td>".format(fmt_int(r.get("n_stable")))
        + "</tr>"
        for r in rows
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 2A context review index</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:7px;text-align:left}}th{{background:#f4f4f4}}</style>
</head><body>
<h1>Stage 2A context review index</h1>
<p>Open each context report for the full candidate tables and figure gallery.</p>
<table><thead><tr><th>Array ID</th><th>Context</th><th>Median top-10 OOF</th><th>Maximum OOF</th><th>N OOF ≥0.60</th><th>N stable</th></tr></thead>
<tbody>{table_rows}</tbody></table>
</body></html>"""
    index_path.write_text(document)
    return index_path


# =============================================================================
# Commands
# =============================================================================


def command_inventory(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    write_json(cfg, output_root / "stage2a_steps1_3_config.resolved.json")

    log("=" * 80)
    log("[STEP 1] Collecting and standardizing Stage 1 results")
    master, schema_audit = collect_inputs(cfg)
    schema_audit.to_csv(output_root / "input_schema_audit.csv", index=False)

    if master.empty:
        raise RuntimeError("No usable Stage 1 feature-level result tables were found.")

    master = apply_project_filters(master, cfg)
    if master.empty:
        raise RuntimeError("No rows remained after project filters. Inspect config and schema audit.")

    master = add_eligibility_flags(master, cfg)
    master = master.sort_values(CONTEXT_COLS + FEATURE_ID_COLS + ["transform_mode"]).reset_index(drop=True)

    master_path = save_table(master, output_root / "stage2a_master_all_transforms.parquet")
    (output_root / "stage2a_master_path.txt").write_text(str(master_path) + "\n")

    context_inputs_root = ensure_dir(output_root / "context_inputs")
    context_rows: List[dict] = []
    grouped = master.groupby(CONTEXT_COLS, dropna=False, sort=True)
    for array_id, (_, context_df) in enumerate(grouped):
        first = context_df.iloc[0]
        slug = safe_context_slug(first, array_id)
        context_path = save_table(context_df, context_inputs_root / f"{slug}.parquet")
        context_rows.append({
            "array_id": int(array_id),
            **{col: first.get(col) for col in CONTEXT_COLS},
            "context_id": first.get("context_id"),
            "context_slug": slug,
            "context_input_path": str(context_path),
            "n_result_rows_all_transforms": int(context_df.shape[0]),
            "n_unique_underlying_variables": int(context_df[FEATURE_ID_COLS].drop_duplicates().shape[0]),
            "n_eligible_result_rows": int(context_df["eligibility_pass"].sum()),
            "n_transforms": int(context_df["transform_mode"].nunique()),
        })

    context_index = pd.DataFrame(context_rows)
    context_index.to_csv(output_root / "stage2a_context_index.csv", index=False)

    inventory_summary = (
        master.groupby(["panel", "cohort", "endpoint", "transform_mode"], dropna=False)
        .agg(
            n_rows=("feature_uid", "size"),
            n_unique_features=("feature_uid", "nunique"),
            n_eligible_rows=("eligibility_pass", "sum"),
        )
        .reset_index()
    )
    inventory_summary.to_csv(output_root / "stage2a_inventory_summary.csv", index=False)

    master.head(5000).to_csv(output_root / "standardized_rows_preview.csv", index=False)
    log(f"[DONE] Inventory created with {len(context_index)} contexts")
    log(f"[SAVE] {output_root / 'stage2a_context_index.csv'}")


def resolve_array_id(args_array_id: Optional[int]) -> int:
    if args_array_id is not None:
        return int(args_array_id)
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_value is None:
        raise ValueError("Provide --array-id or run inside a Slurm array task.")
    return int(env_value)


def command_worker(cfg: Mapping, array_id: int) -> None:
    output_root = Path(cfg["output_root"])
    context_index_path = output_root / "stage2a_context_index.csv"
    if not context_index_path.exists():
        raise FileNotFoundError(f"Run inventory first: {context_index_path}")

    context_index = pd.read_csv(context_index_path)
    row_match = context_index[context_index["array_id"].astype(int) == int(array_id)]
    if row_match.empty:
        raise IndexError(f"array_id={array_id} not present in context index")
    index_row = row_match.iloc[0]

    context_input_path = Path(index_row["context_input_path"])
    context_df = read_table(context_input_path)
    if context_df.empty:
        raise RuntimeError(f"Empty context input: {context_input_path}")

    context_output = ensure_dir(output_root / "contexts" / str(index_row["context_slug"]))
    log("=" * 80)
    log(f"[CONTEXT {array_id}] {index_row['context_id']}")
    log(f"[INFO] rows={context_df.shape[0]}")

    selected, transform_audit = choose_best_transform_for_context(context_df, cfg)
    selected = add_candidate_evidence_score(selected, cfg)

    save_table(selected, context_output / "best_transform_features.parquet")
    transform_audit.to_csv(context_output / "transform_selection_audit.csv", index=False)

    summary = build_context_quality_summary(context_df, selected, array_id=array_id, cfg=cfg)
    summary.to_csv(context_output / "context_quality_summary.csv", index=False)

    top_n = int(cfg.get("top_n_for_manual_review", 200))
    review_columns = [
        *CONTEXT_COLS,
        "candidate_review_rank",
        "feature_source",
        "feature_group",
        "feature",
        "selected_transform_mode",
        "transform_selection_reason",
        "n_valid_transforms",
        "oof_gap_best_vs_second",
        "oof_metric",
        "clinical_oof_metric",
        "combined_oof_metric",
        "delta_clinical",
        "fold_sd",
        "direction_consistency",
        "nonmissing_fraction",
        "effect",
        "p_value",
        "transform_adjusted_p",
        "context_q_value",
        "rank_oof",
        "rank_stability",
        "rank_delta_clinical",
        "rank_completeness",
        "candidate_evidence_score",
        "evidence_score_weight_available",
        "review_evidence_tier",
        "stable_direction_flag",
        "low_fold_sd_flag",
        "stable_flag",
        "n",
        "n_events",
        "n_positive",
        "n_negative",
        "valid_folds",
        "source_file",
    ]
    review_columns = [column for column in review_columns if column in selected.columns]
    selected.head(top_n)[review_columns].to_csv(
        context_output / "top_features_for_manual_review.csv", index=False
    )

    title_prefix = f"{index_row['cohort']} {index_row['panel']} {index_row['endpoint']}"
    tables_dir = ensure_dir(context_output / "tables")
    plots_dir = ensure_dir(context_output / "plots")
    table_paths = build_context_review_tables(selected, transform_audit, tables_dir, cfg)
    plot_paths = save_context_plots(
        selected,
        transform_audit,
        plots_dir,
        title_prefix=title_prefix,
        cfg=cfg,
    )
    report_path = write_context_report(
        context_output=context_output,
        summary=summary,
        selected=selected,
        plot_paths=plot_paths,
        table_paths=table_paths,
        title_prefix=title_prefix,
    )

    write_json(
        {
            "array_id": int(array_id),
            "context_id": index_row["context_id"],
            "context_input_path": str(context_input_path),
            "n_best_transform_variables": int(selected.shape[0]),
            "n_review_plots": int(len(plot_paths)),
            "n_review_tables": int(len(table_paths)),
            "context_report": str(report_path),
        },
        context_output / "worker_complete.json",
    )
    log(f"[DONE] {context_output}")


def command_aggregate(cfg: Mapping) -> None:
    output_root = Path(cfg["output_root"])
    context_index = pd.read_csv(output_root / "stage2a_context_index.csv")

    selected_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []
    audit_parts: List[pd.DataFrame] = []
    top_parts: List[pd.DataFrame] = []
    missing_contexts: List[dict] = []

    for _, row in context_index.iterrows():
        context_output = output_root / "contexts" / str(row["context_slug"])
        summary_path = context_output / "context_quality_summary.csv"
        selected_path = context_output / "best_transform_features.parquet"
        audit_path = context_output / "transform_selection_audit.csv"
        top_path = context_output / "top_features_for_manual_review.csv"

        if not summary_path.exists():
            missing_contexts.append(row.to_dict())
            continue

        summary_parts.append(pd.read_csv(summary_path))
        selected_parts.append(load_saved_table(selected_path))
        if audit_path.exists():
            audit_parts.append(pd.read_csv(audit_path))
        if top_path.exists():
            top_parts.append(pd.read_csv(top_path))

    if missing_contexts:
        pd.DataFrame(missing_contexts).to_csv(output_root / "missing_context_workers.csv", index=False)
        raise RuntimeError(
            f"{len(missing_contexts)} context workers are missing. See missing_context_workers.csv"
        )

    all_selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    all_summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    all_audit = pd.concat(audit_parts, ignore_index=True, sort=False) if audit_parts else pd.DataFrame()
    all_top = pd.concat(top_parts, ignore_index=True, sort=False) if top_parts else pd.DataFrame()

    save_table(all_selected, output_root / "all_context_best_transform_features.parquet")
    if not all_audit.empty:
        all_audit.to_csv(output_root / "all_transform_selection_audit.csv.gz", index=False, compression="gzip")
    if not all_top.empty:
        all_top.to_csv(output_root / "all_context_top_features_for_manual_review.csv.gz", index=False, compression="gzip")

    all_summary = all_summary.sort_values(["panel", "cohort", "endpoint"]).reset_index(drop=True)
    all_summary.to_csv(output_root / "context_quality_review.csv", index=False)

    manual_template = all_summary.copy()
    manual_template["manual_include_context"] = ""
    manual_template["manual_context_strength"] = ""
    manual_template["manual_candidate_limit"] = ""
    manual_template["manual_notes"] = ""
    manual_template.to_csv(output_root / "context_manual_review_template.csv", index=False)

    transform_summary = (
        all_selected.groupby(CONTEXT_COLS + ["selected_transform_mode"], dropna=False)
        .agg(n_variables=("feature_uid", "nunique"))
        .reset_index()
    ) if not all_selected.empty else pd.DataFrame()
    transform_summary.to_csv(output_root / "transform_selection_summary_by_context.csv", index=False)

    # Cross-context visual summaries and an HTML landing page for manual review.
    comparison_plots = plot_cross_context_summaries(all_summary, output_root)
    review_index = write_aggregate_review_index(output_root, context_index, all_summary)

    report_inventory_rows = []
    for _, row in context_index.iterrows():
        context_output = output_root / "contexts" / str(row["context_slug"])
        report_path = context_output / "context_review_report.html"
        plots_dir = context_output / "plots"
        tables_dir = context_output / "tables"
        report_inventory_rows.append({
            "array_id": int(row["array_id"]),
            **{col: row.get(col) for col in CONTEXT_COLS},
            "context_slug": row["context_slug"],
            "context_report": str(report_path),
            "report_exists": bool(report_path.exists()),
            "n_plot_files": len(list(plots_dir.glob("*.png"))) if plots_dir.exists() else 0,
            "n_table_files": len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0,
        })
    pd.DataFrame(report_inventory_rows).to_csv(output_root / "context_review_output_inventory.csv", index=False)

    log("=" * 80)
    log("[DONE] Stage 2A steps 1-3 complete")
    log(f"[REVIEW] {output_root / 'context_manual_review_template.csv'}")
    log(f"[REVIEW] {output_root / 'context_quality_review.csv'}")
    log(f"[REVIEW] {review_index}")
    log(f"[INFO] cross-context plots={len(comparison_plots)}")

# =============================================================================
# Configuration and CLI
# =============================================================================


def default_config() -> dict:
    return {
        "input_tables": [],
        "results_roots": [],
        "include_globs": [
            "**/*summary*.csv",
            "**/*summary*.tsv",
            "**/*results*.csv",
            "**/*results*.parquet",
        ],
        "column_map_json": None,
        "output_root": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/stage2a_steps1_3_context_review",
        "discovery_cohorts": DEFAULT_DISCOVERY_COHORTS,
        "panels": DEFAULT_PANELS,
        "endpoints": DEFAULT_ENDPOINTS,
        "sample_types": DEFAULT_SAMPLE_TYPES,
        "patient_subsets": ["all"],
        "aggs": ["median"],
        "feature_groups": DEFAULT_FEATURE_GROUPS,
        "transforms": ["zscore", "log1p_zscore"],
        "min_n": 20,
        "min_events": 5,
        "min_class_n": 5,
        "min_valid_folds": 4,
        "min_nonmissing_fraction": 0.50,
        "transform_preference_order": ["zscore", "log1p_zscore", "raw"],
        "transform_oof_material_gain": 0.01,
        "transform_fold_sd_material_gain": 0.02,
        "score_weights": {
            "oof": 0.55,
            "stability": 0.20,
            "delta_clinical": 0.15,
            "completeness": 0.10,
        },
        "p_value_use": "annotation_only",
        "stable_direction_cutoff": 0.80,
        "stable_fold_sd_cutoff": 0.10,
        "oof_review_thresholds": [0.55, 0.60, 0.65],
        "delta_review_thresholds": [0.00, 0.02, 0.05],
        "fold_sd_review_thresholds": [0.05, 0.10, 0.15],
        "direction_review_thresholds": [0.60, 0.80, 1.00],
        "p_review_thresholds": [0.05, 0.01],
        "q_review_thresholds": [0.20, 0.10, 0.05],
        "top_n_for_manual_review": 200,
        "plot_top_n": 100,
        "top_n_per_category": 25,
        "scatter_annotate_top_n": 8,
        "hist_bins": 30,
        "composition_top_n_values": [10, 20, 30, 50, 75, 100],
    }


def load_config(path: Optional[str]) -> dict:
    cfg = default_config()
    if path:
        user_cfg = read_json(path)
        cfg.update(user_cfg)
    # Normalize common list values.
    cfg["transforms"] = [normalize_transform(x) for x in cfg.get("transforms", [])]
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Collect results and create context shards.")
    inventory.add_argument("--config", required=True)

    worker = subparsers.add_parser("worker", help="Process one context.")
    worker.add_argument("--config", required=True)
    worker.add_argument("--array-id", type=int, default=None)

    aggregate = subparsers.add_parser("aggregate", help="Merge all context review outputs.")
    aggregate.add_argument("--config", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "inventory":
        command_inventory(cfg)
    elif args.command == "worker":
        command_worker(cfg, array_id=resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
