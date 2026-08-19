#!/usr/bin/env python3
"""
stage2a_rank_compress_candidates_v1.py

Automated post-processing of completed Stage 1 univariate screens into a
redundancy-compressed Stage 2 candidate manifest.

Primary use case
----------------
Use already completed z-score univariate results to:
  1. collect successful feature-level results across cohort x endpoint contexts;
  2. standardize heterogeneous summary-column names;
  3. rank features within each context using OOF performance, improvement over
     the fixed clinical model, fold stability, and completeness;
  4. preselect a generous set of high-ranking features;
  5. rebuild patient-level vectors only for those preselected features;
  6. group structurally comparable features into biological microfamilies;
  7. collapse highly positively correlated features within each microfamily;
  8. choose representatives automatically;
  9. apply soft feature-family caps; and
 10. write a Stage-2B-compatible global_module_candidate_manifest.csv plus a
     complete audit trail.

The script is intentionally conservative about parent/state biology:
base phenotypes, checkpoint-state abundance, state fractions, and state-specific
spatial features are assigned to separate state classes and are not automatically
collapsed across those classes.

Recommended primary analysis
----------------------------
Run on TURBT only. Freeze TURBT-defined modules first. Run RC separately as an
exploratory sensitivity analysis; do not mix TURBT and RC in the same discovery
consensus.

Important
---------
Stage 1 output schemas vary between versions. This script uses column aliases and
path inference. First run with --audit-only. Inspect:
    input_schema_audit.csv
    standardized_rows_preview.csv
    context_input_summary.csv
If required columns cannot be inferred, use --column-map-json to provide explicit
mappings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_DISCOVERY_COHORTS = ["NAC2020", "PURE01", "BLASST", "No-NAC"]
DEFAULT_PANELS = ["AR", "BT"]
DEFAULT_SAMPLE_TYPES = ["TURBT"]
DEFAULT_FEATURE_GROUPS = ["NN", "athena", "cell_features", "triads"]

# A generous preselection is used before patient-vector reconstruction.
DEFAULT_PRESELECT_N = {"AR": 300, "BT": 250}
DEFAULT_FINAL_N = {"AR": 60, "BT": 75}
DEFAULT_CORR_THRESHOLD = {"AR": 0.93, "BT": 0.97}

DEFAULT_FAMILY_CAPS = {
    "AR": {
        "triads": 0.35,
        "cell_features": 0.35,
        "athena": 0.40,
        "NN": 0.40,
    },
    "BT": {
        "triads": 0.45,
        "cell_features": 0.45,
        "athena": 0.50,
        "NN": 0.50,
    },
}

CONTEXT_COLS = [
    "cohort",
    "panel",
    "endpoint",
    "sample_type",
    "patient_subset",
    "agg",
]
MATRIX_CONTEXT_COLS = ["cohort", "panel", "sample_type", "patient_subset", "agg"]
FEATURE_ID_COLS = ["feature_source", "feature_group", "feature"]

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
    # Primary biomarker-only OOF performance aliases.
    "oof_metric": [
        "primary_oof_metric",
        "oof_metric",
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
    ],
    # Added value of clinical + biomarker versus fixed clinical model.
    "delta_clinical": [
        "primary_delta_metric",
        "delta_oof_auc_vs_clinical",
        "delta_oof_cindex_vs_clinical",
        "delta_auc_vs_clinical",
        "delta_cindex_vs_clinical",
        "combined_minus_clinical_oof",
        "delta_combined_vs_clinical",
    ],
    "fold_sd": [
        "primary_oof_sd",
        "biomarker_fold_auc_sd",
        "biomarker_fold_cindex_sd",
        "fold_auc_sd",
        "fold_cindex_sd",
        "oof_sd",
        "cv_sd",
    ],
    "p_value": ["p_value", "p", "biomarker_p", "full_p_value"],
}

KNOWN_COHORTS = ["NAC2020", "NAC2015", "PURE01", "BLASST", "No-NAC", "NoNAC", "KOLL"]
KNOWN_ENDPOINTS = ["complete_response", "any_response", "RFS", "OS"]
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
# Utilities
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_large_dataframe(df: pd.DataFrame, parquet_path: str | Path) -> Path:
    """Prefer parquet; fall back to compressed CSV when parquet engines are absent."""
    parquet_path = Path(parquet_path)
    ensure_dir(parquet_path.parent)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = parquet_path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        log(f"[WARN] parquet engine unavailable; saved compressed CSV: {csv_path}")
        return csv_path


def parse_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None or str(text).strip() == "":
        return None
    return [x.strip() for x in str(text).split(",") if x.strip()]


def import_module_from_path(name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_existing_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if str(a).lower() in lower_map:
            return lower_map[str(a).lower()]
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def percentile_rank(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = safe_numeric(s)
    return x.rank(pct=True, ascending=not higher_better, method="average")


def normalize_cohort(x: str) -> str:
    s = str(x).strip()
    if s.lower().replace("-", "") == "nonac":
        return "No-NAC"
    return s


def normalize_endpoint(x: str) -> str:
    s = str(x).strip()
    low = s.lower().replace("-", "_").replace(" ", "_")
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
    return mapping.get(low, s)


def normalize_panel(x: str) -> str:
    return str(x).strip().upper()


def normalize_feature_group(x: str) -> str:
    s = str(x).strip()
    key = s.lower().replace(" ", "_")
    return KNOWN_FEATURE_GROUPS.get(key, s)


def canonical_feature_name(x: str) -> str:
    """Conservative normalization used only for exact/provenance duplicate grouping."""
    s = str(x).strip().lower()
    s = s.replace("tumour", "tumor")
    s = s.replace("cd8+", "cd8").replace("cd4+", "cd4")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_(),.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def make_feature_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "|"
        + df["feature_group"].astype(str)
        + "|"
        + df["feature"].astype(str)
    )


def context_id_from_row(row: pd.Series) -> str:
    return "__".join(str(row.get(c, "NA")) for c in CONTEXT_COLS)


def matrix_context_id_from_row(row: pd.Series) -> str:
    return "__".join(str(row.get(c, "NA")) for c in MATRIX_CONTEXT_COLS)


# =============================================================================
# Loading and standardizing Stage 1 result tables
# =============================================================================


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def infer_from_path(path: str | Path) -> Dict[str, str]:
    text = str(path)
    low = text.lower()
    out: Dict[str, str] = {}

    for c in KNOWN_COHORTS:
        if c.lower() in low or c.lower().replace("-", "") in low.replace("-", ""):
            out["cohort"] = normalize_cohort(c)
            break

    for p in ["AR", "BT"]:
        if re.search(rf"(^|[/_\-.]){p.lower()}($|[/_\-.])", low):
            out["panel"] = p
            break

    endpoint_patterns = {
        "complete_response": ["complete_response", "complete-response"],
        "any_response": ["any_response", "any-response"],
        "RFS": ["/rfs/", "_rfs_", "endpoint=rfs"],
        "OS": ["/os/", "_os_", "endpoint=os"],
    }
    for endpoint, pats in endpoint_patterns.items():
        if any(p in low for p in pats):
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

    # Common feature-source labels from the expanded AR workflow.
    for fs in [
        "phenotype_only",
        "AR_checkpoint_state",
        "AR_state",
        "compartment_state",
        "compartment",
    ]:
        if fs.lower() in low:
            out["feature_source"] = fs
            break

    if "zscore" in low and "log1p" not in low:
        out["transform_mode"] = "zscore"
    elif "log1p_zscore" in low or "log1pzscore" in low:
        out["transform_mode"] = "log1p_zscore"
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
    audit = {"source_file": str(source_path), "n_rows": int(df.shape[0]), "accepted": False}

    # Metric fields may coexist in the same table (for example oof_auc for
    # response rows and oof_cindex for survival rows). For these targets, combine
    # all matching aliases row-wise rather than selecting only the first column.
    coalesce_targets = {"oof_metric", "delta_clinical", "fold_sd"}

    for target, aliases in COLUMN_ALIASES.items():
        source_col = explicit_map.get(target)
        if source_col is not None and source_col not in df.columns:
            raise ValueError(f"Explicit mapping {target} -> {source_col} not found in {source_path}")

        if source_col is not None:
            out[target] = df[source_col]
            audit[f"col_{target}"] = str(source_col)
            continue

        if target in coalesce_targets:
            lower_map = {str(c).lower(): c for c in df.columns}
            matched = []
            for alias in aliases:
                if alias in df.columns:
                    matched.append(alias)
                elif str(alias).lower() in lower_map:
                    matched.append(lower_map[str(alias).lower()])
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

    # A valid feature-level table must expose a feature and an OOF metric.
    if out["feature"].isna().all() or out["oof_metric"].isna().all():
        audit["reason"] = "missing_feature_or_oof_metric"
        return pd.DataFrame(), audit

    out["source_file"] = str(source_path)
    out["source_row"] = np.arange(len(out), dtype=int)

    # Fill stable defaults only after path inference.
    out["sample_type"] = out["sample_type"].fillna("TURBT")
    out["patient_subset"] = out["patient_subset"].fillna("all")
    out["agg"] = out["agg"].fillna("median")
    out["transform_mode"] = out["transform_mode"].fillna("zscore")
    out["feature_source"] = out["feature_source"].fillna("phenotype_only")

    # Normalize key fields.
    out["cohort"] = out["cohort"].map(normalize_cohort)
    out["panel"] = out["panel"].map(normalize_panel)
    out["endpoint"] = out["endpoint"].map(normalize_endpoint)
    out["feature_group"] = out["feature_group"].map(normalize_feature_group)
    out["feature"] = out["feature"].astype(str)
    out["feature_source"] = out["feature_source"].astype(str)

    for c in [
        "n",
        "n_events",
        "n_positive",
        "n_negative",
        "valid_folds",
        "nonmissing_fraction",
        "oof_metric",
        "delta_clinical",
        "fold_sd",
        "p_value",
    ]:
        out[c] = safe_numeric(out[c])

    out["feature_uid"] = make_feature_uid(out)
    out["canonical_feature"] = out["feature"].map(canonical_feature_name)
    out["context_id"] = out.apply(context_id_from_row, axis=1)
    out["matrix_context_id"] = out.apply(matrix_context_id_from_row, axis=1)

    audit["accepted"] = True
    audit["reason"] = "ok"
    return out, audit


def collect_input_tables(
    input_tables: Sequence[str],
    results_roots: Sequence[str],
    include_globs: Sequence[str],
    explicit_map: Optional[Mapping[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths: List[Path] = []
    for p in input_tables:
        paths.append(Path(p))
    for root_text in results_roots:
        root = Path(root_text)
        if not root.exists():
            log(f"[WARN] Missing results root: {root}")
            continue
        for pattern in include_globs:
            paths.extend(root.glob(pattern))

    # Deduplicate while preserving order.
    seen = set()
    unique_paths = []
    for p in paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique_paths.append(p)

    rows = []
    audits = []
    for i, path in enumerate(unique_paths, start=1):
        try:
            df = read_tabular(path)
            std, audit = standardize_one_table(df, path, explicit_map=explicit_map)
            audits.append(audit)
            if not std.empty:
                rows.append(std)
        except Exception as e:
            audits.append({
                "source_file": str(path),
                "accepted": False,
                "reason": f"{type(e).__name__}: {e}",
            })
        if i % 250 == 0:
            log(f"[INFO] scanned {i}/{len(unique_paths)} files")

    all_rows = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return all_rows, pd.DataFrame(audits)


# =============================================================================
# Eligibility, transform choice, and scoring
# =============================================================================


def status_is_ok(x: object) -> bool:
    if pd.isna(x) or str(x).strip() == "":
        return True
    s = str(x).strip().lower()
    good = {"ok", "success", "successful", "fit", "coxph", "statsmodels_logit", "completed"}
    bad_tokens = ["fail", "error", "skip", "insufficient", "empty", "constant"]
    if any(t in s for t in bad_tokens):
        return False
    return s in good or not any(t in s for t in bad_tokens)


def apply_input_filters(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    d = df.copy()
    d = d[d["cohort"].isin(cfg["discovery_cohorts"])].copy()
    d = d[d["panel"].isin(cfg["panels"])].copy()
    d = d[d["sample_type"].isin(cfg["sample_types"])].copy()
    d = d[d["patient_subset"].isin(cfg["patient_subsets"])].copy()
    d = d[d["agg"].isin(cfg["aggs"])].copy()
    d = d[d["feature_group"].isin(cfg["feature_groups"])].copy()
    d = d[d["transform_mode"].isin(cfg["transforms"])].copy()
    d = d[d["status"].map(status_is_ok)].copy()
    d = d[d["oof_metric"].notna()].copy()

    # Endpoint-specific sample-size filters are applied only when the relevant
    # columns are available; missing metadata is retained and flagged.
    is_response = d["endpoint"].isin(["complete_response", "any_response"])
    is_survival = d["endpoint"].isin(["OS", "RFS"])

    if d["n"].notna().any():
        d = d[(d["n"].isna()) | (d["n"] >= int(cfg["min_n"]))].copy()
    if d["valid_folds"].notna().any():
        d = d[(d["valid_folds"].isna()) | (d["valid_folds"] >= int(cfg["min_valid_folds"]))].copy()
    if d["nonmissing_fraction"].notna().any():
        d = d[(d["nonmissing_fraction"].isna()) | (d["nonmissing_fraction"] >= float(cfg["min_nonmissing_fraction"]))].copy()
    if d["n_events"].notna().any():
        d = d[(~is_survival) | d["n_events"].isna() | (d["n_events"] >= int(cfg["min_events"]))].copy()
    if d["n_positive"].notna().any():
        d = d[(~is_response) | d["n_positive"].isna() | (d["n_positive"] >= int(cfg["min_class_n"]))].copy()
    if d["n_negative"].notna().any():
        d = d[(~is_response) | d["n_negative"].isna() | (d["n_negative"] >= int(cfg["min_class_n"]))].copy()

    return d


def score_within_context(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    parts = []
    weights = cfg["score_weights"]

    for _, g0 in df.groupby(CONTEXT_COLS, dropna=False):
        g = g0.copy()
        g["rank_oof"] = percentile_rank(g["oof_metric"], higher_better=True)
        g["rank_delta_clinical"] = percentile_rank(g["delta_clinical"], higher_better=True)
        g["rank_stability"] = percentile_rank(g["fold_sd"], higher_better=False)
        g["rank_completeness"] = percentile_rank(g["nonmissing_fraction"], higher_better=True)

        components = {
            "rank_oof": float(weights.get("oof", 0.50)),
            "rank_delta_clinical": float(weights.get("delta_clinical", 0.25)),
            "rank_stability": float(weights.get("stability", 0.15)),
            "rank_completeness": float(weights.get("completeness", 0.10)),
        }
        numerator = pd.Series(0.0, index=g.index)
        denominator = pd.Series(0.0, index=g.index)
        for col, w in components.items():
            valid = g[col].notna()
            numerator.loc[valid] += w * g.loc[valid, col]
            denominator.loc[valid] += w
        g["candidate_score"] = numerator / denominator.replace(0, np.nan)
        g["score_components_available"] = denominator
        parts.append(g)

    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return out


def choose_transform_per_feature(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    """
    Choose one transform per underlying feature within each context.

    For z-score-only runs this is a no-op. With multiple transforms, a more complex
    transform must exceed the simpler transform by transform_min_gain unless it has
    meaningfully lower fold variability.
    """
    if df.empty:
        return df

    group_cols = CONTEXT_COLS + FEATURE_ID_COLS
    rows = []
    complexity = {"raw": 0, "zscore": 1, "log1p_zscore": 2}
    min_gain = float(cfg.get("transform_min_gain", 0.01))

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values(
            ["candidate_score", "oof_metric", "fold_sd", "nonmissing_fraction"],
            ascending=[False, False, True, False],
            na_position="last",
        ).copy()
        best = g.iloc[0]

        # Prefer a simpler transform when candidate scores are essentially tied.
        near = g[g["candidate_score"] >= float(best["candidate_score"]) - min_gain].copy()
        if not near.empty:
            near["transform_complexity"] = near["transform_mode"].map(complexity).fillna(99)
            near = near.sort_values(
                ["transform_complexity", "fold_sd", "nonmissing_fraction", "candidate_score"],
                ascending=[True, True, False, False],
                na_position="last",
            )
            best = near.iloc[0]
        rows.append(best)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["selected_transform_mode"] = out["transform_mode"]
    out["feature_uid"] = make_feature_uid(out)
    return out


def collapse_provenance_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse exact canonical feature definitions duplicated across prep roots."""
    keep_rows = []
    audit_rows = []
    group_cols = CONTEXT_COLS + ["feature_group", "canonical_feature", "selected_transform_mode"]

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values(
            ["candidate_score", "oof_metric", "fold_sd", "nonmissing_fraction"],
            ascending=[False, False, True, False],
            na_position="last",
        )
        rep = g.iloc[0].copy()
        rep["provenance_duplicate_group_size"] = int(g.shape[0])
        keep_rows.append(rep)
        for rank, (_, r) in enumerate(g.iterrows(), start=1):
            audit_rows.append({
                **{c: r.get(c) for c in CONTEXT_COLS},
                "canonical_feature": r.get("canonical_feature"),
                "feature_uid": r.get("feature_uid"),
                "representative_feature_uid": rep.get("feature_uid"),
                "is_representative": bool(rank == 1),
                "group_size": int(g.shape[0]),
                "reason": "canonical_definition_duplicate",
            })

    return pd.DataFrame(keep_rows).reset_index(drop=True), pd.DataFrame(audit_rows)


# =============================================================================
# Structural microfamilies and correlation compression
# =============================================================================


def classify_state_class(row: pd.Series) -> str:
    fs = str(row.get("feature_source", "")).lower()
    fg = str(row.get("feature_group", "")).lower()
    feat = str(row.get("feature", "")).lower()
    state_tokens = ["pd1", "pd-1", "pdl1", "pd-l1", "checkpoint", "foxp3", "ki67", "icos", "ctla4"]
    has_state = any(t in feat or t in fs for t in state_tokens)
    is_fraction = any(t in feat for t in ["fraction", "frac", "ratio", "over", "percent", "pct", "prop"])
    is_spatial = fg in {"nn", "athena", "triads"} or any(t in feat for t in ["_to_", "inter_", "ripley", "infiltration"])
    if has_state and is_fraction:
        return "state_composition"
    if has_state and is_spatial:
        return "state_spatial"
    if has_state:
        return "state_abundance"
    return "base"


def build_ontology_for_candidates(df: pd.DataFrame, gm) -> pd.DataFrame:
    ontology = gm.build_feature_ontology(df["feature_uid"].drop_duplicates().tolist(), feature_meta=df)
    ontology = ontology.drop_duplicates("feature_uid")
    return df.merge(
        ontology[[
            "feature_uid",
            "cell_set",
            "tissue_set",
            "metric_family_set",
            "lineage_set",
            "dominant_lineage",
            "n_cells_parsed",
        ]],
        on="feature_uid",
        how="left",
    )


def add_microfamily_keys(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["state_class"] = d.apply(classify_state_class, axis=1)
    for c in ["cell_set", "tissue_set", "metric_family_set", "lineage_set"]:
        d[c] = d[c].fillna("").astype(str)

    # Keep metric family and state class separate. This avoids merging total CD8
    # abundance with PD1+ CD8 composition or spatial behavior solely because they
    # are correlated.
    d["microfamily_key"] = (
        d["feature_group"].astype(str)
        + "||cells=" + d["cell_set"]
        + "||tissue=" + d["tissue_set"]
        + "||metric=" + d["metric_family_set"]
        + "||state=" + d["state_class"]
    )
    return d


class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_components_from_corr(corr: pd.DataFrame, threshold: float) -> Dict[str, int]:
    feats = corr.index.astype(str).tolist()
    uf = UnionFind(feats)
    arr = corr.values.astype(float)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            val = arr[i, j]
            if np.isfinite(val) and val >= threshold:
                uf.union(feats[i], feats[j])
    roots = {}
    root_to_id = {}
    next_id = 1
    for f in feats:
        root = uf.find(f)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        roots[f] = root_to_id[root]
    return roots


def representative_sort(g: pd.DataFrame) -> pd.DataFrame:
    return g.sort_values(
        ["candidate_score", "oof_metric", "fold_sd", "nonmissing_fraction", "feature"],
        ascending=[False, False, True, False, True],
        na_position="last",
    )


def compress_context_by_microfamily(
    context_df: pd.DataFrame,
    matrix: pd.DataFrame,
    corr_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reps = []
    audit = []
    if context_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for mf_index, (mf_key, g0) in enumerate(context_df.groupby("microfamily_key", dropna=False), start=1):
        g = g0.copy()
        features = [f for f in g["feature_uid"].astype(str).tolist() if f in matrix.columns]

        # Features missing from the reconstructed matrix become singleton groups;
        # they remain auditable but are not preferred unless no alternative exists.
        missing = [f for f in g["feature_uid"].astype(str).tolist() if f not in matrix.columns]

        group_map: Dict[str, int] = {}
        if len(features) >= 2:
            X = matrix[features].apply(pd.to_numeric, errors="coerce")
            corr = X.corr(method="spearman")
            group_map = connected_components_from_corr(corr, threshold=corr_threshold)
        elif len(features) == 1:
            group_map = {features[0]: 1}

        next_gid = max(group_map.values(), default=0) + 1
        for f in missing:
            group_map[f] = next_gid
            next_gid += 1

        g["local_redundancy_group"] = g["feature_uid"].map(group_map)
        for local_gid, rg0 in g.groupby("local_redundancy_group", dropna=False):
            rg = representative_sort(rg0)
            rep = rg.iloc[0].copy()
            global_gid = f"{rep['context_id']}::MF{mf_index:05d}::RG{int(local_gid):04d}"
            rep["redundancy_group_id"] = global_gid
            rep["redundancy_group_size"] = int(rg.shape[0])
            rep["representative_reason"] = "highest_candidate_score_within_correlated_microfamily"
            reps.append(rep)

            for rank, (_, r) in enumerate(rg.iterrows(), start=1):
                audit.append({
                    **{c: r.get(c) for c in CONTEXT_COLS},
                    "microfamily_key": mf_key,
                    "redundancy_group_id": global_gid,
                    "feature_uid": r.get("feature_uid"),
                    "feature": r.get("feature"),
                    "candidate_score": r.get("candidate_score"),
                    "oof_metric": r.get("oof_metric"),
                    "fold_sd": r.get("fold_sd"),
                    "nonmissing_fraction": r.get("nonmissing_fraction"),
                    "representative_feature_uid": rep.get("feature_uid"),
                    "is_representative": bool(rank == 1),
                    "rank_within_group": int(rank),
                    "group_size": int(rg.shape[0]),
                    "corr_threshold": float(corr_threshold),
                    "vector_available": bool(r.get("feature_uid") in matrix.columns),
                })

    return pd.DataFrame(reps).reset_index(drop=True), pd.DataFrame(audit)


# =============================================================================
# Soft family caps and context nomination
# =============================================================================


def apply_soft_family_caps(
    reps: pd.DataFrame,
    final_n: int,
    family_caps: Mapping[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top representatives while limiting early domination by one feature group.

    Caps are soft: after the capped pass, remaining slots are filled globally by
    score so a context is not forced to return fewer candidates when other groups
    lack eligible features.
    """
    d = representative_sort(reps).copy()
    if d.empty:
        return d, pd.DataFrame()

    selected_indices: List[int] = []
    group_counts: Dict[str, int] = {}
    cap_counts = {
        group: max(1, int(math.floor(float(frac) * final_n)))
        for group, frac in family_caps.items()
    }

    # Capped pass.
    for idx, row in d.iterrows():
        group = str(row["feature_group"])
        current = group_counts.get(group, 0)
        cap = cap_counts.get(group, final_n)
        if current < cap and len(selected_indices) < final_n:
            selected_indices.append(idx)
            group_counts[group] = current + 1

    # Global fill pass.
    if len(selected_indices) < final_n:
        for idx in d.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= final_n:
                break

    selected = d.loc[selected_indices].copy()
    selected = representative_sort(selected).head(final_n).copy()
    selected["context_candidate_rank"] = np.arange(1, len(selected) + 1)
    selected["selected_for_manifest"] = True

    audit = (
        selected.groupby("feature_group", dropna=False)
        .agg(n_selected=("feature_uid", "size"), mean_candidate_score=("candidate_score", "mean"))
        .reset_index()
    )
    audit["final_n_target"] = int(final_n)
    audit["soft_cap_count"] = audit["feature_group"].map(cap_counts)
    return selected, audit


# =============================================================================
# Matrix reconstruction using existing Stage 1/Stage 2 utilities
# =============================================================================


def build_preselected_matrices(preselected: pd.DataFrame, cfg: Mapping, gm) -> Dict[str, pd.DataFrame]:
    stage1_mod = import_module_from_path("stage1_runtime_for_candidate_compression", cfg["stage1_script_path"])
    matrix_cfg = {
        "output_root": str(Path(cfg["output_root"]) / "matrix_reconstruction"),
        "harmonized_path": cfg["harmonized_path"],
        "koll_metadata_csv": cfg.get("koll_metadata_csv"),
        "spatial_root": cfg.get("spatial_root"),
        "cell_features_path": cfg.get("cell_features_path"),
        "triads_path": cfg.get("triads_path"),
        "qc_acceptability": cfg.get("qc_acceptability", "acceptable_or_borderline"),
        "min_epi_fraction": cfg.get("min_epi_fraction", 0.05),
    }
    ensure_dir(matrix_cfg["output_root"])

    matrices: Dict[str, pd.DataFrame] = {}
    meta_parts = []
    context_rows = []

    for _, ctx in preselected.groupby(MATRIX_CONTEXT_COLS, dropna=False):
        first = ctx.iloc[0]
        mid = matrix_context_id_from_row(first)
        panel_dir = ensure_dir(Path(matrix_cfg["output_root"]) / str(first["panel"]))
        parquet_path = panel_dir / f"{mid}.parquet"
        meta_path = panel_dir / f"{mid}__feature_meta.csv"

        if parquet_path.exists() and meta_path.exists() and not bool(cfg.get("force_rebuild_matrices", False)):
            matrix = pd.read_parquet(parquet_path)
            meta = pd.read_csv(meta_path)
            log(f"[LOAD matrix] {mid} shape={matrix.shape}")
        else:
            log(f"[BUILD matrix] {mid} | features={ctx['feature_uid'].nunique()}")
            matrix, meta = gm.build_context_feature_uid_matrix(
                ctx_manifest=ctx,
                stage1_mod=stage1_mod,
                config=matrix_cfg,
            )
            matrix.to_parquet(parquet_path, index=False)
            meta.to_csv(meta_path, index=False)
            log(f"[SAVE matrix] {parquet_path} shape={matrix.shape}")

        matrices[mid] = matrix
        meta_parts.append(meta)
        context_rows.append({
            **{c: first[c] for c in MATRIX_CONTEXT_COLS},
            "matrix_context_id": mid,
            "n_patients": int(matrix.shape[0]),
            "n_feature_columns": int(max(matrix.shape[1] - 1, 0)),
            "path": str(parquet_path),
        })

    pd.DataFrame(context_rows).to_csv(Path(matrix_cfg["output_root"]) / "matrix_manifest.csv", index=False)
    if meta_parts:
        pd.concat(meta_parts, ignore_index=True, sort=False).to_csv(
            Path(matrix_cfg["output_root"]) / "context_feature_meta.csv", index=False
        )
    return matrices


# =============================================================================
# Output summaries
# =============================================================================


def make_context_summary(
    scored: pd.DataFrame,
    preselected: pd.DataFrame,
    compressed: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    keys = CONTEXT_COLS
    frames = []
    for name, df, value in [
        ("n_scored", scored, "feature_uid"),
        ("n_preselected", preselected, "feature_uid"),
        ("n_compressed_representatives", compressed, "feature_uid"),
        ("n_final_candidates", selected, "feature_uid"),
    ]:
        if df.empty:
            continue
        t = df.groupby(keys, dropna=False)[value].nunique().reset_index(name=name)
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for t in frames[1:]:
        out = out.merge(t, on=keys, how="outer")
    for c in ["n_scored", "n_preselected", "n_compressed_representatives", "n_final_candidates"]:
        if c not in out.columns:
            out[c] = 0
    out["compression_fraction_preselected_to_final"] = (
        out["n_final_candidates"] / out["n_preselected"].replace(0, np.nan)
    )
    return out.sort_values(keys)


def make_recurrence_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    return (
        manifest.groupby(
            ["panel", "feature_uid", "feature_source", "feature_group", "feature"],
            dropna=False,
        )
        .agg(
            n_contexts=("context_id", "nunique"),
            n_cohorts=("cohort", "nunique"),
            n_endpoints=("endpoint", "nunique"),
            cohorts=("cohort", lambda x: ";".join(sorted(set(map(str, x))))),
            endpoints=("endpoint", lambda x: ";".join(sorted(set(map(str, x))))),
            max_candidate_score=("candidate_score", "max"),
            median_candidate_score=("candidate_score", "median"),
            max_oof_metric=("oof_metric", "max"),
            median_oof_metric=("oof_metric", "median"),
        )
        .reset_index()
        .sort_values(
            ["panel", "n_cohorts", "n_contexts", "median_candidate_score"],
            ascending=[True, False, False, False],
        )
    )


# =============================================================================
# Main run
# =============================================================================


def run(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    write_json(cfg, output_root / "run_config.resolved.json")

    explicit_map = read_json(cfg["column_map_json"]) if cfg.get("column_map_json") else None
    all_rows, schema_audit = collect_input_tables(
        input_tables=cfg.get("input_tables", []),
        results_roots=cfg.get("results_roots", []),
        include_globs=cfg.get("include_globs", ["**/*summary*.csv"]),
        explicit_map=explicit_map,
    )
    schema_audit.to_csv(output_root / "input_schema_audit.csv", index=False)

    if all_rows.empty:
        raise RuntimeError(
            "No compatible feature-level result tables were found. Inspect input_schema_audit.csv, "
            "then provide --input-table or --column-map-json."
        )

    all_rows.head(500).to_csv(output_root / "standardized_rows_preview.csv", index=False)
    log(f"[INFO] standardized rows={all_rows.shape[0]:,} from accepted files={schema_audit['accepted'].sum():,}")

    filtered = apply_input_filters(all_rows, cfg)
    if filtered.empty:
        raise RuntimeError("No rows remain after cohort/panel/sample/transform/eligibility filtering.")

    input_summary = (
        filtered.groupby(CONTEXT_COLS, dropna=False)
        .agg(
            n_rows=("feature_uid", "size"),
            n_features=("feature_uid", "nunique"),
            n_sources=("feature_source", "nunique"),
            n_groups=("feature_group", "nunique"),
        )
        .reset_index()
        .sort_values(CONTEXT_COLS)
    )
    input_summary.to_csv(output_root / "context_input_summary.csv", index=False)

    scored = score_within_context(filtered, cfg)
    scored = choose_transform_per_feature(scored, cfg)
    scored, provenance_audit = collapse_provenance_duplicates(scored)
    provenance_audit.to_csv(output_root / "provenance_duplicate_audit.csv", index=False)
    save_large_dataframe(scored, output_root / "all_scored_eligible_features.parquet")

    # Generous preselection before vector reconstruction.
    preselected_parts = []
    for context_key, g in scored.groupby(CONTEXT_COLS, dropna=False):
        panel = str(g.iloc[0]["panel"])
        n_top = int(cfg["preselect_n_by_panel"].get(panel, 250))
        g = representative_sort(g).head(n_top).copy()
        g["preselection_rank"] = np.arange(1, len(g) + 1)
        preselected_parts.append(g)
    preselected = pd.concat(preselected_parts, ignore_index=True, sort=False)

    if cfg.get("audit_only", False):
        # Audit mode deliberately stops before importing project-specific utilities
        # or reconstructing patient vectors.
        save_large_dataframe(preselected, output_root / "preselected_features.parquet")
        log(f"[AUDIT ONLY] Outputs written to {output_root}")
        return

    # Add semantic/microfamily metadata before matrix build so it is preserved.
    gm = import_module_from_path("stage2_global_module_utils_for_compression", cfg["stage2_utils_path"])
    preselected = build_ontology_for_candidates(preselected, gm)
    preselected = add_microfamily_keys(preselected)
    save_large_dataframe(preselected, output_root / "preselected_features.parquet")

    matrices = build_preselected_matrices(preselected, cfg, gm)

    compressed_parts = []
    redundancy_audits = []
    family_audits = []
    final_parts = []

    for _, ctx in preselected.groupby(CONTEXT_COLS, dropna=False):
        first = ctx.iloc[0]
        panel = str(first["panel"])
        mid = matrix_context_id_from_row(first)
        matrix = matrices.get(mid, pd.DataFrame())
        threshold = float(cfg["corr_threshold_by_panel"].get(panel, 0.95))

        reps, audit = compress_context_by_microfamily(ctx, matrix, threshold)
        if not reps.empty:
            compressed_parts.append(reps)
        if not audit.empty:
            redundancy_audits.append(audit)

        final_n = int(cfg["final_n_by_panel"].get(panel, 60))
        caps = cfg["family_caps_by_panel"].get(panel, {})
        selected, fam_audit = apply_soft_family_caps(reps, final_n=final_n, family_caps=caps)
        if not selected.empty:
            final_parts.append(selected)
        if not fam_audit.empty:
            for c in CONTEXT_COLS:
                fam_audit[c] = first[c]
            family_audits.append(fam_audit)

    compressed = pd.concat(compressed_parts, ignore_index=True, sort=False) if compressed_parts else pd.DataFrame()
    redundancy_audit = pd.concat(redundancy_audits, ignore_index=True, sort=False) if redundancy_audits else pd.DataFrame()
    final = pd.concat(final_parts, ignore_index=True, sort=False) if final_parts else pd.DataFrame()
    family_audit = pd.concat(family_audits, ignore_index=True, sort=False) if family_audits else pd.DataFrame()

    save_large_dataframe(compressed, output_root / "compressed_context_representatives.parquet")
    redundancy_audit.to_csv(output_root / "redundancy_group_audit.csv", index=False)
    family_audit.to_csv(output_root / "feature_family_selection_audit.csv", index=False)

    if final.empty:
        raise RuntimeError("No final candidates were selected after redundancy compression.")

    # Stage-2B-compatible names.
    final["selected_transform_mode"] = final["selected_transform_mode"].fillna(final["transform_mode"])
    final["primary_oof_metric"] = final["oof_metric"]
    final["primary_delta_metric"] = final["delta_clinical"]

    manifest_cols = [
        "cohort",
        "panel",
        "endpoint",
        "sample_type",
        "patient_subset",
        "agg",
        "feature_source",
        "feature_group",
        "feature",
        "feature_uid",
        "selected_transform_mode",
        "candidate_score",
        "primary_oof_metric",
        "primary_delta_metric",
        "fold_sd",
        "nonmissing_fraction",
        "n",
        "n_events",
        "n_positive",
        "n_negative",
        "valid_folds",
        "context_candidate_rank",
        "microfamily_key",
        "state_class",
        "redundancy_group_id",
        "redundancy_group_size",
        "representative_reason",
        "source_file",
        "source_row",
        "context_id",
    ]
    manifest_cols = [c for c in manifest_cols if c in final.columns]
    manifest = final[manifest_cols].copy()
    manifest.to_csv(output_root / "global_module_candidate_manifest.csv", index=False)

    context_summary = make_context_summary(scored, preselected, compressed, manifest)
    context_summary.to_csv(output_root / "context_selection_summary.csv", index=False)
    recurrence = make_recurrence_summary(manifest)
    recurrence.to_csv(output_root / "candidate_context_recurrence.csv", index=False)

    panel_summary = (
        manifest.groupby(["panel", "feature_group"], dropna=False)
        .agg(
            n_manifest_rows=("feature_uid", "size"),
            n_unique_features=("feature_uid", "nunique"),
            n_contexts=("context_id", "nunique"),
            n_cohorts=("cohort", "nunique"),
        )
        .reset_index()
    )
    panel_summary.to_csv(output_root / "final_candidate_composition.csv", index=False)

    log("=" * 80)
    log(f"[DONE] Candidate compression complete: {output_root}")
    log(f"[SAVE] {output_root / 'global_module_candidate_manifest.csv'}")
    log("Use this manifest as the candidate_manifest input to Stage 2B.")


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
        "output_root": "/projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v8/discovery_primary_zscore_compressed",
        "stage2_utils_path": "/projects/ovcare/users/nikolay_alabi/immuno/manuscript/modules/stage2_global_module_utils_v7.py",
        "stage1_script_path": "/projects/ovcare/users/nikolay_alabi/immuno/manuscript/univariate_screening/stage1_univariate_cv_screen_v6.py",
        "harmonized_path": "/projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv",
        "koll_metadata_csv": "/projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/KOLL_core_metadata.csv",
        "spatial_root": None,
        "cell_features_path": None,
        "triads_path": None,
        "discovery_cohorts": DEFAULT_DISCOVERY_COHORTS,
        "panels": DEFAULT_PANELS,
        "sample_types": DEFAULT_SAMPLE_TYPES,
        "patient_subsets": ["all"],
        "aggs": ["median"],
        "feature_groups": DEFAULT_FEATURE_GROUPS,
        "transforms": ["zscore"],
        "min_n": 20,
        "min_events": 5,
        "min_class_n": 5,
        "min_valid_folds": 4,
        "min_nonmissing_fraction": 0.50,
        "score_weights": {
            "oof": 0.50,
            "delta_clinical": 0.25,
            "stability": 0.15,
            "completeness": 0.10,
        },
        "transform_min_gain": 0.01,
        "preselect_n_by_panel": DEFAULT_PRESELECT_N,
        "final_n_by_panel": DEFAULT_FINAL_N,
        "corr_threshold_by_panel": DEFAULT_CORR_THRESHOLD,
        "family_caps_by_panel": DEFAULT_FAMILY_CAPS,
        "qc_acceptability": "acceptable_or_borderline",
        "min_epi_fraction": 0.05,
        "force_rebuild_matrices": False,
        "audit_only": False,
        "column_map_json": None,
    }


def load_config(args) -> dict:
    cfg = default_config()
    if args.config:
        cfg.update(read_json(args.config))

    if args.input_table:
        cfg["input_tables"] = args.input_table
    if args.results_root:
        cfg["results_roots"] = args.results_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.stage2_utils_path:
        cfg["stage2_utils_path"] = args.stage2_utils_path
    if args.stage1_script_path:
        cfg["stage1_script_path"] = args.stage1_script_path
    if args.harmonized_path:
        cfg["harmonized_path"] = args.harmonized_path
    if args.column_map_json:
        cfg["column_map_json"] = args.column_map_json
    if args.discovery_cohorts:
        cfg["discovery_cohorts"] = parse_list(args.discovery_cohorts)
    if args.panels:
        cfg["panels"] = parse_list(args.panels)
    if args.sample_types:
        cfg["sample_types"] = parse_list(args.sample_types)
    if args.transforms:
        cfg["transforms"] = parse_list(args.transforms)
    if args.audit_only:
        cfg["audit_only"] = True
    if args.force_rebuild_matrices:
        cfg["force_rebuild_matrices"] = True
    return cfg


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="JSON config; CLI values override it.")
    ap.add_argument("--input-table", action="append", default=[], help="Direct CSV/TSV/parquet input. Repeatable.")
    ap.add_argument("--results-root", action="append", default=[], help="Recursively scan a Stage 1 results root. Repeatable.")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--stage2-utils-path", default=None)
    ap.add_argument("--stage1-script-path", default=None)
    ap.add_argument("--harmonized-path", default=None)
    ap.add_argument("--column-map-json", default=None, help="Optional explicit standard-column -> source-column mapping.")
    ap.add_argument("--discovery-cohorts", default=None, help="Comma-separated.")
    ap.add_argument("--panels", default=None, help="Comma-separated AR,BT.")
    ap.add_argument("--sample-types", default=None, help="Primary recommendation: TURBT only.")
    ap.add_argument("--transforms", default=None, help="Comma-separated; currently use zscore.")
    ap.add_argument("--audit-only", action="store_true", help="Collect/score/preselect but do not reconstruct vectors or compress.")
    ap.add_argument("--force-rebuild-matrices", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    run(cfg)
