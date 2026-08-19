#!/usr/bin/env python3
"""
stage2_module_pipeline_v7.py

Stage 2A candidate-selection pipeline for global module discovery.

This v7 version explicitly lets multiple transformations of the same underlying feature
compete, then carries forward only the best representation per context.

Purpose
-------
This script reads Stage-1 univariate screening outputs and selects a balanced,
high-performing set of candidate biomarkers for each analysis context:

    cohort x panel x endpoint x sample_type x patient_subset x agg

Within each context, zscore/log1p_zscore representations can compete; only the best
transform representation is retained for a given feature_uid.

The outputs are intended to be consumed by a later global-module discovery script.

Key design choices for the expanded rerun
-----------------------------------------
1. Feature spaces / prep roots are pooled within a context, but never treated as
   independent validation evidence. The script retains feature_source and creates
   a unique feature_uid = feature_source|feature_group|feature.
2. Candidate selection is feature-source-aware and feature-group-aware so that
   expanded AR roots do not drown out phenotype-only biology.
3. Discovery defaults are conservative:
       transform = zscore/log1p_zscore can compete
       agg = median
       sample_type = TURBT
       patient_subset = all
       cohorts = NAC2020,PURE01,BLASST,No-NAC
4. NAC2015 and KOLL should usually be held out from module discovery and used for
   frozen-module evaluation, but they can be included by changing --cohorts.

Expected Stage-1 v6 layout
--------------------------
The script is robust to either context columns embedded in the CSVs or the v6
nested result layout:

<results_root>/<feature_source>/<panel>/<feature_group>/<cohort>/<endpoint>/<sample_type>/\
    agg-<agg>/patient_subset-<subset>/transform-<transform>/*__summary.csv

and corresponding *__fullmodels.csv and *__feature_filter.csv files.

Primary outputs
---------------
<output_root>/all_context_ranked_features.csv
<output_root>/all_context_stable_features.csv
<output_root>/all_context_candidates_balanced.csv
<output_root>/global_module_candidate_manifest.csv
<output_root>/<cohort>/<panel>/<endpoint>/<sample_type>/<patient_subset>/... per-context CSVs

Typical primary TURBT discovery run
-----------------------------------
python stage2_module_pipeline_v7.py \
  --stage1-root /projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/results \
  --output-root /projects/ovcare/users/nikolay_alabi/immuno/stage2_global_modules_v6/candidate_selection \
  --cohorts NAC2020,PURE01,BLASST,No-NAC \
  --sample-types TURBT \
  --patient-subsets all \
  --aggs median \
  --transform-modes zscore,log1p_zscore
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Defaults
# =============================================================================

FEATURE_SOURCES_ALL = [
    "phenotype_only",
    "AR_state",
    "AR_checkpoint_state",
    "compartment",
    "compartment_state",
]

# Recommended primary expanded atlas source map.
DEFAULT_PANEL_SOURCE_MAP = {
    "AR": ["phenotype_only", "AR_state", "AR_checkpoint_state", "compartment", "compartment_state"],
    "BT": ["phenotype_only", "compartment"],
}

FEATURE_GROUPS_ALL = ["NN", "athena", "cell_features", "triads"]
PANELS_DEFAULT = ["AR", "BT"]
ENDPOINTS_DEFAULT = ["complete_response", "any_response", "OS", "RFS"]
DISCOVERY_COHORTS_DEFAULT = ["NAC2020", "PURE01", "BLASST", "No-NAC"]
ALL_COHORTS_DEFAULT = ["NAC2020", "PURE01", "BLASST", "No-NAC", "NAC2015", "KOLL"]

RESPONSE_ENDPOINTS = {"complete_response", "any_response"}
SURVIVAL_ENDPOINTS = {"OS", "RFS"}

# Same basic prioritization as the previous pipeline, but applied to feature_uid
# so multiple prep roots can coexist without feature-name collisions.
DEFAULT_RANK_WEIGHTS = {
    "delta": 0.40,
    "oof": 0.30,
    "pval": 0.15,
    "effect": 0.15,
}

# Stability filter; set --no-cv-std-filter to disable.
CV_PLATEAU_FRAC_DEFAULT = 0.95
CV_STD_GRID_N_DEFAULT = 50
CV_TOP_N_DEFAULT = 30

# Candidate selection context: transform is intentionally excluded because
# zscore/log1p_zscore representations of the same feature compete within this context.
SELECTION_CONTEXT_COLS = ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"]



# =============================================================================
# Utility helpers
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_csv(x: str | None, default: Sequence[str]) -> List[str]:
    if x is None or str(x).strip() == "":
        return list(default)
    vals = [v.strip() for v in str(x).split(",") if v.strip()]
    return vals if vals else list(default)


def parse_weights(s: str | None) -> Dict[str, float]:
    if s is None or str(s).strip() == "":
        return dict(DEFAULT_RANK_WEIGHTS)
    out = dict(DEFAULT_RANK_WEIGHTS)
    for part in str(s).split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad --rank-weights component: {part!r}. Expected key=value.")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"Unknown rank weight key: {k}. Allowed: {sorted(out)}")
        out[k] = float(v)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Rank weights must sum to > 0.")
    return {k: v / total for k, v in out.items()}


def sanitize_path_token(x) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def safe_numeric(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def rank01_high_good(s: pd.Series, missing_fill: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    ok = x.notna()
    if ok.sum() == 0:
        return pd.Series(float(missing_fill), index=x.index, dtype=float)
    out.loc[ok] = x.loc[ok].rank(method="average", pct=True, ascending=True)
    return out.fillna(float(missing_fill))


def endpoint_type(endpoint: str) -> str:
    if endpoint in RESPONSE_ENDPOINTS:
        return "response"
    if endpoint in SURVIVAL_ENDPOINTS:
        return "survival"
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def metric_cols_for_endpoint(endpoint: str) -> Dict[str, Optional[str]]:
    if endpoint in RESPONSE_ENDPOINTS:
        return {
            "oof": "biomarker_oof_auc",
            "clinical": "clinical_oof_auc",
            "combo": "clinical_plus_biomarker_oof_auc",
            "delta": "delta_oof_auc_vs_clinical",
            "cv_std": "valid_auc_biomarker_std",
            "fold_mean": "valid_auc_biomarker_mean",
        }
    if endpoint in SURVIVAL_ENDPOINTS:
        return {
            "oof": "biomarker_oof_cindex",
            "clinical": "clinical_oof_cindex",
            "combo": "clinical_plus_biomarker_oof_cindex",
            "delta": "delta_oof_cindex_vs_clinical",
            "cv_std": "valid_cindex_biomarker_std",
            "fold_mean": "valid_cindex_biomarker_mean",
        }
    raise ValueError(endpoint)


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =============================================================================
# Stage-1 file loading and context parsing
# =============================================================================

def infer_context_from_path(path: Path) -> Dict[str, object]:
    """Infer v6 context columns from a nested Stage-1 result path."""
    parts = list(path.parts)
    ctx: Dict[str, object] = {}

    # Locate feature_source in path and infer fixed positions after it.
    fs_indices = [i for i, p in enumerate(parts) if p in FEATURE_SOURCES_ALL]
    if fs_indices:
        i = fs_indices[-1]
        names = ["feature_source", "panel", "feature_group", "cohort", "endpoint", "sample_type"]
        for j, name in enumerate(names):
            k = i + j
            if k < len(parts):
                ctx[name] = parts[k]
        # More nested modifiers.
        for p in parts[i + 1:]:
            if p.startswith("agg-"):
                ctx["agg"] = p.replace("agg-", "", 1)
            elif p.startswith("patient_subset-"):
                ctx["patient_subset"] = p.replace("patient_subset-", "", 1)
            elif p.startswith("transform-"):
                ctx["transform_mode"] = p.replace("transform-", "", 1)

    # Filename fallback:
    # cohort__panel__feature_source__feature_group__endpoint__sample_type__patient_subset__agg-median__chunk...
    stem = path.name
    stem = re.sub(r"__(summary|folds|oof|failures|fullmodels|feature_filter)\.csv$", "", stem)
    tokens = stem.split("__")
    if len(tokens) >= 8:
        fallback = {
            "cohort": tokens[0],
            "panel": tokens[1],
            "feature_source": tokens[2],
            "feature_group": tokens[3],
            "endpoint": tokens[4],
            "sample_type": tokens[5],
            "patient_subset": tokens[6],
        }
        if tokens[7].startswith("agg-"):
            fallback["agg"] = tokens[7].replace("agg-", "", 1)
        for k, v in fallback.items():
            ctx.setdefault(k, v)

    return ctx


def add_context_columns_from_source_file(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source_file" not in df.columns:
        return df
    out = df.copy()
    contexts = [infer_context_from_path(Path(fp)) for fp in out["source_file"].astype(str)]
    ctx_df = pd.DataFrame(contexts, index=out.index)
    for c in ctx_df.columns:
        if c not in out.columns:
            out[c] = ctx_df[c]
        else:
            out[c] = out[c].where(out[c].notna(), ctx_df[c])
    return out


def read_many(root: Path, suffix: str) -> pd.DataFrame:
    files = sorted(root.rglob(f"*__{suffix}.csv"))
    if not files:
        return pd.DataFrame()
    parts = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            df["source_file"] = str(fp)
            parts.append(df)
        except Exception as e:
            parts.append(pd.DataFrame({"source_file": [str(fp)], "read_error": [f"{type(e).__name__}: {e}"]}))
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return add_context_columns_from_source_file(out)


def normalize_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rename = {}
    if "Panel" in out.columns and "panel" not in out.columns:
        rename["Panel"] = "panel"
    if "sample_type" not in out.columns and "TURBT_or_RC" in out.columns:
        rename["TURBT_or_RC"] = "sample_type"
    out = out.rename(columns=rename)

    for c in [
        "cohort", "panel", "feature_source", "feature_group", "endpoint",
        "sample_type", "patient_subset", "agg", "transform_mode", "feature",
    ]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
            out[c] = out[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    # Standardize panel capitalization and sample types where present.
    if "panel" in out.columns:
        out["panel"] = out["panel"].astype("string").str.upper()
    if "sample_type" in out.columns:
        st = out["sample_type"].astype("string").str.upper().str.strip()
        st = st.replace({"RADICAL CYSTECTOMY": "RC", "CYSTECTOMY": "RC"})
        out["sample_type"] = st

    return out


def add_feature_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    needed = ["feature_source", "feature_group", "feature"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Cannot create feature_uid; missing columns: {missing}")
    out["feature_uid"] = (
        out["feature_source"].astype(str) + "|" +
        out["feature_group"].astype(str) + "|" +
        out["feature"].astype(str)
    )
    return out


# =============================================================================
# Filtering and merging
# =============================================================================

def filter_to_requested(
    df: pd.DataFrame,
    *,
    cohorts: Sequence[str],
    panels: Sequence[str],
    endpoints: Sequence[str],
    sample_types: Sequence[str],
    patient_subsets: Sequence[str],
    aggs: Sequence[str],
    transform_modes: Sequence[str],
    feature_sources: Sequence[str],
    feature_groups: Sequence[str],
    use_panel_source_map: bool,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    filters = {
        "cohort": cohorts,
        "panel": [p.upper() for p in panels],
        "endpoint": endpoints,
        "sample_type": [s.upper() for s in sample_types],
        "patient_subset": patient_subsets,
        "agg": aggs,
        "transform_mode": transform_modes,
        "feature_source": feature_sources,
        "feature_group": feature_groups,
    }

    for col, vals in filters.items():
        if col not in out.columns:
            continue
        vals = list(vals)
        if len(vals) == 1 and vals[0].lower() == "all":
            continue
        out = out[out[col].astype(str).isin([str(v) for v in vals])].copy()

    if use_panel_source_map and "panel" in out.columns and "feature_source" in out.columns:
        keep = []
        for _, r in out[["panel", "feature_source"]].iterrows():
            allowed = DEFAULT_PANEL_SOURCE_MAP.get(str(r["panel"]).upper(), FEATURE_SOURCES_ALL)
            keep.append(str(r["feature_source"]) in allowed)
        out = out[pd.Series(keep, index=out.index)].copy()

    return out


def select_best_fullmodel_rows(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame()
    out = normalize_context_columns(full)
    if "feature" not in out.columns:
        return pd.DataFrame()
    out = add_feature_uid(out)

    key_cols = [
        "cohort", "panel", "endpoint", "sample_type", "patient_subset",
        "agg", "transform_mode", "feature_source", "feature_group", "feature", "feature_uid",
    ]
    key_cols = [c for c in key_cols if c in out.columns]

    # Biomarker-only inferential statistics.
    bm = out[out.get("model_name", pd.Series(index=out.index, dtype=object)).astype(str).eq("biomarker_only")].copy()
    bm_cols_keep = key_cols + [
        c for c in [
            "coef", "coef_ci_low", "coef_ci_high", "effect", "effect_ci_low", "effect_ci_high",
            "wald_p_value", "auc_full", "cindex_full", "n", "n_events", "n_positive", "n_negative",
            "model_family", "fit_status",
        ] if c in bm.columns
    ]
    bm = bm[bm_cols_keep].copy() if not bm.empty else pd.DataFrame(columns=key_cols)
    bm = bm.rename(columns={
        "coef": "biomarker_only_coef",
        "coef_ci_low": "biomarker_only_coef_ci_low",
        "coef_ci_high": "biomarker_only_coef_ci_high",
        "effect": "biomarker_only_effect",
        "effect_ci_low": "biomarker_only_effect_ci_low",
        "effect_ci_high": "biomarker_only_effect_ci_high",
        "wald_p_value": "biomarker_only_wald_p_value",
        "auc_full": "biomarker_only_auc_full",
        "cindex_full": "biomarker_only_cindex_full",
    })
    if not bm.empty:
        bm = bm.sort_values(key_cols).drop_duplicates(key_cols, keep="first")

    # Clinical + biomarker rows include LRT and added-value fullfit metrics.
    cp = out[out.get("model_name", pd.Series(index=out.index, dtype=object)).astype(str).eq("clinical_plus_biomarker")].copy()
    cp_cols_keep = key_cols + [
        c for c in [
            "lrt_stat", "lrt_df", "lrt_p_value",
            "clinical_only_auc_full", "clinical_plus_biomarker_auc_full", "delta_auc_full_vs_clinical",
            "clinical_only_cindex_full", "clinical_plus_biomarker_cindex_full", "delta_cindex_full_vs_clinical",
            "wald_p_value", "effect", "coef",
        ] if c in cp.columns
    ]
    cp = cp[cp_cols_keep].copy() if not cp.empty else pd.DataFrame(columns=key_cols)
    cp = cp.rename(columns={
        "wald_p_value": "clinical_plus_biomarker_wald_p_value",
        "effect": "clinical_plus_biomarker_effect",
        "coef": "clinical_plus_biomarker_coef",
    })
    if not cp.empty:
        cp = cp.sort_values(key_cols).drop_duplicates(key_cols, keep="first")

    if bm.empty and cp.empty:
        return pd.DataFrame()
    if bm.empty:
        merged = cp
    elif cp.empty:
        merged = bm
    else:
        merged = bm.merge(cp, on=key_cols, how="outer")
    return merged


def summarize_feature_filter(feature_filter: pd.DataFrame) -> pd.DataFrame:
    if feature_filter.empty or "feature" not in feature_filter.columns:
        return pd.DataFrame()
    ff = normalize_context_columns(feature_filter)
    ff = add_feature_uid(ff)
    key_cols = [
        "cohort", "panel", "endpoint", "sample_type", "patient_subset",
        "agg", "transform_mode", "feature_source", "feature_group", "feature", "feature_uid",
    ]
    key_cols = [c for c in key_cols if c in ff.columns]
    value_cols = [c for c in ["n_patients", "n_nonmissing", "nonmissing_frac", "n_unique", "n_nonzero", "status", "reason"] if c in ff.columns]
    if not value_cols:
        return ff[key_cols].drop_duplicates()
    # Feature filter has one row per feature per job; collapse duplicates if chunks overlap unexpectedly.
    agg_spec = {}
    for c in value_cols:
        if c in {"status", "reason"}:
            agg_spec[c] = lambda x: ";".join(sorted(set(x.dropna().astype(str))))
        else:
            agg_spec[c] = "max"
    out = ff.groupby(key_cols, dropna=False).agg(agg_spec).reset_index()
    out = out.rename(columns={
        "n_patients": "ff_n_patients",
        "n_nonmissing": "ff_n_nonmissing",
        "nonmissing_frac": "ff_nonmissing_frac",
        "n_unique": "ff_n_unique",
        "n_nonzero": "ff_n_nonzero",
        "status": "ff_status",
        "reason": "ff_reason",
    })
    return out


def build_master_feature_table(summary: pd.DataFrame, fullmodels: pd.DataFrame, feature_filter: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        raise RuntimeError("No Stage-1 summary files found after filtering.")
    s = normalize_context_columns(summary)
    if "feature" not in s.columns:
        raise RuntimeError("Stage-1 summary table does not contain a feature column.")
    s = add_feature_uid(s)

    # Keep successful CV summaries only.
    if "status" in s.columns:
        s = s[s["status"].astype(str).eq("ok")].copy()

    full = select_best_fullmodel_rows(fullmodels)
    ff = summarize_feature_filter(feature_filter)

    key_cols = [
        "cohort", "panel", "endpoint", "sample_type", "patient_subset",
        "agg", "transform_mode", "feature_source", "feature_group", "feature", "feature_uid",
    ]
    key_cols = [c for c in key_cols if c in s.columns]

    # Drop duplicate summary rows from repeated scans/chunks if present.
    s = s.sort_values(key_cols).drop_duplicates(key_cols, keep="first")

    out = s
    if not full.empty:
        full_key = [c for c in key_cols if c in full.columns]
        out = out.merge(full, on=full_key, how="left", suffixes=("", "_full"))
    if not ff.empty:
        ff_key = [c for c in key_cols if c in ff.columns]
        out = out.merge(ff, on=ff_key, how="left", suffixes=("", "_ff"))

    return out


# =============================================================================
# Candidate scoring and selection
# =============================================================================

def add_candidate_score(df: pd.DataFrame, rank_weights: Dict[str, float], missing_rank_fill: float) -> pd.DataFrame:
    """
    Add candidate_score within each biological analysis context.

    Important v7 behavior
    ---------------------
    The ranking is performed within:
        cohort x panel x endpoint x sample_type x patient_subset x agg

    transform_mode is NOT part of the ranking context. Therefore, zscore and
    log1p_zscore versions of the same feature compete directly.
    """
    if df.empty:
        return df.copy()
    out = df.copy()

    group_cols = [c for c in SELECTION_CONTEXT_COLS if c in out.columns]
    if not group_cols:
        group_cols = ["endpoint"] if "endpoint" in out.columns else []

    scored_parts = []

    for _, g in out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]:
        endpoint_vals = g["endpoint"].dropna().astype(str).unique().tolist() if "endpoint" in g.columns else []
        endpoint = endpoint_vals[0] if endpoint_vals else ""
        try:
            cols = metric_cols_for_endpoint(endpoint)
        except Exception:
            # If endpoint is unknown, keep rows but make them low priority.
            gg = g.copy()
            gg["candidate_score"] = np.nan
            gg["primary_oof_metric"] = np.nan
            gg["primary_delta_metric"] = np.nan
            gg["primary_cv_std"] = np.nan
            gg["primary_metric_label"] = "unknown"
            scored_parts.append(gg)
            continue

        gg = g.copy()
        for c in [cols["delta"], cols["oof"], cols["cv_std"]]:
            if c is not None and c not in gg.columns:
                gg[c] = np.nan

        p_col = first_existing(gg, ["biomarker_only_wald_p_value", "clinical_plus_biomarker_wald_p_value", "wald_p_value"])
        e_col = first_existing(gg, ["biomarker_only_effect", "clinical_plus_biomarker_effect", "effect"])

        gg["score_delta_rank"] = rank01_high_good(gg[cols["delta"]], missing_fill=missing_rank_fill)
        gg["score_oof_rank"] = rank01_high_good(gg[cols["oof"]], missing_fill=missing_rank_fill)

        if p_col is not None:
            p = pd.to_numeric(gg[p_col], errors="coerce")
            gg["score_pval_rank"] = rank01_high_good(-np.log10(p.clip(lower=1e-300)), missing_fill=missing_rank_fill)
        else:
            gg["score_pval_rank"] = float(missing_rank_fill)

        if e_col is not None:
            eff = pd.to_numeric(gg[e_col], errors="coerce").replace(0, np.nan)
            gg["abs_log_effect"] = np.abs(np.log(eff))
            gg["score_effect_rank"] = rank01_high_good(gg["abs_log_effect"], missing_fill=missing_rank_fill)
        else:
            gg["abs_log_effect"] = np.nan
            gg["score_effect_rank"] = float(missing_rank_fill)

        gg["candidate_score"] = (
            rank_weights["delta"] * gg["score_delta_rank"] +
            rank_weights["oof"] * gg["score_oof_rank"] +
            rank_weights["pval"] * gg["score_pval_rank"] +
            rank_weights["effect"] * gg["score_effect_rank"]
        )

        gg["primary_oof_metric"] = pd.to_numeric(gg[cols["oof"]], errors="coerce")
        gg["primary_delta_metric"] = pd.to_numeric(gg[cols["delta"]], errors="coerce")
        gg["primary_cv_std"] = pd.to_numeric(gg[cols["cv_std"]], errors="coerce")
        gg["primary_metric_label"] = "AUC" if endpoint in RESPONSE_ENDPOINTS else "C-index"
        scored_parts.append(gg)

    return pd.concat(scored_parts, ignore_index=True, sort=False) if scored_parts else out


def choose_best_representation_per_feature_uid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one representation per underlying feature within a selection context.

    The underlying feature ID is:
        feature_source | feature_group | feature

    If both zscore and log1p_zscore exist, the row with the highest candidate_score
    is retained. The selected transform is carried forward in selected_transform_mode.
    Aggregation is still treated as a separate context; this function does not choose
    the best aggregation unless multiple agg values were intentionally placed into the
    same selection context elsewhere.
    """
    if df.empty:
        return df.copy()

    context_cols = [c for c in SELECTION_CONTEXT_COLS if c in df.columns]
    key_cols = context_cols + ["feature_uid"]
    sort_cols = ["candidate_score", "primary_oof_metric", "primary_delta_metric"]
    for c in sort_cols:
        if c not in df.columns:
            df[c] = np.nan

    work = df.copy()

    # Record how many transformations were evaluated for each feature within this context.
    transform_summary = (
        work.groupby(key_cols, dropna=False)
        .agg(
            n_transform_representations_evaluated=("transform_mode", lambda x: int(pd.Series(x).dropna().astype(str).nunique())),
            transform_modes_evaluated=("transform_mode", lambda x: ";".join(sorted(set(pd.Series(x).dropna().astype(str))))),
        )
        .reset_index()
    ) if "transform_mode" in work.columns else pd.DataFrame()

    work = work.sort_values(sort_cols, ascending=[False, False, False], na_position="last")
    out = work.drop_duplicates(key_cols, keep="first").reset_index(drop=True)

    if "transform_mode" in out.columns:
        out["selected_transform_mode"] = out["transform_mode"].astype(str)
    else:
        out["selected_transform_mode"] = pd.NA

    # Since transform is selected within the context, keep a display token that is stable
    # for folder names and context ids.
    out["transform_selection"] = "best_of_requested"

    if not transform_summary.empty:
        out = out.merge(transform_summary, on=key_cols, how="left")

    return out


def cv_std_plateau_filter_context(
    df: pd.DataFrame,
    *,
    plateau_frac: float,
    grid_n: int,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """Context-level version of the previous CV STD plateau filter."""
    if df.empty or "primary_cv_std" not in df.columns:
        return df.copy(), pd.DataFrame(), None

    work = df.dropna(subset=["primary_cv_std", "primary_oof_metric"]).copy()
    if work.empty or work["primary_cv_std"].nunique() <= 1:
        return df.copy(), pd.DataFrame(), None

    vals = np.linspace(work["primary_cv_std"].min(), work["primary_cv_std"].max(), int(grid_n))
    rows = []
    for thr in vals:
        sub = work[work["primary_cv_std"] <= thr].copy()
        if sub.empty:
            rows.append({"cv_std_threshold": float(thr), "n_retained": 0, "mean_perf_topN": np.nan})
            continue
        top = sub.sort_values("candidate_score", ascending=False).head(int(top_n))
        rows.append({
            "cv_std_threshold": float(thr),
            "n_retained": int(sub["feature_uid"].nunique()),
            "mean_perf_topN": float(pd.to_numeric(top["primary_oof_metric"], errors="coerce").mean()),
        })
    diag = pd.DataFrame(rows)
    tmp = diag.dropna(subset=["mean_perf_topN"]).copy()
    if tmp.empty:
        return df.copy(), diag, None
    best_perf = tmp["mean_perf_topN"].max()
    target = float(plateau_frac) * best_perf
    candidates = tmp[tmp["mean_perf_topN"] >= target].copy()
    chosen = float(candidates["cv_std_threshold"].min()) if not candidates.empty else None
    if chosen is None:
        return df.copy(), diag, None
    filtered = df[pd.to_numeric(df["primary_cv_std"], errors="coerce") <= chosen].copy()
    diag["chosen_threshold"] = chosen
    return filtered.reset_index(drop=True), diag, chosen


def select_balanced_candidates(
    stable_df: pd.DataFrame,
    *,
    top_per_source_group: int,
    top_per_source: Optional[int],
    max_candidates_per_context: Optional[int],
    min_candidates_per_context: int,
) -> pd.DataFrame:
    if stable_df.empty:
        return stable_df.copy()

    work = stable_df.sort_values("candidate_score", ascending=False).copy()

    selected_parts = []
    if top_per_source_group is not None and int(top_per_source_group) > 0:
        for _, g in work.groupby(["feature_source", "feature_group"], dropna=False):
            selected_parts.append(g.head(int(top_per_source_group)))
        selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    else:
        selected = work.copy()

    if top_per_source is not None and int(top_per_source) > 0 and not selected.empty:
        selected = (
            selected.sort_values("candidate_score", ascending=False)
            .groupby("feature_source", dropna=False, group_keys=False)
            .head(int(top_per_source))
            .copy()
        )

    if selected.empty:
        selected = work.head(int(min_candidates_per_context)).copy()

    # Rescue globally top features if the balanced cap left too few candidates.
    if selected["feature_uid"].nunique() < int(min_candidates_per_context):
        need = int(min_candidates_per_context) - selected["feature_uid"].nunique()
        rescue = work[~work["feature_uid"].isin(selected["feature_uid"])]
        selected = pd.concat([selected, rescue.head(need)], ignore_index=True, sort=False)

    selected = selected.sort_values("candidate_score", ascending=False).drop_duplicates("feature_uid", keep="first")

    if max_candidates_per_context is not None and int(max_candidates_per_context) > 0:
        selected = selected.head(int(max_candidates_per_context)).copy()

    selected = selected.reset_index(drop=True)
    selected["selection_rank_within_context"] = np.arange(1, selected.shape[0] + 1)
    selected["selected_stage2_candidate"] = True

    # Stratum ranks help audit root balance.
    selected["rank_within_feature_source"] = (
        selected.groupby("feature_source")["candidate_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    selected["rank_within_source_group"] = (
        selected.groupby(["feature_source", "feature_group"])["candidate_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return selected


def context_id_from_row(row: pd.Series) -> str:
    vals = [
        row.get("cohort", "NA"),
        row.get("panel", "NA"),
        row.get("endpoint", "NA"),
        row.get("sample_type", "NA"),
        row.get("patient_subset", "NA"),
        f"agg-{row.get('agg', 'NA')}",
        "best_transform",
    ]
    return "__".join(sanitize_path_token(v) for v in vals)


def context_output_dir(output_root: Path, row: pd.Series) -> Path:
    return (
        output_root
        / sanitize_path_token(row.get("cohort", "NA"))
        / sanitize_path_token(row.get("panel", "NA"))
        / sanitize_path_token(row.get("endpoint", "NA"))
        / sanitize_path_token(row.get("sample_type", "NA"))
        / sanitize_path_token(row.get("patient_subset", "NA"))
        / sanitize_path_token(f"agg-{row.get('agg', 'NA')}")
        / "best_transform"
    )


def write_context_outputs(
    output_root: Path,
    ranked: pd.DataFrame,
    stable: pd.DataFrame,
    candidates: pd.DataFrame,
    cvdiag: pd.DataFrame,
    config: Dict[str, object],
) -> None:
    if ranked.empty:
        return
    outdir = ensure_dir(context_output_dir(output_root, ranked.iloc[0]))
    ranked.to_csv(outdir / "ranked_all_features.csv", index=False)
    stable.to_csv(outdir / "stable_ranked_features.csv", index=False)
    candidates.to_csv(outdir / "candidate_features_balanced.csv", index=False)
    if not cvdiag.empty:
        cvdiag.to_csv(outdir / "cv_std_diagnostics.csv", index=False)

    counts = pd.DataFrame()
    if not candidates.empty:
        counts = (
            candidates.groupby(["feature_source", "feature_group"], dropna=False)
            .size()
            .reset_index(name="n_selected")
            .sort_values(["feature_source", "feature_group"])
        )
        counts.to_csv(outdir / "selected_candidate_counts_by_source_group.csv", index=False)

    summary = {
        **config,
        "context_id": ranked.iloc[0].get("context_id", context_id_from_row(ranked.iloc[0])),
        "n_ranked_features": int(ranked.shape[0]),
        "n_stable_features": int(stable.shape[0]),
        "n_selected_candidates": int(candidates.shape[0]),
        "n_selected_by_source_group": counts.to_dict(orient="records") if not counts.empty else [],
    }
    with open(outdir / "context_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# =============================================================================
# Main driver
# =============================================================================

def run(args: argparse.Namespace) -> None:
    stage1_root = Path(args.stage1_root)
    output_root = ensure_dir(Path(args.output_root))

    cohorts = ALL_COHORTS_DEFAULT if args.evaluate_all_cohorts else split_csv(args.cohorts, DISCOVERY_COHORTS_DEFAULT)
    aggregation_summary_cohorts = split_csv(args.aggregation_summary_cohorts, cohorts)
    panels = split_csv(args.panels, PANELS_DEFAULT)
    endpoints = split_csv(args.endpoints, ENDPOINTS_DEFAULT)
    sample_types = split_csv(args.sample_types, ["TURBT"])
    patient_subsets = split_csv(args.patient_subsets, ["all"])
    aggs = split_csv(args.aggs, ["median"])
    transform_modes = split_csv(args.transform_modes, ["zscore"])
    feature_sources = split_csv(args.feature_sources, FEATURE_SOURCES_ALL)
    feature_groups = split_csv(args.feature_groups, FEATURE_GROUPS_ALL)
    rank_weights = parse_weights(args.rank_weights)

    config = {
        "stage1_root": str(stage1_root),
        "output_root": str(output_root),
        "cohorts_evaluated": cohorts,
        "aggregation_summary_cohorts": aggregation_summary_cohorts,
        "panels": panels,
        "endpoints": endpoints,
        "sample_types": sample_types,
        "patient_subsets": patient_subsets,
        "aggs": aggs,
        "transform_modes": transform_modes,
        "feature_sources": feature_sources,
        "feature_groups": feature_groups,
        "use_panel_source_map": bool(args.use_panel_source_map),
        "rank_weights": rank_weights,
        "use_cv_std_filter": bool(args.use_cv_std_filter),
        "cv_plateau_frac": args.cv_plateau_frac,
        "top_per_source_group": args.top_per_source_group,
        "top_per_source": args.top_per_source,
        "max_candidates_per_context": args.max_candidates_per_context,
        "min_candidates_per_context": args.min_candidates_per_context,
    }
    with open(output_root / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    log(f"[INFO] Reading Stage-1 summaries from: {stage1_root}")
    summary = read_many(stage1_root, "summary")
    fullmodels = read_many(stage1_root, "fullmodels")
    feature_filter = read_many(stage1_root, "feature_filter")

    log(f"[INFO] summary shape before filtering: {summary.shape}")
    log(f"[INFO] fullmodels shape before filtering: {fullmodels.shape}")
    log(f"[INFO] feature_filter shape before filtering: {feature_filter.shape}")

    summary = filter_to_requested(
        normalize_context_columns(summary),
        cohorts=cohorts,
        panels=panels,
        endpoints=endpoints,
        sample_types=sample_types,
        patient_subsets=patient_subsets,
        aggs=aggs,
        transform_modes=transform_modes,
        feature_sources=feature_sources,
        feature_groups=feature_groups,
        use_panel_source_map=bool(args.use_panel_source_map),
    )
    fullmodels = filter_to_requested(
        normalize_context_columns(fullmodels),
        cohorts=cohorts,
        panels=panels,
        endpoints=endpoints,
        sample_types=sample_types,
        patient_subsets=patient_subsets,
        aggs=aggs,
        transform_modes=transform_modes,
        feature_sources=feature_sources,
        feature_groups=feature_groups,
        use_panel_source_map=bool(args.use_panel_source_map),
    )
    feature_filter = filter_to_requested(
        normalize_context_columns(feature_filter),
        cohorts=cohorts,
        panels=panels,
        endpoints=endpoints,
        sample_types=sample_types,
        patient_subsets=patient_subsets,
        aggs=aggs,
        transform_modes=transform_modes,
        feature_sources=feature_sources,
        feature_groups=feature_groups,
        use_panel_source_map=bool(args.use_panel_source_map),
    )

    log(f"[INFO] summary shape after filtering: {summary.shape}")
    log(f"[INFO] fullmodels shape after filtering: {fullmodels.shape}")
    log(f"[INFO] feature_filter shape after filtering: {feature_filter.shape}")

    master = build_master_feature_table(summary, fullmodels, feature_filter)
    master = add_candidate_score(master, rank_weights=rank_weights, missing_rank_fill=args.missing_rank_fill)

    # v7: zscore/log1p_zscore representations compete; only one transform per
    # underlying feature_uid is retained within each selection context.
    all_scored_representations = master.copy()
    all_scored_representations.to_csv(output_root / "all_context_scored_representations_before_best_transform.csv", index=False)
    master = choose_best_representation_per_feature_uid(master)

    # Do not run response contexts for No-NAC/KOLL if they accidentally exist or if empty summaries were created.
    if args.drop_non_treatment_response_contexts:
        mask_bad = master["cohort"].astype(str).isin(["No-NAC", "KOLL"]) & master["endpoint"].astype(str).isin(RESPONSE_ENDPOINTS)
        if mask_bad.any():
            log(f"[INFO] Dropping No-NAC/KOLL response rows: {int(mask_bad.sum())}")
            master = master[~mask_bad].copy()

    if master.empty:
        raise RuntimeError("No valid biomarker rows remain after filtering and scoring.")

    context_cols = [c for c in SELECTION_CONTEXT_COLS if c in master.columns]
    master["context_id"] = master.apply(context_id_from_row, axis=1)

    all_ranked = []
    all_stable = []
    all_candidates = []
    context_summaries = []

    for ctx_vals, ctx_df in master.groupby(context_cols, dropna=False):
        ctx_df = ctx_df.copy()
        ctx_df = ctx_df.sort_values(["candidate_score", "primary_oof_metric"], ascending=[False, False], na_position="last").reset_index(drop=True)
        ctx_df["rank_within_context_all_features"] = np.arange(1, ctx_df.shape[0] + 1)

        if args.use_cv_std_filter:
            stable, cvdiag, chosen_cv = cv_std_plateau_filter_context(
                ctx_df,
                plateau_frac=args.cv_plateau_frac,
                grid_n=args.cv_std_grid_n,
                top_n=args.cv_top_n,
            )
        else:
            stable, cvdiag, chosen_cv = ctx_df.copy(), pd.DataFrame(), None

        stable = stable.sort_values(["candidate_score", "primary_oof_metric"], ascending=[False, False], na_position="last").reset_index(drop=True)
        stable["rank_within_context_stable"] = np.arange(1, stable.shape[0] + 1)

        cand = select_balanced_candidates(
            stable,
            top_per_source_group=args.top_per_source_group,
            top_per_source=args.top_per_source,
            max_candidates_per_context=args.max_candidates_per_context,
            min_candidates_per_context=args.min_candidates_per_context,
        )
        cand["cv_std_chosen_threshold"] = chosen_cv

        write_context_outputs(output_root, ctx_df, stable, cand, cvdiag, config)

        all_ranked.append(ctx_df)
        all_stable.append(stable)
        all_candidates.append(cand)

        first = ctx_df.iloc[0]
        context_summaries.append({
            "context_id": first["context_id"],
            "cohort": first.get("cohort"),
            "panel": first.get("panel"),
            "endpoint": first.get("endpoint"),
            "sample_type": first.get("sample_type"),
            "patient_subset": first.get("patient_subset"),
            "agg": first.get("agg"),
            "transform_selection": first.get("transform_selection", "best_of_requested"),
            "n_ranked_features": int(ctx_df.shape[0]),
            "n_stable_features": int(stable.shape[0]),
            "n_selected_candidates": int(cand.shape[0]),
            "cv_std_chosen_threshold": chosen_cv,
        })

        log(
            "[DONE context] "
            f"{first.get('cohort')} {first.get('panel')} {first.get('endpoint')} "
            f"{first.get('sample_type')} {first.get('patient_subset')} "
            f"ranked={ctx_df.shape[0]} stable={stable.shape[0]} selected={cand.shape[0]}"
        )

    ranked_df = pd.concat(all_ranked, ignore_index=True, sort=False) if all_ranked else pd.DataFrame()
    stable_df = pd.concat(all_stable, ignore_index=True, sort=False) if all_stable else pd.DataFrame()
    candidate_df = pd.concat(all_candidates, ignore_index=True, sort=False) if all_candidates else pd.DataFrame()
    context_summary_df = pd.DataFrame(context_summaries)

    ranked_df.to_csv(output_root / "all_context_ranked_features.csv", index=False)
    stable_df.to_csv(output_root / "all_context_stable_features.csv", index=False)
    candidate_df.to_csv(output_root / "all_context_candidates_balanced.csv", index=False)
    context_summary_df.to_csv(output_root / "context_selection_summary.csv", index=False)

    # Save all evaluated selected candidates, then create the aggregation/module-discovery
    # manifest using only aggregation_summary_cohorts. This lets you audit all cohorts
    # while keeping discovery summaries restricted to the intended cohort set.
    manifest_cols = [
        "context_id", "cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg",
        "transform_selection", "selected_transform_mode", "transform_modes_evaluated",
        "n_transform_representations_evaluated",
        "feature_uid", "feature", "feature_source", "feature_group",
        "candidate_score", "selection_rank_within_context", "rank_within_feature_source", "rank_within_source_group",
        "primary_oof_metric", "primary_delta_metric", "primary_cv_std",
        "biomarker_only_wald_p_value", "biomarker_only_effect", "abs_log_effect",
        "ff_n_patients", "ff_nonmissing_frac", "ff_n_unique", "ff_n_nonzero",
    ]
    manifest_cols = [c for c in manifest_cols if c in candidate_df.columns]
    all_manifest = candidate_df[manifest_cols].copy() if not candidate_df.empty else pd.DataFrame(columns=manifest_cols)
    all_manifest.to_csv(output_root / "all_evaluated_candidate_manifest.csv", index=False)

    if len(aggregation_summary_cohorts) == 1 and str(aggregation_summary_cohorts[0]).lower() == "all":
        aggregation_candidate_df = candidate_df.copy()
    else:
        aggregation_candidate_df = candidate_df[candidate_df["cohort"].astype(str).isin([str(c) for c in aggregation_summary_cohorts])].copy()

    manifest = aggregation_candidate_df[manifest_cols].copy() if not aggregation_candidate_df.empty else pd.DataFrame(columns=manifest_cols)
    manifest.to_csv(output_root / "global_module_candidate_manifest.csv", index=False)

    # Audit tables for all evaluated cohorts.
    if not candidate_df.empty:
        (
            candidate_df.groupby(["panel", "feature_source", "feature_group"], dropna=False)
            .size()
            .reset_index(name="n_selected_candidates")
            .sort_values(["panel", "feature_source", "feature_group"])
            .to_csv(output_root / "all_evaluated_selected_candidate_counts_by_panel_source_group.csv", index=False)
        )
        (
            candidate_df.groupby(["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"], dropna=False)
            .size()
            .reset_index(name="n_selected_candidates")
            .sort_values(["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"])
            .to_csv(output_root / "all_evaluated_selected_candidate_counts_by_context.csv", index=False)
        )

    # Aggregation/module-discovery cohort summaries.
    if not aggregation_candidate_df.empty:
        (
            aggregation_candidate_df.groupby(["panel", "feature_source", "feature_group"], dropna=False)
            .size()
            .reset_index(name="n_selected_candidates")
            .sort_values(["panel", "feature_source", "feature_group"])
            .to_csv(output_root / "aggregation_summary_candidate_counts_by_panel_source_group.csv", index=False)
        )
        (
            aggregation_candidate_df.groupby(["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"], dropna=False)
            .size()
            .reset_index(name="n_selected_candidates")
            .sort_values(["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"])
            .to_csv(output_root / "aggregation_summary_candidate_counts_by_context.csv", index=False)
        )
        support_cols = ["panel", "feature_uid", "feature_source", "feature_group", "feature"]
        support_cols = [c for c in support_cols if c in aggregation_candidate_df.columns]
        support = (
            aggregation_candidate_df.groupby(support_cols, dropna=False)
            .agg(
                n_contexts=("context_id", "nunique"),
                n_cohorts=("cohort", "nunique"),
                cohorts=("cohort", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                endpoints=("endpoint", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                sample_types=("sample_type", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                aggs=("agg", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                selected_transform_modes=("selected_transform_mode", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                max_candidate_score=("candidate_score", "max"),
                median_candidate_score=("candidate_score", "median"),
                median_oof_metric=("primary_oof_metric", "median"),
                median_delta_metric=("primary_delta_metric", "median"),
            )
            .reset_index()
            .sort_values(["panel", "n_cohorts", "n_contexts", "max_candidate_score"], ascending=[True, False, False, False])
        )
        support.to_csv(output_root / "aggregation_summary_feature_support.csv", index=False)

    log("=" * 80)
    log(f"[DONE] ranked features: {ranked_df.shape} -> {output_root / 'all_context_ranked_features.csv'}")
    log(f"[DONE] stable features: {stable_df.shape} -> {output_root / 'all_context_stable_features.csv'}")
    log(f"[DONE] selected candidates: {candidate_df.shape} -> {output_root / 'all_context_candidates_balanced.csv'}")
    log(f"[DONE] aggregation/discovery manifest: {manifest.shape} -> {output_root / 'global_module_candidate_manifest.csv'}")
    log(f"[DONE] all-evaluated manifest: {all_manifest.shape} -> {output_root / 'all_evaluated_candidate_manifest.csv'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 2A candidate selection from Stage-1 v6 outputs.")
    ap.add_argument("--stage1-root", required=True, help="Stage-1 v6 results root to scan recursively.")
    ap.add_argument("--output-root", required=True, help="Output root for context-level candidate files.")

    ap.add_argument("--cohorts", default=",".join(DISCOVERY_COHORTS_DEFAULT), help="Comma-separated cohorts for candidate selection. Ignored if --evaluate-all-cohorts is set.")
    ap.add_argument("--evaluate-all-cohorts", action="store_true", help="Evaluate all known cohorts: NAC2020,PURE01,BLASST,No-NAC,NAC2015,KOLL.")
    ap.add_argument("--aggregation-summary-cohorts", "--module-discovery-cohorts", dest="aggregation_summary_cohorts", default=None, help="Comma-separated cohorts to include in global_module_candidate_manifest and aggregation summaries. Defaults to evaluated cohorts. Use main discovery cohorts here when --evaluate-all-cohorts is used.")
    ap.add_argument("--panels", default=",".join(PANELS_DEFAULT), help="Comma-separated panels.")
    ap.add_argument("--endpoints", default=",".join(ENDPOINTS_DEFAULT), help="Comma-separated endpoints.")
    ap.add_argument("--sample-types", default="TURBT", help="Comma-separated sample types. Primary discovery should usually be TURBT only.")
    ap.add_argument("--patient-subsets", default="all", help="Comma-separated patient subsets. Discovery should usually be all only.")
    ap.add_argument("--aggs", default="median", help="Comma-separated aggregation methods. Primary discovery should usually be median only.")
    ap.add_argument("--transform-modes", default="zscore,log1p_zscore", help="Comma-separated transforms to evaluate. zscore and log1p_zscore compete within each context; only the best is carried forward per feature.")
    ap.add_argument("--feature-sources", default=",".join(FEATURE_SOURCES_ALL), help="Comma-separated feature sources / prep roots.")
    ap.add_argument("--feature-groups", default=",".join(FEATURE_GROUPS_ALL), help="Comma-separated feature groups.")

    ap.add_argument("--use-panel-source-map", action=argparse.BooleanOptionalAction, default=True, help="Use recommended AR/BT feature-source map. AR gets all roots; BT gets phenotype_only+compartment.")
    ap.add_argument("--drop-non-treatment-response-contexts", action=argparse.BooleanOptionalAction, default=True, help="Drop No-NAC/KOLL response contexts if any rows exist.")

    ap.add_argument("--rank-weights", default=None, help="Optional comma-separated weights, e.g. delta=0.4,oof=0.3,pval=0.15,effect=0.15")
    ap.add_argument("--missing-rank-fill", type=float, default=0.0, help="Rank value assigned to missing p/effect/delta components when some values are present.")

    ap.add_argument("--use-cv-std-filter", action=argparse.BooleanOptionalAction, default=True, help="Apply old-style CV STD plateau filter per context.")
    ap.add_argument("--cv-plateau-frac", type=float, default=CV_PLATEAU_FRAC_DEFAULT)
    ap.add_argument("--cv-std-grid-n", type=int, default=CV_STD_GRID_N_DEFAULT)
    ap.add_argument("--cv-top-n", type=int, default=CV_TOP_N_DEFAULT)

    ap.add_argument("--top-per-source-group", type=int, default=5, help="Max selected features per feature_source x feature_group x context.")
    ap.add_argument("--top-per-source", type=int, default=0, help="Optional max selected features per feature_source x context. 0 disables.")
    ap.add_argument("--max-candidates-per-context", type=int, default=80, help="Optional global cap per context. 0 disables.")
    ap.add_argument("--min-candidates-per-context", type=int, default=20, help="Rescue top features until each context has at least this many, if available.")

    args = ap.parse_args()
    if args.top_per_source <= 0:
        args.top_per_source = None
    if args.max_candidates_per_context <= 0:
        args.max_candidates_per_context = None
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        raise
