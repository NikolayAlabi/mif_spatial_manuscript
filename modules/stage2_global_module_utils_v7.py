#!/usr/bin/env python3
"""
stage2_global_module_utils_v7.py

Shared utilities for streamlined Stage 2B global module discovery.

Design goals
------------
1. Keep the notebook compact: the notebook should load saved outputs, inspect plots,
   choose final k, and save final module memberships.
2. Keep heavy computation in scripts: patient-matrix preparation, consensus correlation,
   support filtering, k diagnostics, and heatmap plotting.
3. Support the expanded prep-root feature universe using feature_uid:
       feature_source|feature_group|feature
4. Preserve the older biological k-selection logic:
       - statistical silhouette
       - simple / primitive cell-type semantic silhouette
       - tissue-aware ontology semantic silhouette
       - singleton fraction / max cluster fraction / cluster-size balance
       - heatmaps and module-composition diagnostics

This file is intentionally standalone except for optional import of the Stage 1 v6
script when patient matrices need to be reconstructed from raw feature tables.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DISCOVERY_COHORTS = ["NAC2020", "PURE01", "BLASST", "No-NAC"]
DEFAULT_PANELS = ["AR", "BT"]
DEFAULT_FEATURE_SOURCES_AR = ["phenotype_only", "AR_state", "AR_checkpoint_state", "compartment", "compartment_state"]
DEFAULT_FEATURE_SOURCES_BT = ["phenotype_only", "compartment"]
DEFAULT_FEATURE_GROUPS = ["NN", "athena", "cell_features", "triads"]

CONTEXT_MATRIX_COLS = ["cohort", "panel", "sample_type", "patient_subset", "agg"]
FEATURE_ID_COLS = ["feature_source", "feature_group", "feature"]

# =============================================================================
# Basic IO
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


def save_fig(fig, path: str | Path, dpi: int = 300) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"[SAVE] {path}")


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_candidate_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ["cohort", "panel", "feature_source", "feature_group", "feature"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate manifest missing required columns: {missing}")
    df = df.copy()
    for c in required + ["endpoint", "sample_type", "patient_subset", "agg", "selected_transform_mode"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "feature_uid" not in df.columns:
        df["feature_uid"] = make_feature_uid_series(df)
    return df


def make_feature_uid(feature_source: str, feature_group: str, feature: str) -> str:
    return f"{feature_source}|{feature_group}|{feature}"


def make_feature_uid_series(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "|"
        + df["feature_group"].astype(str)
        + "|"
        + df["feature"].astype(str)
    )


def split_feature_uid(uid: str) -> Tuple[str, str, str]:
    parts = str(uid).split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", str(uid)


def safe_numeric(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, ascending=higher_better)

# =============================================================================
# Candidate manifest validation and summaries
# =============================================================================

def filter_manifest(manifest: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    df = manifest.copy()
    filters = {
        "cohort": config.get("discovery_cohorts"),
        "panel": config.get("panels"),
        "sample_type": config.get("sample_types") or ([config.get("sample_type")] if config.get("sample_type") else None),
        "patient_subset": config.get("patient_subsets") or ([config.get("patient_subset")] if config.get("patient_subset") else None),
        "agg": config.get("aggs") or ([config.get("agg")] if config.get("agg") else None),
        "feature_source": config.get("feature_sources"),
        "feature_group": config.get("feature_groups"),
    }
    for col, vals in filters.items():
        if vals is None or col not in df.columns:
            continue
        vals = [str(v) for v in vals]
        df = df[df[col].astype(str).isin(vals)].copy()
    return df


def validate_candidate_manifest(manifest: pd.DataFrame, config: Mapping) -> Dict[str, pd.DataFrame]:
    df = filter_manifest(manifest, config)
    warnings_rows = []
    if df.empty:
        warnings_rows.append({"level": "ERROR", "message": "No candidate rows remain after config filtering."})
    if df["feature_uid"].isna().any():
        warnings_rows.append({"level": "ERROR", "message": "Some feature_uid values are missing."})
    dup_cols = [c for c in ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg", "feature_uid"] if c in df.columns]
    if dup_cols:
        n_dups = int(df.duplicated(dup_cols).sum())
        if n_dups:
            warnings_rows.append({"level": "WARN", "message": f"{n_dups} duplicated feature_uid rows within context. Highest candidate_score will be used when needed."})

    summary = pd.DataFrame([{
        "n_rows": int(df.shape[0]),
        "n_feature_uids": int(df["feature_uid"].nunique()) if not df.empty else 0,
        "n_cohorts": int(df["cohort"].nunique()) if "cohort" in df.columns and not df.empty else 0,
        "n_panels": int(df["panel"].nunique()) if "panel" in df.columns and not df.empty else 0,
        "n_contexts": int(df[[c for c in ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"] if c in df.columns]].drop_duplicates().shape[0]) if not df.empty else 0,
    }])

    counts_by_context = (
        df.groupby([c for c in ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"] if c in df.columns], dropna=False)
        .agg(n_candidates=("feature_uid", "nunique"), n_rows=("feature_uid", "size"))
        .reset_index()
        .sort_values([c for c in ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"] if c in df.columns])
    ) if not df.empty else pd.DataFrame()

    counts_by_source = (
        df.groupby(["panel", "feature_source", "feature_group"], dropna=False)
        .agg(n_candidates=("feature_uid", "nunique"), n_rows=("feature_uid", "size"))
        .reset_index()
        .sort_values(["panel", "feature_source", "feature_group"])
    ) if not df.empty else pd.DataFrame()

    return {
        "manifest_used": df,
        "summary": summary,
        "counts_by_context": counts_by_context,
        "counts_by_source_group": counts_by_source,
        "warnings": pd.DataFrame(warnings_rows),
    }


def summarize_candidate_composition(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = manifest.copy()
    out = {}
    if df.empty:
        return {"by_panel_source_group": pd.DataFrame(), "by_panel_cohort": pd.DataFrame(), "feature_support": pd.DataFrame()}
    out["by_panel_source_group"] = (
        df.groupby(["panel", "feature_source", "feature_group"], dropna=False)
        .agg(n_feature_uids=("feature_uid", "nunique"), n_rows=("feature_uid", "size"))
        .reset_index()
        .sort_values(["panel", "feature_source", "feature_group"])
    )
    out["by_panel_cohort"] = (
        df.groupby(["panel", "cohort"], dropna=False)
        .agg(n_feature_uids=("feature_uid", "nunique"), n_rows=("feature_uid", "size"))
        .reset_index()
        .sort_values(["panel", "cohort"])
    )
    out["feature_support"] = (
        df.groupby(["panel", "feature_uid", "feature_source", "feature_group", "feature"], dropna=False)
        .agg(
            n_contexts=("context_id", "nunique") if "context_id" in df.columns else ("feature_uid", "size"),
            n_cohorts=("cohort", "nunique"),
            cohorts=("cohort", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
            endpoints=("endpoint", lambda x: ";".join(sorted(set(x.dropna().astype(str)))) if "endpoint" in df.columns else ""),
            selected_transform_modes=("selected_transform_mode", lambda x: ";".join(sorted(set(x.dropna().astype(str)))) if "selected_transform_mode" in df.columns else ""),
            max_candidate_score=("candidate_score", "max") if "candidate_score" in df.columns else ("feature_uid", "size"),
            median_candidate_score=("candidate_score", "median") if "candidate_score" in df.columns else ("feature_uid", "size"),
        )
        .reset_index()
        .sort_values(["panel", "n_cohorts", "n_contexts", "max_candidate_score"], ascending=[True, False, False, False])
    )
    return out

# =============================================================================
# Transformations and patient matrix construction
# =============================================================================

def fit_apply_transform_full(x: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mode = str(mode) if pd.notna(mode) else "zscore"
    if mode == "raw":
        return x
    if mode == "log1p_zscore":
        if (x.dropna() < 0).any():
            return pd.Series(np.nan, index=x.index)
        x = np.log1p(x)
    # zscore default
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def _choose_context_feature_representations(ctx_manifest: pd.DataFrame) -> pd.DataFrame:
    """
    One row per feature_uid within a matrix context. If the same feature_uid enters
    multiple endpoint contexts or has multiple transform choices, keep the highest
    candidate_score row. This defines the vector used for cohort-level correlation.
    """
    work = ctx_manifest.copy()
    if "candidate_score" not in work.columns:
        work["candidate_score"] = np.nan
    if "primary_oof_metric" not in work.columns:
        work["primary_oof_metric"] = np.nan
    if "selected_transform_mode" not in work.columns:
        work["selected_transform_mode"] = work.get("transform_mode", "zscore")
    work = work.sort_values(["candidate_score", "primary_oof_metric"], ascending=[False, False], na_position="last")
    return work.drop_duplicates("feature_uid", keep="first").reset_index(drop=True)


def build_patient_matrix_for_source_group(
    *,
    stage1_mod,
    cohort: str,
    panel: str,
    feature_source: str,
    feature_group: str,
    features: Sequence[str],
    sample_type: str,
    patient_subset: str,
    agg: str,
    qc_acceptability: str,
    min_epi_fraction: Optional[float],
    harmonized_path: str | Path,
    spatial_root: Optional[str | Path] = None,
    cell_features_path: Optional[str | Path] = None,
    triads_path: Optional[str | Path] = None,
    koll_metadata_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Reconstruct a patient-level feature matrix for a source/group using Stage 1 v6 functions."""
    features = list(dict.fromkeys([str(f) for f in features]))
    data_dict = stage1_mod.load_data_dict(
        feature_group=feature_group,
        feature_source=feature_source,
        panels=[panel],
        cohorts=[cohort],
        spatial_root=spatial_root,
        cell_features_path=cell_features_path,
        triads_path=triads_path,
    )
    harm_df = stage1_mod.load_harmonized_df(harmonized_path)
    kwargs = dict(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=qc_acceptability,
        min_epi_fraction=min_epi_fraction,
        sample_type=sample_type,
    )
    if koll_metadata_csv is not None:
        kwargs["koll_metadata_csv"] = koll_metadata_csv
    core_df = stage1_mod.prepare_core_level_feature_table(**kwargs)
    if core_df.empty:
        raise ValueError("No cores remain after requested filters.")
    core_df = stage1_mod.merge_harmonized_to_core_df(core_df, harm_df)
    core_df = stage1_mod.replace_with_harmonized_columns(core_df)
    core_df = stage1_mod.simplify_clinical_vars(core_df)
    core_df = stage1_mod.ensure_patient_id_column(core_df)

    present_features = [f for f in features if f in core_df.columns]
    if not present_features:
        raise ValueError("None of the requested features were found in core_df.")

    patient_df = stage1_mod.aggregate_core_to_patient(core_df, feature_cols=present_features, agg=agg)
    if "cohort" in patient_df.columns:
        patient_df = patient_df[patient_df["cohort"].astype(str) == str(cohort)].copy()

    if cohort in {"No-NAC", "KOLL"} and patient_subset in {"no_adj_chemo", "adj_chemo"}:
        patient_df = stage1_mod.apply_patient_subset(patient_df, patient_subset=patient_subset)
    elif patient_subset != "all":
        # Keep behavior explicit but do not silently drop all rows.
        log(f"[WARN] patient_subset={patient_subset} requested for {cohort}; using all because subset is only defined for No-NAC/KOLL.")

    if patient_df.empty:
        raise ValueError("No patients remain after aggregation/subsetting.")

    keep = [c for c in patient_df.columns if c not in present_features]
    return patient_df[keep + present_features].copy()


def build_context_feature_uid_matrix(
    *,
    ctx_manifest: pd.DataFrame,
    stage1_mod,
    config: Mapping,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one transformed patient matrix for a cohort/panel/sample_type/patient_subset/agg context.

    Returns
    -------
    matrix_df:
        patient_id + feature_uid columns, transformed according to the chosen
        selected_transform_mode for this matrix context.
    matrix_feature_meta:
        one row per feature_uid used in the matrix, including selected transform and source.
    """
    if ctx_manifest.empty:
        return pd.DataFrame(), pd.DataFrame()
    ctx_manifest = ctx_manifest.copy()
    chosen = _choose_context_feature_representations(ctx_manifest)

    first = chosen.iloc[0]
    cohort = str(first["cohort"])
    panel = str(first["panel"])
    sample_type = str(first.get("sample_type", config.get("sample_type", "TURBT")))
    patient_subset = str(first.get("patient_subset", config.get("patient_subset", "all")))
    agg = str(first.get("agg", config.get("agg", "median")))

    merged = None
    meta_rows = []
    failures = []

    for (feature_source, feature_group), g in chosen.groupby(["feature_source", "feature_group"], dropna=False):
        feature_source = str(feature_source)
        feature_group = str(feature_group)
        features = g["feature"].dropna().astype(str).unique().tolist()
        try:
            pdf = build_patient_matrix_for_source_group(
                stage1_mod=stage1_mod,
                cohort=cohort,
                panel=panel,
                feature_source=feature_source,
                feature_group=feature_group,
                features=features,
                sample_type=sample_type,
                patient_subset=patient_subset,
                agg=agg,
                qc_acceptability=str(config.get("qc_acceptability", "acceptable_or_borderline")),
                min_epi_fraction=config.get("min_epi_fraction", 0.05),
                harmonized_path=config["harmonized_path"],
                spatial_root=config.get("spatial_root"),
                cell_features_path=config.get("cell_features_path"),
                triads_path=config.get("triads_path"),
                koll_metadata_csv=config.get("koll_metadata_csv"),
            )
        except Exception as e:
            failures.append({
                "cohort": cohort, "panel": panel, "feature_source": feature_source,
                "feature_group": feature_group, "reason": f"{type(e).__name__}: {e}",
                "n_requested_features": len(features),
            })
            continue

        tmp = pdf[["patient_id"]].copy()
        for _, r in g.iterrows():
            feat = str(r["feature"])
            uid = str(r["feature_uid"])
            mode = str(r.get("selected_transform_mode", "zscore"))
            if feat not in pdf.columns:
                failures.append({
                    "cohort": cohort, "panel": panel, "feature_source": feature_source,
                    "feature_group": feature_group, "feature": feat, "feature_uid": uid,
                    "reason": "feature_missing_from_patient_matrix",
                })
                continue
            z = fit_apply_transform_full(pdf[feat], mode)
            tmp[uid] = z
            meta_rows.append({
                "cohort": cohort,
                "panel": panel,
                "sample_type": sample_type,
                "patient_subset": patient_subset,
                "agg": agg,
                "feature_uid": uid,
                "feature": feat,
                "feature_source": feature_source,
                "feature_group": feature_group,
                "selected_transform_mode": mode,
                "n_patients": int(z.shape[0]),
                "nonmissing_fraction": float(z.notna().mean()),
                "n_unique": int(z.dropna().nunique()),
            })

        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(tmp, on="patient_id", how="outer")

    matrix = merged if merged is not None else pd.DataFrame(columns=["patient_id"])
    meta = pd.DataFrame(meta_rows)
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = Path(config.get("output_root", ".")) / "matrix_build_failures.csv"
        if fail_path.exists():
            old = pd.read_csv(fail_path)
            fail_df = pd.concat([old, fail_df], ignore_index=True, sort=False)
        fail_df.to_csv(fail_path, index=False)
    return matrix, meta


def context_key_from_row(row: pd.Series) -> str:
    return "__".join(str(row.get(c, "NA")) for c in CONTEXT_MATRIX_COLS)



def _choose_global_feature_representations(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    One row per panel-level feature_uid for module discovery.

    Important: the global module atlas should correlate the same feature universe
    in every discovery cohort, not only the features that happened to be selected
    in that specific cohort. Otherwise, pair support mostly reflects co-selection
    rather than whether a feature pair is measurable and correlated across cohorts.

    If a feature_uid has different selected transforms in different contexts, keep
    the representation with the strongest candidate evidence.
    """
    work = manifest.copy()
    if "feature_uid" not in work.columns:
        work["feature_uid"] = make_feature_uid_series(work)
    if "candidate_score" not in work.columns:
        work["candidate_score"] = np.nan
    if "primary_oof_metric" not in work.columns:
        work["primary_oof_metric"] = np.nan
    if "selected_transform_mode" not in work.columns:
        work["selected_transform_mode"] = work.get("transform_mode", "zscore")

    sort_cols = ["candidate_score", "primary_oof_metric"]
    work = work.sort_values(sort_cols, ascending=[False, False], na_position="last")
    spec = work.drop_duplicates("feature_uid", keep="first").reset_index(drop=True)

    keep_cols = [
        "panel", "feature_uid", "feature", "feature_source", "feature_group",
        "selected_transform_mode", "candidate_score", "primary_oof_metric",
        "primary_delta_metric", "n_contexts", "n_cohorts"
    ]
    keep_cols = [c for c in keep_cols if c in spec.columns]
    return spec[keep_cols].copy()


def build_all_context_matrices(
    *,
    manifest: pd.DataFrame,
    stage1_mod,
    config: Mapping,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build and save transformed feature_uid matrices for all cohort/panel matrix contexts.

    v7.1 behavior:
      - Matrix contexts are still cohort x panel x sample_type x patient_subset x agg.
      - Within each panel, every context matrix is built for the *union* of candidate
        feature_uids selected anywhere in the discovery manifest for that panel.
      - This makes pair support reflect cross-cohort measurability/correlation rather
        than requiring the same feature pair to be selected in multiple contexts.
    """
    output_root = ensure_dir(config["output_root"])
    matrix_root = ensure_dir(output_root / "patient_matrices")
    feature_meta_parts = []
    context_rows = []
    matrices = {}

    if manifest.empty:
        raise ValueError("Manifest is empty; cannot build context matrices.")

    global_spec = _choose_global_feature_representations(manifest)
    global_spec.to_csv(matrix_root / "global_panel_feature_spec.csv", index=False)

    group_cols = [c for c in CONTEXT_MATRIX_COLS if c in manifest.columns]
    context_df = manifest[group_cols].drop_duplicates().copy()
    context_df = context_df.sort_values(group_cols).reset_index(drop=True)

    for _, ctx_row in context_df.iterrows():
        cohort = str(ctx_row["cohort"])
        panel = str(ctx_row["panel"])
        sample_type = str(ctx_row.get("sample_type", "TURBT"))
        patient_subset = str(ctx_row.get("patient_subset", "all"))
        agg = str(ctx_row.get("agg", "median"))
        ctx_id = context_key_from_row(ctx_row)

        panel_spec = global_spec[global_spec["panel"].astype(str) == panel].copy()
        if panel_spec.empty:
            log(f"[WARN] No global feature spec for panel={panel}; skipping {ctx_id}")
            continue

        # Inject context fields so build_context_feature_uid_matrix can use the
        # same source/group feature loading code.
        ctx_manifest = panel_spec.copy()
        ctx_manifest["cohort"] = cohort
        ctx_manifest["sample_type"] = sample_type
        ctx_manifest["patient_subset"] = patient_subset
        ctx_manifest["agg"] = agg

        expected_features = set(ctx_manifest["feature_uid"].astype(str))
        expected_n = len(expected_features)

        outdir = ensure_dir(matrix_root / panel)
        outpath = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}.parquet"
        metapath = outdir / f"{cohort}__{sample_type}__{patient_subset}__agg-{agg}__feature_meta.csv"

        can_load = False
        if outpath.exists() and metapath.exists() and not force:
            try:
                old_meta = pd.read_csv(metapath)
                old_features = set(old_meta.get("feature_uid", pd.Series(dtype=str)).astype(str))
                if expected_features.issubset(old_features):
                    can_load = True
                else:
                    log(
                        f"[REBUILD matrix] {ctx_id} existing matrix is narrow/stale: "
                        f"old_features={len(old_features)} expected={expected_n}"
                    )
            except Exception:
                log(f"[REBUILD matrix] {ctx_id} could not inspect existing meta")

        if can_load:
            log(f"[LOAD matrix] {outpath}")
            matrix = pd.read_parquet(outpath)
            meta = pd.read_csv(metapath)
        else:
            log(
                f"[BUILD matrix] {ctx_id} | global_panel_feature_uids={expected_n} "
                f"| source_groups={ctx_manifest[['feature_source','feature_group']].drop_duplicates().shape[0]}"
            )
            matrix, meta = build_context_feature_uid_matrix(
                ctx_manifest=ctx_manifest,
                stage1_mod=stage1_mod,
                config=config,
            )
            matrix.to_parquet(outpath, index=False)
            meta.to_csv(metapath, index=False)
            log(f"[SAVE matrix] {outpath} shape={matrix.shape}")

        matrices[ctx_id] = matrix
        feature_meta_parts.append(meta)
        context_rows.append({
            "context_matrix_id": ctx_id,
            "cohort": cohort,
            "panel": panel,
            "sample_type": sample_type,
            "patient_subset": patient_subset,
            "agg": agg,
            "path": str(outpath),
            "feature_meta_path": str(metapath),
            "n_patients": int(matrix.shape[0]),
            "n_feature_uid_columns": int(max(matrix.shape[1] - 1, 0)),
            "n_expected_panel_feature_uids": int(expected_n),
        })

    context_df_out = pd.DataFrame(context_rows)
    context_df_out.to_csv(matrix_root / "context_matrix_manifest.csv", index=False)
    feature_meta = pd.concat(feature_meta_parts, ignore_index=True, sort=False) if feature_meta_parts else pd.DataFrame()
    feature_meta.to_csv(matrix_root / "context_feature_meta.csv", index=False)
    return {"matrices": matrices, "context_matrix_manifest": context_df_out, "context_feature_meta": feature_meta}

# =============================================================================
# Correlation and consensus matrices
# =============================================================================

def compute_spearman_corr_from_matrix(matrix: pd.DataFrame, min_nonmissing_frac: float = 0.20) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    feature_cols = [c for c in matrix.columns if c != "patient_id"]
    if not feature_cols:
        return pd.DataFrame()
    X = matrix[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = []
    for c in X.columns:
        x = X[c]
        if x.notna().mean() >= min_nonmissing_frac and x.dropna().nunique() > 1:
            keep.append(c)
    X = X[keep]
    if X.shape[1] < 2:
        return pd.DataFrame(index=keep, columns=keep, dtype=float)
    corr = X.corr(method="spearman")
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return corr


def _average_corrs(corrs: Sequence[pd.DataFrame], all_features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feats = list(dict.fromkeys(all_features))
    sum_mat = pd.DataFrame(0.0, index=feats, columns=feats)
    n_mat = pd.DataFrame(0, index=feats, columns=feats, dtype=int)
    for corr in corrs:
        if corr is None or corr.empty:
            continue
        common = [f for f in corr.index if f in feats and f in corr.columns]
        if not common:
            continue
        C = corr.loc[common, common]
        mask = C.notna()
        sum_mat.loc[common, common] = sum_mat.loc[common, common].values + C.fillna(0).values
        n_mat.loc[common, common] = n_mat.loc[common, common].values + mask.astype(int).values
    avg = sum_mat / n_mat.replace(0, np.nan)
    for f in feats:
        avg.loc[f, f] = 1.0
    return avg, n_mat


def build_consensus_for_panel(
    *,
    panel: str,
    matrices: Mapping[str, pd.DataFrame],
    context_manifest: pd.DataFrame,
    min_nonmissing_frac: float = 0.20,
    consensus_level: str = "cohort",
) -> Dict[str, object]:
    """
    Build panel-level signed Spearman consensus.

    If consensus_level='cohort', multiple matrices from the same cohort are first
    averaged within cohort, then the final consensus is averaged across cohorts.
    This prevents TURBT+RC or multiple patient subsets from overweighting one cohort.
    """
    ctx_df = context_manifest.copy()
    ctx_df = ctx_df[ctx_df["panel"].astype(str) == str(panel)].copy()
    all_features = sorted(ctx_df["feature_uid"].dropna().astype(str).unique().tolist())
    context_corr_rows = []
    context_corrs = {}

    matrix_info = []
    for ctx_id, matrix in matrices.items():
        # parse panel/cohort from ctx id by matching context manifest rows
        parts = ctx_id.split("__")
        if len(parts) < 2:
            continue
        cohort = parts[0]
        p = parts[1]
        if p != panel:
            continue
        corr = compute_spearman_corr_from_matrix(matrix, min_nonmissing_frac=min_nonmissing_frac)
        context_corrs[ctx_id] = corr
        matrix_info.append({
            "context_matrix_id": ctx_id,
            "cohort": cohort,
            "panel": p,
            "n_patients": int(matrix.shape[0]),
            "n_features_matrix": int(max(matrix.shape[1] - 1, 0)),
            "n_features_corr": int(corr.shape[0]),
        })
        context_corr_rows.append(corr)

    if consensus_level == "cohort":
        cohort_corrs = {}
        for cohort in sorted(set([r["cohort"] for r in matrix_info])):
            cs = [context_corrs[r["context_matrix_id"]] for r in matrix_info if r["cohort"] == cohort]
            cohort_corr, cohort_support_contexts = _average_corrs(cs, all_features)
            cohort_corrs[cohort] = cohort_corr
        consensus, pair_support = _average_corrs(list(cohort_corrs.values()), all_features)
        support_unit = "cohort"
    else:
        consensus, pair_support = _average_corrs(context_corr_rows, all_features)
        cohort_corrs = {}
        support_unit = "context"

    return {
        "panel": panel,
        "consensus": consensus,
        "pair_support": pair_support,
        "context_corrs": context_corrs,
        "cohort_corrs": cohort_corrs,
        "matrix_qc": pd.DataFrame(matrix_info),
        "support_unit": support_unit,
    }


def apply_support_filter(
    consensus: pd.DataFrame,
    pair_support: pd.DataFrame,
    *,
    min_pair_support: int = 2,
    min_feature_support_frac: float = 0.10,
) -> Dict[str, pd.DataFrame]:
    feats = list(consensus.index)
    if not feats:
        return {"consensus_filtered": consensus.copy(), "pair_support_filtered": pair_support.copy(), "feature_support_summary": pd.DataFrame()}
    support_bool = pair_support >= int(min_pair_support)
    np.fill_diagonal(support_bool.values, True)

    n_possible = max(len(feats) - 1, 1)
    feature_support_frac = []
    for f in feats:
        vals = support_bool.loc[f, [x for x in feats if x != f]]
        feature_support_frac.append({
            "feature_uid": f,
            "n_supported_pairs": int(vals.sum()),
            "feature_support_fraction": float(vals.sum() / n_possible),
            "max_pair_support": int(pd.to_numeric(pair_support.loc[f], errors="coerce").max()),
            "median_pair_support": float(pd.to_numeric(pair_support.loc[f], errors="coerce").median()),
        })
    feature_support = pd.DataFrame(feature_support_frac)
    keep_features = feature_support.loc[feature_support["feature_support_fraction"] >= float(min_feature_support_frac), "feature_uid"].tolist()

    filt = consensus.loc[keep_features, keep_features].copy() if keep_features else pd.DataFrame()
    supp = pair_support.loc[keep_features, keep_features].copy() if keep_features else pd.DataFrame()
    if not filt.empty:
        mask = supp >= int(min_pair_support)
        filt = filt.where(mask, other=0.0)
        np.fill_diagonal(filt.values, 1.0)
    feature_support["passes_feature_support_filter"] = feature_support["feature_uid"].isin(keep_features)
    return {"consensus_filtered": filt, "pair_support_filtered": supp, "feature_support_summary": feature_support}

# =============================================================================
# Feature semantic parsing and ontology
# =============================================================================

CELL_ALIAS_MAP = {
    # tumor / stroma
    "tumor": "tumor_cell", "tumour": "tumor_cell", "tumor_cell": "tumor_cell", "cancer": "tumor_cell", "panck": "tumor_cell",
    "all_neg": "stromal_cell", "all neg": "stromal_cell", "all-negative": "stromal_cell", "stroma": "stromal_cell", "stromal": "stromal_cell", "fibro": "stromal_cell", "fibroblast": "stromal_cell",
    # t lineage
    "t_cell": "t_cell", "t cell": "t_cell", "t cells": "t_cell",
    "cd8": "cd8_t_cell", "cd8_t_cell": "cd8_t_cell", "cd8 t cell": "cd8_t_cell",
    "cd4": "cd4_t_cell", "cd4_t_cell": "cd4_t_cell", "cd4 t cell": "cd4_t_cell",
    "treg": "treg_cell", "treg_cell": "treg_cell", "foxp3": "treg_cell",
    # b / plasma / nk / myeloid
    "b_cell": "b_cell", "b cell": "b_cell", "b cells": "b_cell",
    "plasma": "plasma_cell", "plasma_cell": "plasma_cell", "plasma cell": "plasma_cell",
    "nk": "nk_cell", "nk_cell": "nk_cell", "nk cell": "nk_cell",
    "macrophage": "macrophage", "macrophages": "macrophage", "cd68": "macrophage",
    "immune": "immune_cell",
    # states
    "pd1": "PD1_state", "pdl1": "PDL1_state", "checkpoint_neg": "checkpoint_neg_state", "pd1_pdl1": "PD1_PDL1_state",
}

CELL_LINEAGE = {
    "cd8_t_cell": "T_lineage",
    "cd4_t_cell": "T_lineage",
    "t_cell": "T_lineage",
    "treg_cell": "T_lineage",
    "b_cell": "B_lineage",
    "plasma_cell": "B_lineage",
    "nk_cell": "NK_lineage",
    "macrophage": "myeloid_lineage",
    "tumor_cell": "tumor",
    "stromal_cell": "stroma",
    "immune_cell": "immune",
    "PD1_state": "checkpoint_state",
    "PDL1_state": "checkpoint_state",
    "PD1_PDL1_state": "checkpoint_state",
    "checkpoint_neg_state": "checkpoint_state",
}

TISSUE_PATTERNS = {
    "Tumor": [r"\bTumor\b", r"\bEpi\b", r"epithelial", r"tumou?r"],
    "Stroma": [r"\bStroma\b", r"\bStr\b", r"stromal"],
    "All": [r"\bAll\b", r"whole"],
}

METRIC_PATTERNS = {
    "NN": [r"_to_.*_(Mean|SD|Max|Min|Median|Q1|Q3)$"],
    "ATHENA_interaction": [r"^inter_(diff|z|p)_"],
    "ATHENA_infiltration": [r"^infiltration_"],
    "ATHENA_ripley": [r"^ripley_"],
    "ATHENA_diversity": [r"richness", r"shannon", r"renyi", r"entropy"],
    "composition_density": [r"__density__"],
    "composition_proportion": [r"__prop__"],
    "composition_ratio": [r"__ratio__"],
    "triad": [r"^\("],
}


def canon_cell_token(x: str) -> Optional[str]:
    s0 = str(x).strip()
    candidates = [s0, s0.replace("__", "_"), s0.replace("_", " ").replace("-", " ").lower()]
    for c in candidates:
        key = c.lower().strip()
        if key in CELL_ALIAS_MAP:
            return CELL_ALIAS_MAP[key]
    return None


def extract_cells_from_feature(feature: str) -> set:
    f = str(feature)
    # feature_uid support
    if "|" in f:
        _, _, f = split_feature_uid(f)

    # Triads: "('plasma cell', 't cell', 'treg cell') Str"
    if f.strip().startswith("("):
        try:
            tuple_part = f[: f.rfind(")") + 1]
            vals = ast.literal_eval(tuple_part)
            cells = set()
            for v in vals:
                c = canon_cell_token(str(v)) or canon_cell_token(str(v).replace(" ", "_"))
                if c:
                    cells.add(c)
            if cells:
                return cells
        except Exception:
            pass

    cells = set()
    # Common delimited cell pair patterns
    pair_patterns = [
        r"^(.+?)_to_(.+?)_(Mean|SD|Max|Min|Median|Q1|Q3)$",
        r"^inter_(diff|z|p)_(.+?)__(.+?)__(Tumor|Stroma|All)$",
        r"^(All|Epi|Stroma)__ratio__(.+)__over__(.+)$",
    ]
    for pat in pair_patterns:
        m = re.match(pat, f)
        if m:
            groups = list(m.groups())
            # Remove known metrics/regions
            for token in groups:
                if token in {"diff", "z", "p", "Mean", "SD", "Max", "Min", "Median", "Q1", "Q3", "Tumor", "Stroma", "All", "All", "Epi", "Stroma"}:
                    continue
                c = canon_cell_token(token)
                if c:
                    cells.add(c)
            if cells:
                return cells

    single_patterns = [
        r"^infiltration_(.+?)__(Tumor|Stroma|All)__(min|mean|median|max|pct_non_na)$",
        r"^ripley_(.+?)_(translation|border|ripley|none)_(.+?)__(Tumor|Stroma|All)__(at\d+|peak_abs|auc)$",
        r"^(All|Epi|Stroma)__prop__(.+)$",
        r"^(All|Epi|Stroma)__density__(.+)$",
    ]
    for pat in single_patterns:
        m = re.match(pat, f)
        if m:
            for token in m.groups():
                c = canon_cell_token(token)
                if c:
                    cells.add(c)
            if cells:
                return cells

    f_norm = f.replace("__", " ").replace("_", " ").replace("-", " ").lower()
    for alias, canon in CELL_ALIAS_MAP.items():
        alias_norm = alias.replace("_", " ").replace("-", " ").lower()
        if re.search(rf"(^|\s){re.escape(alias_norm)}($|\s)", f_norm):
            cells.add(canon)
    return cells


def extract_tissues_from_feature(feature: str) -> set:
    f = str(feature)
    if "|" in f:
        _, _, f = split_feature_uid(f)
    tissues = set()
    for label, pats in TISSUE_PATTERNS.items():
        if any(re.search(p, f, flags=re.IGNORECASE) for p in pats):
            tissues.add(label)
    return tissues


def extract_metric_families_from_feature(feature: str, feature_group: Optional[str] = None) -> set:
    f = str(feature)
    if "|" in f:
        _, fg, f0 = split_feature_uid(f)
        if feature_group is None:
            feature_group = fg
        f = f0
    fams = set()
    for label, pats in METRIC_PATTERNS.items():
        if any(re.search(p, f, flags=re.IGNORECASE) for p in pats):
            fams.add(label)
    if not fams and feature_group:
        fams.add(str(feature_group))
    return fams


def jaccard(a: set, b: set, empty_value: float = 0.0) -> float:
    a = set(a); b = set(b)
    if not a and not b:
        return empty_value
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_feature_ontology(features: Sequence[str], feature_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    schema = [
        "feature_uid", "feature_source", "feature_group", "feature",
        "cell_set", "tissue_set", "metric_family_set", "lineage_set",
        "dominant_lineage", "n_cells_parsed", "n_tissues_parsed",
        "n_metric_families_parsed",
    ]
    features = list(map(str, features))
    if len(features) == 0:
        return pd.DataFrame(columns=schema)
    meta_map = {}
    if feature_meta is not None and not feature_meta.empty and "feature_uid" in feature_meta.columns:
        meta_map = feature_meta.drop_duplicates("feature_uid").set_index("feature_uid").to_dict("index")
    rows = []
    for uid in features:
        uid = str(uid)
        fs, fg, feat = split_feature_uid(uid)
        if uid in meta_map:
            fs = str(meta_map[uid].get("feature_source", fs))
            fg = str(meta_map[uid].get("feature_group", fg))
            feat = str(meta_map[uid].get("feature", feat))
        cells = extract_cells_from_feature(uid)
        tissues = extract_tissues_from_feature(uid)
        metrics = extract_metric_families_from_feature(uid, feature_group=fg)
        lineages = {CELL_LINEAGE.get(c, c) for c in cells}
        rows.append({
            "feature_uid": uid,
            "feature_source": fs,
            "feature_group": fg,
            "feature": feat,
            "cell_set": ";".join(sorted(cells)),
            "tissue_set": ";".join(sorted(tissues)),
            "metric_family_set": ";".join(sorted(metrics)),
            "lineage_set": ";".join(sorted(lineages)),
            "dominant_lineage": sorted(lineages)[0] if lineages else "none",
            "n_cells_parsed": len(cells),
            "n_tissues_parsed": len(tissues),
            "n_metric_families_parsed": len(metrics),
        })
    return pd.DataFrame(rows, columns=schema)


def _set_from_semicolon(x: str) -> set:
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return set([v for v in str(x).split(";") if v])


def build_semantic_similarity_matrices(features: Sequence[str], ontology_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    features = list(map(str, features))
    if len(features) == 0:
        ontology_df = build_feature_ontology([])
        empty = pd.DataFrame(dtype=float)
        return {"primitive": empty.copy(), "lineage": empty.copy(), "tissue_aware": empty.copy(), "ontology": ontology_df}
    if ontology_df is None or ontology_df.empty or "feature_uid" not in ontology_df.columns:
        ontology_df = build_feature_ontology(features)
    ont = ontology_df.drop_duplicates("feature_uid").set_index("feature_uid")
    primitive = pd.DataFrame(0.0, index=features, columns=features)
    tissue_aware = pd.DataFrame(0.0, index=features, columns=features)
    lineage_sim = pd.DataFrame(0.0, index=features, columns=features)

    for f1 in features:
        c1 = _set_from_semicolon(ont.loc[f1, "cell_set"]) if f1 in ont.index else set()
        t1 = _set_from_semicolon(ont.loc[f1, "tissue_set"]) if f1 in ont.index else set()
        m1 = _set_from_semicolon(ont.loc[f1, "metric_family_set"]) if f1 in ont.index else set()
        l1 = _set_from_semicolon(ont.loc[f1, "lineage_set"]) if f1 in ont.index else set()
        for f2 in features:
            c2 = _set_from_semicolon(ont.loc[f2, "cell_set"]) if f2 in ont.index else set()
            t2 = _set_from_semicolon(ont.loc[f2, "tissue_set"]) if f2 in ont.index else set()
            m2 = _set_from_semicolon(ont.loc[f2, "metric_family_set"]) if f2 in ont.index else set()
            l2 = _set_from_semicolon(ont.loc[f2, "lineage_set"]) if f2 in ont.index else set()
            prim = jaccard(c1, c2, empty_value=0.0)
            lin = jaccard(l1, l2, empty_value=0.0)
            tis = jaccard(t1, t2, empty_value=0.0)
            met = jaccard(m1, m2, empty_value=0.0)
            primitive.loc[f1, f2] = prim
            lineage_sim.loc[f1, f2] = lin
            # tissue-aware ontology: simple, transparent weighted score
            tissue_aware.loc[f1, f2] = 0.45 * prim + 0.20 * lin + 0.20 * tis + 0.15 * met
    np.fill_diagonal(primitive.values, 1.0)
    np.fill_diagonal(lineage_sim.values, 1.0)
    np.fill_diagonal(tissue_aware.values, 1.0)
    return {"primitive": primitive, "lineage": lineage_sim, "tissue_aware": tissue_aware, "ontology": ontology_df}

# =============================================================================
# Clustering and k diagnostics
# =============================================================================

def build_distance_matrix(consensus_filtered: pd.DataFrame, mode: str = "row_spearman") -> pd.DataFrame:
    C = consensus_filtered.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if C.empty:
        return C
    if mode == "direct_signed":
        D = 1.0 - C
    elif mode == "direct_abs":
        D = 1.0 - C.abs()
    else:
        # Old-style behavior: cluster rows by their consensus-correlation pattern.
        row_corr = C.T.corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        D = 1.0 - row_corr
    D = D.clip(lower=0.0, upper=2.0)
    np.fill_diagonal(D.values, 0.0)
    return D


def linkage_from_distance(D: pd.DataFrame, method: str = "average"):
    if D.shape[0] < 2:
        return None
    arr = D.values.astype(float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    return linkage(squareform(arr, checks=False), method=method)


def _safe_silhouette(D: pd.DataFrame, labels: Sequence[int]) -> float:
    labels = np.asarray(labels)
    if D.shape[0] < 3 or len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return np.nan
    arr = D.values.astype(float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    try:
        return float(silhouette_score(arr, labels, metric="precomputed"))
    except Exception:
        return np.nan


def _cluster_size_stats(labels: Sequence[int]) -> dict:
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    n = len(labels)
    return {
        "n_clusters_observed": int(sizes.shape[0]),
        "median_cluster_size": float(sizes.median()) if not sizes.empty else np.nan,
        "mean_cluster_size": float(sizes.mean()) if not sizes.empty else np.nan,
        "max_cluster_size": int(sizes.max()) if not sizes.empty else 0,
        "max_cluster_fraction": float(sizes.max() / n) if n else np.nan,
        "singleton_fraction": float((sizes == 1).mean()) if not sizes.empty else np.nan,
        "n_singletons": int((sizes == 1).sum()) if not sizes.empty else 0,
    }


def evaluate_k_grid(
    *,
    consensus_filtered: pd.DataFrame,
    feature_meta: Optional[pd.DataFrame] = None,
    k_min: int = 5,
    k_max: int = 30,
    linkage_method: str = "average",
    distance_mode: str = "row_spearman",
) -> Dict[str, object]:
    features = consensus_filtered.index.astype(str).tolist()
    if len(features) < 2:
        ontology_df = build_feature_ontology(features, feature_meta=feature_meta)
        sims = build_semantic_similarity_matrices(features, ontology_df=ontology_df)
        return {
            "k_diagnostics": pd.DataFrame(),
            "memberships_all_k": pd.DataFrame(),
            "linkage": None,
            "distance": pd.DataFrame(index=features, columns=features, dtype=float),
            "ontology": ontology_df,
            "semantic_matrices": sims,
        }
    D_stat = build_distance_matrix(consensus_filtered, mode=distance_mode)
    Z = linkage_from_distance(D_stat, method=linkage_method)
    ontology_df = build_feature_ontology(features, feature_meta=feature_meta)
    sims = build_semantic_similarity_matrices(features, ontology_df=ontology_df)
    D_primitive = 1.0 - sims["primitive"]
    D_tissue = 1.0 - sims["tissue_aware"]
    D_lineage = 1.0 - sims["lineage"]
    for D in [D_primitive, D_tissue, D_lineage]:
        np.fill_diagonal(D.values, 0.0)

    rows = []
    membership_rows = []
    if Z is None:
        return {"k_diagnostics": pd.DataFrame(), "memberships_all_k": pd.DataFrame(), "linkage": Z, "distance": D_stat, "ontology": ontology_df, "semantic_matrices": sims}

    max_possible = max(2, min(int(k_max), len(features) - 1))
    for k in range(int(k_min), max_possible + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        stat_sil = _safe_silhouette(D_stat, labels)
        primitive_sil = _safe_silhouette(D_primitive.loc[features, features], labels)
        tissue_sil = _safe_silhouette(D_tissue.loc[features, features], labels)
        lineage_sil = _safe_silhouette(D_lineage.loc[features, features], labels)
        size_stats = _cluster_size_stats(labels)
        rows.append({
            "requested_k": int(k),
            "n_features": int(len(features)),
            "stat_silhouette": stat_sil,
            "primitive_semantic_silhouette": primitive_sil,
            "tissue_aware_semantic_silhouette": tissue_sil,
            "lineage_semantic_silhouette": lineage_sil,
            **size_stats,
        })
        for f, lab in zip(features, labels):
            membership_rows.append({"requested_k": int(k), "feature_uid": f, "raw_cluster_id": int(lab)})

    kdiag = pd.DataFrame(rows)
    # Rank-normalized composite makes metrics with different scales comparable.
    if not kdiag.empty:
        kdiag["stat_rank"] = rank01(kdiag["stat_silhouette"], higher_better=True)
        kdiag["primitive_rank"] = rank01(kdiag["primitive_semantic_silhouette"], higher_better=True)
        kdiag["tissue_aware_rank"] = rank01(kdiag["tissue_aware_semantic_silhouette"], higher_better=True)
        kdiag["lineage_rank"] = rank01(kdiag["lineage_semantic_silhouette"], higher_better=True)
        # Penalize giant clusters and excessive singletons, but gently.
        kdiag["cluster_balance_score"] = 1.0 - kdiag["max_cluster_fraction"].fillna(1.0)
        kdiag["non_singleton_score"] = 1.0 - kdiag["singleton_fraction"].fillna(1.0)
        kdiag["balance_rank"] = rank01(kdiag["cluster_balance_score"], higher_better=True)
        kdiag["non_singleton_rank"] = rank01(kdiag["non_singleton_score"], higher_better=True)
        kdiag["k_composite_score"] = (
            0.40 * kdiag["stat_rank"].fillna(0)
            + 0.20 * kdiag["primitive_rank"].fillna(0)
            + 0.25 * kdiag["tissue_aware_rank"].fillna(0)
            + 0.05 * kdiag["lineage_rank"].fillna(0)
            + 0.05 * kdiag["balance_rank"].fillna(0)
            + 0.05 * kdiag["non_singleton_rank"].fillna(0)
        )
        best = kdiag["k_composite_score"].max(skipna=True)
        kdiag["within_95pct_best_composite"] = kdiag["k_composite_score"] >= 0.95 * best if pd.notna(best) else False

    mem = pd.DataFrame(membership_rows)
    return {"k_diagnostics": kdiag, "memberships_all_k": mem, "linkage": Z, "distance": D_stat, "ontology": ontology_df, "semantic_matrices": sims}


def remap_modules_by_size(membership_df: pd.DataFrame, k: int) -> pd.DataFrame:
    m = membership_df[membership_df["requested_k"].astype(int) == int(k)].copy()
    if m.empty:
        raise ValueError(f"No membership rows for k={k}")
    size_order = m["raw_cluster_id"].value_counts().sort_values(ascending=False)
    remap = {old: i + 1 for i, old in enumerate(size_order.index)}
    m["module_num"] = m["raw_cluster_id"].map(remap).astype(int)
    m["module_id"] = m["module_num"].map(lambda x: f"M{int(x):02d}")
    return m.sort_values(["module_num", "feature_uid"]).reset_index(drop=True)


def summarize_modules(
    membership: pd.DataFrame,
    manifest: pd.DataFrame,
    consensus_filtered: Optional[pd.DataFrame] = None,
    pair_support_filtered: Optional[pd.DataFrame] = None,
    ontology_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    m = membership.copy()
    if "feature_uid" not in m.columns:
        raise ValueError("membership needs feature_uid")
    manifest_meta = (
        manifest.sort_values("candidate_score", ascending=False, na_position="last")
        .drop_duplicates("feature_uid")
        if "candidate_score" in manifest.columns else manifest.drop_duplicates("feature_uid")
    )
    join_cols = [c for c in ["feature_uid", "feature", "feature_source", "feature_group", "selected_transform_mode"] if c in manifest_meta.columns]
    m = m.merge(manifest_meta[join_cols], on="feature_uid", how="left")
    if ontology_df is None or ontology_df.empty:
        ontology_df = build_feature_ontology(m["feature_uid"].tolist(), feature_meta=m)
    ont = ontology_df.drop_duplicates("feature_uid")
    m = m.merge(ont[[c for c in ont.columns if c not in m.columns or c == "feature_uid"]], on="feature_uid", how="left")

    rows = []
    for module_num, g in m.groupby("module_num"):
        feats = g["feature_uid"].astype(str).tolist()
        mean_abs_corr = np.nan
        mean_pair_support = np.nan
        if consensus_filtered is not None and not consensus_filtered.empty:
            common = [f for f in feats if f in consensus_filtered.index]
            if len(common) > 1:
                sub = consensus_filtered.loc[common, common]
                vals = sub.where(~np.eye(sub.shape[0], dtype=bool)).stack().abs()
                mean_abs_corr = float(vals.mean()) if len(vals) else np.nan
        if pair_support_filtered is not None and not pair_support_filtered.empty:
            common = [f for f in feats if f in pair_support_filtered.index]
            if len(common) > 1:
                sub = pair_support_filtered.loc[common, common]
                vals = sub.where(~np.eye(sub.shape[0], dtype=bool)).stack()
                mean_pair_support = float(vals.mean()) if len(vals) else np.nan
        source_counts = g["feature_source"].value_counts(dropna=False).to_dict() if "feature_source" in g.columns else {}
        group_counts = g["feature_group"].value_counts(dropna=False).to_dict() if "feature_group" in g.columns else {}
        transform_col = "selected_transform_mode" if "selected_transform_mode" in g.columns else ("transform_mode" if "transform_mode" in g.columns else None)
        transform_counts = g[transform_col].value_counts(dropna=False).to_dict() if transform_col is not None else {}
        cell_counts = pd.Series(";".join(g.get("cell_set", pd.Series(dtype=str)).dropna().astype(str)).split(";")).replace("", np.nan).dropna().value_counts().to_dict()
        tissue_counts = pd.Series(";".join(g.get("tissue_set", pd.Series(dtype=str)).dropna().astype(str)).split(";")).replace("", np.nan).dropna().value_counts().to_dict()
        metric_counts = pd.Series(";".join(g.get("metric_family_set", pd.Series(dtype=str)).dropna().astype(str)).split(";")).replace("", np.nan).dropna().value_counts().to_dict()
        rows.append({
            "module_num": int(module_num),
            "module_id": f"M{int(module_num):02d}",
            "n_features": int(g["feature_uid"].nunique()),
            "feature_source_counts": format_counts(source_counts),
            "feature_group_counts": format_counts(group_counts),
            "transform_mode_counts": format_counts(transform_counts),
            "cell_type_counts": format_counts(cell_counts),
            "tissue_counts": format_counts(tissue_counts),
            "metric_family_counts": format_counts(metric_counts),
            "dominant_feature_source": max(source_counts, key=source_counts.get) if source_counts else "",
            "dominant_feature_group": max(group_counts, key=group_counts.get) if group_counts else "",
            "dominant_cell_type": max(cell_counts, key=cell_counts.get) if cell_counts else "",
            "dominant_tissue": max(tissue_counts, key=tissue_counts.get) if tissue_counts else "",
            "dominant_metric_family": max(metric_counts, key=metric_counts.get) if metric_counts else "",
            "mean_abs_consensus_corr": mean_abs_corr,
            "mean_pair_support": mean_pair_support,
            "features": " | ".join(feats),
        })
    return pd.DataFrame(rows).sort_values("module_num")


def format_counts(counts: Mapping) -> str:
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return "; ".join([f"{k}:{int(v)}" for k, v in items])


def choose_module_representatives(membership: pd.DataFrame, manifest: pd.DataFrame, feature_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    m = membership.copy()
    score_cols = ["candidate_score", "primary_oof_metric", "primary_delta_metric"]
    meta_cols = ["feature_uid", "feature", "feature_source", "feature_group", "selected_transform_mode"] + score_cols
    meta_cols = [c for c in meta_cols if c in manifest.columns]
    mm = manifest[meta_cols].copy()
    if "candidate_score" not in mm.columns:
        mm["candidate_score"] = np.nan
    agg = (
        mm.groupby([c for c in ["feature_uid", "feature", "feature_source", "feature_group", "selected_transform_mode"] if c in mm.columns], dropna=False)
        .agg(
            max_candidate_score=("candidate_score", "max"),
            median_candidate_score=("candidate_score", "median"),
            max_oof_metric=("primary_oof_metric", "max") if "primary_oof_metric" in mm.columns else ("candidate_score", "max"),
            max_delta_metric=("primary_delta_metric", "max") if "primary_delta_metric" in mm.columns else ("candidate_score", "max"),
        )
        .reset_index()
    )
    m = m.merge(agg, on="feature_uid", how="left")
    rows = []
    for module_num, g in m.groupby("module_num"):
        rep = g.sort_values(["max_candidate_score", "median_candidate_score", "max_oof_metric"], ascending=[False, False, False], na_position="last").iloc[0]
        rows.append({
            "module_num": int(module_num),
            "module_id": f"M{int(module_num):02d}",
            "representative_feature_uid": rep["feature_uid"],
            "representative_feature": rep.get("feature", split_feature_uid(rep["feature_uid"])[2]),
            "representative_feature_source": rep.get("feature_source", split_feature_uid(rep["feature_uid"])[0]),
            "representative_feature_group": rep.get("feature_group", split_feature_uid(rep["feature_uid"])[1]),
            "representative_transform_mode": rep.get("selected_transform_mode", "zscore"),
            "representative_max_candidate_score": rep.get("max_candidate_score", np.nan),
            "module_n_features": int(g["feature_uid"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("module_num")

# =============================================================================
# Plotting
# =============================================================================

def plot_candidate_composition(manifest: pd.DataFrame, output_dir: str | Path) -> None:
    if manifest.empty:
        return
    output_dir = ensure_dir(output_dir)
    counts = manifest.groupby(["panel", "feature_source"], dropna=False)["feature_uid"].nunique().reset_index(name="n_features")
    for panel, g in counts.groupby("panel"):
        fig, ax = plt.subplots(figsize=(8, 4))
        g = g.sort_values("n_features", ascending=False)
        ax.bar(g["feature_source"].astype(str), g["n_features"])
        ax.set_title(f"{panel}: selected candidates by feature source")
        ax.set_ylabel("Unique feature_uid")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        save_fig(fig, output_dir / f"{panel}_candidate_counts_by_feature_source.png")


def plot_support_diagnostics(feature_support: pd.DataFrame, output_dir: str | Path, panel: str) -> None:
    if feature_support.empty:
        return
    output_dir = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(feature_support["feature_support_fraction"].dropna(), bins=30)
    ax.set_xlabel("Feature support fraction")
    ax.set_ylabel("Number of features")
    ax.set_title(f"{panel}: feature support filtering")
    fig.tight_layout()
    save_fig(fig, output_dir / f"{panel}_feature_support_fraction_hist.png")


def plot_k_diagnostics(kdiag: pd.DataFrame, output_dir: str | Path, panel: str) -> None:
    if kdiag.empty:
        return
    output_dir = ensure_dir(output_dir)
    metrics = [
        "stat_silhouette",
        "primitive_semantic_silhouette",
        "tissue_aware_semantic_silhouette",
        "k_composite_score",
        "max_cluster_fraction",
        "singleton_fraction",
    ]
    for metric in metrics:
        if metric not in kdiag.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(kdiag["requested_k"], kdiag[metric], marker="o")
        ax.set_xlabel("Requested k")
        ax.set_ylabel(metric)
        ax.set_title(f"{panel}: {metric}")
        fig.tight_layout()
        save_fig(fig, output_dir / f"{panel}_kdiag_{metric}.png")

    # Compact combined plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["stat_silhouette", "primitive_semantic_silhouette", "tissue_aware_semantic_silhouette", "k_composite_score"]:
        if metric in kdiag.columns:
            vals = pd.to_numeric(kdiag[metric], errors="coerce")
            if vals.notna().any():
                ax.plot(kdiag["requested_k"], vals, marker="o", label=metric)
    ax.set_xlabel("Requested k")
    ax.set_ylabel("Score")
    ax.set_title(f"{panel}: k-selection diagnostics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, output_dir / f"{panel}_k_selection_combined.png")


def plot_consensus_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str,
    membership: Optional[pd.DataFrame] = None,
    feature_meta: Optional[pd.DataFrame] = None,
    figsize_scale: float = 0.12,
    max_figsize: int = 28,
) -> None:
    if matrix.empty or sns is None:
        return
    M = matrix.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = M.shape[0]
    if n < 2:
        return
    row_colors = None
    if membership is not None and not membership.empty and "feature_uid" in membership.columns:
        mem = membership.drop_duplicates("feature_uid").set_index("feature_uid")
        if "module_num" in mem.columns:
            modules = mem.reindex(M.index)["module_num"].fillna(0).astype(int)
            palette = sns.color_palette("tab20", n_colors=max(int(modules.max()), 1) + 1)
            row_colors = modules.map(lambda x: palette[int(x) % len(palette)])
    size = min(max(8, figsize_scale * n), max_figsize)
    try:
        g = sns.clustermap(
            M,
            cmap="vlag",
            center=0,
            figsize=(size, size),
            row_colors=row_colors,
            xticklabels=False,
            yticklabels=False,
            dendrogram_ratio=(0.10, 0.10),
            cbar_pos=(0.02, 0.82, 0.03, 0.12),
        )
        g.fig.suptitle(title, y=1.02)
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        g.fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(g.fig)
        log(f"[SAVE] {output_path}")
    except Exception as e:
        log(f"[WARN] heatmap failed for {output_path}: {type(e).__name__}: {e}")


def plot_module_composition(module_summary: pd.DataFrame, output_dir: str | Path, panel: str) -> None:
    if module_summary.empty:
        return
    output_dir = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * module_summary.shape[0])))
    d = module_summary.sort_values("module_num")
    ax.barh(d["module_id"], d["n_features"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of features")
    ax.set_title(f"{panel}: final module sizes")
    fig.tight_layout()
    save_fig(fig, output_dir / f"{panel}_final_module_sizes.png")

# =============================================================================
# Saving/loading prepared outputs and finalization
# =============================================================================

def save_panel_prepared_outputs(
    *,
    panel: str,
    output_root: str | Path,
    consensus: pd.DataFrame,
    pair_support: pd.DataFrame,
    consensus_filtered: pd.DataFrame,
    pair_support_filtered: pd.DataFrame,
    feature_support_summary: pd.DataFrame,
    k_results: Dict[str, object],
    manifest_panel: pd.DataFrame,
    matrix_qc: pd.DataFrame,
) -> None:
    panel_dir = ensure_dir(Path(output_root) / panel)
    consensus.to_parquet(panel_dir / f"{panel}_consensus_similarity.parquet")
    pair_support.to_parquet(panel_dir / f"{panel}_pair_support.parquet")
    consensus_filtered.to_parquet(panel_dir / f"{panel}_consensus_similarity_support_filtered.parquet")
    pair_support_filtered.to_parquet(panel_dir / f"{panel}_pair_support_filtered.parquet")
    feature_support_summary.to_csv(panel_dir / f"{panel}_feature_support_summary.csv", index=False)
    k_results["k_diagnostics"].to_csv(panel_dir / f"{panel}_k_selection_diagnostics.csv", index=False)
    k_results["memberships_all_k"].to_csv(panel_dir / f"{panel}_memberships_all_k.csv", index=False)
    k_results["ontology"].to_csv(panel_dir / f"{panel}_feature_ontology.csv", index=False)
    manifest_panel.to_csv(panel_dir / f"{panel}_manifest_used.csv", index=False)
    matrix_qc.to_csv(panel_dir / f"{panel}_matrix_qc.csv", index=False)
    # Save distance matrix used for clustering.
    if isinstance(k_results.get("distance"), pd.DataFrame):
        k_results["distance"].to_parquet(panel_dir / f"{panel}_clustering_distance.parquet")


def load_prepared_panel(output_root: str | Path, panel: str) -> Dict[str, object]:
    panel_dir = Path(output_root) / panel
    out = {
        "panel_dir": panel_dir,
        "consensus": pd.read_parquet(panel_dir / f"{panel}_consensus_similarity.parquet"),
        "pair_support": pd.read_parquet(panel_dir / f"{panel}_pair_support.parquet"),
        "consensus_filtered": pd.read_parquet(panel_dir / f"{panel}_consensus_similarity_support_filtered.parquet"),
        "pair_support_filtered": pd.read_parquet(panel_dir / f"{panel}_pair_support_filtered.parquet"),
        "feature_support_summary": pd.read_csv(panel_dir / f"{panel}_feature_support_summary.csv"),
        "k_diagnostics": pd.read_csv(panel_dir / f"{panel}_k_selection_diagnostics.csv"),
        "memberships_all_k": pd.read_csv(panel_dir / f"{panel}_memberships_all_k.csv"),
        "ontology": pd.read_csv(panel_dir / f"{panel}_feature_ontology.csv"),
        "manifest_used": pd.read_csv(panel_dir / f"{panel}_manifest_used.csv"),
        "matrix_qc": pd.read_csv(panel_dir / f"{panel}_matrix_qc.csv"),
    }
    return out


def finalize_panel_modules(output_root: str | Path, panel: str, final_k: int) -> Dict[str, pd.DataFrame]:
    prep = load_prepared_panel(output_root, panel)
    membership = remap_modules_by_size(prep["memberships_all_k"], final_k)
    manifest = prep["manifest_used"]
    ontology = prep["ontology"]

    # Merge display/source metadata and selected transform for old evaluation script compatibility.
    meta = (
        manifest.sort_values("candidate_score", ascending=False, na_position="last")
        .drop_duplicates("feature_uid")
    )
    keep = [c for c in ["feature_uid", "feature", "feature_source", "feature_group", "selected_transform_mode", "candidate_score", "primary_oof_metric", "primary_delta_metric"] if c in meta.columns]
    membership = membership.merge(meta[keep], on="feature_uid", how="left")
    membership = membership.rename(columns={"selected_transform_mode": "transform_mode"})
    membership["requested_k"] = int(final_k)
    # Old scripts often expect feature as the measurable column and requested_k/module_num.
    membership["module_label"] = membership["module_id"]

    module_summary = summarize_modules(
        membership=membership,
        manifest=manifest,
        consensus_filtered=prep["consensus_filtered"],
        pair_support_filtered=prep["pair_support_filtered"],
        ontology_df=ontology,
    )
    reps = choose_module_representatives(membership, manifest)

    final_dir = ensure_dir(Path(output_root) / "final_modules" / panel)
    membership.to_csv(final_dir / f"{panel}_global_module_memberships_k{final_k}.csv", index=False)
    # Also save an evaluation-script-friendly generic name.
    membership.to_csv(final_dir / f"{panel}_dendrogram_k_memberships.csv", index=False)
    module_summary.to_csv(final_dir / f"{panel}_module_summary_k{final_k}.csv", index=False)
    reps.to_csv(final_dir / f"{panel}_module_representatives_k{final_k}.csv", index=False)

    plot_consensus_heatmap(
        prep["consensus_filtered"],
        final_dir / f"{panel}_consensus_heatmap_final_k{final_k}.png",
        title=f"{panel} consensus similarity | final k={final_k}",
        membership=membership,
        feature_meta=manifest,
    )
    plot_module_composition(module_summary, final_dir, panel)
    log(f"[DONE final] {panel} k={final_k} -> {final_dir}")
    return {"membership": membership, "module_summary": module_summary, "representatives": reps}


def make_k_shortlist(kdiag: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if kdiag.empty:
        return kdiag
    cols = [
        "requested_k", "k_composite_score", "stat_silhouette", "primitive_semantic_silhouette",
        "tissue_aware_semantic_silhouette", "max_cluster_fraction", "singleton_fraction",
        "median_cluster_size", "within_95pct_best_composite",
    ]
    cols = [c for c in cols if c in kdiag.columns]
    out = kdiag.sort_values("k_composite_score", ascending=False, na_position="last")[cols].head(top_n).copy()
    return out
