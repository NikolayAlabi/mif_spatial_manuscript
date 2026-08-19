#!/usr/bin/env python
"""
stage1_univariate_cv_screen.py

Stage-1 univariate screening for mIF biomarkers.

Design goals
------------
1. Load biomarker tables + lightweight metadata/QC/tissue segmentation.
2. Merge core-level biomarkers to harmonized patient-level clinical data.
3. Apply core-level filters:
   - cohort
   - panel
   - QC acceptability
   - minimum epithelial fraction
4. Aggregate cores to the patient level.
5. Run robust cross-validated screening for:
   - response endpoints (complete_response, any_response)
   - survival endpoints (OS, RFS)
6. Save:
   - fold-level metrics
   - pooled out-of-fold metrics
   - patient-level OOF predictions
   - skips / failures / reasons

This script is intended to be parallelized by submitting separate jobs for:
    cohort x panel x feature_group x feature_chunk x endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import warnings
from scipy.stats import chi2
import statsmodels.api as sm

from mif_dataframe_builders_v2 import (
    extract_coord_token,
    load_tma_metadata,
    load_tma_qc,
    load_tma_tissue_segmentation,
    load_blasst_metadata,
    load_blasst_tissue_segmentation,
    add_blasst_sample_type_to_dataframes,
)

RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
MIN_PATIENTS_RESPONSE = 20
MIN_POS_PER_CLASS = 5
MIN_PATIENTS_SURV = 10
MIN_EVENTS_SURV = 5
CLINICAL_MISSING_THRESHOLD = 0.50
DEFAULT_CLINICAL_VARS = ["cN", "cT", "Age", "Sex"]

import re

BLASST_UNDERSCORE_COORD_RE = re.compile(r"^\s*'?(\d+)\s*_\s*(\d+)'?\s*$")
BRACKET_COORD_RE = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")

from glob import glob

TRIAD_FILES = [
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_AR_3_collapse_label_blasst.csv",
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_AR_3_collapse_label_tma.csv",
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_BT_3_collapse_label_blasst.csv",
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_BT_3_collapse_label_tma.csv",
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_MY_3_collapse_label_blasst.csv",
    "/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables/go_MY_3_collapse_label_tma.csv",
]

CORE_COORD_RE = re.compile(r"_\[\s*(\d+)\s*,\s*(\d+)\s*\](?:\.im3)?\s*$")


def extract_coord_from_core_name(core_val) -> str | pd.NA:
    if pd.isna(core_val):
        return pd.NA
    s = str(core_val).strip()
    m = CORE_COORD_RE.search(s)
    if not m:
        return pd.NA
    return f"[{m.group(1)},{m.group(2)}]"


def load_triads_table(filepaths: Sequence[str]) -> pd.DataFrame:
    parts = []

    for fp in filepaths:
        df = pd.read_csv(fp, low_memory=False).copy()
        base = Path(fp).name

        # infer panel from filename
        m_panel = re.search(r"go_(AR|BT|MY)_3_", base)
        if not m_panel:
            raise ValueError(f"Could not infer panel from triad filename: {base}")
        panel = m_panel.group(1)

        # infer broad source from filename
        source_kind = "BLASST" if "blasst" in base.lower() else "TMA"

        # standardize the identifier column
        if "Core" not in df.columns:
            raise ValueError(f"Triad file missing 'Core' column: {fp}")

        df = df.rename(columns={"Core": "sample_name"})
        df["coord"] = df["sample_name"].map(extract_coord_from_core_name)
        df["Panel"] = panel

        # keep a coarse source tag; cohort itself will come from metadata merge
        df["triad_source"] = source_kind

        parts.append(df)

    triads = pd.concat(parts, ignore_index=True)

    # fail early if coord extraction broke
    n_missing = triads["coord"].isna().sum()
    if n_missing > 0:
        bad = triads.loc[triads["coord"].isna(), "sample_name"].astype(str).head(10).tolist()
        raise ValueError(
            f"Triads table has {n_missing} rows with missing coord after parsing Core. "
            f"Examples: {bad}"
        )

    return triads

def normalize_yes_no_series(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()

    yes_vals = {"1", "1.0", "yes", "y", "true", "t"}
    no_vals  = {"0", "0.0", "no", "n", "false", "f"}

    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[x.isin(yes_vals)] = "yes"
    out[x.isin(no_vals)] = "no"

    return out

def normalize_sample_type(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper()

    mapping = {
        "TURBT": "TURBT",
        "RC": "RC",
        "CYSTECTOMY": "RC",
        "RADICAL CYSTECTOMY": "RC",
    }
    return mapping.get(s, s)


def apply_sample_type_filter(core_df: pd.DataFrame, sample_type: str = "all") -> pd.DataFrame:
    out = core_df.copy()

    if sample_type == "all":
        return out

    if sample_type not in {"TURBT", "RC"}:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    if "TURBT_or_RC" not in out.columns:
        raise ValueError("Requested sample_type filtering but 'TURBT_or_RC' column is missing.")

    samp = out["TURBT_or_RC"].map(normalize_sample_type)

    out = out[samp == sample_type].copy()
    return out

def apply_patient_subset(patient_df: pd.DataFrame, patient_subset: str) -> pd.DataFrame:
    out = patient_df.copy()

    if patient_subset == "all":
        return out

    if patient_subset not in {"no_adj_chemo", "adj_chemo"}:
        raise ValueError(f"Unknown patient_subset: {patient_subset}")

    if "adjuvant_chemo" not in out.columns:
        raise ValueError("Requested patient subset requires 'adjuvant_chemo', but column is missing.")

    adj = normalize_yes_no_series(out["adjuvant_chemo"])

    if patient_subset == "no_adj_chemo":
        out = out[adj == "no"].copy()
    elif patient_subset == "adj_chemo":
        out = out[adj == "yes"].copy()

    return out

def standardize_coord_series(
    series: pd.Series,
    *,
    allow_blasst_underscore: bool = True,
) -> pd.Series:
    """
    Standardize coord values to bracket format: [x,y]

    Handles examples like:
      [47334,8859]
      [ 47334 , 8859 ]
      '55560_10933'
      55560_10933

    Returns
    -------
    pd.Series
        Standardized coord strings in the form [x,y], or NA if unparsable.
    """
    out = pd.Series(pd.NA, index=series.index, dtype="object")

    s = series.astype(str).str.strip()
    s = s.replace({
        "<NA>": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "": pd.NA,
    })

    # First try already-bracketed coords
    m_bracket = s.str.extract(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")
    ok_bracket = m_bracket[0].notna() & m_bracket[1].notna()
    out.loc[ok_bracket] = (
        "[" + m_bracket.loc[ok_bracket, 0].astype(str) + "," + m_bracket.loc[ok_bracket, 1].astype(str) + "]"
    )

    # Then try BLASST underscore coords like 55560_10933
    if allow_blasst_underscore:
        m_us = s.str.extract(r"^\s*'?\s*(\d+)\s*_\s*(\d+)\s*'?\s*$")
        ok_us = m_us[0].notna() & m_us[1].notna() & out.isna()
        out.loc[ok_us] = (
            "[" + m_us.loc[ok_us, 0].astype(str) + "," + m_us.loc[ok_us, 1].astype(str) + "]"
        )

    return out

def standardize_and_validate_all_coords(
    data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Apply coord standardization/validation to every dataframe in data_dict
    that contains a coord column.
    """
    out = dict(data_dict)

    keys_to_check = [
        "ratios",
        "cell_features",
        "athena",
        "NN",
        "triads",
        "qc_df",
        "meta_df",
        "tissue_seg_df",
        "tma_meta_df",
        "tma_qc_df",
        "blasst_meta_df",
        "blasst_tissue_seg_df",
    ]

    for key in keys_to_check:
        if key in out and out[key] is not None:
            out[key] = standardize_and_validate_coord_df(
                out[key],
                df_name=key,
                fail_on_missing=True,
                drop_missing=False,
            )

    return out

def standardize_and_validate_coord_df(
    df: Optional[pd.DataFrame],
    df_name: str,
    *,
    cohort_col: str = "cohort",
    coord_col: str = "coord",
    fail_on_missing: bool = True,
    drop_missing: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Standardize coord format within a dataframe and validate completeness.

    Important BLASST behavior:
    - If cohort == BLASST and coord is underscore-style, convert to [x,y]
    - For all cohorts, bracket-style coords are normalized to [x,y]

    Parameters
    ----------
    df : DataFrame or None
    df_name : str
        Name used in error/warning messages.
    fail_on_missing : bool
        If True, raise an error when coord is missing after standardization.
    drop_missing : bool
        If True, drop rows with missing/unparseable coord instead of erroring.

    Returns
    -------
    DataFrame or None
    """
    if df is None:
        return None

    out = df.copy()

    if coord_col not in out.columns:
        raise ValueError(f"{df_name} is missing required column '{coord_col}'.")

    # Standardize all TMA/BLASST coords, including BLASST underscore form.
    # KOLL does not use [x,y] TMA coordinate tokens; preserve its sample/core id
    # as coord if bracket/underscore parsing fails.
    original_coord = out[coord_col].copy()
    standardized_coord = standardize_coord_series(original_coord)
    if cohort_col in out.columns:
        is_koll = out[cohort_col].astype(str).str.strip().eq("KOLL")
        m = is_koll & standardized_coord.isna() & original_coord.notna()
        standardized_coord.loc[m] = original_coord.loc[m].astype(str).str.strip()
    out[coord_col] = standardized_coord

    n_missing = out[coord_col].isna().sum()

    if n_missing > 0:
        msg = (
            f"{df_name} has {n_missing} rows with missing/unparseable coord "
            f"after standardization."
        )

        if drop_missing:
            print(f"[WARN] {msg} Dropping those rows.", flush=True)
            out = out[out[coord_col].notna()].copy()
        elif fail_on_missing:
            bad_examples = df.loc[out[coord_col].isna(), coord_col].astype(str).head(10).tolist()
            raise ValueError(f"{msg} Example bad values: {bad_examples}")
        else:
            print(f"[WARN] {msg}", flush=True)

    return out

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_coord_bracket(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    out = out.str.replace(r"\[\s*", "[", regex=True)
    out = out.str.replace(r"\s*,\s*", ",", regex=True)
    out = out.str.replace(r"\s*\]", "]", regex=True)
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def collapse_duplicate_rows(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    value_cols = [c for c in df.columns if c not in key_cols]
    if not value_cols:
        return df.drop_duplicates(list(key_cols)).copy()
    out = (
        df.groupby(list(key_cols), dropna=False)[value_cols]
        .agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
        .reset_index()
    )
    return out


def safe_mode(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def pick_threshold_from_train(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.clip(y_prob, 0, 1))
    if len(thresholds) == 0:
        return 0.5
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        y_hat = (y_prob >= thr).astype(int)
        try:
            score = balanced_accuracy_score(y_true, y_hat)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def sens_spec_from_preds(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    except Exception:
        return np.nan, np.nan
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec


def pooled_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {
        "oof_auc": np.nan,
        "oof_auprc": np.nan,
        "oof_accuracy": np.nan,
        "oof_balanced_accuracy": np.nan,
        "oof_sensitivity": np.nan,
        "oof_specificity": np.nan,
    }
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return out
    try:
        out["oof_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        pass
    try:
        out["oof_auprc"] = average_precision_score(y_true, y_prob)
    except Exception:
        pass
    try:
        thr = pick_threshold_from_train(y_true, y_prob)
        y_hat = (y_prob >= thr).astype(int)
        out["oof_accuracy"] = accuracy_score(y_true, y_hat)
        out["oof_balanced_accuracy"] = balanced_accuracy_score(y_true, y_hat)
        sens, spec = sens_spec_from_preds(y_true, y_hat)
        out["oof_sensitivity"] = sens
        out["oof_specificity"] = spec
    except Exception:
        pass
    return out

# -----------------------------------------------------------------------------
# Full-dataset inferential layer + transform helpers
# -----------------------------------------------------------------------------

TRANSFORM_MODES = ("raw", "zscore", "log1p_zscore")


def _is_binary_series(s: pd.Series) -> bool:
    x = safe_numeric(s).dropna().unique()
    return set(x).issubset({0, 1}) and len(x) >= 1


def _fit_zscore_params(series: pd.Series) -> Tuple[float, float]:
    x = safe_numeric(series)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        raise ValueError("Feature has zero variance.")
    return float(mu), float(sd)


def _apply_transform_to_feature(
    series: pd.Series,
    transform_mode: str,
) -> pd.Series:
    """
    Full-dataset transform for inferential/descriptive models.

    Modes
    -----
    raw
    zscore
    log1p_zscore   # requires feature >= 0
    """
    x = safe_numeric(series).copy()

    if transform_mode == "raw":
        return x

    if transform_mode == "zscore":
        mu, sd = _fit_zscore_params(x)
        return (x - mu) / sd

    if transform_mode == "log1p_zscore":
        if (x.dropna() < 0).any():
            raise ValueError("log1p_zscore requested but feature has negative values.")
        x = np.log1p(x)
        mu, sd = _fit_zscore_params(x)
        return (x - mu) / sd

    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def _transform_train_valid_feature(
    train_s: pd.Series,
    valid_s: pd.Series,
    transform_mode: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Leakage-safe transform for CV.

    IMPORTANT:
    - fit transform parameters on training fold only
    - apply same parameters to validation fold
    """
    xtr = safe_numeric(train_s).copy()
    xva = safe_numeric(valid_s).copy()

    if transform_mode == "raw":
        return xtr, xva

    if transform_mode == "zscore":
        mu, sd = _fit_zscore_params(xtr)
        return (xtr - mu) / sd, (xva - mu) / sd

    if transform_mode == "log1p_zscore":
        if (xtr.dropna() < 0).any():
            raise ValueError("Training fold has negative values; cannot use log1p_zscore.")
        if (xva.dropna() < 0).any():
            raise ValueError("Validation fold has negative values; cannot use log1p_zscore.")
        xtr = np.log1p(xtr)
        xva = np.log1p(xva)
        mu, sd = _fit_zscore_params(xtr)
        return (xtr - mu) / sd, (xva - mu) / sd

    raise ValueError(f"Unknown transform_mode: {transform_mode}")


def _make_design_matrix(
    df: pd.DataFrame,
    feature_name: Optional[str],
    clinical_covars: Sequence[str],
    transform_mode: str = "raw",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build model matrix for full-dataset inferential layer.

    Returns
    -------
    X : DataFrame
    feature_cols_used : list[str]
        includes feature_name if present
    """
    parts = []

    keep_covars = [c for c in clinical_covars if c in df.columns]
    if keep_covars:
        X_clin = pd.get_dummies(df[keep_covars].copy(), drop_first=True)
        if X_clin.shape[1] > 0:
            keep = X_clin.nunique(dropna=False) > 1
            X_clin = X_clin.loc[:, keep]
        parts.append(X_clin)

    feature_cols_used = []
    if feature_name is not None:
        x = _apply_transform_to_feature(df[feature_name], transform_mode=transform_mode)
        X_feat = pd.DataFrame({feature_name: x}, index=df.index)
        parts.append(X_feat)
        feature_cols_used.append(feature_name)

    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, feature_cols_used


def _make_design_matrix_train_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_name: Optional[str],
    clinical_covars: Sequence[str],
    transform_mode: str = "raw",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe train/valid design matrices for CV.
    """
    keep_covars = [c for c in clinical_covars if c in train_df.columns and c in valid_df.columns]

    Xc_tr = pd.DataFrame(index=train_df.index)
    Xc_va = pd.DataFrame(index=valid_df.index)
    if keep_covars:
        Xc_tr = pd.get_dummies(train_df[keep_covars].copy(), drop_first=True)
        Xc_va = pd.get_dummies(valid_df[keep_covars].copy(), drop_first=True)
        Xc_va = Xc_va.reindex(columns=Xc_tr.columns, fill_value=0)

        if Xc_tr.shape[1] > 0:
            keep = Xc_tr.nunique(dropna=False) > 1
            Xc_tr = Xc_tr.loc[:, keep]
            Xc_va = Xc_va.loc[:, Xc_tr.columns]

    parts_tr = [Xc_tr] if Xc_tr.shape[1] > 0 else []
    parts_va = [Xc_va] if Xc_va.shape[1] > 0 else []

    if feature_name is not None:
        ftr_tr, ftr_va = _transform_train_valid_feature(
            train_df[feature_name], valid_df[feature_name], transform_mode=transform_mode
        )
        parts_tr.append(pd.DataFrame({feature_name: ftr_tr}, index=train_df.index))
        parts_va.append(pd.DataFrame({feature_name: ftr_va}, index=valid_df.index))

    Xtr = pd.concat(parts_tr, axis=1) if parts_tr else pd.DataFrame(index=train_df.index)
    Xva = pd.concat(parts_va, axis=1) if parts_va else pd.DataFrame(index=valid_df.index)

    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xva = Xva.replace([np.inf, -np.inf], np.nan)
    return Xtr, Xva


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    y = safe_numeric(y_true)
    if y.nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, y_prob))
    except Exception:
        return np.nan


def _safe_auprc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    y = safe_numeric(y_true)
    if y.nunique() < 2:
        return np.nan
    try:
        return float(average_precision_score(y, y_prob))
    except Exception:
        return np.nan


def _safe_cindex(time: pd.Series, event: pd.Series, risk: np.ndarray) -> float:
    tt = safe_numeric(time)
    ee = safe_numeric(event)
    if ee.sum() <= 0:
        return np.nan
    try:
        return float(concordance_index(tt, -risk, ee))
    except Exception:
        return np.nan


def fit_full_logistic_model(
    df: pd.DataFrame,
    outcome_col: str,
    clinical_covars: Sequence[str],
    feature_name: Optional[str] = None,
    transform_mode: str = "raw",
) -> Dict[str, object]:
    """
    Classical full-dataset logistic regression.
    """
    work = df.copy()
    cols = [outcome_col] + [c for c in clinical_covars if c in work.columns]
    if feature_name is not None:
        cols.append(feature_name)
    cols = [c for c in cols if c in work.columns]

    work = work[cols].copy()
    work[outcome_col] = safe_numeric(work[outcome_col])

    X, feature_cols_used = _make_design_matrix(
        work,
        feature_name=feature_name,
        clinical_covars=clinical_covars,
        transform_mode=transform_mode,
    )

    model_df = pd.concat([X, work[[outcome_col]]], axis=1).dropna().copy()
    if model_df.empty:
        raise ValueError("No rows remain after dropping missing inputs.")
    if model_df[outcome_col].nunique() < 2:
        raise ValueError("Outcome has a single class.")

    y = model_df[outcome_col].astype(int)
    X_fit = model_df.drop(columns=[outcome_col])
    X_fit = sm.add_constant(X_fit, has_constant="add")

    fit = sm.Logit(y, X_fit).fit(disp=False)
    prob = fit.predict(X_fit)

    out = {
        "model_family": "logistic",
        "model_name": (
            "clinical_only" if feature_name is None and len(clinical_covars) > 0
            else "biomarker_only" if feature_name is not None and len(clinical_covars) == 0
            else "clinical_plus_biomarker"
        ),
        "transform_mode": transform_mode,
        "n": int(len(model_df)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "auc_full": _safe_auc(y, prob),
        "auprc_full": _safe_auprc(y, prob),
        "aic": float(fit.aic),
        "loglik": float(fit.llf),
        "clinical_covars_used": [c for c in clinical_covars if c in df.columns],
        "fit_object": fit,
        "design_columns": [c for c in X_fit.columns if c != "const"],
    }

    if feature_name is not None:
        params = fit.params
        conf = fit.conf_int()
        pvals = fit.pvalues

        if feature_name not in params.index:
            raise ValueError(f"Feature '{feature_name}' missing from fitted logistic model.")

        coef = float(params[feature_name])
        ci_low = float(conf.loc[feature_name, 0])
        ci_high = float(conf.loc[feature_name, 1])

        out.update({
            "feature": feature_name,
            "coef": coef,
            "coef_ci_low": ci_low,
            "coef_ci_high": ci_high,
            "effect": float(np.exp(coef)),
            "effect_ci_low": float(np.exp(ci_low)),
            "effect_ci_high": float(np.exp(ci_high)),
            "wald_p_value": float(pvals[feature_name]),
        })

    return out


def fit_full_cox_model(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    clinical_covars: Sequence[str],
    feature_name: Optional[str] = None,
    transform_mode: str = "raw",
    penalizer: float = 0.0,
) -> Dict[str, object]:
    """
    Classical full-dataset Cox regression.

    For inferential tables, penalizer=0.0 is more classical than the stage-1
    screening Cox path, which currently uses penalizer=0.01. :contentReference[oaicite:3]{index=3}
    """
    work = df.copy()
    cols = [time_col, event_col] + [c for c in clinical_covars if c in work.columns]
    if feature_name is not None:
        cols.append(feature_name)
    cols = [c for c in cols if c in work.columns]

    work = work[cols].copy()
    work[time_col] = safe_numeric(work[time_col])
    work[event_col] = safe_numeric(work[event_col])

    X, feature_cols_used = _make_design_matrix(
        work,
        feature_name=feature_name,
        clinical_covars=clinical_covars,
        transform_mode=transform_mode,
    )

    model_df = pd.concat([work[[time_col, event_col]], X], axis=1).dropna().copy()
    if model_df.empty:
        raise ValueError("No rows remain after dropping missing inputs.")
    model_df[event_col] = safe_numeric(model_df[event_col]).astype(int)
    model_df[time_col] = safe_numeric(model_df[time_col])
    if model_df[event_col].sum() <= 0:
        raise ValueError("No events remain after filtering.")

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_df, duration_col=time_col, event_col=event_col)

    risk = cph.predict_partial_hazard(model_df.drop(columns=[time_col, event_col])).values.reshape(-1)

    out = {
        "model_family": "cox",
        "model_name": (
            "clinical_only" if feature_name is None and len(clinical_covars) > 0
            else "biomarker_only" if feature_name is not None and len(clinical_covars) == 0
            else "clinical_plus_biomarker"
        ),
        "transform_mode": transform_mode,
        "n": int(len(model_df)),
        "n_events": int(model_df[event_col].sum()),
        "cindex_full": _safe_cindex(model_df[time_col], model_df[event_col], risk),
        "partial_loglik": float(cph.log_likelihood_),
        "aic": float(cph.AIC_partial_) if hasattr(cph, "AIC_partial_") else np.nan,
        "clinical_covars_used": [c for c in clinical_covars if c in df.columns],
        "fit_object": cph,
        "design_columns": [c for c in model_df.columns if c not in {time_col, event_col}],
        "time_col": time_col,
        "event_col": event_col,
    }

    if feature_name is not None:
        summ = cph.summary
        if feature_name not in summ.index:
            raise ValueError(f"Feature '{feature_name}' missing from Cox summary table.")

        row = summ.loc[feature_name]
        out.update({
            "feature": feature_name,
            "coef": float(row["coef"]),
            "coef_ci_low": float(row["coef lower 95%"]),
            "coef_ci_high": float(row["coef upper 95%"]),
            "effect": float(row["exp(coef)"]),
            "effect_ci_low": float(row["exp(coef) lower 95%"]),
            "effect_ci_high": float(row["exp(coef) upper 95%"]),
            "wald_p_value": float(row["p"]),
        })

    return out


def compare_nested_logistic_models_lrt(
    df: pd.DataFrame,
    outcome_col: str,
    clinical_covars: Sequence[str],
    feature_name: str,
    transform_mode: str = "raw",
) -> Dict[str, float]:
    """
    LRT: clinical-only vs clinical+biomarker for logistic regression.
    """
    fit0 = fit_full_logistic_model(
        df=df,
        outcome_col=outcome_col,
        clinical_covars=clinical_covars,
        feature_name=None,
        transform_mode=transform_mode,
    )
    fit1 = fit_full_logistic_model(
        df=df,
        outcome_col=outcome_col,
        clinical_covars=clinical_covars,
        feature_name=feature_name,
        transform_mode=transform_mode,
    )

    ll0 = fit0["loglik"]
    ll1 = fit1["loglik"]
    df0 = len(fit0["design_columns"]) + 1  # + intercept
    df1 = len(fit1["design_columns"]) + 1

    lrt_stat = 2.0 * (ll1 - ll0)
    df_diff = max(df1 - df0, 0)
    p = 1.0 - chi2.cdf(lrt_stat, df=df_diff) if df_diff > 0 else np.nan

    return {
        "lrt_stat": float(lrt_stat),
        "lrt_df": int(df_diff),
        "lrt_p_value": float(p) if pd.notna(p) else np.nan,
        "clinical_only_auc_full": fit0["auc_full"],
        "clinical_plus_biomarker_auc_full": fit1["auc_full"],
        "delta_auc_full_vs_clinical": (
            fit1["auc_full"] - fit0["auc_full"]
            if pd.notna(fit1["auc_full"]) and pd.notna(fit0["auc_full"])
            else np.nan
        ),
    }


def compare_nested_cox_models_lrt(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    clinical_covars: Sequence[str],
    feature_name: str,
    transform_mode: str = "raw",
    penalizer: float = 0.0,
) -> Dict[str, float]:
    """
    LRT: clinical-only vs clinical+biomarker for Cox regression.
    """
    fit0 = fit_full_cox_model(
        df=df,
        time_col=time_col,
        event_col=event_col,
        clinical_covars=clinical_covars,
        feature_name=None,
        transform_mode=transform_mode,
        penalizer=penalizer,
    )
    fit1 = fit_full_cox_model(
        df=df,
        time_col=time_col,
        event_col=event_col,
        clinical_covars=clinical_covars,
        feature_name=feature_name,
        transform_mode=transform_mode,
        penalizer=penalizer,
    )

    ll0 = fit0["partial_loglik"]
    ll1 = fit1["partial_loglik"]
    df0 = len(fit0["design_columns"])
    df1 = len(fit1["design_columns"])

    lrt_stat = 2.0 * (ll1 - ll0)
    df_diff = max(df1 - df0, 0)
    p = 1.0 - chi2.cdf(lrt_stat, df=df_diff) if df_diff > 0 else np.nan

    return {
        "lrt_stat": float(lrt_stat),
        "lrt_df": int(df_diff),
        "lrt_p_value": float(p) if pd.notna(p) else np.nan,
        "clinical_only_cindex_full": fit0["cindex_full"],
        "clinical_plus_biomarker_cindex_full": fit1["cindex_full"],
        "delta_cindex_full_vs_clinical": (
            fit1["cindex_full"] - fit0["cindex_full"]
            if pd.notna(fit1["cindex_full"]) and pd.notna(fit0["cindex_full"])
            else np.nan
        ),
    }


def run_full_dataset_inference_single_feature(
    patient_df: pd.DataFrame,
    feature_name: str,
    endpoint: str,
    clinical_covars: Sequence[str],
    transform_modes: Sequence[str] = TRANSFORM_MODES,
    cox_penalizer: float = 0.0,
) -> Dict[str, object]:
    """
    Full-dataset inferential layer for one biomarker.
    Returns a dict with:
      - rows: list[dict]
      - status / reason
    """
    out = {
        "feature": feature_name,
        "status": "ok",
        "reason": "",
        "rows": [],
    }

    try:
        if endpoint in {"complete_response", "any_response"}:
            df_ready, outcome_col = build_response_endpoint(patient_df.copy(), endpoint)
            base_cols = ["patient_id", outcome_col, feature_name] + [c for c in clinical_covars if c in df_ready.columns]
            base_cols = [c for c in base_cols if c in df_ready.columns]
            df_ready = df_ready[base_cols].copy()
            df_ready[feature_name] = safe_numeric(df_ready[feature_name])
            df_ready = df_ready.dropna(subset=[outcome_col, feature_name]).copy()
            if df_ready.empty or df_ready[outcome_col].nunique() < 2:
                out["status"] = "skip"
                out["reason"] = "no_rows_or_single_class_after_filter"
                return out

            for tm in transform_modes:
                try:
                    biomarker_only = fit_full_logistic_model(
                        df=df_ready,
                        outcome_col=outcome_col,
                        clinical_covars=[],
                        feature_name=feature_name,
                        transform_mode=tm,
                    )
                    biomarker_only.update({
                        "endpoint": endpoint,
                        "feature": feature_name,
                    })
                    out["rows"].append(biomarker_only)

                    if len(clinical_covars) > 0:
                        clinical_only = fit_full_logistic_model(
                            df=df_ready,
                            outcome_col=outcome_col,
                            clinical_covars=clinical_covars,
                            feature_name=None,
                            transform_mode=tm,
                        )
                        clinical_only.update({
                            "endpoint": endpoint,
                            "feature": feature_name,
                        })
                        out["rows"].append(clinical_only)

                        clinical_plus = fit_full_logistic_model(
                            df=df_ready,
                            outcome_col=outcome_col,
                            clinical_covars=clinical_covars,
                            feature_name=feature_name,
                            transform_mode=tm,
                        )
                        clinical_plus.update({
                            "endpoint": endpoint,
                            "feature": feature_name,
                        })

                        lrt = compare_nested_logistic_models_lrt(
                            df=df_ready,
                            outcome_col=outcome_col,
                            clinical_covars=clinical_covars,
                            feature_name=feature_name,
                            transform_mode=tm,
                        )
                        clinical_plus.update(lrt)
                        out["rows"].append(clinical_plus)

                except Exception as e:
                    out["rows"].append({
                        "feature": feature_name,
                        "endpoint": endpoint,
                        "model_family": "logistic",
                        "model_name": "transform_failed",
                        "transform_mode": tm,
                        "status": "fail",
                        "reason": f"{type(e).__name__}: {e}",
                    })

            return out

        if endpoint in {"OS", "RFS"}:
            df_ready, time_col, event_col = build_survival_endpoint(patient_df.copy(), endpoint)
            base_cols = ["patient_id", time_col, event_col, feature_name] + [c for c in clinical_covars if c in df_ready.columns]
            base_cols = [c for c in base_cols if c in df_ready.columns]
            df_ready = df_ready[base_cols].copy()
            df_ready[feature_name] = safe_numeric(df_ready[feature_name])
            df_ready = df_ready.dropna(subset=[time_col, event_col, feature_name]).copy()
            if df_ready.empty or safe_numeric(df_ready[event_col]).sum() <= 0:
                out["status"] = "skip"
                out["reason"] = "no_rows_or_no_events_after_filter"
                return out

            for tm in transform_modes:
                try:
                    biomarker_only = fit_full_cox_model(
                        df=df_ready,
                        time_col=time_col,
                        event_col=event_col,
                        clinical_covars=[],
                        feature_name=feature_name,
                        transform_mode=tm,
                        penalizer=cox_penalizer,
                    )
                    biomarker_only.update({
                        "endpoint": endpoint,
                        "feature": feature_name,
                    })
                    out["rows"].append(biomarker_only)

                    if len(clinical_covars) > 0:
                        clinical_only = fit_full_cox_model(
                            df=df_ready,
                            time_col=time_col,
                            event_col=event_col,
                            clinical_covars=clinical_covars,
                            feature_name=None,
                            transform_mode=tm,
                            penalizer=cox_penalizer,
                        )
                        clinical_only.update({
                            "endpoint": endpoint,
                            "feature": feature_name,
                        })
                        out["rows"].append(clinical_only)

                        clinical_plus = fit_full_cox_model(
                            df=df_ready,
                            time_col=time_col,
                            event_col=event_col,
                            clinical_covars=clinical_covars,
                            feature_name=feature_name,
                            transform_mode=tm,
                            penalizer=cox_penalizer,
                        )
                        clinical_plus.update({
                            "endpoint": endpoint,
                            "feature": feature_name,
                        })

                        lrt = compare_nested_cox_models_lrt(
                            df=df_ready,
                            time_col=time_col,
                            event_col=event_col,
                            clinical_covars=clinical_covars,
                            feature_name=feature_name,
                            transform_mode=tm,
                            penalizer=cox_penalizer,
                        )
                        clinical_plus.update(lrt)
                        out["rows"].append(clinical_plus)

                except Exception as e:
                    out["rows"].append({
                        "feature": feature_name,
                        "endpoint": endpoint,
                        "model_family": "cox",
                        "model_name": "transform_failed",
                        "transform_mode": tm,
                        "status": "fail",
                        "reason": f"{type(e).__name__}: {e}",
                    })

            return out

        raise ValueError(f"Unsupported endpoint: {endpoint}")

    except Exception as e:
        out["status"] = "fail"
        out["reason"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        return out

NN_STATS = ["Mean", "SD", "Max", "Min", "Median", "Q1", "Q3"]
DISTANCE_TO_NNSTATS_RENAME = {
    "Distance_Mean": "Mean",
    "Distance_SD": "SD",
    "Distance_Max": "Max",
    "Distance_Min": "Min",
    "Distance_Median": "Median",
    "Distance_Q1": "Q1",
    "Distance_Q3": "Q3",
}

DEFAULT_COHORTS = ["PURE01", "NAC2015", "NAC2020", "No-NAC", "BLASST", "KOLL"]
DEFAULT_PANELS = ["AR", "BT"]

# Five reviewed feature-source namespaces generated by the STROMA/checkpoint recode workflow.
FEATURE_SOURCE_CONFIG = {
    "phenotype_only": {
        "spatial_root": Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_phenotype_only"),
        "cell_features_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/phenotype_only/cell_features_phenotype_only.csv"),
        "triads_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/triads_phenotype_only/triad_features_phenotype_only.csv"),
        "panels": ["AR", "BT"],
    },
    "AR_state": {
        "spatial_root": Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_state"),
        "cell_features_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/AR_state/cell_features_AR_state.csv"),
        "triads_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/triads_AR_state/triad_features_AR_state.csv"),
        "panels": ["AR"],
    },
    "AR_checkpoint_state": {
        "spatial_root": Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_AR_checkpoint_state"),
        "cell_features_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/AR_checkpoint_state/cell_features_AR_checkpoint_state.csv"),
        "triads_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/triads_AR_checkpoint_state/triad_features_AR_checkpoint_state.csv"),
        "panels": ["AR"],
    },
    "compartment": {
        "spatial_root": Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_compartment"),
        "cell_features_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/compartment/cell_features_compartment.csv"),
        "triads_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/triads_compartment/triad_features_compartment.csv"),
        "panels": ["AR", "BT"],
    },
    "compartment_state": {
        "spatial_root": Path("/projects/ovcare/users/nikolay_alabi/immuno/weibull/run_reviewed_compartment_state"),
        "cell_features_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/compartment_state/cell_features_compartment_state.csv"),
        "triads_path": Path("/projects/ovcare/users/nikolay_alabi/immuno/cell_feature_tables_reviewed/triads_compartment_state/triad_features_compartment_state.csv"),
        "panels": ["AR"],
    },
}

DEFAULT_TMA_CLINICAL_CSV = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_NAC_NoNAC_PURE01_NAC2.csv")
DEFAULT_TMA_QC_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data")
DEFAULT_TMA_TISSUE_SEG_CSV = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/tissue_segmentation/tma_tissue_region_summary_compact.csv")
DEFAULT_BLASST_METADATA_CSV = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/ClinicalData_Core_BLASST.csv")
DEFAULT_BLASST_TISSUE_SEG_CSV = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/tissue_segmentation/whole_sections_tissue_region_summary_compact.csv")
DEFAULT_KOLL_METADATA_CSV = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/KOLL_cohort/KOLL_core_metadata.csv")


def _coerce_path(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path)


def _default_spatial_root_for_source(feature_source: str) -> Path:
    try:
        return FEATURE_SOURCE_CONFIG[feature_source]["spatial_root"]
    except KeyError:
        raise ValueError(f"Unknown feature_source: {feature_source}. Expected one of {sorted(FEATURE_SOURCE_CONFIG)}")


def _default_cell_features_path_for_source(feature_source: str) -> Path:
    try:
        return FEATURE_SOURCE_CONFIG[feature_source]["cell_features_path"]
    except KeyError:
        raise ValueError(f"Unknown feature_source: {feature_source}. Expected one of {sorted(FEATURE_SOURCE_CONFIG)}")


def _default_triads_path_for_source(feature_source: str) -> Path:
    try:
        return FEATURE_SOURCE_CONFIG[feature_source]["triads_path"]
    except KeyError:
        raise ValueError(f"Unknown feature_source: {feature_source}. Expected one of {sorted(FEATURE_SOURCE_CONFIG)}")


def discover_chunk_files(
    cohort_paths: Optional[dict[str, Path]] = None,
    panels: Optional[list[str]] = None,
    filename: str = "NNstats.tsv",
    *,
    spatial_root: Optional[str | Path] = None,
    cohorts: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Discover chunk-level feature files.

    Supports both the legacy layout:
        <cohort_root>/<panel>/chunk_*/<filename>

    and the reviewed v3/v4 layout:
        <spatial_root>/<dataset>/<cohort>/<panel>/chunk_*/<filename>
    """
    rows = []
    panels_set = set(panels or []) if panels is not None else None
    cohorts_set = set(str(c) for c in cohorts) if cohorts is not None else None

    if spatial_root is not None:
        root = Path(spatial_root)
        if not root.exists():
            raise FileNotFoundError(f"spatial_root does not exist: {root}")
        for fp in sorted(root.rglob(filename)):
            if fp.parent.name.startswith("chunk_"):
                chunk_dir = fp.parent.name
                panel = fp.parent.parent.name
                cohort = fp.parent.parent.parent.name
                dataset = fp.parent.parent.parent.parent.name
            else:
                continue

            if panels_set is not None and panel not in panels_set:
                continue
            if cohorts_set is not None and str(cohort) not in cohorts_set:
                continue

            rows.append({
                "dataset": dataset,
                "cohort": cohort,
                "Panel": panel,
                "chunk_dir": chunk_dir,
                "path": fp,
            })
        return pd.DataFrame(rows)

    if cohort_paths is None:
        raise ValueError("Either spatial_root or cohort_paths must be provided.")

    for cohort, cohort_root in cohort_paths.items():
        if cohorts_set is not None and str(cohort) not in cohorts_set:
            continue
        for panel in panels or []:
            panel_root = Path(cohort_root) / panel
            if not panel_root.exists():
                continue
            for fp in sorted(panel_root.glob(f"chunk_*/{filename}")):
                rows.append({
                    "dataset": "legacy",
                    "cohort": cohort,
                    "Panel": panel,
                    "chunk_dir": fp.parent.name,
                    "path": fp,
                })

    return pd.DataFrame(rows)


def standardize_sample_id_column(
    df: pd.DataFrame,
    sample_candidates: tuple[str, ...] = ("sample_id", "sample_name"),
) -> pd.DataFrame:
    out = df.copy()

    if "sample_id" in out.columns:
        return out

    for col in sample_candidates:
        if col in out.columns:
            out = out.rename(columns={col: "sample_id"})
            return out

    raise ValueError(
        f"Could not find a sample column. Expected one of {sample_candidates}. "
        f"Available columns: {list(out.columns)}"
    )


def add_coord_from_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        raise ValueError("Dataframe must contain 'sample_id' before coord extraction.")
    out["coord"] = extract_coord_token(out["sample_id"])
    if out["coord"].isna().any():
        # Fallback to generic bracket/underscore standardization if extract_coord_token missed BLASST-style IDs.
        fallback = standardize_coord_series(out["sample_id"], allow_blasst_underscore=True)
        out.loc[out["coord"].isna(), "coord"] = fallback.loc[out["coord"].isna()]

    # KOLL is not a TMA with [x,y] coordinate tokens. For KOLL, sample_id is the
    # core/sample identifier, so preserve it as coord when coordinate parsing fails.
    if "cohort" in out.columns and out["coord"].isna().any():
        is_koll = out["cohort"].astype(str).str.strip().eq("KOLL")
        m = is_koll & out["coord"].isna() & out["sample_id"].notna()
        out.loc[m, "coord"] = out.loc[m, "sample_id"].astype(str).str.strip()
    return out


def load_one_nnstats(path: str | Path, cohort: str, panel: str, dataset: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={k: v for k, v in DISTANCE_TO_NNSTATS_RENAME.items() if k in df.columns})

    required = {"sample_id", "phenotype_combo", *NN_STATS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}. Available: {list(df.columns)[:30]}")

    long_df = df.melt(
        id_vars=["sample_id", "phenotype_combo"],
        value_vars=NN_STATS,
        var_name="stat",
        value_name="value",
    )

    long_df["feature"] = long_df["phenotype_combo"].astype(str) + "_" + long_df["stat"].astype(str)

    wide_df = (
        long_df.pivot_table(
            index="sample_id",
            columns="feature",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_df.columns.name = None

    wide_df["cohort"] = cohort
    wide_df["Panel"] = panel
    wide_df["dataset"] = dataset
    wide_df = add_coord_from_sample_id(wide_df)

    return wide_df


def build_nn_dataframe(
    *,
    panels: list[str],
    filename: str = "NNstats.tsv",
    spatial_root: Optional[str | Path] = None,
    cohorts: Optional[Sequence[str]] = None,
    cohort_paths: Optional[dict[str, Path]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nn_files = discover_chunk_files(
        cohort_paths=cohort_paths,
        panels=panels,
        filename=filename,
        spatial_root=spatial_root,
        cohorts=cohorts,
    )

    if nn_files.empty:
        raise ValueError(f"No {filename} files found under {spatial_root}.")

    parts = []
    for row in nn_files.itertuples(index=False):
        dataset = getattr(row, "dataset", "")
        parts.append(load_one_nnstats(row.path, row.cohort, row.Panel, dataset=dataset))

    nn_df = pd.concat(parts, ignore_index=True)

    key_cols = [c for c in ["sample_id", "coord", "Panel", "cohort", "dataset"] if c in nn_df.columns]
    nn_df = collapse_duplicate_rows(nn_df, key_cols=key_cols)

    return nn_df, nn_files


def load_one_athena(path: str | Path, cohort: str, panel: str, dataset: str = "") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_sample_id_column(df, sample_candidates=("sample_id", "sample_name"))

    if "Panel" not in df.columns:
        df["Panel"] = panel
    if "cohort" not in df.columns:
        df["cohort"] = cohort
    if "dataset" not in df.columns:
        df["dataset"] = dataset
    if "coord" not in df.columns:
        df = add_coord_from_sample_id(df)

    df["cohort"] = df["cohort"].astype(str)
    df["Panel"] = df["Panel"].astype(str)
    return df


def build_athena_dataframe(
    *,
    panels: list[str],
    filename: str = "athena_features.csv",
    spatial_root: Optional[str | Path] = None,
    cohorts: Optional[Sequence[str]] = None,
    cohort_paths: Optional[dict[str, Path]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    athena_files = discover_chunk_files(
        cohort_paths=cohort_paths,
        panels=panels,
        filename=filename,
        spatial_root=spatial_root,
        cohorts=cohorts,
    )

    if athena_files.empty:
        raise ValueError(f"No {filename} files found under {spatial_root}.")

    parts = []
    for row in athena_files.itertuples(index=False):
        dataset = getattr(row, "dataset", "")
        parts.append(load_one_athena(row.path, row.cohort, row.Panel, dataset=dataset))

    athena_df = pd.concat(parts, ignore_index=True)

    key_cols = [c for c in ["sample_id", "coord", "Panel", "cohort", "dataset"] if c in athena_df.columns]
    athena_df = collapse_duplicate_rows(athena_df, key_cols=key_cols)

    return athena_df, athena_files


def load_wide_core_feature_table(path: str | Path, feature_group: str) -> pd.DataFrame:
    """
    Load a wide core-level feature table produced from the reviewed prep workflow.

    Used for cell_features/ratios and triads. Keeps ALL_NEG features if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{feature_group} file does not exist: {path}")

    df = pd.read_csv(path, low_memory=False).copy()

    rename = {}
    if "panel" in df.columns and "Panel" not in df.columns:
        rename["panel"] = "Panel"
    if "COHORT" in df.columns and "cohort" not in df.columns:
        rename["COHORT"] = "cohort"
    if "Core" in df.columns and "sample_id" not in df.columns:
        rename["Core"] = "sample_id"
    if "sample_name" in df.columns and "sample_id" not in df.columns:
        rename["sample_name"] = "sample_id"
    df = df.rename(columns=rename)

    if "sample_id" not in df.columns:
        if "coord" in df.columns:
            df["sample_id"] = df["coord"].astype(str)
        else:
            df = standardize_sample_id_column(df, sample_candidates=("sample_id", "sample_name", "Core"))

    if "coord" not in df.columns:
        df = add_coord_from_sample_id(df)

    required = {"coord", "Panel", "cohort"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{feature_group} file is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)[:50]}"
        )

    df["Panel"] = df["Panel"].astype(str).str.strip()
    df["cohort"] = df["cohort"].astype(str).str.strip()
    return df


def load_ratios_dataframe(ratios_path: str | Path) -> pd.DataFrame:
    # Backward-compatible alias. In v4 this is usually the reviewed cell_features table.
    return load_wide_core_feature_table(ratios_path, feature_group="ratios")


def build_tma_meta_qc(
    clinical_csv: str | Path,
    qc_dir: str | Path,
    tissue_seg_csv: Optional[str | Path] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    meta_df = load_tma_metadata(clinical_csv)
    qc_df = load_tma_qc(qc_dir)
    tissue_seg_df = (
        load_tma_tissue_segmentation(tissue_seg_csv)
        if tissue_seg_csv is not None
        else None
    )

    return {
        "meta_df": meta_df,
        "qc_df": qc_df,
        "tissue_seg_df": tissue_seg_df,
    }


def build_blasst_meta_qc(
    metadata_csv: str | Path,
    tissue_seg_csv: Optional[str | Path] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    meta_df = load_blasst_metadata(metadata_csv)
    tissue_seg_df = (
        load_blasst_tissue_segmentation(tissue_seg_csv)
        if tissue_seg_csv is not None
        else None
    )

    blasst_dict = {
        "meta_df": meta_df,
        "qc_df": None,
        "tissue_seg_df": tissue_seg_df,
    }

    blasst_dict = add_blasst_sample_type_to_dataframes(blasst_dict)

    return blasst_dict


def combine_meta_qc_dicts(
    tma_dict: dict[str, Optional[pd.DataFrame]],
    blasst_dict: dict[str, Optional[pd.DataFrame]],
) -> dict[str, Optional[pd.DataFrame]]:
    out = {}

    for key in ["meta_df", "qc_df", "tissue_seg_df"]:
        frames = []
        for d in [tma_dict, blasst_dict]:
            if d.get(key) is not None:
                frames.append(d[key])

        if frames:
            out[key] = pd.concat(frames, ignore_index=True, sort=False)
        else:
            out[key] = None

    return out


def load_data_dict(
    *,
    feature_group: str,
    feature_source: str = "phenotype_only",
    panels: Optional[Sequence[str]] = None,
    cohorts: Optional[Sequence[str]] = None,
    spatial_root: Optional[str | Path] = None,
    cell_features_path: Optional[str | Path] = None,
    triads_path: Optional[str | Path] = None,
    tma_clinical_csv: str | Path = DEFAULT_TMA_CLINICAL_CSV,
    tma_qc_dir: str | Path = DEFAULT_TMA_QC_DIR,
    tma_tissue_seg_csv: Optional[str | Path] = DEFAULT_TMA_TISSUE_SEG_CSV,
    blasst_metadata_csv: str | Path = DEFAULT_BLASST_METADATA_CSV,
    blasst_tissue_seg_csv: Optional[str | Path] = DEFAULT_BLASST_TISSUE_SEG_CSV,
) -> Dict[str, pd.DataFrame]:
    """
    Load only the requested feature family plus shared metadata/QC/tissue tables.
    This avoids failing a cell-feature job because ATHENA outputs are not ready, etc.
    """
    if feature_group == "ratios":
        feature_group_to_load = "cell_features"
    else:
        feature_group_to_load = feature_group

    if feature_group_to_load not in {"NN", "athena", "cell_features", "triads"}:
        raise ValueError("feature_group must be one of NN, athena, cell_features, ratios, triads.")

    panels = list(panels) if panels is not None else list(DEFAULT_PANELS)
    cohorts = list(cohorts) if cohorts is not None else list(DEFAULT_COHORTS)

    spatial_root = _coerce_path(spatial_root) or _default_spatial_root_for_source(feature_source)
    cell_features_path = _coerce_path(cell_features_path) or _default_cell_features_path_for_source(feature_source)
    triads_path = _coerce_path(triads_path) or _default_triads_path_for_source(feature_source)

    tma_dict = build_tma_meta_qc(
        clinical_csv=tma_clinical_csv,
        qc_dir=tma_qc_dir,
        tissue_seg_csv=tma_tissue_seg_csv,
    )
    blasst_dict = build_blasst_meta_qc(
        metadata_csv=blasst_metadata_csv,
        tissue_seg_csv=blasst_tissue_seg_csv,
    )
    combined_meta_qc = combine_meta_qc_dicts(tma_dict, blasst_dict)

    out = {
        "qc_df": combined_meta_qc["qc_df"],
        "meta_df": combined_meta_qc["meta_df"],
        "tissue_seg_df": combined_meta_qc["tissue_seg_df"],
        "tma_meta_df": tma_dict["meta_df"],
        "tma_qc_df": tma_dict["qc_df"],
        "blasst_meta_df": blasst_dict["meta_df"],
        "blasst_tissue_seg_df": blasst_dict["tissue_seg_df"],
    }

    if feature_group_to_load == "NN":
        nn_df, nn_files = build_nn_dataframe(
            panels=panels,
            spatial_root=spatial_root,
            cohorts=cohorts,
            filename="NNstats.tsv",
        )
        out["NN"] = nn_df
        out["nn_files"] = nn_files

    elif feature_group_to_load == "athena":
        athena_df, athena_files = build_athena_dataframe(
            panels=panels,
            spatial_root=spatial_root,
            cohorts=cohorts,
            filename="athena_features.csv",
        )
        out["athena"] = athena_df
        out["athena_files"] = athena_files

    elif feature_group_to_load == "cell_features":
        cf = load_wide_core_feature_table(cell_features_path, feature_group="cell_features")
        out["cell_features"] = cf
        out["ratios"] = cf  # backwards-compatible alias

    elif feature_group_to_load == "triads":
        out["triads"] = load_wide_core_feature_table(triads_path, feature_group="triads")

    out = standardize_and_validate_all_coords(out)
    return out

def load_harmonized_df(path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()
    if "patient_id" not in df.columns:
        raise ValueError("harmonized_df must contain 'patient_id'.")
    if "cohort" not in df.columns:
        raise ValueError("harmonized_df must contain 'cohort'.")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["cohort"] = df["cohort"].astype(str).str.strip()
    return df

def ensure_patient_id_column(
    df: pd.DataFrame,
    *,
    candidate_cols: Sequence[str] = (
        "patient_id",
        "Patient_ID",
        "PATIENT_ID",
        "patient",
        "Sample_ID_Adjusted",
        "Sample_ID",
        "case_id",
        "Case_ID",
    ),
) -> pd.DataFrame:
    """
    Create a clean patient_id column from the first candidate column that
    contains at least some real non-missing values.

    Important:
    - avoids converting NaN -> 'nan'
    - forces empty / pseudo-missing strings back to NA
    - fails loudly if no usable patient ID source exists
    """
    out = df.copy()

    chosen = None
    for col in candidate_cols:
        if col in out.columns:
            s = out[col].copy()
            s = s.astype("object")
            s = s.where(pd.notna(s), pd.NA)
            s = s.astype("string").str.strip()
            s = s.replace({
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
                "": pd.NA,
            })
            if s.notna().any():
                chosen = col
                break

    if chosen is None:
        available = [c for c in candidate_cols if c in out.columns]
        raise ValueError(
            "Could not find any usable patient ID column after metadata merge. "
            f"Available candidate columns: {available}"
        )

    s = out[chosen].copy()
    s = s.astype("object")
    s = s.where(pd.notna(s), pd.NA)
    s = s.astype("string").str.strip()
    s = s.replace({
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
        "": pd.NA,
    })

    out["patient_id"] = s

    n_missing = out["patient_id"].isna().sum()
    if n_missing > 0:
        print(
            f"[WARN] patient_id missing for {n_missing}/{len(out)} rows "
            f"after choosing source column '{chosen}'",
            flush=True,
        )

    return out


# -----------------------------------------------------------------------------
# KOLL core/sample metadata adapter
# -----------------------------------------------------------------------------

def _clean_id_value(x) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>", "null"}:
        return pd.NA
    return s


def _norm_match_key(x) -> object:
    s = _clean_id_value(x)
    if pd.isna(s):
        return pd.NA
    return re.sub(r"\s+", " ", str(s).strip()).upper()


def _first_nonmissing_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c].map(_clean_id_value)
        out = out.where(out.notna(), s)
    return out


def load_koll_metadata(path: str | Path = DEFAULT_KOLL_METADATA_CSV) -> pd.DataFrame:
    """Load the explicit KOLL core metadata/crosswalk.

    Expected columns are compatible with the notebook-built file:
        sample_id, sample_name, coord, patient_id, cohort, TURBT_or_RC, Panel

    The important linkage is feature core/sample -> patient_id, while the
    sample type is taken from TURBT_or_RC so KOLL TURBT and RC samples can be
    analyzed separately if both exist.
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(
            f"KOLL metadata file not found: {fp}. Build KOLL_core_metadata.csv before running KOLL Stage 1."
        )

    meta = pd.read_csv(fp, low_memory=False).copy()

    rename = {}
    if "panel" in meta.columns and "Panel" not in meta.columns:
        rename["panel"] = "Panel"
    if "Patient_ID" in meta.columns and "patient_id" not in meta.columns:
        rename["Patient_ID"] = "patient_id"
    if "specimen_type" in meta.columns and "TURBT_or_RC" not in meta.columns:
        rename["specimen_type"] = "TURBT_or_RC"
    meta = meta.rename(columns=rename)

    if "cohort" not in meta.columns:
        meta["cohort"] = "KOLL"
    meta["cohort"] = "KOLL"

    if "Panel" not in meta.columns:
        raise ValueError(f"KOLL metadata is missing Panel column: {fp}")
    if "patient_id" not in meta.columns:
        raise ValueError(f"KOLL metadata is missing patient_id column: {fp}")
    if "TURBT_or_RC" not in meta.columns:
        raise ValueError(f"KOLL metadata is missing TURBT_or_RC column: {fp}")

    for c in ["sample_id", "sample_name", "coord", "patient_id", "Panel", "TURBT_or_RC", "specimen_type", "cohort"]:
        if c in meta.columns:
            meta[c] = meta[c].map(_clean_id_value)

    # If coord/sample IDs are incomplete, fill from the best available core identifier.
    if "coord" not in meta.columns:
        meta["coord"] = pd.NA
    if "sample_id" not in meta.columns:
        meta["sample_id"] = pd.NA
    if "sample_name" not in meta.columns:
        meta["sample_name"] = pd.NA

    best_core = _first_nonmissing_series(meta, ["coord", "sample_id", "sample_name"])
    meta["coord"] = meta["coord"].where(meta["coord"].notna(), best_core)
    meta["sample_id"] = meta["sample_id"].where(meta["sample_id"].notna(), best_core)
    meta["sample_name"] = meta["sample_name"].where(meta["sample_name"].notna(), best_core)

    # Match keys are deliberately string-based, because KOLL cores are not [x,y]
    # TMA coordinates. They are named cores from the UBC summary sheet.
    meta["__koll_key"] = _first_nonmissing_series(meta, ["coord", "sample_id", "sample_name"]).map(_norm_match_key)
    meta["Panel"] = meta["Panel"].astype(str).str.strip()
    meta["TURBT_or_RC"] = meta["TURBT_or_RC"].map(normalize_sample_type)

    meta = meta.dropna(subset=["__koll_key", "patient_id", "Panel"]).copy()
    meta = meta.drop_duplicates(["Panel", "__koll_key"], keep="first")
    return meta


def merge_koll_metadata_to_feature_df(
    feat_df: pd.DataFrame,
    koll_metadata_csv: str | Path = DEFAULT_KOLL_METADATA_CSV,
) -> pd.DataFrame:
    """Attach KOLL patient_id and TURBT_or_RC to core-level features.

    This replaces the older shortcut of assuming sample_id == patient_id and
    assuming all KOLL samples are RC. The crosswalk comes from
    KOLL_core_metadata.csv generated from Summary_UBC TMA + CE_summary.
    """
    feat = feat_df.copy()
    if feat.empty:
        return feat

    meta = load_koll_metadata(koll_metadata_csv)

    if "Panel" not in feat.columns:
        raise ValueError("KOLL feature table is missing Panel column.")
    if "cohort" not in feat.columns:
        feat["cohort"] = "KOLL"

    # Make a KOLL string matching key from coord/sample_id/sample_name.
    for c in ["coord", "sample_id", "sample_name"]:
        if c not in feat.columns:
            feat[c] = pd.NA
    best_core = _first_nonmissing_series(feat, ["coord", "sample_id", "sample_name"])
    feat["coord"] = feat["coord"].where(feat["coord"].notna(), best_core)
    feat["sample_id"] = feat["sample_id"].where(feat["sample_id"].notna(), best_core)
    feat["sample_name"] = feat["sample_name"].where(feat["sample_name"].notna(), best_core)
    feat["__koll_key"] = _first_nonmissing_series(feat, ["coord", "sample_id", "sample_name"]).map(_norm_match_key)
    feat["Panel"] = feat["Panel"].astype(str).str.strip()

    keep_meta_cols = [
        "Panel", "__koll_key", "patient_id", "TURBT_or_RC", "specimen_type",
        "sample_id", "sample_name", "coord", "cohort",
    ]
    keep_meta_cols = [c for c in keep_meta_cols if c in meta.columns]
    meta_small = meta[keep_meta_cols].copy()
    meta_small = meta_small.rename(columns={
        c: f"{c}_kollmeta" for c in keep_meta_cols if c not in {"Panel", "__koll_key"}
    })

    merged = feat.merge(meta_small, on=["Panel", "__koll_key"], how="left")

    n_unmatched = int(merged["patient_id_kollmeta"].isna().sum()) if "patient_id_kollmeta" in merged.columns else len(merged)
    if n_unmatched > 0:
        examples = (
            merged.loc[merged.get("patient_id_kollmeta", pd.Series(index=merged.index)).isna(), ["Panel", "coord", "sample_id", "sample_name"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        log(f"[WARN] KOLL metadata did not match {n_unmatched}/{len(merged)} feature rows. Examples: {examples}")

    # Prefer explicit KOLL metadata for patient/specimen linkage.
    for base_col in ["patient_id", "TURBT_or_RC", "specimen_type"]:
        meta_col = f"{base_col}_kollmeta"
        if meta_col in merged.columns:
            merged[base_col] = merged[meta_col].combine_first(merged[base_col] if base_col in merged.columns else pd.Series(pd.NA, index=merged.index))

    # Preserve core identifiers where missing.
    for base_col in ["sample_id", "sample_name", "coord", "cohort"]:
        meta_col = f"{base_col}_kollmeta"
        if meta_col in merged.columns:
            merged[base_col] = merged[base_col].combine_first(merged[meta_col]) if base_col in merged.columns else merged[meta_col]

    merged["cohort"] = "KOLL"
    if "TURBT_or_RC" in merged.columns:
        merged["TURBT_or_RC"] = merged["TURBT_or_RC"].map(normalize_sample_type)
    merged["matched_koll_metadata"] = merged.get("patient_id_kollmeta", pd.Series(pd.NA, index=merged.index)).notna()

    drop_cols = [c for c in merged.columns if c.endswith("_kollmeta") or c == "__koll_key"]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged

def prepare_core_level_feature_table(
    data_dict: Dict[str, pd.DataFrame],
    feature_group: str,
    cohort: str,
    panel: str,
    qc_acceptability: str,
    min_epi_fraction: Optional[float],
    sample_type="TURBT",
    koll_metadata_csv: Optional[str | Path] = DEFAULT_KOLL_METADATA_CSV,
) -> pd.DataFrame:
    if feature_group not in {"NN", "athena", "ratios", "cell_features", "triads"}:
        raise ValueError("feature_group must be one of {'NN', 'athena', 'ratios', 'cell_features', 'triads'}.")

    feat_df = data_dict[feature_group].copy()
    cohort = str(cohort)
    panel = str(panel)

    feat_df = standardize_and_validate_coord_df(feat_df, "feature_df", fail_on_missing=False)

    # Filter feature table to requested cohort / panel first. This works for
    # TMA, BLASST, and KOLL because prep_inputs.py writes cohort/Panel columns
    # into every feature family.
    if "cohort" in feat_df.columns:
        feat_df = feat_df[feat_df["cohort"].astype(str) == cohort].copy()
    if "Panel" in feat_df.columns:
        feat_df = feat_df[feat_df["Panel"].astype(str) == panel].copy()

    # KOLL adapter path ------------------------------------------------------
    # KOLL uses an explicit core metadata/crosswalk built from Summary_UBC TMA
    # and CE_summary_UBC. This lets KOLL behave like the other cohorts:
    # feature core/sample -> KOLL_core_metadata.csv -> patient_id/TURBT_or_RC.
    if cohort == "KOLL":
        core_df = merge_koll_metadata_to_feature_df(
            feat_df,
            koll_metadata_csv=koll_metadata_csv or DEFAULT_KOLL_METADATA_CSV,
        )
        if core_df.empty:
            return core_df

        core_df = apply_sample_type_filter(core_df, sample_type=sample_type)

        if min_epi_fraction is not None:
            log("[WARN] KOLL has no matching tissue segmentation table in this Stage-1 loader. Skipping min_epi_fraction filter for KOLL.")

        # Collapse to one row per KOLL core/sample, not directly to patient.
        # Patient-level aggregation happens later using patient_id.
        core_df = collapse_duplicate_rows(core_df, key_cols=["coord"])
        return core_df

    # Use cohort-appropriate metadata sources for TMA/BLASST -----------------
    if cohort == "BLASST":
        meta_df = data_dict["blasst_meta_df"].copy()
        qc_df = None
        tissue_seg_df = (
            data_dict["blasst_tissue_seg_df"].copy()
            if data_dict["blasst_tissue_seg_df"] is not None else None
        )
    else:
        meta_df = data_dict["tma_meta_df"].copy()
        qc_df = data_dict["tma_qc_df"].copy() if data_dict["tma_qc_df"] is not None else None
        tissue_seg_df = (
            data_dict["tissue_seg_df"].copy()
            if data_dict["tissue_seg_df"] is not None else None
        )

    meta_df = standardize_and_validate_coord_df(meta_df, "meta_df")
    qc_df = standardize_and_validate_coord_df(qc_df, "qc_df") if qc_df is not None else None
    tissue_seg_df = (
        standardize_and_validate_coord_df(tissue_seg_df, "tissue_seg_df")
        if tissue_seg_df is not None else None
    )

    core_df = feat_df.merge(meta_df, on="coord", how="left", suffixes=("", "_meta"))
    core_df = apply_sample_type_filter(core_df, sample_type=sample_type)
    if qc_df is not None:
        qc_small = qc_df.drop_duplicates("coord").copy()
        core_df = core_df.merge(qc_small, on="coord", how="left", suffixes=("", "_qc"))

    if tissue_seg_df is not None:
        tissue_small = tissue_seg_df.drop_duplicates("coord").copy()
        core_df = core_df.merge(tissue_small, on="coord", how="left", suffixes=("", "_seg"))

    # QC filter only for TMA cohorts
    if qc_df is not None and qc_acceptability != "all" and "structural_acceptability" in core_df.columns:
        sa = core_df["structural_acceptability"].astype(str).str.strip().str.lower()

        if qc_acceptability == "acceptable_only":
            core_df = core_df[sa.eq("acceptable")].copy()
        elif qc_acceptability == "acceptable_or_borderline":
            core_df = core_df[sa.isin(["acceptable", "borderline"])].copy()
        else:
            raise ValueError("qc_acceptability must be acceptable_only, acceptable_or_borderline, or all.")

    if min_epi_fraction is not None:
        epi_col = first_existing(
            core_df,
            ["Epi_region_area_percent", "Epi_region_area_fraction", "epi_region_area_percent", "epi_area_percent"]
        )
        if epi_col is None:
            log("[WARN] No epithelium area column found. Skipping epithelium filter.")
        else:
            epi = safe_numeric(core_df[epi_col])
            if epi.max(skipna=True) > 1.0:
                epi = epi / 100.0
            core_df = core_df[epi >= float(min_epi_fraction)].copy()

    core_df = collapse_duplicate_rows(core_df, key_cols=["coord"])
    return core_df


def merge_harmonized_to_core_df(
    core_df: pd.DataFrame,
    harmonized_df: pd.DataFrame,
    *,
    drop_missing_patient_id: bool = True,
) -> pd.DataFrame:
    """
    Link harmonized patient-level data through patient_id.

    If some rows lack patient_id after metadata merge, either drop them with
    a warning or fail loudly.
    """
    out = core_df.copy()

    out = ensure_patient_id_column(out)

    n_missing_pid = out["patient_id"].isna().sum()
    if n_missing_pid > 0:
        msg = (
            f"core_df has {n_missing_pid}/{len(out)} rows with missing patient_id "
            "before harmonized_df merge."
        )
        if drop_missing_patient_id:
            print(f"[WARN] {msg} Dropping those rows.", flush=True)
            out = out[out["patient_id"].notna()].copy()
        else:
            raise ValueError(msg)

    harm = harmonized_df.copy()
    harm["patient_id"] = harm["patient_id"].astype("string").str.strip()
    harm["patient_id"] = harm["patient_id"].replace({
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
        "": pd.NA,
    })

    rename_map = {
        c: f"{c}_harm"
        for c in harm.columns
        if c != "patient_id"
    }
    harm = harm.rename(columns=rename_map)

    merged = out.merge(
        harm,
        on="patient_id",
        how="left",
    )

    return merged

def harmonize_clinical_column_names(
    df: pd.DataFrame,
    *,
    drop_harm_columns: bool = True,
) -> pd.DataFrame:
    """
    Force modeling clinical/outcome columns to come from the harmonized dataframe.

    Assumes merge_harmonized_to_core_df(...) renamed harmonized columns with `_harm`.

    For each modeling column:
      - if <col>_harm exists, overwrite <col> with it
      - optionally drop all *_harm columns afterward
    """
    out = df.copy()

    force_from_harm = [
        "cohort",
        "cN",
        "pN",
        "cT",
        "pT",
        "Age",
        "Sex",
        "variant",
        "complete_response",
        "any_response",
        "OS_months_RC",
        "OS_months_TUR",
        "OS_event",
        "REC_months_RC",
        "REC_months_TURBT",
        "REC",
        "adjuvant_chemo",
        "TURBT_primary_histology",
        "TURBT_secondary_histology",
        "RC_primary_histology",
        "RC_secondary_histology",
    ]

    used_harm_cols = []

    for col in force_from_harm:
        harm_col = f"{col}_harm"
        if harm_col in out.columns:
            out[col] = out[harm_col]
            used_harm_cols.append(harm_col)

    if drop_harm_columns:
        harm_cols = [c for c in out.columns if c.endswith("_harm")]
        out = out.drop(columns=harm_cols, errors="ignore")

    return out

def choose_clinical_covariates(patient_df: pd.DataFrame, requested_covars: Sequence[str], missing_threshold: float = CLINICAL_MISSING_THRESHOLD) -> List[str]:
    keep = []
    for col in requested_covars:
        if col not in patient_df.columns:
            continue
        miss_frac = patient_df[col].isna().mean()
        nunique = patient_df[col].dropna().nunique()
        if miss_frac > missing_threshold:
            continue
        if nunique <= 1:
            continue
        keep.append(col)
    return keep


def impute_clinical_columns(df: pd.DataFrame, clinical_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in clinical_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            med = s.dropna().median() if s.dropna().shape[0] else np.nan
            out[col] = s.fillna(med)
        else:
            md = safe_mode(s)
            out[col] = s.fillna(md)
    return out


def get_feature_columns(core_df: pd.DataFrame, feature_group: str) -> List[str]:
    reserved = {
        "coord", "sample_id", "sample_name", "Panel", "cohort", "dataset", "feature_source", "chunk_dir", "path", "patient_id",
        "structural_acceptability", "__qc_file", "segmentation_comments",
        "tma", "TURBT_or_RC", "specimen_type", "sample_type", "matched_koll_metadata",
        "complete_response", "any_response", "OS_months_RC", "OS_months_TUR",
        "OS_event", "REC_months_RC", "REC_months_TURBT", "REC",
        "cN", "pN", "cT", "pT", "Age", "Sex", "variant", "adjuvant_chemo",
        "TURBT_primary_histology", "TURBT_secondary_histology",
        "RC_primary_histology", "RC_secondary_histology",
        "Epi_region_area_percent", "Other_region_area_percent", "Str_region_area_percent",
        "Epi_region_area_sq_microns", "Other_region_area_sq_microns", "Str_region_area_sq_microns",
    }
    feature_cols = [c for c in core_df.columns if c not in reserved]
    numeric_like = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(core_df[c]):
            numeric_like.append(c)
            continue
        coerced = pd.to_numeric(core_df[c], errors="coerce")
        if coerced.notna().any():
            core_df[c] = coerced
            numeric_like.append(c)
    return numeric_like


def chunk_feature_list(features: Sequence[str], chunk_idx: int, n_chunks: int) -> List[str]:
    if n_chunks <= 1:
        return list(features)
    features = list(features)
    chunks = np.array_split(features, n_chunks)
    if chunk_idx < 0 or chunk_idx >= len(chunks):
        raise ValueError(f"chunk_idx must be in [0, {len(chunks)-1}]")
    return list(chunks[chunk_idx])


def aggregate_core_to_patient(
    core_df: pd.DataFrame,
    feature_cols: Sequence[str],
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate core-level features to patient level.

    Feature columns are aggregated numerically.
    Metadata columns keep the first non-null value per patient.

    This function requires a valid non-missing patient_id column.
    """
    if "patient_id" not in core_df.columns:
        raise ValueError("core_df must contain patient_id before patient aggregation.")

    df = core_df.copy()

    df["patient_id"] = df["patient_id"].astype("string").str.strip()
    df["patient_id"] = df["patient_id"].replace({
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA,
        "": pd.NA,
    })

    n_missing_pid = df["patient_id"].isna().sum()
    if n_missing_pid > 0:
        raise ValueError(
            f"Found {n_missing_pid} rows with missing patient_id before aggregation."
        )

    # Feature aggregation
    if agg == "mean":
        feat_pat = df.groupby("patient_id", dropna=False)[list(feature_cols)].mean()
    elif agg == "median":
        feat_pat = df.groupby("patient_id", dropna=False)[list(feature_cols)].median()
    elif agg == "max":
        feat_pat = df.groupby("patient_id", dropna=False)[list(feature_cols)].max()
    elif agg == "min":
        feat_pat = df.groupby("patient_id", dropna=False)[list(feature_cols)].min()
    else:
        raise ValueError("agg must be one of: mean, median, max, min.")

    # Metadata aggregation
    meta_cols = [c for c in df.columns if c not in set(feature_cols)]
    meta_cols = [c for c in meta_cols if c != "patient_id"]

    meta_pat = (
        df.groupby("patient_id", dropna=False)[meta_cols]
        .agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
    )

    # Core counts
    ncore_pat = df.groupby("patient_id", dropna=False).size().rename("n_cores")

    out = pd.concat([ncore_pat, meta_pat, feat_pat], axis=1).reset_index()

    return out



def filter_feature_columns_by_coverage(
    patient_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    min_nonmissing_frac: float = 0.70,
    min_unique: int = 3,
    min_nonzero: int = 5,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Drop sparse/degenerate features after patient aggregation and endpoint filtering.

    This is intentionally done at the patient level because the modeling unit is patient,
    not core. It is especially important for all-label triads, where many motif columns
    can be zero for most patients.
    """
    keep_features: List[str] = []
    rows = []

    n_patients = int(patient_df.shape[0])

    for f in feature_cols:
        if f not in patient_df.columns:
            rows.append({
                "feature": f,
                "status": "drop",
                "reason": "missing_from_patient_df",
                "n_patients": n_patients,
                "n_nonmissing": 0,
                "nonmissing_frac": 0.0,
                "n_unique": 0,
                "n_nonzero": 0,
            })
            continue

        x = safe_numeric(patient_df[f]).replace([np.inf, -np.inf], np.nan)
        n_nonmissing = int(x.notna().sum())
        nonmissing_frac = float(n_nonmissing / n_patients) if n_patients > 0 else 0.0
        n_unique = int(x.dropna().nunique())
        n_nonzero = int((x.fillna(0) != 0).sum())

        status = "keep"
        reason = ""

        if nonmissing_frac < float(min_nonmissing_frac):
            status = "drop"
            reason = "low_nonmissing"
        elif n_unique < int(min_unique):
            status = "drop"
            reason = "low_unique"
        elif n_nonzero < int(min_nonzero):
            status = "drop"
            reason = "low_nonzero"

        rows.append({
            "feature": f,
            "status": status,
            "reason": reason,
            "n_patients": n_patients,
            "n_nonmissing": n_nonmissing,
            "nonmissing_frac": nonmissing_frac,
            "n_unique": n_unique,
            "n_nonzero": n_nonzero,
        })

        if status == "keep":
            keep_features.append(f)

    return keep_features, pd.DataFrame(rows)


def build_response_endpoint(df: pd.DataFrame, endpoint: str) -> Tuple[pd.DataFrame, str]:
    if endpoint not in {"complete_response", "any_response"}:
        raise ValueError("Response endpoint must be complete_response or any_response.")
    out = df.copy()
    out[endpoint] = safe_numeric(out[endpoint])
    out = out.dropna(subset=[endpoint]).copy()
    out[endpoint] = out[endpoint].astype(int)
    return out, endpoint


def build_survival_endpoint(df: pd.DataFrame, endpoint: str) -> Tuple[pd.DataFrame, str, str]:
    out = df.copy()
    if endpoint == "OS":
        event_col = "OS_event"
        rc_time = "OS_months_RC"
        tur_time = "OS_months_TUR"
    elif endpoint == "RFS":
        event_col = "REC"
        rc_time = "REC_months_RC"
        tur_time = "REC_months_TURBT"
    else:
        raise ValueError("Survival endpoint must be OS or RFS.")
    if event_col not in out.columns:
        raise ValueError(f"Missing event column: {event_col}")
    type_col = first_existing(out, ["TURBT_or_RC", "tma", "sample_type", "specimen_type"])
    if type_col is not None:
        sample_type = out[type_col].astype(str).str.upper().str.strip()
        use_rc = sample_type.str.contains("RC", na=False)
    else:
        use_rc = safe_numeric(out.get(rc_time, pd.Series(index=out.index, dtype=float))).notna()
    out["_time"] = np.where(use_rc, safe_numeric(out.get(rc_time, pd.Series(index=out.index))), safe_numeric(out.get(tur_time, pd.Series(index=out.index))))
    out["_event"] = safe_numeric(out[event_col])
    out = out.dropna(subset=["_time", "_event"]).copy()
    out["_event"] = out["_event"].astype(int)
    return out, "_time", "_event"

def fit_logistic_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    transform_mode: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage-1 predictive logistic path.

    Notes
    -----
    - raw: no scaling
    - zscore: fit StandardScaler on train, apply to valid
    - log1p_zscore: requires all values >= 0, applies log1p then train-fitted zscore
    """
    Xtr = X_train.copy()
    Xva = X_valid.copy()

    if transform_mode == "raw":
        X_train_sc = Xtr.values
        X_valid_sc = Xva.values

    elif transform_mode == "zscore":
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(Xtr)
        X_valid_sc = scaler.transform(Xva)

    elif transform_mode == "log1p_zscore":
        if (Xtr < 0).any().any() or (Xva < 0).any().any():
            raise ValueError("log1p_zscore requested but design matrix has negative values.")
        Xtr = np.log1p(Xtr)
        Xva = np.log1p(Xva)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(Xtr)
        X_valid_sc = scaler.transform(Xva)

    else:
        raise ValueError(f"Unknown transform_mode: {transform_mode}")

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sc, y_train)
    train_prob = model.predict_proba(X_train_sc)[:, 1]
    valid_prob = model.predict_proba(X_valid_sc)[:, 1]
    return train_prob, valid_prob


def run_response_cv_single_feature(
    patient_df: pd.DataFrame,
    feature_name: str,
    endpoint_col: str,
    clinical_covars: Sequence[str],
    n_splits: int,
    random_state: int = RANDOM_STATE,
    transform_mode: str = "zscore",
) -> Dict[str, object]:
    result = {"feature": feature_name, "status": "ok", "reason": "", "summary": None, "fold_df": None, "oof_df": None}
    try:
        cols = ["patient_id", endpoint_col, feature_name] + list(clinical_covars)
        cols = [c for c in cols if c in patient_df.columns]
        df = patient_df[cols].copy()
        df[endpoint_col] = safe_numeric(df[endpoint_col])
        df[feature_name] = safe_numeric(df[feature_name])
        df = df.dropna(subset=[endpoint_col, feature_name]).copy()
        if df.shape[0] < MIN_PATIENTS_RESPONSE:
            result["status"] = "skip"
            result["reason"] = f"too_few_patients_after_filter:{df.shape[0]}"
            return result
        y = df[endpoint_col].astype(int)
        if y.nunique() < 2:
            result["status"] = "skip"
            result["reason"] = "outcome_has_single_class"
            return result
        if y.sum() < MIN_POS_PER_CLASS or (len(y) - y.sum()) < MIN_POS_PER_CLASS:
            result["status"] = "skip"
            result["reason"] = f"class_too_small:pos={int(y.sum())},neg={int(len(y)-y.sum())}"
            return result
        df = impute_clinical_columns(df, clinical_covars)
        X_feat = df[[feature_name]].copy()
        X_clin = pd.get_dummies(df[list(clinical_covars)].copy(), drop_first=True) if len(clinical_covars) else pd.DataFrame(index=df.index)
        X_full = pd.concat([X_clin, X_feat], axis=1)
        def drop_zero_var(X):
            if X.shape[1] == 0:
                return X
            keep = X.nunique(dropna=False) > 1
            return X.loc[:, keep]
        X_feat = drop_zero_var(X_feat)
        X_clin = drop_zero_var(X_clin)
        X_full = drop_zero_var(X_full)
        if X_feat.shape[1] == 0:
            result["status"] = "skip"
            result["reason"] = "feature_zero_variance"
            return result
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_rows, oof_rows = [], []
        for fold_idx, (tr, va) in enumerate(splitter.split(df, y), start=1):
            tr_df = df.iloc[tr].copy()
            va_df = df.iloc[va].copy()
            y_tr = tr_df[endpoint_col].astype(int)
            y_va = va_df[endpoint_col].astype(int)
            if y_tr.nunique() < 2 or y_va.nunique() < 2:
                fold_rows.append({"feature": feature_name, "fold": fold_idx, "status": "skip_fold", "reason": "degenerate_train_or_valid_class"})
                continue
            Xf_tr, Xf_va = X_feat.iloc[tr].copy(), X_feat.iloc[va].copy()
            Xc_tr = X_clin.iloc[tr].copy() if len(X_clin.columns) else pd.DataFrame(index=tr_df.index)
            Xc_va = X_clin.iloc[va].copy() if len(X_clin.columns) else pd.DataFrame(index=va_df.index)
            Xfull_tr, Xfull_va = X_full.iloc[tr].copy(), X_full.iloc[va].copy()
            def align_train_valid(Xtr, Xva):
                if Xtr.shape[1] == 0:
                    return Xtr, Xva
                keep = Xtr.nunique(dropna=False) > 1
                Xtr2 = Xtr.loc[:, keep]
                Xva2 = Xva.loc[:, Xtr2.columns]
                return Xtr2, Xva2
            Xf_tr, Xf_va = align_train_valid(Xf_tr, Xf_va)
            Xc_tr, Xc_va = align_train_valid(Xc_tr, Xc_va)
            Xfull_tr, Xfull_va = align_train_valid(Xfull_tr, Xfull_va)
            if Xf_tr.shape[1] == 0:
                fold_rows.append({"feature": feature_name, "fold": fold_idx, "status": "skip_fold", "reason": "feature_zero_variance_in_train_fold"})
                continue
            try:
                prob_tr_feat, prob_va_feat = fit_logistic_and_predict(Xf_tr, y_tr, Xf_va, transform_mode=transform_mode)
                auc_tr_feat = roc_auc_score(y_tr, prob_tr_feat) if y_tr.nunique() == 2 else np.nan
                auc_va_feat = roc_auc_score(y_va, prob_va_feat) if y_va.nunique() == 2 else np.nan
                auprc_va_feat = average_precision_score(y_va, prob_va_feat) if y_va.nunique() == 2 else np.nan
                thr_feat = pick_threshold_from_train(y_tr.values, prob_tr_feat)
                hat_va_feat = (prob_va_feat >= thr_feat).astype(int)
                acc_va_feat = accuracy_score(y_va, hat_va_feat)
                bal_va_feat = balanced_accuracy_score(y_va, hat_va_feat)
                sens_va_feat, spec_va_feat = sens_spec_from_preds(y_va.values, hat_va_feat)
            except Exception as e:
                fold_rows.append({"feature": feature_name, "fold": fold_idx, "status": "skip_fold", "reason": f"biomarker_model_failed:{type(e).__name__}"})
                continue
            prob_tr_clin = np.repeat(np.nan, len(y_tr))
            prob_va_clin = np.repeat(np.nan, len(y_va))
            auc_va_clin = np.nan
            if Xc_tr.shape[1] > 0:
                try:
                    prob_tr_clin, prob_va_clin = fit_logistic_and_predict(Xc_tr, y_tr, Xc_va, transform_mode=transform_mode)
                    auc_va_clin = roc_auc_score(y_va, prob_va_clin) if y_va.nunique() == 2 else np.nan
                except Exception:
                    pass
            prob_tr_full = np.repeat(np.nan, len(y_tr))
            prob_va_full = np.repeat(np.nan, len(y_va))
            auc_va_full = np.nan
            if Xfull_tr.shape[1] > 0:
                try:
                    prob_tr_full, prob_va_full = fit_logistic_and_predict(Xfull_tr, y_tr, Xfull_va, transform_mode=transform_mode)
                    auc_va_full = roc_auc_score(y_va, prob_va_full) if y_va.nunique() == 2 else np.nan
                except Exception:
                    pass
            fold_rows.append({
                "feature": feature_name, "fold": fold_idx, "status": "ok",
                "n_train": len(tr), "n_valid": len(va),
                "n_train_pos": int(y_tr.sum()), "n_valid_pos": int(y_va.sum()),
                "train_auc_biomarker": auc_tr_feat,
                "valid_auc_biomarker": auc_va_feat,
                "valid_auprc_biomarker": auprc_va_feat,
                "valid_accuracy_biomarker": acc_va_feat,
                "valid_balanced_accuracy_biomarker": bal_va_feat,
                "valid_sensitivity_biomarker": sens_va_feat,
                "valid_specificity_biomarker": spec_va_feat,
                "valid_auc_clinical": auc_va_clin,
                "valid_auc_clinical_plus_biomarker": auc_va_full,
                "delta_auc_vs_clinical": auc_va_full - auc_va_clin if pd.notna(auc_va_full) and pd.notna(auc_va_clin) else np.nan,
            })
            for pid, yy, p1, p2, p3 in zip(va_df["patient_id"].astype(str), y_va.values, prob_va_feat, prob_va_clin, prob_va_full):
                oof_rows.append({
                    "feature": feature_name, "fold": fold_idx, "patient_id": pid,
                    "y_true": int(yy),
                    "prob_biomarker": float(p1) if pd.notna(p1) else np.nan,
                    "prob_clinical": float(p2) if pd.notna(p2) else np.nan,
                    "prob_clinical_plus_biomarker": float(p3) if pd.notna(p3) else np.nan,
                })
        fold_df = pd.DataFrame(fold_rows)
        oof_df = pd.DataFrame(oof_rows)
        if oof_df.empty:
            result["status"] = "skip"
            result["reason"] = "no_valid_oof_predictions"
            result["fold_df"] = fold_df
            result["oof_df"] = oof_df
            return result
        summary = {"feature": feature_name, "status": "ok"}
        pooled_bio = pooled_classification_metrics(oof_df["y_true"].values, oof_df["prob_biomarker"].values)
        summary.update({f"biomarker_{k}": v for k, v in pooled_bio.items()})
        if oof_df["prob_clinical"].notna().sum() > 0:
            pooled_clin = pooled_classification_metrics(oof_df.loc[oof_df["prob_clinical"].notna(), "y_true"].values, oof_df.loc[oof_df["prob_clinical"].notna(), "prob_clinical"].values)
            summary.update({f"clinical_{k}": v for k, v in pooled_clin.items()})
        else:
            summary.update({f"clinical_{k}": np.nan for k in pooled_bio.keys()})
        if oof_df["prob_clinical_plus_biomarker"].notna().sum() > 0:
            pooled_full = pooled_classification_metrics(oof_df.loc[oof_df["prob_clinical_plus_biomarker"].notna(), "y_true"].values, oof_df.loc[oof_df["prob_clinical_plus_biomarker"].notna(), "prob_clinical_plus_biomarker"].values)
            summary.update({f"clinical_plus_biomarker_{k}": v for k, v in pooled_full.items()})
        else:
            summary.update({f"clinical_plus_biomarker_{k}": np.nan for k in pooled_bio.keys()})
        summary["delta_oof_auc_vs_clinical"] = summary["clinical_plus_biomarker_oof_auc"] - summary["clinical_oof_auc"] if pd.notna(summary.get("clinical_plus_biomarker_oof_auc")) and pd.notna(summary.get("clinical_oof_auc")) else np.nan
        ok_folds = fold_df[fold_df["status"] == "ok"].copy()
        summary["n_folds_ok"] = int(ok_folds.shape[0])
        for col in ["valid_auc_biomarker", "valid_auprc_biomarker", "valid_balanced_accuracy_biomarker", "valid_auc_clinical", "valid_auc_clinical_plus_biomarker", "delta_auc_vs_clinical"]:
            if col in ok_folds.columns and ok_folds[col].notna().any():
                summary[f"{col}_mean"] = ok_folds[col].mean()
                summary[f"{col}_std"] = ok_folds[col].std()
            else:
                summary[f"{col}_mean"] = np.nan
                summary[f"{col}_std"] = np.nan
        result["summary"] = summary
        result["fold_df"] = fold_df
        result["oof_df"] = oof_df
        return result
    except Exception as e:
        result["status"] = "fail"
        result["reason"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        return result


def fit_cox_and_predict_risk(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    predictor_cols: Sequence[str],
    transform_mode: str = "raw",
    penalizer: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage-1 predictive Cox path with leakage-safe feature transforms.

    Continuous columns are transformed using training-fold parameters only.
    Binary dummy columns are left unchanged.
    """
    Xtr = train_df[list(predictor_cols)].copy()
    Xva = valid_df[list(predictor_cols)].copy()

    Xtr = pd.get_dummies(Xtr, drop_first=True)
    Xva = pd.get_dummies(Xva, drop_first=True)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0)

    if Xtr.shape[1] == 0:
        raise ValueError("No usable predictors after encoding.")

    keep = Xtr.nunique(dropna=False) > 1
    Xtr = Xtr.loc[:, keep]
    Xva = Xva.loc[:, Xtr.columns]

    if Xtr.shape[1] == 0:
        raise ValueError("All predictors are constant in training fold.")

    # Apply transform only to non-binary columns
    cont_cols = [c for c in Xtr.columns if not _is_binary_series(Xtr[c])]

    if len(cont_cols) > 0:
        for c in cont_cols:
            tr_s, va_s = _transform_train_valid_feature(
                Xtr[c], Xva[c], transform_mode=transform_mode
            )
            Xtr[c] = tr_s
            Xva[c] = va_s

    cox_train = pd.concat(
        [train_df[[time_col, event_col]].reset_index(drop=True), Xtr.reset_index(drop=True)],
        axis=1,
    )
    cox_valid = pd.concat(
        [valid_df[[time_col, event_col]].reset_index(drop=True), Xva.reset_index(drop=True)],
        axis=1,
    )

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_train, duration_col=time_col, event_col=event_col)

    risk_train = cph.predict_partial_hazard(cox_train[Xtr.columns]).values.reshape(-1)
    risk_valid = cph.predict_partial_hazard(cox_valid[Xtr.columns]).values.reshape(-1)
    return risk_train, risk_valid


def run_survival_cv_single_feature(
    patient_df: pd.DataFrame,
    feature_name: str,
    time_col: str,
    event_col: str,
    clinical_covars: Sequence[str],
    n_splits: int,
    random_state: int = RANDOM_STATE,
    transform_mode: str = "raw",
    cox_penalizer: float = 0.01,
) -> Dict[str, object]:
    result = {"feature": feature_name, "status": "ok", "reason": "", "summary": None, "fold_df": None, "oof_df": None}
    try:
        cols = ["patient_id", time_col, event_col, feature_name] + list(clinical_covars)
        cols = [c for c in cols if c in patient_df.columns]
        df = patient_df[cols].copy()
        df[time_col] = safe_numeric(df[time_col])
        df[event_col] = safe_numeric(df[event_col])
        df[feature_name] = safe_numeric(df[feature_name])
        df = df.dropna(subset=[time_col, event_col, feature_name]).copy()
        df[event_col] = df[event_col].astype(int)
        if df.shape[0] < MIN_PATIENTS_SURV:
            result["status"] = "skip"
            result["reason"] = f"too_few_patients_after_filter:{df.shape[0]}"
            return result
        if df[event_col].sum() < MIN_EVENTS_SURV:
            result["status"] = "skip"
            result["reason"] = f"too_few_events:{int(df[event_col].sum())}"
            return result
        df = impute_clinical_columns(df, clinical_covars)
        event_counts = df[event_col].value_counts(dropna=False)
        if df[event_col].nunique() == 2 and event_counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iter = splitter.split(df, df[event_col].astype(int))
            splitter_name = "event_stratified_kfold"
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iter = splitter.split(df)
            splitter_name = "kfold"
        fold_rows, oof_rows = [], []
        for fold_idx, (tr, va) in enumerate(split_iter, start=1):
            tr_df = df.iloc[tr].copy()
            va_df = df.iloc[va].copy()
            n_events_tr = int(tr_df[event_col].sum())
            n_events_va = int(va_df[event_col].sum())
            if n_events_tr < MIN_EVENTS_SURV:
                fold_rows.append({"feature": feature_name, "fold": fold_idx, "status": "skip_fold", "reason": f"too_few_train_events:{n_events_tr}"})
                continue
            try:
                risk_tr_bio, risk_va_bio = fit_cox_and_predict_risk(tr_df, va_df, time_col, event_col, predictor_cols=[feature_name], transform_mode=transform_mode, penalizer=cox_penalizer)
                c_train_bio = concordance_index(tr_df[time_col], -risk_tr_bio, tr_df[event_col])
                c_valid_bio = concordance_index(va_df[time_col], -risk_va_bio, va_df[event_col]) if n_events_va > 0 else np.nan
            except Exception as e:
                fold_rows.append({"feature": feature_name, "fold": fold_idx, "status": "skip_fold", "reason": f"biomarker_cox_failed:{type(e).__name__}"})
                continue
            risk_va_clin = np.repeat(np.nan, len(va_df))
            c_valid_clin = np.nan
            clinical_use = [c for c in clinical_covars if c in tr_df.columns]
            clinical_fail_reason = ""

            if len(clinical_use):
                try:
                    _, risk_va_clin = fit_cox_and_predict_risk(
                        tr_df, va_df, time_col, event_col, predictor_cols=clinical_use, transform_mode=transform_mode, penalizer=cox_penalizer)
                    c_valid_clin = (
                        concordance_index(va_df[time_col], -risk_va_clin, va_df[event_col])
                        if n_events_va > 0 else np.nan
                    )
                except Exception as e:
                    clinical_fail_reason = f"{type(e).__name__}: {e}"
                    print(
                        f"[WARN] clinical-only Cox failed | feature={feature_name} | fold={fold_idx} | reason={clinical_fail_reason}",
                        flush=True,
                    )
            risk_va_full = np.repeat(np.nan, len(va_df))
            c_valid_full = np.nan
            full_use = clinical_use + [feature_name]
            try:
                _, risk_va_full = fit_cox_and_predict_risk(tr_df, va_df, time_col, event_col, predictor_cols=full_use, transform_mode=transform_mode, penalizer=cox_penalizer)
                c_valid_full = concordance_index(va_df[time_col], -risk_va_full, va_df[event_col]) if n_events_va > 0 else np.nan
            except Exception:
                pass
            fold_rows.append({
                "feature": feature_name, "fold": fold_idx, "status": "ok",
                "n_train": len(tr_df), "n_valid": len(va_df),
                "n_train_events": n_events_tr, "n_valid_events": n_events_va,
                "train_cindex_biomarker": c_train_bio,
                "valid_cindex_biomarker": c_valid_bio,
                "valid_cindex_clinical": c_valid_clin,
                "valid_cindex_clinical_plus_biomarker": c_valid_full,
                "delta_cindex_vs_clinical": c_valid_full - c_valid_clin if pd.notna(c_valid_full) and pd.notna(c_valid_clin) else np.nan,
                "clinical_fail_reason": clinical_fail_reason,
            })
            for pid, tt, ee, r1, r2, r3 in zip(va_df["patient_id"].astype(str), va_df[time_col].values, va_df[event_col].values, risk_va_bio, risk_va_clin, risk_va_full):
                oof_rows.append({
                    "feature": feature_name, "fold": fold_idx, "patient_id": pid,
                    "time": float(tt), "event": int(ee),
                    "risk_biomarker": float(r1) if pd.notna(r1) else np.nan,
                    "risk_clinical": float(r2) if pd.notna(r2) else np.nan,
                    "risk_clinical_plus_biomarker": float(r3) if pd.notna(r3) else np.nan,
                })
        fold_df = pd.DataFrame(fold_rows)
        oof_df = pd.DataFrame(oof_rows)
        if oof_df.empty:
            result["status"] = "skip"
            result["reason"] = "no_valid_oof_predictions"
            result["fold_df"] = fold_df
            result["oof_df"] = oof_df
            return result
        summary = {"feature": feature_name, "status": "ok"}
        try:
            summary["biomarker_oof_cindex"] = concordance_index(oof_df["time"], -oof_df["risk_biomarker"], oof_df["event"])
        except Exception:
            summary["biomarker_oof_cindex"] = np.nan
        if oof_df["risk_clinical"].notna().sum() > 0:
            try:
                tmp = oof_df.loc[oof_df["risk_clinical"].notna()]
                summary["clinical_oof_cindex"] = concordance_index(tmp["time"], -tmp["risk_clinical"], tmp["event"])
            except Exception:
                summary["clinical_oof_cindex"] = np.nan
        else:
            summary["clinical_oof_cindex"] = np.nan
        if oof_df["risk_clinical_plus_biomarker"].notna().sum() > 0:
            try:
                tmp = oof_df.loc[oof_df["risk_clinical_plus_biomarker"].notna()]
                summary["clinical_plus_biomarker_oof_cindex"] = concordance_index(tmp["time"], -tmp["risk_clinical_plus_biomarker"], tmp["event"])
            except Exception:
                summary["clinical_plus_biomarker_oof_cindex"] = np.nan
        else:
            summary["clinical_plus_biomarker_oof_cindex"] = np.nan
        summary["delta_oof_cindex_vs_clinical"] = summary["clinical_plus_biomarker_oof_cindex"] - summary["clinical_oof_cindex"] if pd.notna(summary["clinical_plus_biomarker_oof_cindex"]) and pd.notna(summary["clinical_oof_cindex"]) else np.nan
        ok_folds = fold_df[fold_df["status"] == "ok"].copy()
        summary["n_folds_ok"] = int(ok_folds.shape[0])
        for col in ["valid_cindex_biomarker", "valid_cindex_clinical", "valid_cindex_clinical_plus_biomarker", "delta_cindex_vs_clinical"]:
            if col in ok_folds.columns and ok_folds[col].notna().any():
                summary[f"{col}_mean"] = ok_folds[col].mean()
                summary[f"{col}_std"] = ok_folds[col].std()
            else:
                summary[f"{col}_mean"] = np.nan
                summary[f"{col}_std"] = np.nan
        result["summary"] = summary
        result["fold_df"] = fold_df
        result["oof_df"] = oof_df
        return result
    except Exception as e:
        result["status"] = "fail"
        result["reason"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        return result


def _combine_repeated_single_feature_results(
    results: Sequence[Dict[str, object]],
    feature_name: str,
    metric_keys: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Combine repeated-CV runs for a single feature."""
    ok_results = [r for r in results if r.get("summary") is not None]
    fold_parts, oof_parts, fail_reasons = [], [], []
    summary_rows = []

    for rep_idx, r in enumerate(results, start=1):
        if isinstance(r.get("fold_df"), pd.DataFrame) and not r["fold_df"].empty:
            tmp = r["fold_df"].copy()
            tmp["repeat"] = rep_idx
            fold_parts.append(tmp)
        if isinstance(r.get("oof_df"), pd.DataFrame) and not r["oof_df"].empty:
            tmp = r["oof_df"].copy()
            tmp["repeat"] = rep_idx
            oof_parts.append(tmp)
        if r.get("summary") is not None:
            row = dict(r["summary"])
            row["repeat"] = rep_idx
            summary_rows.append(row)
        if r.get("status") in {"skip", "fail"}:
            fail_reasons.append(str(r.get("reason", "")))

    if not ok_results:
        return {
            "feature": feature_name,
            "status": "skip",
            "reason": ";".join([x for x in fail_reasons if x]) or "no_repeats_with_valid_summary",
            "summary": None,
            "fold_df": pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame(),
            "oof_df": pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(),
        }

    summary_df = pd.DataFrame(summary_rows)
    combined = {"feature": feature_name, "status": "ok", "n_repeats_ok": int(summary_df.shape[0])}

    # Preserve legacy summary column names as the repeat-mean values.
    for col in summary_df.columns:
        if col in {"feature", "status", "repeat"}:
            continue
        if pd.api.types.is_numeric_dtype(summary_df[col]):
            combined[col] = summary_df[col].mean(skipna=True)
            combined[f"{col}_repeat_std"] = summary_df[col].std(skipna=True)
        else:
            vals = summary_df[col].dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                combined[col] = vals[0]

    fold_df = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return {"feature": feature_name, "status": "ok", "reason": "", "summary": combined, "fold_df": fold_df, "oof_df": oof_df}


def run_response_cv_single_feature_repeated(
    patient_df: pd.DataFrame,
    feature_name: str,
    endpoint_col: str,
    clinical_covars: Sequence[str],
    n_splits: int,
    n_repeats: int,
    random_state: int = RANDOM_STATE,
    transform_mode: str = "zscore",
) -> Dict[str, object]:
    results = []
    for rep in range(int(n_repeats)):
        results.append(
            run_response_cv_single_feature(
                patient_df=patient_df,
                feature_name=feature_name,
                endpoint_col=endpoint_col,
                clinical_covars=clinical_covars,
                n_splits=n_splits,
                random_state=random_state + rep * 1009,
                transform_mode=transform_mode,
            )
        )
    return _combine_repeated_single_feature_results(results, feature_name=feature_name)


def run_survival_cv_single_feature_repeated(
    patient_df: pd.DataFrame,
    feature_name: str,
    time_col: str,
    event_col: str,
    clinical_covars: Sequence[str],
    n_splits: int,
    n_repeats: int,
    random_state: int = RANDOM_STATE,
    transform_mode: str = "raw",
    cox_penalizer: float = 0.01,
) -> Dict[str, object]:
    results = []
    for rep in range(int(n_repeats)):
        results.append(
            run_survival_cv_single_feature(
                patient_df=patient_df,
                feature_name=feature_name,
                time_col=time_col,
                event_col=event_col,
                clinical_covars=clinical_covars,
                n_splits=n_splits,
                random_state=random_state + rep * 1009,
                transform_mode=transform_mode,
                cox_penalizer=cox_penalizer,
            )
        )
    return _combine_repeated_single_feature_results(results, feature_name=feature_name)


def get_rightmost_series(df: pd.DataFrame, colname: str) -> Optional[pd.Series]:
    """
    Return the rightmost column with this name as a Series.
    Handles duplicate column names safely.
    """
    mask = (df.columns == colname)
    if mask.sum() == 0:
        return None
    sub = df.loc[:, mask]
    return sub.iloc[:, -1]

def simplify_clinical_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert harmonized clinical variables into modeling-friendly numeric formats.

    Output:
      - Age -> numeric
      - Sex -> 1 for male, 0 for female
      - cN  -> extracted numeric stage
      - cT  -> extracted numeric stage
      - pN  -> extracted numeric stage
      - pT  -> extracted numeric stage where possible
    """
    out = df.copy()

    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")

    if "Sex" in out.columns:
        sex = (
            out["Sex"]
            .astype(object)          # important: do NOT keep StringDtype here
            .where(pd.notna(out["Sex"]), np.nan)
        )
        sex = pd.Series(sex, index=out.index).astype(str).str.strip().str.upper()
        sex = sex.replace({
            "MALE": 1,
            "M": 1,
            "FEMALE": 0,
            "F": 0,
            "NAN": np.nan,
            "NONE": np.nan,
            "<NA>": np.nan,
            "": np.nan,
        })
        out["Sex"] = pd.to_numeric(sex, errors="coerce")

    if "cN" in out.columns:
        cN = out["cN"].astype(object).where(pd.notna(out["cN"]), np.nan)
        cN = pd.Series(cN, index=out.index).astype(str).str.extract(r"(\d+)")[0]
        out["cN"] = pd.to_numeric(cN, errors="coerce")

    if "cT" in out.columns:
        cT = out["cT"].astype(object).where(pd.notna(out["cT"]), np.nan)
        cT = pd.Series(cT, index=out.index).astype(str).str.extract(r"(\d+)")[0]
        out["cT"] = pd.to_numeric(cT, errors="coerce")

    if "pN" in out.columns:
        pN = out["pN"].astype(object).where(pd.notna(out["pN"]), np.nan)
        pN = pd.Series(pN, index=out.index).astype(str).str.extract(r"(\d+)")[0]
        out["pN"] = pd.to_numeric(pN, errors="coerce")

    if "pT" in out.columns:
        pT = out["pT"].astype(object).where(pd.notna(out["pT"]), np.nan)
        # handles values like pT0, pT2, pTa, pTis by extracting numeric part if present
        pT = pd.Series(pT, index=out.index).astype(str).str.extract(r"(\d+)")[0]
        out["pT"] = pd.to_numeric(pT, errors="coerce")

    return out

def replace_with_harmonized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace unsuffixed columns with their *_harm counterparts in a clean,
    deterministic way.

    Strategy
    --------
    1. Identify all columns ending with '_harm'
    2. Compute the corresponding base names
    3. Drop BOTH:
         - all *_harm columns
         - any existing unsuffixed base columns for those names
    4. Add the harmonized columns back with the '_harm' suffix removed

    This guarantees the harmonized dataframe is the source of truth.
    """
    out = df.copy()

    # Do not let patient-level harmonized columns overwrite specimen/core-level
    # metadata. TURBT_or_RC should come from core metadata, especially for KOLL
    # where a patient could theoretically have separate TURBT and RC cores.
    protected_core_cols = {"TURBT_or_RC", "specimen_type", "sample_type"}
    harm_cols = [c for c in out.columns if c.endswith("_harm") and c[:-5] not in protected_core_cols]
    protected_harm_cols = [c for c in out.columns if c.endswith("_harm") and c[:-5] in protected_core_cols]
    if not harm_cols:
        return out.drop(columns=protected_harm_cols, errors="ignore")

    # base names like cN_harm -> cN
    base_names = [c[:-5] for c in harm_cols]

    # Build a clean harmonized dataframe with renamed columns
    harm_df = out[harm_cols].copy()
    harm_df.columns = base_names

    # Drop:
    #   1) all *_harm columns
    #   2) any pre-existing unsuffixed versions of those same columns
    cols_to_drop = set(harm_cols) | set(base_names) | set(protected_harm_cols)
    keep_cols = [c for c in out.columns if c not in cols_to_drop]

    out_clean = out[keep_cols].copy()

    # Concatenate clean base dataframe + harmonized replacements
    out_final = pd.concat([out_clean, harm_df], axis=1)

    return out_final

def run_job(cohort: str, 
            panel: str, 
            feature_group: str, 
            endpoint: str, 
            qc_acceptability: str, 
            min_epi_fraction: Optional[float], 
            agg: str, 
            n_splits: int, 
            n_repeats: int,
            chunk_idx: int, 
            n_chunks: int, 
            outdir: str,
            harmonized_path: str,
            clinical_vars: Sequence[str],
            patient_subset: str = "all",
            sample_type="TURBT",
            transform_mode: str = "zscore",
            feature_source: str = "phenotype_only",
            spatial_root: Optional[str | Path] = None,
            cell_features_path: Optional[str | Path] = None,
            triads_path: Optional[str | Path] = None,
            koll_metadata_csv: Optional[str | Path] = DEFAULT_KOLL_METADATA_CSV,
            min_feature_nonmissing_frac: float = 0.70,
            min_feature_unique: int = 3,
            min_feature_nonzero: int = 5,
            ) -> None:
    ensure_dir(outdir)
    log("=" * 80)
    log(f"[START] cohort={cohort} | panel={panel} | feature_source={feature_source} | feature_group={feature_group} | endpoint={endpoint}")
    log(f"[START] qc={qc_acceptability} | min_epi_fraction={min_epi_fraction} | agg={agg}")
    log(f"[START] chunk_idx={chunk_idx}/{n_chunks} | n_splits={n_splits} | n_repeats={n_repeats}")
    data_dict = load_data_dict(
        feature_group=feature_group,
        feature_source=feature_source,
        panels=[panel],
        cohorts=[cohort],
        spatial_root=spatial_root,
        cell_features_path=cell_features_path,
        triads_path=triads_path,
    )
    harm_df = load_harmonized_df(harmonized_path)
    core_df = prepare_core_level_feature_table(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=qc_acceptability,
        min_epi_fraction=min_epi_fraction,
        sample_type=sample_type,
        koll_metadata_csv=koll_metadata_csv,
    )
    if core_df.empty:
        raise ValueError("No cores remain after requested filters.")
    core_df = merge_harmonized_to_core_df(core_df, harm_df)
    core_df = replace_with_harmonized_columns(core_df)
    core_df = simplify_clinical_vars(core_df)
    feature_cols = get_feature_columns(core_df, feature_group=feature_group)
    feature_cols = chunk_feature_list(feature_cols, chunk_idx=chunk_idx, n_chunks=n_chunks)
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found for this chunk.")

    core_df = ensure_patient_id_column(core_df)

    n_missing_pid = core_df["patient_id"].isna().sum()
    if n_missing_pid > 0:
        raise ValueError(
            f"core_df has {n_missing_pid}/{len(core_df)} rows with missing patient_id "
            "before aggregation."
        )
    patient_df = aggregate_core_to_patient(core_df, feature_cols=feature_cols, agg=agg)    
    if "cohort" in patient_df.columns:
        patient_df = patient_df[patient_df["cohort"].astype(str) == str(cohort)].copy()
    # Apply adjuvant-chemotherapy patient subsets only where biologically relevant:
    # RC survival analyses in RC-only / cystectomy cohorts that have adjuvant_chemo.
    if cohort in {"No-NAC", "KOLL"} and endpoint in {"OS", "RFS"} and sample_type in {"TURBT", "RC", "all"}:
        patient_df = apply_patient_subset(patient_df, patient_subset=patient_subset)
    else:
        if patient_subset != "all":
            log(f"[WARN] patient_subset={patient_subset} requested outside No-NAC/KOLL survival context; resetting to all.")
        patient_subset = "all"
    log(f"[INFO] patient_subset={patient_subset} -> n={len(patient_df)}")
    if patient_df.empty:
        raise ValueError("No patients remain after aggregation.")
    clinical_covars = choose_clinical_covariates(patient_df, requested_covars=clinical_vars)
    log(f"[INFO] clinical_covars_kept={clinical_covars}")
    summary_rows, fold_parts, oof_parts, fail_rows = [], [], [], []
    full_model_rows = []
    feature_filter_df = pd.DataFrame()
    if endpoint in {"complete_response", "any_response"}:
        patient_df, y_col = build_response_endpoint(patient_df, endpoint=endpoint)
        feature_cols, feature_filter_df = filter_feature_columns_by_coverage(
            patient_df,
            feature_cols,
            min_nonmissing_frac=min_feature_nonmissing_frac,
            min_unique=min_feature_unique,
            min_nonzero=min_feature_nonzero,
        )
        log(f"[INFO] features_after_coverage_filter={len(feature_cols)}")
        if len(feature_cols) == 0:
            raise ValueError("No features remain after patient-level coverage filtering.")
        for i, feature_name in enumerate(feature_cols, start=1):
            if i % 50 == 0:
                log(f"[PROGRESS] {i}/{len(feature_cols)} features")
            res = run_response_cv_single_feature_repeated(patient_df=patient_df, feature_name=feature_name, endpoint_col=y_col, clinical_covars=clinical_covars, n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE, transform_mode=transform_mode)
            if res["summary"] is not None:
                row = dict(res["summary"])
                row.update({"cohort": cohort, "panel": panel, "feature_source": feature_source, "feature_group": feature_group, "endpoint": endpoint, "patient_subset": patient_subset, "sample_type": sample_type, "qc_acceptability": qc_acceptability, "min_epi_fraction": min_epi_fraction, "agg": agg, "n_splits": n_splits, "n_repeats": n_repeats, "chunk_idx": chunk_idx, "n_chunks": n_chunks, "transform_mode": transform_mode})
                summary_rows.append(row)
            if isinstance(res["fold_df"], pd.DataFrame) and not res["fold_df"].empty:
                tmp = res["fold_df"].copy(); tmp["cohort"] = cohort; tmp["panel"] = panel; tmp["feature_source"] = feature_source; tmp["feature_group"] = feature_group; tmp["endpoint"] = endpoint; tmp["patient_subset"] = patient_subset; tmp["sample_type"] = sample_type; tmp["transform_mode"] = transform_mode; tmp["chunk_idx"] = chunk_idx; fold_parts.append(tmp)
            if isinstance(res["oof_df"], pd.DataFrame) and not res["oof_df"].empty:
                tmp = res["oof_df"].copy(); tmp["cohort"] = cohort; tmp["panel"] = panel; tmp["feature_source"] = feature_source; tmp["feature_group"] = feature_group; tmp["endpoint"] = endpoint; tmp["patient_subset"] = patient_subset; tmp["sample_type"] = sample_type; tmp["transform_mode"] = transform_mode; tmp["chunk_idx"] = chunk_idx; oof_parts.append(tmp)
            if res["status"] in {"skip", "fail"}:
                fail_rows.append({"feature": feature_name, "status": res["status"], "reason": res.get("reason", ""), "cohort": cohort, "panel": panel, "feature_source": feature_source, "feature_group": feature_group, "endpoint": endpoint, "patient_subset": patient_subset, "sample_type": sample_type, "transform_mode": transform_mode, "chunk_idx": chunk_idx})
                        # Full-dataset inferential layer
            full_res = run_full_dataset_inference_single_feature(
                patient_df=patient_df,
                feature_name=feature_name,
                endpoint=endpoint,
                clinical_covars=clinical_covars,
                transform_modes=[transform_mode],
                cox_penalizer=0.0,
            )
            if full_res["rows"]:
                for rr in full_res["rows"]:
                    rr.update({
                        "cohort": cohort,
                        "panel": panel,
                        "feature_source": feature_source,
                        "feature_group": feature_group,
                        "endpoint": endpoint,
                        "patient_subset": patient_subset,
                        "sample_type": sample_type,
                        "transform_mode": transform_mode,
                        "qc_acceptability": qc_acceptability,
                        "min_epi_fraction": min_epi_fraction,
                        "agg": agg,
                        "chunk_idx": chunk_idx,
                        "n_chunks": n_chunks,
                    })
                    full_model_rows.append(rr)
    elif endpoint in {"OS", "RFS"}:
        patient_df, time_col, event_col = build_survival_endpoint(patient_df, endpoint=endpoint)
        feature_cols, feature_filter_df = filter_feature_columns_by_coverage(
            patient_df,
            feature_cols,
            min_nonmissing_frac=min_feature_nonmissing_frac,
            min_unique=min_feature_unique,
            min_nonzero=min_feature_nonzero,
        )
        log(f"[INFO] features_after_coverage_filter={len(feature_cols)}")
        if len(feature_cols) == 0:
            raise ValueError("No features remain after patient-level coverage filtering.")
        for i, feature_name in enumerate(feature_cols, start=1):
            if i % 50 == 0:
                log(f"[PROGRESS] {i}/{len(feature_cols)} features")
            res = run_survival_cv_single_feature_repeated(patient_df=patient_df, feature_name=feature_name, time_col=time_col, event_col=event_col, clinical_covars=clinical_covars, n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE, transform_mode=transform_mode)
            if res["summary"] is not None:
                row = dict(res["summary"])
                row.update({"cohort": cohort, "panel": panel, "feature_source": feature_source, "feature_group": feature_group, "endpoint": endpoint, "patient_subset": patient_subset, "sample_type": sample_type, "qc_acceptability": qc_acceptability, "min_epi_fraction": min_epi_fraction, "agg": agg, "n_splits": n_splits, "n_repeats": n_repeats, "chunk_idx": chunk_idx, "n_chunks": n_chunks, "transform_mode": transform_mode})
                summary_rows.append(row)
            if isinstance(res["fold_df"], pd.DataFrame) and not res["fold_df"].empty:
                tmp = res["fold_df"].copy(); tmp["cohort"] = cohort; tmp["panel"] = panel; tmp["feature_source"] = feature_source; tmp["feature_group"] = feature_group; tmp["endpoint"] = endpoint; tmp["patient_subset"] = patient_subset; tmp["sample_type"] = sample_type; tmp["transform_mode"] = transform_mode; tmp["chunk_idx"] = chunk_idx; fold_parts.append(tmp)
            if isinstance(res["oof_df"], pd.DataFrame) and not res["oof_df"].empty:
                tmp = res["oof_df"].copy(); tmp["cohort"] = cohort; tmp["panel"] = panel; tmp["feature_source"] = feature_source; tmp["feature_group"] = feature_group; tmp["endpoint"] = endpoint; tmp["patient_subset"] = patient_subset; tmp["sample_type"] = sample_type; tmp["transform_mode"] = transform_mode; tmp["chunk_idx"] = chunk_idx; oof_parts.append(tmp)
            if res["status"] in {"skip", "fail"}:
                fail_rows.append({"feature": feature_name, "status": res["status"], "reason": res.get("reason", ""), "cohort": cohort, "panel": panel, "feature_source": feature_source, "feature_group": feature_group, "endpoint": endpoint, "patient_subset": patient_subset,"sample_type": sample_type, "transform_mode": transform_mode, "chunk_idx": chunk_idx})
                        # Full-dataset inferential layer
            full_res = run_full_dataset_inference_single_feature(
                patient_df=patient_df,
                feature_name=feature_name,
                endpoint=endpoint,
                clinical_covars=clinical_covars,
                transform_modes=[transform_mode],
                cox_penalizer=0.0,
            )
            if full_res["rows"]:
                for rr in full_res["rows"]:
                    rr.update({
                        "cohort": cohort,
                        "panel": panel,
                        "feature_source": feature_source,
                        "feature_group": feature_group,
                        "endpoint": endpoint,
                        "patient_subset": patient_subset,
                        "sample_type": sample_type,
                        "transform_mode": transform_mode,
                        "qc_acceptability": qc_acceptability,
                        "min_epi_fraction": min_epi_fraction,
                        "agg": agg,
                        "chunk_idx": chunk_idx,
                        "n_chunks": n_chunks,
                    })
                    full_model_rows.append(rr)
    else:
        raise ValueError("endpoint must be one of OS, RFS, complete_response, any_response.")
    stem = f"{cohort}__{panel}__{feature_source}__{feature_group}__{endpoint}__{sample_type}__{patient_subset}__agg-{agg}__transform-{transform_mode}__chunk{chunk_idx:03d}of{n_chunks:03d}"
    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    fail_df = pd.DataFrame(fail_rows)
    full_model_df = pd.DataFrame(full_model_rows)
    summary_fp = Path(outdir) / f"{stem}__summary.csv"
    fold_fp = Path(outdir) / f"{stem}__folds.csv"
    oof_fp = Path(outdir) / f"{stem}__oof.csv"
    fail_fp = Path(outdir) / f"{stem}__failures.csv"
    meta_fp = Path(outdir) / f"{stem}__runmeta.json"
    full_model_fp = Path(outdir) / f"{stem}__fullmodels.csv"
    feature_filter_fp = Path(outdir) / f"{stem}__feature_filter.csv"
    summary_df.to_csv(summary_fp, index=False)
    fold_df.to_csv(fold_fp, index=False)
    oof_df.to_csv(oof_fp, index=False)
    fail_df.to_csv(fail_fp, index=False)
    full_model_df.to_csv(full_model_fp, index=False)
    feature_filter_df.to_csv(feature_filter_fp, index=False)
    runmeta = {
        "cohort": cohort, "panel": panel, "feature_source": feature_source, "feature_group": feature_group, "endpoint": endpoint,
        "sample_type": sample_type, "patient_subset": patient_subset,
        "qc_acceptability": qc_acceptability, "min_epi_fraction": min_epi_fraction, "agg": agg, "transform_mode": transform_mode,
        "n_splits": n_splits, "n_repeats": n_repeats, "chunk_idx": chunk_idx, "n_chunks": n_chunks,
        "min_feature_nonmissing_frac": min_feature_nonmissing_frac,
        "min_feature_unique": min_feature_unique,
        "min_feature_nonzero": min_feature_nonzero,
        "clinical_covars_kept": list(clinical_covars),
        "n_features_after_coverage_filter": len(feature_cols),
        "n_features_dropped_by_coverage": int((feature_filter_df["status"] == "drop").sum()) if not feature_filter_df.empty and "status" in feature_filter_df.columns else 0,
        "n_features_with_summary": int(summary_df.shape[0]),
        "n_fail_rows": int(fail_df.shape[0]),
        "n_full_model_rows": int(full_model_df.shape[0]),
    }
    with open(meta_fp, "w") as f:
        json.dump(runmeta, f, indent=2)
    log(f"[DONE] summary -> {summary_fp}")
    log(f"[DONE] folds   -> {fold_fp}")
    log(f"[DONE] oof     -> {oof_fp}")
    log(f"[DONE] fails   -> {fail_fp}")
    log(f"[DONE] runmeta -> {meta_fp}")
    log(f"[DONE] fullmodels -> {full_model_fp}")
    log(f"[DONE] feature_filter -> {feature_filter_fp}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 univariate CV screen for mIF biomarkers.")
    p.add_argument("--cohort", required=True, type=str)
    p.add_argument("--panel", required=True, type=str, choices=["AR", "BT", "MY"])
    p.add_argument("--feature-group", required=True, type=str, choices=["NN", "athena", "cell_features", "ratios", "triads"])
    p.add_argument("--endpoint", required=True, type=str, choices=["OS", "RFS", "complete_response", "any_response"])
    p.add_argument("--qc-acceptability", default="acceptable_or_borderline", type=str, choices=["acceptable_only", "acceptable_or_borderline", "all"])
    p.add_argument("--min-epi-fraction", default=None, type=float)
    p.add_argument("--agg", default="median", type=str, choices=["mean", "median", "max", "min"])
    p.add_argument("--n-splits", default=DEFAULT_N_SPLITS, type=int)
    p.add_argument("--n-repeats", default=DEFAULT_N_REPEATS, type=int, help="Number of repeated CV runs. Use repeated 5-fold by setting --n-splits 5 --n-repeats >1.")
    p.add_argument("--chunk-idx", default=0, type=int)
    p.add_argument("--n-chunks", default=1, type=int)
    p.add_argument("--harmonized-path", default="/projects/ovcare/users/nikolay_alabi/immuno/data/harmonized_modeling_dataframe.csv", type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--clinical-vars", nargs="+", default=DEFAULT_CLINICAL_VARS, help="Requested clinical variables. Variables >50% missing are dropped per job.")
    p.add_argument(
        "--patient-subset",
        default="all",
        type=str,
        choices=["all", "no_adj_chemo", "adj_chemo"],
    )
    p.add_argument(
        "--sample-type",
        default="TURBT",
        choices=["all", "TURBT", "RC"],
    )
    p.add_argument(
        "--transform-mode",
        default="zscore",
        type=str,
        choices=["raw", "zscore", "log1p_zscore"],
        help="Transform used in stage-1 CV predictive models.",
    )
    p.add_argument(
        "--feature-source",
        default="phenotype_only",
        choices=sorted(FEATURE_SOURCE_CONFIG.keys()),
        help="Which reviewed feature namespace to use.",
    )
    p.add_argument("--spatial-root", default=None, type=str, help="Root containing chunked NNstats/ATHENA outputs. Defaults depend on --feature-source.")
    p.add_argument("--cell-features-path", default=None, type=str, help="Reviewed wide cell-feature CSV. Defaults depend on --feature-source.")
    p.add_argument("--triads-path", default=None, type=str, help="Reviewed wide triad CSV. Defaults depend on --feature-source.")
    p.add_argument("--koll-metadata-csv", default=str(DEFAULT_KOLL_METADATA_CSV), type=str, help="KOLL core metadata/crosswalk CSV with sample/core -> patient_id and TURBT_or_RC.")
    p.add_argument("--min-feature-nonmissing-frac", default=0.70, type=float)
    p.add_argument("--min-feature-unique", default=3, type=int)
    p.add_argument("--min-feature-nonzero", default=5, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_job(
        cohort=args.cohort,
        panel=args.panel,
        feature_group=args.feature_group,
        endpoint=args.endpoint,
        qc_acceptability=args.qc_acceptability,
        min_epi_fraction=args.min_epi_fraction,
        agg=args.agg,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        chunk_idx=args.chunk_idx,
        n_chunks=args.n_chunks,
        outdir=args.outdir,
        harmonized_path=args.harmonized_path,
        clinical_vars=args.clinical_vars,
        patient_subset=args.patient_subset,
        sample_type=args.sample_type,
        transform_mode=args.transform_mode,
        feature_source=args.feature_source,
        spatial_root=args.spatial_root,
        cell_features_path=args.cell_features_path,
        triads_path=args.triads_path,
        koll_metadata_csv=args.koll_metadata_csv,
        min_feature_nonmissing_frac=args.min_feature_nonmissing_frac,
        min_feature_unique=args.min_feature_unique,
        min_feature_nonzero=args.min_feature_nonzero,
    )


if __name__ == "__main__":
    main()
