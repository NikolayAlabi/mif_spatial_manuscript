#!/usr/bin/env python3
"""
stage3a_turbt_module_univariate_v1.py

Outcome evaluation of FROZEN Stage2C root modules and Stage2D cross-root meta-modules.

Architecture
------------
1) setup
   - validates the frozen Stage2C / Stage2D definitions
   - reads discovery score tables once
   - writes small cohort x panel score caches
   - creates a two-row NAC2015 scoring index

2) nac2015-worker
   - one CPU per panel
   - reconstructs NAC2015 TURBT patient-level raw feature matrices using the same
     Stage1-v6 loading/QC/median-aggregation logic as Stage2B1
   - applies the frozen Stage2C root-module definitions
   - applies the frozen Stage2D meta-module definitions
   - DOES NOT use NAC2015 outcomes while constructing scores

3) finalize
   - checks all cohort x panel score caches
   - creates the cohort x panel x endpoint worker index
   - marks low-N / low-event contexts as exploratory rather than silently dropping them

4) worker
   - one CPU per cohort x panel x endpoint
   - full-fit univariate logistic/Cox association
   - repeated OOF CV discrimination
   - fold stability / direction-consistency diagnostics
   - root modules and meta-modules are both evaluated

Primary Stage3A outputs per context
-----------------------------------
program_univariate_metrics.csv
program_fullfit_forest.csv
program_fold_metrics.csv
program_oof_predictions.csv.gz
program_scores_used.parquet
context_summary.csv
plots/*.png

NAC2015 interpretation
----------------------
Module / meta-module DEFINITIONS are frozen before NAC2015 is scored.
The univariate coefficient is nevertheless refit within NAC2015; therefore this is
an independent-cohort evaluation of a frozen score, not a locked external model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESPONSE_ENDPOINTS = {"complete_response", "any_response"}
SURVIVAL_ENDPOINTS = {"OS", "RFS"}
ROOT_COL = "feature_source"
RANDOM_STATE = 42
COX_PENALIZER = 0.01


# =============================================================================
# Generic helpers
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, p: Path) -> None:
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_config(path: str | Path) -> dict:
    p = Path(path)
    cfg = read_json(p)
    cfg["_config_path"] = str(p)
    return cfg


def safe_slug(x: object) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")
    return s or "NA"


def clean_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA, "": pd.NA})
    )


def safe_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def first_existing(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def import_module_from_path(name: str, path: str | Path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_array_id(arg: Optional[int]) -> int:
    if arg is not None:
        return int(arg)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise RuntimeError("Need --array-id or SLURM_ARRAY_TASK_ID")
    return int(env)


def load_index_row(path: Path, array_id: int) -> pd.Series:
    d = pd.read_csv(path)
    z = d[pd.to_numeric(d["array_id"], errors="coerce").eq(int(array_id))]
    if len(z) != 1:
        raise RuntimeError(f"Expected one row for array_id={array_id} in {path}; found {len(z)}")
    return z.iloc[0]


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["stage3a_output_root"])


def score_cache_dir(cfg: Mapping, cohort: str, panel: str) -> Path:
    return output_root(cfg) / "score_cache" / f"{safe_slug(cohort)}__{safe_slug(panel)}"


def context_dir(
    cfg: Mapping,
    cohort: str,
    panel: str,
    endpoint: str,
    patient_subset: str = "all",
) -> Path:
    return (
        output_root(cfg)
        / "contexts"
        / safe_slug(cohort)
        / safe_slug(panel)
        / safe_slug(endpoint)
        / safe_slug(patient_subset)
    )


# =============================================================================
# Frozen Stage2 definitions
# =============================================================================

def validate_frozen_meta_thresholds(cfg: Mapping) -> None:
    """Fail if Stage2D primary output is not the requested frozen threshold."""
    s2d = Path(cfg["stage2d_output_root"])
    p = s2d / "stage2d_primary_meta_summary.csv"
    expected = cfg.get("expected_meta_rho_thresholds", {"AR": 0.35, "BT": 0.35})
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    if "panel" not in d.columns:
        raise KeyError(f"panel missing from {p}")
    threshold_col = first_existing(
        d,
        ["primary_rho_threshold", "rho_threshold", "threshold"],
    )
    if threshold_col is None:
        log(f"[WARN] Could not verify frozen rho threshold from {p}; no threshold column.")
        return
    problems = []
    for panel, target in expected.items():
        z = d[d["panel"].astype(str).eq(str(panel))]
        if z.empty:
            problems.append(f"{panel}: missing")
            continue
        observed = float(pd.to_numeric(z.iloc[0][threshold_col], errors="coerce"))
        if not np.isfinite(observed) or abs(observed - float(target)) > 1e-8:
            problems.append(f"{panel}: observed={observed}, expected={target}")
    if problems:
        raise RuntimeError(
            "Stage2D final memberships are not frozen at the requested thresholds. "
            "Edit Stage2D primary_rho_thresholds to AR=0.35, BT=0.35 and rerun Stage2D. "
            "Problems: " + "; ".join(problems)
        )


def load_root_membership(cfg: Mapping) -> pd.DataFrame:
    p = Path(cfg["stage2c_output_root"]) / "final_root_module_membership.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    req = ["panel", ROOT_COL, "root_module_id", "feature_uid", "feature_group", "feature"]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise KeyError(f"Root membership missing {miss}: {p}")
    return d


def load_meta_membership(cfg: Mapping) -> pd.DataFrame:
    p = Path(cfg["stage2d_output_root"]) / "final_meta_module_membership.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    req = ["panel", ROOT_COL, "root_module_id", "meta_module_id"]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise KeyError(f"Meta membership missing {miss}: {p}")
    return d


def build_program_registry(cfg: Mapping) -> pd.DataFrame:
    rm = load_root_membership(cfg)
    mm = load_meta_membership(cfg)

    roots = (
        rm.groupby(["panel", ROOT_COL, "root_module_id"], as_index=False)
        .agg(n_features=("feature_uid", "nunique"))
    )
    roots["program_id"] = roots["root_module_id"].astype(str)
    roots["program_level"] = "root_module"
    roots["program_type"] = "root_module"
    roots["n_member_root_modules"] = 1
    roots["member_root_modules"] = roots["root_module_id"].astype(str)

    true_meta = mm[mm["meta_module_id"].notna()].copy()
    metas = (
        true_meta.groupby(["panel", "meta_module_id"], as_index=False)
        .agg(
            n_member_root_modules=("root_module_id", "nunique"),
            n_prep_roots=(ROOT_COL, "nunique"),
            prep_roots=(ROOT_COL, lambda x: ";".join(sorted(set(map(str, x))))),
            member_root_modules=("root_module_id", lambda x: ";".join(map(str, x))),
        )
    )
    metas["program_id"] = metas["meta_module_id"].astype(str)
    metas["program_level"] = "meta_module"
    metas["program_type"] = "meta_module"
    metas[ROOT_COL] = "cross_root"
    metas["n_features"] = np.nan

    cols = [
        "panel", "program_id", "program_level", "program_type", ROOT_COL,
        "n_features", "n_member_root_modules", "member_root_modules",
    ]
    for c in cols:
        if c not in roots.columns:
            roots[c] = np.nan
        if c not in metas.columns:
            metas[c] = np.nan

    out = pd.concat([roots[cols], metas[cols]], ignore_index=True, sort=False)
    return out.sort_values(["panel", "program_level", ROOT_COL, "program_id"]).reset_index(drop=True)


# =============================================================================
# Score cache setup for the four discovery cohorts
# =============================================================================

def discovery_score_tables(cfg: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s2c = Path(cfg["stage2c_output_root"])
    s2d = Path(cfg["stage2d_output_root"])
    rp = s2c / "all_discovery_root_module_scores_wide.parquet"
    mp = s2d / "all_discovery_meta_module_scores_wide.parquet"
    if not rp.exists():
        raise FileNotFoundError(rp)
    if not mp.exists():
        raise FileNotFoundError(mp)
    return pd.read_parquet(rp), pd.read_parquet(mp)


def normalize_score_wide(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    pid = first_existing(x, ["patient_id", "patient_ID", "Patient_ID"])
    if pid is None:
        raise KeyError("No patient ID column in score table")
    if pid != "patient_id":
        x = x.rename(columns={pid: "patient_id"})
    x["patient_id"] = clean_id_series(x["patient_id"])
    x = x[x["patient_id"].notna()].copy()
    if "cohort" not in x.columns or "panel" not in x.columns:
        raise KeyError("Discovery score tables must contain cohort and panel")
    return x


def make_cache_from_existing(
    root_scores: pd.DataFrame,
    meta_scores: pd.DataFrame,
    registry: pd.DataFrame,
    cohort: str,
    panel: str,
    outdir: Path,
) -> None:
    r = normalize_score_wide(root_scores)
    m = normalize_score_wide(meta_scores)
    r = r[r["cohort"].astype(str).eq(cohort) & r["panel"].astype(str).eq(panel)].copy()
    m = m[m["cohort"].astype(str).eq(cohort) & m["panel"].astype(str).eq(panel)].copy()
    if r.empty:
        raise RuntimeError(f"No root scores for {cohort}/{panel}")
    reg = registry[registry["panel"].astype(str).eq(panel)].copy()
    root_ids = reg.loc[reg["program_level"].eq("root_module"), "program_id"].astype(str).tolist()
    meta_ids = reg.loc[reg["program_level"].eq("meta_module"), "program_id"].astype(str).tolist()
    rkeep = [c for c in root_ids if c in r.columns]
    mkeep = [c for c in meta_ids if c in m.columns]

    rr = r[["patient_id"] + rkeep].drop_duplicates("patient_id")
    mm = m[["patient_id"] + mkeep].drop_duplicates("patient_id")
    cache = rr.merge(mm, on="patient_id", how="outer")
    cache.insert(1, "cohort", cohort)
    cache.insert(2, "panel", panel)

    ensure_dir(outdir)
    cache.to_parquet(outdir / "program_scores.parquet", index=False)

    audit = pd.DataFrame({
        "program_id": root_ids + meta_ids,
        "program_level": ["root_module"] * len(root_ids) + ["meta_module"] * len(meta_ids),
    })
    audit["present_in_score_cache"] = audit["program_id"].isin(cache.columns)
    audit.to_csv(outdir / "program_score_availability.csv", index=False)

    pd.DataFrame([{
        "cohort": cohort,
        "panel": panel,
        "n_patients": int(cache["patient_id"].nunique()),
        "n_root_programs_expected": len(root_ids),
        "n_root_programs_present": len(rkeep),
        "n_meta_programs_expected": len(meta_ids),
        "n_meta_programs_present": len(mkeep),
    }]).to_csv(outdir / "score_cache_summary.csv", index=False)
    (outdir / ".done").write_text("complete\n")


# =============================================================================
# NAC2015 scoring
# =============================================================================

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
) -> pd.DataFrame:
    """Same tested Stage1-v6 loading/aggregation pattern used by Stage2B1."""
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
    core_df = stage1_mod.prepare_core_level_feature_table(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=qc_acceptability,
        min_epi_fraction=min_epi_fraction,
        sample_type=sample_type,
    )
    if core_df.empty:
        raise ValueError("No cores remain after requested filters")
    core_df = stage1_mod.merge_harmonized_to_core_df(core_df, harm_df)
    core_df = stage1_mod.replace_with_harmonized_columns(core_df)
    core_df = stage1_mod.simplify_clinical_vars(core_df)
    core_df = stage1_mod.ensure_patient_id_column(core_df)

    present = [f for f in features if f in core_df.columns]
    if not present:
        raise ValueError("None of the requested features were found in core_df")

    patient_df = stage1_mod.aggregate_core_to_patient(core_df, feature_cols=present, agg=agg)
    if "cohort" in patient_df.columns:
        patient_df = patient_df[patient_df["cohort"].astype(str).eq(str(cohort))].copy()
    if patient_df.empty:
        raise ValueError("No patients remain after aggregation")

    keep = [c for c in patient_df.columns if c not in present]
    return patient_df[keep + present].copy()


def zscore_feature(s: pd.Series, ddof: int = 0) -> Tuple[pd.Series, dict]:
    x = safe_numeric(s)
    n = int(x.notna().sum())
    mu = float(x.mean()) if n else np.nan
    sd = float(x.std(ddof=ddof)) if n else np.nan
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        z = pd.Series(np.nan, index=x.index, dtype=float)
    else:
        z = (x - mu) / sd
    return z, {"n_nonmissing": n, "mean": mu, "sd": sd, "n_unique": int(x.dropna().nunique())}


def score_root_modules_for_nac2015(cfg: Mapping, panel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    membership = load_root_membership(cfg)
    membership = membership[membership["panel"].astype(str).eq(panel)].copy()
    if membership.empty:
        raise RuntimeError(f"No root membership for {panel}")

    stage1_mod = import_module_from_path(
        f"stage1_v6_stage3a_nac2015_{panel}", cfg["stage1_script_path"]
    )

    # Reconstruct each prep-root union matrix, exactly as Stage2B1 did.
    root_matrices: Dict[str, pd.DataFrame] = {}
    matrix_audit: List[dict] = []

    for root, mem_root in membership.groupby(ROOT_COL, sort=True):
        merged: Optional[pd.DataFrame] = None
        for fg, g in mem_root.groupby("feature_group", sort=True):
            features = g["feature"].astype(str).drop_duplicates().tolist()
            try:
                pdf = build_patient_matrix_for_source_group(
                    stage1_mod=stage1_mod,
                    cohort="NAC2015",
                    panel=panel,
                    feature_source=str(root),
                    feature_group=str(fg),
                    features=features,
                    sample_type=str(cfg.get("sample_type", "TURBT")),
                    patient_subset="all",
                    agg=str(cfg.get("agg", "median")),
                    qc_acceptability=str(cfg.get("qc_acceptability", "acceptable_or_borderline")),
                    min_epi_fraction=float(cfg.get("min_epi_fraction", 0.05)),
                    harmonized_path=cfg["harmonized_path"],
                    spatial_root=cfg.get("spatial_root"),
                    cell_features_path=cfg.get("cell_features_path"),
                    triads_path=cfg.get("triads_path"),
                )
            except Exception as exc:
                matrix_audit.append({
                    "panel": panel, ROOT_COL: root, "feature_group": fg,
                    "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                    "n_requested_features": len(features),
                })
                log(f"[WARN] NAC2015 {panel}/{root}/{fg} failed: {type(exc).__name__}: {exc}")
                continue

            tmp = pdf[["patient_id"]].copy()
            tmp["patient_id"] = clean_id_series(tmp["patient_id"])
            n_ok = 0
            for _, rr in g.drop_duplicates("feature_uid").iterrows():
                feat = str(rr["feature"])
                uid = str(rr["feature_uid"])
                if feat in pdf.columns:
                    tmp[uid] = safe_numeric(pdf[feat]).to_numpy()
                    n_ok += 1
            if merged is None:
                merged = tmp
            else:
                add = [c for c in tmp.columns if c == "patient_id" or c not in merged.columns]
                merged = merged.merge(tmp[add], on="patient_id", how="outer")
            matrix_audit.append({
                "panel": panel, ROOT_COL: root, "feature_group": fg,
                "status": "ok", "reason": "",
                "n_requested_features": len(features), "n_measured_features": n_ok,
                "n_patients": int(pdf["patient_id"].nunique()),
            })

        if merged is None:
            merged = pd.DataFrame(columns=["patient_id"])
        for uid in mem_root["feature_uid"].astype(str).drop_duplicates():
            if uid not in merged.columns:
                merged[uid] = np.nan
        root_matrices[str(root)] = merged

    # Stage2C scoring rules.
    min_feature_nonmissing = float(cfg.get("min_feature_nonmissing_fraction_for_scoring", 0.20))
    min_patient_frac = float(cfg.get("min_patient_module_feature_fraction", 0.50))
    min_features_present_multi = int(cfg.get("min_features_present_multifeature_module", 1))
    ddof = int(cfg.get("zscore_ddof", 0))

    score_parts = []
    module_qc = []

    for root, mem_root in membership.groupby(ROOT_COL, sort=True):
        X = root_matrices[str(root)].copy()
        if X.empty:
            continue
        X["patient_id"] = clean_id_series(X["patient_id"])
        zmat = pd.DataFrame(index=X.index)
        eligible: Dict[str, bool] = {}

        for uid in mem_root["feature_uid"].astype(str).drop_duplicates():
            if uid not in X.columns:
                zmat[uid] = np.nan
                eligible[uid] = False
                continue
            z, qc = zscore_feature(X[uid], ddof=ddof)
            zmat[uid] = z
            frac = float(qc["n_nonmissing"] / len(X)) if len(X) else 0.0
            eligible[uid] = bool(
                frac >= min_feature_nonmissing
                and qc["n_unique"] >= 2
                and np.isfinite(qc["sd"])
                and qc["sd"] > 0
            )

        for module_id, gm in mem_root.groupby("root_module_id", sort=True):
            uids = gm["feature_uid"].astype(str).drop_duplicates().tolist()
            eligible_uids = [u for u in uids if eligible.get(u, False)]
            n_total = len(uids)
            if eligible_uids:
                block = zmat[eligible_uids]
                n_present = block.notna().sum(axis=1)
                coverage = n_present / max(n_total, 1)
                min_present = max(1, int(math.ceil(min_patient_frac * n_total)))
                if n_total > 1:
                    min_present = max(min_present, min_features_present_multi)
                score = block.mean(axis=1, skipna=True)
                score[(coverage < min_patient_frac) | (n_present < min_present)] = np.nan
            else:
                n_present = pd.Series(0, index=X.index, dtype=int)
                coverage = pd.Series(0.0, index=X.index)
                score = pd.Series(np.nan, index=X.index)

            part = pd.DataFrame({
                "patient_id": X["patient_id"].values,
                "cohort": "NAC2015",
                "panel": panel,
                ROOT_COL: str(root),
                "root_module_id": str(module_id),
                "score_meanz": score.values,
            })
            score_parts.append(part)
            module_qc.append({
                "cohort": "NAC2015", "panel": panel, ROOT_COL: root,
                "root_module_id": str(module_id),
                "n_features_total": n_total,
                "n_features_available_cohort": len(eligible_uids),
                "feature_availability_fraction": len(eligible_uids) / n_total if n_total else np.nan,
                "n_patients": len(X),
                "n_patients_scored": int(score.notna().sum()),
                "patient_score_fraction": float(score.notna().mean()) if len(score) else np.nan,
            })

    if not score_parts:
        raise RuntimeError(f"No NAC2015 root module scores generated for {panel}")
    scores = pd.concat(score_parts, ignore_index=True, sort=False)
    return scores, pd.DataFrame(module_qc), pd.DataFrame(matrix_audit)


def zscore_series(s: pd.Series) -> pd.Series:
    x = safe_numeric(s)
    sd = x.std(ddof=0)
    if x.notna().sum() < 2 or not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def score_meta_modules_from_root_scores(
    root_scores: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: Mapping,
    panel: str,
) -> pd.DataFrame:
    pmem = membership[
        membership["panel"].astype(str).eq(panel)
        & membership["meta_module_id"].notna()
    ].copy()
    if pmem.empty:
        return pd.DataFrame(columns=["patient_id", "cohort", "panel", "meta_module_id", "score_meta_meanz"])

    X = root_scores.pivot_table(
        index="patient_id", columns="root_module_id", values="score_meanz", aggfunc="first"
    )
    Z = X.apply(zscore_series, axis=0)
    min_frac = float(cfg.get("min_meta_member_fraction", 0.50))
    min_abs = int(cfg.get("min_meta_members_present", 2))
    rows = []

    for meta_id, gm in pmem.groupby("meta_module_id", sort=True):
        members = gm["root_module_id"].astype(str).tolist()
        present = [m for m in members if m in Z.columns]
        if present:
            block = Z[present]
            n_present = block.notna().sum(axis=1)
            min_req = max(min_abs, int(math.ceil(min_frac * len(members))))
            score = block.mean(axis=1, skipna=True)
            score[n_present < min_req] = np.nan
        else:
            score = pd.Series(np.nan, index=Z.index)
            n_present = pd.Series(0, index=Z.index)
        for pid in Z.index:
            rows.append({
                "patient_id": str(pid), "cohort": "NAC2015", "panel": panel,
                "meta_module_id": str(meta_id),
                "score_meta_meanz": score.loc[pid],
                "n_member_root_modules_total": len(members),
                "n_member_root_modules_present": int(n_present.loc[pid]),
            })
    return pd.DataFrame(rows)



# =============================================================================
# Fixed TURBT clinical-variable screen
# =============================================================================

AGE_CANDIDATES = ["Age", "age"]
SEX_CANDIDATES = ["Sex", "gender"]
CT_CANDIDATES = [
    "cT_STAGE_ordinal", "TURBT_Tstage", "clinical_T_stage",
    "cstage", "cT_STAGE", "TURBT_Tstage_ordinal",
]
CN_CANDIDATES = [
    "cN_STAGE_ordinal", "TURBT_Nstage", "clinical_N_stage",
    "cN_STAGE", "TURBT_Nstage_ordinal", "cN",
]


def stage_to_ordinal(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    s = (
        s.replace("CT", "")
         .replace("CN", "")
         .replace("PT", "")
         .replace("PN", "")
         .replace("T", "")
         .replace("N", "")
         .replace("Y", "")
         .replace("P", "")
    )
    if s in {"IS", "A", "TA", "TIS"}:
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def load_turbt_clinical_variables(cfg: Mapping, cohort: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the fixed TURBT clinical baseline variables:
      Age, Sex, cT, cN

    The first usable column from each prespecified candidate list is used,
    mirroring the older combined clinical/module evaluator.
    """
    root = Path(cfg["clinical_matrix_root"])
    # Clinical prefix differs for No-NAC in the existing clean-matrix convention.
    clinical_cohort = "NoNAC" if cohort == "No-NAC" else cohort
    p = root / f"{clinical_cohort}_TURBT_clinical_matrix.csv"
    if not p.exists():
        return pd.DataFrame(columns=["patient_id"]), pd.DataFrame([{
            "cohort": cohort, "clinical_variable": "ALL",
            "status": "missing_clinical_matrix", "source_column": "", "path": str(p),
        }])

    d = pd.read_csv(p)
    pid = first_existing(d, ["patient_ID", "patient_id", "Patient_ID"])
    if pid is None:
        raise KeyError(f"No patient ID column in {p}")
    d["patient_id"] = clean_id_series(d[pid])
    d = d[d["patient_id"].notna()].drop_duplicates("patient_id").copy()

    groups = {
        "Age": AGE_CANDIDATES,
        "Sex": SEX_CANDIDATES,
        "cT": CT_CANDIDATES,
        "cN": CN_CANDIDATES,
    }

    out = pd.DataFrame({"patient_id": d["patient_id"].values})
    audit = []

    for label, candidates in groups.items():
        source = None
        for c in candidates:
            if c in d.columns and d[c].notna().sum() >= 3 and d[c].dropna().nunique() > 1:
                source = c
                break

        if source is None:
            audit.append({
                "cohort": cohort, "clinical_variable": label, "status": "not_available",
                "source_column": "", "path": str(p),
            })
            continue

        s = d[source]
        if label == "Age":
            x = safe_numeric(s)

        elif label in {"cT", "cN"}:
            x = safe_numeric(s)
            if x.notna().mean() < 0.8:
                x = s.map(stage_to_ordinal)

        elif label == "Sex":
            xnum = safe_numeric(s)
            if xnum.notna().mean() >= 0.8 and xnum.dropna().nunique() == 2:
                vals = sorted(xnum.dropna().unique())
                x = xnum.map({vals[0]: 0.0, vals[1]: 1.0})
            else:
                ss = s.astype("string").str.strip()
                levels = sorted(ss.dropna().unique().tolist())
                if len(levels) == 2:
                    x = ss.map({levels[0]: 0.0, levels[1]: 1.0}).astype(float)
                else:
                    x = pd.Series(np.nan, index=s.index, dtype=float)

        else:
            x = safe_numeric(s)

        out[label] = x.to_numpy()
        audit.append({
            "cohort": cohort, "clinical_variable": label,
            "status": "ok" if out[label].notna().sum() >= 3 and out[label].dropna().nunique() > 1 else "unusable_after_encoding",
            "source_column": source, "path": str(p),
            "n_nonmissing": int(out[label].notna().sum()),
            "n_unique": int(out[label].dropna().nunique()),
        })

    return out, pd.DataFrame(audit)


# =============================================================================
# Endpoint preparation
# =============================================================================

def load_harmonized(cfg: Mapping) -> pd.DataFrame:
    d = pd.read_csv(cfg["harmonized_path"], low_memory=False)
    if "patient_id" not in d.columns or "cohort" not in d.columns:
        raise KeyError("harmonized dataframe needs patient_id and cohort")
    d["patient_id"] = clean_id_series(d["patient_id"])
    d["cohort"] = d["cohort"].astype(str).str.strip()
    d = d[d["patient_id"].notna()].drop_duplicates(["cohort", "patient_id"]).copy()
    return d


def normalize_yes_no_series(s: pd.Series) -> pd.Series:
    """Mirror Stage1 apply_patient_subset normalization exactly."""
    x = s.astype("string").str.strip().str.lower()
    yes_vals = {"1", "1.0", "yes", "y", "true", "t"}
    no_vals = {"0", "0.0", "no", "n", "false", "f"}
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[x.isin(yes_vals)] = "yes"
    out[x.isin(no_vals)] = "no"
    return out


def apply_patient_subset_harmonized(
    d: pd.DataFrame,
    cohort: str,
    patient_subset: str,
) -> pd.DataFrame:
    """
    Apply the same No-NAC adjuvant-chemotherapy subset definition used upstream.

    no_adj_chemo = adjuvant_chemo normalized to "no"
    adj_chemo    = adjuvant_chemo normalized to "yes"
    """
    if patient_subset == "all":
        return d.copy()

    if cohort != "No-NAC":
        raise ValueError(
            f"patient_subset={patient_subset} requested for {cohort}; "
            "Stage3A subset evaluation is currently defined only for No-NAC."
        )

    if patient_subset not in {"no_adj_chemo", "adj_chemo"}:
        raise ValueError(f"Unknown patient_subset: {patient_subset}")

    if "adjuvant_chemo" not in d.columns:
        raise ValueError(
            "Requested No-NAC adjuvant-chemotherapy subset, but harmonized "
            "dataframe lacks 'adjuvant_chemo'."
        )

    adj = normalize_yes_no_series(d["adjuvant_chemo"])
    if patient_subset == "no_adj_chemo":
        return d[adj.eq("no")].copy()
    return d[adj.eq("yes")].copy()


def endpoint_table(
    harm: pd.DataFrame,
    cohort: str,
    endpoint: str,
    patient_subset: str = "all",
) -> pd.DataFrame:
    d = harm[harm["cohort"].astype(str).eq(cohort)].copy()
    d = apply_patient_subset_harmonized(d, cohort=cohort, patient_subset=patient_subset)
    if endpoint in RESPONSE_ENDPOINTS:
        if endpoint not in d.columns:
            return pd.DataFrame(columns=["patient_id", "y"])
        out = d[["patient_id", endpoint]].rename(columns={endpoint: "y"})
        out["y"] = safe_numeric(out["y"])
        out = out.dropna(subset=["y"]).copy()
        out["y"] = out["y"].astype(int)
        return out

    if endpoint == "OS":
        tcol = first_existing(d, ["OS_months_TUR", "OS_months_TURBT", "OS_TURBT_months"])
        ecol = first_existing(d, ["OS_event", "OS"])
    elif endpoint == "RFS":
        tcol = first_existing(d, ["REC_months_TURBT", "RFS_months_TURBT", "RFS_TURBT"])
        ecol = first_existing(d, ["REC", "RFS_event"])
    else:
        raise ValueError(endpoint)

    if tcol is None or ecol is None:
        return pd.DataFrame(columns=["patient_id", "time", "event"])
    out = d[["patient_id", tcol, ecol]].rename(columns={tcol: "time", ecol: "event"})
    out["time"] = safe_numeric(out["time"])
    out["event"] = safe_numeric(out["event"])
    out = out.dropna(subset=["time", "event"]).copy()
    out = out[out["time"] >= 0].copy()
    out["event"] = out["event"].astype(int)
    return out


def context_quality(endpoint: str, ep: pd.DataFrame, cfg: Mapping) -> dict:
    if endpoint in RESPONSE_ENDPOINTS:
        n = len(ep)
        counts = ep["y"].value_counts()
        n_pos = int((ep["y"] == 1).sum())
        n_neg = int((ep["y"] == 0).sum())
        min_class = int(counts.min()) if len(counts) >= 2 else 0
        primary = (
            n >= int(cfg.get("primary_min_n", 20))
            and min_class >= int(cfg.get("primary_min_response_class", 5))
        )
        can_fit = n >= int(cfg.get("minimum_fit_n", 10)) and min_class >= 2
        return {
            "n_endpoint": n, "n_events": n_pos, "n_positive": n_pos, "n_negative": n_neg,
            "n_signal": min_class, "signal_name": "min_class_count",
            "primary_eligible": bool(primary), "can_fit": bool(can_fit),
        }
    n = len(ep)
    events = int(ep["event"].sum()) if n else 0
    primary = (
        n >= int(cfg.get("primary_min_n", 20))
        and events >= int(cfg.get("primary_min_survival_events", 5))
    )
    can_fit = n >= int(cfg.get("minimum_fit_n", 10)) and events >= 2
    return {
        "n_endpoint": n, "n_events": events, "n_positive": np.nan, "n_negative": np.nan,
        "n_signal": events, "signal_name": "n_events",
        "primary_eligible": bool(primary), "can_fit": bool(can_fit),
    }


# =============================================================================
# Univariate modeling
# =============================================================================

def standardize_train_valid(train: pd.Series, valid: pd.Series) -> Tuple[pd.Series, pd.Series]:
    tr = safe_numeric(train)
    va = safe_numeric(valid)
    mu = tr.mean()
    sd = tr.std(ddof=0)
    if tr.notna().sum() < 2 or not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=tr.index), pd.Series(np.nan, index=va.index)
    return (tr - mu) / sd, (va - mu) / sd


def fullfit_response(score: pd.Series, y: pd.Series) -> dict:
    d = pd.DataFrame({"score": safe_numeric(score), "y": safe_numeric(y)}).dropna()
    if len(d) < 3 or d["y"].nunique() < 2 or d["score"].nunique() < 2:
        return {"fit_status": "insufficient_data"}
    z = (d["score"] - d["score"].mean()) / d["score"].std(ddof=0)
    try:
        X = sm.add_constant(pd.DataFrame({"score": z}), has_constant="add")
        fit = sm.Logit(d["y"].astype(int), X).fit(disp=False, maxiter=300)
        coef = float(fit.params["score"])
        lo, hi = fit.conf_int().loc["score"].tolist()
        p = float(fit.pvalues["score"])
        prob = fit.predict(X)
        auc = float(roc_auc_score(d["y"].astype(int), prob))
        return {
            "coef": coef, "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "effect": float(np.exp(coef)), "effect_ci_low": float(np.exp(lo)),
            "effect_ci_high": float(np.exp(hi)), "full_metric": auc,
            "fit_status": "statsmodels_logit", "n": len(d), "n_events": int(d["y"].sum()),
        }
    except Exception as exc:
        try:
            model = LogisticRegression(
                penalty="l2", C=1.0, solver="liblinear", class_weight="balanced",
                max_iter=1000, random_state=RANDOM_STATE,
            )
            model.fit(z.to_frame("score"), d["y"].astype(int))
            coef = float(model.coef_.ravel()[0])
            prob = model.predict_proba(z.to_frame("score"))[:, 1]
            return {
                "coef": coef, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                "effect": float(np.exp(coef)), "effect_ci_low": np.nan, "effect_ci_high": np.nan,
                "full_metric": float(roc_auc_score(d["y"].astype(int), prob)),
                "fit_status": f"sklearn_fallback_after_{type(exc).__name__}",
                "n": len(d), "n_events": int(d["y"].sum()),
            }
        except Exception as exc2:
            return {"fit_status": f"failed:{type(exc2).__name__}"}


def fullfit_survival(score: pd.Series, time: pd.Series, event: pd.Series) -> dict:
    d = pd.DataFrame({
        "score": safe_numeric(score), "time": safe_numeric(time), "event": safe_numeric(event)
    }).dropna()
    if len(d) < 3 or d["event"].sum() < 1 or d["score"].nunique() < 2:
        return {"fit_status": "insufficient_data"}
    d["score"] = (d["score"] - d["score"].mean()) / d["score"].std(ddof=0)
    try:
        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(d[["time", "event", "score"]], duration_col="time", event_col="event")
        s = cph.summary.loc["score"]
        risk = cph.predict_partial_hazard(d[["score"]]).values.ravel()
        cidx = float(concordance_index(d["time"], -risk, d["event"]))
        return {
            "coef": float(s["coef"]), "ci_low": float(s["coef lower 95%"]),
            "ci_high": float(s["coef upper 95%"]), "p_value": float(s["p"]),
            "effect": float(s["exp(coef)"]),
            "effect_ci_low": float(s["exp(coef) lower 95%"]),
            "effect_ci_high": float(s["exp(coef) upper 95%"]),
            "full_metric": cidx, "fit_status": "coxph",
            "n": len(d), "n_events": int(d["event"].sum()),
        }
    except Exception as exc:
        return {"fit_status": f"failed:{type(exc).__name__}"}


def repeated_oof_response(
    score: pd.Series, y: pd.Series, patient_ids: pd.Series,
    n_splits: int, n_repeats: int, seed: int,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "patient_id": clean_id_series(patient_ids),
        "score_raw": safe_numeric(score),
        "y": safe_numeric(y),
    }).dropna()
    if base["y"].nunique() < 2 or base["score_raw"].nunique() < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()
    min_class = int(base["y"].astype(int).value_counts().min())
    k = min(int(n_splits), min_class)
    if k < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()

    pred_rows, fold_rows = [], []
    for rep in range(int(n_repeats)):
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        for fold, (tr, va) in enumerate(splitter.split(base, base["y"].astype(int)), start=1):
            train, valid = base.iloc[tr].copy(), base.iloc[va].copy()
            ztr, zva = standardize_train_valid(train["score_raw"], valid["score_raw"])
            if ztr.notna().sum() < 2 or zva.notna().sum() < 1:
                continue
            try:
                model = LogisticRegression(
                    penalty="l2", C=1.0, solver="liblinear", class_weight="balanced",
                    max_iter=1000, random_state=seed + rep,
                )
                model.fit(ztr.to_frame("score"), train["y"].astype(int))
                prob = model.predict_proba(zva.to_frame("score"))[:, 1]
                coef = float(model.coef_.ravel()[0])
                fold_auc = (
                    float(roc_auc_score(valid["y"].astype(int), prob))
                    if valid["y"].nunique() == 2 else np.nan
                )
                fold_rows.append({
                    "repeat": rep + 1, "fold": fold, "fold_metric": fold_auc,
                    "fold_coef": coef, "n_train": len(train), "n_valid": len(valid),
                    "n_valid_events": int(valid["y"].sum()),
                })
                for pid, yy, pp in zip(valid["patient_id"], valid["y"].astype(int), prob):
                    pred_rows.append({
                        "patient_id": pid, "repeat": rep + 1, "fold": fold,
                        "y_true": int(yy), "prediction": float(pp),
                    })
            except Exception:
                continue

    pred = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)
    if pred.empty:
        return np.nan, pred, folds
    avg = pred.groupby("patient_id", as_index=False).agg(
        y_true=("y_true", "first"), prediction=("prediction", "mean"),
        n_oof_predictions=("prediction", "size"),
    )
    metric = float(roc_auc_score(avg["y_true"], avg["prediction"])) if avg["y_true"].nunique() == 2 else np.nan
    return metric, avg, folds


def repeated_oof_survival(
    score: pd.Series, time: pd.Series, event: pd.Series, patient_ids: pd.Series,
    n_splits: int, n_repeats: int, seed: int,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({
        "patient_id": clean_id_series(patient_ids),
        "score_raw": safe_numeric(score),
        "time": safe_numeric(time),
        "event": safe_numeric(event),
    }).dropna()
    if base["event"].sum() < 2 or base["score_raw"].nunique() < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()

    n_events = int(base["event"].sum())
    k = min(int(n_splits), n_events, len(base))
    if k < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()

    pred_rows, fold_rows = [], []
    stratify_ok = base["event"].value_counts().min() >= k

    for rep in range(int(n_repeats)):
        if stratify_ok:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
            split_iter = splitter.split(base, base["event"].astype(int))
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=seed + rep)
            split_iter = splitter.split(base)

        for fold, (tr, va) in enumerate(split_iter, start=1):
            train, valid = base.iloc[tr].copy(), base.iloc[va].copy()
            if train["event"].sum() < 1:
                continue
            ztr, zva = standardize_train_valid(train["score_raw"], valid["score_raw"])
            fitdf = pd.DataFrame({
                "time": train["time"].values, "event": train["event"].values, "score": ztr.values
            }).dropna()
            if fitdf["event"].sum() < 1 or fitdf["score"].nunique() < 2:
                continue
            try:
                cph = CoxPHFitter(penalizer=COX_PENALIZER)
                cph.fit(fitdf, duration_col="time", event_col="event")
                coef = float(cph.params_["score"])
                linpred = coef * zva.to_numpy()
                vv = valid.copy()
                vv["risk_score"] = linpred
                fold_c = np.nan
                if vv["event"].sum() > 0 and vv["risk_score"].notna().sum() >= 2:
                    try:
                        fold_c = float(concordance_index(vv["time"], -vv["risk_score"], vv["event"]))
                    except Exception:
                        pass
                fold_rows.append({
                    "repeat": rep + 1, "fold": fold, "fold_metric": fold_c,
                    "fold_coef": coef, "n_train": len(train), "n_valid": len(valid),
                    "n_valid_events": int(valid["event"].sum()),
                })
                for pid, tt, ee, rr in zip(vv["patient_id"], vv["time"], vv["event"].astype(int), vv["risk_score"]):
                    if np.isfinite(rr):
                        pred_rows.append({
                            "patient_id": pid, "repeat": rep + 1, "fold": fold,
                            "time": float(tt), "event": int(ee), "prediction": float(rr),
                        })
            except Exception:
                continue

    pred = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)
    if pred.empty:
        return np.nan, pred, folds
    avg = pred.groupby("patient_id", as_index=False).agg(
        time=("time", "first"), event=("event", "first"),
        prediction=("prediction", "mean"), n_oof_predictions=("prediction", "size"),
    )
    try:
        metric = float(concordance_index(avg["time"], -avg["prediction"], avg["event"]))
    except Exception:
        metric = np.nan
    return metric, avg, folds


# =============================================================================
# Plots
# =============================================================================

def save_fig(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_forest(metrics: pd.DataFrame, endpoint: str, outpath: Path) -> None:
    d = metrics.copy()
    d = d[d["effect"].notna()].sort_values(["program_level", "p_value"], na_position="last")
    if d.empty:
        return
    d["label"] = d["program_id"].astype(str)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.25 * len(d))))
    has = d["effect_ci_low"].notna() & d["effect_ci_high"].notna()
    if has.any():
        dd = d[has]
        yy = y[has.to_numpy()]
        ax.errorbar(
            dd["effect"], yy,
            xerr=[dd["effect"] - dd["effect_ci_low"], dd["effect_ci_high"] - dd["effect"]],
            fmt="o", capsize=2, linewidth=1,
        )
    if (~has).any():
        ax.scatter(d.loc[~has, "effect"], y[(~has).to_numpy()], s=25)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio per SD" if endpoint in RESPONSE_ENDPOINTS else "Hazard ratio per SD")
    ax.set_title("Frozen spatial-program univariate associations")
    fig.tight_layout()
    save_fig(fig, outpath)


def plot_oof_performance(metrics: pd.DataFrame, endpoint: str, outpath: Path) -> None:
    d = metrics.dropna(subset=["oof_metric"]).copy()
    if d.empty:
        return
    d = d.sort_values("oof_metric")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.22 * len(d))))
    y = np.arange(len(d))
    ax.scatter(d["oof_metric"], y, s=28)
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["program_id"], fontsize=7)
    ax.set_xlabel("Repeated-OOF AUC" if endpoint in RESPONSE_ENDPOINTS else "Repeated-OOF C-index")
    ax.set_title("Univariate discrimination of frozen spatial programs")
    fig.tight_layout()
    save_fig(fig, outpath)


# =============================================================================
# Commands
# =============================================================================

def command_validate(cfg: Mapping) -> None:
    required = [
        Path(cfg["stage2c_output_root"]) / "final_root_module_membership.csv",
        Path(cfg["stage2c_output_root"]) / "all_discovery_root_module_scores_wide.parquet",
        Path(cfg["stage2d_output_root"]) / "final_meta_module_membership.csv",
        Path(cfg["stage2d_output_root"]) / "all_discovery_meta_module_scores_wide.parquet",
        Path(cfg["stage1_script_path"]),
        Path(cfg["harmonized_path"]),
        Path(cfg["clinical_matrix_root"]),
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    validate_frozen_meta_thresholds(cfg)
    reg = build_program_registry(cfg)
    log(f"[VALID] programs={len(reg)} root={int((reg.program_level=='root_module').sum())} meta={int((reg.program_level=='meta_module').sum())}")


def command_setup(cfg: Mapping) -> None:
    command_validate(cfg)
    out = ensure_dir(output_root(cfg))
    write_json(cfg, out / "stage3a_config.resolved.json")

    registry = build_program_registry(cfg)
    registry.to_csv(out / "frozen_program_registry.csv", index=False)

    # Read each discovery aggregate score table ONCE.
    log("[SETUP] Reading discovery Stage2C / Stage2D score tables once...")
    root_scores, meta_scores = discovery_score_tables(cfg)

    discovery = [str(x) for x in cfg.get("discovery_cohorts", ["NAC2020", "PURE01", "BLASST", "No-NAC"])]
    panels = [str(x) for x in cfg.get("panels", ["AR", "BT"])]

    for cohort in discovery:
        for panel in panels:
            make_cache_from_existing(
                root_scores, meta_scores, registry, cohort, panel,
                score_cache_dir(cfg, cohort, panel),
            )

    # Only NAC2015 requires raw-feature reconstruction.
    nidx = pd.DataFrame({
        "array_id": np.arange(1, len(panels) + 1, dtype=int),
        "cohort": "NAC2015",
        "panel": panels,
    })
    nidx.to_csv(out / "stage3a_nac2015_score_worker_index.csv", index=False)
    log(f"[DONE SETUP] discovery caches={len(discovery)*len(panels)} NAC2015 workers={len(nidx)}")


def command_nac2015_worker(cfg: Mapping, array_id: int) -> None:
    out = output_root(cfg)
    row = load_index_row(out / "stage3a_nac2015_score_worker_index.csv", array_id)
    panel = str(row["panel"])
    wdir = score_cache_dir(cfg, "NAC2015", panel)
    ensure_dir(wdir)

    root_scores, module_qc, matrix_audit = score_root_modules_for_nac2015(cfg, panel)
    meta_membership = load_meta_membership(cfg)
    meta_scores = score_meta_modules_from_root_scores(root_scores, meta_membership, cfg, panel)

    rw = root_scores.pivot_table(
        index=["patient_id", "cohort", "panel"],
        columns="root_module_id", values="score_meanz", aggfunc="first"
    ).reset_index()
    rw.columns.name = None
    if not meta_scores.empty:
        mw = meta_scores.pivot_table(
            index=["patient_id", "cohort", "panel"],
            columns="meta_module_id", values="score_meta_meanz", aggfunc="first"
        ).reset_index()
        mw.columns.name = None
        cache = rw.merge(mw, on=["patient_id", "cohort", "panel"], how="outer")
    else:
        cache = rw

    cache.to_parquet(wdir / "program_scores.parquet", index=False)
    root_scores.to_parquet(wdir / "nac2015_root_module_scores_long.parquet", index=False)
    meta_scores.to_parquet(wdir / "nac2015_meta_module_scores_long.parquet", index=False)
    module_qc.to_csv(wdir / "nac2015_root_module_score_qc.csv", index=False)
    matrix_audit.to_csv(wdir / "nac2015_matrix_build_audit.csv", index=False)

    registry = pd.read_csv(out / "frozen_program_registry.csv")
    reg = registry[registry["panel"].astype(str).eq(panel)]
    audit = reg[["program_id", "program_level"]].copy()
    audit["present_in_score_cache"] = audit["program_id"].astype(str).isin(cache.columns)
    audit.to_csv(wdir / "program_score_availability.csv", index=False)

    pd.DataFrame([{
        "cohort": "NAC2015", "panel": panel,
        "n_patients": int(cache["patient_id"].nunique()),
        "n_programs_present": int(audit["present_in_score_cache"].sum()),
        "n_programs_expected": len(audit),
    }]).to_csv(wdir / "score_cache_summary.csv", index=False)
    (wdir / ".done").write_text("complete\n")
    log(f"[DONE NAC2015] {panel}: patients={cache.patient_id.nunique()} programs={audit.present_in_score_cache.sum()}/{len(audit)}")


def command_finalize(cfg: Mapping) -> None:
    out = output_root(cfg)
    cohorts = [str(x) for x in cfg.get("cohorts", ["NAC2020", "PURE01", "BLASST", "No-NAC", "NAC2015"])]
    panels = [str(x) for x in cfg.get("panels", ["AR", "BT"])]
    endpoints = [str(x) for x in cfg.get("endpoints", ["complete_response", "any_response", "OS", "RFS"])]

    missing = []
    for cohort in cohorts:
        for panel in panels:
            p = score_cache_dir(cfg, cohort, panel)
            if not (p / ".done").exists() or not (p / "program_scores.parquet").exists():
                missing.append({"cohort": cohort, "panel": panel, "score_cache_dir": str(p)})
    pd.DataFrame(missing).to_csv(out / "missing_score_caches.csv", index=False)
    if missing:
        raise RuntimeError(f"{len(missing)} cohort/panel score caches missing")

    harm = load_harmonized(cfg)
    rows, audit_rows = [], []
    aid = 0

    subset_map = cfg.get(
        "cohort_patient_subsets",
        {"_default": ["all"], "No-NAC": ["all", "no_adj_chemo"]},
    )
    subset_endpoint_map = cfg.get(
        "patient_subset_endpoints",
        {"no_adj_chemo": ["OS", "RFS"], "adj_chemo": ["OS", "RFS"]},
    )

    for cohort in cohorts:
        patient_subsets = subset_map.get(cohort, subset_map.get("_default", ["all"]))
        for panel in panels:
            scores = pd.read_parquet(score_cache_dir(cfg, cohort, panel) / "program_scores.parquet")
            score_ids = set(clean_id_series(scores["patient_id"]).dropna().astype(str))
            for endpoint in endpoints:
                for patient_subset in patient_subsets:
                    allowed = subset_endpoint_map.get(patient_subset)
                    if patient_subset != "all" and allowed is not None and endpoint not in allowed:
                        continue

                    ep = endpoint_table(
                        harm, cohort, endpoint, patient_subset=patient_subset
                    )
                    if ep.empty:
                        q = {
                            "n_endpoint": 0, "n_events": 0, "n_positive": np.nan, "n_negative": np.nan,
                            "n_signal": 0, "signal_name": "NA", "primary_eligible": False, "can_fit": False,
                        }
                        n_matched = 0
                    else:
                        ep = ep[clean_id_series(ep["patient_id"]).astype(str).isin(score_ids)].copy()
                        n_matched = len(ep)
                        q = context_quality(endpoint, ep, cfg)

                    row = {
                        "cohort": cohort, "panel": panel, "endpoint": endpoint,
                        "sample_type": cfg.get("sample_type", "TURBT"),
                        "patient_subset": patient_subset,
                        "n_matched_endpoint": n_matched,
                        **q,
                    }
                    audit_rows.append(row)
                    if q["can_fit"]:
                        aid += 1
                        rows.append({"array_id": aid, **row})

    idx = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    idx.to_csv(out / "stage3a_context_index.csv", index=False)
    audit.to_csv(out / "stage3a_context_availability_audit.csv", index=False)
    log(f"[DONE FINALIZE] evaluable contexts={len(idx)}; primary eligible={int(idx['primary_eligible'].sum()) if len(idx) else 0}")


def command_worker(cfg: Mapping, array_id: int) -> None:
    out = output_root(cfg)
    row = load_index_row(out / "stage3a_context_index.csv", array_id)
    cohort, panel, endpoint = str(row["cohort"]), str(row["panel"]), str(row["endpoint"])
    patient_subset = str(row.get("patient_subset", "all"))
    cdir = ensure_dir(context_dir(cfg, cohort, panel, endpoint, patient_subset))
    pdir = ensure_dir(cdir / "plots")

    scores = pd.read_parquet(score_cache_dir(cfg, cohort, panel) / "program_scores.parquet")
    scores["patient_id"] = clean_id_series(scores["patient_id"])
    registry = pd.read_csv(out / "frozen_program_registry.csv")
    registry = registry[registry["panel"].astype(str).eq(panel)].copy()

    harm = load_harmonized(cfg)
    ep = endpoint_table(harm, cohort, endpoint, patient_subset=patient_subset)
    clinical, clinical_audit = load_turbt_clinical_variables(cfg, cohort)
    data = scores.merge(ep, on="patient_id", how="inner")
    if not clinical.empty:
        data = data.merge(clinical, on="patient_id", how="left")
    if data.empty:
        raise RuntimeError(f"No matched score/outcome rows for {cohort}/{panel}/{endpoint}")

    clinical_audit.to_csv(cdir / "clinical_variable_availability.csv", index=False)

    # Clinical variables are evaluated alongside, but remain explicitly labeled.
    clinical_programs = [c for c in ["Age", "Sex", "cT", "cN"] if c in data.columns]
    program_ids = [p for p in registry["program_id"].astype(str) if p in data.columns]
    metric_rows, fold_parts, pred_parts = [], [], []

    n_splits = int(cfg.get("cv_n_splits", 5))
    n_repeats = int(cfg.get("cv_n_repeats", 5))
    seed = int(cfg.get("random_state", RANDOM_STATE))

    eval_programs = []
    for c in clinical_programs:
        eval_programs.append({
            "program_id": c,
            "program_level": "clinical_variable",
            "program_type": "clinical_variable",
            ROOT_COL: "clinical",
            "n_features": 1,
            "n_member_root_modules": np.nan,
            "member_root_modules": "",
        })
    for pid in program_ids:
        meta = registry[registry["program_id"].astype(str).eq(pid)].iloc[0].to_dict()
        eval_programs.append(meta)

    for ii, meta in enumerate(eval_programs):
        pid = str(meta["program_id"])
        score = safe_numeric(data[pid])

        if endpoint in RESPONSE_ENDPOINTS:
            full = fullfit_response(score, data["y"])
            oof_metric, pred, folds = repeated_oof_response(
                score, data["y"], data["patient_id"], n_splits, n_repeats, seed + ii * 100
            )
        else:
            full = fullfit_survival(score, data["time"], data["event"])
            oof_metric, pred, folds = repeated_oof_survival(
                score, data["time"], data["event"], data["patient_id"],
                n_splits, n_repeats, seed + ii * 100,
            )

        coef = full.get("coef", np.nan)
        if not folds.empty:
            fold_sd = float(pd.to_numeric(folds["fold_metric"], errors="coerce").std(ddof=1))
            mean_fold = float(pd.to_numeric(folds["fold_metric"], errors="coerce").mean())
            valid_folds = int(pd.to_numeric(folds["fold_metric"], errors="coerce").notna().sum())
            if np.isfinite(coef) and "fold_coef" in folds.columns:
                fc = pd.to_numeric(folds["fold_coef"], errors="coerce").dropna()
                direction_consistency = float((np.sign(fc) == np.sign(coef)).mean()) if len(fc) else np.nan
            else:
                direction_consistency = np.nan
        else:
            fold_sd = mean_fold = direction_consistency = np.nan
            valid_folds = 0

        metric_rows.append({
            "array_id": array_id, "cohort": cohort, "panel": panel, "endpoint": endpoint,
            "sample_type": cfg.get("sample_type", "TURBT"), "patient_subset": patient_subset,
            "program_id": pid, "program_level": meta.get("program_level"),
            "program_type": meta.get("program_type"), ROOT_COL: meta.get(ROOT_COL),
            "n_features": meta.get("n_features"), "n_member_root_modules": meta.get("n_member_root_modules"),
            "member_root_modules": meta.get("member_root_modules"),
            "coef": coef, "ci_low": full.get("ci_low", np.nan), "ci_high": full.get("ci_high", np.nan),
            "p_value": full.get("p_value", np.nan),
            "effect": full.get("effect", np.nan),
            "effect_ci_low": full.get("effect_ci_low", np.nan),
            "effect_ci_high": full.get("effect_ci_high", np.nan),
            "full_metric": full.get("full_metric", np.nan),
            "oof_metric": oof_metric,
            "mean_fold_metric": mean_fold, "fold_sd": fold_sd,
            "direction_consistency": direction_consistency,
            "valid_folds": valid_folds,
            "fit_status": full.get("fit_status", "unknown"),
            "n": int(full.get("n", score.notna().sum()) if pd.notna(full.get("n", np.nan)) else score.notna().sum()),
            "n_events": int(full.get("n_events", np.nan)) if pd.notna(full.get("n_events", np.nan)) else np.nan,
            "context_primary_eligible": bool(row["primary_eligible"]),
            "context_n_signal": row["n_signal"],
            "evaluation_set": "NAC2015_independent_score_evaluation" if cohort == "NAC2015" else "discovery_cohort_evaluation",
        })

        if not folds.empty:
            folds = folds.copy()
            folds["program_id"] = pid
            folds["program_level"] = meta.get("program_level")
            fold_parts.append(folds)
        if not pred.empty:
            pred = pred.copy()
            pred["program_id"] = pid
            pred["program_level"] = meta.get("program_level")
            pred_parts.append(pred)

    metrics = pd.DataFrame(metric_rows)
    folds_all = pd.concat(fold_parts, ignore_index=True, sort=False) if fold_parts else pd.DataFrame()
    preds_all = pd.concat(pred_parts, ignore_index=True, sort=False) if pred_parts else pd.DataFrame()

    metrics.to_csv(cdir / "program_univariate_metrics.csv", index=False)
    metrics[[
        c for c in [
            "program_id", "program_level", ROOT_COL, "coef", "ci_low", "ci_high",
            "p_value", "effect", "effect_ci_low", "effect_ci_high", "fit_status"
        ] if c in metrics.columns
    ]].to_csv(cdir / "program_fullfit_forest.csv", index=False)
    folds_all.to_csv(cdir / "program_fold_metrics.csv", index=False)
    preds_all.to_csv(cdir / "program_oof_predictions.csv.gz", index=False, compression="gzip")

    keep_cols = ["patient_id"] + clinical_programs + program_ids
    data[keep_cols].to_parquet(cdir / "program_scores_used.parquet", index=False)

    pd.DataFrame([{
        "array_id": array_id, "cohort": cohort, "panel": panel, "endpoint": endpoint,
        "patient_subset": patient_subset,
        "n_matched_patients": int(data["patient_id"].nunique()),
        "n_programs_evaluated": len(metrics),
        "n_clinical_variables": int((metrics["program_level"] == "clinical_variable").sum()),
        "n_root_modules": int((metrics["program_level"] == "root_module").sum()),
        "n_meta_modules": int((metrics["program_level"] == "meta_module").sum()),
        "context_primary_eligible": bool(row["primary_eligible"]),
        "context_n_signal": row["n_signal"],
    }]).to_csv(cdir / "context_summary.csv", index=False)

    # Forest plots: all, clinical, every prep root, and meta-modules.
    plot_forest(metrics, endpoint, pdir / "01_all_univariate_forest.png")
    plot_oof_performance(metrics, endpoint, pdir / "02_all_oof_performance.png")

    clin_plot = metrics[metrics["program_level"].eq("clinical_variable")].copy()
    plot_forest(clin_plot, endpoint, pdir / "03_clinical_univariate_forest.png")

    root_plot = metrics[metrics["program_level"].eq("root_module")].copy()
    for root_name, gg in root_plot.groupby(ROOT_COL, dropna=False):
        slug = safe_slug(root_name)
        plot_forest(
            gg, endpoint,
            pdir / f"04_root_{slug}_univariate_forest.png",
        )
        plot_oof_performance(
            gg, endpoint,
            pdir / f"05_root_{slug}_oof_performance.png",
        )

    meta_plot = metrics[metrics["program_level"].eq("meta_module")].copy()
    plot_forest(meta_plot, endpoint, pdir / "06_meta_module_univariate_forest.png")
    plot_oof_performance(meta_plot, endpoint, pdir / "07_meta_module_oof_performance.png")

    (cdir / ".done").write_text("complete\n")
    log(f"[DONE] {cohort}/{panel}/{endpoint}/{patient_subset}: programs={len(metrics)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "setup", "nac2015-worker", "finalize", "worker"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--array-id", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "setup":
        command_setup(cfg)
    elif args.command == "nac2015-worker":
        command_nac2015_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "finalize":
        command_finalize(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))


if __name__ == "__main__":
    main()
