#!/usr/bin/env python3
"""
delta_rc_common_v1.py

Shared utilities for the matched TURBT->RC shift, delta-outcome, and RC-only
evaluation of frozen Stage2C root modules and Stage2D meta-modules.

Key scoring decisions
---------------------
1. Root/meta memberships are frozen before this stage.
2. Matched TURBT->RC deltas use TURBT-reference scaling:
     - feature z-score parameters are fit on all eligible TURBT patients
       within cohort x panel x prep-root;
     - those same parameters are applied to RC;
     - delta = RC - TURBT on that common pretreatment reference scale.
   This avoids using RC values to define the coordinate system of the change.
3. RC-only prognostic analyses use an RC-internal cross-sectional scale,
   analogous to Stage2C but fit only on the RC cohort.
4. Meta-modules are rescored from root-module scores using the same
   mean-of-z-scored-root-modules rule used in Stage2D.
5. Survival analyses involving delta or RC-only scores use RC as the time
   origin. No automatic fallback to TURBT-origin survival time is allowed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT_COL = "feature_source"
RESPONSE_ENDPOINTS = {"any_response", "complete_response"}
SURVIVAL_ENDPOINTS = {"OS", "RFS"}
RANDOM_STATE = 42
COX_PENALIZER = 0.01


def log(x: str) -> None:
    print(x, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(p: str | Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def write_json(obj, p: str | Path) -> None:
    p = Path(p)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def clean_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def safe_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_slug(x: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(x)).strip("-")
    return s or "NA"


def first_existing(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def import_from_path(name: str, path: str | Path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_array_id(arg: Optional[int]) -> int:
    if arg is not None:
        return int(arg)
    x = os.environ.get("SLURM_ARRAY_TASK_ID")
    if x is None:
        raise RuntimeError("Need --array-id or SLURM_ARRAY_TASK_ID")
    return int(x)


def load_index_row(p: str | Path, array_id: int) -> pd.Series:
    d = pd.read_csv(p)
    z = d[pd.to_numeric(d["array_id"], errors="coerce").eq(int(array_id))]
    if len(z) != 1:
        raise RuntimeError(f"Expected one row for array_id={array_id} in {p}; found {len(z)}")
    return z.iloc[0]


def bh_adjust(p: pd.Series) -> pd.Series:
    x = safe_numeric(p)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    ok = x.notna()
    if not ok.any():
        return out
    vals = x[ok].clip(0, 1).to_numpy(float)
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(vals)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    idx = x[ok].index.to_numpy()
    out.loc[idx[order]] = q
    return out


# -----------------------------------------------------------------------------
# Frozen definitions
# -----------------------------------------------------------------------------

def root_membership(cfg: Mapping, panel: Optional[str] = None) -> pd.DataFrame:
    p = Path(cfg["stage2c_output_root"]) / "final_root_module_membership.csv"
    d = pd.read_csv(p)
    required = {"panel", ROOT_COL, "root_module_id", "feature_uid", "feature_group", "feature"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"{p} missing {sorted(missing)}")
    if panel is not None:
        d = d[d["panel"].astype(str).eq(str(panel))].copy()
    return d


def meta_membership(cfg: Mapping, panel: Optional[str] = None) -> pd.DataFrame:
    p = Path(cfg["stage2d_output_root"]) / "final_meta_module_membership.csv"
    d = pd.read_csv(p)
    required = {"panel", ROOT_COL, "root_module_id", "meta_module_id"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"{p} missing {sorted(missing)}")
    if panel is not None:
        d = d[d["panel"].astype(str).eq(str(panel))].copy()
    return d


def validate_frozen_definitions(cfg: Mapping) -> None:
    for p in [
        Path(cfg["stage2c_output_root"]) / "final_root_module_membership.csv",
        Path(cfg["stage2d_output_root"]) / "final_meta_module_membership.csv",
        Path(cfg["stage1_script_path"]),
        Path(cfg["harmonized_path"]),
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    expected = cfg.get("expected_meta_rho_thresholds", {"AR": 0.35, "BT": 0.35})
    p = Path(cfg["stage2d_output_root"]) / "stage2d_primary_meta_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    tcol = first_existing(d, ["primary_rho_threshold", "rho_threshold", "threshold"])
    if tcol is not None:
        problems = []
        for panel, target in expected.items():
            z = d[d["panel"].astype(str).eq(str(panel))]
            if z.empty:
                problems.append(f"{panel}:missing")
                continue
            observed = float(safe_numeric(pd.Series([z.iloc[0][tcol]])).iloc[0])
            if not np.isfinite(observed) or abs(observed - float(target)) > 1e-8:
                problems.append(f"{panel}:observed={observed},expected={target}")
        if problems:
            raise RuntimeError("Stage2D frozen-threshold mismatch: " + "; ".join(problems))


def program_registry(cfg: Mapping) -> pd.DataFrame:
    r = root_membership(cfg)
    m = meta_membership(cfg)
    roots = (
        r.groupby(["panel", ROOT_COL, "root_module_id"], as_index=False)
        .agg(n_features=("feature_uid", "nunique"))
        .rename(columns={"root_module_id": "program_id"})
    )
    roots["program_level"] = "root_module"
    roots["n_member_root_modules"] = 1
    roots["member_root_modules"] = roots["program_id"]

    mm = m[m["meta_module_id"].notna()].copy()
    metas = (
        mm.groupby(["panel", "meta_module_id"], as_index=False)
        .agg(
            n_member_root_modules=("root_module_id", "nunique"),
            member_root_modules=("root_module_id", lambda x: ";".join(sorted(set(map(str, x))))),
            member_prep_roots=(ROOT_COL, lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .rename(columns={"meta_module_id": "program_id"})
    )
    metas["program_level"] = "meta_module"
    metas[ROOT_COL] = "cross_root"
    metas["n_features"] = np.nan

    cols = ["panel", "program_id", "program_level", ROOT_COL, "n_features",
            "n_member_root_modules", "member_root_modules"]
    for c in cols:
        if c not in roots:
            roots[c] = np.nan
        if c not in metas:
            metas[c] = np.nan
    return pd.concat([roots[cols], metas[cols]], ignore_index=True)


# -----------------------------------------------------------------------------
# Stage1-based raw patient matrix reconstruction
# -----------------------------------------------------------------------------

def build_patient_matrix_for_group(
    stage1_mod,
    cfg: Mapping,
    cohort: str,
    panel: str,
    sample_type: str,
    feature_source: str,
    feature_group: str,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    features = feature_rows["feature"].astype(str).drop_duplicates().tolist()

    data_dict = stage1_mod.load_data_dict(
        feature_group=feature_group,
        feature_source=feature_source,
        panels=[panel],
        cohorts=[cohort],
        spatial_root=cfg.get("spatial_root"),
        cell_features_path=cfg.get("cell_features_path"),
        triads_path=cfg.get("triads_path"),
    )
    harm = stage1_mod.load_harmonized_df(cfg["harmonized_path"])
    core = stage1_mod.prepare_core_level_feature_table(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=cfg.get("qc_acceptability", "acceptable_or_borderline"),
        min_epi_fraction=float(cfg.get("min_epi_fraction", 0.05)),
        sample_type=sample_type,
    )
    if core.empty:
        raise ValueError("no_cores_after_qc")
    core = stage1_mod.merge_harmonized_to_core_df(core, harm)

    # Use whichever harmonization helpers exist in this Stage1 version.
    if hasattr(stage1_mod, "replace_with_harmonized_columns"):
        core = stage1_mod.replace_with_harmonized_columns(core)
    elif hasattr(stage1_mod, "harmonize_clinical_column_names"):
        core = stage1_mod.harmonize_clinical_column_names(core)
    if hasattr(stage1_mod, "simplify_clinical_vars"):
        core = stage1_mod.simplify_clinical_vars(core)
    core = stage1_mod.ensure_patient_id_column(core)

    present = [f for f in features if f in core.columns]
    if not present:
        raise ValueError("none_of_requested_features_present")

    pat = stage1_mod.aggregate_core_to_patient(
        core, feature_cols=present, agg=str(cfg.get("core_agg", "median"))
    )
    pat["patient_id"] = clean_id_series(pat["patient_id"])
    pat = pat[pat["patient_id"].notna()].drop_duplicates("patient_id").copy()

    out = pd.DataFrame({"patient_id": pat["patient_id"].values})
    for _, rr in feature_rows.drop_duplicates("feature_uid").iterrows():
        feat, uid = str(rr["feature"]), str(rr["feature_uid"])
        if feat in pat.columns:
            out[uid] = safe_numeric(pat[feat]).to_numpy()
    return out


def build_root_raw_matrix(
    stage1_mod,
    cfg: Mapping,
    cohort: str,
    panel: str,
    sample_type: str,
    mem_root: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = str(mem_root[ROOT_COL].iloc[0])
    merged = None
    audit = []

    for fg, g in mem_root.groupby("feature_group", sort=True):
        try:
            x = build_patient_matrix_for_group(
                stage1_mod, cfg, cohort, panel, sample_type, root, str(fg), g
            )
            status, reason = "ok", ""
        except Exception as exc:
            x = pd.DataFrame(columns=["patient_id"])
            status, reason = "failed", f"{type(exc).__name__}:{exc}"
        audit.append({
            "cohort": cohort, "panel": panel, "sample_type": sample_type,
            ROOT_COL: root, "feature_group": fg, "status": status, "reason": reason,
            "n_requested_features": int(g["feature_uid"].nunique()),
            "n_measured_features": max(0, x.shape[1] - 1),
            "n_patients": int(x["patient_id"].nunique()) if "patient_id" in x else 0,
        })
        if x.empty and len(x.columns) == 1:
            continue
        if merged is None:
            merged = x
        else:
            add = ["patient_id"] + [c for c in x.columns if c != "patient_id" and c not in merged.columns]
            merged = merged.merge(x[add], on="patient_id", how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=["patient_id"])
    for uid in mem_root["feature_uid"].astype(str).drop_duplicates():
        if uid not in merged.columns:
            merged[uid] = np.nan
    return merged, pd.DataFrame(audit)


# -----------------------------------------------------------------------------
# Root / meta scoring
# -----------------------------------------------------------------------------

def fit_z_params(x: pd.Series, ddof: int = 0) -> dict:
    s = safe_numeric(x)
    n = int(s.notna().sum())
    return {
        "n_nonmissing": n,
        "nonmissing_fraction": float(s.notna().mean()) if len(s) else 0.0,
        "mean": float(s.mean()) if n else np.nan,
        "sd": float(s.std(ddof=ddof)) if n else np.nan,
        "n_unique": int(s.dropna().nunique()),
    }


def apply_z(x: pd.Series, p: Mapping) -> pd.Series:
    s = safe_numeric(x)
    sd = p.get("sd", np.nan)
    mu = p.get("mean", np.nan)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def score_root_modules_reference(
    reference_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    mem_root: pd.DataFrame,
    cfg: Mapping,
    cohort: str,
    panel: str,
    reference_label: str,
    target_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_nonmissing = float(cfg.get("min_feature_nonmissing_fraction_for_scoring", 0.20))
    min_patient_frac = float(cfg.get("min_patient_module_feature_fraction", 0.50))
    min_multi = int(cfg.get("min_features_present_multifeature_module", 1))
    ddof = int(cfg.get("zscore_ddof", 0))

    ref = reference_raw.copy()
    tar = target_raw.copy()
    ref["patient_id"] = clean_id_series(ref["patient_id"])
    tar["patient_id"] = clean_id_series(tar["patient_id"])

    ref_z, tar_z = pd.DataFrame(index=ref.index), pd.DataFrame(index=tar.index)
    feature_qc = []
    eligible = {}

    for uid in mem_root["feature_uid"].astype(str).drop_duplicates():
        if uid in ref.columns:
            p = fit_z_params(ref[uid], ddof=ddof)
            elig = bool(
                p["nonmissing_fraction"] >= min_nonmissing
                and p["n_unique"] >= 2
                and np.isfinite(p["sd"]) and p["sd"] > 0
            )
        else:
            p = {"n_nonmissing": 0, "nonmissing_fraction": 0.0, "mean": np.nan,
                 "sd": np.nan, "n_unique": 0}
            elig = False
        eligible[uid] = elig
        ref_z[uid] = apply_z(ref[uid], p) if uid in ref.columns and elig else np.nan
        tar_z[uid] = apply_z(tar[uid], p) if uid in tar.columns and elig else np.nan
        feature_qc.append({
            "cohort": cohort, "panel": panel, ROOT_COL: mem_root[ROOT_COL].iloc[0],
            "feature_uid": uid, "reference_sample_type": reference_label,
            "target_sample_type": target_label, "eligible": elig, **p,
        })

    rows = []
    module_qc = []
    for rid, gm in mem_root.groupby("root_module_id", sort=True):
        uids = gm["feature_uid"].astype(str).drop_duplicates().tolist()
        euids = [u for u in uids if eligible.get(u, False)]
        n_total = len(uids)
        min_present = max(1, int(math.ceil(min_patient_frac * n_total)))
        if n_total > 1:
            min_present = max(min_present, min_multi)

        for label, raw, zmat in [(reference_label, ref, ref_z), (target_label, tar, tar_z)]:
            if euids:
                block = zmat[euids]
                n_present = block.notna().sum(axis=1)
                coverage = n_present / max(1, n_total)
                score = block.mean(axis=1, skipna=True)
                score[(coverage < min_patient_frac) | (n_present < min_present)] = np.nan
            else:
                score = pd.Series(np.nan, index=raw.index)
                n_present = pd.Series(0, index=raw.index)
                coverage = pd.Series(0.0, index=raw.index)
            for i, pid in enumerate(raw["patient_id"]):
                rows.append({
                    "patient_id": str(pid), "cohort": cohort, "panel": panel,
                    "sample_type": label, ROOT_COL: str(mem_root[ROOT_COL].iloc[0]),
                    "root_module_id": str(rid), "score": score.iloc[i],
                    "n_features_total": n_total,
                    "n_features_available_reference": len(euids),
                    "n_features_present_patient": int(n_present.iloc[i]),
                    "feature_fraction_present_patient": float(coverage.iloc[i]),
                })
        module_qc.append({
            "cohort": cohort, "panel": panel, ROOT_COL: str(mem_root[ROOT_COL].iloc[0]),
            "root_module_id": str(rid), "reference_sample_type": reference_label,
            "n_features_total": n_total, "n_features_available_reference": len(euids),
            "feature_availability_fraction": len(euids) / n_total if n_total else np.nan,
        })

    return pd.DataFrame(rows), pd.DataFrame(feature_qc), pd.DataFrame(module_qc)


def zscore_reference_pair(ref: pd.Series, tar: pd.Series) -> Tuple[pd.Series, pd.Series]:
    p = fit_z_params(ref, ddof=0)
    if p["n_unique"] < 2 or not np.isfinite(p["sd"]) or p["sd"] <= 0:
        return pd.Series(np.nan, index=ref.index), pd.Series(np.nan, index=tar.index)
    return apply_z(ref, p), apply_z(tar, p)


def score_meta_modules_reference(
    root_long: pd.DataFrame,
    meta_mem: pd.DataFrame,
    cfg: Mapping,
    reference_label: str,
    target_label: str,
) -> pd.DataFrame:
    if root_long.empty:
        return pd.DataFrame()
    cohort = str(root_long["cohort"].iloc[0])
    panel = str(root_long["panel"].iloc[0])

    ref = root_long[root_long["sample_type"].eq(reference_label)].pivot_table(
        index="patient_id", columns="root_module_id", values="score", aggfunc="first"
    )
    tar = root_long[root_long["sample_type"].eq(target_label)].pivot_table(
        index="patient_id", columns="root_module_id", values="score", aggfunc="first"
    )

    refz = pd.DataFrame(index=ref.index)
    tarz = pd.DataFrame(index=tar.index)
    for rid in sorted(set(ref.columns).union(tar.columns)):
        rs = ref[rid] if rid in ref else pd.Series(np.nan, index=ref.index)
        ts = tar[rid] if rid in tar else pd.Series(np.nan, index=tar.index)
        rz, tz = zscore_reference_pair(rs, ts)
        refz[rid] = rz
        tarz[rid] = tz

    min_frac = float(cfg.get("min_meta_member_fraction", 0.50))
    min_abs = int(cfg.get("min_meta_members_present", 2))
    rows = []

    pm = meta_mem[meta_mem["panel"].astype(str).eq(panel) & meta_mem["meta_module_id"].notna()].copy()
    for mid, gm in pm.groupby("meta_module_id", sort=True):
        members = gm["root_module_id"].astype(str).drop_duplicates().tolist()
        min_req = max(min_abs, int(math.ceil(min_frac * len(members))))
        for label, Z in [(reference_label, refz), (target_label, tarz)]:
            present_cols = [m for m in members if m in Z.columns]
            block = Z[present_cols] if present_cols else pd.DataFrame(index=Z.index)
            n_present = block.notna().sum(axis=1) if len(present_cols) else pd.Series(0, index=Z.index)
            score = block.mean(axis=1, skipna=True) if len(present_cols) else pd.Series(np.nan, index=Z.index)
            score[n_present < min_req] = np.nan
            for pid in Z.index:
                rows.append({
                    "patient_id": str(pid), "cohort": cohort, "panel": panel,
                    "sample_type": label, "meta_module_id": str(mid),
                    "score": score.loc[pid],
                    "n_member_root_modules_total": len(members),
                    "n_member_root_modules_present": int(n_present.loc[pid]),
                })
    return pd.DataFrame(rows)


def unified_program_long(root_scores: pd.DataFrame, meta_scores: pd.DataFrame) -> pd.DataFrame:
    r = root_scores.rename(columns={"root_module_id": "program_id"}).copy()
    r["program_level"] = "root_module"
    r = r[["patient_id", "cohort", "panel", "sample_type", ROOT_COL, "program_id", "program_level", "score"]]

    if meta_scores is None or meta_scores.empty:
        return r
    m = meta_scores.rename(columns={"meta_module_id": "program_id"}).copy()
    m["program_level"] = "meta_module"
    m[ROOT_COL] = "cross_root"
    m = m[["patient_id", "cohort", "panel", "sample_type", ROOT_COL, "program_id", "program_level", "score"]]
    return pd.concat([r, m], ignore_index=True, sort=False)


# -----------------------------------------------------------------------------
# Outcomes and subsets
# -----------------------------------------------------------------------------

def load_harmonized(cfg: Mapping) -> pd.DataFrame:
    d = pd.read_csv(cfg["harmonized_path"], low_memory=False)
    if "patient_id" not in d.columns or "cohort" not in d.columns:
        raise KeyError("harmonized file needs patient_id and cohort")
    d["patient_id"] = clean_id_series(d["patient_id"])
    d["cohort"] = d["cohort"].astype(str).str.strip()
    return d[d["patient_id"].notna()].drop_duplicates(["cohort", "patient_id"]).copy()


def normalize_yes_no(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out[x.isin({"1", "1.0", "yes", "y", "true", "t"})] = "yes"
    out[x.isin({"0", "0.0", "no", "n", "false", "f"})] = "no"
    return out


def apply_subset(d: pd.DataFrame, cohort: str, subset: str) -> pd.DataFrame:
    if subset == "all":
        return d.copy()
    if cohort != "No-NAC":
        raise ValueError(f"{subset} only defined for No-NAC")
    if "adjuvant_chemo" not in d.columns:
        raise KeyError("adjuvant_chemo")
    a = normalize_yes_no(d["adjuvant_chemo"])
    if subset == "no_adj_chemo":
        return d[a.eq("no")].copy()
    if subset == "adj_chemo":
        return d[a.eq("yes")].copy()
    raise ValueError(subset)


def endpoint_table(cfg: Mapping, cohort: str, endpoint: str, subset: str = "all") -> pd.DataFrame:
    d = load_harmonized(cfg)
    d = d[d["cohort"].astype(str).eq(str(cohort))].copy()
    d = apply_subset(d, cohort, subset)

    if endpoint in RESPONSE_ENDPOINTS:
        if endpoint not in d.columns:
            return pd.DataFrame(columns=["patient_id", "y"])
        out = d[["patient_id", endpoint]].rename(columns={endpoint: "y"})
        out["y"] = safe_numeric(out["y"])
        out = out.dropna(subset=["y"]).copy()
        out["y"] = out["y"].astype(int)
        return out

    if endpoint == "OS":
        t = first_existing(d, ["OS_months_RC", "OS_RC_time"])
        e = first_existing(d, ["OS_event", "OS"])
    elif endpoint == "RFS":
        t = first_existing(d, ["REC_months_RC", "RFS_RC_time"])
        e = first_existing(d, ["REC", "RFS_event"])
    else:
        raise ValueError(endpoint)

    # Deliberately NO fallback to TURBT time origin: score/delta is only known at RC.
    if t is None or e is None:
        return pd.DataFrame(columns=["patient_id", "time", "event"])
    out = d[["patient_id", t, e]].rename(columns={t: "time", e: "event"})
    out["time"] = safe_numeric(out["time"])
    out["event"] = safe_numeric(out["event"])
    out = out.dropna(subset=["time", "event"])
    out = out[out["time"] >= 0].copy()
    out["event"] = out["event"].astype(int)
    return out


def context_quality(endpoint: str, ep: pd.DataFrame, cfg: Mapping) -> dict:
    if endpoint in RESPONSE_ENDPOINTS:
        n = len(ep)
        pos, neg = int((ep["y"] == 1).sum()), int((ep["y"] == 0).sum())
        signal = min(pos, neg)
        return {
            "n": n, "events": pos, "n_positive": pos, "n_negative": neg,
            "n_signal": signal,
            "can_fit": bool(n >= int(cfg.get("minimum_fit_n", 8)) and signal >= 2),
            "primary_eligible": bool(
                n >= int(cfg.get("primary_min_n", 20))
                and signal >= int(cfg.get("primary_min_response_class", 5))
            ),
        }
    n = len(ep)
    events = int(ep["event"].sum()) if n else 0
    return {
        "n": n, "events": events, "n_positive": np.nan, "n_negative": np.nan,
        "n_signal": events,
        "can_fit": bool(
            n >= int(cfg.get("minimum_fit_n", 8))
            and events >= int(cfg.get("minimum_survival_events", 3))
        ),
        "primary_eligible": bool(
            n >= int(cfg.get("primary_min_n", 20))
            and events >= int(cfg.get("primary_min_survival_events", 5))
        ),
    }


# -----------------------------------------------------------------------------
# RC clinical variables
# -----------------------------------------------------------------------------

AGE_CANDIDATES = ["Age", "age"]
SEX_CANDIDATES = ["Sex", "gender"]
YPT_CANDIDATES = ["ypT_STAGE_ordinal", "Cystectomy_ypTstage", "pT_STAGE_RC", "pstage", "ypT_STAGE"]
YPN_CANDIDATES = ["ypN_STAGE_ordinal", "Cystectomy_ypNstage", "pN_STAGE_RC", "ypN_STAGE", "pN"]


def stage_to_ordinal(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    s = s.replace("YPT", "").replace("YPN", "").replace("PT", "").replace("PN", "")
    s = s.replace("T", "").replace("N", "").replace("Y", "").replace("P", "")
    if s in {"IS", "A", "TA", "TIS"}:
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def load_rc_clinical_variables(cfg: Mapping, cohort: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clinical_cohort = "NoNAC" if cohort == "No-NAC" else cohort
    p = Path(cfg["clinical_matrix_root"]) / f"{clinical_cohort}_RC_clinical_matrix.csv"
    if not p.exists():
        return pd.DataFrame(columns=["patient_id"]), pd.DataFrame([{
            "cohort": cohort, "clinical_variable": "ALL", "status": "missing_matrix",
            "source_column": "", "path": str(p)
        }])

    d = pd.read_csv(p)
    pid = first_existing(d, ["patient_ID", "patient_id", "Patient_ID"])
    if pid is None:
        raise KeyError(f"No patient ID in {p}")
    d["patient_id"] = clean_id_series(d[pid])
    d = d[d["patient_id"].notna()].drop_duplicates("patient_id").copy()

    groups = {"Age": AGE_CANDIDATES, "Sex": SEX_CANDIDATES, "ypT": YPT_CANDIDATES, "ypN": YPN_CANDIDATES}
    out = pd.DataFrame({"patient_id": d["patient_id"].values})
    audit = []
    for label, candidates in groups.items():
        source = None
        for c in candidates:
            if c in d.columns and d[c].notna().sum() >= 3 and d[c].dropna().nunique() > 1:
                source = c
                break
        if source is None:
            audit.append({"cohort": cohort, "clinical_variable": label, "status": "not_available",
                          "source_column": "", "path": str(p)})
            continue
        s = d[source]
        if label == "Age":
            x = safe_numeric(s)
        elif label in {"ypT", "ypN"}:
            x = safe_numeric(s)
            if x.notna().mean() < 0.8:
                x = s.map(stage_to_ordinal)
        else:
            xn = safe_numeric(s)
            if xn.notna().mean() >= 0.8 and xn.dropna().nunique() == 2:
                levels = sorted(xn.dropna().unique())
                x = xn.map({levels[0]: 0.0, levels[1]: 1.0})
            else:
                ss = s.astype("string").str.strip()
                levels = sorted(ss.dropna().unique().tolist())
                x = ss.map({levels[0]: 0.0, levels[1]: 1.0}).astype(float) if len(levels) == 2 else pd.Series(np.nan, index=s.index)
        out[label] = x.to_numpy()
        audit.append({
            "cohort": cohort, "clinical_variable": label,
            "status": "ok" if out[label].notna().sum() >= 3 and out[label].dropna().nunique() > 1 else "unusable",
            "source_column": source, "path": str(p),
            "n_nonmissing": int(out[label].notna().sum()),
            "n_unique": int(out[label].dropna().nunique()),
        })
    return out, pd.DataFrame(audit)


# -----------------------------------------------------------------------------
# Univariate fits and repeated OOF
# -----------------------------------------------------------------------------

def standardize_train_valid(tr: pd.Series, va: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x = safe_numeric(tr)
    z = safe_numeric(va)
    mu, sd = x.mean(), x.std(ddof=0)
    if x.notna().sum() < 2 or not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index), pd.Series(np.nan, index=z.index)
    return (x - mu) / sd, (z - mu) / sd


def fullfit_response(score: pd.Series, y: pd.Series) -> dict:
    d = pd.DataFrame({"score": safe_numeric(score), "y": safe_numeric(y)}).dropna()
    if len(d) < 3 or d["y"].nunique() < 2 or d["score"].nunique() < 2:
        return {"fit_status": "insufficient"}
    z = (d["score"] - d["score"].mean()) / d["score"].std(ddof=0)
    try:
        X = sm.add_constant(z.to_frame("score"), has_constant="add")
        fit = sm.Logit(d["y"].astype(int), X).fit(disp=False, maxiter=300)
        lo, hi = fit.conf_int().loc["score"]
        coef = float(fit.params["score"])
        pred = fit.predict(X)
        return {
            "coef": coef, "ci_low": float(lo), "ci_high": float(hi),
            "effect": float(np.exp(coef)), "effect_ci_low": float(np.exp(lo)),
            "effect_ci_high": float(np.exp(hi)), "p_value": float(fit.pvalues["score"]),
            "full_metric": float(roc_auc_score(d["y"].astype(int), pred)),
            "n": len(d), "events": int(d["y"].sum()), "fit_status": "logit",
        }
    except Exception as exc:
        return {"fit_status": f"failed:{type(exc).__name__}"}


def fullfit_survival(score: pd.Series, time: pd.Series, event: pd.Series) -> dict:
    d = pd.DataFrame({"score": safe_numeric(score), "time": safe_numeric(time), "event": safe_numeric(event)}).dropna()
    if len(d) < 3 or d["event"].sum() < 1 or d["score"].nunique() < 2:
        return {"fit_status": "insufficient"}
    d["score"] = (d["score"] - d["score"].mean()) / d["score"].std(ddof=0)
    try:
        cph = CoxPHFitter(penalizer=COX_PENALIZER)
        cph.fit(d, duration_col="time", event_col="event")
        s = cph.summary.loc["score"]
        risk = cph.predict_partial_hazard(d[["score"]]).to_numpy().ravel()
        return {
            "coef": float(s["coef"]), "ci_low": float(s["coef lower 95%"]),
            "ci_high": float(s["coef upper 95%"]), "effect": float(s["exp(coef)"]),
            "effect_ci_low": float(s["exp(coef) lower 95%"]),
            "effect_ci_high": float(s["exp(coef) upper 95%"]),
            "p_value": float(s["p"]),
            "full_metric": float(concordance_index(d["time"], -risk, d["event"])),
            "n": len(d), "events": int(d["event"].sum()), "fit_status": "cox",
        }
    except Exception as exc:
        return {"fit_status": f"failed:{type(exc).__name__}"}


def repeated_oof_response(score, y, ids, n_splits=5, n_repeats=5, seed=42):
    d = pd.DataFrame({"patient_id": clean_id_series(ids), "score": safe_numeric(score), "y": safe_numeric(y)}).dropna()
    if d["y"].nunique() < 2 or d["score"].nunique() < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()
    k = min(int(n_splits), int(d["y"].value_counts().min()))
    if k < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()
    pred, folds = [], []
    for rep in range(int(n_repeats)):
        sp = StratifiedKFold(k, shuffle=True, random_state=seed + rep)
        for fold, (tr, va) in enumerate(sp.split(d, d["y"].astype(int)), 1):
            a, b = d.iloc[tr].copy(), d.iloc[va].copy()
            ztr, zva = standardize_train_valid(a["score"], b["score"])
            try:
                model = LogisticRegression(penalty="l2", C=1, solver="liblinear", class_weight="balanced",
                                           max_iter=1000, random_state=seed + rep)
                model.fit(ztr.to_frame("score"), a["y"].astype(int))
                pr = model.predict_proba(zva.to_frame("score"))[:, 1]
                coef = float(model.coef_.ravel()[0])
                fm = float(roc_auc_score(b["y"].astype(int), pr)) if b["y"].nunique() == 2 else np.nan
                folds.append({"repeat": rep+1, "fold": fold, "fold_metric": fm, "fold_coef": coef})
                for pid, yy, pp in zip(b["patient_id"], b["y"].astype(int), pr):
                    pred.append({"patient_id": pid, "repeat": rep+1, "fold": fold, "y_true": yy, "prediction": pp})
            except Exception:
                pass
    pred = pd.DataFrame(pred); folds = pd.DataFrame(folds)
    if pred.empty:
        return np.nan, pred, folds
    avg = pred.groupby("patient_id", as_index=False).agg(y_true=("y_true", "first"), prediction=("prediction", "mean"))
    auc = float(roc_auc_score(avg["y_true"], avg["prediction"])) if avg["y_true"].nunique() == 2 else np.nan
    return auc, avg, folds


def repeated_oof_survival(score, time, event, ids, n_splits=5, n_repeats=5, seed=42):
    d = pd.DataFrame({
        "patient_id": clean_id_series(ids), "score": safe_numeric(score),
        "time": safe_numeric(time), "event": safe_numeric(event)
    }).dropna()
    if d["event"].sum() < 2 or d["score"].nunique() < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()
    k = min(int(n_splits), int(d["event"].sum()), len(d))
    if k < 2:
        return np.nan, pd.DataFrame(), pd.DataFrame()
    pred, folds = [], []
    strat_ok = d["event"].value_counts().min() >= k
    for rep in range(int(n_repeats)):
        if strat_ok:
            sp = StratifiedKFold(k, shuffle=True, random_state=seed + rep)
            it = sp.split(d, d["event"].astype(int))
        else:
            sp = KFold(k, shuffle=True, random_state=seed + rep)
            it = sp.split(d)
        for fold, (tr, va) in enumerate(it, 1):
            a, b = d.iloc[tr].copy(), d.iloc[va].copy()
            if a["event"].sum() < 1:
                continue
            ztr, zva = standardize_train_valid(a["score"], b["score"])
            fit = pd.DataFrame({"time": a["time"], "event": a["event"], "score": ztr}).dropna()
            if fit["event"].sum() < 1 or fit["score"].nunique() < 2:
                continue
            try:
                cph = CoxPHFitter(penalizer=COX_PENALIZER)
                cph.fit(fit, duration_col="time", event_col="event")
                coef = float(cph.params_["score"])
                risk = coef * zva.to_numpy()
                fm = np.nan
                try:
                    if b["event"].sum() > 0:
                        fm = float(concordance_index(b["time"], -risk, b["event"]))
                except Exception:
                    pass
                folds.append({"repeat": rep+1, "fold": fold, "fold_metric": fm, "fold_coef": coef})
                for pid, tt, ee, rr in zip(b["patient_id"], b["time"], b["event"].astype(int), risk):
                    if np.isfinite(rr):
                        pred.append({"patient_id": pid, "repeat": rep+1, "fold": fold,
                                     "time": tt, "event": ee, "prediction": rr})
            except Exception:
                pass
    pred = pd.DataFrame(pred); folds = pd.DataFrame(folds)
    if pred.empty:
        return np.nan, pred, folds
    avg = pred.groupby("patient_id", as_index=False).agg(
        time=("time", "first"), event=("event", "first"), prediction=("prediction", "mean")
    )
    try:
        ci = float(concordance_index(avg["time"], -avg["prediction"], avg["event"]))
    except Exception:
        ci = np.nan
    return ci, avg, folds
