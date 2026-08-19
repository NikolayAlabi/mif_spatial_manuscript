#!/usr/bin/env python3
"""
stage2c_score_root_modules_v1.py

Stage 2C for the root -> meta-module workflow.

Purpose
-------
1. Freeze the final root-module memberships from Stage 2B-2 manual K selections.
2. Score those frozen root modules in the discovery TURBT cohort x root matrices
   already built in Stage 2B-1.
3. Write canonical long/wide patient root-module score tables for outcome-blind
   Stage 2D meta-module discovery.

Important design choices
------------------------
* No outcomes are read or used.
* No feature/module selection occurs here.
* Primary score = mean of within-cohort feature z-scores ("meanz").
* Features are never sign-flipped. The Stage 2B direct-cohesion diagnostics are
  therefore an important guardrail for the chosen modules.
* This v1 intentionally scores the discovery TURBT matrices cached by Stage 2B-1.
  After meta-modules are frozen, the same definitions can be applied to RC,
  NAC2015, and future validation cohorts in the downstream evaluation scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT_COL = "feature_source"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_slug(x: object) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")
    return s or "NA"


def read_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["output_root"])


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".gz" and path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_parquet(path, index=False)
    return path


def load_b1_manifest(stage2b1_root: Path) -> pd.DataFrame:
    p = stage2b1_root / "stage2b1_root_consensus_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    need = {"panel", ROOT_COL}
    miss = need - set(d.columns)
    if miss:
        raise KeyError(f"{p} missing columns: {sorted(miss)}")
    return d


def load_matrix_summary(stage2b1_root: Path) -> pd.DataFrame:
    p = stage2b1_root / "all_cohort_root_matrix_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        if "matrix_path" in d.columns:
            return d

    # Fallback to worker index + worker summaries.
    idxp = stage2b1_root / "stage2b1_matrix_worker_index.csv"
    if not idxp.exists():
        raise FileNotFoundError(p)
    idx = pd.read_csv(idxp)
    rows = []
    for _, r in idx.iterrows():
        mdir = stage2b1_root / "cohort_root_matrices" / str(r["matrix_slug"])
        sp = mdir / "matrix_worker_summary.csv"
        if sp.exists():
            rows.append(pd.read_csv(sp))
    if not rows:
        raise RuntimeError("No completed Stage 2B-1 matrix summaries found")
    return pd.concat(rows, ignore_index=True, sort=False)


def load_square_consensus(stage2b1_root: Path, panel: str, root: str, manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest[
        manifest["panel"].astype(str).eq(str(panel))
        & manifest[ROOT_COL].astype(str).eq(str(root))
    ]
    if m.empty:
        raise KeyError(f"No Stage2B1 manifest row for {panel}/{root}")
    r = m.iloc[0]
    wdir = None
    if "root_output_dir" in m.columns and pd.notna(r.get("root_output_dir")):
        q = Path(str(r["root_output_dir"]))
        if q.exists():
            wdir = q
    if wdir is None:
        slug = r.get("consensus_slug")
        if pd.notna(slug):
            q = stage2b1_root / "root_consensus" / str(slug)
            if q.exists():
                wdir = q
    if wdir is None:
        wdir = stage2b1_root / "root_consensus" / f"{safe_slug(panel)}__{safe_slug(root)}"
    p = wdir / "consensus_signed_spearman_support_filtered_clustering.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    C = pd.read_parquet(p)
    if "feature_uid" in C.columns:
        C = C.set_index("feature_uid")
    C.index = C.index.astype(str)
    C.columns = C.columns.astype(str)
    common = [f for f in C.index if f in C.columns]
    C = C.loc[common, common]
    return C


def load_feature_universe(stage2b1_root: Path, panel: str, root: str, manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest[
        manifest["panel"].astype(str).eq(str(panel))
        & manifest[ROOT_COL].astype(str).eq(str(root))
    ]
    if m.empty:
        return pd.DataFrame()
    r = m.iloc[0]
    candidates = []
    if "root_output_dir" in m.columns and pd.notna(r.get("root_output_dir")):
        candidates.append(Path(str(r["root_output_dir"])) / "feature_universe.csv")
    if "feature_universe_path" in m.columns and pd.notna(r.get("feature_universe_path")):
        candidates.append(Path(str(r["feature_universe_path"])))
    if "consensus_slug" in m.columns and pd.notna(r.get("consensus_slug")):
        candidates.append(stage2b1_root / "root_consensus" / str(r["consensus_slug"]) / "feature_universe.csv")
    candidates.append(stage2b1_root / "root_consensus" / f"{safe_slug(panel)}__{safe_slug(root)}" / "feature_universe.csv")
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def build_distance(C: pd.DataFrame, mode: str) -> pd.DataFrame:
    Z = C.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "direct_signed":
        D = 1.0 - Z
    elif mode == "direct_abs":
        D = 1.0 - Z.abs()
    elif mode == "row_spearman":
        rc = Z.T.corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        D = 1.0 - rc
    else:
        raise ValueError(f"Unknown distance mode: {mode}")
    D = D.clip(lower=0.0, upper=2.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D.values, 0.0)
    return D


def cluster_order_map(C: pd.DataFrame, raw_membership: pd.DataFrame, mode: str, linkage_method: str) -> Dict[int, int]:
    D = build_distance(C, mode)
    arr = D.to_numpy(float)
    Z = linkage(squareform(arr, checks=False), method=linkage_method)
    leaves = list(leaves_list(Z))
    feature_order = [D.index[i] for i in leaves]
    fmap = dict(zip(raw_membership["feature_uid"].astype(str), raw_membership["raw_cluster_id"].astype(int)))
    seen: List[int] = []
    for f in feature_order:
        lab = fmap.get(str(f))
        if lab is not None and lab not in seen:
            seen.append(int(lab))
    # Any unexpected labels not represented in the matrix are appended deterministically.
    for lab in sorted(set(fmap.values())):
        if int(lab) not in seen:
            seen.append(int(lab))
    return {lab: i + 1 for i, lab in enumerate(seen)}


def load_final_k(cfg: Mapping) -> pd.DataFrame:
    p = Path(cfg["final_k_csv"])
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    need = {"panel", ROOT_COL, "manual_selected_k"}
    miss = need - set(d.columns)
    if miss:
        raise KeyError(f"{p} missing columns: {sorted(miss)}")
    d["manual_selected_k"] = pd.to_numeric(d["manual_selected_k"], errors="coerce")
    d = d[d["manual_selected_k"].notna()].copy()
    d["manual_selected_k"] = d["manual_selected_k"].astype(int)
    if d.empty:
        raise RuntimeError("No manual_selected_k values found in final K CSV")
    if d.duplicated(["panel", ROOT_COL]).any():
        raise RuntimeError("Duplicate panel/root rows in final K CSV")
    return d.sort_values(["panel", ROOT_COL]).reset_index(drop=True)


def membership_path(cfg: Mapping, panel: str, root: str, mode: str) -> Path:
    b2 = Path(cfg["stage2b2_output_root"])
    p = b2 / "roots" / f"{safe_slug(panel)}__{safe_slug(root)}" / safe_slug(mode) / "memberships_all_k.csv.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def command_validate(cfg: Mapping) -> None:
    b1 = Path(cfg["stage2b1_output_root"])
    b2 = Path(cfg["stage2b2_output_root"])
    for p in [b1, b2, Path(cfg["final_k_csv"])]:
        if not p.exists():
            raise FileNotFoundError(p)
    fk = load_final_k(cfg)
    manifest = load_b1_manifest(b1)
    matrix = load_matrix_summary(b1)
    problems = []
    for _, r in fk.iterrows():
        panel, root = str(r["panel"]), str(r[ROOT_COL])
        mode = str(r.get("primary_distance_mode", cfg.get("primary_distance_mode", "row_spearman")))
        if mode in {"", "nan", "None"}:
            mode = str(cfg.get("primary_distance_mode", "row_spearman"))
        try:
            membership_path(cfg, panel, root, mode)
        except Exception as exc:
            problems.append({"panel": panel, ROOT_COL: root, "problem": str(exc)})
        if manifest[(manifest.panel.astype(str) == panel) & (manifest[ROOT_COL].astype(str) == root)].empty:
            problems.append({"panel": panel, ROOT_COL: root, "problem": "missing B1 consensus manifest row"})
        if matrix[(matrix.panel.astype(str) == panel) & (matrix[ROOT_COL].astype(str) == root)].empty:
            problems.append({"panel": panel, ROOT_COL: root, "problem": "missing B1 cohort-root matrices"})
    out = ensure_dir(output_root(cfg))
    pd.DataFrame(problems).to_csv(out / "stage2c_validation_problems.csv", index=False)
    if problems:
        raise RuntimeError(f"Stage2C validation found {len(problems)} problem(s)")
    log(f"[VALID] final panel-roots={len(fk)} cohort-root matrices={len(matrix)}")


def command_setup(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    b1 = Path(cfg["stage2b1_output_root"])
    fk = load_final_k(cfg)
    manifest = load_b1_manifest(b1)
    matrix = load_matrix_summary(b1)
    linkage_method = str(cfg.get("linkage_method", "average"))

    all_members = []
    summaries = []

    for _, sel in fk.iterrows():
        panel = str(sel["panel"])
        root = str(sel[ROOT_COL])
        k = int(sel["manual_selected_k"])
        mode = str(sel.get("primary_distance_mode", cfg.get("primary_distance_mode", "row_spearman")))
        if mode in {"", "nan", "None"}:
            mode = str(cfg.get("primary_distance_mode", "row_spearman"))

        mem = pd.read_csv(membership_path(cfg, panel, root, mode))
        mem = mem[
            mem["requested_k"].astype(int).eq(k)
            & mem["distance_mode"].astype(str).eq(mode)
        ].copy()
        if mem.empty:
            raise RuntimeError(f"No membership for {panel}/{root}, mode={mode}, K={k}")
        mem["feature_uid"] = mem["feature_uid"].astype(str)

        C = load_square_consensus(b1, panel, root, manifest)
        common = [f for f in C.index if f in set(mem["feature_uid"])]
        mem = mem[mem["feature_uid"].isin(common)].copy()
        if mem.empty:
            raise RuntimeError(f"No common consensus/membership features for {panel}/{root}")
        remap = cluster_order_map(C.loc[common, common], mem, mode, linkage_method)
        mem["module_num"] = mem["raw_cluster_id"].astype(int).map(remap).astype(int)
        mem["module_label"] = mem["module_num"].map(lambda x: f"M{x:02d}")
        mem["root_module_id"] = mem.apply(
            lambda r: f"{panel}__{root}__{r['module_label']}", axis=1
        )
        mem["final_k"] = k
        mem["primary_distance_mode"] = mode

        uni = load_feature_universe(b1, panel, root, manifest)
        if not uni.empty:
            uid_col = "stage2b_feature_uid" if "stage2b_feature_uid" in uni.columns else "feature_uid"
            if uid_col in uni.columns:
                uni = uni.rename(columns={uid_col: "feature_uid"})
                uni["feature_uid"] = uni["feature_uid"].astype(str)
                keep = [c for c in uni.columns if c not in mem.columns or c == "feature_uid"]
                mem = mem.merge(uni[keep], on="feature_uid", how="left")

        all_members.append(mem)
        for module_id, g in mem.groupby("root_module_id", sort=True):
            summaries.append({
                "panel": panel,
                ROOT_COL: root,
                "final_k": k,
                "primary_distance_mode": mode,
                "module_num": int(g["module_num"].iloc[0]),
                "module_label": str(g["module_label"].iloc[0]),
                "root_module_id": str(module_id),
                "n_features": int(g["feature_uid"].nunique()),
            })

    members = pd.concat(all_members, ignore_index=True, sort=False)
    module_summary = pd.DataFrame(summaries).sort_values(["panel", ROOT_COL, "module_num"])
    members.to_csv(out / "final_root_module_membership.csv", index=False)
    module_summary.to_csv(out / "final_root_module_summary.csv", index=False)
    fk.to_csv(out / "final_k_snapshot.csv", index=False)

    # One worker per discovery cohort x panel. Each worker loops over roots.
    workers = (
        matrix[["cohort", "panel"]]
        .drop_duplicates()
        .sort_values(["cohort", "panel"])
        .reset_index(drop=True)
    )
    workers.insert(0, "array_id", np.arange(1, len(workers) + 1, dtype=int))
    workers.to_csv(out / "stage2c_worker_index.csv", index=False)

    setup_summary = {
        "n_panel_roots": int(len(fk)),
        "n_root_modules": int(module_summary["root_module_id"].nunique()),
        "n_feature_memberships": int(len(members)),
        "n_workers": int(len(workers)),
        "score_method": "mean_of_within_cohort_feature_zscores",
    }
    write_json(setup_summary, out / "stage2c_setup_summary.json")
    log(f"[DONE SETUP] root_modules={setup_summary['n_root_modules']} workers={len(workers)}")


def load_index_row(path: Path, array_id: int) -> pd.Series:
    d = pd.read_csv(path)
    z = d[pd.to_numeric(d["array_id"], errors="coerce").eq(int(array_id))]
    if z.empty:
        raise KeyError(f"array_id={array_id} not found in {path}")
    return z.iloc[0]


def zscore_feature(s: pd.Series, ddof: int = 0) -> Tuple[pd.Series, dict]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n = int(x.notna().sum())
    mu = float(x.mean()) if n else np.nan
    sd = float(x.std(ddof=ddof)) if n else np.nan
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        z = pd.Series(np.nan, index=x.index, dtype=float)
    else:
        z = (x - mu) / sd
    return z, {"n_nonmissing": n, "mean": mu, "sd": sd, "n_unique": int(x.dropna().nunique())}


def command_worker(cfg: Mapping, array_id: int) -> None:
    out = output_root(cfg)
    row = load_index_row(out / "stage2c_worker_index.csv", array_id)
    cohort = str(row["cohort"])
    panel = str(row["panel"])
    wdir = ensure_dir(out / "workers" / f"{safe_slug(cohort)}__{safe_slug(panel)}")

    membership = pd.read_csv(out / "final_root_module_membership.csv")
    membership = membership[membership["panel"].astype(str).eq(panel)].copy()
    matrix_summary = load_matrix_summary(Path(cfg["stage2b1_output_root"]))
    matrix_summary = matrix_summary[
        matrix_summary["cohort"].astype(str).eq(cohort)
        & matrix_summary["panel"].astype(str).eq(panel)
    ].copy()
    if matrix_summary.empty:
        raise RuntimeError(f"No B1 matrices for {cohort}/{panel}")

    min_feature_nonmissing = float(cfg.get("min_feature_nonmissing_fraction_for_scoring", 0.20))
    min_patient_frac = float(cfg.get("min_patient_module_feature_fraction", 0.50))
    ddof = int(cfg.get("zscore_ddof", 0))

    score_parts = []
    feature_qc_rows = []
    module_qc_rows = []

    for root, mem_root in membership.groupby(ROOT_COL, sort=True):
        mr = matrix_summary[matrix_summary[ROOT_COL].astype(str).eq(str(root))]
        if mr.empty:
            log(f"[WARN] no matrix for {cohort}/{panel}/{root}")
            continue
        matrix_path = Path(str(mr.iloc[0]["matrix_path"]))
        if not matrix_path.exists():
            raise FileNotFoundError(matrix_path)
        X = read_table(matrix_path)
        if "patient_id" not in X.columns:
            raise KeyError(f"patient_id missing from {matrix_path}")
        X["patient_id"] = X["patient_id"].astype(str)

        uids = mem_root["feature_uid"].astype(str).drop_duplicates().tolist()
        zmat = pd.DataFrame(index=X.index)
        feature_eligible: Dict[str, bool] = {}
        for uid in uids:
            if uid not in X.columns:
                zmat[uid] = np.nan
                qc = {"n_nonmissing": 0, "mean": np.nan, "sd": np.nan, "n_unique": 0}
                frac = 0.0
            else:
                z, qc = zscore_feature(X[uid], ddof=ddof)
                zmat[uid] = z
                frac = float(qc["n_nonmissing"] / len(X)) if len(X) else 0.0
            eligible = bool(frac >= min_feature_nonmissing and qc["n_unique"] >= 2 and np.isfinite(qc["sd"]) and qc["sd"] > 0)
            feature_eligible[uid] = eligible
            feature_qc_rows.append({
                "array_id": array_id,
                "cohort": cohort,
                "panel": panel,
                ROOT_COL: root,
                "feature_uid": uid,
                "nonmissing_fraction": frac,
                "eligible_for_scoring": eligible,
                **qc,
            })

        for module_id, gm in mem_root.groupby("root_module_id", sort=True):
            module_label = str(gm["module_label"].iloc[0])
            module_uids = gm["feature_uid"].astype(str).drop_duplicates().tolist()
            eligible_uids = [u for u in module_uids if feature_eligible.get(u, False)]
            n_total = len(module_uids)
            n_available = len(eligible_uids)

            if eligible_uids:
                block = zmat[eligible_uids]
                n_present = block.notna().sum(axis=1)
                denom = max(n_total, 1)
                coverage = n_present / denom
                min_present = max(1, int(math.ceil(min_patient_frac * n_total)))
                if n_total > 1:
                    min_present = max(min_present, int(cfg.get("min_features_present_multifeature_module", 1)))
                score = block.mean(axis=1, skipna=True)
                score[(coverage < min_patient_frac) | (n_present < min_present)] = np.nan
            else:
                n_present = pd.Series(0, index=X.index, dtype=int)
                coverage = pd.Series(0.0, index=X.index, dtype=float)
                score = pd.Series(np.nan, index=X.index, dtype=float)

            part = pd.DataFrame({
                "patient_id": X["patient_id"].values,
                "cohort": cohort,
                "panel": panel,
                ROOT_COL: root,
                "module_label": module_label,
                "root_module_id": str(module_id),
                "score_meanz": score.values,
                "n_features_total": n_total,
                "n_features_available_cohort": n_available,
                "n_features_present_patient": n_present.values,
                "feature_fraction_present_patient": coverage.values,
            })
            score_parts.append(part)
            module_qc_rows.append({
                "array_id": array_id,
                "cohort": cohort,
                "panel": panel,
                ROOT_COL: root,
                "module_label": module_label,
                "root_module_id": str(module_id),
                "n_features_total": n_total,
                "n_features_available_cohort": n_available,
                "feature_availability_fraction": float(n_available / n_total) if n_total else np.nan,
                "n_patients": int(len(X)),
                "n_patients_scored": int(score.notna().sum()),
                "patient_score_fraction": float(score.notna().mean()) if len(score) else np.nan,
                "score_mean": float(score.mean()) if score.notna().any() else np.nan,
                "score_sd": float(score.std(ddof=0)) if score.notna().sum() >= 2 else np.nan,
            })

    scores = pd.concat(score_parts, ignore_index=True, sort=False) if score_parts else pd.DataFrame()
    if scores.empty:
        raise RuntimeError(f"No root-module scores generated for {cohort}/{panel}")

    scores.to_parquet(wdir / "root_module_scores_long.parquet", index=False)
    wide = scores.pivot_table(index=["patient_id", "cohort", "panel"], columns="root_module_id", values="score_meanz", aggfunc="first").reset_index()
    wide.columns.name = None
    wide.to_parquet(wdir / "root_module_scores_wide.parquet", index=False)
    pd.DataFrame(feature_qc_rows).to_csv(wdir / "feature_standardization_qc.csv", index=False)
    pd.DataFrame(module_qc_rows).to_csv(wdir / "root_module_score_qc.csv", index=False)

    summary = {
        "array_id": array_id,
        "cohort": cohort,
        "panel": panel,
        "n_patients": int(scores["patient_id"].nunique()),
        "n_root_modules": int(scores["root_module_id"].nunique()),
        "n_scores_nonmissing": int(scores["score_meanz"].notna().sum()),
        "score_long_path": str(wdir / "root_module_scores_long.parquet"),
        "score_wide_path": str(wdir / "root_module_scores_wide.parquet"),
    }
    pd.DataFrame([summary]).to_csv(wdir / "worker_summary.csv", index=False)
    (wdir / ".done").write_text("complete\n")
    log(f"[DONE] {cohort}/{panel}: patients={summary['n_patients']} modules={summary['n_root_modules']}")


def command_aggregate(cfg: Mapping) -> None:
    out = output_root(cfg)
    idx = pd.read_csv(out / "stage2c_worker_index.csv")
    score_parts = []
    qc_parts = []
    fq_parts = []
    summary_parts = []
    missing = []

    for _, r in idx.iterrows():
        wdir = out / "workers" / f"{safe_slug(r['cohort'])}__{safe_slug(r['panel'])}"
        if not (wdir / ".done").exists():
            missing.append({"array_id": r["array_id"], "cohort": r["cohort"], "panel": r["panel"]})
            continue
        score_parts.append(pd.read_parquet(wdir / "root_module_scores_long.parquet"))
        qc_parts.append(pd.read_csv(wdir / "root_module_score_qc.csv"))
        fq_parts.append(pd.read_csv(wdir / "feature_standardization_qc.csv"))
        summary_parts.append(pd.read_csv(wdir / "worker_summary.csv"))

    pd.DataFrame(missing).to_csv(out / "stage2c_missing_workers.csv", index=False)
    if missing:
        raise RuntimeError(f"{len(missing)} Stage2C worker(s) missing")

    scores = pd.concat(score_parts, ignore_index=True, sort=False)
    qc = pd.concat(qc_parts, ignore_index=True, sort=False)
    fq = pd.concat(fq_parts, ignore_index=True, sort=False)
    ws = pd.concat(summary_parts, ignore_index=True, sort=False)

    scores.to_parquet(out / "all_discovery_root_module_scores_long.parquet", index=False)
    scores.to_csv(out / "all_discovery_root_module_scores_long.csv.gz", index=False, compression="gzip")
    wide = scores.pivot_table(index=["patient_id", "cohort", "panel"], columns="root_module_id", values="score_meanz", aggfunc="first").reset_index()
    wide.columns.name = None
    wide.to_parquet(out / "all_discovery_root_module_scores_wide.parquet", index=False)
    qc.to_csv(out / "all_discovery_root_module_score_qc.csv", index=False)
    fq.to_csv(out / "all_discovery_feature_standardization_qc.csv.gz", index=False, compression="gzip")
    ws.to_csv(out / "stage2c_worker_summary.csv", index=False)

    module_summary = pd.read_csv(out / "final_root_module_summary.csv")
    manifest = module_summary.merge(
        qc.groupby(["panel", ROOT_COL, "root_module_id"], as_index=False).agg(
            n_cohorts_scored=("cohort", "nunique"),
            median_feature_availability_fraction=("feature_availability_fraction", "median"),
            median_patient_score_fraction=("patient_score_fraction", "median"),
        ),
        on=["panel", ROOT_COL, "root_module_id"], how="left",
    )
    manifest.to_csv(out / "stage2c_root_module_manifest.csv", index=False)

    summary = {
        "n_workers": int(len(ws)),
        "n_cohorts": int(scores["cohort"].nunique()),
        "n_panels": int(scores["panel"].nunique()),
        "n_root_modules": int(scores["root_module_id"].nunique()),
        "n_patients_panel_specific": int(scores[["cohort", "panel", "patient_id"]].drop_duplicates().shape[0]),
        "score_method": "mean_of_within_cohort_feature_zscores",
    }
    write_json(summary, out / "stage2c_aggregate_summary.json")
    log(f"[DONE AGGREGATE] cohorts={summary['n_cohorts']} root_modules={summary['n_root_modules']}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("setup")
    w = sub.add_parser("worker")
    w.add_argument("--array-id", type=int, default=None)
    sub.add_parser("aggregate")
    return ap.parse_args()


def resolve_array_id(cli: Optional[int]) -> int:
    if cli is not None:
        return int(cli)
    import os
    val = os.environ.get("SLURM_ARRAY_TASK_ID")
    if val is None:
        raise RuntimeError("Provide --array-id or run as a Slurm array task")
    return int(val)


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "setup":
        command_setup(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
