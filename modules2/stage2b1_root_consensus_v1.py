#!/usr/bin/env python3
"""
stage2b1_root_consensus_v1.py

Root-aware Stage 2B-1 for the finalized mIF root -> meta-module workflow.

Purpose
-------
Convert the endpoint-specific Stage 2A-5 survivors into endpoint-independent,
cohort-level feature matrices and cross-cohort consensus correlation structures,
separately for each panel x prep root.

Design
------
1. SETUP reads the large Stage 2A-5 aggregate candidate table ONCE.
   It freezes the union feature universe for each panel x prep root, writes small
   per-root universe CSVs, and builds worker indices.
2. MATRIX WORKERS run one CPU per cohort x panel x prep root. Each worker rebuilds
   all features in that root universe for that cohort from the tested Stage 1 v6
   data-loading/aggregation functions. A feature does not need to have been
   nominated in that cohort to be measured there.
3. CONSENSUS WORKERS run one CPU per panel x prep root. Each worker computes
   within-cohort signed Spearman correlations, pairwise-complete N, equal-weight
   cross-cohort consensus rho, pair support, sign consistency, and support filters.
4. AGGREGATE collects root summaries and writes the Stage 2B-2-ready manifest.

Important scientific behavior
-----------------------------
* Discovery cohorts are equal-weighted; endpoint count never weights a cohort.
* Stage 2B-1 is endpoint-independent after feature nomination.
* No requirement that a raw feature was nominated in >=2 cohorts. Nomination
  support is retained as annotation only.
* Correlations are SIGNED Spearman correlations.
* Raw patient-level values are sufficient for Spearman geometry: z-score and
  log1p+z-score choices used in Stage 1 are monotonic transformations and do not
  change rank ordering when defined. No outcome is used here.
* Pair support is the number of cohorts with an estimable pairwise correlation.
* Support-filtered consensus requires pair support >= min_pair_support; unsupported
  pairs are set to 0 only in the clustering-ready matrix (neutral correlation).
* No clustering or K selection occurs here. That is Stage 2B-2.

Commands
--------
validate          Validate inputs/config.
setup             Build root universes + matrix/consensus worker indices.
matrix-worker     Build one cohort x panel x prep-root union patient matrix.
consensus-worker  Build one panel x prep-root consensus package.
aggregate         Verify workers and combine summaries/manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_COL = "feature_source"
DISCOVERY_CONTEXT_COLS = ["cohort", "panel", "sample_type", "patient_subset", "agg"]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, path: Union[str, Path]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


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
    low = p.name.lower()
    if low.endswith(".parquet"):
        return pd.read_parquet(p)
    if low.endswith(".tsv") or low.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def save_table(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=index)
            return p
        except (ImportError, ModuleNotFoundError):
            p = p.with_suffix(".csv.gz")
            df.to_csv(p, index=index, compression="gzip")
            return p
    df.to_csv(p, index=index)
    return p


def save_square(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    out.index.name = "feature_uid"
    return save_table(out.reset_index(), path, index=False)


def load_square(path: Union[str, Path]) -> pd.DataFrame:
    d = read_table(path)
    if "feature_uid" not in d.columns:
        raise ValueError(f"Square matrix missing feature_uid column: {path}")
    return d.set_index("feature_uid")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def import_module_from_path(name: str, path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["output_root"])


def stage2a5_root(cfg: Mapping) -> Path:
    return Path(cfg["stage2a5_output_root"])


def stable_feature_uid(panel: str, root: str, group: str, feature: str) -> str:
    key = f"{panel}\x1f{root}\x1f{group}\x1f{feature}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"S2B1_{panel}_{h}"


def safe_slug(x: object) -> str:
    s = str(x)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "NA"


def first_nonmissing(series: pd.Series):
    for x in series:
        if pd.notna(x) and str(x) != "":
            return x
    return np.nan


def resolve_array_id(arg: Optional[int]) -> int:
    if arg is not None:
        return int(arg)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise RuntimeError("Need --array-id or SLURM_ARRAY_TASK_ID")
    return int(env)


def validate_config(cfg: Mapping) -> None:
    required = [
        "stage2a5_output_root", "output_root", "stage1_script_path",
        "harmonized_path", "discovery_cohorts", "panels", "sample_type",
        "patient_subset", "agg",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    for p in [
        stage2a5_root(cfg) / "all_context_root_final_candidates.parquet",
        stage2a5_root(cfg) / "stage2a5_root_matrix_manifest.csv",
        Path(cfg["stage1_script_path"]),
        Path(cfg["harmonized_path"]),
    ]:
        if not p.exists() and not p.with_suffix(".csv.gz").exists() and not p.with_suffix(".csv").exists():
            raise FileNotFoundError(str(p))


def load_final_candidates(cfg: Mapping) -> pd.DataFrame:
    p = stage2a5_root(cfg) / "all_context_root_final_candidates.parquet"
    d = read_table(p)
    if d.empty:
        raise RuntimeError("Stage2A5 final candidate table is empty")
    required = ["cohort", "panel", ROOT_COL, "feature_group", "feature"]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise KeyError(f"Stage2A5 final candidate table missing columns: {miss}")
    for c in ["sample_type", "patient_subset", "agg"]:
        if c not in d.columns:
            d[c] = cfg[c]
    return d


def filter_discovery_candidates(d: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    x = d.copy()
    x = x[x["cohort"].astype(str).isin([str(v) for v in cfg["discovery_cohorts"]])]
    x = x[x["panel"].astype(str).isin([str(v) for v in cfg["panels"]])]
    x = x[x["sample_type"].astype(str).eq(str(cfg["sample_type"]))]
    x = x[x["patient_subset"].astype(str).eq(str(cfg["patient_subset"]))]
    x = x[x["agg"].astype(str).eq(str(cfg["agg"]))]
    if x.empty:
        raise RuntimeError("No Stage2A5 candidates remain after discovery filters")
    return x


def command_validate(cfg: Mapping) -> None:
    validate_config(cfg)
    d = filter_discovery_candidates(load_final_candidates(cfg), cfg)
    roots = d[["panel", ROOT_COL]].drop_duplicates().sort_values(["panel", ROOT_COL])
    log(f"[VALID] candidate_rows={len(d)} panel_roots={len(roots)}")
    log(roots.to_string(index=False))


def build_root_universe(g: pd.DataFrame, panel: str, root: str) -> pd.DataFrame:
    work = g.copy()
    work["feature_group"] = work["feature_group"].astype(str)
    work["feature"] = work["feature"].astype(str)

    key_cols = ["feature_group", "feature"]
    meta_preferred = [
        "parser_status", "parser_warnings", "parsed_feature_type",
        "parsed_feature_subtype", "parsed_entities_json", "parsed_metric_params_json",
        "parsed_compartment", "parsed_summary_stat", "summary_class",
        "root_safe_summary_key", "exact_semantic_key",
    ]

    rows: List[dict] = []
    for (fg, feat), z in work.groupby(key_cols, dropna=False, sort=False):
        row = {
            "panel": panel,
            ROOT_COL: root,
            "feature_group": str(fg),
            "feature": str(feat),
            "stage2b_feature_uid": stable_feature_uid(panel, root, str(fg), str(feat)),
            "n_nominating_contexts": int(z["context_slug"].nunique()) if "context_slug" in z.columns else int(len(z)),
            "n_nominating_cohorts": int(z["cohort"].astype(str).nunique()),
            "n_nominating_endpoints": int(z["endpoint"].astype(str).nunique()) if "endpoint" in z.columns else np.nan,
            "n_stage2a5_rows": int(len(z)),
            "n_stage2a5_unique_uids": int(z["feature_uid"].astype(str).nunique()) if "feature_uid" in z.columns else np.nan,
            "stage2a5_feature_uids": ";".join(sorted(set(z["feature_uid"].dropna().astype(str)))) if "feature_uid" in z.columns else "",
            "nominated_cohorts": ";".join(sorted(set(z["cohort"].astype(str)))),
            "nominated_endpoints": ";".join(sorted(set(z["endpoint"].astype(str)))) if "endpoint" in z.columns else "",
            "best_stage2a5_root_rank": float(safe_numeric(z["stage2a5_root_rank"]).min()) if "stage2a5_root_rank" in z.columns else np.nan,
            "best_stage2a4_root_rank": float(safe_numeric(z["stage2a4_root_rank"]).min()) if "stage2a4_root_rank" in z.columns else np.nan,
            "median_oof_metric": float(safe_numeric(z["oof_metric"]).median()) if "oof_metric" in z.columns else np.nan,
            "median_fold_sd": float(safe_numeric(z["fold_sd"]).median()) if "fold_sd" in z.columns else np.nan,
        }
        for c in meta_preferred:
            if c in z.columns:
                row[c] = first_nonmissing(z[c])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["n_nominating_cohorts", "n_nominating_contexts", "best_stage2a5_root_rank", "feature_group", "feature"],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    out["stage2b_universe_rank"] = np.arange(1, len(out) + 1)
    return out


def command_setup(cfg: Mapping) -> None:
    validate_config(cfg)
    out = ensure_dir(output_root(cfg))
    write_json(cfg, out / "stage2b1_config.resolved.json")

    log("[SETUP] Reading Stage2A5 aggregate candidate table ONCE...")
    cand = filter_discovery_candidates(load_final_candidates(cfg), cfg)
    # Save a small filtered snapshot for provenance; setup is the only stage reading the large aggregate parquet.
    small_cols = [c for c in cand.columns if c not in {"represented_seed_feature_uids"}]
    save_table(cand[small_cols], out / "stage2a5_discovery_candidates_snapshot.parquet")

    universe_root = ensure_dir(out / "root_universes")
    universe_rows: List[dict] = []
    matrix_rows: List[dict] = []
    consensus_rows: List[dict] = []

    aid_matrix = 0
    aid_cons = 0
    for (panel, root), g in cand.groupby(["panel", ROOT_COL], dropna=False, sort=True):
        panel = str(panel)
        root = str(root)
        universe = build_root_universe(g, panel, root)
        pdir = ensure_dir(universe_root / safe_slug(panel))
        upath = pdir / f"{safe_slug(root)}__feature_universe.csv"
        universe.to_csv(upath, index=False)

        universe_rows.append({
            "panel": panel,
            ROOT_COL: root,
            "n_universe_features": int(len(universe)),
            "n_features_nominated_ge2_cohorts": int((universe["n_nominating_cohorts"] >= 2).sum()),
            "n_features_nominated_ge3_cohorts": int((universe["n_nominating_cohorts"] >= 3).sum()),
            "feature_universe_path": str(upath),
        })

        for cohort in cfg["discovery_cohorts"]:
            aid_matrix += 1
            matrix_rows.append({
                "array_id": aid_matrix,
                "cohort": str(cohort),
                "panel": panel,
                ROOT_COL: root,
                "sample_type": str(cfg["sample_type"]),
                "patient_subset": str(cfg["patient_subset"]),
                "agg": str(cfg["agg"]),
                "n_universe_features": int(len(universe)),
                "feature_universe_path": str(upath),
                "matrix_slug": "__".join(map(safe_slug, [cohort, panel, root])),
            })

        aid_cons += 1
        consensus_rows.append({
            "array_id": aid_cons,
            "panel": panel,
            ROOT_COL: root,
            "n_universe_features": int(len(universe)),
            "feature_universe_path": str(upath),
            "consensus_slug": "__".join(map(safe_slug, [panel, root])),
        })

    uni = pd.DataFrame(universe_rows)
    midx = pd.DataFrame(matrix_rows)
    cidx = pd.DataFrame(consensus_rows)
    uni.to_csv(out / "stage2b1_root_universe_index.csv", index=False)
    midx.to_csv(out / "stage2b1_matrix_worker_index.csv", index=False)
    cidx.to_csv(out / "stage2b1_consensus_worker_index.csv", index=False)

    summary = {
        "n_discovery_candidate_rows": int(len(cand)),
        "n_panel_roots": int(len(uni)),
        "n_matrix_workers": int(len(midx)),
        "n_consensus_workers": int(len(cidx)),
        "large_stage2a5_parquet_read_policy": "setup_only",
        "discovery_cohorts": list(map(str, cfg["discovery_cohorts"])),
        "sample_type": str(cfg["sample_type"]),
        "patient_subset": str(cfg["patient_subset"]),
        "agg": str(cfg["agg"]),
    }
    write_json(summary, out / "stage2b1_setup_summary.json")
    log(f"[DONE SETUP] panel_roots={len(uni)} matrix_workers={len(midx)} consensus_workers={len(cidx)}")


def load_index_row(path: Path, array_id: int) -> pd.Series:
    d = pd.read_csv(path)
    hit = d[pd.to_numeric(d["array_id"], errors="coerce").eq(int(array_id))]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one row for array_id={array_id} in {path}; found {len(hit)}")
    return hit.iloc[0]


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
    """Tested Stage1-v6 loading/aggregation pattern used in prior Stage2 code."""
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
        raise ValueError("No cores remain after requested filters")
    core_df = stage1_mod.merge_harmonized_to_core_df(core_df, harm_df)
    core_df = stage1_mod.replace_with_harmonized_columns(core_df)
    core_df = stage1_mod.simplify_clinical_vars(core_df)
    core_df = stage1_mod.ensure_patient_id_column(core_df)

    present_features = [f for f in features if f in core_df.columns]
    if not present_features:
        raise ValueError("None of the requested features were found in core_df")

    patient_df = stage1_mod.aggregate_core_to_patient(core_df, feature_cols=present_features, agg=agg)
    if "cohort" in patient_df.columns:
        patient_df = patient_df[patient_df["cohort"].astype(str).eq(str(cohort))].copy()

    if cohort in {"No-NAC", "KOLL"} and patient_subset in {"no_adj_chemo", "adj_chemo"}:
        patient_df = stage1_mod.apply_patient_subset(patient_df, patient_subset=patient_subset)
    elif patient_subset != "all":
        log(f"[WARN] patient_subset={patient_subset} requested for {cohort}; keeping all because Stage1 subset is not defined")

    if patient_df.empty:
        raise ValueError("No patients remain after aggregation/subsetting")
    keep = [c for c in patient_df.columns if c not in present_features]
    return patient_df[keep + present_features].copy()


def matrix_worker_dir(cfg: Mapping, row: pd.Series) -> Path:
    return output_root(cfg) / "cohort_root_matrices" / str(row["matrix_slug"])


def command_matrix_worker(cfg: Mapping, array_id: int) -> None:
    out = output_root(cfg)
    idx_path = out / "stage2b1_matrix_worker_index.csv"
    row = load_index_row(idx_path, array_id)
    wdir = ensure_dir(matrix_worker_dir(cfg, row))

    cohort = str(row["cohort"])
    panel = str(row["panel"])
    root = str(row[ROOT_COL])
    universe = pd.read_csv(str(row["feature_universe_path"]))
    if universe.empty:
        raise RuntimeError("Feature universe is empty")

    stage1_mod = import_module_from_path(
        f"stage1_v6_s2b1_{array_id}", cfg["stage1_script_path"]
    )

    merged: Optional[pd.DataFrame] = None
    feature_meta_rows: List[dict] = []
    failure_rows: List[dict] = []

    log(f"[MATRIX] array={array_id} cohort={cohort} panel={panel} root={root} features={len(universe)}")

    for feature_group, g in universe.groupby("feature_group", dropna=False, sort=True):
        fg = str(feature_group)
        features = g["feature"].astype(str).drop_duplicates().tolist()
        try:
            pdf = build_patient_matrix_for_source_group(
                stage1_mod=stage1_mod,
                cohort=cohort,
                panel=panel,
                feature_source=root,
                feature_group=fg,
                features=features,
                sample_type=str(row["sample_type"]),
                patient_subset=str(row["patient_subset"]),
                agg=str(row["agg"]),
                qc_acceptability=str(cfg.get("qc_acceptability", "acceptable_or_borderline")),
                min_epi_fraction=cfg.get("min_epi_fraction", 0.05),
                harmonized_path=cfg["harmonized_path"],
                spatial_root=cfg.get("spatial_root"),
                cell_features_path=cfg.get("cell_features_path"),
                triads_path=cfg.get("triads_path"),
                koll_metadata_csv=cfg.get("koll_metadata_csv"),
            )
        except Exception as exc:
            for _, rr in g.iterrows():
                failure_rows.append({
                    "array_id": array_id, "cohort": cohort, "panel": panel,
                    ROOT_COL: root, "feature_group": fg,
                    "stage2b_feature_uid": rr["stage2b_feature_uid"],
                    "feature": rr["feature"],
                    "reason": f"{type(exc).__name__}: {exc}",
                })
            log(f"[WARN] source/group failed {root}/{fg}: {type(exc).__name__}: {exc}")
            continue

        tmp = pdf[["patient_id"]].copy()
        tmp["patient_id"] = tmp["patient_id"].astype(str)
        for _, rr in g.iterrows():
            feat = str(rr["feature"])
            uid = str(rr["stage2b_feature_uid"])
            if feat not in pdf.columns:
                failure_rows.append({
                    "array_id": array_id, "cohort": cohort, "panel": panel,
                    ROOT_COL: root, "feature_group": fg,
                    "stage2b_feature_uid": uid, "feature": feat,
                    "reason": "feature_missing_from_patient_matrix",
                })
                continue
            values = safe_numeric(pdf[feat])
            tmp[uid] = values.to_numpy()
            feature_meta_rows.append({
                "array_id": array_id, "cohort": cohort, "panel": panel,
                ROOT_COL: root, "feature_group": fg,
                "stage2b_feature_uid": uid, "feature": feat,
                "n_patients_total": int(len(values)),
                "n_nonmissing": int(values.notna().sum()),
                "nonmissing_fraction": float(values.notna().mean()) if len(values) else np.nan,
                "n_unique": int(values.dropna().nunique()),
                "measured_successfully": True,
            })

        if merged is None:
            merged = tmp
        else:
            # Avoid duplicate columns when a source/group contributed no valid feature columns.
            add_cols = [c for c in tmp.columns if c == "patient_id" or c not in merged.columns]
            merged = merged.merge(tmp[add_cols], on="patient_id", how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=["patient_id"])

    all_uids = universe["stage2b_feature_uid"].astype(str).tolist()
    for uid in all_uids:
        if uid not in merged.columns:
            merged[uid] = np.nan
    merged = merged[["patient_id"] + all_uids]

    meta = pd.DataFrame(feature_meta_rows)
    if meta.empty:
        meta = universe[["stage2b_feature_uid", "feature_group", "feature"]].copy()
        meta["array_id"] = array_id
        meta["cohort"] = cohort
        meta["panel"] = panel
        meta[ROOT_COL] = root
        meta["n_patients_total"] = len(merged)
        meta["n_nonmissing"] = 0
        meta["nonmissing_fraction"] = 0.0
        meta["n_unique"] = 0
        meta["measured_successfully"] = False
    else:
        meta = universe.merge(meta, on=["stage2b_feature_uid", "feature_group", "feature"], how="left", suffixes=("", "_measured"))
        for c, val in [
            ("array_id", array_id), ("cohort", cohort), ("panel", panel), (ROOT_COL, root),
            ("n_patients_total", len(merged)), ("n_nonmissing", 0),
            ("nonmissing_fraction", 0.0), ("n_unique", 0), ("measured_successfully", False),
        ]:
            if c not in meta.columns:
                meta[c] = val
            else:
                meta[c] = meta[c].fillna(val)

    matrix_path = save_table(merged, wdir / "patient_feature_matrix.parquet")
    meta.to_csv(wdir / "feature_measurement_qc.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(wdir / "matrix_build_failures.csv", index=False)

    n_measured = int(sum(uid in merged.columns and merged[uid].notna().any() for uid in all_uids))
    summary = {
        "array_id": array_id,
        "cohort": cohort,
        "panel": panel,
        ROOT_COL: root,
        "sample_type": str(row["sample_type"]),
        "patient_subset": str(row["patient_subset"]),
        "agg": str(row["agg"]),
        "n_patients": int(len(merged)),
        "n_universe_features": int(len(universe)),
        "n_features_measured": n_measured,
        "n_features_missing": int(len(universe) - n_measured),
        "n_build_failures": int(len(failure_rows)),
        "matrix_path": str(matrix_path),
        "feature_qc_path": str(wdir / "feature_measurement_qc.csv"),
        "failures_path": str(wdir / "matrix_build_failures.csv"),
    }
    pd.DataFrame([summary]).to_csv(wdir / "matrix_worker_summary.csv", index=False)
    (wdir / ".done").write_text("complete\n")
    log(f"[DONE MATRIX] patients={len(merged)} measured={n_measured}/{len(universe)} failures={len(failure_rows)}")


def compute_cohort_correlation(
    matrix: pd.DataFrame,
    feature_uids: Sequence[str],
    min_nonmissing_frac: float,
    min_pairwise_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = [str(f) for f in feature_uids if str(f) in matrix.columns]
    if not feats:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    X = matrix[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_patients = int(len(X))
    qc_rows = []
    keep = []
    for c in feats:
        x = X[c]
        frac = float(x.notna().mean()) if n_patients else 0.0
        nunique = int(x.dropna().nunique())
        eligible = bool(frac >= min_nonmissing_frac and nunique > 1)
        qc_rows.append({
            "stage2b_feature_uid": c,
            "n_patients": n_patients,
            "n_nonmissing": int(x.notna().sum()),
            "nonmissing_fraction": frac,
            "n_unique": nunique,
            "eligible_for_corr": eligible,
        })
        if eligible:
            keep.append(c)
    X = X[keep]
    if not keep:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(qc_rows)

    valid = X.notna().astype(np.int16)
    pair_n = valid.T.dot(valid).astype(int)
    corr = X.corr(method="spearman", min_periods=int(min_pairwise_n))
    corr = corr.where(pair_n >= int(min_pairwise_n))
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return corr, pair_n, pd.DataFrame(qc_rows)


def matrix_summary_paths(cfg: Mapping, panel: str, root: str) -> pd.DataFrame:
    idx = pd.read_csv(output_root(cfg) / "stage2b1_matrix_worker_index.csv")
    return idx[
        idx["panel"].astype(str).eq(str(panel))
        & idx[ROOT_COL].astype(str).eq(str(root))
    ].copy()


def average_consensus(
    cohort_corrs: Mapping[str, pd.DataFrame],
    all_features: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    feats = list(map(str, all_features))
    n = len(feats)
    sum_rho = np.zeros((n, n), dtype=float)
    sum_sq = np.zeros((n, n), dtype=float)
    support = np.zeros((n, n), dtype=np.int16)
    n_pos = np.zeros((n, n), dtype=np.int16)
    n_neg = np.zeros((n, n), dtype=np.int16)
    n_zero = np.zeros((n, n), dtype=np.int16)

    pos = {f: i for i, f in enumerate(feats)}
    for cohort, corr in cohort_corrs.items():
        if corr is None or corr.empty:
            continue
        common = [f for f in corr.index.astype(str) if f in pos and f in corr.columns]
        if not common:
            continue
        ii = np.array([pos[f] for f in common], dtype=int)
        C = corr.loc[common, common].to_numpy(dtype=float)
        mask = np.isfinite(C)
        block_sum = np.where(mask, C, 0.0)
        block_sq = np.where(mask, C * C, 0.0)
        idx = np.ix_(ii, ii)
        sum_rho[idx] += block_sum
        sum_sq[idx] += block_sq
        support[idx] += mask.astype(np.int16)
        n_pos[idx] += (mask & (C > 0)).astype(np.int16)
        n_neg[idx] += (mask & (C < 0)).astype(np.int16)
        n_zero[idx] += (mask & (C == 0)).astype(np.int16)

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = sum_rho / np.where(support > 0, support, np.nan)
        variance = sum_sq / np.where(support > 0, support, np.nan) - consensus ** 2
        variance = np.maximum(variance, 0)
        rho_sd = np.sqrt(variance)

    sign_den = n_pos + n_neg
    sign_consistency = np.maximum(n_pos, n_neg) / np.where(sign_den > 0, sign_den, np.nan)

    for i in range(n):
        consensus[i, i] = 1.0
        rho_sd[i, i] = 0.0
        sign_consistency[i, i] = 1.0

    def df(a):
        return pd.DataFrame(a, index=feats, columns=feats)

    return {
        "consensus": df(consensus),
        "pair_support": df(support.astype(int)),
        "rho_sd": df(rho_sd),
        "n_pos": df(n_pos.astype(int)),
        "n_neg": df(n_neg.astype(int)),
        "n_zero": df(n_zero.astype(int)),
        "sign_consistency": df(sign_consistency),
    }


def apply_support_filter(
    consensus: pd.DataFrame,
    pair_support: pd.DataFrame,
    min_pair_support: int,
    min_feature_support_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = list(consensus.index.astype(str))
    if not feats:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    supported = pair_support >= int(min_pair_support)
    np.fill_diagonal(supported.values, True)
    denom = max(len(feats) - 1, 1)
    rows = []
    for f in feats:
        others = [x for x in feats if x != f]
        vals = supported.loc[f, others] if others else pd.Series(dtype=bool)
        supp_vals = pd.to_numeric(pair_support.loc[f, others], errors="coerce") if others else pd.Series(dtype=float)
        rows.append({
            "stage2b_feature_uid": f,
            "n_supported_pairs": int(vals.sum()) if len(vals) else 0,
            "feature_support_fraction": float(vals.sum() / denom) if len(vals) else 0.0,
            "max_pair_support": int(supp_vals.max()) if len(supp_vals) and pd.notna(supp_vals.max()) else 0,
            "median_pair_support": float(supp_vals.median()) if len(supp_vals) else np.nan,
        })
    fs = pd.DataFrame(rows)
    keep = fs.loc[fs["feature_support_fraction"] >= float(min_feature_support_frac), "stage2b_feature_uid"].astype(str).tolist()
    fs["passes_feature_support_filter"] = fs["stage2b_feature_uid"].isin(keep)

    if not keep:
        return pd.DataFrame(), pd.DataFrame(), fs, pd.DataFrame()
    cons_sub = consensus.loc[keep, keep].copy()
    supp_sub = pair_support.loc[keep, keep].copy()
    mask = supp_sub >= int(min_pair_support)
    cons_masked_nan = cons_sub.where(mask)
    consensus_clustering = cons_sub.where(mask, other=0.0)
    np.fill_diagonal(consensus_clustering.values, 1.0)
    np.fill_diagonal(cons_masked_nan.values, 1.0)
    return consensus_clustering, supp_sub, fs, cons_masked_nan


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "coolwarm",
) -> None:
    if matrix is None or matrix.empty:
        return
    n = matrix.shape[0]
    fig_side = min(max(5.5, n * 0.10), 13)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"Features (n={n})")
    ax.set_ylabel(f"Features (n={n})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_feature_support_plot(feature_support: pd.DataFrame, path: Path, title: str) -> None:
    if feature_support.empty:
        return
    x = pd.to_numeric(feature_support["feature_support_fraction"], errors="coerce").dropna()
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x, bins=min(20, max(5, int(math.sqrt(len(x))))))
    ax.set_xlabel("Fraction of feature pairs supported in >= min_pair_support cohorts")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def consensus_worker_dir(cfg: Mapping, row: pd.Series) -> Path:
    return output_root(cfg) / "root_consensus" / str(row["consensus_slug"])


def command_consensus_worker(cfg: Mapping, array_id: int) -> None:
    out = output_root(cfg)
    row = load_index_row(out / "stage2b1_consensus_worker_index.csv", array_id)
    panel = str(row["panel"])
    root = str(row[ROOT_COL])
    wdir = ensure_dir(consensus_worker_dir(cfg, row))
    plots = ensure_dir(wdir / "plots")
    cohort_dir = ensure_dir(wdir / "cohort_correlations")

    universe = pd.read_csv(str(row["feature_universe_path"]))
    feats = universe["stage2b_feature_uid"].astype(str).tolist()
    mrows = matrix_summary_paths(cfg, panel, root)
    if mrows.empty:
        raise RuntimeError(f"No matrix workers indexed for panel={panel}, root={root}")

    cohort_corrs: Dict[str, pd.DataFrame] = {}
    cohort_pair_n: Dict[str, pd.DataFrame] = {}
    cohort_feature_qc_parts: List[pd.DataFrame] = []
    matrix_summary_parts: List[pd.DataFrame] = []
    missing_matrix_workers: List[dict] = []

    min_nonmissing = float(cfg.get("min_nonmissing_frac_for_corr", 0.20))
    min_pairwise_n = int(cfg.get("min_pairwise_n", 20))

    log(f"[CONSENSUS] array={array_id} panel={panel} root={root} features={len(feats)}")

    for _, mr in mrows.sort_values("cohort").iterrows():
        cohort = str(mr["cohort"])
        mdir = matrix_worker_dir(cfg, mr)
        done = mdir / ".done"
        sp = mdir / "matrix_worker_summary.csv"
        if not done.exists() or not sp.exists():
            missing_matrix_workers.append({"cohort": cohort, "panel": panel, ROOT_COL: root, "reason": "matrix_worker_incomplete"})
            continue
        summ = pd.read_csv(sp)
        matrix_summary_parts.append(summ)
        matrix_path = Path(str(summ.iloc[0]["matrix_path"]))
        matrix = read_table(matrix_path)
        corr, pair_n, fqc = compute_cohort_correlation(
            matrix=matrix,
            feature_uids=feats,
            min_nonmissing_frac=min_nonmissing,
            min_pairwise_n=min_pairwise_n,
        )
        cohort_corrs[cohort] = corr
        cohort_pair_n[cohort] = pair_n
        if not fqc.empty:
            fqc["cohort"] = cohort
            fqc["panel"] = panel
            fqc[ROOT_COL] = root
            cohort_feature_qc_parts.append(fqc)
        save_square(corr, cohort_dir / f"{safe_slug(cohort)}__spearman.parquet")
        save_square(pair_n, cohort_dir / f"{safe_slug(cohort)}__pairwise_n.parquet")
        log(f"[COHORT] {cohort}: patients={len(matrix)} corr_features={corr.shape[0]}")

    if missing_matrix_workers:
        pd.DataFrame(missing_matrix_workers).to_csv(wdir / "missing_matrix_workers.csv", index=False)
        raise RuntimeError(f"{len(missing_matrix_workers)} matrix workers missing for {panel}/{root}")

    res = average_consensus(cohort_corrs, feats)
    consensus = res["consensus"]
    pair_support = res["pair_support"]
    rho_sd = res["rho_sd"]
    sign_consistency = res["sign_consistency"]

    clustering_consensus, support_filtered, feature_support, masked_nan = apply_support_filter(
        consensus,
        pair_support,
        min_pair_support=int(cfg.get("min_pair_support", 2)),
        min_feature_support_frac=float(cfg.get("min_feature_support_frac", 0.10)),
    )

    # Add measurement/nominating support annotations to feature support.
    cohort_feature_qc = pd.concat(cohort_feature_qc_parts, ignore_index=True, sort=False) if cohort_feature_qc_parts else pd.DataFrame()
    if not cohort_feature_qc.empty:
        elig = (
            cohort_feature_qc.groupby("stage2b_feature_uid", dropna=False)
            .agg(
                n_cohorts_measured=("cohort", lambda s: int(s.nunique())),
                n_cohorts_corr_eligible=("eligible_for_corr", "sum"),
                median_nonmissing_fraction=("nonmissing_fraction", "median"),
                min_nonmissing_fraction=("nonmissing_fraction", "min"),
            )
            .reset_index()
        )
        feature_support = feature_support.merge(elig, on="stage2b_feature_uid", how="left")
    add_cols = [
        "stage2b_feature_uid", "feature", "feature_group", "n_nominating_contexts",
        "n_nominating_cohorts", "nominated_cohorts", "nominated_endpoints",
        "best_stage2a5_root_rank", "median_oof_metric", "median_fold_sd",
        "parsed_feature_type", "parsed_feature_subtype", "parsed_entities_json",
        "parsed_compartment", "parsed_summary_stat", "root_safe_summary_key", "exact_semantic_key",
    ]
    add_cols = [c for c in add_cols if c in universe.columns]
    feature_support = feature_support.merge(universe[add_cols], on="stage2b_feature_uid", how="left")

    # Pair-level long audit including cohort-specific rho values.
    pair_rows = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i], feats[j]
            row_pair = {
                "feature_uid_1": a,
                "feature_uid_2": b,
                "consensus_rho": consensus.loc[a, b],
                "pair_support": int(pair_support.loc[a, b]),
                "rho_sd_across_cohorts": rho_sd.loc[a, b],
                "sign_consistency": sign_consistency.loc[a, b],
                "n_positive_cohorts": int(res["n_pos"].loc[a, b]),
                "n_negative_cohorts": int(res["n_neg"].loc[a, b]),
                "n_zero_cohorts": int(res["n_zero"].loc[a, b]),
            }
            for cohort in cfg["discovery_cohorts"]:
                c = str(cohort)
                cr = cohort_corrs.get(c, pd.DataFrame())
                pn = cohort_pair_n.get(c, pd.DataFrame())
                row_pair[f"rho_{c}"] = cr.loc[a, b] if (not cr.empty and a in cr.index and b in cr.columns) else np.nan
                row_pair[f"pair_n_{c}"] = int(pn.loc[a, b]) if (not pn.empty and a in pn.index and b in pn.columns and pd.notna(pn.loc[a, b])) else np.nan
            pair_rows.append(row_pair)
    pair_long = pd.DataFrame(pair_rows)

    save_square(consensus, wdir / "consensus_signed_spearman_unfiltered.parquet")
    save_square(pair_support, wdir / "pair_support_unfiltered.parquet")
    save_square(rho_sd, wdir / "consensus_rho_sd_across_cohorts.parquet")
    save_square(sign_consistency, wdir / "sign_consistency_unfiltered.parquet")
    save_square(clustering_consensus, wdir / "consensus_signed_spearman_support_filtered_clustering.parquet")
    save_square(masked_nan, wdir / "consensus_signed_spearman_support_filtered_nan.parquet")
    save_square(support_filtered, wdir / "pair_support_support_filtered.parquet")
    feature_support.to_csv(wdir / "feature_support_summary.csv", index=False)
    cohort_feature_qc.to_csv(wdir / "cohort_feature_correlation_qc.csv", index=False)
    pd.concat(matrix_summary_parts, ignore_index=True, sort=False).to_csv(wdir / "cohort_matrix_summary.csv", index=False)
    pair_long.to_csv(wdir / "pairwise_consensus_audit.csv.gz", index=False, compression="gzip")
    universe.to_csv(wdir / "feature_universe.csv", index=False)

    save_heatmap(consensus, plots / "01_consensus_unfiltered.png", f"{panel} | {root} | signed Spearman consensus", -1, 1)
    save_heatmap(pair_support, plots / "02_pair_support.png", f"{panel} | {root} | pair support across cohorts", 0, len(cfg["discovery_cohorts"]), cmap="viridis")
    save_heatmap(sign_consistency, plots / "03_sign_consistency.png", f"{panel} | {root} | sign consistency", 0, 1, cmap="viridis")
    save_heatmap(clustering_consensus, plots / "04_consensus_support_filtered.png", f"{panel} | {root} | support-filtered consensus", -1, 1)
    save_feature_support_plot(feature_support, plots / "05_feature_support_fraction.png", f"{panel} | {root} | feature support")

    # Root-level summary.
    n_features = len(feats)
    upper = np.triu_indices(n_features, k=1) if n_features >= 2 else (np.array([], dtype=int), np.array([], dtype=int))
    supp_vals = pair_support.to_numpy(dtype=float)[upper] if n_features >= 2 else np.array([])
    cons_vals = consensus.to_numpy(dtype=float)[upper] if n_features >= 2 else np.array([])
    sign_vals = sign_consistency.to_numpy(dtype=float)[upper] if n_features >= 2 else np.array([])
    sd_vals = rho_sd.to_numpy(dtype=float)[upper] if n_features >= 2 else np.array([])
    supported_mask = supp_vals >= int(cfg.get("min_pair_support", 2)) if len(supp_vals) else np.array([], dtype=bool)

    def med(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if len(arr) else np.nan

    supported_sign = sign_vals[supported_mask] if len(sign_vals) else np.array([])
    supported_cons = cons_vals[supported_mask] if len(cons_vals) else np.array([])
    supported_sd = sd_vals[supported_mask] if len(sd_vals) else np.array([])

    summary = {
        "array_id": array_id,
        "panel": panel,
        ROOT_COL: root,
        "n_discovery_cohorts": int(len(cfg["discovery_cohorts"])),
        "n_features_universe": int(n_features),
        "n_features_support_filtered": int(clustering_consensus.shape[0]),
        "feature_retention_fraction": float(clustering_consensus.shape[0] / n_features) if n_features else np.nan,
        "n_pairs_total": int(len(supp_vals)),
        "n_pairs_support_ge_min": int(supported_mask.sum()) if len(supported_mask) else 0,
        "pair_support_ge_min_fraction": float(supported_mask.mean()) if len(supported_mask) else np.nan,
        "median_pair_support": med(supp_vals),
        "median_abs_consensus_rho_supported": med(np.abs(supported_cons)),
        "median_sign_consistency_supported": med(supported_sign),
        "fraction_supported_pairs_same_sign": float(np.mean(supported_sign >= 1.0 - 1e-12)) if np.isfinite(supported_sign).any() else np.nan,
        "median_rho_sd_across_cohorts_supported": med(supported_sd),
        "min_pair_support": int(cfg.get("min_pair_support", 2)),
        "min_feature_support_frac": float(cfg.get("min_feature_support_frac", 0.10)),
        "min_nonmissing_frac_for_corr": min_nonmissing,
        "min_pairwise_n": min_pairwise_n,
        "consensus_path": str(wdir / "consensus_signed_spearman_unfiltered.parquet"),
        "consensus_support_filtered_path": str(wdir / "consensus_signed_spearman_support_filtered_clustering.parquet"),
        "pair_support_path": str(wdir / "pair_support_unfiltered.parquet"),
        "sign_consistency_path": str(wdir / "sign_consistency_unfiltered.parquet"),
        "feature_support_path": str(wdir / "feature_support_summary.csv"),
        "pairwise_audit_path": str(wdir / "pairwise_consensus_audit.csv.gz"),
        "feature_universe_path": str(row["feature_universe_path"]),
        "root_output_dir": str(wdir),
    }
    pd.DataFrame([summary]).to_csv(wdir / "root_consensus_summary.csv", index=False)
    (wdir / ".done").write_text("complete\n")
    log(f"[DONE CONSENSUS] {panel}/{root}: universe={n_features} support_filtered={clustering_consensus.shape[0]} supported_pairs={summary['n_pairs_support_ge_min']}")


def command_aggregate(cfg: Mapping) -> None:
    out = output_root(cfg)
    midx = pd.read_csv(out / "stage2b1_matrix_worker_index.csv")
    cidx = pd.read_csv(out / "stage2b1_consensus_worker_index.csv")

    matrix_summaries = []
    missing_matrix = []
    for _, row in midx.iterrows():
        wdir = matrix_worker_dir(cfg, row)
        if not (wdir / ".done").exists():
            missing_matrix.append({"array_id": int(row["array_id"]), "matrix_slug": row["matrix_slug"], "reason": "missing_done"})
            continue
        p = wdir / "matrix_worker_summary.csv"
        if p.exists():
            matrix_summaries.append(pd.read_csv(p))

    root_summaries = []
    missing_cons = []
    manifest_rows = []
    feature_support_parts = []
    for _, row in cidx.iterrows():
        wdir = consensus_worker_dir(cfg, row)
        if not (wdir / ".done").exists():
            missing_cons.append({"array_id": int(row["array_id"]), "consensus_slug": row["consensus_slug"], "reason": "missing_done"})
            continue
        p = wdir / "root_consensus_summary.csv"
        if p.exists():
            s = pd.read_csv(p)
            root_summaries.append(s)
            manifest_rows.append(s.iloc[0].to_dict())
        fp = wdir / "feature_support_summary.csv"
        if fp.exists():
            f = pd.read_csv(fp)
            f["panel"] = row["panel"]
            f[ROOT_COL] = row[ROOT_COL]
            feature_support_parts.append(f)

    ms = pd.concat(matrix_summaries, ignore_index=True, sort=False) if matrix_summaries else pd.DataFrame()
    rs = pd.concat(root_summaries, ignore_index=True, sort=False) if root_summaries else pd.DataFrame()
    man = pd.DataFrame(manifest_rows)
    fs = pd.concat(feature_support_parts, ignore_index=True, sort=False) if feature_support_parts else pd.DataFrame()

    ms.to_csv(out / "all_cohort_root_matrix_summary.csv", index=False)
    rs.to_csv(out / "all_panel_root_consensus_summary.csv", index=False)
    man.to_csv(out / "stage2b1_root_consensus_manifest.csv", index=False)
    fs.to_csv(out / "all_panel_root_feature_support_summary.csv", index=False)
    pd.DataFrame(missing_matrix).to_csv(out / "stage2b1_missing_matrix_workers.csv", index=False)
    pd.DataFrame(missing_cons).to_csv(out / "stage2b1_missing_consensus_workers.csv", index=False)

    if not ms.empty:
        msum = (
            ms.groupby(["panel", ROOT_COL], dropna=False)
            .agg(
                n_cohort_matrices=("cohort", "nunique"),
                median_patients=("n_patients", "median"),
                min_patients=("n_patients", "min"),
                n_universe_features=("n_universe_features", "max"),
                median_features_measured=("n_features_measured", "median"),
                min_features_measured=("n_features_measured", "min"),
                total_build_failures=("n_build_failures", "sum"),
            )
            .reset_index()
        )
    else:
        msum = pd.DataFrame()

    review = rs.copy()
    if not msum.empty and not review.empty:
        review = review.merge(msum, on=["panel", ROOT_COL], how="left", suffixes=("", "_matrix"))
    review.to_csv(out / "stage2b1_panel_root_review_summary.csv", index=False)

    lines = []
    lines.append("STAGE 2B-1 ROOT CONSENSUS SUMMARY")
    lines.append("=" * 88)
    lines.append("Endpoint-independent root feature universes; equal-weighted discovery cohorts.")
    lines.append(f"Discovery cohorts: {', '.join(map(str, cfg['discovery_cohorts']))}")
    lines.append(f"Spearman pairwise N >= {int(cfg.get('min_pairwise_n', 20))}; pair support >= {int(cfg.get('min_pair_support', 2))} cohorts")
    lines.append("")
    if not review.empty:
        for _, r in review.sort_values(["panel", ROOT_COL]).iterrows():
            lines.append(
                f"{r['panel']} | {r[ROOT_COL]}: universe={int(r['n_features_universe'])}, "
                f"support-filtered={int(r['n_features_support_filtered'])}, "
                f"supported-pair-frac={r['pair_support_ge_min_fraction']:.3f}, "
                f"median sign-consistency={r['median_sign_consistency_supported']:.3f}, "
                f"median |consensus rho|={r['median_abs_consensus_rho_supported']:.3f}"
            )
    (out / "stage2b1_summary.txt").write_text("\n".join(lines) + "\n")

    write_json({
        "n_expected_matrix_workers": int(len(midx)),
        "n_completed_matrix_workers": int(len(ms)),
        "n_missing_matrix_workers": int(len(missing_matrix)),
        "n_expected_consensus_workers": int(len(cidx)),
        "n_completed_consensus_workers": int(len(rs)),
        "n_missing_consensus_workers": int(len(missing_cons)),
        "large_stage2a5_parquet_read_policy": "setup_only",
        "consensus_level": "equal_weighted_cohort",
        "clustering_performed": False,
        "k_selection_performed": False,
    }, out / "stage2b1_aggregate_summary.json")

    if missing_matrix or missing_cons:
        raise RuntimeError(
            f"Missing workers: matrix={len(missing_matrix)} consensus={len(missing_cons)}; "
            "see stage2b1_missing_*_workers.csv"
        )
    log(f"[DONE AGGREGATE] matrix_workers={len(ms)} consensus_roots={len(rs)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ["validate", "setup", "matrix-worker", "consensus-worker", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name in {"matrix-worker", "consensus-worker"}:
            p.add_argument("--array-id", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "setup":
        command_setup(cfg)
    elif args.command == "matrix-worker":
        command_matrix_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "consensus-worker":
        command_consensus_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
