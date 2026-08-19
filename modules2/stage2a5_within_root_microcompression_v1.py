#!/usr/bin/env python3
"""
stage2a5_within_root_microcompression_v1.py

Finalized root-aware Stage 2A-5 for the mIF global-module pipeline.

Purpose
-------
Consume the fixed-cap, root-specific Stage 2A-4 candidate registries and
patient-feature matrices and perform only conservative WITHIN-PREP-ROOT
redundancy compression.

What this script DOES
---------------------
For each context and each prep root independently:
  1. retain only Stage 2A-4 candidates successfully present in the root matrix;
  2. annotate candidates with a grammar-aware semantic parse that preserves
     checkpoint/state identity and measurement compartment;
  3. collapse exact patient-vector duplicates only within the same semantic
     measurement;
  4. optionally collapse Mean <-> Median variants of the same fully state-aware
     measurement when they are highly positively correlated and the retained
     higher-ranked candidate is within a small OOF tolerance;
  5. collapse residual duplicate representations of the exact same semantic
     measurement only at a very high positive correlation threshold;
  6. emit before/after matrices, complete decision audits, pairwise correlation
     diagnostics, and root-level manifests for Stage 2B.

What this script explicitly DOES NOT do
---------------------------------------
  * no cross-root comparison, rescue, replacement, or compression;
  * no state stripping / parent-phenotype rescue;
  * no tissue-compartment simplification;
  * no generic high-correlation pruning across biologically distinct features;
  * no addition of rescue-only variables outside the frozen Stage 2A-4 seeds.

Moderately/highly correlated but biologically distinct features are retained on
purpose: their correlation is signal for downstream root-module discovery.

Commands
--------
validate   Validate Stage 2A-4 aggregate inputs and parser availability.
inventory  Build one worker row per endpoint context (one CPU per context).
worker     Process all prep roots for one context.
aggregate  Aggregate context/root outputs and Stage 2B-ready manifests.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_COLS = [
    "cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"
]
ROOT_COL = "feature_source"


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


def save_table(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return p
        except (ImportError, ModuleNotFoundError):
            p = p.with_suffix(".csv.gz")
            df.to_csv(p, index=False, compression="gzip")
            return p
    df.to_csv(p, index=False)
    return p


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def numeric_value(row: Union[pd.Series, Mapping], col: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(col, default))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def import_module_from_path(name: str, path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError("Could not import {}".format(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage2a4_root(cfg: Mapping) -> Path:
    return Path(cfg["stage2a4_output_root"])


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["output_root"])


def root_manifest_path(cfg: Mapping) -> Path:
    return stage2a4_root(cfg) / "stage2a4_root_matrix_manifest.csv"


def get_parser(cfg: Mapping):
    return import_module_from_path(
        "stage2_feature_parser_for_root_microcompression",
        cfg["feature_parser_path"],
    )


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def semantic_enrich(registry: pd.DataFrame, parser) -> pd.DataFrame:
    """
    Add only the semantic keys needed for safe within-root compression.

    Crucially, root_safe_summary_key retains FULL entity state identities and
    measurement compartment; it excludes only the Mean/Median summary label.
    """
    rows: List[dict] = []
    for _, row in registry.iterrows():
        feature = str(row.get("feature", ""))
        root = None if pd.isna(row.get(ROOT_COL)) else str(row.get(ROOT_COL))
        group = None if pd.isna(row.get("feature_group")) else str(row.get("feature_group"))
        parsed = parser.parse_feature(
            feature=feature,
            feature_source=root,
            feature_group=group,
        )
        entities = parsed.get("entities", []) or []
        params = parsed.get("metric_params", {}) or {}
        summary = parsed.get("summary_stat")
        summary_norm = str(summary).strip().lower() if summary is not None else ""
        summary_class = "location" if summary_norm in {"mean", "median"} else (
            "none" if summary_norm == "" else "other"
        )

        full_entities = [
            {
                "role": e.get("role"),
                "cell": e.get("cell"),
                "state": e.get("state"),
            }
            for e in entities
        ]
        base = {
            "feature_type": parsed.get("feature_type"),
            "feature_subtype": parsed.get("feature_subtype"),
            "entities": full_entities,
            "compartment": parsed.get("compartment"),
            "metric_params": params,
        }
        exact = dict(base)
        exact["summary_stat"] = summary_norm

        rows.append({
            "parser_status": parsed.get("parse_status", "unknown"),
            "parser_warnings": ";".join(map(str, parsed.get("warnings", []) or [])),
            "parsed_feature_type": parsed.get("feature_type"),
            "parsed_feature_subtype": parsed.get("feature_subtype"),
            "parsed_entities_json": stable_json(full_entities),
            "parsed_metric_params_json": stable_json(params),
            "parsed_compartment": parsed.get("compartment") if parsed.get("compartment") is not None else "",
            "parsed_summary_stat": summary if summary is not None else "",
            "summary_class": summary_class,
            "root_safe_summary_key": stable_json(base),
            "exact_semantic_key": stable_json(exact),
        })

    parsed_df = pd.DataFrame(rows, index=registry.index)
    out = registry.copy()
    for c in parsed_df.columns:
        if c in out.columns:
            out = out.drop(columns=[c])
    return pd.concat([out, parsed_df], axis=1)


def representative_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Evidence-first ordering inherited from the frozen Stage 2A-4 root rank."""
    out = df.copy().reset_index(drop=True)
    sort_cols: List[str] = []
    ascending: List[bool] = []
    for col, asc in [
        ("stage2a4_root_rank", True),
        ("eligible_root_rank", True),
        ("root_candidate_evidence_score", False),
        ("oof_metric", False),
        ("fold_sd", True),
        ("nonmissing_fraction", False),
        ("feature_uid", True),
    ]:
        if col in out.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return out.sort_values("feature_uid")
    return out.sort_values(sort_cols, ascending=ascending, na_position="last")


def vector_hash(series: pd.Series, decimals: int = 12) -> str:
    x = safe_numeric(series)
    tokens = ["NA" if pd.isna(v) else ("{:.{}f}".format(float(v), decimals)) for v in x]
    return hashlib.sha1("|".join(tokens).encode("utf-8")).hexdigest()


def exact_vector_duplicate(a: pd.Series, b: pd.Series, atol: float) -> Tuple[bool, int]:
    x = safe_numeric(a)
    y = safe_numeric(b)
    # Exact duplicate requires same missingness pattern.
    if not x.isna().equals(y.isna()):
        return False, int((x.notna() & y.notna()).sum())
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n == 0:
        return False, 0
    return bool(np.allclose(x.loc[mask].to_numpy(), y.loc[mask].to_numpy(), rtol=0.0, atol=atol)), n


def safe_spearman(a: pd.Series, b: pd.Series, min_n: int) -> Tuple[float, int]:
    x = safe_numeric(a)
    y = safe_numeric(b)
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < int(min_n):
        return np.nan, n
    xx = x.loc[mask]
    yy = y.loc[mask]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan, n
    return float(xx.corr(yy, method="spearman")), n


def resolve_rep(uid: str, replacement: Dict[str, str]) -> str:
    seen: Set[str] = set()
    current = str(uid)
    while current in replacement and replacement[current] != current:
        if current in seen:
            raise RuntimeError("Replacement cycle involving {}".format(uid))
        seen.add(current)
        current = replacement[current]
    return current


def exact_duplicate_pass(
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    active: Set[str],
    replacement: Dict[str, str],
    atol: float,
) -> Tuple[List[dict], List[dict]]:
    audit: List[dict] = []
    pairs: List[dict] = []
    meta = registry.set_index("feature_uid", drop=False)
    groups: Dict[Tuple[str, str], List[str]] = {}

    for uid in sorted(active):
        if uid not in matrix.columns:
            continue
        semantic = str(meta.loc[uid].get("exact_semantic_key", ""))
        # Parser failures do not get broad semantic grouping: require raw feature identity.
        if str(meta.loc[uid].get("parser_status", "")) == "unparsed":
            semantic = "raw_feature::" + str(meta.loc[uid].get("feature", uid))
        groups.setdefault((semantic, vector_hash(matrix[uid])), []).append(uid)

    for (_, _), uids in groups.items():
        if len(uids) < 2:
            continue
        ordered = representative_sort(meta.loc[uids].copy())
        rep = str(ordered.iloc[0]["feature_uid"])
        for uid in ordered["feature_uid"].astype(str).tolist()[1:]:
            if uid not in active:
                continue
            exact, n = exact_vector_duplicate(matrix[rep], matrix[uid], atol=atol)
            pairs.append({
                "compression_rule": "exact_semantic_vector_duplicate",
                "feature_uid_a": rep,
                "feature_uid_b": uid,
                "spearman_rho": 1.0 if exact else np.nan,
                "pairwise_n": n,
                "passes_rule": bool(exact),
            })
            if not exact:
                continue
            active.remove(uid)
            replacement[uid] = rep
            audit.append({
                "removed_feature_uid": uid,
                "representative_feature_uid": rep,
                "compression_rule": "exact_semantic_vector_duplicate",
                "spearman_rho": 1.0,
                "pairwise_n": n,
                "oof_loss": numeric_value(meta.loc[uid], "oof_metric") - numeric_value(meta.loc[rep], "oof_metric"),
                "decision_reason": "same semantic measurement and identical transformed patient vector; higher-ranked Stage2A4 candidate retained",
            })
    return audit, pairs


def greedy_semantic_pass(
    *,
    rule_name: str,
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    active: Set[str],
    replacement: Dict[str, str],
    group_key: str,
    corr_threshold: float,
    max_oof_loss: float,
    min_pairwise_n: int,
    location_only: bool,
    require_different_summary: bool,
) -> Tuple[List[dict], List[dict]]:
    audit: List[dict] = []
    pairs: List[dict] = []
    meta = registry.set_index("feature_uid", drop=False)
    work = registry[registry["feature_uid"].astype(str).isin(active)].copy()
    if location_only:
        work = work[work["summary_class"].astype(str).eq("location")].copy()
    work = work[~work["parser_status"].astype(str).eq("unparsed")].copy()

    for _, group in work.groupby(group_key, dropna=False, sort=False):
        group = group[group["feature_uid"].astype(str).isin(active)].copy()
        if len(group) < 2:
            continue
        ordered = representative_sort(group)
        representatives: List[str] = []

        for _, row in ordered.iterrows():
            uid = str(row["feature_uid"])
            if uid not in active or uid not in matrix.columns:
                continue
            assigned = False
            for rep in representatives:
                if rep not in active or rep not in matrix.columns:
                    continue
                if require_different_summary:
                    s1 = str(meta.loc[uid].get("parsed_summary_stat", "")).strip().lower()
                    s2 = str(meta.loc[rep].get("parsed_summary_stat", "")).strip().lower()
                    if s1 == s2:
                        continue
                rho, n = safe_spearman(matrix[uid], matrix[rep], min_n=min_pairwise_n)
                uid_oof = numeric_value(meta.loc[uid], "oof_metric")
                rep_oof = numeric_value(meta.loc[rep], "oof_metric")
                oof_loss = uid_oof - rep_oof
                passes = bool(
                    np.isfinite(rho)
                    and rho >= float(corr_threshold)
                    and np.isfinite(oof_loss)
                    and oof_loss <= float(max_oof_loss)
                )
                pairs.append({
                    "compression_rule": rule_name,
                    "feature_uid_a": uid,
                    "feature_uid_b": rep,
                    "spearman_rho": rho,
                    "pairwise_n": n,
                    "oof_loss": oof_loss,
                    "passes_rule": passes,
                })
                if passes:
                    active.remove(uid)
                    replacement[uid] = rep
                    audit.append({
                        "removed_feature_uid": uid,
                        "representative_feature_uid": rep,
                        "compression_rule": rule_name,
                        "spearman_rho": rho,
                        "pairwise_n": n,
                        "oof_loss": oof_loss,
                        "decision_reason": "same root-safe semantic family; higher-ranked Stage2A4 representative retained after correlation and OOF-tolerance checks",
                    })
                    assigned = True
                    break
            if not assigned:
                representatives.append(uid)

    return audit, pairs


def all_pairwise_diagnostics(
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    min_pairwise_n: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ordered = representative_sort(registry)
    uids = [u for u in ordered["feature_uid"].astype(str) if u in matrix.columns]
    meta = registry.set_index("feature_uid", drop=False)
    rows: List[dict] = []
    vals: List[float] = []

    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            a, b = uids[i], uids[j]
            rho, n = safe_spearman(matrix[a], matrix[b], min_n=min_pairwise_n)
            absrho = abs(rho) if np.isfinite(rho) else np.nan
            if np.isfinite(absrho):
                vals.append(absrho)
            rows.append({
                "feature_uid_a": a,
                "feature_uid_b": b,
                "rank_a": numeric_value(meta.loc[a], "stage2a4_root_rank"),
                "rank_b": numeric_value(meta.loc[b], "stage2a4_root_rank"),
                "spearman_rho": rho,
                "abs_spearman_rho": absrho,
                "pairwise_n": n,
                "same_exact_semantic": str(meta.loc[a].get("exact_semantic_key", "")) == str(meta.loc[b].get("exact_semantic_key", "")),
                "same_root_safe_summary_family": str(meta.loc[a].get("root_safe_summary_key", "")) == str(meta.loc[b].get("root_safe_summary_key", "")),
            })

    arr = np.asarray(vals, dtype=float)
    summary = {
        "n_pairs_total": int(len(rows)),
        "n_pairs_estimable": int(len(vals)),
        "median_abs_rho": float(np.nanmedian(arr)) if arr.size else np.nan,
        "q90_abs_rho": float(np.nanquantile(arr, 0.90)) if arr.size else np.nan,
        "max_abs_rho": float(np.nanmax(arr)) if arr.size else np.nan,
        "n_pairs_abs_rho_ge_090": int(np.sum(arr >= 0.90)) if arr.size else 0,
        "n_pairs_abs_rho_ge_095": int(np.sum(arr >= 0.95)) if arr.size else 0,
        "n_pairs_abs_rho_ge_098": int(np.sum(arr >= 0.98)) if arr.size else 0,
    }
    return pd.DataFrame(rows), summary


def save_corr_heatmap(
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    min_pairwise_n: int,
) -> None:
    ordered = representative_sort(registry)
    uids = [u for u in ordered["feature_uid"].astype(str) if u in matrix.columns]
    if len(uids) < 2:
        return
    corr = np.full((len(uids), len(uids)), np.nan, dtype=float)
    np.fill_diagonal(corr, 1.0)
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            rho, _ = safe_spearman(matrix[uids[i]], matrix[uids[j]], min_n=min_pairwise_n)
            corr[i, j] = rho
            corr[j, i] = rho

    labels = []
    meta = registry.set_index("feature_uid", drop=False)
    for uid in uids:
        rank = numeric_value(meta.loc[uid], "stage2a4_root_rank")
        feat = str(meta.loc[uid].get("feature", uid))
        if len(feat) > 34:
            feat = feat[:31] + "..."
        labels.append("{}: {}".format(int(rank) if np.isfinite(rank) else "?", feat))

    size = max(6.5, min(15.0, 4.5 + 0.42 * len(uids)))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(uids)))
    ax.set_yticks(np.arange(len(uids)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_flow_plot(summary_row: Mapping, path: Path, title: str) -> None:
    labels = ["Stage2A4 seeds", "After exact", "After Mean/Median", "Final"]
    vals = [
        int(summary_row.get("n_input_features", 0)),
        int(summary_row.get("n_after_exact", 0)),
        int(summary_row.get("n_after_summary", 0)),
        int(summary_row.get("n_final_features", 0)),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(len(vals)), vals, marker="o")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Candidate features")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_stage2a4_root_manifest(cfg: Mapping) -> pd.DataFrame:
    p = root_manifest_path(cfg)
    if not p.exists():
        raise FileNotFoundError("Run Stage 2A-4 aggregate first: {}".format(p))
    m = pd.read_csv(p)
    required = {
        "array_id", "context_slug", ROOT_COL, "matrix_path", "candidate_registry_path",
        *CONTEXT_COLS,
    }
    missing = sorted(required - set(m.columns))
    if missing:
        raise ValueError("Stage2A4 root manifest missing columns: {}".format(missing))
    return m


def command_validate(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    manifest = load_stage2a4_root_manifest(cfg)
    parser = get_parser(cfg)
    # Tiny parser smoke test without imposing biological assumptions.
    _ = parser.parse_feature("All__density__tumour", feature_source="phenotype_only", feature_group="cell_features")

    problems: List[dict] = []
    audit: List[dict] = []
    for i, row in manifest.iterrows():
        mp = Path(str(row["matrix_path"]))
        rp = Path(str(row["candidate_registry_path"]))
        ok_m = mp.exists() or mp.with_suffix(".csv.gz").exists()
        ok_r = rp.exists()
        audit.append({
            "manifest_row": i,
            **{c: row[c] for c in CONTEXT_COLS},
            ROOT_COL: row[ROOT_COL],
            "matrix_exists": bool(ok_m),
            "registry_exists": bool(ok_r),
        })
        if not ok_m or not ok_r:
            problems.append({
                "manifest_row": i,
                "context_slug": row["context_slug"],
                ROOT_COL: row[ROOT_COL],
                "matrix_exists": bool(ok_m),
                "registry_exists": bool(ok_r),
            })

    pd.DataFrame(audit).to_csv(out / "stage2a5_validation_root_audit.csv", index=False)
    pd.DataFrame(problems).to_csv(out / "stage2a5_validation_problems.csv", index=False)
    write_json(dict(cfg), out / "config.resolved.json")
    if problems:
        raise RuntimeError("Stage2A5 validation failed for {} root inputs".format(len(problems)))
    log("[VALID] contexts={} root_inputs={}".format(manifest["context_slug"].nunique(), len(manifest)))


def command_inventory(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    manifest = load_stage2a4_root_manifest(cfg)
    # One CPU per endpoint context; worker loops only its few roots.
    keep_cols = ["array_id", "context_slug"] + CONTEXT_COLS
    idx = manifest[keep_cols].drop_duplicates().sort_values("array_id").reset_index(drop=True)
    idx.insert(0, "stage2a5_array_id", np.arange(len(idx), dtype=int))
    idx.to_csv(out / "stage2a5_context_index.csv", index=False)
    # Snapshot source root manifest so downstream runs are reproducible.
    manifest.to_csv(out / "stage2a4_root_matrix_manifest.snapshot.csv", index=False)
    write_json(dict(cfg), out / "config.resolved.json")
    log("[SAVE] {} contexts={}".format(out / "stage2a5_context_index.csv", len(idx)))


def resolve_array_id(value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or use a Slurm array")
    return int(env)


def load_context_index(cfg: Mapping) -> pd.DataFrame:
    p = output_root(cfg) / "stage2a5_context_index.csv"
    if not p.exists():
        raise FileNotFoundError("Run inventory first: {}".format(p))
    return pd.read_csv(p)


def get_context_row(cfg: Mapping, array_id: int) -> pd.Series:
    idx = load_context_index(cfg)
    m = idx[idx["stage2a5_array_id"].astype(int).eq(int(array_id))]
    if m.empty:
        raise IndexError("stage2a5_array_id={} not found".format(array_id))
    return m.iloc[0]


def context_dir(cfg: Mapping, row: pd.Series) -> Path:
    return output_root(cfg) / "contexts" / str(row["context_slug"])


def root_dir(cdir: Path, root: str) -> Path:
    return cdir / "roots" / str(root).replace("/", "_")


def context_root_inputs(cfg: Mapping, row: pd.Series) -> pd.DataFrame:
    m = load_stage2a4_root_manifest(cfg)
    return m[m["context_slug"].astype(str).eq(str(row["context_slug"]))].copy()


def compress_one_root(
    cfg: Mapping,
    parser,
    input_row: pd.Series,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, Path]:
    root = str(input_row[ROOT_COL])
    registry = pd.read_csv(str(input_row["candidate_registry_path"]))
    matrix = read_table(str(input_row["matrix_path"]))
    if "patient_id" not in matrix.columns:
        raise ValueError("Root matrix lacks patient_id: {}".format(input_row["matrix_path"]))
    if ROOT_COL not in registry.columns:
        raise ValueError("Root registry lacks feature_source")
    roots = set(registry[ROOT_COL].dropna().astype(str))
    if roots != {root}:
        raise ValueError("Root registry contamination: expected {} observed {}".format(root, sorted(roots)))

    input_registry_n = int(len(registry))
    registry = registry[registry["feature_uid"].astype(str).isin(matrix.columns)].copy()
    registry = registry.drop_duplicates("feature_uid", keep="first").reset_index(drop=True)
    registry = semantic_enrich(registry, parser)
    registry = representative_sort(registry).reset_index(drop=True)

    built_uids = registry["feature_uid"].astype(str).tolist()
    matrix = matrix[["patient_id"] + built_uids].copy()
    active: Set[str] = set(built_uids)
    replacement: Dict[str, str] = {}
    all_audit: List[dict] = []
    all_rule_pairs: List[dict] = []

    min_n = int(cfg.get("min_pairwise_n", 20))
    exact_atol = float(cfg.get("exact_duplicate_atol", 1e-12))
    summary_rho = float(cfg.get("summary_corr_threshold", 0.95))
    summary_oof_loss = float(cfg.get("summary_max_oof_loss", 0.005))
    residual_rho = float(cfg.get("exact_semantic_corr_threshold", 0.98))
    residual_oof_loss = float(cfg.get("exact_semantic_max_oof_loss", 0.005))

    before_pairs, before_stats = all_pairwise_diagnostics(registry, matrix, min_pairwise_n=min_n)
    before_pairs.to_csv(out_dir / "pairwise_correlations_before.csv", index=False)
    save_corr_heatmap(
        registry, matrix, out_dir / "correlation_heatmap_before.png",
        "{} | correlations before Stage2A5".format(root), min_pairwise_n=min_n,
    )

    audit, pairs = exact_duplicate_pass(registry, matrix, active, replacement, exact_atol)
    all_audit.extend(audit)
    all_rule_pairs.extend(pairs)
    n_after_exact = len(active)

    # Mean/Median only. Full cell/state identities and compartment are retained
    # in root_safe_summary_key, so this cannot perform state or compartment rescue.
    audit, pairs = greedy_semantic_pass(
        rule_name="mean_median_same_measure",
        registry=registry,
        matrix=matrix,
        active=active,
        replacement=replacement,
        group_key="root_safe_summary_key",
        corr_threshold=summary_rho,
        max_oof_loss=summary_oof_loss,
        min_pairwise_n=min_n,
        location_only=True,
        require_different_summary=True,
    )
    all_audit.extend(audit)
    all_rule_pairs.extend(pairs)
    n_after_summary = len(active)

    # Residual duplicate provenance/encoding of the exact same semantic measurement.
    audit, pairs = greedy_semantic_pass(
        rule_name="exact_semantic_near_duplicate",
        registry=registry,
        matrix=matrix,
        active=active,
        replacement=replacement,
        group_key="exact_semantic_key",
        corr_threshold=residual_rho,
        max_oof_loss=residual_oof_loss,
        min_pairwise_n=min_n,
        location_only=False,
        require_different_summary=False,
    )
    all_audit.extend(audit)
    all_rule_pairs.extend(pairs)

    final_uids = [
        uid for uid in representative_sort(registry)["feature_uid"].astype(str).tolist()
        if uid in active
    ]
    final = registry[registry["feature_uid"].astype(str).isin(final_uids)].copy()
    final = representative_sort(final).reset_index(drop=True)
    final["stage2a5_root_rank"] = np.arange(1, len(final) + 1)
    final["stage2a5_selection_policy"] = "within_root_conservative_microcompression"
    final["cross_root_compression"] = False
    final["state_simplification"] = False
    final["compartment_simplification"] = False

    # Map every input seed to its final representative.
    seed_map_rows: List[dict] = []
    represented: Dict[str, List[str]] = {}
    for uid in built_uids:
        rep = resolve_rep(uid, replacement)
        represented.setdefault(rep, []).append(uid)
        seed_map_rows.append({
            "seed_feature_uid": uid,
            "final_representative_uid": rep,
            "seed_changed_representative": bool(uid != rep),
        })
    final["represented_seed_feature_uids"] = final["feature_uid"].astype(str).map(
        lambda u: ";".join(sorted(set(represented.get(u, [u]))))
    )
    final["n_represented_seeds"] = final["feature_uid"].astype(str).map(
        lambda u: len(set(represented.get(u, [u])))
    )

    final_matrix = matrix[["patient_id"] + [u for u in final["feature_uid"].astype(str) if u in matrix.columns]].copy()
    final_matrix_path = save_table(final_matrix, out_dir / "compressed_patient_feature_matrix.parquet")
    final.to_csv(out_dir / "compressed_root_candidates.csv", index=False)
    pd.DataFrame(seed_map_rows).to_csv(out_dir / "seed_to_final_representative.csv", index=False)

    audit_df = pd.DataFrame(all_audit)
    if not audit_df.empty:
        meta_cols = [c for c in ["feature_uid", "feature", ROOT_COL, "feature_group", "stage2a4_root_rank", "oof_metric", "fold_sd"] if c in registry.columns]
        removed_meta = registry[meta_cols].rename(columns={c: "removed_" + c for c in meta_cols})
        rep_meta = registry[meta_cols].rename(columns={c: "representative_" + c for c in meta_cols})
        audit_df = audit_df.merge(removed_meta, left_on="removed_feature_uid", right_on="removed_feature_uid", how="left")
        audit_df = audit_df.merge(rep_meta, left_on="representative_feature_uid", right_on="representative_feature_uid", how="left")
    audit_df.to_csv(out_dir / "compression_decision_audit.csv", index=False)
    pd.DataFrame(all_rule_pairs).to_csv(out_dir / "evaluated_compression_pairs.csv", index=False)

    after_pairs, after_stats = all_pairwise_diagnostics(final, final_matrix, min_pairwise_n=min_n)
    after_pairs.to_csv(out_dir / "pairwise_correlations_after.csv", index=False)
    save_corr_heatmap(
        final, final_matrix, out_dir / "correlation_heatmap_after.png",
        "{} | correlations after Stage2A5".format(root), min_pairwise_n=min_n,
    )

    summary = {
        "array_id": int(input_row["array_id"]),
        **{c: input_row[c] for c in CONTEXT_COLS},
        "context_slug": input_row["context_slug"],
        ROOT_COL: root,
        "fixed_root_cap": int(input_row["fixed_root_cap"]) if "fixed_root_cap" in input_row and pd.notna(input_row["fixed_root_cap"]) else np.nan,
        "n_stage2a4_registry_rows": input_registry_n,
        "n_input_features": int(len(built_uids)),
        "n_parser_unparsed": int(registry["parser_status"].astype(str).eq("unparsed").sum()),
        "n_after_exact": int(n_after_exact),
        "n_after_summary": int(n_after_summary),
        "n_final_features": int(len(final)),
        "n_removed_exact": int(sum(1 for r in all_audit if r.get("compression_rule") == "exact_semantic_vector_duplicate")),
        "n_removed_mean_median": int(sum(1 for r in all_audit if r.get("compression_rule") == "mean_median_same_measure")),
        "n_removed_exact_semantic": int(sum(1 for r in all_audit if r.get("compression_rule") == "exact_semantic_near_duplicate")),
        "compression_fraction": float(len(final) / len(built_uids)) if built_uids else np.nan,
        **{"before_" + k: v for k, v in before_stats.items()},
        **{"after_" + k: v for k, v in after_stats.items()},
        "cross_root_compression_performed": False,
        "state_simplification_performed": False,
        "compartment_simplification_performed": False,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "root_compression_summary.csv", index=False)
    save_flow_plot(summary, out_dir / "compression_flow.png", "{} | {} | {} | {}".format(
        input_row["cohort"], input_row["panel"], input_row["endpoint"], root
    ))

    return final, audit_df, pd.DataFrame(seed_map_rows), summary, final_matrix_path


def command_worker(cfg: Mapping, array_id: int) -> None:
    row = get_context_row(cfg, array_id)
    cdir = ensure_dir(context_dir(cfg, row))
    parser = get_parser(cfg)
    root_inputs = context_root_inputs(cfg, row)
    if root_inputs.empty:
        raise RuntimeError("No Stage2A4 root inputs for {}".format(row["context_slug"]))

    log("=" * 100)
    log("[STAGE2A5 WITHIN-ROOT] array={} | {}".format(
        array_id, " | ".join("{}={}".format(c, row[c]) for c in CONTEXT_COLS)
    ))

    summaries: List[dict] = []
    manifest_rows: List[dict] = []
    for _, rr in root_inputs.sort_values(ROOT_COL).iterrows():
        root = str(rr[ROOT_COL])
        rdir = ensure_dir(root_dir(cdir, root))
        log("[ROOT] {}".format(root))
        final, audit, seedmap, summary, matrix_path = compress_one_root(cfg, parser, rr, rdir)
        summaries.append(summary)
        manifest_rows.append({
            "stage2a5_array_id": int(array_id),
            "array_id": int(rr["array_id"]),
            **{c: rr[c] for c in CONTEXT_COLS},
            "context_slug": rr["context_slug"],
            ROOT_COL: root,
            "fixed_root_cap": rr.get("fixed_root_cap", np.nan),
            "n_stage2a4_selected": int(rr.get("n_selected", len(final))) if pd.notna(rr.get("n_selected", np.nan)) else np.nan,
            "n_stage2a4_built": int(rr.get("n_built", len(final))) if pd.notna(rr.get("n_built", np.nan)) else np.nan,
            "n_stage2a5_final": int(len(final)),
            "matrix_path": str(matrix_path),
            "candidate_registry_path": str(rdir / "compressed_root_candidates.csv"),
            "compression_summary_path": str(rdir / "root_compression_summary.csv"),
            "compression_audit_path": str(rdir / "compression_decision_audit.csv"),
            "seed_map_path": str(rdir / "seed_to_final_representative.csv"),
        })

    pd.DataFrame(summaries).to_csv(cdir / "context_root_compression_summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(cdir / "root_matrix_manifest.csv", index=False)
    context_summary = {
        "stage2a5_array_id": int(array_id),
        **{c: row[c] for c in CONTEXT_COLS},
        "context_slug": row["context_slug"],
        "n_roots": int(len(summaries)),
        "n_input_features": int(sum(int(x["n_input_features"]) for x in summaries)),
        "n_final_features": int(sum(int(x["n_final_features"]) for x in summaries)),
        "n_removed_total": int(sum(int(x["n_input_features"]) - int(x["n_final_features"]) for x in summaries)),
        "cross_root_compression_performed": False,
    }
    pd.DataFrame([context_summary]).to_csv(cdir / "context_stage2a5_summary.csv", index=False)
    (cdir / ".done").write_text("complete\n")
    log("[DONE] roots={} input={} final={}".format(
        context_summary["n_roots"], context_summary["n_input_features"], context_summary["n_final_features"]
    ))


def command_aggregate(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    idx = load_context_index(cfg)
    context_summaries: List[pd.DataFrame] = []
    root_summaries: List[pd.DataFrame] = []
    manifests: List[pd.DataFrame] = []
    final_candidates: List[pd.DataFrame] = []
    audits: List[pd.DataFrame] = []
    seedmaps: List[pd.DataFrame] = []
    missing: List[dict] = []

    for _, row in idx.sort_values("stage2a5_array_id").iterrows():
        cdir = context_dir(cfg, row)
        if not (cdir / ".done").exists():
            missing.append({
                "stage2a5_array_id": int(row["stage2a5_array_id"]),
                "context_slug": row["context_slug"],
                "reason": "missing_done_marker",
            })
            continue
        cp = cdir / "context_stage2a5_summary.csv"
        rp = cdir / "context_root_compression_summary.csv"
        mp = cdir / "root_matrix_manifest.csv"
        if cp.exists(): context_summaries.append(pd.read_csv(cp))
        if rp.exists(): root_summaries.append(pd.read_csv(rp))
        if mp.exists():
            man = pd.read_csv(mp)
            manifests.append(man)
            for _, mr in man.iterrows():
                regp = Path(str(mr["candidate_registry_path"]))
                if regp.exists():
                    d = pd.read_csv(regp)
                    if not d.empty:
                        final_candidates.append(d)
                ap = Path(str(mr["compression_audit_path"]))
                if ap.exists() and ap.stat().st_size > 0:
                    try:
                        d = pd.read_csv(ap)
                    except pd.errors.EmptyDataError:
                        d = pd.DataFrame()
                    if not d.empty:
                        d["context_slug"] = mr["context_slug"]
                        d[ROOT_COL] = mr[ROOT_COL]
                        audits.append(d)
                sp = Path(str(mr["seed_map_path"]))
                if sp.exists() and sp.stat().st_size > 0:
                    try:
                        d = pd.read_csv(sp)
                    except pd.errors.EmptyDataError:
                        d = pd.DataFrame()
                    if not d.empty:
                        d["context_slug"] = mr["context_slug"]
                        d[ROOT_COL] = mr[ROOT_COL]
                        seedmaps.append(d)

    cs = pd.concat(context_summaries, ignore_index=True, sort=False) if context_summaries else pd.DataFrame()
    rs = pd.concat(root_summaries, ignore_index=True, sort=False) if root_summaries else pd.DataFrame()
    man = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()
    cand = pd.concat(final_candidates, ignore_index=True, sort=False) if final_candidates else pd.DataFrame()
    aud = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    smap = pd.concat(seedmaps, ignore_index=True, sort=False) if seedmaps else pd.DataFrame()

    cs.to_csv(out / "all_context_stage2a5_summary.csv", index=False)
    rs.to_csv(out / "all_context_root_compression_summary.csv", index=False)
    man.to_csv(out / "stage2a5_root_matrix_manifest.csv", index=False)
    save_table(cand, out / "all_context_root_final_candidates.parquet")
    aud.to_csv(out / "all_context_compression_decision_audit.csv", index=False)
    smap.to_csv(out / "all_context_seed_to_final_representative.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "stage2a5_missing_context_outputs.csv", index=False)

    if not rs.empty:
        panel_root = (
            rs.groupby(["panel", ROOT_COL], dropna=False)
            .agg(
                n_context_roots=("context_slug", "nunique"),
                total_input_features=("n_input_features", "sum"),
                total_final_features=("n_final_features", "sum"),
                median_input_features=("n_input_features", "median"),
                median_final_features=("n_final_features", "median"),
                median_compression_fraction=("compression_fraction", "median"),
                total_removed_exact=("n_removed_exact", "sum"),
                total_removed_mean_median=("n_removed_mean_median", "sum"),
                total_removed_exact_semantic=("n_removed_exact_semantic", "sum"),
                median_before_q90_abs_rho=("before_q90_abs_rho", "median"),
                median_after_q90_abs_rho=("after_q90_abs_rho", "median"),
            )
            .reset_index()
        )
        panel_root["total_removed"] = panel_root["total_input_features"] - panel_root["total_final_features"]
        panel_root.to_csv(out / "panel_root_microcompression_summary.csv", index=False)

    if not cand.empty:
        support = (
            cand.groupby(["panel", ROOT_COL, "feature_uid"], dropna=False)
            .agg(
                feature=("feature", "first"),
                feature_group=("feature_group", "first"),
                n_contexts=("context_slug", "nunique"),
                n_cohorts=("cohort", "nunique"),
                endpoints=("endpoint", lambda x: ";".join(sorted(set(map(str, x))))),
                best_root_rank=("stage2a5_root_rank", "min"),
                median_oof=("oof_metric", "median"),
                median_fold_sd=("fold_sd", "median"),
            )
            .reset_index()
            .sort_values(["panel", ROOT_COL, "n_cohorts", "n_contexts", "best_root_rank"], ascending=[True, True, False, False, True])
        )
        support.to_csv(out / "stage2a5_final_feature_support.csv", index=False)

    write_json({
        "n_expected_contexts": int(len(idx)),
        "n_completed_contexts": int(len(cs)),
        "n_missing_contexts": int(len(missing)),
        "n_root_matrix_rows": int(len(man)),
        "n_final_candidate_rows": int(len(cand)),
        "cross_root_compression_performed": False,
        "state_simplification_performed": False,
        "compartment_simplification_performed": False,
        "generic_correlation_pruning_performed": False,
        "summary_rule": "Mean/Median only; full state identity + measurement compartment preserved",
    }, out / "stage2a5_aggregate_summary.json")

    if missing:
        raise RuntimeError("{} context workers missing; see stage2a5_missing_context_outputs.csv".format(len(missing)))
    log("[DONE] contexts={} root_matrices={} final_candidate_rows={}".format(len(cs), len(man), len(cand)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ["validate", "inventory", "worker", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "worker":
            p.add_argument("--array-id", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "validate":
        command_validate(cfg)
    elif args.command == "inventory":
        command_inventory(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
