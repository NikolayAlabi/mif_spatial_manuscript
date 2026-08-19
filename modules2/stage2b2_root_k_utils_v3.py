#!/usr/bin/env python3
"""
stage2b2_root_k_utils_v3.py

Helper functions for Stage 2B-2 root-specific K review.

Purpose
-------
Keep the Stage 2B-2 notebook small while providing a reproducible K-selection
workflow for each panel x prep-root consensus matrix produced by Stage 2B-1.

Primary design choices
----------------------
1. Root-specific module discovery only.
2. Row-Spearman consensus-profile geometry is primary; direct signed consensus
   distance is the sensitivity analysis. Direct-absolute geometry is available
   but disabled by default. Because row-Spearman groups features by similar
   network position, direct signed within-module cohesion is reported as an
   explicit diagnostic/guardrail for downstream mean-z scoreability.
3. Statistical silhouette is retained.
4. Semantic silhouette is retained as a diagnostic only.
5. Semantic COHERENCE is the semantic quantity used in the overall K score.
   Coherence is calculated for three nested root-aware semantic views:
       - primitive: root-appropriate biological identities/states
       - tissue-aware: primitive biology + measurement tissue/compartment
       - metric-aware: primitive biology + tissue + measurement family/metric
6. For coherence, both the raw within-module similarity and enrichment above a
   cluster-size-preserving random-label null are reported. The exact expected
   null mean is the global pairwise semantic similarity; optional permutations
   provide a null SD/z-score/p-value.
7. The overall K score is rank-normalized WITHIN each panel x root x distance
   mode. It uses coherence enrichment, not semantic silhouette.
8. Cross-cohort reproducibility is calculated as a diagnostic, not used in the
   composite by default, so the module geometry and semantic interpretation do
   not become over-tuned to another arbitrary weighted term.

Expected Stage 2B-1 inputs
--------------------------
<stage2b1_root>/stage2b1_root_consensus_manifest.csv
<stage2b1_root>/root_consensus/<panel__root>/
    consensus_signed_spearman_support_filtered_clustering.parquet
    pairwise_consensus_audit.csv.gz
    feature_universe.csv

The module is intentionally notebook-friendly but can also be imported from a
script. Plotting uses matplotlib only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_DISTANCE_MODES = ("row_spearman", "direct_signed")
DEFAULT_PRIMARY_MODE = "row_spearman"
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 25
DEFAULT_LINKAGE_METHOD = "average"
DEFAULT_N_SEMANTIC_PERMUTATIONS = 100
DEFAULT_RANDOM_SEED = 20260818

# Primary K selection is intentionally constrained to a parsimonious range.
# K values above this are still evaluated and plotted as extended diagnostics.
DEFAULT_PRIMARY_SELECTION_K_MAX = 10
DEFAULT_SELECTION_WITHIN_BEST_FRACTION = 0.90

# Composite focuses on statistical structure, primitive semantic coherence,
# modest tissue-aware coherence, and explicit control of giant clusters/singletons.
# Metric-aware coherence and all semantic silhouettes are diagnostic only.
DEFAULT_K_SCORE_WEIGHTS = {
    # Primary evidence for K: statistical structure + primitive biology.
    "stat_silhouette": 0.45,
    "primitive_coherence": 0.30,
    # Tissue-aware coherence contributes modestly because it is nested with
    # primitive semantics. Metric-aware coherence is diagnostic only.
    "tissue_coherence": 0.10,
    "metric_coherence": 0.00,
    # Explicitly reward usable module-size structure.
    "cluster_balance": 0.075,
    "non_singleton": 0.075,
}

# These are review flags, not eligibility filters for the score itself.
DEFAULT_REVIEW_THRESHOLDS = {
    # Soft review guardrails, not hard exclusions from the composite.
    "max_cluster_fraction": 0.50,
    "singleton_fraction": 0.25,
    "within_best_fraction": 0.95,
    # Row-Spearman modules should still be directly coordinated enough to
    # support downstream signed mean-z scoring. These are warning thresholds
    # only and do not enter the K composite.
    "min_q25_module_mean_consensus_rho": 0.00,
    "min_median_module_fraction_positive_pairs": 0.60,
}

COMPARTMENT_IDENTITY_MAP = {
    "immune_cell": "immune",
    "tumor_cell": "tumor",
    "stromal_cell": "stroma",
}

ROOT_STATE_AWARE = {"AR_state", "compartment_state"}
ROOT_CHECKPOINT_PRIMARY = {"AR_checkpoint_state"}
ROOT_COMPARTMENT_PRIMARY = {"compartment", "compartment_state"}


# =============================================================================
# Small utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_slug(x: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_")


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # pandas rank ascending=True -> largest value gets highest percentile.
    return x.rank(pct=True, ascending=higher_better)


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json_load_maybe(x, default):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _jaccard(a: Iterable[str], b: Iterable[str], both_empty: float = np.nan) -> float:
    a = set(v for v in a if v is not None and str(v) != "")
    b = set(v for v in b if v is not None and str(v) != "")
    if not a and not b:
        return float(both_empty) if not pd.isna(both_empty) else np.nan
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _weighted_available(parts: Sequence[Tuple[float, float | None]]) -> float:
    vals = [(float(w), float(v)) for w, v in parts if v is not None and np.isfinite(v)]
    if not vals:
        return 0.0
    denom = sum(w for w, _ in vals)
    if denom <= 0:
        return 0.0
    return float(sum(w * v for w, v in vals) / denom)


def _safe_silhouette(distance: pd.DataFrame, labels: Sequence[int]) -> float:
    labels = np.asarray(labels)
    n = len(labels)
    if n < 3:
        return np.nan
    n_groups = len(np.unique(labels))
    if n_groups < 2 or n_groups >= n:
        return np.nan
    arr = distance.to_numpy(dtype=float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    try:
        return float(silhouette_score(arr, labels, metric="precomputed"))
    except Exception:
        return np.nan


def _cluster_size_stats(labels: Sequence[int]) -> dict:
    s = pd.Series(np.asarray(labels)).value_counts().sort_values(ascending=False)
    n = int(s.sum())
    return {
        "n_clusters_observed": int(len(s)),
        "min_cluster_size": int(s.min()) if len(s) else 0,
        "median_cluster_size": float(s.median()) if len(s) else np.nan,
        "mean_cluster_size": float(s.mean()) if len(s) else np.nan,
        "max_cluster_size": int(s.max()) if len(s) else 0,
        "max_cluster_fraction": float(s.max() / n) if n else np.nan,
        # Fraction of MODULES that are singletons (same convention as prior code).
        "singleton_fraction": float((s == 1).mean()) if len(s) else np.nan,
        "n_singletons": int((s == 1).sum()) if len(s) else 0,
        "n_non_singleton_modules": int((s >= 2).sum()) if len(s) else 0,
    }


def _labels_for_k(Z, k: int) -> np.ndarray:
    # cut_tree gives the exact requested number of clusters when feasible and
    # avoids maxclust tie behavior that can occasionally return fewer clusters.
    return cut_tree(Z, n_clusters=[int(k)]).reshape(-1).astype(int) + 1


# =============================================================================
# Stage 2B-1 loading
# =============================================================================

def load_b1_manifest(stage2b1_root: str | Path) -> pd.DataFrame:
    root = Path(stage2b1_root)
    p = root / "stage2b1_root_consensus_manifest.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "feature_source" not in df.columns:
            raise KeyError(f"{p} lacks feature_source")
        return df.sort_values(["panel", "feature_source"]).reset_index(drop=True)

    # Fallback if aggregate is not yet run but individual root outputs are done.
    idx = root / "stage2b1_consensus_worker_index.csv"
    if not idx.exists():
        raise FileNotFoundError(
            f"Could not find {p} or {idx}. Finish Stage 2B-1 setup/consensus first."
        )
    df = pd.read_csv(idx)
    rows = []
    for _, r in df.iterrows():
        wdir = root / "root_consensus" / str(r["consensus_slug"])
        sp = wdir / "root_consensus_summary.csv"
        if not sp.exists():
            continue
        z = pd.read_csv(sp)
        rows.append(z.iloc[0].to_dict())
    if not rows:
        raise RuntimeError("No completed Stage 2B-1 root consensus summaries found")
    return pd.DataFrame(rows).sort_values(["panel", "feature_source"]).reset_index(drop=True)


def _find_root_dir(stage2b1_root: Path, row: Mapping[str, object]) -> Path:
    if "root_output_dir" in row and pd.notna(row.get("root_output_dir")):
        p = Path(str(row["root_output_dir"]))
        if p.exists():
            return p
    slug = row.get("consensus_slug")
    if slug is not None and pd.notna(slug):
        p = stage2b1_root / "root_consensus" / str(slug)
        if p.exists():
            return p
    p = stage2b1_root / "root_consensus" / "__".join(
        [safe_slug(row["panel"]), safe_slug(row["feature_source"])]
    )
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_root_inputs(stage2b1_root: str | Path, row: Mapping[str, object]) -> Dict[str, object]:
    root = Path(stage2b1_root)
    wdir = _find_root_dir(root, row)

    cpath = wdir / "consensus_signed_spearman_support_filtered_clustering.parquet"
    if not cpath.exists():
        raise FileNotFoundError(cpath)
    consensus = pd.read_parquet(cpath)
    # Stage2B1 save_square() resets the feature index into a feature_uid column.
    if "feature_uid" in consensus.columns:
        consensus = consensus.set_index("feature_uid")
    consensus.index = consensus.index.astype(str)
    consensus.columns = consensus.columns.astype(str)
    common = [f for f in consensus.index if f in consensus.columns]
    consensus = consensus.loc[common, common].copy()

    upath = wdir / "feature_universe.csv"
    if not upath.exists() and "feature_universe_path" in row:
        upath = Path(str(row["feature_universe_path"]))
    universe = pd.read_csv(upath)

    apath = wdir / "pairwise_consensus_audit.csv.gz"
    audit = pd.read_csv(apath) if apath.exists() else pd.DataFrame()

    return {
        "root_dir": wdir,
        "consensus": consensus,
        "universe": universe,
        "pairwise_audit": audit,
    }


# =============================================================================
# Root-aware semantic ontology
# =============================================================================

def _parse_feature_meta_row(row: Mapping[str, object], parser=None) -> dict:
    """Return normalized semantic ingredients for one B1 universe feature."""
    root = str(row.get("feature_source", ""))
    group = str(row.get("feature_group", ""))
    feature = str(row.get("feature", ""))

    entities = _json_load_maybe(row.get("parsed_entities_json"), [])
    params = _json_load_maybe(row.get("parsed_metric_params_json"), {})
    ftype = row.get("parsed_feature_type")
    fsub = row.get("parsed_feature_subtype")
    compartment = row.get("parsed_compartment")
    summary = row.get("parsed_summary_stat")
    status = row.get("parser_status", "")

    # Fall back to the grammar-aware parser if B1 universe metadata is missing.
    if (not entities and parser is not None) or pd.isna(ftype) or str(ftype).strip() == "":
        parsed = parser.parse_feature(
            feature=feature,
            feature_source=root,
            feature_group=group,
        )
        entities = parsed.get("entities", []) or []
        params = parsed.get("metric_params", {}) or {}
        ftype = parsed.get("feature_type")
        fsub = parsed.get("feature_subtype")
        compartment = parsed.get("compartment")
        summary = parsed.get("summary_stat")
        status = parsed.get("parse_status", status)

    identities: List[str] = []
    states: List[str] = []
    role_identity_tokens: List[str] = []
    role_state_tokens: List[str] = []

    for e in entities or []:
        role = str(e.get("role") or "entity")
        cell = e.get("cell")
        state = e.get("state")

        if root in ROOT_CHECKPOINT_PRIMARY:
            if state:
                tok = str(state)
                identities.append(tok)
                states.append(tok)
                role_identity_tokens.append(f"{role}:state:{tok}")
        elif root in ROOT_COMPARTMENT_PRIMARY:
            if cell:
                ident = COMPARTMENT_IDENTITY_MAP.get(str(cell), str(cell))
                identities.append(ident)
                role_identity_tokens.append(f"{role}:identity:{ident}")
            if root in ROOT_STATE_AWARE and state:
                states.append(str(state))
                role_state_tokens.append(f"{role}:state:{state}")
        else:
            # phenotype_only and AR_state primary identity = native phenotype.
            if cell:
                identities.append(str(cell))
                role_identity_tokens.append(f"{role}:identity:{cell}")
            if root in ROOT_STATE_AWARE and state:
                states.append(str(state))
                role_state_tokens.append(f"{role}:state:{state}")

    primitive_tokens = set(role_identity_tokens)
    if root in ROOT_STATE_AWARE:
        primitive_tokens.update(role_state_tokens)
    if root in ROOT_CHECKPOINT_PRIMARY and not primitive_tokens:
        # Defensive fallback for any state-only parser record without roles.
        primitive_tokens.update(f"state:{s}" for s in states)

    tissue_tokens = set()
    if compartment is not None and not pd.isna(compartment) and str(compartment).strip():
        tissue_tokens.add(f"tissue:{str(compartment)}")

    # Metric tokens intentionally keep family/subtype distinct from summary and
    # detailed parameters; metric-aware similarity combines these transparently.
    metric_family_tokens = {
        f"feature_group:{group}",
        f"feature_type:{str(ftype)}" if ftype is not None and not pd.isna(ftype) else "",
        f"feature_subtype:{str(fsub)}" if fsub is not None and not pd.isna(fsub) else "",
    }
    metric_family_tokens.discard("")

    summary_tokens = set()
    if summary is not None and not pd.isna(summary) and str(summary).strip():
        summary_tokens.add(f"summary:{str(summary).strip().lower()}")

    param_tokens = set()
    for k, v in sorted((params or {}).items()):
        param_tokens.add(f"param:{k}={json.dumps(v, sort_keys=True, default=str)}")

    return {
        "panel": str(row.get("panel", "")),
        "feature_source": root,
        "feature_group": group,
        "feature": feature,
        "stage2b_feature_uid": str(row.get("stage2b_feature_uid", row.get("feature_uid", feature))),
        "parser_status": str(status),
        "primitive_tokens": primitive_tokens,
        "tissue_tokens": tissue_tokens,
        "metric_family_tokens": metric_family_tokens,
        "summary_tokens": summary_tokens,
        "param_tokens": param_tokens,
        "identities": sorted(set(identities)),
        "states": sorted(set(states)),
        "compartment": "" if compartment is None or pd.isna(compartment) else str(compartment),
        "feature_type": "" if ftype is None or pd.isna(ftype) else str(ftype),
        "feature_subtype": "" if fsub is None or pd.isna(fsub) else str(fsub),
        "summary_stat": "" if summary is None or pd.isna(summary) else str(summary),
    }


def build_root_semantic_ontology(
    universe: pd.DataFrame,
    features: Sequence[str],
    parser_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build root-aware primitive, tissue-aware, and metric-aware similarities.

    Similarity definitions
    ----------------------
    primitive:
        Jaccard of root-appropriate role-aware biological primitive tokens.
        For state-aware roots, state tokens are included. For
        AR_checkpoint_state, checkpoint states are the PRIMARY identity.

    tissue-aware:
        75% primitive + 25% measurement tissue/compartment, with adaptive
        re-normalization if neither feature has an explicit tissue token.

    metric-aware:
        55% primitive + 20% tissue + 25% measurement-metric similarity.
        Metric similarity itself is an adaptive combination of feature
        family/subtype, summary statistic, and metric parameters.

    The biology remains dominant by design; metric awareness refines rather than
    replaces biological similarity.
    """
    features = list(map(str, features))
    parser = import_module_from_path("stage2b2_feature_parser", parser_path) if parser_path else None

    u = universe.copy()
    uid_col = "stage2b_feature_uid" if "stage2b_feature_uid" in u.columns else (
        "feature_uid" if "feature_uid" in u.columns else None
    )
    if uid_col is None:
        raise KeyError("Feature universe must contain stage2b_feature_uid or feature_uid")
    u[uid_col] = u[uid_col].astype(str)
    u = u.drop_duplicates(uid_col).set_index(uid_col, drop=False)

    records: Dict[str, dict] = {}
    flat_rows: List[dict] = []
    for f in features:
        if f not in u.index:
            raise KeyError(f"Feature {f!r} not found in Stage2B1 feature universe")
        rec = _parse_feature_meta_row(u.loc[f].to_dict(), parser=parser)
        records[f] = rec
        flat_rows.append({
            **{k: v for k, v in rec.items() if not isinstance(v, set)},
            "primitive_tokens": ";".join(sorted(rec["primitive_tokens"])),
            "tissue_tokens": ";".join(sorted(rec["tissue_tokens"])),
            "metric_family_tokens": ";".join(sorted(rec["metric_family_tokens"])),
            "summary_tokens": ";".join(sorted(rec["summary_tokens"])),
            "param_tokens": ";".join(sorted(rec["param_tokens"])),
            "identities": ";".join(rec["identities"]),
            "states": ";".join(rec["states"]),
        })

    ontology = pd.DataFrame(flat_rows)

    primitive = pd.DataFrame(0.0, index=features, columns=features)
    tissue_aware = pd.DataFrame(0.0, index=features, columns=features)
    metric_aware = pd.DataFrame(0.0, index=features, columns=features)
    tissue_only = pd.DataFrame(np.nan, index=features, columns=features)
    metric_only = pd.DataFrame(np.nan, index=features, columns=features)

    for i, a in enumerate(features):
        ra = records[a]
        for j in range(i, len(features)):
            b = features[j]
            rb = records[b]

            prim = _jaccard(ra["primitive_tokens"], rb["primitive_tokens"], both_empty=0.0)

            # Tissue is only informative when at least one feature explicitly
            # carries a tissue/measurement compartment annotation.
            if ra["tissue_tokens"] or rb["tissue_tokens"]:
                tis = _jaccard(ra["tissue_tokens"], rb["tissue_tokens"], both_empty=np.nan)
            else:
                tis = np.nan

            fam = _jaccard(ra["metric_family_tokens"], rb["metric_family_tokens"], both_empty=0.0)
            summ = (
                _jaccard(ra["summary_tokens"], rb["summary_tokens"], both_empty=np.nan)
                if (ra["summary_tokens"] or rb["summary_tokens"])
                else np.nan
            )
            pars = (
                _jaccard(ra["param_tokens"], rb["param_tokens"], both_empty=np.nan)
                if (ra["param_tokens"] or rb["param_tokens"])
                else np.nan
            )
            met = _weighted_available([
                (0.60, fam),
                (0.20, summ),
                (0.20, pars),
            ])

            tissue_cum = _weighted_available([
                (0.75, prim),
                (0.25, tis),
            ])
            metric_cum = _weighted_available([
                (0.55, prim),
                (0.20, tis),
                (0.25, met),
            ])

            primitive.iat[i, j] = primitive.iat[j, i] = prim
            tissue_only.iat[i, j] = tissue_only.iat[j, i] = tis
            metric_only.iat[i, j] = metric_only.iat[j, i] = met
            tissue_aware.iat[i, j] = tissue_aware.iat[j, i] = tissue_cum
            metric_aware.iat[i, j] = metric_aware.iat[j, i] = metric_cum

    for M in (primitive, tissue_aware, metric_aware, metric_only):
        np.fill_diagonal(M.values, 1.0)
    # tissue_only remains NaN on diagonal when tissue isn't applicable; it is
    # an audit matrix rather than a semantic layer used for clustering metrics.

    return ontology, {
        "primitive": primitive,
        "tissue_aware": tissue_aware,
        "metric_aware": metric_aware,
        "tissue_only": tissue_only,
        "metric_only": metric_only,
    }


# =============================================================================
# Statistical distance / clustering
# =============================================================================

def build_distance_matrix(consensus: pd.DataFrame, mode: str) -> pd.DataFrame:
    C = consensus.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "direct_signed":
        D = 1.0 - C
    elif mode == "direct_abs":
        D = 1.0 - C.abs()
    elif mode == "row_spearman":
        row_corr = C.T.corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        D = 1.0 - row_corr
    else:
        raise ValueError(f"Unknown distance_mode={mode!r}")
    D = D.clip(lower=0.0, upper=2.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D.values, 0.0)
    return D


def linkage_from_distance(D: pd.DataFrame, method: str = DEFAULT_LINKAGE_METHOD):
    if D.shape[0] < 2:
        return None
    arr = D.to_numpy(dtype=float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    return linkage(squareform(arr, checks=False), method=method)


# =============================================================================
# Semantic coherence
# =============================================================================

def semantic_coherence_stats(
    similarity: pd.DataFrame,
    labels: Sequence[int],
    *,
    n_permutations: int = DEFAULT_N_SEMANTIC_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """
    Calculate semantic coherence of a proposed partition.

    raw_coherence
        Mean semantic similarity across all within-module feature pairs,
        weighted naturally by the number of feature pairs in each module.

    null_mean
        Exact expected mean under a cluster-size-preserving random partition.
        Because every feature pair has equal probability of being assigned to
        the same cluster, this is simply the global mean pairwise similarity.

    enrichment
        raw_coherence - null_mean. This is the main semantic quantity used in
        the K composite because it corrects the mechanical baseline similarity
        of the feature universe.

    z / empirical p
        Optional cluster-size-preserving permutation diagnostics.
    """
    labels = np.asarray(labels)
    S = similarity.to_numpy(dtype=float)
    n = len(labels)
    if n < 2:
        return {
            "raw": np.nan, "null_mean": np.nan, "enrichment": np.nan,
            "null_sd": np.nan, "z": np.nan, "empirical_p": np.nan,
            "n_within_pairs": 0,
        }

    iu = np.triu_indices(n, k=1)
    vals = S[iu]
    finite = np.isfinite(vals)
    same = labels[iu[0]] == labels[iu[1]]
    within_mask = finite & same
    all_mask = finite

    raw = float(np.mean(vals[within_mask])) if within_mask.any() else np.nan
    null_mean = float(np.mean(vals[all_mask])) if all_mask.any() else np.nan
    enrichment = raw - null_mean if np.isfinite(raw) and np.isfinite(null_mean) else np.nan

    null_sd = np.nan
    z = np.nan
    p = np.nan
    if n_permutations and n_permutations > 0 and within_mask.any() and all_mask.any():
        rng = np.random.default_rng(int(random_seed))
        null_vals = np.empty(int(n_permutations), dtype=float)
        for b in range(int(n_permutations)):
            plab = rng.permutation(labels)
            pmask = finite & (plab[iu[0]] == plab[iu[1]])
            null_vals[b] = float(np.mean(vals[pmask])) if pmask.any() else np.nan
        null_vals = null_vals[np.isfinite(null_vals)]
        if len(null_vals):
            null_sd = float(np.std(null_vals, ddof=1)) if len(null_vals) > 1 else 0.0
            if null_sd > 1e-12 and np.isfinite(raw):
                z = float((raw - np.mean(null_vals)) / null_sd)
            if np.isfinite(raw):
                p = float((1 + np.sum(null_vals >= raw)) / (len(null_vals) + 1))

    return {
        "raw": raw,
        "null_mean": null_mean,
        "enrichment": enrichment,
        "null_sd": null_sd,
        "z": z,
        "empirical_p": p,
        "n_within_pairs": int(within_mask.sum()),
    }


def module_semantic_coherence(similarity: pd.DataFrame, labels: Sequence[int]) -> pd.DataFrame:
    labels = np.asarray(labels)
    feats = similarity.index.astype(str).tolist()
    rows = []
    for lab in sorted(np.unique(labels)):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            raw = np.nan
            n_pairs = 0
        else:
            sub = similarity.iloc[idx, idx].to_numpy(dtype=float)
            iu = np.triu_indices(len(idx), k=1)
            vals = sub[iu]
            vals = vals[np.isfinite(vals)]
            raw = float(np.mean(vals)) if len(vals) else np.nan
            n_pairs = int(len(vals))
        rows.append({
            "raw_cluster_id": int(lab),
            "module_size": int(len(idx)),
            "semantic_coherence": raw,
            "n_semantic_pairs": n_pairs,
            "features": ";".join(feats[i] for i in idx),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Cross-cohort reproducibility
# =============================================================================

def _audit_pair_key_columns(audit: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("feature_uid_1", "feature_uid_2"),
        ("feature_uid_a", "feature_uid_b"),
        ("feature_a", "feature_b"),
        ("feature1", "feature2"),
    ]
    for a, b in candidates:
        if a in audit.columns and b in audit.columns:
            return a, b
    raise KeyError("Could not identify feature-pair columns in pairwise_consensus_audit")


def cross_cohort_reproducibility(
    audit: pd.DataFrame,
    labels: Sequence[int],
    features: Sequence[str],
) -> Tuple[dict, pd.DataFrame]:
    """Summarize within-module signed rho separately in each discovery cohort."""
    if audit is None or audit.empty:
        return {
            "median_module_cohort_mean_rho": np.nan,
            "q25_module_cohort_mean_rho": np.nan,
            "fraction_module_cohort_positive": np.nan,
            "fraction_modules_positive_all_available_cohorts": np.nan,
            "median_module_cross_cohort_sd": np.nan,
        }, pd.DataFrame()

    a_col, b_col = _audit_pair_key_columns(audit)
    lab_map = {str(f): int(l) for f, l in zip(features, labels)}
    z = audit.copy()
    z[a_col] = z[a_col].astype(str)
    z[b_col] = z[b_col].astype(str)
    z = z[z[a_col].isin(lab_map) & z[b_col].isin(lab_map)].copy()
    z["cluster_a"] = z[a_col].map(lab_map)
    z["cluster_b"] = z[b_col].map(lab_map)
    z = z[z["cluster_a"] == z["cluster_b"]].copy()

    rho_cols = [c for c in z.columns if c.startswith("rho_")]
    rows = []
    for lab, g in z.groupby("cluster_a", sort=True):
        for c in rho_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals):
                rows.append({
                    "raw_cluster_id": int(lab),
                    "cohort": c[len("rho_"):],
                    "mean_within_module_rho": float(vals.mean()),
                    "median_within_module_rho": float(vals.median()),
                    "n_pairs": int(len(vals)),
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "median_module_cohort_mean_rho": np.nan,
            "q25_module_cohort_mean_rho": np.nan,
            "fraction_module_cohort_positive": np.nan,
            "fraction_modules_positive_all_available_cohorts": np.nan,
            "median_module_cross_cohort_sd": np.nan,
        }, detail

    vals = pd.to_numeric(detail["mean_within_module_rho"], errors="coerce")
    per_mod = detail.groupby("raw_cluster_id")["mean_within_module_rho"].agg(list)
    all_pos = []
    sds = []
    for _, arr in per_mod.items():
        a = np.asarray([v for v in arr if np.isfinite(v)], dtype=float)
        if len(a):
            all_pos.append(bool(np.all(a > 0)))
            sds.append(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)

    summary = {
        "median_module_cohort_mean_rho": float(vals.median()) if vals.notna().any() else np.nan,
        "q25_module_cohort_mean_rho": float(vals.quantile(0.25)) if vals.notna().any() else np.nan,
        "fraction_module_cohort_positive": float((vals > 0).mean()) if vals.notna().any() else np.nan,
        "fraction_modules_positive_all_available_cohorts": float(np.mean(all_pos)) if all_pos else np.nan,
        "median_module_cross_cohort_sd": float(np.median(sds)) if sds else np.nan,
    }
    return summary, detail


def consensus_within_module_stats(consensus: pd.DataFrame, labels: Sequence[int]) -> Tuple[dict, pd.DataFrame]:
    labels = np.asarray(labels)
    rows = []
    for lab in sorted(np.unique(labels)):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            vals = np.array([], dtype=float)
        else:
            sub = consensus.iloc[idx, idx].to_numpy(dtype=float)
            iu = np.triu_indices(len(idx), k=1)
            vals = sub[iu]
            vals = vals[np.isfinite(vals)]
        rows.append({
            "raw_cluster_id": int(lab),
            "module_size": int(len(idx)),
            "mean_consensus_rho": float(np.mean(vals)) if len(vals) else np.nan,
            "median_consensus_rho": float(np.median(vals)) if len(vals) else np.nan,
            "fraction_positive_pairs": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "n_pairs": int(len(vals)),
        })
    detail = pd.DataFrame(rows)
    usable = detail[detail["module_size"] >= 2]
    return {
        "median_module_mean_consensus_rho": float(usable["mean_consensus_rho"].median()) if len(usable) else np.nan,
        "q25_module_mean_consensus_rho": float(usable["mean_consensus_rho"].quantile(0.25)) if len(usable) else np.nan,
        "median_module_median_consensus_rho": float(usable["median_consensus_rho"].median()) if len(usable) else np.nan,
        "q25_module_median_consensus_rho": float(usable["median_consensus_rho"].quantile(0.25)) if len(usable) else np.nan,
        "median_module_fraction_positive_pairs": float(usable["fraction_positive_pairs"].median()) if len(usable) else np.nan,
        "q25_module_fraction_positive_pairs": float(usable["fraction_positive_pairs"].quantile(0.25)) if len(usable) else np.nan,
        "fraction_modules_positive_mean_consensus": float((usable["mean_consensus_rho"] > 0).mean()) if len(usable) else np.nan,
    }, detail


# =============================================================================
# K grid
# =============================================================================

def evaluate_k_grid(
    *,
    panel: str,
    root: str,
    consensus: pd.DataFrame,
    universe: pd.DataFrame,
    pairwise_audit: pd.DataFrame,
    distance_mode: str,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    parser_path: Optional[str | Path] = None,
    n_semantic_permutations: int = DEFAULT_N_SEMANTIC_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    score_weights: Optional[Mapping[str, float]] = None,
    review_thresholds: Optional[Mapping[str, float]] = None,
    primary_k_max: int = DEFAULT_PRIMARY_SELECTION_K_MAX,
    selection_within_best_fraction: float = DEFAULT_SELECTION_WITHIN_BEST_FRACTION,
) -> Dict[str, object]:
    features = consensus.index.astype(str).tolist()
    if len(features) < 3:
        raise RuntimeError(f"{panel}/{root}: only {len(features)} support-filtered features; K grid not meaningful")

    ontology, sims = build_root_semantic_ontology(universe, features, parser_path=parser_path)
    D_stat = build_distance_matrix(consensus, distance_mode)
    Z = linkage_from_distance(D_stat, method=linkage_method)
    if Z is None:
        raise RuntimeError(f"{panel}/{root}: linkage could not be built")

    semantic_distances = {}
    for name in ("primitive", "tissue_aware", "metric_aware"):
        D = 1.0 - sims[name]
        D = D.clip(lower=0.0, upper=1.0)
        np.fill_diagonal(D.values, 0.0)
        semantic_distances[name] = D

    max_possible = min(int(k_max), len(features) - 1)
    min_possible = max(2, int(k_min))
    if max_possible < min_possible:
        min_possible = 2

    rows = []
    memberships = []
    module_rows = []
    repro_rows = []

    for k in range(min_possible, max_possible + 1):
        labels = _labels_for_k(Z, k)
        size_stats = _cluster_size_stats(labels)
        stat_sil = _safe_silhouette(D_stat, labels)

        sem_sil = {
            name: _safe_silhouette(semantic_distances[name], labels)
            for name in ("primitive", "tissue_aware", "metric_aware")
        }
        sem_coh = {
            name: semantic_coherence_stats(
                sims[name], labels,
                n_permutations=n_semantic_permutations,
                random_seed=int(random_seed) + 1000 * k + ii,
            )
            for ii, name in enumerate(("primitive", "tissue_aware", "metric_aware"))
        }

        cons_summary, cons_detail = consensus_within_module_stats(consensus, labels)
        repro_summary, repro_detail = cross_cohort_reproducibility(pairwise_audit, labels, features)

        row = {
            "panel": panel,
            "feature_source": root,
            "distance_mode": distance_mode,
            "requested_k": int(k),
            "n_features": int(len(features)),
            "stat_silhouette": stat_sil,
            "primitive_semantic_silhouette": sem_sil["primitive"],
            "tissue_aware_semantic_silhouette": sem_sil["tissue_aware"],
            "metric_aware_semantic_silhouette": sem_sil["metric_aware"],
            **size_stats,
            **cons_summary,
            **repro_summary,
        }
        for name in ("primitive", "tissue_aware", "metric_aware"):
            prefix = {
                "primitive": "primitive",
                "tissue_aware": "tissue_aware",
                "metric_aware": "metric_aware",
            }[name]
            for stat_name, val in sem_coh[name].items():
                row[f"{prefix}_coherence_{stat_name}"] = val
        rows.append(row)

        for f, lab in zip(features, labels):
            memberships.append({
                "panel": panel,
                "feature_source": root,
                "distance_mode": distance_mode,
                "requested_k": int(k),
                "feature_uid": f,
                "raw_cluster_id": int(lab),
            })

        # Module-level diagnostics for all three semantic layers + consensus.
        for sem_name in ("primitive", "tissue_aware", "metric_aware"):
            md = module_semantic_coherence(sims[sem_name], labels)
            for _, rr in md.iterrows():
                module_rows.append({
                    "panel": panel,
                    "feature_source": root,
                    "distance_mode": distance_mode,
                    "requested_k": int(k),
                    "semantic_layer": sem_name,
                    **rr.to_dict(),
                })
        if not cons_detail.empty:
            for _, rr in cons_detail.iterrows():
                module_rows.append({
                    "panel": panel,
                    "feature_source": root,
                    "distance_mode": distance_mode,
                    "requested_k": int(k),
                    "semantic_layer": "statistical_consensus",
                    **rr.to_dict(),
                })
        if not repro_detail.empty:
            tmp = repro_detail.copy()
            tmp["panel"] = panel
            tmp["feature_source"] = root
            tmp["distance_mode"] = distance_mode
            tmp["requested_k"] = int(k)
            repro_rows.append(tmp)

    kdiag = pd.DataFrame(rows)
    mem = pd.DataFrame(memberships)
    module_diag = pd.DataFrame(module_rows)
    repro_detail_all = pd.concat(repro_rows, ignore_index=True, sort=False) if repro_rows else pd.DataFrame()

    weights = dict(DEFAULT_K_SCORE_WEIGHTS)
    if score_weights:
        weights.update({k: float(v) for k, v in score_weights.items()})
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError(f"K score weights must sum to 1; got {weights} sum={sum(weights.values())}")

    thresholds = dict(DEFAULT_REVIEW_THRESHOLDS)
    if review_thresholds:
        thresholds.update({k: float(v) for k, v in review_thresholds.items()})

    if not kdiag.empty:
        kdiag["stat_rank"] = rank01(kdiag["stat_silhouette"], higher_better=True)
        kdiag["primitive_coherence_rank"] = rank01(kdiag["primitive_coherence_enrichment"], higher_better=True)
        kdiag["tissue_coherence_rank"] = rank01(kdiag["tissue_aware_coherence_enrichment"], higher_better=True)
        kdiag["metric_coherence_rank"] = rank01(kdiag["metric_aware_coherence_enrichment"], higher_better=True)
        kdiag["cluster_balance_score"] = 1.0 - kdiag["max_cluster_fraction"]
        kdiag["non_singleton_score"] = 1.0 - kdiag["singleton_fraction"]
        kdiag["balance_rank"] = rank01(kdiag["cluster_balance_score"], higher_better=True)
        kdiag["non_singleton_rank"] = rank01(kdiag["non_singleton_score"], higher_better=True)

        kdiag["k_composite_score"] = (
            weights["stat_silhouette"] * kdiag["stat_rank"].fillna(0.0)
            + weights["primitive_coherence"] * kdiag["primitive_coherence_rank"].fillna(0.0)
            + weights["tissue_coherence"] * kdiag["tissue_coherence_rank"].fillna(0.0)
            + weights["metric_coherence"] * kdiag["metric_coherence_rank"].fillna(0.0)
            + weights["cluster_balance"] * kdiag["balance_rank"].fillna(0.0)
            + weights["non_singleton"] * kdiag["non_singleton_rank"].fillna(0.0)
        )

        kdiag["passes_size_review"] = (
            (kdiag["max_cluster_fraction"] <= thresholds["max_cluster_fraction"])
            & (kdiag["singleton_fraction"] <= thresholds["singleton_fraction"])
        )
        kdiag["passes_direct_cohesion_review"] = (
            (kdiag["q25_module_mean_consensus_rho"] >= thresholds["min_q25_module_mean_consensus_rho"])
            & (kdiag["median_module_fraction_positive_pairs"] >= thresholds["min_median_module_fraction_positive_pairs"])
        )
        kdiag["review_warning_large_cluster"] = (
            kdiag["max_cluster_fraction"] > thresholds["max_cluster_fraction"]
        )
        kdiag["review_warning_singletons"] = (
            kdiag["singleton_fraction"] > thresholds["singleton_fraction"]
        )
        kdiag["review_warning_weak_direct_cohesion"] = ~kdiag["passes_direct_cohesion_review"]
        best = float(kdiag["k_composite_score"].max())
        kdiag["within_best_fraction"] = (
            kdiag["k_composite_score"] >= thresholds["within_best_fraction"] * best
        )
        kdiag["within_best_and_size_healthy"] = kdiag["within_best_fraction"] & kdiag["passes_size_review"]

        # Re-rank the same components inside the prespecified parsimonious
        # primary range (normally K<=10). Larger K remain diagnostic only.
        kdiag = add_primary_k_selection_scores(
            kdiag,
            score_weights=weights,
            primary_k_max=primary_k_max,
            within_best_fraction=selection_within_best_fraction,
        )

    return {
        "k_diagnostics": kdiag,
        "memberships": mem,
        "module_diagnostics": module_diag,
        "cross_cohort_detail": repro_detail_all,
        "distance": D_stat,
        "linkage": Z,
        "ontology": ontology,
        "semantic_matrices": sims,
        "score_weights": weights,
        "review_thresholds": thresholds,
    }


# =============================================================================
# Parsimonious primary K selection
# =============================================================================

def add_primary_k_selection_scores(
    kdiag: pd.DataFrame,
    *,
    score_weights: Optional[Mapping[str, float]] = None,
    primary_k_max: int = DEFAULT_PRIMARY_SELECTION_K_MAX,
    within_best_fraction: float = DEFAULT_SELECTION_WITHIN_BEST_FRACTION,
) -> pd.DataFrame:
    """Add a K<=primary_k_max selection score without discarding larger-K diagnostics.

    The full K grid remains available for sensitivity/visualization.  For the
    *primary* K choice, component percentile ranks are recalculated only over
    the prespecified parsimonious range (normally K=2..10).  This prevents
    mechanically increasing semantic coherence at K=20-30 from dominating the
    ranking used for the manuscript-facing module solution.

    The recommended K is subsequently the smallest K within
    `within_best_fraction` of the best primary-range score, preferentially
    satisfying the module-size review guardrails.
    """
    d = kdiag.copy()
    weights = dict(DEFAULT_K_SCORE_WEIGHTS)
    if score_weights:
        weights.update({k: float(v) for k, v in score_weights.items()})

    d["primary_k_selection_eligible"] = (
        pd.to_numeric(d["requested_k"], errors="coerce") <= int(primary_k_max)
    )

    # Initialize columns for all K; only the primary range gets values.
    cols = [
        "primary_stat_rank",
        "primary_primitive_coherence_rank",
        "primary_tissue_coherence_rank",
        "primary_metric_coherence_rank",
        "primary_balance_rank",
        "primary_non_singleton_rank",
        "k_primary_selection_score",
    ]
    for c in cols:
        d[c] = np.nan

    mask = d["primary_k_selection_eligible"].fillna(False)
    if mask.any():
        z = d.loc[mask].copy()
        d.loc[mask, "primary_stat_rank"] = rank01(z["stat_silhouette"], True).values
        d.loc[mask, "primary_primitive_coherence_rank"] = rank01(
            z["primitive_coherence_enrichment"], True
        ).values
        d.loc[mask, "primary_tissue_coherence_rank"] = rank01(
            z["tissue_aware_coherence_enrichment"], True
        ).values
        d.loc[mask, "primary_metric_coherence_rank"] = rank01(
            z["metric_aware_coherence_enrichment"], True
        ).values
        d.loc[mask, "primary_balance_rank"] = rank01(
            z["cluster_balance_score"], True
        ).values
        d.loc[mask, "primary_non_singleton_rank"] = rank01(
            z["non_singleton_score"], True
        ).values

        d.loc[mask, "k_primary_selection_score"] = (
            weights["stat_silhouette"] * d.loc[mask, "primary_stat_rank"].fillna(0.0)
            + weights["primitive_coherence"] * d.loc[mask, "primary_primitive_coherence_rank"].fillna(0.0)
            + weights["tissue_coherence"] * d.loc[mask, "primary_tissue_coherence_rank"].fillna(0.0)
            + weights["metric_coherence"] * d.loc[mask, "primary_metric_coherence_rank"].fillna(0.0)
            + weights["cluster_balance"] * d.loc[mask, "primary_balance_rank"].fillna(0.0)
            + weights["non_singleton"] * d.loc[mask, "primary_non_singleton_rank"].fillna(0.0)
        )

        best = pd.to_numeric(
            d.loc[mask, "k_primary_selection_score"], errors="coerce"
        ).max()
        d["within_primary_best_fraction"] = False
        if np.isfinite(best):
            d.loc[mask, "within_primary_best_fraction"] = (
                d.loc[mask, "k_primary_selection_score"]
                >= float(within_best_fraction) * float(best)
            )
    else:
        d["within_primary_best_fraction"] = False

    d["within_primary_best_and_size_healthy"] = (
        d["within_primary_best_fraction"].fillna(False)
        & d["passes_size_review"].fillna(False)
    )
    d["primary_selection_k_max"] = int(primary_k_max)
    d["primary_selection_within_best_fraction"] = float(within_best_fraction)
    return d


# =============================================================================
# Recommendation summaries
# =============================================================================

def summarize_k_recommendation(kdiag: pd.DataFrame) -> dict:
    if kdiag.empty:
        return {}
    d = kdiag.sort_values("requested_k").copy()

    # Full-grid best is retained as an extended diagnostic.
    best_full = d.sort_values(
        ["k_composite_score", "requested_k"], ascending=[False, True]
    ).iloc[0]

    # Primary manuscript-facing selection is restricted to the prespecified
    # parsimonious range, normally K<=10.
    if "primary_k_selection_eligible" in d.columns:
        primary = d[d["primary_k_selection_eligible"].fillna(False)].copy()
    else:
        primary = d.copy()

    score_col = (
        "k_primary_selection_score"
        if "k_primary_selection_score" in primary.columns
        and primary["k_primary_selection_score"].notna().any()
        else "k_composite_score"
    )
    best_primary = primary.sort_values(
        [score_col, "requested_k"], ascending=[False, True]
    ).iloc[0]

    if "within_primary_best_fraction" in primary.columns:
        near = primary[primary["within_primary_best_fraction"].fillna(False)].copy()
        near_healthy = primary[
            primary.get("within_primary_best_and_size_healthy", False)
        ].copy()
    else:
        near = primary[primary["within_best_fraction"]].copy()
        near_healthy = primary[primary["within_best_and_size_healthy"]].copy()

    if not near_healthy.empty:
        pars = near_healthy.sort_values("requested_k").iloc[0]
        pars_reason = "smallest_K_within_90pct_primary_best_and_size_healthy"
    elif not near.empty:
        pars = near.sort_values("requested_k").iloc[0]
        pars_reason = "smallest_K_within_90pct_primary_best_no_size_healthy_candidate"
    else:
        pars = best_primary
        pars_reason = "best_primary_range_score_only"

    return {
        "panel": best_primary["panel"],
        "feature_source": best_primary["feature_source"],
        "distance_mode": best_primary["distance_mode"],
        "n_features": int(best_primary["n_features"]),
        "extended_grid_best_k": int(best_full["requested_k"]),
        "extended_grid_best_score": float(best_full["k_composite_score"]),
        "primary_selection_k_max": int(best_primary.get("primary_selection_k_max", primary["requested_k"].max())),
        "best_score_k": int(best_primary["requested_k"]),
        "best_score": float(best_primary[score_col]),
        "recommended_parsimonious_k": int(pars["requested_k"]),
        "recommendation_reason": pars_reason,
        "recommended_score": float(pars[score_col]),
        "recommended_stat_silhouette": float(pars["stat_silhouette"]) if pd.notna(pars["stat_silhouette"]) else np.nan,
        "recommended_primitive_coherence_enrichment": float(pars["primitive_coherence_enrichment"]) if pd.notna(pars["primitive_coherence_enrichment"]) else np.nan,
        "recommended_tissue_coherence_enrichment": float(pars["tissue_aware_coherence_enrichment"]) if pd.notna(pars["tissue_aware_coherence_enrichment"]) else np.nan,
        "recommended_metric_coherence_enrichment": float(pars["metric_aware_coherence_enrichment"]) if pd.notna(pars["metric_aware_coherence_enrichment"]) else np.nan,
        "recommended_max_cluster_fraction": float(pars["max_cluster_fraction"]),
        "recommended_singleton_fraction": float(pars["singleton_fraction"]),
        "recommended_q25_module_mean_consensus_rho": float(pars["q25_module_mean_consensus_rho"]) if pd.notna(pars["q25_module_mean_consensus_rho"]) else np.nan,
        "recommended_median_module_fraction_positive_pairs": float(pars["median_module_fraction_positive_pairs"]) if pd.notna(pars["median_module_fraction_positive_pairs"]) else np.nan,
        "recommended_passes_direct_cohesion_review": bool(pars["passes_direct_cohesion_review"]) if pd.notna(pars["passes_direct_cohesion_review"]) else False,
        "recommended_cross_cohort_median_rho": float(pars["median_module_cohort_mean_rho"]) if pd.notna(pars["median_module_cohort_mean_rho"]) else np.nan,
        "recommended_fraction_module_cohort_positive": float(pars["fraction_module_cohort_positive"]) if pd.notna(pars["fraction_module_cohort_positive"]) else np.nan,
    }


# =============================================================================
# Plotting
# =============================================================================

def _plot_lines(
    d: pd.DataFrame,
    ycols: Sequence[str],
    labels: Sequence[str],
    *,
    title: str,
    ylabel: str,
    path: Path,
    hline: Optional[float] = None,
):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for y, label in zip(ycols, labels):
        ax.plot(d["requested_k"], d[y], marker="o", label=label)
    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1)
    ax.set_xlabel("K (number of root modules)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_k_diagnostics(
    result: Mapping[str, object],
    output_dir: str | Path,
    *,
    panel: str,
    root: str,
    distance_mode: str,
) -> None:
    out = ensure_dir(output_dir)
    d = result["k_diagnostics"].sort_values("requested_k").copy()
    if d.empty:
        return

    prefix = f"{panel} | {root} | {distance_mode}"

    _plot_lines(
        d,
        ["stat_silhouette"],
        ["Statistical silhouette"],
        title=f"{prefix}: statistical silhouette",
        ylabel="Silhouette",
        path=out / "01_statistical_silhouette.png",
        hline=0.0,
    )

    _plot_lines(
        d,
        [
            "primitive_semantic_silhouette",
            "tissue_aware_semantic_silhouette",
            "metric_aware_semantic_silhouette",
        ],
        ["Primitive", "Tissue-aware", "Metric-aware"],
        title=f"{prefix}: semantic silhouette (diagnostic only)",
        ylabel="Semantic silhouette",
        path=out / "02_semantic_silhouette.png",
        hline=0.0,
    )

    # Raw coherence with its exact random-partition expectation.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    specs = [
        ("primitive", "Primitive"),
        ("tissue_aware", "Tissue-aware"),
        ("metric_aware", "Metric-aware"),
    ]
    for ax, (pfx, lab) in zip(axes, specs):
        ax.plot(d["requested_k"], d[f"{pfx}_coherence_raw"], marker="o", label="Observed")
        ax.plot(d["requested_k"], d[f"{pfx}_coherence_null_mean"], linestyle="--", label="Random-partition expectation")
        ax.set_title(lab)
        ax.set_xlabel("K")
        ax.set_ylabel("Within-module semantic similarity")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{prefix}: semantic coherence")
    fig.tight_layout()
    fig.savefig(out / "03_semantic_coherence_raw_vs_null.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    _plot_lines(
        d,
        [
            "primitive_coherence_enrichment",
            "tissue_aware_coherence_enrichment",
            "metric_aware_coherence_enrichment",
        ],
        ["Primitive", "Tissue-aware", "Metric-aware"],
        title=f"{prefix}: semantic coherence enrichment",
        ylabel="Observed coherence − random expectation",
        path=out / "04_semantic_coherence_enrichment.png",
        hline=0.0,
    )

    _plot_lines(
        d,
        [
            "primitive_coherence_z",
            "tissue_aware_coherence_z",
            "metric_aware_coherence_z",
        ],
        ["Primitive", "Tissue-aware", "Metric-aware"],
        title=f"{prefix}: semantic coherence permutation Z",
        ylabel="Permutation Z-score",
        path=out / "05_semantic_coherence_z.png",
        hline=0.0,
    )

    _plot_lines(
        d,
        ["max_cluster_fraction", "singleton_fraction"],
        ["Largest-module fraction", "Singleton-module fraction"],
        title=f"{prefix}: module-size health",
        ylabel="Fraction",
        path=out / "06_module_size_health.png",
    )

    _plot_lines(
        d,
        [
            "median_module_mean_consensus_rho",
            "q25_module_mean_consensus_rho",
        ],
        ["Median module mean consensus rho", "25th percentile module mean consensus rho"],
        title=f"{prefix}: within-module consensus correlation",
        ylabel="Signed consensus Spearman rho",
        path=out / "07_within_module_consensus_rho.png",
        hline=0.0,
    )

    _plot_lines(
        d,
        [
            "median_module_cohort_mean_rho",
            "q25_module_cohort_mean_rho",
        ],
        ["Median cohort × module mean rho", "25th percentile cohort × module mean rho"],
        title=f"{prefix}: cross-cohort module reproducibility",
        ylabel="Within-module Spearman rho",
        path=out / "08_cross_cohort_reproducibility.png",
        hline=0.0,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(d["requested_k"], d["k_composite_score"], marker="o", alpha=0.45, label="Extended-grid score")
    if "k_primary_selection_score" in d.columns:
        dp = d[d["primary_k_selection_eligible"].fillna(False)]
        ax.plot(dp["requested_k"], dp["k_primary_selection_score"], marker="o", linewidth=2, label="Primary K≤10 score")
    rec = summarize_k_recommendation(d)
    ax.axvline(int(rec.get("primary_selection_k_max", 10)), linestyle="--", linewidth=1, label=f"Primary K max={int(rec.get('primary_selection_k_max', 10))}")
    ax.axvline(int(rec["recommended_parsimonious_k"]), linestyle=":", linewidth=1.8, label=f"Parsimonious K={int(rec['recommended_parsimonious_k'])}")
    ax.set_xlabel("K")
    ax.set_ylabel("Rank-normalized K score")
    ax.set_title(f"{prefix}: overall K score\n(primary selection constrained to K≤10)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "09_k_composite_score.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Compact dashboard for rapid review.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(d["requested_k"], d["stat_silhouette"], marker="o", label="Statistical silhouette")
    ax.set_title("Statistical structure")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.axhline(0, linestyle="--", linewidth=0.8)

    ax = axes[0, 1]
    for y, lab in [
        ("primitive_coherence_enrichment", "Primitive"),
        ("tissue_aware_coherence_enrichment", "Tissue-aware"),
        ("metric_aware_coherence_enrichment", "Metric-aware (diagnostic)"),
    ]:
        ax.plot(d["requested_k"], d[y], marker="o", label=lab)
    ax.set_title("Semantic coherence enrichment")
    ax.set_xlabel("K"); ax.set_ylabel("Observed − random expectation"); ax.axhline(0, linestyle="--", linewidth=0.8); ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(d["requested_k"], d["max_cluster_fraction"], marker="o", label="Largest module")
    ax.plot(d["requested_k"], d["singleton_fraction"], marker="o", label="Singleton modules")
    ax.set_title("Module-size health")
    ax.set_xlabel("K"); ax.set_ylabel("Fraction"); ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(d["requested_k"], d["k_composite_score"], marker="o", label="Composite")
    ax.plot(d["requested_k"], rank01(d["q25_module_mean_consensus_rho"], True), marker="o", label="Direct-cohesion rank")
    ax.plot(d["requested_k"], rank01(d["median_module_cohort_mean_rho"], True), marker="o", label="Reproducibility rank")
    ax.set_title("Composite + diagnostic guardrails")
    ax.set_xlabel("K"); ax.set_ylabel("Relative score/rank"); ax.legend(frameon=False)

    fig.suptitle(prefix)
    fig.tight_layout()
    fig.savefig(out / "00_k_review_dashboard.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_matrices(result: Mapping[str, object], output_dir: str | Path, *, panel: str, root: str) -> None:
    out = ensure_dir(output_dir)
    for ii, name in enumerate(("primitive", "tissue_aware", "metric_aware"), start=1):
        M = result["semantic_matrices"][name]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(M.to_numpy(dtype=float), vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{panel} | {root}: {name.replace('_', ' ')} semantic similarity")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"{ii:02d}_{name}_semantic_similarity.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_dendrogram(result: Mapping[str, object], output_path: str | Path, *, panel: str, root: str, distance_mode: str) -> None:
    Z = result["linkage"]
    features = result["distance"].index.astype(str).tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.12), 5.5))
    dendrogram(Z, labels=features, leaf_rotation=90, leaf_font_size=5, ax=ax)
    ax.set_title(f"{panel} | {root} | {distance_mode}: hierarchical dendrogram")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _module_order(labels: np.ndarray, Z) -> Tuple[List[int], Dict[int, int]]:
    leaves = list(leaves_list(Z))
    # Keep dendrogram order but group modules contiguously according to first appearance.
    mod_first = []
    for i in leaves:
        lab = int(labels[i])
        if lab not in mod_first:
            mod_first.append(lab)
    order = []
    for lab in mod_first:
        order.extend([i for i in leaves if int(labels[i]) == lab])
    remap = {lab: ii + 1 for ii, lab in enumerate(mod_first)}
    return order, remap


def plot_k_heatmap(
    consensus: pd.DataFrame,
    result: Mapping[str, object],
    k: int,
    output_path: str | Path,
    *,
    panel: str,
    root: str,
    distance_mode: str,
) -> pd.DataFrame:
    features = consensus.index.astype(str).tolist()
    labels = _labels_for_k(result["linkage"], int(k))
    order_idx, remap = _module_order(labels, result["linkage"])
    ordered_features = [features[i] for i in order_idx]
    ordered_labels = np.asarray([remap[int(labels[i])] for i in order_idx])
    M = consensus.loc[ordered_features, ordered_features]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M.to_numpy(dtype=float), vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"{panel} | {root} | {distance_mode} | K={k}\nSupport-filtered signed Spearman consensus")
    ax.set_xticks([]); ax.set_yticks([])
    # Module boundaries.
    changes = np.where(np.diff(ordered_labels) != 0)[0] + 1
    for pos in changes:
        ax.axhline(pos - 0.5, linewidth=1)
        ax.axvline(pos - 0.5, linewidth=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({
        "feature_uid": ordered_features,
        "raw_cluster_id": [int(labels[i]) for i in order_idx],
        "module_num_ordered": ordered_labels.astype(int),
    })


def plot_annotated_k_heatmap(
    consensus: pd.DataFrame,
    result: Mapping[str, object],
    k: int,
    output_path: str | Path,
    *,
    panel: str,
    root: str,
    distance_mode: str,
    show_feature_labels: bool = False,
    module_prefix: str = "M",
    figsize: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Plot a red/blue signed-consensus heatmap with a module bar on top.

    Features are ordered by dendrogram leaves and then grouped contiguously by
    the selected K solution. Positive consensus rho is red, negative is blue,
    and zero is white. The top categorical strip marks module membership and
    labels modules M01, M02, ... in heatmap order.
    """
    features = consensus.index.astype(str).tolist()
    labels = _labels_for_k(result["linkage"], int(k))
    order_idx, remap = _module_order(labels, result["linkage"])
    ordered_features = [features[i] for i in order_idx]
    ordered_labels = np.asarray([remap[int(labels[i])] for i in order_idx], dtype=int)
    M = consensus.loc[ordered_features, ordered_features]

    n = len(ordered_features)
    if figsize is None:
        side = min(max(8.0, 0.16 * n + 5.5), 15.0)
        figsize = (side, side + 1.0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.55, 12],
        width_ratios=[12, 0.45],
        hspace=0.04, wspace=0.08,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Signed consensus: blue negative, white zero, red positive.
    im = ax.imshow(
        M.to_numpy(dtype=float),
        vmin=-1, vmax=1,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Signed consensus Spearman ρ")

    unique_mods = list(dict.fromkeys(ordered_labels.tolist()))
    base_cmap = plt.get_cmap("tab20")
    mod_colors = [base_cmap(i % base_cmap.N) for i in range(len(unique_mods))]
    mod_to_idx = {m: i for i, m in enumerate(unique_mods)}
    bar_values = np.asarray([[mod_to_idx[m] for m in ordered_labels]], dtype=int)
    ax_bar.imshow(
        bar_values,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(mod_colors),
        vmin=-0.5,
        vmax=max(len(unique_mods) - 0.5, 0.5),
    )
    ax_bar.set_xlim(ax.get_xlim())
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Boundaries and module labels.
    changes = np.where(np.diff(ordered_labels) != 0)[0] + 1
    for boundary in changes:
        x = boundary - 0.5
        ax.axhline(x, color="black", linewidth=0.8)
        ax.axvline(x, color="black", linewidth=0.8)
        ax_bar.axvline(x, color="white", linewidth=1.0)

    starts = np.r_[0, changes]
    ends = np.r_[changes, n]
    for module_num, (start, end) in enumerate(zip(starts, ends), start=1):
        center = (start + end - 1) / 2.0
        ax_bar.text(
            center, -0.75,
            f"{module_prefix}{module_num:02d}",
            ha="center", va="bottom",
            fontsize=8, rotation=0,
            clip_on=False,
        )

    if show_feature_labels:
        fs = 5 if n > 30 else 6
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(ordered_features, rotation=90, fontsize=fs)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(ordered_features, fontsize=fs)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(
        f"{panel} | {root} | {distance_mode} | K={int(k)}\n"
        "Support-filtered signed Spearman consensus",
        pad=12,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({
        "feature_uid": ordered_features,
        "raw_cluster_id": [int(labels[i]) for i in order_idx],
        "module_num_ordered": ordered_labels.astype(int),
        "module_label": [f"{module_prefix}{m:02d}" for m in ordered_labels],
    })


def plot_primary_k_yield(
    kdiag: pd.DataFrame,
    output_path: str | Path,
    *,
    panel: str,
    root: str,
    distance_mode: str,
    primary_k_max: int = DEFAULT_PRIMARY_SELECTION_K_MAX,
) -> None:
    """Show core K-selection yields only over the prespecified primary range."""
    d = kdiag[
        pd.to_numeric(kdiag["requested_k"], errors="coerce") <= int(primary_k_max)
    ].sort_values("requested_k").copy()
    if d.empty:
        return
    rec = summarize_k_recommendation(kdiag)
    rk = int(rec["recommended_parsimonious_k"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax = axes[0, 0]
    ax.plot(d["requested_k"], d["stat_silhouette"], marker="o")
    ax.set_title("Statistical silhouette")
    ax.set_ylabel("Silhouette")

    ax = axes[0, 1]
    ax.plot(d["requested_k"], d["primitive_coherence_enrichment"], marker="o", label="Primitive")
    ax.plot(d["requested_k"], d["tissue_aware_coherence_enrichment"], marker="o", label="Tissue-aware")
    ax.set_title("Semantic coherence enrichment")
    ax.set_ylabel("Observed − random expectation")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(d["requested_k"], d["max_cluster_fraction"], marker="o", label="Largest module")
    ax.plot(d["requested_k"], d["singleton_fraction"], marker="o", label="Singleton modules")
    ax.set_title("Module-size health")
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    score_col = "k_primary_selection_score" if "k_primary_selection_score" in d.columns else "k_composite_score"
    ax.plot(d["requested_k"], d[score_col], marker="o", label="Primary selection score")
    ax.axvline(rk, linestyle="--", linewidth=1.2, label=f"Parsimonious K={rk}")
    ax.set_title("Parsimonious primary K score")
    ax.set_ylabel("Rank-normalized score")
    ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.set_xlabel("K")
        ax.set_xticks(d["requested_k"].astype(int).tolist())
    fig.suptitle(f"{panel} | {root} | {distance_mode}: primary K range (K≤{int(primary_k_max)})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_top_k_heatmaps(
    consensus: pd.DataFrame,
    result: Mapping[str, object],
    output_dir: str | Path,
    *,
    panel: str,
    root: str,
    distance_mode: str,
    n_top: int = 3,
) -> List[int]:
    out = ensure_dir(output_dir)
    d = result["k_diagnostics"].sort_values(["k_composite_score", "requested_k"], ascending=[False, True])
    ks = d["requested_k"].head(int(n_top)).astype(int).tolist()
    rec = summarize_k_recommendation(result["k_diagnostics"])
    pk = int(rec["recommended_parsimonious_k"])
    if pk not in ks:
        ks = [pk] + ks[: max(0, int(n_top) - 1)]
    ks = list(dict.fromkeys(ks))
    for k in ks:
        plot_k_heatmap(
            consensus, result, k, out / f"K{k:02d}_consensus_heatmap.png",
            panel=panel, root=root, distance_mode=distance_mode,
        )
    return ks


# =============================================================================
# Root runner / full notebook workflow
# =============================================================================

def evaluate_one_root(
    *,
    stage2b1_root: str | Path,
    manifest_row: Mapping[str, object],
    output_root: str | Path,
    parser_path: Optional[str | Path],
    distance_modes: Sequence[str] = DEFAULT_DISTANCE_MODES,
    primary_mode: str = DEFAULT_PRIMARY_MODE,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    n_semantic_permutations: int = DEFAULT_N_SEMANTIC_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    score_weights: Optional[Mapping[str, float]] = None,
    review_thresholds: Optional[Mapping[str, float]] = None,
    primary_k_max: int = DEFAULT_PRIMARY_SELECTION_K_MAX,
    selection_within_best_fraction: float = DEFAULT_SELECTION_WITHIN_BEST_FRACTION,
    top_heatmaps: int = 3,
) -> Dict[str, object]:
    panel = str(manifest_row["panel"])
    root = str(manifest_row["feature_source"])
    inputs = load_root_inputs(stage2b1_root, manifest_row)
    consensus = inputs["consensus"]
    universe = inputs["universe"]
    audit = inputs["pairwise_audit"]

    root_out = ensure_dir(Path(output_root) / f"{safe_slug(panel)}__{safe_slug(root)}")
    mode_results = {}
    rec_rows = []

    for mode in distance_modes:
        mdir = ensure_dir(root_out / mode)
        res = evaluate_k_grid(
            panel=panel,
            root=root,
            consensus=consensus,
            universe=universe,
            pairwise_audit=audit,
            distance_mode=mode,
            k_min=k_min,
            k_max=k_max,
            linkage_method=linkage_method,
            parser_path=parser_path,
            n_semantic_permutations=n_semantic_permutations,
            random_seed=random_seed,
            score_weights=score_weights,
            review_thresholds=review_thresholds,
            primary_k_max=primary_k_max,
            selection_within_best_fraction=selection_within_best_fraction,
        )
        mode_results[mode] = res

        res["k_diagnostics"].to_csv(mdir / "k_diagnostics.csv", index=False)
        res["memberships"].to_csv(mdir / "memberships_all_k.csv.gz", index=False, compression="gzip")
        res["module_diagnostics"].to_csv(mdir / "module_diagnostics_all_k.csv.gz", index=False, compression="gzip")
        if not res["cross_cohort_detail"].empty:
            res["cross_cohort_detail"].to_csv(mdir / "cross_cohort_reproducibility_all_k.csv.gz", index=False, compression="gzip")
        res["ontology"].to_csv(mdir / "feature_semantic_ontology.csv", index=False)

        plot_k_diagnostics(res, mdir / "plots", panel=panel, root=root, distance_mode=mode)
        plot_primary_k_yield(
            res["k_diagnostics"],
            mdir / "plots" / "11_primary_k_yield.png",
            panel=panel, root=root, distance_mode=mode,
            primary_k_max=primary_k_max,
        )
        plot_dendrogram(res, mdir / "plots" / "10_dendrogram.png", panel=panel, root=root, distance_mode=mode)
        if mode == primary_mode:
            plot_semantic_matrices(res, mdir / "plots" / "semantic_matrices", panel=panel, root=root)
        ks = plot_top_k_heatmaps(
            consensus, res, mdir / "plots" / "top_k_heatmaps",
            panel=panel, root=root, distance_mode=mode, n_top=top_heatmaps,
        )

        rec = summarize_k_recommendation(res["k_diagnostics"])
        rec["top_heatmap_ks"] = ";".join(map(str, ks))
        rec_rows.append(rec)
        pd.DataFrame([rec]).to_csv(mdir / "k_recommendation.csv", index=False)

    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(root_out / "distance_mode_recommendations.csv", index=False)

    # Root-level manual review row: primary mode recommendation plus sensitivity.
    primary = rec_df[rec_df["distance_mode"].astype(str).eq(str(primary_mode))]
    if primary.empty:
        primary = rec_df.iloc[[0]]
    pr = primary.iloc[0].to_dict()
    sens = rec_df[~rec_df.index.isin(primary.index)]
    review = {
        "panel": panel,
        "feature_source": root,
        "n_features": int(consensus.shape[0]),
        "primary_distance_mode": str(primary_mode),
        "primary_best_score_k": pr.get("best_score_k"),
        "primary_recommended_k": pr.get("recommended_parsimonious_k"),
        "primary_recommended_score": pr.get("recommended_score"),
        "primary_stat_silhouette": pr.get("recommended_stat_silhouette"),
        "primary_primitive_coherence_enrichment": pr.get("recommended_primitive_coherence_enrichment"),
        "primary_tissue_coherence_enrichment": pr.get("recommended_tissue_coherence_enrichment"),
        "primary_metric_coherence_enrichment": pr.get("recommended_metric_coherence_enrichment"),
        "primary_max_cluster_fraction": pr.get("recommended_max_cluster_fraction"),
        "primary_singleton_fraction": pr.get("recommended_singleton_fraction"),
        "primary_q25_module_mean_consensus_rho": pr.get("recommended_q25_module_mean_consensus_rho"),
        "primary_median_module_fraction_positive_pairs": pr.get("recommended_median_module_fraction_positive_pairs"),
        "primary_passes_direct_cohesion_review": pr.get("recommended_passes_direct_cohesion_review"),
        "primary_cross_cohort_median_rho": pr.get("recommended_cross_cohort_median_rho"),
        "sensitivity_mode_recommended_ks": ";".join(
            f"{r.distance_mode}:{int(r.recommended_parsimonious_k)}" for r in sens.itertuples()
        ),
        "manual_selected_k": "",
        "manual_notes": "",
    }
    pd.DataFrame([review]).to_csv(root_out / "root_k_manual_review_row.csv", index=False)

    return {
        "panel": panel,
        "feature_source": root,
        "inputs": inputs,
        "mode_results": mode_results,
        "recommendations": rec_df,
        "manual_review": review,
        "output_dir": root_out,
    }


def run_full_stage2b2_review(
    *,
    stage2b1_root: str | Path,
    output_root: str | Path,
    parser_path: Optional[str | Path],
    distance_modes: Sequence[str] = DEFAULT_DISTANCE_MODES,
    primary_mode: str = DEFAULT_PRIMARY_MODE,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    n_semantic_permutations: int = DEFAULT_N_SEMANTIC_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    score_weights: Optional[Mapping[str, float]] = None,
    review_thresholds: Optional[Mapping[str, float]] = None,
    primary_k_max: int = DEFAULT_PRIMARY_SELECTION_K_MAX,
    selection_within_best_fraction: float = DEFAULT_SELECTION_WITHIN_BEST_FRACTION,
    top_heatmaps: int = 3,
) -> Dict[str, object]:
    stage2b1_root = Path(stage2b1_root)
    output_root = ensure_dir(output_root)
    manifest = load_b1_manifest(stage2b1_root)

    all_k = []
    all_rec = []
    all_review = []
    root_results = {}
    errors = []

    for _, row in manifest.iterrows():
        panel = str(row["panel"])
        root = str(row["feature_source"])
        key = (panel, root)
        print(f"[STAGE2B2] {panel} / {root}", flush=True)
        try:
            rr = evaluate_one_root(
                stage2b1_root=stage2b1_root,
                manifest_row=row.to_dict(),
                output_root=output_root / "roots",
                parser_path=parser_path,
                distance_modes=distance_modes,
                primary_mode=primary_mode,
                k_min=k_min,
                k_max=k_max,
                linkage_method=linkage_method,
                n_semantic_permutations=n_semantic_permutations,
                random_seed=random_seed,
                score_weights=score_weights,
                review_thresholds=review_thresholds,
                primary_k_max=primary_k_max,
                selection_within_best_fraction=selection_within_best_fraction,
                top_heatmaps=top_heatmaps,
            )
            root_results[key] = rr
            all_rec.append(rr["recommendations"])
            all_review.append(pd.DataFrame([rr["manual_review"]]))
            for mode, res in rr["mode_results"].items():
                all_k.append(res["k_diagnostics"])
        except Exception as exc:
            errors.append({
                "panel": panel,
                "feature_source": root,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"[ERROR] {panel}/{root}: {type(exc).__name__}: {exc}", flush=True)

    k_all = pd.concat(all_k, ignore_index=True, sort=False) if all_k else pd.DataFrame()
    rec_all = pd.concat(all_rec, ignore_index=True, sort=False) if all_rec else pd.DataFrame()
    review_all = pd.concat(all_review, ignore_index=True, sort=False) if all_review else pd.DataFrame()
    err_df = pd.DataFrame(errors)

    k_all.to_csv(output_root / "all_panel_root_k_diagnostics.csv", index=False)
    rec_all.to_csv(output_root / "all_panel_root_k_recommendations.csv", index=False)
    review_all.to_csv(output_root / "stage2b2_manual_k_review_template.csv", index=False)
    err_df.to_csv(output_root / "stage2b2_errors.csv", index=False)

    # Primary-mode compact ranking table: top 5 K per root.
    if not k_all.empty:
        primary_k = k_all[k_all["distance_mode"].astype(str).eq(str(primary_mode))].copy()
        top_rows = []
        for (panel, root), g in primary_k.groupby(["panel", "feature_source"], sort=True):
            z = g.sort_values(["k_composite_score", "requested_k"], ascending=[False, True]).head(5).copy()
            z["score_rank_within_root"] = np.arange(1, len(z) + 1)
            top_rows.append(z)
        top = pd.concat(top_rows, ignore_index=True, sort=False) if top_rows else pd.DataFrame()
        top.to_csv(output_root / "primary_mode_top5_k_per_root.csv", index=False)

        # Panel-level comparison plot of composite curves.
        for panel, gp in primary_k.groupby("panel", sort=True):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for root, gr in gp.groupby("feature_source", sort=True):
                gr = gr.sort_values("requested_k")
                ax.plot(gr["requested_k"], gr["k_composite_score"], marker="o", label=root)
            ax.set_xlabel("K")
            ax.set_ylabel("Composite K score")
            ax.set_title(f"{panel}: primary ({primary_mode}) K score across prep roots")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(output_root / f"{panel}_primary_mode_k_score_by_root.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

    settings = {
        "stage2b1_root": str(stage2b1_root),
        "output_root": str(output_root),
        "parser_path": None if parser_path is None else str(parser_path),
        "distance_modes": list(distance_modes),
        "primary_mode": str(primary_mode),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "linkage_method": str(linkage_method),
        "n_semantic_permutations": int(n_semantic_permutations),
        "random_seed": int(random_seed),
        "score_weights": dict(DEFAULT_K_SCORE_WEIGHTS if score_weights is None else score_weights),
        "review_thresholds": dict(DEFAULT_REVIEW_THRESHOLDS if review_thresholds is None else review_thresholds),
        "primary_k_max": int(primary_k_max),
        "selection_within_best_fraction": float(selection_within_best_fraction),
        "semantic_composite_policy": "45% statistical silhouette + 30% primitive coherence enrichment + 10% tissue-aware coherence enrichment + 7.5% largest-cluster control + 7.5% singleton control; metric-aware coherence and all semantic silhouettes diagnostic only; primary K selection re-ranks these components within K<=primary_k_max",
        "primary_geometry_policy": "row_spearman primary; direct_signed sensitivity; direct signed within-module cohesion reported as diagnostic guardrail",
    }
    with open(output_root / "stage2b2_review_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    return {
        "manifest": manifest,
        "root_results": root_results,
        "k_diagnostics": k_all,
        "recommendations": rec_all,
        "manual_review": review_all,
        "errors": err_df,
        "output_root": output_root,
    }


# =============================================================================
# Notebook conveniences
# =============================================================================

def show_root_summary(
    review_result: Mapping[str, object],
    panel: str,
    root: str,
    mode: str = DEFAULT_PRIMARY_MODE,
    top_n: int = 10,
) -> pd.DataFrame:
    d = review_result["k_diagnostics"].copy()
    z = d[
        d["panel"].astype(str).eq(str(panel))
        & d["feature_source"].astype(str).eq(str(root))
        & d["distance_mode"].astype(str).eq(str(mode))
    ].copy()
    cols = [
        "requested_k", "k_composite_score", "stat_silhouette",
        "primitive_semantic_silhouette", "tissue_aware_semantic_silhouette", "metric_aware_semantic_silhouette",
        "primitive_coherence_enrichment", "tissue_aware_coherence_enrichment", "metric_aware_coherence_enrichment",
        "max_cluster_fraction", "singleton_fraction",
        "median_module_mean_consensus_rho", "q25_module_mean_consensus_rho",
        "median_module_fraction_positive_pairs", "passes_direct_cohesion_review",
        "median_module_cohort_mean_rho",
        "fraction_module_cohort_positive", "within_best_fraction", "passes_size_review",
    ]
    cols = [c for c in cols if c in z.columns]
    return z.sort_values(["k_composite_score", "requested_k"], ascending=[False, True])[cols].head(int(top_n)).reset_index(drop=True)


def save_manual_k_selections(
    review_csv: str | Path,
    selections: Mapping[Tuple[str, str], int],
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    p = Path(review_csv)
    df = pd.read_csv(p)
    for (panel, root), k in selections.items():
        m = df["panel"].astype(str).eq(str(panel)) & df["feature_source"].astype(str).eq(str(root))
        df.loc[m, "manual_selected_k"] = int(k)
    out = Path(output_path) if output_path else p
    df.to_csv(out, index=False)
    return df
