#!/usr/bin/env python3
"""
stage2a_interpretability_utils_v2.py

Single ontology/interpretability adapter for corrected Stage 2A-4 rescue,
Stage 2A-5 microcompression, grid canonicalization, and later Stage 2B review.

This module delegates feature grammar parsing to stage2_feature_parser_v8.py
and deliberately keeps these dimensions separate:
  * biological cell identity / entity role
  * checkpoint/state identity
  * tissue compartment
  * feature type and subtype
  * summary statistic / metric parameters

Microcompression policy is intentionally conservative:
  * state rescue: same measurement after removing states, only when an
    underlying cell identity is known;
  * metric-summary rescue: only Median <-> Mean (Median preferred);
  * compartment rescue: same measurement except compartment, with
    All preferred over Tumor/Epi preferred over Stroma;
  * residual microfamily: exact structured biological identity, excluding
    prep-root/provenance duplication.

This avoids collapsing Q1/Q3 with Mean, Mean with SD, density with ratio,
interaction_diff with interaction_z, or different Renyi parameters.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PARSER_VERSION = "stage2_feature_parser_v8_2"
DEFAULT_PARSER_PATH = Path(__file__).with_name("stage2_feature_parser_v8_2.py")


def import_module_from_path(module_name: str, path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_parser(parser_path: Optional[Union[str, Path]] = None):
    return import_module_from_path(
        "stage2_feature_parser_v8_for_interpretability",
        parser_path or DEFAULT_PARSER_PATH,
    )


def _stable(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _norm_compartment(x: object) -> str:
    if x is None or pd.isna(x):
        return "none"
    s = str(x).strip().lower()
    if s in {"epi", "tumor", "tumour"}:
        return "tumor"
    if s in {"stroma", "str"}:
        return "stroma"
    if s == "all":
        return "all"
    return "none"


def _summary_class_and_priority(summary: object) -> Tuple[str, int]:
    """
    Only Mean/Median are considered microcompression-compatible location
    summaries. Quantiles/extremes/dispersion remain distinct measurements.
    """
    if summary is None or pd.isna(summary):
        return "none", 0
    s = str(summary).strip().lower()

    if s == "median":
        return "location", 0
    if s == "mean":
        return "location", 1

    if s in {"q1", "q3"}:
        return "quantile", 10
    if s in {"min", "max"}:
        return "extreme", 20
    if s in {"sd", "std", "stdev"}:
        return "dispersion", 30

    # ATHENA / triad summaries are distinct and are not interchangeable
    # during metric-summary rescue.
    return f"distinct:{s}", 40


def _state_marker_complexity(state: Optional[str]) -> int:
    if state is None or str(state).strip() in {"", "None", "nan"}:
        return 0
    s = str(state)
    if s == "PD1_PDL1":
        return 2
    return 1


def _entity_full(entities: Sequence[Mapping]) -> list:
    return [
        {
            "role": e.get("role"),
            "cell": e.get("cell"),
            "state": e.get("state"),
        }
        for e in entities
    ]


def _entity_cells_only(entities: Sequence[Mapping]) -> list:
    return [
        {
            "role": e.get("role"),
            "cell": e.get("cell"),
        }
        for e in entities
    ]


def _has_underlying_cell(entities: Sequence[Mapping]) -> bool:
    return any(e.get("cell") not in {None, "", "nan"} for e in entities)


def _source_priority(source: object) -> int:
    # Prefer simpler/base provenance when two representations are otherwise
    # biologically identical.
    order = {
        "phenotype_only": 0,
        "compartment": 1,
        "AR_state": 2,
        "compartment_state": 3,
        "AR_checkpoint_state": 4,
    }
    return order.get(str(source), 50)


def _feature_kind(parsed: Mapping) -> str:
    return str(parsed.get("feature_type") or "unknown")


def _metric_kind(parsed: Mapping) -> str:
    subtype = parsed.get("feature_subtype")
    if subtype is None:
        return str(parsed.get("feature_type") or "unknown")
    return str(subtype)


def enrich_one(row: Mapping, parser) -> Dict[str, object]:
    feature = str(row.get("feature", ""))
    feature_source = row.get("feature_source")
    feature_group = row.get("feature_group")

    parsed = parser.parse_feature(
        feature=feature,
        feature_source=None if pd.isna(feature_source) else str(feature_source),
        feature_group=None if pd.isna(feature_group) else str(feature_group),
    )

    entities = parsed.get("entities", []) or []
    metric_params = parsed.get("metric_params", {}) or {}
    compartment = _norm_compartment(parsed.get("compartment"))
    summary = parsed.get("summary_stat")
    summary_class, summary_priority = _summary_class_and_priority(summary)

    state_complexity = sum(
        _state_marker_complexity(e.get("state"))
        for e in entities
    )

    full_entities = _entity_full(entities)
    cell_entities = _entity_cells_only(entities)

    # Base identity shared across state variants. For checkpoint-state-only
    # features with no underlying cell identity, we intentionally prevent
    # state rescue by including the full state entities in the key.
    if _has_underlying_cell(entities):
        state_entities = cell_entities
        state_rescue_allowed = True
    else:
        state_entities = full_entities
        state_rescue_allowed = False

    feature_type = parsed.get("feature_type")
    feature_subtype = parsed.get("feature_subtype")

    base_without_state = {
        "feature_type": feature_type,
        "feature_subtype": feature_subtype,
        "entities": state_entities,
        "metric_params": metric_params,
    }

    exact_biology = {
        "feature_type": feature_type,
        "feature_subtype": feature_subtype,
        "entities": full_entities,
        "compartment": compartment,
        "summary_stat": summary,
        "metric_params": metric_params,
    }

    state_key = _stable({
        **base_without_state,
        "compartment": compartment,
        "summary_stat": summary,
        "state_rescue_allowed": state_rescue_allowed,
    })

    # Summary excluded, but full state identity retained. Since only
    # summary_class == "location" is eligible in Stage 2A-4/5, this key
    # can only lead to Median <-> Mean simplification.
    metric_key = _stable({
        "feature_type": feature_type,
        "feature_subtype": feature_subtype,
        "entities": full_entities,
        "compartment": compartment,
        "metric_params": metric_params,
    })

    compartment_key = _stable({
        "feature_type": feature_type,
        "feature_subtype": feature_subtype,
        "entities": full_entities,
        "summary_stat": summary,
        "metric_params": metric_params,
    })

    exact_key = _stable(exact_biology)

    warnings = parsed.get("warnings", []) or []
    cells = parsed.get("cells", []) or []
    states = parsed.get("states", []) or []
    lineages = parsed.get("lineages", []) or []

    return {
        "parser_version": PARSER_VERSION,
        "parser_status": parsed.get("parse_status", "unknown"),
        "candidate_eligible": bool(parsed.get("candidate_eligible", True)),
        "parser_warnings": ";".join(map(str, warnings)),
        "parsed_feature_type": feature_type,
        "parsed_feature_subtype": feature_subtype,
        "parsed_cells": ";".join(map(str, cells)),
        "parsed_states": ";".join(map(str, states)),
        "parsed_lineages": ";".join(map(str, lineages)),
        "parsed_entities_json": _stable(full_entities),
        "parsed_metric_params_json": _stable(metric_params),

        # Legacy-compatible columns consumed by Stage 2A-4/5/grid.
        "feature_kind": _feature_kind(parsed),
        "metric_kind": _metric_kind(parsed),
        "summary_stat": summary if summary is not None else "",
        "summary_class": summary_class,
        "summary_priority": int(summary_priority),
        "compartment": compartment,
        "compartment_priority": {
            "all": 0,
            "tumor": 1,
            "stroma": 2,
            "none": 0,
        }.get(compartment, 10),
        "state_complexity": int(state_complexity),
        "state_rescue_allowed": bool(state_rescue_allowed),
        "source_priority": _source_priority(feature_source),

        "state_simplification_key": state_key,
        "metric_simplification_key": metric_key,
        "compartment_simplification_key": compartment_key,

        # Residual compression is deliberately limited to exact structured
        # biological identity (prep-root duplication excluded).
        "residual_microfamily_key": exact_key,
        "full_semantic_key": exact_key,
        "exact_semantic_key": exact_key,
    }


def add_interpretability_columns(
    df: pd.DataFrame,
    parser_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    out = df.copy()

    # Idempotent: reparse only if this exact parser version has not already
    # enriched the rows.
    if (
        "parser_version" in out.columns
        and len(out) > 0
        and out["parser_version"].fillna("").eq(PARSER_VERSION).all()
        and "full_semantic_key" in out.columns
    ):
        return out

    parser = get_parser(parser_path)
    parsed_rows = [
        enrich_one(row, parser)
        for _, row in out.iterrows()
    ]
    parsed_df = pd.DataFrame(parsed_rows, index=out.index)

    # Replace stale legacy columns from the old parser.
    for col in parsed_df.columns:
        if col in out.columns:
            out = out.drop(columns=[col])

    return pd.concat([out, parsed_df], axis=1)


def safe_spearman(
    a: pd.Series,
    b: pd.Series,
    min_n: int = 3,
) -> Tuple[float, int]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < int(min_n):
        return np.nan, n
    xx = x.loc[mask]
    yy = y.loc[mask]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan, n
    return float(xx.corr(yy, method="spearman")), n


def is_exact_vector_duplicate(
    a: pd.Series,
    b: pd.Series,
    atol: float = 1e-12,
) -> Tuple[bool, int]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")

    both_na = x.isna() & y.isna()
    both_obs = x.notna() & y.notna()
    mismatch_missing = x.isna() ^ y.isna()

    if bool(mismatch_missing.any()):
        return False, int(both_obs.sum())

    n = int(both_obs.sum())
    if n == 0:
        return False, 0

    exact = bool(
        np.allclose(
            x.loc[both_obs].to_numpy(dtype=float),
            y.loc[both_obs].to_numpy(dtype=float),
            atol=float(atol),
            rtol=0.0,
            equal_nan=True,
        )
    )
    return exact, n
