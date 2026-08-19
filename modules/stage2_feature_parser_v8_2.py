#!/usr/bin/env python3
"""
stage2_feature_parser_v8.py

Grammar-aware parser for Stage 2 mIF candidate features.

Goals
-----
1. Parse the feature family/subtype BEFORE parsing biological entities.
2. Use feature_source (prep root) whenever available.
3. Keep cell identity, cell state/checkpoint state, tissue compartment,
   metric subtype, and summary statistic as separate concepts.
4. Avoid generic substring scanning that can:
   - turn PD1/PDL1 into "cells"
   - turn Stroma/Tumor tissue labels into cells
   - double-count CD8 T cells as both cd8_t_cell and t_cell
5. Provide structural keys that can later be used safely for rescue/microcompression.
6. Fall back to prep-root inference only when feature_source is missing.

The parser is intentionally independent of Stage 2A/2B scripts so it can be
validated on existing candidate tables before integration.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# Canonical vocabularies
# =============================================================================

CELL_ALIASES: Dict[str, str] = {
    # tumor / stromal / broad compartment phenotypes
    "tumor": "tumor_cell",
    "tumour": "tumor_cell",
    "tumor_cell": "tumor_cell",
    "tumour_cell": "tumor_cell",
    "cancer": "tumor_cell",
    "cancer_cell": "tumor_cell",
    "panck": "tumor_cell",

    "stroma": "stromal_cell",
    "stromal": "stromal_cell",
    "stromal_cell": "stromal_cell",
    "all_neg": "stromal_cell",
    "all-negative": "stromal_cell",
    "fibro": "stromal_cell",
    "fibroblast": "stromal_cell",

    "immune": "immune_cell",
    "immune_cell": "immune_cell",

    # T lineage
    "t_cell": "t_cell",
    "tcell": "t_cell",
    "cd8": "cd8_t_cell",
    "cd8_t_cell": "cd8_t_cell",
    "cd4": "cd4_t_cell",
    "cd4_t_cell": "cd4_t_cell",
    "treg": "treg_cell",
    "treg_cell": "treg_cell",
    "foxp3": "treg_cell",

    # B / plasma / NK / myeloid
    "b_cell": "b_cell",
    "bcell": "b_cell",
    "plasma": "plasma_cell",
    "plasma_cell": "plasma_cell",
    "nk": "nk_cell",
    "nk_cell": "nk_cell",
    "macrophage": "macrophage",
    "macrophages": "macrophage",
    "cd68": "macrophage",
}

CELL_LINEAGE: Dict[str, str] = {
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
}

# IMPORTANT: states are deliberately NOT in CELL_ALIASES.
STATE_ALIASES: Dict[str, str] = {
    "pd1": "PD1",
    "pdl1": "PDL1",
    "pd1_pdl1": "PD1_PDL1",
    "checkpoint_neg": "checkpoint_neg",
}
STATE_SUFFIXES = sorted(STATE_ALIASES.keys(), key=len, reverse=True)

COMPARTMENT_MAP: Dict[str, str] = {
    "all": "All",
    "tumor": "Tumor",
    "tumour": "Tumor",
    "epi": "Tumor",
    "stroma": "Stroma",
    "str": "Stroma",
}

KNOWN_FEATURE_SOURCES = {
    "phenotype_only",
    "AR_state",
    "AR_checkpoint_state",
    "compartment",
    "compartment_state",
}
KNOWN_FEATURE_GROUPS = {"NN", "athena", "cell_features", "triads"}

NN_SUMMARIES = ("Mean", "SD", "Max", "Min", "Median", "Q1", "Q3")
LOCAL_SUMMARIES = ("min", "mean", "median", "max", "pct_non_na")
TRIAD_SUMMARIES = ("count", "frac_center", "n_cells")

# ATHENA families supported by the parser. Not every family must be present
# among the current selected candidates.
ATHENA_DIVERSITY_METRICS = {
    "richness",
    "shannon",
    "simpson",
    "renyi",
    "hill",
    "quadratic",
    "rao",
}
ATHENA_INTERACTION_PREFIXES = {
    "inter_diff": "interaction_diff",
    "inter_z": "interaction_z",
    "inter_p": "interaction_p",
}


# =============================================================================
# Utilities
# =============================================================================

def _norm_token(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"_+", "_", s)


def canon_cell(x: str) -> Optional[str]:
    return CELL_ALIASES.get(_norm_token(x))


def canon_state(x: str) -> Optional[str]:
    return STATE_ALIASES.get(_norm_token(x))


def canon_compartment(x: str) -> Optional[str]:
    return COMPARTMENT_MAP.get(_norm_token(x))


def split_feature_uid(uid: str) -> Tuple[str, str, str]:
    parts = str(uid).split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "", "", str(uid)


def _jsonable_tuple(x):
    if isinstance(x, tuple):
        return [_jsonable_tuple(v) for v in x]
    if isinstance(x, list):
        return [_jsonable_tuple(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable_tuple(v) for k, v in x.items()}
    return x


def _dedupe_keep_order(values: Sequence[Optional[str]]) -> List[str]:
    out: List[str] = []
    for x in values:
        if x is not None and x not in out:
            out.append(x)
    return out


# =============================================================================
# Feature-group and prep-root inference
# =============================================================================

def infer_feature_group(feature: str) -> Optional[str]:
    """Infer broad feature group from grammar only when metadata is absent."""
    f = str(feature)

    if "_to_" in f and re.search(
        r"_(Mean|SD|Max|Min|Median|Q1|Q3)$", f
    ):
        return "NN"

    if f.startswith(
        (
            "inter_",
            "infiltration_",
            "ripley_",
            "richness_",
            "shannon_",
            "simpson_",
            "renyi_",
            "hill_",
            "quadratic_",
            "rao_",
            "modularity_",
        )
    ):
        return "athena"

    if re.match(r"^(All|Epi|Stroma)__", f):
        return "cell_features"

    if f.startswith(("triad_centered__", "triad__", "(")):
        return "triads"

    return None


def infer_feature_source(
    feature: str,
    feature_group: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Best-effort prep-root inference.

    This is deliberately secondary to explicit feature_source metadata.
    Ambiguous features return a lower-confidence assignment.
    """
    f = str(feature)
    group = feature_group or infer_feature_group(f)

    state_tokens_present = any(
        re.search(
            rf"(^|_|__){re.escape(s)}($|_|__)",
            f,
            flags=re.IGNORECASE,
        )
        for s in STATE_ALIASES
    )

    # Use exact tokenization heuristics only for fallback inference.
    cells_seen = set()
    scrubbed = f
    for s in sorted(STATE_ALIASES, key=len, reverse=True):
        scrubbed = re.sub(re.escape(s), "", scrubbed, flags=re.IGNORECASE)

    token_candidates = re.split(r"__|_to_|__over__", scrubbed)
    for tok in token_candidates:
        # Strip known suffixes/prefixes before exact cell canonicalization.
        tok = re.sub(
            r"_(Mean|SD|Max|Min|Median|Q1|Q3)$",
            "",
            tok,
        )
        c = canon_cell(tok)
        if c:
            cells_seen.add(c)

    if state_tokens_present:
        if not cells_seen:
            return "AR_checkpoint_state", "high"

        if cells_seen.issubset(
            {"immune_cell", "stromal_cell", "tumor_cell"}
        ):
            return "compartment_state", "medium"

        return "AR_state", "medium"

    if cells_seen and cells_seen.issubset(
        {"immune_cell", "stromal_cell", "tumor_cell"}
    ):
        return "compartment", "medium"

    if group is not None:
        return "phenotype_only", "low"

    return None, "none"


# =============================================================================
# Prep-root-aware biological entity parsing
# =============================================================================

def parse_entity(
    token: str,
    feature_source: Optional[str],
) -> Dict[str, object]:
    """
    Parse one biological entity token into separate cell and state fields.

    Examples
    --------
    phenotype_only:
        cd8_t_cell -> cell=cd8_t_cell, state=None

    AR_state / compartment_state:
        t_cell_PD1 -> cell=t_cell, state=PD1
        cd8_t_cell_PD1_PDL1 -> cell=cd8_t_cell, state=PD1_PDL1
        cd8_t_cell__PD1_PDL1 -> same result

    AR_checkpoint_state:
        checkpoint_neg -> cell=None, state=checkpoint_neg
    """
    raw = str(token).strip()
    source = str(feature_source) if feature_source is not None else ""
    warnings: List[str] = []

    if source == "AR_checkpoint_state":
        state = canon_state(raw)
        if state:
            return {
                "raw": raw,
                "cell": None,
                "state": state,
                "warnings": warnings,
            }

        cell = canon_cell(raw)
        if cell:
            warnings.append("checkpoint_root_contains_cell_token")
            return {
                "raw": raw,
                "cell": cell,
                "state": None,
                "warnings": warnings,
            }

        return {
            "raw": raw,
            "cell": None,
            "state": None,
            "warnings": ["unparsed_entity"],
        }

    if source in {"AR_state", "compartment_state"}:
        # NN feature names use cell__state within each endpoint.
        if "__" in raw:
            bits = raw.split("__")
            if len(bits) == 2 and canon_state(bits[1]):
                cell = canon_cell(bits[0])
                state = canon_state(bits[1])
                if cell is None:
                    warnings.append("unparsed_cell_before_state")
                return {
                    "raw": raw,
                    "cell": cell,
                    "state": state,
                    "warnings": warnings,
                }

        # cell_features / triads use cell_STATE suffixes.
        low = raw.lower()
        for suffix in STATE_SUFFIXES:
            marker = "_" + suffix
            if low.endswith(marker):
                base = raw[: -len(marker)]
                cell = canon_cell(base)
                state = canon_state(suffix)
                if cell is None:
                    warnings.append("unparsed_cell_before_state")
                return {
                    "raw": raw,
                    "cell": cell,
                    "state": state,
                    "warnings": warnings,
                }

        # Defensive fallbacks.
        state = canon_state(raw)
        if state:
            warnings.append("state_only_entity_in_cell_state_root")
            return {
                "raw": raw,
                "cell": None,
                "state": state,
                "warnings": warnings,
            }

        cell = canon_cell(raw)
        if cell:
            warnings.append("state_missing_in_state_root")
            return {
                "raw": raw,
                "cell": cell,
                "state": None,
                "warnings": warnings,
            }

        return {
            "raw": raw,
            "cell": None,
            "state": None,
            "warnings": ["unparsed_entity"],
        }

    # phenotype_only / compartment / unknown non-state source
    cell = canon_cell(raw)
    if cell:
        return {
            "raw": raw,
            "cell": cell,
            "state": None,
            "warnings": warnings,
        }

    state = canon_state(raw)
    if state:
        warnings.append("state_token_in_nonstate_root")
        return {
            "raw": raw,
            "cell": None,
            "state": state,
            "warnings": warnings,
        }

    return {
        "raw": raw,
        "cell": None,
        "state": None,
        "warnings": ["unparsed_entity"],
    }


def _entity(
    token: str,
    role: str,
    source: Optional[str],
) -> Dict[str, object]:
    out = parse_entity(token, source)
    out["role"] = role
    return out


# =============================================================================
# Grammar-specific feature parsers
# =============================================================================

def _parse_nn(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    m = re.match(
        r"^(.+)_to_(.+)_(Mean|SD|Max|Min|Median|Q1|Q3)$",
        feature,
    )
    if not m:
        raise ValueError("NN grammar mismatch")

    left, right, stat = m.groups()

    return {
        "feature_type": "nearest_neighbor",
        "feature_subtype": "1NN_distance",
        "summary_stat": stat,
        "compartment": None,
        "entities": [
            _entity(left, "reference", source),
            _entity(right, "neighbor", source),
        ],
        "metric_params": {},
    }


def _parse_cell_features(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    # Technical bookkeeping/support columns in the cell-feature tables.
    # These describe how many cells were available for downstream feature
    # construction; they are not biological candidate biomarkers.
    technical = re.match(
        r"^(All|Epi|Stroma)__(n_cells|n_resolved_for_ratio)$",
        feature,
        flags=re.IGNORECASE,
    )
    if technical:
        compartment_raw, subtype_raw = technical.groups()
        return {
            "feature_type": "technical_qc",
            "feature_subtype": subtype_raw.lower(),
            "summary_stat": None,
            "compartment": canon_compartment(compartment_raw),
            "entities": [],
            "metric_params": {},
            "candidate_eligible": False,
        }

    m = re.match(
        r"^(All|Epi|Stroma)__(count|density|prop|ratio)__(.+)$",
        feature,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("cell_features grammar mismatch")

    compartment_raw, subtype_raw, rest = m.groups()
    subtype = {
        "prop": "proportion",
    }.get(subtype_raw.lower(), subtype_raw.lower())

    entities: List[Dict[str, object]]

    if subtype_raw.lower() == "ratio":
        if "__over__" not in rest:
            raise ValueError("ratio feature missing __over__")

        numerator, denominator = rest.split("__over__", 1)
        entities = [
            _entity(numerator, "numerator", source),
            _entity(denominator, "denominator", source),
        ]
    else:
        entities = [
            _entity(rest, "target", source),
        ]

    return {
        "feature_type": "composition",
        "feature_subtype": subtype,
        "summary_stat": None,
        "compartment": canon_compartment(compartment_raw),
        "entities": entities,
        "metric_params": {},
    }


def _parse_triads(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    # Technical bookkeeping/support columns emitted with triad tables.
    # Examples include n_cells_total_input. These describe the input/support
    # available to calculate triad features and are not biological biomarkers.
    if re.match(r"^n_cells(?:_[A-Za-z0-9]+)*$", feature):
        return {
            "feature_type": "technical_qc",
            "feature_subtype": feature.lower(),
            "summary_stat": None,
            "compartment": None,
            "entities": [],
            "metric_params": {},
            "candidate_eligible": False,
        }

    # Current expanded grammar.
    if feature.startswith("triad_centered__"):
        parts = feature.split("__")
        if len(parts) != 6:
            raise ValueError(
                "triad_centered expected 6 double-underscore fields; "
                f"observed {len(parts)}"
            )

        _, e1, e2, e3, compartment_raw, summary = parts

        return {
            "feature_type": "triad",
            "feature_subtype": "centered",
            "summary_stat": summary,
            "compartment": canon_compartment(compartment_raw),
            "entities": [
                _entity(e1, "triad_1", source),
                _entity(e2, "triad_2", source),
                _entity(e3, "triad_3", source),
            ],
            "metric_params": {},
        }

    # Checkpoint-state legacy grammar seen in the current candidates:
    # triad__All__center__checkpoint_neg__n_cells
    if feature.startswith("triad__"):
        parts = feature.split("__")

        entities: List[Dict[str, object]] = []
        if len(parts) >= 4:
            entities.append(
                _entity(parts[3], "triad_state_or_target", source)
            )

        return {
            "feature_type": "triad",
            "feature_subtype": "legacy",
            "summary_stat": parts[-1] if len(parts) >= 2 else None,
            "compartment": (
                canon_compartment(parts[1]) if len(parts) >= 2 else None
            ),
            "entities": entities,
            "metric_params": {
                "legacy_parts": parts[2:-1],
            },
        }

    # Older tuple-style triads.
    if feature.strip().startswith("("):
        tuple_part = feature[: feature.rfind(")") + 1]
        vals = ast.literal_eval(tuple_part)
        tail = feature[feature.rfind(")") + 1 :].strip()

        return {
            "feature_type": "triad",
            "feature_subtype": "legacy_tuple",
            "summary_stat": None,
            "compartment": canon_compartment(tail),
            "entities": [
                _entity(str(v), f"triad_{i + 1}", source)
                for i, v in enumerate(vals)
            ],
            "metric_params": {},
        }

    raise ValueError("triad grammar mismatch")


def _parse_athena_interaction(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    m = re.match(r"^inter_(diff|z|p)_(.+)$", feature)
    if not m:
        raise ValueError("ATHENA interaction grammar mismatch")

    flavor, rest = m.groups()
    parts = rest.split("__")
    entities: List[Dict[str, object]]

    if source in {"AR_state", "compartment_state"}:
        if len(parts) != 5:
            raise ValueError(
                "stateful ATHENA interaction expected "
                "cell1__state1__cell2__state2__compartment"
            )

        cell1, state1, cell2, state2, compartment_raw = parts
        entities = [
            _entity(
                f"{cell1}_{state1}",
                "source",
                source,
            ),
            _entity(
                f"{cell2}_{state2}",
                "target",
                source,
            ),
        ]

    elif source == "AR_checkpoint_state":
        if len(parts) != 3:
            raise ValueError(
                "checkpoint ATHENA interaction expected "
                "state1__state2__compartment"
            )

        state1, state2, compartment_raw = parts
        entities = [
            _entity(state1, "source", source),
            _entity(state2, "target", source),
        ]

    else:
        if len(parts) != 3:
            raise ValueError(
                "ATHENA interaction expected "
                "cell1__cell2__compartment"
            )

        cell1, cell2, compartment_raw = parts
        entities = [
            _entity(cell1, "source", source),
            _entity(cell2, "target", source),
        ]

    return {
        "feature_type": "ATHENA_interaction",
        "feature_subtype": f"interaction_{flavor}",
        "summary_stat": None,
        "compartment": canon_compartment(compartment_raw),
        "entities": entities,
        "metric_params": {
            "interaction_component": flavor,
        },
    }


def _parse_athena_infiltration(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    m = re.match(
        r"^infiltration_(.+?)__(All|Tumor|Stroma|Epi)__"
        r"(min|mean|median|max|pct_non_na)$",
        feature,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("ATHENA infiltration grammar mismatch")

    entity_raw, compartment_raw, summary = m.groups()

    return {
        "feature_type": "ATHENA_infiltration",
        "feature_subtype": "infiltration",
        "summary_stat": summary.lower(),
        "compartment": canon_compartment(compartment_raw),
        "entities": [
            _entity(entity_raw, "infiltrating", source),
        ],
        "metric_params": {},
    }


def _parse_athena_ripley(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    """
    Flexible parser for known Ripley-like feature naming.

    Expected final structure:
        ripley_<body>__<compartment>__<summary>

    The <body> may include an entity, edge correction, and/or variant.
    """
    m = re.match(
        r"^ripley_(.+?)__(All|Tumor|Stroma|Epi)__"
        r"(at\d+|peak_abs|auc)$",
        feature,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("ATHENA Ripley grammar mismatch")

    body, compartment_raw, summary = m.groups()
    metric_params: Dict[str, object] = {}
    entities: List[Dict[str, object]] = []

    mm = re.match(
        r"^(.+?)_(translation|border|ripley|none)(?:_(.+))?$",
        body,
        flags=re.IGNORECASE,
    )

    if mm:
        entity_raw, correction, extra = mm.groups()
        entities.append(
            _entity(entity_raw, "target", source)
        )
        metric_params["edge_correction"] = correction
        if extra:
            metric_params["ripley_variant"] = extra
    else:
        entities.append(
            _entity(body, "target", source)
        )

    return {
        "feature_type": "ATHENA_spatial_statistics",
        "feature_subtype": "ripley",
        "summary_stat": summary.lower(),
        "compartment": canon_compartment(compartment_raw),
        "entities": entities,
        "metric_params": metric_params,
    }


def _parse_athena_diversity_or_graph(
    feature: str,
) -> Dict[str, object]:
    m = re.match(
        r"^(richness|shannon|simpson|renyi|hill|quadratic|rao|modularity)_"
        r"(.+?)__(All|Tumor|Stroma|Epi)__"
        r"(min|mean|median|max|pct_non_na)$",
        feature,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("ATHENA diversity/graph grammar mismatch")

    metric, body, compartment_raw, summary = m.groups()
    metric = metric.lower()

    metric_params: Dict[str, object] = {
        "basis": body,
    }

    q_match = re.search(
        r"(?:^|_)q([0-9.]+)(?:_|$)",
        body,
        flags=re.IGNORECASE,
    )
    if q_match:
        metric_params["q"] = float(q_match.group(1))

    alpha_match = re.search(
        r"(?:^|_)alpha([0-9.]+)(?:_|$)",
        body,
        flags=re.IGNORECASE,
    )
    if alpha_match:
        metric_params["alpha"] = float(alpha_match.group(1))

    for graph in ("radius", "contact", "knn"):
        if re.search(
            rf"(?:^|_){graph}(?:_|$)",
            body,
            flags=re.IGNORECASE,
        ):
            metric_params["graph"] = graph

    return {
        "feature_type": (
            "ATHENA_graph"
            if metric == "modularity"
            else "ATHENA_diversity"
        ),
        "feature_subtype": metric,
        "summary_stat": summary.lower(),
        "compartment": canon_compartment(compartment_raw),
        "entities": [],
        "metric_params": metric_params,
    }


def _parse_athena(
    feature: str,
    source: Optional[str],
) -> Dict[str, object]:
    if feature.startswith(
        ("inter_diff_", "inter_z_", "inter_p_")
    ):
        return _parse_athena_interaction(feature, source)

    if feature.startswith("infiltration_"):
        return _parse_athena_infiltration(feature, source)

    if feature.startswith("ripley_"):
        return _parse_athena_ripley(feature, source)

    return _parse_athena_diversity_or_graph(feature)


# =============================================================================
# Public parser
# =============================================================================

def parse_feature(
    feature: str,
    feature_source: Optional[str] = None,
    feature_group: Optional[str] = None,
) -> Dict[str, object]:
    """
    Parse one feature into a structured ontology record.

    The parser always routes by feature_group first, and uses feature_source
    for prep-root-aware entity/state parsing.
    """
    raw_input = str(feature)
    f = raw_input

    # Support feature_uid passed directly.
    if "|" in f and (
        feature_source is None or feature_group is None
    ):
        uid_source, uid_group, uid_feature = split_feature_uid(f)
        if feature_source is None:
            feature_source = uid_source
        if feature_group is None:
            feature_group = uid_group
        f = uid_feature

    if feature_group is None or str(feature_group) in {"", "nan", "None"}:
        feature_group = infer_feature_group(f)

    source_inferred = False
    source_inference_confidence: Optional[str] = None

    if feature_source is None or str(feature_source) in {"", "nan", "None"}:
        feature_source, source_inference_confidence = infer_feature_source(
            f,
            feature_group,
        )
        source_inferred = True

    out: Dict[str, object] = {
        "feature": f,
        "feature_source": feature_source,
        "feature_group": feature_group,
        "source_inferred": source_inferred,
        "source_inference_confidence": source_inference_confidence,
        "feature_type": None,
        "feature_subtype": None,
        "summary_stat": None,
        "compartment": None,
        "cells": [],
        "states": [],
        "lineages": [],
        "entities": [],
        "metric_params": {},
        "parse_status": "ok",
        "candidate_eligible": True,
        "warnings": [],
    }

    try:
        if feature_group == "NN":
            parsed = _parse_nn(f, feature_source)

        elif feature_group == "cell_features":
            parsed = _parse_cell_features(f, feature_source)

        elif feature_group == "triads":
            parsed = _parse_triads(f, feature_source)

        elif feature_group == "athena":
            parsed = _parse_athena(f, feature_source)

        else:
            raise ValueError(
                f"unknown feature_group={feature_group!r}"
            )

        out.update(parsed)

    except Exception as exc:
        out["parse_status"] = "unparsed"
        out["warnings"].append(
            f"{type(exc).__name__}: {exc}"
        )
        return out

    entities = out.get("entities", []) or []

    cells = _dedupe_keep_order(
        [e.get("cell") for e in entities]
    )
    states = _dedupe_keep_order(
        [e.get("state") for e in entities]
    )
    lineages = _dedupe_keep_order(
        [CELL_LINEAGE.get(c) for c in cells]
    )

    warnings: List[str] = list(out.get("warnings", []))
    for entity in entities:
        warnings.extend(entity.get("warnings", []))

    out["cells"] = cells
    out["states"] = states
    out["lineages"] = lineages
    out["warnings"] = _dedupe_keep_order(warnings)

    if any(
        str(w).startswith(
            (
                "unparsed",
                "state_missing",
                "checkpoint_root_contains",
                "state_only_entity",
                "state_token_in_nonstate",
            )
        )
        for w in out["warnings"]
    ):
        out["parse_status"] = "partial"

    # -------------------------------------------------------------------------
    # Structural keys for future rescue/microcompression
    # -------------------------------------------------------------------------
    role_cells = tuple(
        (e.get("role"), e.get("cell"))
        for e in entities
    )
    role_cells_states = tuple(
        (e.get("role"), e.get("cell"), e.get("state"))
        for e in entities
    )

    params_tuple = tuple(
        sorted(
            (
                str(k),
                json.dumps(v, sort_keys=True, default=str),
            )
            for k, v in (out.get("metric_params") or {}).items()
        )
    )

    # Same biological measurement after stripping state.
    state_base = (
        out["feature_type"],
        out["feature_subtype"],
        role_cells,
        params_tuple,
    )

    out["state_rescue_key"] = repr(
        (
            state_base,
            out["compartment"],
            out["summary_stat"],
        )
    )

    # Same measurement except for summary statistic.
    out["metric_rescue_key"] = repr(
        (
            state_base,
            out["compartment"],
        )
    )

    # Same measurement except for tissue compartment.
    out["compartment_rescue_key"] = repr(
        (
            state_base,
            out["summary_stat"],
        )
    )

    # Full structured identity.
    out["exact_semantic_key"] = repr(
        (
            out["feature_type"],
            out["feature_subtype"],
            role_cells_states,
            out["compartment"],
            out["summary_stat"],
            params_tuple,
        )
    )

    return out


def parse_feature_row(
    row: Mapping[str, object],
) -> Dict[str, object]:
    feature = str(row.get("feature", ""))

    if not feature and row.get("feature_uid"):
        _, _, feature = split_feature_uid(
            str(row.get("feature_uid"))
        )

    return parse_feature(
        feature=feature,
        feature_source=(
            str(row.get("feature_source"))
            if row.get("feature_source") is not None
            else None
        ),
        feature_group=(
            str(row.get("feature_group"))
            if row.get("feature_group") is not None
            else None
        ),
    )


def flatten_parse_result(
    parsed: Mapping[str, object],
) -> Dict[str, object]:
    """Flatten nested parser output into CSV-friendly columns."""
    out = dict(parsed)

    entities = out.pop("entities", [])
    metric_params = out.pop("metric_params", {})
    warnings = out.pop("warnings", [])
    cells = out.pop("cells", [])
    states = out.pop("states", [])
    lineages = out.pop("lineages", [])

    out["cells"] = ";".join(cells)
    out["states"] = ";".join(states)
    out["lineages"] = ";".join(lineages)
    out["n_cells"] = len(cells)
    out["n_states"] = len(states)
    out["n_lineages"] = len(lineages)

    out["entities_json"] = json.dumps(
        entities,
        sort_keys=True,
        default=str,
    )
    out["metric_params_json"] = json.dumps(
        metric_params,
        sort_keys=True,
        default=str,
    )
    out["warnings"] = ";".join(warnings)

    return out


def parse_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the grammar-aware parser to a candidate table."""
    rows = []

    for _, row in df.iterrows():
        parsed = parse_feature_row(row)
        flat = flatten_parse_result(parsed)

        keep = row.to_dict()
        # Avoid duplicate source/group/feature columns from parser output.
        for col in (
            "feature",
            "feature_source",
            "feature_group",
        ):
            flat.pop(col, None)

        keep.update(flat)
        rows.append(keep)

    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="CSV containing feature, feature_source, feature_group columns.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output parsed CSV.",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    parsed = parse_dataframe(df)
    parsed.to_csv(out, index=False)

    print(
        parsed["parse_status"]
        .value_counts(dropna=False)
        .to_string()
    )
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    main()
