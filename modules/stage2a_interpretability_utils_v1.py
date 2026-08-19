#!/usr/bin/env python3
"""
stage2a_interpretability_utils_v1.py

Shared parsing and interpretability helpers for Stage 2A-4 and Stage 2A-5.
The parser is deliberately rule-based and auditable. It recognizes the feature
name patterns used by the global-module pipeline and exposes canonical keys for:
  * state simplification;
  * metric-summary simplification;
  * compartment simplification; and
  * residual within-family redundancy compression.

Unknown feature names are retained with conservative generic keys rather than
being silently discarded.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SUMMARY_PRIORITY = {
    "median": 0,
    "mean": 1,
    "q1": 2,
    "q3": 2,
    "min": 3,
    "max": 3,
    # SD measures dispersion rather than location and is kept in its own class.
    "sd": 10,
    "auc": 0,
    "peak_abs": 0,
    "pct_non_na": 0,
    "none": 0,
}

COMPARTMENT_PRIORITY = {
    "all": 0,
    "tumor": 1,
    "epi": 1,
    "epithelial": 1,
    "stroma": 2,
    "str": 2,
    "unknown": 9,
}

SOURCE_PRIORITY = {
    "phenotype_only": 0,
    "compartment": 1,
    "ar_state": 2,
    "ar_checkpoint_state": 3,
    "compartment_state": 4,
}

# Only adaptive-resistance/checkpoint state tokens are stripped. Lineage-defining
# tokens such as FoxP3/Treg are intentionally preserved.
STATE_PATTERNS = [
    r"pd[\-_ ]?1(?:\+|pos|positive|state)?",
    r"pd[\-_ ]?l[\-_ ]?1(?:\+|pos|positive|state)?",
    r"pd1[\-_ ]?pdl1(?:\+|pos|positive|state)?",
    r"pdl1[\-_ ]?pd1(?:\+|pos|positive|state)?",
    r"checkpoint[\-_ ]?(?:neg|negative|pos|positive|state)?",
    r"ckpt[\-_ ]?(?:neg|negative|pos|positive|state)?",
    r"ar[\-_ ]?state",
    r"state[\-_ ]?(?:pos|positive|neg|negative)",
]


def _clean_token(text: object) -> str:
    s = str(text).strip().lower()
    s = s.replace("+", "_pos").replace("−", "-")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def canonical_compartment(value: object) -> str:
    s = _clean_token(value)
    if s in {"all", "whole", "whole_tissue"}:
        return "all"
    if s in {"tumor", "tumour", "epi", "epithelial"}:
        return "tumor"
    if s in {"stroma", "stromal", "str"}:
        return "stroma"
    return "unknown"


def _state_hits(text: str) -> List[str]:
    low = str(text).lower()
    hits: List[str] = []
    for pattern in STATE_PATTERNS:
        for match in re.finditer(pattern, low, flags=re.IGNORECASE):
            hits.append(_clean_token(match.group(0)))
    return sorted(set(hits))


def strip_checkpoint_state(text: object) -> str:
    original = str(text)
    # Preserve the biologically meaningful ALL_NEG cell class.
    sentinel = "__ALL_NEG_SENTINEL__"
    s = re.sub(r"all[\-_ ]?neg", sentinel, original, flags=re.IGNORECASE)
    for pattern in STATE_PATTERNS:
        s = re.sub(pattern, " ", s, flags=re.IGNORECASE)
    s = s.replace(sentinel, "all_neg")
    s = re.sub(r"(?:^|[\-_ ])(?:pos|positive|neg|negative)(?:$|[\-_ ])", " ", s, flags=re.IGNORECASE)
    return _clean_token(s)


def state_complexity(feature_source: object, entities: Sequence[str], feature: object) -> int:
    fs = _clean_token(feature_source)
    text = " ".join(list(entities) + [str(feature), str(feature_source)])
    n_hits = len(_state_hits(text))
    source_floor = 0
    if fs == "ar_state":
        source_floor = 1
    elif fs == "ar_checkpoint_state":
        source_floor = 2
    elif fs == "compartment_state":
        source_floor = 2
    return max(n_hits, source_floor)


def summary_class(summary: object) -> str:
    s = _clean_token(summary)
    if s in {"mean", "median", "q1", "q3", "min", "max"}:
        return "location"
    if s == "sd":
        return "dispersion"
    return s


def summary_priority(summary: object) -> int:
    return int(SUMMARY_PRIORITY.get(_clean_token(summary), 9))


def compartment_priority(compartment: object) -> int:
    return int(COMPARTMENT_PRIORITY.get(canonical_compartment(compartment), 9))


def source_priority(feature_source: object) -> int:
    return int(SOURCE_PRIORITY.get(_clean_token(feature_source), 8))


@dataclass
class ParsedFeature:
    feature_uid: str
    feature_source: str
    feature_group: str
    feature: str
    feature_kind: str
    metric_kind: str
    entities_raw: Tuple[str, ...]
    entities_parent: Tuple[str, ...]
    state_signature: str
    state_complexity: int
    compartment: str
    compartment_priority: int
    summary_stat: str
    summary_class: str
    summary_priority: int
    source_priority: int
    parser_status: str
    full_semantic_key: str
    state_simplification_key: str
    metric_simplification_key: str
    compartment_simplification_key: str
    residual_microfamily_key: str

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["entities_raw"] = ";".join(self.entities_raw)
        out["entities_parent"] = ";".join(self.entities_parent)
        return out


def split_feature_uid(uid: object, feature_source: object = "", feature_group: object = "", feature: object = "") -> Tuple[str, str, str]:
    text = str(uid)
    parts = text.split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return str(feature_source), str(feature_group), str(feature)


def _key(parts: Sequence[object]) -> str:
    return "||".join(_clean_token(x) for x in parts)


def parse_feature(
    feature_uid: object,
    feature_source: object = "",
    feature_group: object = "",
    feature: object = "",
) -> ParsedFeature:
    fs, fg, feat = split_feature_uid(feature_uid, feature_source, feature_group, feature)
    fg_clean = _clean_token(fg)
    f = str(feat)

    kind = "generic"
    metric = fg_clean
    entities: Tuple[str, ...] = tuple()
    compartment = "unknown"
    summary = "none"
    status = "generic_fallback"

    # 1-NN: directional from -> to and a distribution summary.
    m = re.match(r"^(.+?)_to_(.+?)_(Mean|SD|Max|Min|Median|Q1|Q3)$", f, flags=re.IGNORECASE)
    if m:
        kind = "nn_distance"
        metric = "nn_distance"
        entities = (_clean_token(m.group(1)), _clean_token(m.group(2)))
        summary = _clean_token(m.group(3))
        status = "parsed_nn"
    else:
        m = re.match(r"^inter_(diff|z|p)_(.+?)__(.+?)__(Tumor|Stroma|All)$", f, flags=re.IGNORECASE)
        if m:
            kind = "athena_interaction"
            metric = "athena_inter_" + _clean_token(m.group(1))
            entities = (_clean_token(m.group(2)), _clean_token(m.group(3)))
            compartment = canonical_compartment(m.group(4))
            status = "parsed_athena_interaction"
        else:
            m = re.match(r"^infiltration_(.+?)__(Tumor|Stroma|All)__(min|mean|median|max|pct_non_na)$", f, flags=re.IGNORECASE)
            if m:
                kind = "athena_infiltration"
                metric = "athena_infiltration"
                entities = (_clean_token(m.group(1)),)
                compartment = canonical_compartment(m.group(2))
                summary = _clean_token(m.group(3))
                status = "parsed_athena_infiltration"
            else:
                m = re.match(r"^ripley_(.+?)_(translation|border|ripley|none)_(.+?)__(Tumor|Stroma|All)__(at\d+|peak_abs|auc)$", f, flags=re.IGNORECASE)
                if m:
                    kind = "athena_ripley"
                    metric = "athena_ripley_" + _clean_token(m.group(2))
                    entities = (_clean_token(m.group(1)), _clean_token(m.group(3)))
                    compartment = canonical_compartment(m.group(4))
                    summary = _clean_token(m.group(5))
                    status = "parsed_athena_ripley"
                else:
                    m = re.match(r"^(All|Epi|Stroma)__ratio__(.+)__over__(.+)$", f, flags=re.IGNORECASE)
                    if m:
                        kind = "composition_ratio"
                        metric = "composition_ratio"
                        compartment = canonical_compartment(m.group(1))
                        entities = (_clean_token(m.group(2)), _clean_token(m.group(3)))
                        status = "parsed_composition_ratio"
                    else:
                        m = re.match(r"^(All|Epi|Stroma)__(prop|density)__(.+)$", f, flags=re.IGNORECASE)
                        if m:
                            kind = "composition_" + _clean_token(m.group(2))
                            metric = kind
                            compartment = canonical_compartment(m.group(1))
                            entities = (_clean_token(m.group(3)),)
                            status = "parsed_composition_single"
                        elif f.strip().startswith("("):
                            try:
                                end = f.rfind(")")
                                vals = ast.literal_eval(f[: end + 1])
                                if not isinstance(vals, tuple):
                                    vals = tuple(vals)
                                entities = tuple(_clean_token(v) for v in vals)
                                suffix = f[end + 1 :].strip(" _-")
                                compartment = canonical_compartment(suffix)
                                kind = "triad"
                                metric = "triad"
                                status = "parsed_triad"
                            except Exception:
                                pass

    if not entities:
        # Conservative generic entity: the complete name. This avoids accidental
        # collapsing of unknown feature formats.
        entities = (_clean_token(f),)

    parent_entities = tuple(strip_checkpoint_state(x) for x in entities)
    state_hits = _state_hits(" ".join(list(entities) + [f, fs]))
    state_sig = ";".join(state_hits) if state_hits else "base"
    st_complexity = state_complexity(fs, entities, f)
    comp = canonical_compartment(compartment)
    summ = _clean_token(summary)
    summ_class = summary_class(summ)

    # All keys preserve direction by preserving entity order.
    full_key = _key([kind, metric, *entities, comp, summ])
    state_key = _key([kind, metric, *parent_entities, comp, summ])
    metric_key = _key([kind, metric, *entities, comp, summ_class])
    compartment_key = _key([kind, metric, *entities, summ])
    residual_key = _key([kind, metric, *entities, comp, summ])

    return ParsedFeature(
        feature_uid=str(feature_uid),
        feature_source=str(fs),
        feature_group=str(fg),
        feature=str(f),
        feature_kind=kind,
        metric_kind=metric,
        entities_raw=entities,
        entities_parent=parent_entities,
        state_signature=state_sig,
        state_complexity=int(st_complexity),
        compartment=comp,
        compartment_priority=compartment_priority(comp),
        summary_stat=summ,
        summary_class=summ_class,
        summary_priority=summary_priority(summ),
        source_priority=source_priority(fs),
        parser_status=status,
        full_semantic_key=full_key,
        state_simplification_key=state_key,
        metric_simplification_key=metric_key,
        compartment_simplification_key=compartment_key,
        residual_microfamily_key=residual_key,
    )


def add_interpretability_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for _, row in df.iterrows():
        uid = row.get("feature_uid")
        if pd.isna(uid) or str(uid).strip() == "":
            uid = "{}|{}|{}".format(
                row.get("feature_source", ""), row.get("feature_group", ""), row.get("feature", "")
            )
        rows.append(
            parse_feature(
                uid,
                feature_source=row.get("feature_source", ""),
                feature_group=row.get("feature_group", ""),
                feature=row.get("feature", ""),
            ).to_dict()
        )
    parsed = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in parsed.columns:
        if col in {"feature_uid", "feature_source", "feature_group", "feature"}:
            continue
        out[col] = parsed[col]
    return out


def safe_spearman(x: pd.Series, y: pd.Series, min_n: int = 10) -> Tuple[float, int]:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    n = int(pair.shape[0])
    if n < int(min_n):
        return np.nan, n
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, n
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")), n


def is_exact_vector_duplicate(x: pd.Series, y: pd.Series, atol: float = 1e-12) -> Tuple[bool, int]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    same_missing = x_num.isna().equals(y_num.isna())
    pair = pd.concat([x_num, y_num], axis=1).dropna()
    n = int(pair.shape[0])
    if not same_missing or n == 0:
        return False, n
    return bool(np.allclose(pair.iloc[:, 0].values, pair.iloc[:, 1].values, atol=atol, rtol=0.0)), n
