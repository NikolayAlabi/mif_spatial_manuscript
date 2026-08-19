#!/usr/bin/env python3
"""
stage2a4_filter_build_context_matrices_v2_2.py

Stage 2A-4: apply manually reviewed, context-specific candidate thresholds;
identify simpler structural rescue alternatives; and build one transformed
patient-by-feature matrix per context.

This stage intentionally does NOT perform correlation compression. It creates the
reusable matrices and complete candidate registry consumed by Stage 2A-5.

Commands
--------
init-rules  Create a complete rules CSV from the Stage 2A steps 1-3 context index.
validate    Validate rule coverage and configuration.
worker      Process one context (one CPU; supports SLURM_ARRAY_TASK_ID).
aggregate   Aggregate all completed context outputs.

Examples
--------
python stage2a4_filter_build_context_matrices_v2_2.py init-rules --config CONFIG.json
python stage2a4_filter_build_context_matrices_v2_2.py validate --config CONFIG.json
python stage2a4_filter_build_context_matrices_v2_2.py worker --config CONFIG.json --array-id 0
python stage2a4_filter_build_context_matrices_v2_2.py aggregate --config CONFIG.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


CONTEXT_COLS = ["cohort", "panel", "endpoint", "sample_type", "patient_subset", "agg"]
FEATURE_ID_COLS = ["feature_source", "feature_group", "feature"]

RULE_COLUMNS = [
    "context_id",
    *CONTEXT_COLS,
    "include_context",
    "context_strength",
    "min_oof_metric",
    "min_delta_clinical",
    "max_fold_sd",
    "min_direction_consistency",
    "min_nonmissing_fraction",
    "min_candidate_evidence_score",
    "min_valid_folds",
    "max_nominal_p",
    "max_context_q",
    "max_candidates",
    "enable_state_rescue",
    "enable_metric_rescue",
    "enable_compartment_rescue",
    "rescue_search_max_oof_loss",
    "manual_notes",
]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(obj: Mapping, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # Accommodate parquet fallback from Stage 2A steps 1-3.
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


def import_module_from_path(module_name: str, path: Union[str, Path]):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load module from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "include", "included"}


def optional_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def optional_int(value: object) -> Optional[int]:
    x = optional_float(value)
    return None if x is None else int(x)


def make_feature_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["feature_source"].astype(str)
        + "|"
        + df["feature_group"].astype(str)
        + "|"
        + df["feature"].astype(str)
    )


def get_interpretability_utils(cfg: Mapping):
    path = cfg.get("interpretability_utils_path")
    if path is None:
        path = Path(__file__).with_name("stage2a_interpretability_utils_v2_2.py")
    return import_module_from_path("stage2a_interpretability_utils_for_stage2a4", path)


def stage13_root(cfg: Mapping) -> Path:
    return Path(cfg["stage2a_steps1_3_output_root"])


def context_index_path(cfg: Mapping) -> Path:
    return stage13_root(cfg) / "stage2a_context_index.csv"


def rules_path(cfg: Mapping) -> Path:
    return Path(cfg["context_rules_csv"])


def load_context_index(cfg: Mapping) -> pd.DataFrame:
    path = context_index_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Missing Stage 2A steps 1-3 context index: {}".format(path))
    index = pd.read_csv(path)
    required = {"array_id", "context_id", "context_slug", *CONTEXT_COLS}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError("Context index missing columns: {}".format(missing))
    return index


def load_rules(cfg: Mapping) -> pd.DataFrame:
    path = rules_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Missing context rules CSV: {}".format(path))
    rules = pd.read_csv(path)
    missing = sorted(set(RULE_COLUMNS) - set(rules.columns))
    if missing:
        raise ValueError("Rules CSV missing required columns: {}".format(missing))
    if rules["context_id"].duplicated().any():
        dup = rules.loc[rules["context_id"].duplicated(False), "context_id"].tolist()
        raise ValueError("Duplicate context_id values in rules CSV: {}".format(dup[:10]))
    return rules


def best_transform_path(cfg: Mapping, index_row: pd.Series) -> Path:
    return stage13_root(cfg) / "contexts" / str(index_row["context_slug"]) / "best_transform_features.parquet"


def context_output_dir(cfg: Mapping, index_row: pd.Series) -> Path:
    return Path(cfg["output_root"]) / "contexts" / str(index_row["context_slug"])


def default_rules_row(row: pd.Series) -> dict:
    endpoint = str(row["endpoint"])
    is_response = endpoint in {"complete_response", "any_response"}
    return {
        "context_id": row["context_id"],
        **{c: row[c] for c in CONTEXT_COLS},
        "include_context": 0,
        "context_strength": "REVIEW_REQUIRED",
        "min_oof_metric": 0.58 if is_response else 0.56,
        "min_delta_clinical": "",
        "max_fold_sd": 0.15,
        "min_direction_consistency": 0.70,
        "min_nonmissing_fraction": 0.50,
        "min_candidate_evidence_score": "",
        "min_valid_folds": 4,
        "max_nominal_p": "",
        "max_context_q": "",
        "max_candidates": 50,
        "enable_state_rescue": 1,
        "enable_metric_rescue": 1,
        "enable_compartment_rescue": 1,
        "rescue_search_max_oof_loss": 0.05,
        "manual_notes": "EDIT AFTER REVIEW; p/q thresholds intentionally blank",
    }


def command_init_rules(cfg: Mapping, force: bool = False) -> None:
    index = load_context_index(cfg)
    outpath = rules_path(cfg)
    if outpath.exists() and not force:
        raise FileExistsError("Rules file already exists. Use --force to overwrite: {}".format(outpath))
    rows = [default_rules_row(row) for _, row in index.sort_values("array_id").iterrows()]
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS)
    ensure_dir(outpath.parent)
    rules.to_csv(outpath, index=False)
    log("[SAVE] {} rows={}".format(outpath, len(rules)))
    log("[IMPORTANT] All contexts default to include_context=0. Review and edit before workers are run.")


def validate_rules(index: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    index_ids = set(index["context_id"].astype(str))
    rule_ids = set(rules["context_id"].astype(str))
    rows: List[dict] = []
    for context_id in sorted(index_ids | rule_ids):
        in_index = context_id in index_ids
        in_rules = context_id in rule_ids
        status = "ok" if in_index and in_rules else ("missing_rule" if in_index else "extra_rule")
        rows.append({"context_id": context_id, "in_context_index": in_index, "in_rules": in_rules, "coverage_status": status})
    return pd.DataFrame(rows)


def command_validate(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    index = load_context_index(cfg)
    rules = load_rules(cfg)
    coverage = validate_rules(index, rules)
    coverage.to_csv(output_root / "context_rule_coverage_audit.csv", index=False)

    problems: List[str] = []

    iu_path = Path(
        cfg.get(
            "interpretability_utils_path",
            Path(__file__).with_name("stage2a_interpretability_utils_v2_2.py"),
        )
    )
    parser_path = Path(
        cfg.get(
            "feature_parser_path",
            Path(__file__).with_name("stage2_feature_parser_v8_2.py"),
        )
    )
    if not iu_path.exists():
        problems.append("missing interpretability_utils_path: {}".format(iu_path))
    if not parser_path.exists():
        problems.append("missing feature_parser_path: {}".format(parser_path))

    if (coverage["coverage_status"] == "missing_rule").any():
        problems.append("one or more Stage 2A-3 contexts are missing from the rules CSV")

    for _, r in rules.iterrows():
        if not parse_bool(r["include_context"]):
            continue
        if optional_float(r["min_oof_metric"]) is None:
            problems.append("included context {} has blank min_oof_metric".format(r["context_id"]))
        if optional_int(r["max_candidates"]) is not None and optional_int(r["max_candidates"]) < 1:
            problems.append("included context {} has max_candidates < 1".format(r["context_id"]))

    report = {
        "n_contexts": int(len(index)),
        "n_rules": int(len(rules)),
        "n_included": int(rules["include_context"].map(parse_bool).sum()),
        "n_excluded": int((~rules["include_context"].map(parse_bool)).sum()),
        "problems": problems,
    }
    write_json(report, output_root / "stage2a4_validation_report.json")
    if problems:
        raise ValueError("Stage 2A-4 validation failed:\n- " + "\n- ".join(problems))
    log("[VALID] contexts={} included={}".format(report["n_contexts"], report["n_included"]))


def apply_context_thresholds(features: pd.DataFrame, rule: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = features.copy()
    if "feature_uid" not in df.columns:
        df["feature_uid"] = make_feature_uid(df)

    specifications = [
        ("oof_metric", "min", optional_float(rule["min_oof_metric"])),
        ("delta_clinical", "min", optional_float(rule["min_delta_clinical"])),
        ("fold_sd", "max", optional_float(rule["max_fold_sd"])),
        ("direction_consistency", "min", optional_float(rule["min_direction_consistency"])),
        ("nonmissing_fraction", "min", optional_float(rule["min_nonmissing_fraction"])),
        ("candidate_evidence_score", "min", optional_float(rule["min_candidate_evidence_score"])),
        ("valid_folds", "min", optional_float(rule["min_valid_folds"])),
        ("p_value", "max", optional_float(rule["max_nominal_p"])),
        ("context_q_value", "max", optional_float(rule["max_context_q"])),
    ]

    pass_columns: List[str] = []
    for column, direction, threshold in specifications:
        flag_col = "pass_{}".format(column)
        pass_columns.append(flag_col)
        if threshold is None:
            df[flag_col] = True
            continue
        if column not in df.columns:
            df[flag_col] = False
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if direction == "min":
            df[flag_col] = values.notna() & (values >= threshold)
        else:
            df[flag_col] = values.notna() & (values <= threshold)

    df["passes_context_thresholds"] = df[pass_columns].all(axis=1)

    def reasons(row: pd.Series) -> str:
        failed = [col.replace("pass_", "") for col in pass_columns if not bool(row[col])]
        return ";".join(failed)

    df["threshold_failure_reasons"] = df.apply(reasons, axis=1)
    seeds = df[df["passes_context_thresholds"]].copy()
    seeds = seeds.sort_values(
        ["candidate_evidence_score", "oof_metric", "fold_sd", "nonmissing_fraction"],
        ascending=[False, False, True, False],
        na_position="last",
    )
    max_candidates = optional_int(rule["max_candidates"])
    if max_candidates is not None:
        seeds = seeds.head(max_candidates).copy()
    seeds["seed_rank"] = np.arange(1, len(seeds) + 1)
    selected_uids = set(seeds["feature_uid"].astype(str))
    df["selected_as_seed"] = df["feature_uid"].astype(str).isin(selected_uids)
    return seeds.reset_index(drop=True), df.reset_index(drop=True)


def identify_rescue_alternatives(
    all_features: pd.DataFrame,
    seeds: pd.DataFrame,
    rule: pd.Series,
    cfg: Mapping,
    iu,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seeds.empty:
        return pd.DataFrame(columns=all_features.columns), pd.DataFrame()

    all_parsed = iu.add_interpretability_columns(all_features)
    seeds_parsed = all_parsed[all_parsed["feature_uid"].astype(str).isin(seeds["feature_uid"].astype(str))].copy()
    max_oof_loss = optional_float(rule.get("rescue_search_max_oof_loss"))
    if max_oof_loss is None:
        max_oof_loss = float(cfg.get("rescue_search_max_oof_loss", 0.05))
    max_per_seed = int(cfg.get("max_rescues_per_seed_per_rule", 3))

    enabled = {
        "state_rescue": parse_bool(rule.get("enable_state_rescue")),
        "metric_rescue": parse_bool(rule.get("enable_metric_rescue")),
        "compartment_rescue": parse_bool(rule.get("enable_compartment_rescue")),
    }

    link_rows: List[dict] = []
    for _, seed in seeds_parsed.iterrows():
        seed_uid = str(seed["feature_uid"])
        seed_oof = pd.to_numeric(pd.Series([seed.get("oof_metric")]), errors="coerce").iloc[0]

        rule_specs = []
        if enabled["state_rescue"]:
            rule_specs.append((
                "state_rescue", "state_simplification_key", "state_complexity", int(seed["state_complexity"])
            ))
        if enabled["metric_rescue"] and str(seed["summary_class"]) == "location":
            rule_specs.append((
                "metric_rescue", "metric_simplification_key", "summary_priority", int(seed["summary_priority"])
            ))
        if enabled["compartment_rescue"] and str(seed["compartment"]) != "all":
            rule_specs.append((
                "compartment_rescue", "compartment_simplification_key", "compartment_priority", int(seed["compartment_priority"])
            ))

        for rescue_rule, key_col, priority_col, seed_priority in rule_specs:
            candidates = all_parsed[
                (all_parsed[key_col].astype(str) == str(seed[key_col]))
                & (pd.to_numeric(all_parsed[priority_col], errors="coerce") < seed_priority)
                & (all_parsed["feature_uid"].astype(str) != seed_uid)
            ].copy()
            if candidates.empty:
                continue
            candidates["rescue_oof_loss_vs_seed"] = seed_oof - pd.to_numeric(candidates["oof_metric"], errors="coerce")
            candidates = candidates[
                candidates["rescue_oof_loss_vs_seed"].notna()
                & (candidates["rescue_oof_loss_vs_seed"] <= max_oof_loss)
            ].copy()
            if candidates.empty:
                continue
            candidates = candidates.sort_values(
                [priority_col, "candidate_evidence_score", "oof_metric", "fold_sd"],
                ascending=[True, False, False, True],
                na_position="last",
            ).head(max_per_seed)
            for _, candidate in candidates.iterrows():
                link_rows.append({
                    "context_id": seed.get("context_id"),
                    "seed_feature_uid": seed_uid,
                    "rescue_feature_uid": candidate["feature_uid"],
                    "rescue_rule": rescue_rule,
                    "matching_key": seed[key_col],
                    "seed_priority": seed_priority,
                    "rescue_priority": candidate[priority_col],
                    "seed_oof_metric": seed_oof,
                    "rescue_oof_metric": candidate.get("oof_metric"),
                    "rescue_oof_loss_vs_seed": candidate["rescue_oof_loss_vs_seed"],
                })

    links = pd.DataFrame(link_rows)
    if links.empty:
        return pd.DataFrame(columns=all_parsed.columns), links
    rescue_uids = set(links["rescue_feature_uid"].astype(str))
    rescues = all_parsed[all_parsed["feature_uid"].astype(str).isin(rescue_uids)].copy()
    role_map = links.groupby("rescue_feature_uid")["rescue_rule"].agg(lambda x: ";".join(sorted(set(x)))).to_dict()
    seed_map = links.groupby("rescue_feature_uid")["seed_feature_uid"].agg(lambda x: ";".join(sorted(set(x)))).to_dict()
    rescues["rescue_roles"] = rescues["feature_uid"].map(role_map)
    rescues["rescues_seed_feature_uids"] = rescues["feature_uid"].map(seed_map)
    return rescues.reset_index(drop=True), links.reset_index(drop=True)


def make_candidate_registry(seeds: pd.DataFrame, rescues: pd.DataFrame, iu) -> pd.DataFrame:
    seed_uids = set(seeds["feature_uid"].astype(str)) if not seeds.empty else set()
    rescue_uids = set(rescues["feature_uid"].astype(str)) if not rescues.empty else set()
    parts = []
    if not seeds.empty:
        parts.append(seeds.copy())
    if not rescues.empty:
        parts.append(rescues.copy())
    if not parts:
        return pd.DataFrame()
    registry = pd.concat(parts, ignore_index=True, sort=False)
    registry = registry.sort_values(
        ["candidate_evidence_score", "oof_metric"], ascending=[False, False], na_position="last"
    ).drop_duplicates("feature_uid", keep="first")
    registry = iu.add_interpretability_columns(registry)
    registry["seed_pass_thresholds"] = registry["feature_uid"].astype(str).isin(seed_uids)
    registry["included_as_rescue"] = registry["feature_uid"].astype(str).isin(rescue_uids)
    registry["candidate_role"] = np.where(
        registry["seed_pass_thresholds"] & registry["included_as_rescue"],
        "seed_and_rescue",
        np.where(registry["seed_pass_thresholds"], "seed", "rescue_only"),
    )
    return registry.reset_index(drop=True)


def resolve_array_id(value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("Provide --array-id or run inside a Slurm array task")
    return int(env)


def get_index_and_rule(cfg: Mapping, array_id: int) -> Tuple[pd.Series, pd.Series]:
    index = load_context_index(cfg)
    match = index[index["array_id"].astype(int) == int(array_id)]
    if match.empty:
        raise IndexError("array_id={} not found".format(array_id))
    index_row = match.iloc[0]
    rules = load_rules(cfg)
    rmatch = rules[rules["context_id"].astype(str) == str(index_row["context_id"])]
    if rmatch.empty:
        raise KeyError("No rule for context_id={}".format(index_row["context_id"]))
    return index_row, rmatch.iloc[0]


def fit_apply_transform_full(x: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mode = str(mode) if pd.notna(mode) else "zscore"
    if mode == "raw":
        return x
    if mode == "log1p_zscore":
        if (x.dropna() < 0).any():
            return pd.Series(np.nan, index=x.index)
        x = np.log1p(x)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def build_patient_matrix_for_source_group_local(
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
    cfg: Mapping,
) -> pd.DataFrame:
    features = list(dict.fromkeys([str(f) for f in features]))
    data_dict = stage1_mod.load_data_dict(
        feature_group=feature_group,
        feature_source=feature_source,
        panels=[panel],
        cohorts=[cohort],
        spatial_root=cfg.get("spatial_root"),
        cell_features_path=cfg.get("cell_features_path"),
        triads_path=cfg.get("triads_path"),
    )
    harm_df = stage1_mod.load_harmonized_df(cfg["harmonized_path"])
    kwargs = dict(
        data_dict=data_dict,
        feature_group=feature_group,
        cohort=cohort,
        panel=panel,
        qc_acceptability=str(cfg.get("qc_acceptability", "acceptable_or_borderline")),
        min_epi_fraction=cfg.get("min_epi_fraction", 0.05),
        sample_type=sample_type,
    )
    if cfg.get("koll_metadata_csv") is not None:
        kwargs["koll_metadata_csv"] = cfg.get("koll_metadata_csv")
    core_df = stage1_mod.prepare_core_level_feature_table(**kwargs)
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
        patient_df = patient_df[patient_df["cohort"].astype(str) == str(cohort)].copy()
    if cohort in {"No-NAC", "KOLL"} and patient_subset in {"no_adj_chemo", "adj_chemo"}:
        patient_df = stage1_mod.apply_patient_subset(patient_df, patient_subset=patient_subset)
    if patient_df.empty:
        raise ValueError("No patients remain after aggregation/subsetting")
    return patient_df[["patient_id"] + present].copy()


def build_patient_matrix(registry: pd.DataFrame, cfg: Mapping, context_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if registry.empty:
        return pd.DataFrame(), pd.DataFrame()
    stage1_mod = import_module_from_path("stage1_univariate_module_for_stage2a4", cfg["stage1_script_path"])
    first = registry.iloc[0]
    cohort = str(first["cohort"])
    panel = str(first["panel"])
    sample_type = str(first.get("sample_type", "TURBT"))
    patient_subset = str(first.get("patient_subset", "all"))
    agg = str(first.get("agg", "median"))
    merged = None
    meta_rows: List[dict] = []
    failure_rows: List[dict] = []

    work = registry.sort_values(
        ["candidate_evidence_score", "oof_metric"], ascending=[False, False], na_position="last"
    ).drop_duplicates("feature_uid", keep="first")

    for (feature_source, feature_group), group in work.groupby(["feature_source", "feature_group"], dropna=False):
        features = group["feature"].dropna().astype(str).unique().tolist()
        try:
            patient_df = build_patient_matrix_for_source_group_local(
                stage1_mod=stage1_mod,
                cohort=cohort,
                panel=panel,
                feature_source=str(feature_source),
                feature_group=str(feature_group),
                features=features,
                sample_type=sample_type,
                patient_subset=patient_subset,
                agg=agg,
                cfg=cfg,
            )
        except Exception as exc:
            failure_rows.append({
                "cohort": cohort,
                "panel": panel,
                "feature_source": feature_source,
                "feature_group": feature_group,
                "reason": "{}: {}".format(type(exc).__name__, exc),
                "n_requested_features": len(features),
            })
            continue

        tmp = patient_df[["patient_id"]].copy()
        for _, feature_row in group.iterrows():
            feature = str(feature_row["feature"])
            uid = str(feature_row["feature_uid"])
            mode = str(feature_row.get("selected_transform_mode", "zscore"))
            if feature not in patient_df.columns:
                failure_rows.append({
                    "cohort": cohort,
                    "panel": panel,
                    "feature_source": feature_source,
                    "feature_group": feature_group,
                    "feature": feature,
                    "feature_uid": uid,
                    "reason": "feature_missing_from_patient_matrix",
                })
                continue
            transformed = fit_apply_transform_full(patient_df[feature], mode)
            tmp[uid] = transformed
            meta_rows.append({
                "cohort": cohort,
                "panel": panel,
                "sample_type": sample_type,
                "patient_subset": patient_subset,
                "agg": agg,
                "feature_uid": uid,
                "feature": feature,
                "feature_source": feature_source,
                "feature_group": feature_group,
                "selected_transform_mode": mode,
                "n_patients": int(len(transformed)),
                "nonmissing_fraction": float(transformed.notna().mean()),
                "n_unique": int(transformed.dropna().nunique()),
            })
        merged = tmp if merged is None else merged.merge(tmp, on="patient_id", how="outer")

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(context_dir / "matrix_build_failures.csv", index=False)
    matrix = merged if merged is not None else pd.DataFrame(columns=["patient_id"])
    return matrix, pd.DataFrame(meta_rows)



def previous_context_dir(cfg: Mapping, index_row: pd.Series) -> Optional[Path]:
    root = cfg.get("reuse_stage2a4_root")
    if root is None or str(root).strip() == "":
        return None
    return Path(root) / "contexts" / str(index_row["context_slug"])


def _find_existing_table(path: Path) -> Optional[Path]:
    candidates = [
        path,
        path.with_suffix(".csv.gz"),
        path.with_suffix(".csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_patient_matrix_incremental(
    registry: pd.DataFrame,
    cfg: Mapping,
    context_dir: Path,
    index_row: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reuse matching feature_uid columns from an earlier Stage 2A-4 matrix and
    reconstruct only newly required rescue columns.

    The previous matrix is read-only. Same feature_uid implies the same
    feature_source / feature_group / feature identity. Stage 2A steps 1-3 are
    unchanged, so selected_transform_mode should also be unchanged.
    """
    if registry.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_uids = list(dict.fromkeys(registry["feature_uid"].astype(str)))
    reuse_dir = previous_context_dir(cfg, index_row)

    old_matrix = pd.DataFrame()
    old_meta = pd.DataFrame()
    reuse_path = None

    if reuse_dir is not None:
        reuse_path = _find_existing_table(
            reuse_dir / "patient_feature_matrix.parquet"
        )
        if reuse_path is not None:
            old_matrix = read_table(reuse_path)
        old_meta_path = reuse_dir / "matrix_feature_meta.csv"
        if old_meta_path.exists():
            old_meta = pd.read_csv(old_meta_path)

    reusable_uids = [
        uid
        for uid in required_uids
        if not old_matrix.empty and uid in old_matrix.columns
    ]
    missing_uids = [
        uid
        for uid in required_uids
        if uid not in set(reusable_uids)
    ]

    audit_rows: List[dict] = []
    for uid in reusable_uids:
        audit_rows.append({
            "feature_uid": uid,
            "matrix_source": "reused_previous_stage2a4",
            "reuse_matrix_path": str(reuse_path) if reuse_path else "",
            "build_attempted": False,
            "build_success": True,
        })

    # Start with reused columns only.
    if reusable_uids:
        matrix = old_matrix[["patient_id"] + reusable_uids].copy()
    else:
        matrix = pd.DataFrame()

    # Reuse prior metadata where available; otherwise generate lightweight meta.
    if reusable_uids and not old_meta.empty and "feature_uid" in old_meta.columns:
        reused_meta = old_meta[
            old_meta["feature_uid"].astype(str).isin(reusable_uids)
        ].copy()
    else:
        reused_meta = pd.DataFrame()

    missing_registry = registry[
        registry["feature_uid"].astype(str).isin(missing_uids)
    ].copy()

    built_matrix = pd.DataFrame()
    built_meta = pd.DataFrame()

    if not missing_registry.empty:
        built_matrix, built_meta = build_patient_matrix(
            missing_registry,
            cfg,
            context_dir,
        )

        built_success = (
            set(built_meta["feature_uid"].astype(str))
            if not built_meta.empty and "feature_uid" in built_meta.columns
            else set()
        )

        for uid in missing_uids:
            audit_rows.append({
                "feature_uid": uid,
                "matrix_source": (
                    "newly_built"
                    if uid in built_success
                    else "build_failed"
                ),
                "reuse_matrix_path": "",
                "build_attempted": True,
                "build_success": uid in built_success,
            })

    # Merge reused + newly built matrices.
    if matrix.empty:
        matrix = built_matrix.copy()
    elif not built_matrix.empty:
        matrix = matrix.merge(
            built_matrix,
            on="patient_id",
            how="outer",
            validate="one_to_one",
        )

    # Build complete metadata table.
    meta_parts = []

    if not reused_meta.empty:
        meta_parts.append(reused_meta)

    # If metadata was missing for reused columns, reconstruct descriptive rows.
    reused_meta_uids = (
        set(reused_meta["feature_uid"].astype(str))
        if not reused_meta.empty and "feature_uid" in reused_meta.columns
        else set()
    )

    for uid in reusable_uids:
        if uid in reused_meta_uids:
            continue
        r = registry[
            registry["feature_uid"].astype(str).eq(uid)
        ].iloc[0]
        vec = old_matrix[uid]
        meta_parts.append(pd.DataFrame([{
            "cohort": r.get("cohort"),
            "panel": r.get("panel"),
            "sample_type": r.get("sample_type"),
            "patient_subset": r.get("patient_subset"),
            "agg": r.get("agg"),
            "feature_uid": uid,
            "feature": r.get("feature"),
            "feature_source": r.get("feature_source"),
            "feature_group": r.get("feature_group"),
            "selected_transform_mode": r.get(
                "selected_transform_mode",
                r.get("transform_mode", "zscore"),
            ),
            "n_patients": int(len(vec)),
            "nonmissing_fraction": float(vec.notna().mean()),
            "n_unique": int(vec.dropna().nunique()),
        }]))

    if not built_meta.empty:
        meta_parts.append(built_meta)

    meta = (
        pd.concat(meta_parts, ignore_index=True, sort=False)
        if meta_parts else pd.DataFrame()
    )
    if not meta.empty and "feature_uid" in meta.columns:
        meta = meta.drop_duplicates("feature_uid", keep="last")

    audit = pd.DataFrame(audit_rows)
    return matrix, meta, audit

def command_worker(cfg: Mapping, array_id: int) -> None:
    index_row, rule = get_index_and_rule(cfg, array_id)
    context_dir = ensure_dir(context_output_dir(cfg, index_row))
    write_json({k: (None if pd.isna(v) else v) for k, v in rule.to_dict().items()}, context_dir / "context_rule.resolved.json")

    context_id = str(index_row["context_id"])
    log("=" * 80)
    log("[STAGE2A-4 context {}] {}".format(array_id, context_id))

    if not parse_bool(rule["include_context"]):
        summary = pd.DataFrame([{
            "array_id": array_id,
            "context_id": context_id,
            **{c: index_row[c] for c in CONTEXT_COLS},
            "context_strength": rule["context_strength"],
            "included": False,
            "status": "excluded_by_manual_rule",
            "n_best_transform_features": 0,
            "n_seed_candidates": 0,
            "n_rescue_candidates": 0,
            "n_matrix_features": 0,
        }])
        summary.to_csv(context_dir / "context_stage2a4_summary.csv", index=False)
        (context_dir / ".done").write_text("excluded\n")
        log("[SKIP] context excluded by rules CSV")
        return

    features = read_table(best_transform_path(cfg, index_row))
    if features.empty:
        raise RuntimeError("Empty best-transform feature table for {}".format(context_id))
    if "feature_uid" not in features.columns:
        features["feature_uid"] = make_feature_uid(features)

    iu = get_interpretability_utils(cfg)

    # Parse the full context feature universe once using the corrected grammar.
    # This is needed because rescue alternatives are searched among all
    # best-transform features, not only among the selected seeds.
    features = iu.add_interpretability_columns(features)

    parser_summary = (
        features.groupby(
            ["feature_source", "feature_group", "parser_status"],
            dropna=False,
        )
        .size()
        .rename("n_features")
        .reset_index()
    )
    parser_summary.to_csv(
        context_dir / "feature_parser_status_summary.csv",
        index=False,
    )

    parser_problems = features[
        features["parser_status"].astype(str).ne("ok")
    ].copy()
    if not parser_problems.empty:
        parser_problems.to_csv(
            context_dir / "feature_parser_problem_features.csv",
            index=False,
        )

    # Explicitly exclude technical/QC support variables from biomarker
    # nomination and rescue searches. Keep them in a dedicated audit.
    if "candidate_eligible" not in features.columns:
        features["candidate_eligible"] = True

    technical_excluded = features[
        ~features["candidate_eligible"].fillna(True).astype(bool)
    ].copy()
    technical_excluded.to_csv(
        context_dir / "technical_non_candidate_features.csv",
        index=False,
    )

    candidate_features = features[
        features["candidate_eligible"].fillna(True).astype(bool)
    ].copy()

    seeds, filter_audit = apply_context_thresholds(
        candidate_features,
        rule,
    )

    # A selected seed must be structurally parseable; otherwise rescue and
    # microcompression logic for that seed would be unreliable.
    bad_seed_parse = seeds[
        seeds["parser_status"].astype(str).ne("ok")
    ]
    if not bad_seed_parse.empty:
        bad_seed_parse.to_csv(
            context_dir / "ERROR_selected_seed_parser_failures.csv",
            index=False,
        )
        raise RuntimeError(
            "{} selected seed(s) failed corrected feature parsing in {}".format(
                len(bad_seed_parse), context_id
            )
        )

    rescues, rescue_links = identify_rescue_alternatives(
        candidate_features,
        seeds,
        rule,
        cfg,
        iu,
    )
    registry = make_candidate_registry(seeds, rescues, iu)

    filter_audit.to_csv(context_dir / "candidate_threshold_filter_audit.csv", index=False)
    seeds.to_csv(context_dir / "seed_candidates.csv", index=False)
    rescues.to_csv(context_dir / "rescue_candidates.csv", index=False)
    rescue_links.to_csv(context_dir / "rescue_candidate_links.csv", index=False)
    registry.to_csv(context_dir / "candidate_registry.csv", index=False)

    if registry.empty:
        summary = pd.DataFrame([{
            "array_id": array_id,
            "context_id": context_id,
            **{c: index_row[c] for c in CONTEXT_COLS},
            "context_strength": rule["context_strength"],
            "included": True,
            "status": "included_but_zero_seed_candidates",
            "n_best_transform_features": int(len(features)),
            "n_seed_candidates": 0,
            "n_rescue_candidates": 0,
            "n_matrix_features": 0,
        }])
        summary.to_csv(context_dir / "context_stage2a4_summary.csv", index=False)
        (context_dir / ".done").write_text("zero_candidates\n")
        log("[DONE] no features passed context thresholds")
        return

    matrix_reuse_audit = pd.DataFrame()
    if bool(cfg.get("skip_matrix_build", False)):
        matrix = pd.DataFrame()
        matrix_meta = pd.DataFrame()
        status = "dry_run_registry_only"
    else:
        matrix, matrix_meta, matrix_reuse_audit = build_patient_matrix_incremental(
            registry,
            cfg,
            context_dir,
            index_row,
        )
        if matrix.empty or matrix.shape[1] <= 1:
            status = "matrix_build_zero_features"
        else:
            status = "complete"
            save_table(
                matrix,
                context_dir / "patient_feature_matrix.parquet",
            )
        matrix_meta.to_csv(
            context_dir / "matrix_feature_meta.csv",
            index=False,
        )
        matrix_reuse_audit.to_csv(
            context_dir / "matrix_reuse_audit.csv",
            index=False,
        )

    registry_uids = set(registry["feature_uid"].astype(str))
    built_uids = set(matrix_meta["feature_uid"].astype(str)) if not matrix_meta.empty else set()
    build_audit = registry[[c for c in ["feature_uid", "feature_source", "feature_group", "feature", "candidate_role"] if c in registry.columns]].copy()
    build_audit["matrix_build_success"] = build_audit["feature_uid"].astype(str).isin(built_uids)
    build_audit["matrix_build_failure_reason"] = np.where(
        build_audit["matrix_build_success"], "", "not_returned_by_matrix_builder"
    )
    build_audit.to_csv(context_dir / "matrix_feature_build_audit.csv", index=False)

    summary = pd.DataFrame([{
        "array_id": array_id,
        "context_id": context_id,
        **{c: index_row[c] for c in CONTEXT_COLS},
        "context_strength": rule["context_strength"],
        "included": True,
        "status": status,
        "n_best_transform_features": int(len(features)),
        "n_passing_raw_thresholds": int(filter_audit["passes_context_thresholds"].sum()),
        "n_seed_candidates": int(len(seeds)),
        "n_rescue_candidates": int(len(set(rescues["feature_uid"])) if not rescues.empty else 0),
        "n_candidate_registry": int(len(registry)),
        "n_matrix_patients": int(matrix.shape[0]) if not matrix.empty else 0,
        "n_matrix_features": int(max(matrix.shape[1] - 1, 0)) if not matrix.empty else 0,
        "n_matrix_build_success": int(len(built_uids)),
        "n_matrix_build_failed": int(len(registry_uids - built_uids)) if not bool(cfg.get("skip_matrix_build", False)) else 0,
        "n_matrix_features_reused": int(
            (
                matrix_reuse_audit.get(
                    "matrix_source",
                    pd.Series(dtype=str),
                ) == "reused_previous_stage2a4"
            ).sum()
        ) if not matrix_reuse_audit.empty else 0,
        "n_matrix_features_newly_built": int(
            (
                matrix_reuse_audit.get(
                    "matrix_source",
                    pd.Series(dtype=str),
                ) == "newly_built"
            ).sum()
        ) if not matrix_reuse_audit.empty else 0,
    }])
    summary.to_csv(context_dir / "context_stage2a4_summary.csv", index=False)
    (context_dir / ".done").write_text(status + "\n")
    log("[DONE] seeds={} rescues={} matrix_features={}".format(len(seeds), len(rescues), summary.iloc[0]["n_matrix_features"]))


def command_aggregate(cfg: Mapping) -> None:
    output_root = ensure_dir(cfg["output_root"])
    index = load_context_index(cfg)
    summary_parts: List[pd.DataFrame] = []
    registry_parts: List[pd.DataFrame] = []
    seed_parts: List[pd.DataFrame] = []
    build_parts: List[pd.DataFrame] = []
    matrix_rows: List[dict] = []
    missing: List[dict] = []

    for _, row in index.sort_values("array_id").iterrows():
        cdir = context_output_dir(cfg, row)
        summary_path = cdir / "context_stage2a4_summary.csv"
        if not summary_path.exists():
            missing.append({"array_id": row["array_id"], "context_id": row["context_id"], "reason": "missing_summary"})
            continue
        summary_parts.append(pd.read_csv(summary_path))
        for filename, collection in [
            ("candidate_registry.csv", registry_parts),
            ("seed_candidates.csv", seed_parts),
            ("matrix_feature_build_audit.csv", build_parts),
        ]:
            path = cdir / filename
            if path.exists():
                df = pd.read_csv(path)
                df["array_id"] = row["array_id"]
                df["context_slug"] = row["context_slug"]
                collection.append(df)
        matrix_path = cdir / "patient_feature_matrix.parquet"
        if not matrix_path.exists() and matrix_path.with_suffix(".csv.gz").exists():
            matrix_path = matrix_path.with_suffix(".csv.gz")
        if matrix_path.exists():
            matrix_rows.append({
                "array_id": row["array_id"],
                "context_id": row["context_id"],
                "context_slug": row["context_slug"],
                **{c: row[c] for c in CONTEXT_COLS},
                "matrix_path": str(matrix_path),
                "candidate_registry_path": str(cdir / "candidate_registry.csv"),
                "matrix_feature_meta_path": str(cdir / "matrix_feature_meta.csv"),
            })

    summary = pd.concat(summary_parts, ignore_index=True, sort=False) if summary_parts else pd.DataFrame()
    registry = pd.concat(registry_parts, ignore_index=True, sort=False) if registry_parts else pd.DataFrame()
    seeds = pd.concat(seed_parts, ignore_index=True, sort=False) if seed_parts else pd.DataFrame()
    build = pd.concat(build_parts, ignore_index=True, sort=False) if build_parts else pd.DataFrame()

    summary.to_csv(output_root / "all_context_stage2a4_summary.csv", index=False)
    save_table(registry, output_root / "all_context_candidate_registry.parquet")
    save_table(seeds, output_root / "all_context_seed_candidates.parquet")
    build.to_csv(output_root / "all_context_matrix_feature_build_audit.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(output_root / "stage2a4_matrix_manifest.csv", index=False)
    pd.DataFrame(missing).to_csv(output_root / "stage2a4_missing_context_outputs.csv", index=False)

    if not registry.empty:
        composition = (
            registry.groupby(["panel", "candidate_role", "feature_group", "feature_source"], dropna=False)
            .agg(n_rows=("feature_uid", "size"), n_unique_features=("feature_uid", "nunique"), n_contexts=("context_id", "nunique"))
            .reset_index()
        )
        composition.to_csv(output_root / "stage2a4_candidate_composition.csv", index=False)

    log("[SAVE] {}".format(output_root / "all_context_stage2a4_summary.csv"))
    log("[DONE] matrices={} missing_context_outputs={}".format(len(matrix_rows), len(missing)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["init-rules", "validate", "worker", "aggregate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "worker":
            p.add_argument("--array-id", type=int, default=None)
        if name == "init-rules":
            p.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    ensure_dir(cfg["output_root"])
    write_json(cfg, Path(cfg["output_root"]) / "stage2a4_config.resolved.json")
    if args.command == "init-rules":
        command_init_rules(cfg, force=args.force)
    elif args.command == "validate":
        command_validate(cfg)
    elif args.command == "worker":
        command_worker(cfg, resolve_array_id(args.array_id))
    elif args.command == "aggregate":
        command_aggregate(cfg)


if __name__ == "__main__":
    main()
