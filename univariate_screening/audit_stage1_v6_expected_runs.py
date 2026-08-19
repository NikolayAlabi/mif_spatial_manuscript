#!/usr/bin/env python3
"""
audit_stage1_v6_expected_runs.py

Builds an EXPECTED Stage-1 v6 run grid and compares it to output files
actually present under stage1_univariate_v6/results.

This is different from a simple file scan: a file scan only tells you what
exists. This script also lists runs that are missing or incomplete.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from itertools import product
import pandas as pd

FEATURE_SOURCE_PANELS = {
    "phenotype_only": ["AR", "BT"],
    "AR_state": ["AR"],
    "AR_checkpoint_state": ["AR"],
    "compartment": ["AR", "BT"],
    "compartment_state": ["AR"],
}

FEATURE_SOURCES = list(FEATURE_SOURCE_PANELS.keys())
FEATURE_GROUPS = ["NN", "athena", "cell_features", "triads"]
ENDPOINTS = ["complete_response", "any_response", "OS", "RFS"]
SURVIVAL_ENDPOINTS = {"OS", "RFS"}
RESPONSE_ENDPOINTS = {"complete_response", "any_response"}
NO_RESPONSE_COHORTS = {"No-NAC", "KOLL"}
NO_ADJ_COHORTS = {"No-NAC", "KOLL"}
REQUIRED_SUFFIXES = ["summary", "fullmodels", "feature_filter"]
ALL_SUFFIXES = ["summary", "fullmodels", "feature_filter", "failures", "folds", "oof"]


def split_csv(x: str | None, default: list[str]) -> list[str]:
    if x is None or str(x).strip() == "":
        return list(default)
    return [v.strip() for v in str(x).split(",") if v.strip()]


def infer_phase(sample_type: str, agg: str, transform_mode: str) -> str:
    if agg == "median" and transform_mode == "zscore" and sample_type == "TURBT":
        return "primary_turbt"
    if agg == "median" and transform_mode == "zscore" and sample_type == "RC":
        return "primary_rc"
    if agg == "median" and transform_mode == "log1p_zscore" and sample_type == "TURBT":
        return "log1p_turbt"
    if agg == "median" and transform_mode == "log1p_zscore" and sample_type == "RC":
        return "log1p_rc"
    if agg in {"mean", "max", "min"} and transform_mode == "zscore":
        return "other_aggs_zscore"
    if agg in {"mean", "max", "min"} and transform_mode == "log1p_zscore":
        return "other_aggs_log1p"
    return "custom_or_unknown"


def infer_context_from_path(path: Path, root: Path) -> dict:
    parts = list(path.parts)
    ctx = {"path": str(path)}
    fs_indices = [i for i, p in enumerate(parts) if p in FEATURE_SOURCES]
    if fs_indices:
        i = fs_indices[-1]
        names = ["feature_source", "panel", "feature_group", "cohort", "endpoint", "sample_type"]
        for j, name in enumerate(names):
            k = i + j
            if k < len(parts):
                ctx[name] = parts[k]
        for p in parts[i + 1:]:
            if p.startswith("agg-"):
                ctx["agg"] = p.replace("agg-", "", 1)
            elif p.startswith("patient_subset-"):
                ctx["patient_subset"] = p.replace("patient_subset-", "", 1)
            elif p.startswith("transform-"):
                ctx["transform_mode"] = p.replace("transform-", "", 1)
    m = re.search(r"__(summary|fullmodels|feature_filter|failures|folds|oof)\.csv$", path.name)
    ctx["suffix"] = m.group(1) if m else None
    return ctx


def scan_actual(root: Path) -> pd.DataFrame:
    rows = []
    for fp in root.rglob("*.csv"):
        if re.search(r"__(summary|fullmodels|feature_filter|failures|folds|oof)\.csv$", fp.name):
            rows.append(infer_context_from_path(fp, root))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_actual_wide(actual: pd.DataFrame) -> pd.DataFrame:
    if actual.empty:
        cols = ["cohort", "panel", "feature_source", "feature_group", "endpoint", "sample_type", "patient_subset", "agg", "transform_mode"] + ALL_SUFFIXES
        return pd.DataFrame(columns=cols)

    keys = [
        "cohort", "panel", "feature_source", "feature_group", "endpoint",
        "sample_type", "patient_subset", "agg", "transform_mode",
    ]
    wide = (
        actual.groupby(keys + ["suffix"], dropna=False)
        .size()
        .reset_index(name="n_files")
        .pivot_table(index=keys, columns="suffix", values="n_files", fill_value=0, aggfunc="sum")
        .reset_index()
    )
    for c in ALL_SUFFIXES:
        if c not in wide.columns:
            wide[c] = 0
    return wide


def build_expected(args) -> pd.DataFrame:
    cohorts = split_csv(args.cohorts, ["NAC2020", "PURE01", "BLASST", "No-NAC", "NAC2015", "KOLL"])
    endpoints = split_csv(args.endpoints, ENDPOINTS)
    sample_types = split_csv(args.sample_types, ["TURBT", "RC"])
    aggs = split_csv(args.aggs, ["median", "mean", "max", "min"])
    transforms = split_csv(args.transform_modes, ["zscore", "log1p_zscore"])
    feature_sources = split_csv(args.feature_sources, FEATURE_SOURCES)
    feature_groups = split_csv(args.feature_groups, FEATURE_GROUPS)

    rows = []
    for fs in feature_sources:
        panels = FEATURE_SOURCE_PANELS.get(fs, [])
        for panel, fg, cohort, endpoint, sample_type, agg, transform in product(
            panels, feature_groups, cohorts, endpoints, sample_types, aggs, transforms
        ):
            if cohort in NO_RESPONSE_COHORTS and endpoint in RESPONSE_ENDPOINTS:
                continue

            if args.mirror_auto_patient_subsets:
                patient_subsets = ["all"]
                if cohort in NO_ADJ_COHORTS and endpoint in SURVIVAL_ENDPOINTS:
                    patient_subsets.append("no_adj_chemo")
            else:
                patient_subsets = split_csv(args.patient_subsets, ["all", "no_adj_chemo"])

            for patient_subset in patient_subsets:
                rows.append({
                    "cohort": cohort,
                    "panel": panel,
                    "feature_source": fs,
                    "feature_group": fg,
                    "endpoint": endpoint,
                    "sample_type": sample_type,
                    "patient_subset": patient_subset,
                    "agg": agg,
                    "transform_mode": transform,
                    "phase": infer_phase(sample_type, agg, transform),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/results")
    ap.add_argument("--outdir", default="/projects/ovcare/users/nikolay_alabi/immuno/stage1_univariate_v6/audits")
    ap.add_argument("--cohorts", default="NAC2020,PURE01,BLASST,No-NAC,NAC2015,KOLL")
    ap.add_argument("--endpoints", default=",".join(ENDPOINTS))
    ap.add_argument("--sample-types", default="TURBT,RC")
    ap.add_argument("--aggs", default="median,mean,max,min")
    ap.add_argument("--transform-modes", default="zscore,log1p_zscore")
    ap.add_argument("--feature-sources", default=",".join(FEATURE_SOURCES))
    ap.add_argument("--feature-groups", default=",".join(FEATURE_GROUPS))
    ap.add_argument("--patient-subsets", default="all,no_adj_chemo")
    ap.add_argument("--mirror-auto-patient-subsets", action="store_true", default=True,
                    help="Mirror the v6 submitter: no_adj_chemo only for No-NAC/KOLL survival; all otherwise.")
    ap.add_argument("--no-mirror-auto-patient-subsets", dest="mirror_auto_patient_subsets", action="store_false")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    actual = scan_actual(root)
    actual_wide = build_actual_wide(actual)
    expected = build_expected(args)

    keys = [
        "cohort", "panel", "feature_source", "feature_group", "endpoint",
        "sample_type", "patient_subset", "agg", "transform_mode",
    ]

    merged = expected.merge(actual_wide, on=keys, how="left")
    for c in ALL_SUFFIXES:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0).astype(int)

    merged["complete_required"] = True
    for c in REQUIRED_SUFFIXES:
        merged["complete_required"] &= merged[c] > 0
    merged["has_any_output"] = merged[ALL_SUFFIXES].sum(axis=1) > 0

    def missing_req(row):
        return ";".join([c for c in REQUIRED_SUFFIXES if row.get(c, 0) <= 0])

    merged["missing_required_suffixes"] = merged.apply(missing_req, axis=1)

    # Status labels
    merged["status"] = "complete"
    merged.loc[~merged["has_any_output"], "status"] = "not_started_or_no_outputs"
    merged.loc[merged["has_any_output"] & ~merged["complete_required"], "status"] = "incomplete"

    phase_summary = (
        merged.groupby(["phase", "sample_type", "agg", "transform_mode", "status"], dropna=False)
        .size()
        .reset_index(name="n_contexts")
        .pivot_table(index=["phase", "sample_type", "agg", "transform_mode"], columns="status", values="n_contexts", fill_value=0, aggfunc="sum")
        .reset_index()
    )
    for c in ["complete", "incomplete", "not_started_or_no_outputs"]:
        if c not in phase_summary.columns:
            phase_summary[c] = 0
    phase_summary["total_expected"] = phase_summary[["complete", "incomplete", "not_started_or_no_outputs"]].sum(axis=1)
    phase_summary["complete_frac"] = phase_summary["complete"] / phase_summary["total_expected"].replace(0, pd.NA)

    combo_summary = (
        merged.groupby(["sample_type", "agg", "transform_mode", "status"], dropna=False)
        .size()
        .reset_index(name="n_contexts")
        .pivot_table(index=["sample_type", "agg", "transform_mode"], columns="status", values="n_contexts", fill_value=0, aggfunc="sum")
        .reset_index()
    )
    for c in ["complete", "incomplete", "not_started_or_no_outputs"]:
        if c not in combo_summary.columns:
            combo_summary[c] = 0
    combo_summary["total_expected"] = combo_summary[["complete", "incomplete", "not_started_or_no_outputs"]].sum(axis=1)
    combo_summary["complete_frac"] = combo_summary["complete"] / combo_summary["total_expected"].replace(0, pd.NA)

    missing = merged[~merged["complete_required"]].copy()

    actual_out = outdir / "stage1_v6_actual_found_outputs.csv"
    expected_out = outdir / "stage1_v6_expected_completion_matrix.csv"
    phase_out = outdir / "stage1_v6_phase_completion_summary.csv"
    combo_out = outdir / "stage1_v6_combo_completion_summary.csv"
    missing_out = outdir / "stage1_v6_missing_or_incomplete_contexts.csv"

    actual.to_csv(actual_out, index=False)
    merged.to_csv(expected_out, index=False)
    phase_summary.to_csv(phase_out, index=False)
    combo_summary.to_csv(combo_out, index=False)
    missing.to_csv(missing_out, index=False)

    print("=" * 100)
    print(f"Stage1 root: {root}")
    print(f"Actual output CSV files found: {0 if actual.empty else actual.shape[0]}")
    print(f"Expected context rows: {merged.shape[0]}")
    print(f"Complete context rows: {int(merged['complete_required'].sum())}")
    print(f"Missing/incomplete context rows: {int((~merged['complete_required']).sum())}")
    print("=" * 100)

    print("\n=== Completion by phase ===")
    print(phase_summary.sort_values(["phase", "sample_type", "agg", "transform_mode"]).to_string(index=False))

    print("\n=== Completion by sample_type / agg / transform ===")
    print(combo_summary.sort_values(["sample_type", "agg", "transform_mode"]).to_string(index=False))

    print("\nSaved:")
    for p in [actual_out, expected_out, phase_out, combo_out, missing_out]:
        print(f"  {p}")


if __name__ == "__main__":
    main()