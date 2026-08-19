#!/usr/bin/env python3
"""
make_stage2_manifest_min2ctx_mixed_v7.py

Build a stricter but panel-aware Stage 2B candidate manifest from the Stage 2A
candidate manifest.

Default behavior:
  AR: keep features with >=2 predictive contexts using stricter predictive gates.
  BT: keep features selected in >=2 contexts and predictive in >=1 context using
      relaxed predictive gates, because BT has fewer candidates and the AR-style
      strict gate can over-prune it.

The script also writes a new Stage 2B config JSON pointing to the new manifest
and output root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RESPONSE_ENDPOINTS = {"complete_response", "any_response"}
SURVIVAL_ENDPOINTS = {"OS", "RFS"}


def pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_gate_columns(
    df: pd.DataFrame,
    min_response_auc: float,
    min_survival_cindex: float,
    min_delta: float,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()

    # Stage1/Stage2A column names have changed over versions, so be flexible.
    resp_metric_col = pick_col(out, [
        "biomarker_oof_auc", "oof_auc", "primary_oof_metric", "biomarker_oof_metric",
    ])
    surv_metric_col = pick_col(out, [
        "biomarker_oof_cindex", "oof_cindex", "primary_oof_metric", "biomarker_oof_metric",
    ])
    resp_delta_col = pick_col(out, [
        "delta_oof_auc_vs_clinical", "delta_oof_vs_clinical", "primary_delta_metric",
    ])
    surv_delta_col = pick_col(out, [
        "delta_oof_cindex_vs_clinical", "delta_oof_vs_clinical", "primary_delta_metric",
    ])

    print(f"[{prefix}] response metric col: {resp_metric_col}")
    print(f"[{prefix}] survival metric col: {surv_metric_col}")
    print(f"[{prefix}] response delta col : {resp_delta_col}")
    print(f"[{prefix}] survival delta col : {surv_delta_col}")

    out[f"{prefix}_gate_metric"] = np.nan
    out[f"{prefix}_gate_delta"] = np.nan

    response_mask = out["endpoint"].astype(str).isin(RESPONSE_ENDPOINTS)
    survival_mask = out["endpoint"].astype(str).isin(SURVIVAL_ENDPOINTS)

    if resp_metric_col is not None:
        out.loc[response_mask, f"{prefix}_gate_metric"] = pd.to_numeric(out.loc[response_mask, resp_metric_col], errors="coerce")
    if surv_metric_col is not None:
        out.loc[survival_mask, f"{prefix}_gate_metric"] = pd.to_numeric(out.loc[survival_mask, surv_metric_col], errors="coerce")
    if resp_delta_col is not None:
        out.loc[response_mask, f"{prefix}_gate_delta"] = pd.to_numeric(out.loc[response_mask, resp_delta_col], errors="coerce")
    if surv_delta_col is not None:
        out.loc[survival_mask, f"{prefix}_gate_delta"] = pd.to_numeric(out.loc[survival_mask, surv_delta_col], errors="coerce")

    out[f"{prefix}_passes_predictive_gate"] = False
    out.loc[response_mask, f"{prefix}_passes_predictive_gate"] = (
        (out.loc[response_mask, f"{prefix}_gate_metric"] >= min_response_auc)
        & (out.loc[response_mask, f"{prefix}_gate_delta"] >= min_delta)
    )
    out.loc[survival_mask, f"{prefix}_passes_predictive_gate"] = (
        (out.loc[survival_mask, f"{prefix}_gate_metric"] >= min_survival_cindex)
        & (out.loc[survival_mask, f"{prefix}_gate_delta"] >= min_delta)
    )

    return out


def summarize_support(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    good = df[df[f"{prefix}_passes_predictive_gate"]].copy()

    selected_support = (
        df.groupby(["panel", "feature_uid"], dropna=False)
        .agg(
            n_selected_rows=("feature_uid", "size"),
            n_selected_contexts=("context_id", "nunique"),
            n_selected_cohorts=("cohort", "nunique"),
            n_selected_endpoints=("endpoint", "nunique"),
            max_candidate_score=("candidate_score", "max"),
            mean_candidate_score=("candidate_score", "mean"),
        )
        .reset_index()
    )

    if good.empty:
        pred_support = selected_support[["panel", "feature_uid"]].copy()
        pred_support["n_predictive_rows"] = 0
        pred_support["n_predictive_contexts"] = 0
        pred_support["n_predictive_cohorts"] = 0
        pred_support["n_predictive_endpoints"] = 0
        pred_support["max_gate_metric"] = np.nan
        pred_support["mean_gate_metric"] = np.nan
        pred_support["max_gate_delta"] = np.nan
        pred_support["mean_gate_delta"] = np.nan
    else:
        pred_support = (
            good.groupby(["panel", "feature_uid"], dropna=False)
            .agg(
                n_predictive_rows=("feature_uid", "size"),
                n_predictive_contexts=("context_id", "nunique"),
                n_predictive_cohorts=("cohort", "nunique"),
                n_predictive_endpoints=("endpoint", "nunique"),
                max_gate_metric=(f"{prefix}_gate_metric", "max"),
                mean_gate_metric=(f"{prefix}_gate_metric", "mean"),
                max_gate_delta=(f"{prefix}_gate_delta", "max"),
                mean_gate_delta=(f"{prefix}_gate_delta", "mean"),
            )
            .reset_index()
        )

    support = selected_support.merge(pred_support, on=["panel", "feature_uid"], how="left")
    for c in ["n_predictive_rows", "n_predictive_contexts", "n_predictive_cohorts", "n_predictive_endpoints"]:
        support[c] = support[c].fillna(0).astype(int)
    return support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--input-config", required=True)
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--new-output-root", required=True)

    ap.add_argument("--ar-min-predictive-contexts", type=int, default=2)
    ap.add_argument("--ar-response-auc", type=float, default=0.60)
    ap.add_argument("--ar-survival-cindex", type=float, default=0.58)
    ap.add_argument("--ar-delta", type=float, default=0.02)

    ap.add_argument("--bt-min-selected-contexts", type=int, default=2)
    ap.add_argument("--bt-min-predictive-contexts", type=int, default=1)
    ap.add_argument("--bt-response-auc", type=float, default=0.58)
    ap.add_argument("--bt-survival-cindex", type=float, default=0.56)
    ap.add_argument("--bt-delta", type=float, default=0.00)

    args = ap.parse_args()

    in_manifest = Path(args.input_manifest)
    out_manifest = Path(args.output_manifest)
    in_config = Path(args.input_config)
    out_config = Path(args.output_config)
    new_output_root = Path(args.new_output_root)

    df0 = pd.read_csv(in_manifest)
    required = {"panel", "feature_uid", "cohort", "endpoint"}
    missing = sorted(required - set(df0.columns))
    if missing:
        raise ValueError(f"Input manifest missing required columns: {missing}")

    df0["context_id"] = (
        df0["cohort"].astype(str) + "|" +
        df0["endpoint"].astype(str) + "|" +
        df0.get("sample_type", "TURBT").astype(str) + "|" +
        df0.get("patient_subset", "all").astype(str) + "|" +
        df0.get("agg", "median").astype(str)
    )

    # Build AR and BT gates separately, then merge support columns.
    df_ar_gate = add_gate_columns(
        df0,
        min_response_auc=args.ar_response_auc,
        min_survival_cindex=args.ar_survival_cindex,
        min_delta=args.ar_delta,
        prefix="ar",
    )
    ar_support = summarize_support(df_ar_gate, prefix="ar")

    df_bt_gate = add_gate_columns(
        df0,
        min_response_auc=args.bt_response_auc,
        min_survival_cindex=args.bt_survival_cindex,
        min_delta=args.bt_delta,
        prefix="bt",
    )
    bt_support = summarize_support(df_bt_gate, prefix="bt")

    ar_keep = ar_support[
        (ar_support["panel"].astype(str).eq("AR"))
        & (ar_support["n_predictive_contexts"] >= args.ar_min_predictive_contexts)
    ][["panel", "feature_uid"]].copy()

    bt_keep = bt_support[
        (bt_support["panel"].astype(str).eq("BT"))
        & (bt_support["n_selected_contexts"] >= args.bt_min_selected_contexts)
        & (bt_support["n_predictive_contexts"] >= args.bt_min_predictive_contexts)
    ][["panel", "feature_uid"]].copy()

    keep = pd.concat([ar_keep, bt_keep], ignore_index=True).drop_duplicates()

    out = df0.merge(keep, on=["panel", "feature_uid"], how="inner")

    # Attach support information with common columns.
    ar_support = ar_support.add_prefix("ar_").rename(columns={"ar_panel": "panel", "ar_feature_uid": "feature_uid"})
    bt_support = bt_support.add_prefix("bt_").rename(columns={"bt_panel": "panel", "bt_feature_uid": "feature_uid"})
    out = out.merge(ar_support, on=["panel", "feature_uid"], how="left")
    out = out.merge(bt_support, on=["panel", "feature_uid"], how="left")

    # Pick one global transform per feature_uid/panel according to best candidate evidence.
    tx_col = "selected_transform_mode" if "selected_transform_mode" in out.columns else ("transform_mode" if "transform_mode" in out.columns else None)
    if tx_col is not None:
        sort_cols = ["panel", "feature_uid", "candidate_score"]
        extra = []
        for c in ["ar_max_gate_metric", "ar_max_gate_delta", "bt_max_gate_metric", "bt_max_gate_delta"]:
            if c in out.columns:
                extra.append(c)
        best_tx = (
            out.sort_values(sort_cols + extra, ascending=[True, True, False] + [False] * len(extra), na_position="last")
            .drop_duplicates(["panel", "feature_uid"], keep="first")
            [["panel", "feature_uid", tx_col]]
            .rename(columns={tx_col: "global_selected_transform_mode"})
        )
        out = out.merge(best_tx, on=["panel", "feature_uid"], how="left")
        if "selected_transform_mode" in out.columns:
            out["selected_transform_mode"] = out["global_selected_transform_mode"]
        if "transform_mode" in out.columns:
            out["transform_mode"] = out["global_selected_transform_mode"]

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_manifest, index=False)

    support_out = out_manifest.with_name(out_manifest.stem + "__support_summary.csv")
    pd.concat([
        ar_support.assign(rule_panel="AR_strict"),
        bt_support.assign(rule_panel="BT_relaxed"),
    ], ignore_index=True, sort=False).to_csv(support_out, index=False)

    cfg = json.loads(in_config.read_text())
    cfg["candidate_manifest"] = str(out_manifest)
    cfg["output_root"] = str(new_output_root)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(cfg, indent=2))

    print("=" * 80)
    print("Input rows:", df0.shape[0])
    print("Input feature_uids by panel:", df0.groupby("panel")["feature_uid"].nunique().to_dict())
    print("Output rows:", out.shape[0])
    print("Output feature_uids by panel:", out.groupby("panel")["feature_uid"].nunique().to_dict())
    print("Output composition:")
    print(
        out.groupby(["panel", "feature_source", "feature_group"], dropna=False)
        .agg(n_rows=("feature_uid", "size"), n_feature_uids=("feature_uid", "nunique"))
        .reset_index()
        .sort_values(["panel", "n_feature_uids"], ascending=[True, False])
        .to_string(index=False)
    )
    print("Saved manifest:", out_manifest)
    print("Saved support summary:", support_out)
    print("Saved config:", out_config)
    print("New output root:", new_output_root)


if __name__ == "__main__":
    main()
