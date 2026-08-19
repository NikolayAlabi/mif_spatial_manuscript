#!/usr/bin/env python3
"""
stage3b_aggregate_turbt_module_results_v1.py

Aggregate Stage3A clinical-variable, root-module, and meta-module univariate evaluation across
cohorts, panels, and response/survival contexts.

Color convention for effect plots
---------------------------------
Red = favorable biology in BOTH endpoint families:
  response: positive coefficient (greater odds of response)
  survival: negative Cox coefficient (lower hazard / longer survival)
Blue = unfavorable.

Primary outputs
---------------
all_stage3a_program_metrics.csv
all_stage3a_program_metrics_with_qvalues.csv
all_stage3a_fold_metrics.csv.gz
all_stage3a_context_summary.csv
program_cross_context_summary.csv
program_response_summary.csv
program_survival_summary.csv
nac2015_program_evaluation_summary.csv
discovery_vs_nac2015_concordance.csv
stage3b_context_inventory.csv
plots/

Multiplicity
------------
Two BH q-values are reported:
  q_within_program_level:
      cohort x panel x endpoint x {root_module/meta_module}
  q_all_programs_in_context:
      cohort x panel x endpoint across BOTH root and meta programs

Primary recurrence summaries use contexts satisfying the frozen Stage3A
minimum-N / minimum-event eligibility rules. Exploratory low-signal contexts
remain in the long tables and plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

RESPONSE_ENDPOINTS = {"complete_response", "any_response"}
SURVIVAL_ENDPOINTS = {"OS", "RFS"}
ROOT_COL = "feature_source"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def bh_adjust(p: pd.Series) -> pd.Series:
    x = safe_numeric(p)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    ok = x.notna()
    if not ok.any():
        return out
    vals = x[ok].clip(lower=0, upper=1).to_numpy(float)
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    idx = x[ok].index.to_numpy()
    out.loc[idx[order]] = q
    return out


def module_sort_key(s: object):
    x = str(s)
    m = re.search(r"(?:META|M)(\d+)", x)
    return int(m.group(1)) if m else 9999


def favorable_effect(df: pd.DataFrame) -> pd.Series:
    coef = safe_numeric(df["coef"])
    return np.where(df["endpoint"].isin(SURVIVAL_ENDPOINTS), -coef, coef)


def save_fig(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_effect_dotplot(
    d: pd.DataFrame,
    panel: str,
    level: str,
    endpoints: Sequence[str],
    path: Path,
    title: str,
    feature_source: Optional[str] = None,
) -> None:
    x = d[
        d["panel"].astype(str).eq(panel)
        & d["program_level"].astype(str).eq(level)
        & d["endpoint"].isin(endpoints)
    ].copy()
    if feature_source is not None:
        x = x[x[ROOT_COL].astype(str).eq(str(feature_source))].copy()
    if x.empty:
        return
    x["fav_effect"] = favorable_effect(x)
    x["context"] = (
        x["endpoint"].astype(str)
        + " | " + x["cohort"].astype(str)
        + np.where(
            x["patient_subset"].astype(str).ne("all"),
            " | " + x["patient_subset"].astype(str),
            "",
        )
    )
    cohort_order = [c for c in ["NAC2020", "PURE01", "BLASST", "No-NAC", "NAC2015"] if c in x["cohort"].unique()]
    context_order = []
    for e in endpoints:
        for c in cohort_order:
            z = x[(x["endpoint"] == e) & (x["cohort"] == c)]
            for subset in ["all", "no_adj_chemo", "adj_chemo"]:
                zz = z[z["patient_subset"].astype(str).eq(subset)]
                if zz.empty:
                    continue
                label = f"{e} | {c}" if subset == "all" else f"{e} | {c} | {subset}"
                context_order.append(label)
    programs = sorted(x["program_id"].astype(str).unique(), key=module_sort_key)
    xpos = {c: i for i, c in enumerate(context_order)}
    ypos = {m: i for i, m in enumerate(programs)}
    x["xx"] = x["context"].map(xpos)
    x["yy"] = x["program_id"].astype(str).map(ypos)
    p = safe_numeric(x["p_value"]).clip(lower=1e-300)
    nl = -np.log10(p)
    denom = np.nanquantile(nl, 0.95) if nl.notna().any() else 1.0
    denom = denom if np.isfinite(denom) and denom > 0 else 1.0
    sizes = (35 + 180 * nl / denom).clip(35, 260)
    eff = safe_numeric(x["fav_effect"])
    lim = np.nanquantile(np.abs(eff), 0.95) if eff.notna().any() else 1.0
    lim = max(float(lim), 0.1)

    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(context_order)), max(5, 0.25 * len(programs))))
    sc = ax.scatter(
        x["xx"], x["yy"], c=eff, s=sizes,
        cmap="coolwarm", vmin=-lim, vmax=lim,
        edgecolor="black", linewidth=0.35, alpha=0.9,
    )
    ax.set_xticks(range(len(context_order)))
    ax.set_xticklabels(context_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(programs, fontsize=6)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Favorable signed coefficient\n(+ response / - survival)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_oof_heatmap(d: pd.DataFrame, panel: str, level: str, path: Path) -> None:
    x = d[
        d["panel"].astype(str).eq(panel)
        & d["program_level"].astype(str).eq(level)
    ].copy()
    if x.empty:
        return
    x["context"] = (
        x["endpoint"].astype(str)
        + " | " + x["cohort"].astype(str)
        + np.where(
            x["patient_subset"].astype(str).ne("all"),
            " | " + x["patient_subset"].astype(str),
            "",
        )
    )
    heat = x.pivot_table(index="program_id", columns="context", values="oof_metric", aggfunc="first")
    heat = heat.reindex(sorted(heat.index, key=module_sort_key))
    if heat.empty:
        return
    M = heat.to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * heat.shape[1]), max(5, 0.23 * heat.shape[0])))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index, fontsize=6)
    ax.set_title(f"{panel} {level}: repeated-OOF discrimination")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="OOF AUC / C-index")
    fig.tight_layout()
    save_fig(fig, path)


def plot_discovery_vs_nac2015(conc: pd.DataFrame, panel: str, level: str, path: Path) -> None:
    d = conc[
        conc["panel"].astype(str).eq(panel)
        & conc["program_level"].astype(str).eq(level)
    ].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for endpoint, g in d.groupby("endpoint"):
        ax.scatter(g["discovery_median_favorable_coef"], g["nac2015_favorable_coef"], label=endpoint, alpha=0.8)
    lim = np.nanmax(np.abs(d[["discovery_median_favorable_coef", "nac2015_favorable_coef"]].to_numpy(float)))
    lim = max(float(lim), 0.1)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Median favorable coefficient across discovery cohorts")
    ax.set_ylabel("NAC2015 favorable coefficient")
    ax.set_title(f"{panel} {level}: discovery vs NAC2015 effect direction")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    s3a = Path(cfg["stage3a_output_root"])
    out = ensure_dir(Path(cfg["stage3b_output_root"]))
    plot_root = ensure_dir(out / "plots")

    idx_path = s3a / "stage3a_context_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    idx = pd.read_csv(idx_path)

    metric_parts, fold_parts, context_parts, inventory_rows, missing = [], [], [], [], []
    for _, r in idx.iterrows():
        patient_subset = str(r.get("patient_subset", "all"))
        cdir = (
            s3a / "contexts" / str(r["cohort"]) / str(r["panel"])
            / str(r["endpoint"]) / patient_subset
        )
        mp = cdir / "program_univariate_metrics.csv"
        fp = cdir / "program_fold_metrics.csv"
        cp = cdir / "context_summary.csv"
        done = cdir / ".done"
        inventory_rows.append({
            "cohort": r["cohort"], "panel": r["panel"], "endpoint": r["endpoint"],
            "patient_subset": patient_subset,
            "context_dir": str(cdir), "complete": done.exists(),
            "metrics_exists": mp.exists(), "folds_exists": fp.exists(), "summary_exists": cp.exists(),
        })
        if not done.exists() or not mp.exists():
            missing.append({"cohort": r["cohort"], "panel": r["panel"], "endpoint": r["endpoint"]})
            continue
        metric_parts.append(pd.read_csv(mp))
        if fp.exists():
            f = pd.read_csv(fp)
            if not f.empty:
                for c in ["cohort", "panel", "endpoint", "patient_subset"]:
                    f[c] = r[c] if c in r.index else "all"
                fold_parts.append(f)
        if cp.exists():
            context_parts.append(pd.read_csv(cp))

    pd.DataFrame(inventory_rows).to_csv(out / "stage3b_context_inventory.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "stage3b_missing_contexts.csv", index=False)
    if missing:
        raise RuntimeError(f"{len(missing)} Stage3A contexts are incomplete")
    if not metric_parts:
        raise RuntimeError("No Stage3A metrics found")

    d = pd.concat(metric_parts, ignore_index=True, sort=False)
    folds = pd.concat(fold_parts, ignore_index=True, sort=False) if fold_parts else pd.DataFrame()
    contexts = pd.concat(context_parts, ignore_index=True, sort=False) if context_parts else pd.DataFrame()

    # Standardize numeric columns.
    for c in ["coef", "p_value", "effect", "full_metric", "oof_metric", "fold_sd", "direction_consistency"]:
        if c in d.columns:
            d[c] = safe_numeric(d[c])

    # Multiple-testing correction.
    d["q_within_program_level"] = np.nan
    for _, g in d.groupby(["cohort", "panel", "endpoint", "patient_subset", "program_level"], dropna=False):
        d.loc[g.index, "q_within_program_level"] = bh_adjust(g["p_value"])
    d["q_all_programs_in_context"] = np.nan
    for _, g in d.groupby(["cohort", "panel", "endpoint", "patient_subset"], dropna=False):
        d.loc[g.index, "q_all_programs_in_context"] = bh_adjust(g["p_value"])

    d["favorable_coef"] = favorable_effect(d)
    d["effect_direction"] = np.where(d["favorable_coef"] > 0, "favorable", np.where(d["favorable_coef"] < 0, "unfavorable", "neutral"))
    d["nominal_p_lt_005"] = d["p_value"] < 0.05
    d["nominal_p_lt_010"] = d["p_value"] < 0.10
    d["q_level_lt_010"] = d["q_within_program_level"] < 0.10
    d["q_level_lt_020"] = d["q_within_program_level"] < 0.20

    d.to_csv(out / "all_stage3a_program_metrics.csv", index=False)
    d.to_csv(out / "all_stage3a_program_metrics_with_qvalues.csv", index=False)
    folds.to_csv(out / "all_stage3a_fold_metrics.csv.gz", index=False, compression="gzip")
    contexts.to_csv(out / "all_stage3a_context_summary.csv", index=False)

    primary = d[d["context_primary_eligible"].astype(str).str.lower().isin(["true", "1"])].copy()

    def summarize(g):
        return pd.Series({
            "n_contexts": int(len(g)),
            "n_cohorts": int(g["cohort"].nunique()),
            "n_endpoints": int(g["endpoint"].nunique()),
            "favorable_contexts": int((g["favorable_coef"] > 0).sum()),
            "unfavorable_contexts": int((g["favorable_coef"] < 0).sum()),
            "nominal_favorable_p005": int(((g["favorable_coef"] > 0) & (g["p_value"] < 0.05)).sum()),
            "nominal_unfavorable_p005": int(((g["favorable_coef"] < 0) & (g["p_value"] < 0.05)).sum()),
            "q010_favorable_contexts": int(((g["favorable_coef"] > 0) & (g["q_within_program_level"] < 0.10)).sum()),
            "median_favorable_coef": float(g["favorable_coef"].median()),
            "mean_oof_metric": float(g["oof_metric"].mean()),
            "median_oof_metric": float(g["oof_metric"].median()),
            "max_oof_metric": float(g["oof_metric"].max()),
            "median_fold_sd": float(g["fold_sd"].median()),
            "median_direction_consistency": float(g["direction_consistency"].median()),
        })

    cross = (
        primary.groupby(["patient_subset", "panel", "program_level", ROOT_COL, "program_id"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["panel", "program_level", "q010_favorable_contexts", "median_oof_metric"], ascending=[True, True, False, False])
    )
    cross.to_csv(out / "program_cross_context_summary.csv", index=False)

    resp = (
        primary[primary["endpoint"].isin(RESPONSE_ENDPOINTS)]
        .groupby(["patient_subset", "panel", "program_level", ROOT_COL, "program_id"], dropna=False)
        .apply(summarize).reset_index()
    )
    resp.to_csv(out / "program_response_summary.csv", index=False)

    surv = (
        primary[primary["endpoint"].isin(SURVIVAL_ENDPOINTS)]
        .groupby(["patient_subset", "panel", "program_level", ROOT_COL, "program_id"], dropna=False)
        .apply(summarize).reset_index()
    )
    surv.to_csv(out / "program_survival_summary.csv", index=False)

    nac = d[d["cohort"].astype(str).eq("NAC2015")].copy()
    nac.to_csv(out / "nac2015_program_evaluation_summary.csv", index=False)

    # Discovery -> NAC2015 directional concordance, matched by panel/level/program/endpoint.
    disc = d[
        ~d["cohort"].astype(str).eq("NAC2015")
        & d["patient_subset"].astype(str).eq("all")
    ].copy()
    disc_med = (
        disc.groupby(["panel", "program_level", ROOT_COL, "program_id", "endpoint"], as_index=False)
        .agg(
            discovery_median_favorable_coef=("favorable_coef", "median"),
            discovery_mean_oof_metric=("oof_metric", "mean"),
            discovery_n_cohorts=("cohort", "nunique"),
        )
    )
    nac_small = nac[["panel", "program_level", ROOT_COL, "program_id", "endpoint", "favorable_coef", "oof_metric", "p_value"]].rename(columns={
        "favorable_coef": "nac2015_favorable_coef",
        "oof_metric": "nac2015_oof_metric",
        "p_value": "nac2015_p_value",
    })
    conc = disc_med.merge(nac_small, on=["panel", "program_level", ROOT_COL, "program_id", "endpoint"], how="inner")
    conc["same_effect_direction"] = np.sign(conc["discovery_median_favorable_coef"]) == np.sign(conc["nac2015_favorable_coef"])
    conc.to_csv(out / "discovery_vs_nac2015_concordance.csv", index=False)

    # Review plots.
    # Root modules are NEVER pooled into one giant effect dotplot.
    # Each prep root gets its own response and survival landscape.
    for panel in ["AR", "BT"]:
        root_rows = d[
            d["panel"].astype(str).eq(panel)
            & d["program_level"].astype(str).eq("root_module")
        ].copy()

        for root_name in sorted(root_rows[ROOT_COL].dropna().astype(str).unique()):
            pdir = ensure_dir(plot_root / panel / "roots" / str(root_name))
            plot_effect_dotplot(
                d, panel, "root_module", ["any_response", "complete_response"],
                pdir / "01_response_effect_dotplot.png",
                f"{panel} / {root_name}: response associations",
                feature_source=root_name,
            )
            plot_effect_dotplot(
                d, panel, "root_module", ["OS", "RFS"],
                pdir / "02_survival_effect_dotplot.png",
                f"{panel} / {root_name}: survival associations",
                feature_source=root_name,
            )

        # Cross-root meta-modules are shown together because this is already
        # the intentionally integrated level of the hierarchy.
        mdir = ensure_dir(plot_root / panel / "meta_modules")
        plot_effect_dotplot(
            d, panel, "meta_module", ["any_response", "complete_response"],
            mdir / "01_response_effect_dotplot.png",
            f"{panel}: meta-module response associations",
        )
        plot_effect_dotplot(
            d, panel, "meta_module", ["OS", "RFS"],
            mdir / "02_survival_effect_dotplot.png",
            f"{panel}: meta-module survival associations",
        )
        plot_oof_heatmap(d, panel, "meta_module", mdir / "03_oof_performance_heatmap.png")
        plot_discovery_vs_nac2015(conc, panel, "meta_module", mdir / "04_discovery_vs_nac2015_effects.png")

        # Clinical variables are a separate benchmark plot.
        cdir = ensure_dir(plot_root / panel / "clinical_variables")
        plot_effect_dotplot(
            d, panel, "clinical_variable", ["any_response", "complete_response"],
            cdir / "01_response_effect_dotplot.png",
            f"{panel}: clinical-variable response associations",
        )
        plot_effect_dotplot(
            d, panel, "clinical_variable", ["OS", "RFS"],
            cdir / "02_survival_effect_dotplot.png",
            f"{panel}: clinical-variable survival associations",
        )

    summary_lines = [
        "STAGE 3B TURBT MODULE EVALUATION SUMMARY",
        "=" * 64,
        f"Stage3A contexts aggregated: {idx.shape[0]}",
        f"Program-result rows: {len(d)}",
        f"Primary-eligible result rows: {len(primary)}",
        f"NAC2015 result rows: {len(nac)}",
        "",
        "Primary analyses use only contexts meeting frozen minimum N / event or class-count criteria.",
        "Low-signal contexts remain available in the all-results tables as exploratory analyses.",
        "NAC2015 uses frozen module definitions, but the univariate coefficient is refit within NAC2015.",
    ]
    (out / "stage3b_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"[DONE] Stage3B -> {out}")


if __name__ == "__main__":
    main()
