#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot total segmented cells per TMA core by structural acceptability.

Report-specific plot:
    row 1 = AR panel, NAC2020 / No-NAC / PURE01
    row 2 = BT panel, NAC2020 / No-NAC / PURE01

Input:
    qc_check_rebuild/source_presence_all.csv

Output:
    PDF, PNG, and source-data CSV.

Notes:
    - This plot is intended for TMA cohorts with manual structural review.
    - BLASST is excluded because it is a whole-section cohort and does not have
      the same TMA structural_acceptability review field.
    - NAC2015 is excluded for the DOD report version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_QC_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild")
DEFAULT_OUT_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/figure_outputs")

REPORT_COHORTS = ["NAC2020", "No-NAC", "PURE01"]
PANEL_ORDER = ["AR", "BT"]

QC_ORDER = ["Unusable", "Borderline", "Acceptable", "Missing QC"]
QC_COLORS = {
    "Unusable": "red",
    "Borderline": "orange",
    "Acceptable": "forestgreen",
    "Missing QC": "lightgray",
}


def normalize_status(x) -> str:
    """Normalize structural acceptability labels to the four plot groups."""
    if pd.isna(x):
        return "Missing QC"

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "<na>", "unknown"}:
        return "Missing QC"

    sl = s.lower()

    if sl in {"acceptable", "accept", "accepted", "good", "pass", "usable"}:
        return "Acceptable"

    if sl in {"borderline", "borderline/usable", "borderline usable"}:
        return "Borderline"

    if sl in {"unusable", "fail", "failed", "bad", "reject", "rejected"}:
        return "Unusable"

    # Preserve unexpected labels but keep them visible.
    return s


def load_plot_data(
    qc_dir: Path = DEFAULT_QC_DIR,
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
) -> pd.DataFrame:
    """Load and clean source_presence_all.csv for this plot."""
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER

    fp = qc_dir / "source_presence_all.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Could not find: {fp}")

    df = pd.read_csv(fp, low_memory=False)

    required = ["branch", "panel", "cohort_label", "coord", "n_parquet_cells"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} is missing required columns: {missing}")

    # TMA only; exclude BLASST whole sections and NAC2015.
    d = df.copy()
    d = d[d["branch"].astype(str).eq("TMA")].copy()
    d = d[d["panel"].isin(panels)].copy()
    d = d[d["cohort_label"].isin(cohorts)].copy()

    d["n_parquet_cells"] = pd.to_numeric(d["n_parquet_cells"], errors="coerce")
    d = d[d["n_parquet_cells"].notna()].copy()
    d = d[d["n_parquet_cells"] > 0].copy()

    if "structural_acceptability" in d.columns:
        d["qc_status"] = d["structural_acceptability"].map(normalize_status)
    else:
        d["structural_acceptability"] = pd.NA
        d["qc_status"] = "Missing QC"

    # One row per panel/cohort/core coordinate. If duplicates exist, retain the
    # first after sorting but also keep the maximum cell count to be conservative.
    d = (
        d.sort_values(["panel", "cohort_label", "coord"])
        .groupby(["panel", "cohort_label", "coord"], as_index=False)
        .agg(
            n_parquet_cells=("n_parquet_cells", "max"),
            qc_status=("qc_status", "first"),
            structural_acceptability=("structural_acceptability", "first"),
            sample_type=("sample_type", "first") if "sample_type" in d.columns else ("coord", "first"),
            has_all_primary_inputs=("has_all_primary_inputs", "first") if "has_all_primary_inputs" in d.columns else ("coord", "first"),
        )
    )

    d["panel"] = pd.Categorical(d["panel"], categories=panels, ordered=True)
    d["cohort_label"] = pd.Categorical(d["cohort_label"], categories=cohorts, ordered=True)

    return d.sort_values(["panel", "cohort_label", "coord"]).reset_index(drop=True)


def make_hist_grid(
    data: pd.DataFrame,
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
    bins: int = 24,
    same_x_by_panel: bool = True,
    title: str = "Total segmented cells per core by structural acceptability",
):
    """Make 2 x 3 grid: AR row, BT row, cohorts as columns."""
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=len(cohorts),
        figsize=(14, 7.5),
        sharey=False,
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = np.array([axes])
    if len(cohorts) == 1:
        axes = axes.reshape(len(panels), 1)

    # Panel-level x ranges keep AR and BT rows internally comparable.
    panel_bins = {}
    for panel in panels:
        vals = data.loc[data["panel"].astype(str).eq(panel), "n_parquet_cells"].dropna()
        if vals.empty:
            panel_bins[panel] = np.linspace(0, 1, bins + 1)
            continue

        max_val = vals.quantile(0.995)
        max_val = max(max_val, vals.max())
        max_val = np.ceil(max_val / 1000) * 1000
        panel_bins[panel] = np.linspace(0, max_val, bins + 1)

    for r, panel in enumerate(panels):
        for c, cohort in enumerate(cohorts):
            ax = axes[r, c]
            sub = data[
                data["panel"].astype(str).eq(panel)
                & data["cohort_label"].astype(str).eq(cohort)
            ].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cohort}\n(n cores=0)")
                ax.set_xlabel("Total cells per core")
                if c == 0:
                    ax.set_ylabel("Number of cores")
                continue

            hist_data = []
            labels = []
            colors = []
            for status in QC_ORDER:
                vals = sub.loc[sub["qc_status"].eq(status), "n_parquet_cells"].dropna()
                hist_data.append(vals.values)
                labels.append(status)
                colors.append(QC_COLORS[status])

            # Include any unexpected statuses after the standard ones.
            extra_statuses = [
                s for s in sorted(sub["qc_status"].dropna().unique())
                if s not in QC_ORDER
            ]
            for status in extra_statuses:
                vals = sub.loc[sub["qc_status"].eq(status), "n_parquet_cells"].dropna()
                hist_data.append(vals.values)
                labels.append(status)
                colors.append("white")

            use_bins = panel_bins[panel] if same_x_by_panel else bins

            ax.hist(
                hist_data,
                bins=use_bins,
                stacked=True,
                label=labels,
                color=colors,
                edgecolor="black",
                linewidth=0.4,
            )

            median = float(np.median(sub["n_parquet_cells"]))
            ax.axvline(median, linestyle="--", linewidth=1.5, color="blue")

            n_cores = sub["coord"].nunique()
            ax.set_title(f"{cohort}\n(n cores={n_cores}, median={median:.0f})", fontsize=10)
            ax.set_xlabel("Total cells per core")
            if c == 0:
                ax.set_ylabel("Number of cores")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add panel label at left side of row.
            if c == 0:
                ax.text(
                    -0.28,
                    0.5,
                    panel,
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                )

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        title="Structural acceptability",
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)

    return fig, axes


def save_outputs(fig, data: pd.DataFrame, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = out_dir / f"{prefix}.pdf"
    png = out_dir / f"{prefix}.png"
    csv = out_dir / f"{prefix}.source_data.csv"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    data.to_csv(csv, index=False)

    return {"pdf": pdf, "png": png, "source_data": csv}


def build_plot(
    qc_dir: Path = DEFAULT_QC_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    prefix: str = "fig1_total_cells_per_core_by_qc_report_AR_BT",
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
    save: bool = True,
):
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER

    data = load_plot_data(qc_dir=qc_dir, cohorts=cohorts, panels=panels)

    fig, axes = make_hist_grid(
        data,
        cohorts=cohorts,
        panels=panels,
        title="AR and BT total cells per core by structural acceptability",
    )

    paths = {}
    if save:
        paths = save_outputs(fig, data, out_dir, prefix)

    return fig, axes, data, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-dir", default=str(DEFAULT_QC_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--prefix", default="fig1_total_cells_per_core_by_qc_report_AR_BT")
    ap.add_argument("--cohorts", nargs="+", default=REPORT_COHORTS)
    ap.add_argument("--panels", nargs="+", default=PANEL_ORDER)
    args = ap.parse_args()

    fig, axes, data, paths = build_plot(
        qc_dir=Path(args.qc_dir),
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
        cohorts=args.cohorts,
        panels=args.panels,
        save=True,
    )

    print("\nSaved outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\nData summary:")
    print(
        data.groupby(["panel", "cohort_label", "qc_status"], observed=False)
        .size()
        .reset_index(name="n_cores")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()