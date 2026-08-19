#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot adjusted epithelial/stromal tissue area by cohort, sample type, and panel.

Report plot:
    row 1 = AR
    row 2 = BT
    columns = NAC2020, PURE01, No-NAC, BLASST

For each panel/cohort subplot:
    x-axis = TURBT vs RC
    y-axis = adjusted area percent, where Epi + Stroma = 100
    hue = Epi vs Stroma

Bad core handling:
    - For TMA cohorts, exclude Unusable cores.
    - By default, keep Acceptable and Borderline cores only.
    - Missing QC is excluded for TMA.
    - For BLASST, manual TMA-style structural QC is not expected, so rows are kept
      if they pass source completeness / have tissue data.

Inputs:
    qc_check_rebuild/source_presence_all.csv
    qc_check_rebuild/tissue_inventory_all.csv

Outputs:
    PDF, PNG, and source-data CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_QC_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/data/qc_check_rebuild")
DEFAULT_OUT_DIR = Path("/projects/ovcare/users/nikolay_alabi/immuno/manuscript/dod_report/figure_outputs")

REPORT_COHORTS = ["NAC2020", "PURE01", "No-NAC", "BLASST"]
PANEL_ORDER = ["AR", "BT"]
SAMPLE_TYPE_ORDER = ["TURBT", "RC"]

TISSUE_COLORS = {
    "Epi": "#ff3333",
    "Stroma": "#228B22",
}


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_panel(x) -> str:
    s = norm_str(x).upper()
    if s in {"AR", "ARP"}:
        return "AR"
    if s in {"BT", "B&T", "B+T", "B_T", "B T"}:
        return "BT"
    if s in {"MY", "M", "MYELOID"}:
        return "MY"
    return norm_str(x)


def normalize_cohort(x) -> str:
    s = norm_str(x)
    su = s.upper().replace("_", " ").replace("-", "")
    if su in {"BCA2020", "BCA 2020", "NAC2020"}:
        return "NAC2020"
    if su in {"NONAC", "NO NAC"}:
        return "No-NAC"
    if su == "PURE01":
        return "PURE01"
    if su == "BLASST":
        return "BLASST"
    if su == "NAC2015" or su.startswith("BLADDER"):
        return "NAC2015"
    if su == "KOLL" or su == "FLORESTAN":
        return "KOLL"
    return s


def normalize_sample_type(x) -> str:
    s = norm_str(x).upper()
    if "TURBT" in s:
        return "TURBT"
    if s == "RC" or "CYSTECTOMY" in s:
        return "RC"
    return norm_str(x) if norm_str(x) else "Unknown"


def normalize_qc(x) -> str:
    s = norm_str(x)
    if s == "" or s.lower() in {"nan", "none", "na", "<na>", "unknown"}:
        return "Missing QC"
    sl = s.lower()
    if sl in {"acceptable", "accepted", "accept", "good", "pass", "usable"}:
        return "Acceptable"
    if sl in {"borderline", "borderline usable", "borderline/usable"}:
        return "Borderline"
    if sl in {"unusable", "fail", "failed", "bad", "reject", "rejected"}:
        return "Unusable"
    return s


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_tissue_area_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Locate epithelial and stromal area columns.

    Works with columns like:
        Epi_region_area_percent
        Str_region_area_percent
        Epi_region_area_sq_microns
        Str_region_area_sq_microns
    """
    epi_candidates = [
        "Epi_region_area_percent",
        "Epi_region_area_sq_microns",
        "Epi_region_area_pixels",
        "Epi_percent",
        "Epi_area",
    ]
    str_candidates = [
        "Str_region_area_percent",
        "Str_region_area_sq_microns",
        "Str_region_area_pixels",
        "Str_percent",
        "Str_area",
    ]

    epi_col = first_existing_col(df, epi_candidates)
    str_col = first_existing_col(df, str_candidates)

    if epi_col is not None and str_col is not None:
        return epi_col, str_col

    # fallback: search by pattern
    cols = list(df.columns)
    epi_like = [
        c for c in cols
        if re.search(r"(^|_)Epi(_|$)", c, flags=re.IGNORECASE)
        and re.search(r"percent|sq_microns|pixels|area", c, flags=re.IGNORECASE)
    ]
    str_like = [
        c for c in cols
        if re.search(r"(^|_)Str(_|$)", c, flags=re.IGNORECASE)
        and re.search(r"percent|sq_microns|pixels|area", c, flags=re.IGNORECASE)
    ]

    epi_col = epi_like[0] if epi_like else None
    str_col = str_like[0] if str_like else None
    return epi_col, str_col


def merge_tissue_if_needed(presence: pd.DataFrame, tissue: pd.DataFrame) -> pd.DataFrame:
    """
    source_presence_all.csv often already contains tissue columns after merge.
    If not, merge tissue_inventory_all.csv by panel/coord.
    """
    epi_col, str_col = find_tissue_area_columns(presence)
    if epi_col is not None and str_col is not None:
        return presence.copy()

    if tissue.empty:
        return presence.copy()

    t = tissue.copy()
    if "panel" in t.columns:
        t["panel"] = t["panel"].map(normalize_panel)
    if "coord" not in t.columns:
        return presence.copy()

    tissue_cols = ["panel", "coord"]
    epi_col_t, str_col_t = find_tissue_area_columns(t)
    if epi_col_t is None or str_col_t is None:
        return presence.copy()

    keep = ["panel", "coord", epi_col_t, str_col_t]
    extra = [
        c for c in [
            "sample_name",
            "cohort_label_inferred",
            "tma_inferred",
            "source_root",
            "core_token",
            "coord_token",
        ]
        if c in t.columns
    ]
    keep = keep + extra

    t = t[keep].drop_duplicates(["panel", "coord"]).copy()

    out = presence.merge(
        t,
        on=["panel", "coord"],
        how="left",
        suffixes=("", "_tissue_inventory"),
    )
    return out


def load_tissue_plot_data(
    qc_dir: Path = DEFAULT_QC_DIR,
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
    keep_qc_statuses: list[str] | None = None,
    require_all_primary_inputs: bool = True,
) -> pd.DataFrame:
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER
    keep_qc_statuses = keep_qc_statuses or ["Acceptable", "Borderline"]

    presence_fp = qc_dir / "source_presence_all.csv"
    tissue_fp = qc_dir / "tissue_inventory_all.csv"

    if not presence_fp.exists():
        raise FileNotFoundError(f"Missing required file: {presence_fp}")

    presence = pd.read_csv(presence_fp, low_memory=False)
    tissue = pd.read_csv(tissue_fp, low_memory=False) if tissue_fp.exists() else pd.DataFrame()

    d = merge_tissue_if_needed(presence, tissue)

    for c in ["panel", "cohort_label", "sample_type", "branch"]:
        if c not in d.columns:
            d[c] = pd.NA

    d["panel"] = d["panel"].map(normalize_panel)
    d["cohort_label"] = d["cohort_label"].map(normalize_cohort)
    d["sample_type"] = d["sample_type"].map(normalize_sample_type)

    d = d[d["panel"].isin(panels)].copy()
    d = d[d["cohort_label"].isin(cohorts)].copy()
    d = d[d["sample_type"].isin(SAMPLE_TYPE_ORDER)].copy()

    if require_all_primary_inputs and "has_all_primary_inputs" in d.columns:
        d["has_all_primary_inputs"] = d["has_all_primary_inputs"].fillna(False).astype(bool)
        d = d[d["has_all_primary_inputs"]].copy()

    if "structural_acceptability" in d.columns:
        d["qc_status"] = d["structural_acceptability"].map(normalize_qc)
    else:
        d["qc_status"] = "Missing QC"

    # Remove bad TMA cores. BLASST does not have TMA-style manual structural review.
    is_tma = d["branch"].astype(str).eq("TMA")
    d = d[
        (~is_tma) |
        (is_tma & d["qc_status"].isin(keep_qc_statuses))
    ].copy()

    epi_col, str_col = find_tissue_area_columns(d)
    if epi_col is None or str_col is None:
        available = "\n".join(d.columns.astype(str).tolist())
        raise ValueError(
            "Could not find epithelial/stromal tissue area columns. "
            "Expected columns like Epi_region_area_percent and Str_region_area_percent.\n\n"
            f"Available columns:\n{available}"
        )

    d["epi_area_raw"] = pd.to_numeric(d[epi_col], errors="coerce")
    d["stroma_area_raw"] = pd.to_numeric(d[str_col], errors="coerce")
    d = d[d["epi_area_raw"].notna() & d["stroma_area_raw"].notna()].copy()

    d["epi_stroma_total"] = d["epi_area_raw"] + d["stroma_area_raw"]
    d = d[d["epi_stroma_total"] > 0].copy()

    d["Epi"] = 100.0 * d["epi_area_raw"] / d["epi_stroma_total"]
    d["Stroma"] = 100.0 * d["stroma_area_raw"] / d["epi_stroma_total"]

    # Deduplicate to one row per panel/cohort/sample_type/entity.
    entity_col = "entity_id" if "entity_id" in d.columns else "coord"
    d = (
        d.sort_values(["panel", "cohort_label", "sample_type", entity_col])
        .drop_duplicates(["panel", "cohort_label", "sample_type", entity_col])
        .copy()
    )

    long = d.melt(
        id_vars=[
            c for c in [
                "entity_id",
                "branch",
                "panel",
                "cohort_label",
                "sample_type",
                "coord",
                "qc_status",
                "has_all_primary_inputs",
                "n_parquet_cells",
            ]
            if c in d.columns
        ],
        value_vars=["Epi", "Stroma"],
        var_name="tissue_class",
        value_name="adjusted_area_percent",
    )

    long["panel"] = pd.Categorical(long["panel"], categories=panels, ordered=True)
    long["cohort_label"] = pd.Categorical(long["cohort_label"], categories=cohorts, ordered=True)
    long["sample_type"] = pd.Categorical(long["sample_type"], categories=SAMPLE_TYPE_ORDER, ordered=True)
    long["tissue_class"] = pd.Categorical(long["tissue_class"], categories=["Epi", "Stroma"], ordered=True)

    return long.sort_values(["panel", "cohort_label", "sample_type", "tissue_class"]).reset_index(drop=True)


def boxplot_with_points(ax, sub: pd.DataFrame, jitter_seed: int = 1):
    rng = np.random.default_rng(jitter_seed)

    positions = {
        ("TURBT", "Epi"): 0.85,
        ("TURBT", "Stroma"): 1.15,
        ("RC", "Epi"): 1.85,
        ("RC", "Stroma"): 2.15,
    }

    for (sample_type, tissue_class), pos in positions.items():
        vals = sub.loc[
            sub["sample_type"].astype(str).eq(sample_type)
            & sub["tissue_class"].astype(str).eq(tissue_class),
            "adjusted_area_percent",
        ].dropna().to_numpy()

        if len(vals) == 0:
            continue

        color = TISSUE_COLORS[tissue_class]

        bp = ax.boxplot(
            vals,
            positions=[pos],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#E69F00", "linewidth": 1.2},
            boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 1.0, "alpha": 0.78},
            whiskerprops={"color": "black", "linewidth": 0.9},
            capprops={"color": "black", "linewidth": 0.9},
        )

        jitter = rng.normal(loc=0, scale=0.035, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=10,
            color=color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )


def make_tissue_composition_grid(
    data: pd.DataFrame,
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
    title: str = "Adjusted epithelial and stromal tissue area by cohort, sample type, and panel",
):
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=len(cohorts),
        figsize=(17, 7.2),
        sharey=True,
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = np.array([axes])
    if len(cohorts) == 1:
        axes = axes.reshape(len(panels), 1)

    for r, panel in enumerate(panels):
        for c, cohort in enumerate(cohorts):
            ax = axes[r, c]
            sub = data[
                data["panel"].astype(str).eq(panel)
                & data["cohort_label"].astype(str).eq(cohort)
            ].copy()

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cohort}\nTURBT: n=0 | RC: n=0", fontsize=10)
            else:
                boxplot_with_points(ax, sub, jitter_seed=100 + r * 10 + c)

                n_turbt = sub.loc[sub["sample_type"].astype(str).eq("TURBT"), "coord"].nunique()
                n_rc = sub.loc[sub["sample_type"].astype(str).eq("RC"), "coord"].nunique()
                ax.set_title(f"{cohort}\nTURBT: n={n_turbt} | RC: n={n_rc}", fontsize=10)

            ax.set_xticks([1.0, 2.0])
            ax.set_xticklabels(["TURBT", "RC"])
            ax.set_xlim(0.45, 2.55)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if c == 0:
                ax.set_ylabel("Adjusted area % (Epi + Stroma = 100)")
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

    handles = [
        Patch(facecolor=TISSUE_COLORS["Epi"], edgecolor="black", label="Epi"),
        Patch(facecolor=TISSUE_COLORS["Stroma"], edgecolor="black", label="Stroma"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        title="Tissue class",
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
    prefix: str = "fig1_tissue_composition_AR_BT_report",
    cohorts: list[str] | None = None,
    panels: list[str] | None = None,
    keep_qc_statuses: list[str] | None = None,
    require_all_primary_inputs: bool = True,
    save: bool = True,
):
    cohorts = cohorts or REPORT_COHORTS
    panels = panels or PANEL_ORDER

    data = load_tissue_plot_data(
        qc_dir=qc_dir,
        cohorts=cohorts,
        panels=panels,
        keep_qc_statuses=keep_qc_statuses,
        require_all_primary_inputs=require_all_primary_inputs,
    )

    fig, axes = make_tissue_composition_grid(
        data,
        cohorts=cohorts,
        panels=panels,
        title="Epithelial/Stromal Area",
    )

    paths = {}
    if save:
        paths = save_outputs(fig, data, out_dir, prefix)

    return fig, axes, data, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-dir", default=str(DEFAULT_QC_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--prefix", default="fig1_tissue_composition_AR_BT_report")
    ap.add_argument("--cohorts", nargs="+", default=REPORT_COHORTS)
    ap.add_argument("--panels", nargs="+", default=PANEL_ORDER)
    ap.add_argument(
        "--keep-qc-statuses",
        nargs="+",
        default=["Acceptable", "Borderline"],
        help="For TMA cohorts, retain only these structural acceptability labels.",
    )
    ap.add_argument(
        "--no-require-all-primary-inputs",
        action="store_true",
        help="Do not filter to has_all_primary_inputs==True.",
    )
    args = ap.parse_args()

    fig, axes, data, paths = build_plot(
        qc_dir=Path(args.qc_dir),
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
        cohorts=args.cohorts,
        panels=args.panels,
        keep_qc_statuses=args.keep_qc_statuses,
        require_all_primary_inputs=not args.no_require_all_primary_inputs,
        save=True,
    )

    print("\nSaved outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\nCore/region counts used:")
    counts = (
        data
        .drop_duplicates(["panel", "cohort_label", "sample_type", "coord"])
        .groupby(["panel", "cohort_label", "sample_type"], observed=False)
        .size()
        .reset_index(name="n_units")
    )
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()