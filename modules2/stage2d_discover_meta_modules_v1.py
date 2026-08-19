#!/usr/bin/env python3
"""
stage2d_discover_meta_modules_v1.py

Outcome-blind Stage 2D meta-module discovery from frozen Stage 2C root-module scores.

Primary concept
---------------
Root modules are already biologically interpretable programs. Meta-modules are
therefore defined only when root-module scores from >=2 distinct prep roots are
reproducibly positively coordinated across discovery cohorts.

The script:
  * computes cohort-specific root-module score Spearman matrices;
  * equal-weights cohorts to a signed consensus matrix;
  * requires pair support and sign consistency for clustering;
  * explores a transparent consensus-rho threshold grid;
  * uses direct-signed hierarchical clustering (average linkage primary);
  * accepts a cluster as a meta-module only if it contains >=2 root modules
    from >=2 prep roots;
  * leaves same-root-only clusters and singletons as standalone root programs;
  * scores the primary meta-module solution per patient using mean z-scored
    root-module scores.

No outcome columns are read anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT_COL = "feature_source"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_slug(x: object) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(x)).strip("-")
    return s or "NA"


def read_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Mapping, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def output_root(cfg: Mapping) -> Path:
    return Path(cfg["output_root"])


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 2:
        return pd.Series(np.nan, index=x.index, dtype=float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - x.mean()) / sd


def pairwise_corr_and_n(X: pd.DataFrame, min_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(X.columns)
    A = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = A.corr(method="spearman", min_periods=int(min_n)).reindex(index=cols, columns=cols)
    mask = A.notna().astype(np.int16)
    nmat = mask.T.dot(mask).reindex(index=cols, columns=cols).astype(float)
    return corr, nmat


def build_panel_consensus(scores: pd.DataFrame, panel: str, min_pairwise_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = scores[scores["panel"].astype(str).eq(panel)].copy()
    modules = sorted(d["root_module_id"].astype(str).unique())
    cohort_corr: Dict[str, pd.DataFrame] = {}
    cohort_n: Dict[str, pd.DataFrame] = {}

    for cohort, g in d.groupby("cohort", sort=True):
        X = g.pivot_table(index="patient_id", columns="root_module_id", values="score_meanz", aggfunc="first")
        X = X.reindex(columns=modules)
        corr, nmat = pairwise_corr_and_n(X, min_pairwise_n)
        cohort_corr[str(cohort)] = corr
        cohort_n[str(cohort)] = nmat

    C = pd.DataFrame(np.eye(len(modules)), index=modules, columns=modules, dtype=float)
    S = pd.DataFrame(0, index=modules, columns=modules, dtype=int)
    SC = pd.DataFrame(np.nan, index=modules, columns=modules, dtype=float)
    SD = pd.DataFrame(np.nan, index=modules, columns=modules, dtype=float)
    audit_rows = []

    for i, a in enumerate(modules):
        S.loc[a, a] = len(cohort_corr)
        SC.loc[a, a] = 1.0
        SD.loc[a, a] = 0.0
        for j in range(i + 1, len(modules)):
            b = modules[j]
            vals = []
            detail = {}
            for cohort in cohort_corr:
                rho = cohort_corr[cohort].loc[a, b]
                nn = cohort_n[cohort].loc[a, b]
                detail[f"rho__{cohort}"] = rho
                detail[f"n__{cohort}"] = nn
                if np.isfinite(rho) and np.isfinite(nn) and nn >= int(min_pairwise_n):
                    vals.append(float(rho))
            support = len(vals)
            consensus = float(np.mean(vals)) if vals else np.nan
            sd = float(np.std(vals, ddof=0)) if vals else np.nan
            n_pos = sum(v > 0 for v in vals)
            n_neg = sum(v < 0 for v in vals)
            n_zero = sum(v == 0 for v in vals)
            if support:
                sign_consistency = float(max(n_pos, n_neg, n_zero) / support)
            else:
                sign_consistency = np.nan
            C.loc[a, b] = C.loc[b, a] = consensus
            S.loc[a, b] = S.loc[b, a] = support
            SC.loc[a, b] = SC.loc[b, a] = sign_consistency
            SD.loc[a, b] = SD.loc[b, a] = sd
            audit_rows.append({
                "panel": panel,
                "module_a": a,
                "module_b": b,
                "consensus_rho": consensus,
                "pair_support": support,
                "sign_consistency": sign_consistency,
                "cross_cohort_rho_sd": sd,
                "n_positive_cohorts": n_pos,
                "n_negative_cohorts": n_neg,
                **detail,
            })

    return C, S, SC, SD, pd.DataFrame(audit_rows)


def clustering_consensus(C: pd.DataFrame, support: pd.DataFrame, sign_cons: pd.DataFrame, min_support: int, min_sign_consistency: float) -> pd.DataFrame:
    Z = C.copy()
    ok = (support >= int(min_support)) & (sign_cons >= float(min_sign_consistency))
    Z = Z.where(ok, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Z = (Z + Z.T) / 2.0
    np.fill_diagonal(Z.values, 1.0)
    return Z


def direct_signed_linkage(C: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, np.ndarray]:
    D = 1.0 - C
    D = D.clip(lower=0.0, upper=2.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D.values, 0.0)
    Z = linkage(squareform(D.to_numpy(float), checks=False), method=method)
    return D, Z


def ordered_cluster_map(labels: np.ndarray, Z: np.ndarray) -> Dict[int, int]:
    seen = []
    for i in leaves_list(Z):
        lab = int(labels[i])
        if lab not in seen:
            seen.append(lab)
    return {lab: ii + 1 for ii, lab in enumerate(seen)}


def cluster_solution(
    C_raw: pd.DataFrame,
    C_cluster: pd.DataFrame,
    Z: np.ndarray,
    module_meta: pd.DataFrame,
    rho_threshold: float,
    panel: str,
    linkage_method: str,
    min_roots: int,
    min_modules: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = fcluster(Z, t=float(1.0 - rho_threshold), criterion="distance").astype(int)
    remap = ordered_cluster_map(labels, Z)
    ordered_labels = np.asarray([remap[int(x)] for x in labels], dtype=int)
    mem = pd.DataFrame({
        "panel": panel,
        "root_module_id": C_cluster.index.astype(str),
        "raw_cluster_id": labels,
        "ordered_cluster_id": ordered_labels,
    })
    mm = module_meta[["root_module_id", ROOT_COL, "module_label"]].drop_duplicates("root_module_id")
    mem = mem.merge(mm, on="root_module_id", how="left")

    cluster_rows = []
    accepted_cluster_ids = []
    for cid, g in mem.groupby("ordered_cluster_id", sort=True):
        ids = g["root_module_id"].astype(str).tolist()
        roots = sorted(g[ROOT_COL].dropna().astype(str).unique().tolist())
        vals = []
        pos = 0
        total = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                v = C_raw.loc[ids[i], ids[j]]
                if np.isfinite(v):
                    vals.append(float(v))
                    total += 1
                    pos += int(v > 0)
        accepted = len(ids) >= int(min_modules) and len(roots) >= int(min_roots)
        if accepted:
            accepted_cluster_ids.append(int(cid))
        cluster_rows.append({
            "panel": panel,
            "linkage_method": linkage_method,
            "rho_threshold": float(rho_threshold),
            "ordered_cluster_id": int(cid),
            "n_root_modules": int(len(ids)),
            "n_prep_roots": int(len(roots)),
            "prep_roots": ";".join(roots),
            "is_cross_root_meta_module": bool(accepted),
            "mean_within_consensus_rho": float(np.mean(vals)) if vals else np.nan,
            "median_within_consensus_rho": float(np.median(vals)) if vals else np.nan,
            "min_within_consensus_rho": float(np.min(vals)) if vals else np.nan,
            "fraction_positive_pairs": float(pos / total) if total else np.nan,
        })
    csum = pd.DataFrame(cluster_rows)

    # Stable meta labels only for accepted cross-root clusters.
    meta_order = [cid for cid in sorted(accepted_cluster_ids)]
    cid_to_meta = {cid: f"{panel}__META{ii+1:02d}" for ii, cid in enumerate(meta_order)}
    mem["meta_module_id"] = mem["ordered_cluster_id"].map(cid_to_meta)
    mem["program_type"] = np.where(mem["meta_module_id"].notna(), "meta_module", "standalone_root_module")
    mem["integrated_program_id"] = mem["meta_module_id"].fillna(mem["root_module_id"])
    mem["linkage_method"] = linkage_method
    mem["rho_threshold"] = float(rho_threshold)

    n_modules = len(mem)
    integrated = int(mem[mem["meta_module_id"].notna()]["root_module_id"].nunique())
    meta_clusters = csum[csum["is_cross_root_meta_module"]]
    diag = {
        "panel": panel,
        "linkage_method": linkage_method,
        "rho_threshold": float(rho_threshold),
        "n_root_modules": int(n_modules),
        "n_raw_clusters": int(csum["ordered_cluster_id"].nunique()),
        "n_cross_root_meta_modules": int(len(meta_clusters)),
        "n_root_modules_integrated": integrated,
        "fraction_root_modules_integrated": float(integrated / n_modules) if n_modules else np.nan,
        "n_standalone_root_modules": int(n_modules - integrated),
        "median_meta_module_size": float(meta_clusters["n_root_modules"].median()) if len(meta_clusters) else np.nan,
        "max_meta_module_size": int(meta_clusters["n_root_modules"].max()) if len(meta_clusters) else 0,
        "median_prep_roots_per_meta": float(meta_clusters["n_prep_roots"].median()) if len(meta_clusters) else np.nan,
        "median_meta_within_consensus_rho": float(meta_clusters["mean_within_consensus_rho"].median()) if len(meta_clusters) else np.nan,
        "min_meta_within_consensus_rho": float(meta_clusters["min_within_consensus_rho"].min()) if len(meta_clusters) else np.nan,
    }
    return mem, csum, diag


def plot_consensus_heatmap(C: pd.DataFrame, module_meta: pd.DataFrame, Z: np.ndarray, path: Path, panel: str) -> None:
    order = list(leaves_list(Z))
    ids = [C.index[i] for i in order]
    M = C.loc[ids, ids]
    meta = module_meta.drop_duplicates("root_module_id").set_index("root_module_id").reindex(ids)
    roots = meta[ROOT_COL].fillna("NA").astype(str).tolist()
    root_levels = list(dict.fromkeys(roots))
    cmap0 = plt.get_cmap("tab20")
    root_to_i = {r: i for i, r in enumerate(root_levels)}
    bar = np.array([[root_to_i[r] for r in roots]])

    fig = plt.figure(figsize=(max(9, len(ids)*0.22+4), max(8, len(ids)*0.20+4)))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 12], width_ratios=[12, 0.45], hspace=0.04, wspace=0.08)
    axb = fig.add_subplot(gs[0,0]); ax = fig.add_subplot(gs[1,0]); cax = fig.add_subplot(gs[1,1])
    im = ax.imshow(M.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest", aspect="auto")
    fig.colorbar(im, cax=cax).set_label("Root-module score consensus Spearman ρ")
    axb.imshow(bar, aspect="auto", interpolation="nearest", cmap=ListedColormap([cmap0(i % cmap0.N) for i in range(len(root_levels))]), vmin=-0.5, vmax=max(len(root_levels)-0.5,0.5))
    axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{panel}: discovery-cohort consensus correlation of root-module scores")
    fig.savefig(path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_primary_heatmap(C: pd.DataFrame, primary_mem: pd.DataFrame, Z: np.ndarray, path: Path, panel: str) -> None:
    leaves = [C.index[i] for i in leaves_list(Z)]
    # Group by integrated program while preserving dendrogram first appearance.
    mm = primary_mem.set_index("root_module_id")
    prog_order = []
    for m in leaves:
        p = str(mm.loc[m, "integrated_program_id"])
        if p not in prog_order:
            prog_order.append(p)
    ids = []
    for p in prog_order:
        ids.extend([m for m in leaves if str(mm.loc[m, "integrated_program_id"]) == p])
    M = C.loc[ids, ids]
    meta = mm.reindex(ids)
    roots = meta[ROOT_COL].fillna("NA").astype(str).tolist()
    programs = meta["integrated_program_id"].astype(str).tolist()
    root_levels = list(dict.fromkeys(roots)); prog_levels = list(dict.fromkeys(programs))
    cm = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(max(10, len(ids)*0.23+4), max(8, len(ids)*0.20+4)))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.38,0.38,12], width_ratios=[12,0.45], hspace=0.03, wspace=0.08)
    axr=fig.add_subplot(gs[0,0]); axp=fig.add_subplot(gs[1,0]); ax=fig.add_subplot(gs[2,0]); cax=fig.add_subplot(gs[2,1])
    im=ax.imshow(M.to_numpy(float),vmin=-1,vmax=1,cmap="RdBu_r",interpolation="nearest",aspect="auto")
    fig.colorbar(im,cax=cax).set_label("Root-module score consensus Spearman ρ")
    rmap={r:i for i,r in enumerate(root_levels)}; pmap={p:i for i,p in enumerate(prog_levels)}
    axr.imshow(np.array([[rmap[r] for r in roots]]),aspect="auto",interpolation="nearest",cmap=ListedColormap([cm(i%cm.N) for i in range(len(root_levels))]),vmin=-0.5,vmax=max(len(root_levels)-0.5,0.5))
    axp.imshow(np.array([[pmap[p] for p in programs]]),aspect="auto",interpolation="nearest",cmap=ListedColormap([cm(i%cm.N) for i in range(len(prog_levels))]),vmin=-0.5,vmax=max(len(prog_levels)-0.5,0.5))
    for a in [axr,axp]:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
    changes=np.where(np.array(programs[1:])!=np.array(programs[:-1]))[0]+1
    for b in changes:
        x=b-0.5; ax.axhline(x,color="black",linewidth=0.8); ax.axvline(x,color="black",linewidth=0.8); axp.axvline(x,color="white",linewidth=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{panel}: primary outcome-blind meta-module solution\nTop bars: prep root, integrated program")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(diag: pd.DataFrame, path: Path, panel: str, primary_threshold: float) -> None:
    d = diag.sort_values("rho_threshold")
    fig, ax = plt.subplots(figsize=(8.5,5.2))
    ax.plot(d["rho_threshold"], d["n_cross_root_meta_modules"], marker="o", label="Cross-root meta-modules")
    ax.plot(d["rho_threshold"], d["fraction_root_modules_integrated"], marker="o", label="Fraction root modules integrated")
    ax.plot(d["rho_threshold"], d["max_meta_module_size"], marker="o", label="Largest meta-module size")
    ax.axvline(primary_threshold, linestyle="--", linewidth=1, label="Primary threshold")
    ax.set_xlabel("Minimum consensus Spearman ρ used to cut hierarchy")
    ax.set_ylabel("Diagnostic value")
    ax.set_title(f"{panel}: meta-module threshold sensitivity")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path,dpi=250,bbox_inches="tight"); plt.close(fig)


def score_primary_meta_modules(scores: pd.DataFrame, membership: pd.DataFrame, cfg: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_mem = membership[membership["meta_module_id"].notna()].copy()
    out_rows = []
    integrated_rows = []
    min_frac = float(cfg.get("min_meta_member_fraction", 0.50))
    min_present_abs = int(cfg.get("min_meta_members_present", 2))

    for (cohort, panel), g in scores.groupby(["cohort", "panel"], sort=True):
        X = g.pivot_table(index="patient_id", columns="root_module_id", values="score_meanz", aggfunc="first")
        Z = X.apply(zscore, axis=0)
        pmem = membership[membership["panel"].astype(str).eq(str(panel))]

        # True cross-root meta-modules.
        for meta_id, gm in pmem[pmem["meta_module_id"].notna()].groupby("meta_module_id", sort=True):
            members = gm["root_module_id"].astype(str).tolist()
            present_cols = [m for m in members if m in Z.columns]
            block = Z[present_cols] if present_cols else pd.DataFrame(index=Z.index)
            if len(present_cols):
                n_present = block.notna().sum(axis=1)
                min_req = max(min_present_abs, int(math.ceil(min_frac * len(members))))
                score = block.mean(axis=1, skipna=True)
                score[n_present < min_req] = np.nan
            else:
                n_present = pd.Series(0,index=Z.index,dtype=int); score=pd.Series(np.nan,index=Z.index)
            for pid in Z.index:
                out_rows.append({
                    "patient_id": str(pid), "cohort": str(cohort), "panel": str(panel),
                    "meta_module_id": str(meta_id), "score_meta_meanz": score.loc[pid],
                    "n_member_root_modules_total": len(members),
                    "n_member_root_modules_present": int(n_present.loc[pid]),
                })
                integrated_rows.append({
                    "patient_id": str(pid), "cohort": str(cohort), "panel": str(panel),
                    "integrated_program_id": str(meta_id), "program_type": "meta_module",
                    "score_integrated_z": score.loc[pid],
                })

        # Standalone root modules are carried forward rather than silently discarded.
        standalone = pmem[pmem["meta_module_id"].isna()]["root_module_id"].astype(str).drop_duplicates().tolist()
        for rid in standalone:
            zz = Z[rid] if rid in Z.columns else pd.Series(np.nan,index=Z.index)
            for pid in Z.index:
                integrated_rows.append({
                    "patient_id": str(pid), "cohort": str(cohort), "panel": str(panel),
                    "integrated_program_id": rid, "program_type": "standalone_root_module",
                    "score_integrated_z": zz.loc[pid],
                })

    return pd.DataFrame(out_rows), pd.DataFrame(integrated_rows)


def command_run(cfg: Mapping) -> None:
    out = ensure_dir(output_root(cfg))
    plot_root = ensure_dir(out / "plots")
    s2c = Path(cfg["stage2c_output_root"])
    score_path = s2c / "all_discovery_root_module_scores_long.parquet"
    meta_path = s2c / "final_root_module_summary.csv"
    if not score_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Finish Stage2C aggregate first: {score_path} / {meta_path}")
    scores = pd.read_parquet(score_path)
    module_meta = pd.read_csv(meta_path)

    min_n = int(cfg.get("min_pairwise_n",20))
    min_support = int(cfg.get("min_pair_support",2))
    min_sign = float(cfg.get("min_sign_consistency",0.75))
    threshold_grid = [float(x) for x in cfg.get("rho_threshold_grid",[0.2,0.25,0.3,0.35,0.4,0.45,0.5])]
    primary_linkage = str(cfg.get("primary_linkage_method","average"))
    linkage_methods = list(dict.fromkeys([primary_linkage] + list(cfg.get("sensitivity_linkage_methods",["complete"]))))
    primary_thresholds = cfg.get("primary_rho_thresholds",{})
    default_thr = float(cfg.get("default_primary_rho_threshold",0.30))
    min_roots = int(cfg.get("min_roots_per_meta_module",2))
    min_modules = int(cfg.get("min_root_modules_per_meta_module",2))

    all_pair=[]; all_diag=[]; all_mem=[]; all_csum=[]; primary_mems=[]; panel_summaries=[]

    for panel in sorted(scores["panel"].astype(str).unique()):
        pdir=ensure_dir(out / "panels" / safe_slug(panel)); ppdir=ensure_dir(plot_root / safe_slug(panel))
        pm = module_meta[module_meta["panel"].astype(str).eq(panel)].copy()
        C,S,SC,SD,audit=build_panel_consensus(scores,panel,min_n)
        audit.to_csv(pdir / "root_module_pairwise_consensus_audit.csv.gz",index=False,compression="gzip")
        all_pair.append(audit)
        C.to_parquet(pdir / "root_module_score_consensus_spearman.parquet")
        S.to_parquet(pdir / "root_module_pair_support.parquet")
        SC.to_parquet(pdir / "root_module_sign_consistency.parquet")
        SD.to_parquet(pdir / "root_module_cross_cohort_rho_sd.parquet")
        Ccl=clustering_consensus(C,S,SC,min_support,min_sign)
        Ccl.to_parquet(pdir / "root_module_consensus_for_meta_clustering.parquet")

        panel_diags=[]; solution_store={}
        for method in linkage_methods:
            D,Z=direct_signed_linkage(Ccl,method)
            if method == primary_linkage:
                plot_consensus_heatmap(C,pm,Z,ppdir / "01_root_module_score_consensus_heatmap.png",panel)
                fig,ax=plt.subplots(figsize=(max(10,len(C)*0.15),5.5)); dendrogram(Z,labels=C.index.tolist(),leaf_rotation=90,leaf_font_size=5,ax=ax); ax.set_ylabel("1 - reproducibility-filtered consensus rho"); ax.set_title(f"{panel}: root-module meta clustering dendrogram ({method})"); fig.tight_layout(); fig.savefig(ppdir / "02_meta_clustering_dendrogram.png",dpi=220,bbox_inches="tight"); plt.close(fig)
            for thr in threshold_grid:
                mem,csum,diag=cluster_solution(C,Ccl,Z,pm,thr,panel,method,min_roots,min_modules)
                all_diag.append(diag); panel_diags.append(diag); all_mem.append(mem); all_csum.append(csum)
                solution_store[(method,float(thr))]=(mem,csum,Z)

        diag_df=pd.DataFrame(panel_diags)
        diag_df.to_csv(pdir / "meta_module_threshold_diagnostics.csv",index=False)
        primary_thr=float(primary_thresholds.get(panel,default_thr))
        # If exact float not in grid, calculate it directly.
        key=(primary_linkage,primary_thr)
        if key not in solution_store:
            _,Z=direct_signed_linkage(Ccl,primary_linkage)
            mem,csum,diag=cluster_solution(C,Ccl,Z,pm,primary_thr,panel,primary_linkage,min_roots,min_modules)
            solution_store[key]=(mem,csum,Z); all_diag.append(diag); all_mem.append(mem); all_csum.append(csum)
        pmem,pcsum,pZ=solution_store[key]
        pmem.to_csv(pdir / "primary_meta_module_membership.csv",index=False)
        pcsum.to_csv(pdir / "primary_meta_module_cluster_summary.csv",index=False)
        primary_mems.append(pmem)
        plot_threshold_sensitivity(diag_df[diag_df["linkage_method"].eq(primary_linkage)],ppdir / "03_meta_threshold_sensitivity.png",panel,primary_thr)
        plot_primary_heatmap(C,pmem,pZ,ppdir / "04_primary_meta_module_heatmap.png",panel)

        meta_only=pcsum[pcsum["is_cross_root_meta_module"]]
        panel_summaries.append({
            "panel":panel,"primary_linkage_method":primary_linkage,"primary_rho_threshold":primary_thr,
            "n_root_modules":int(len(pmem)),"n_meta_modules":int(len(meta_only)),
            "n_root_modules_integrated":int(pmem["meta_module_id"].notna().sum()),
            "n_standalone_root_modules":int(pmem["meta_module_id"].isna().sum()),
            "fraction_root_modules_integrated":float(pmem["meta_module_id"].notna().mean()) if len(pmem) else np.nan,
            "median_meta_module_size":float(meta_only["n_root_modules"].median()) if len(meta_only) else np.nan,
            "max_meta_module_size":int(meta_only["n_root_modules"].max()) if len(meta_only) else 0,
        })

    pair_all=pd.concat(all_pair,ignore_index=True,sort=False) if all_pair else pd.DataFrame()
    diag_all=pd.DataFrame(all_diag)
    mem_all=pd.concat(all_mem,ignore_index=True,sort=False) if all_mem else pd.DataFrame()
    csum_all=pd.concat(all_csum,ignore_index=True,sort=False) if all_csum else pd.DataFrame()
    primary=pd.concat(primary_mems,ignore_index=True,sort=False) if primary_mems else pd.DataFrame()
    psummary=pd.DataFrame(panel_summaries)

    pair_all.to_csv(out / "all_panel_root_module_pairwise_consensus_audit.csv.gz",index=False,compression="gzip")
    diag_all.to_csv(out / "stage2d_meta_threshold_diagnostics.csv",index=False)
    mem_all.to_csv(out / "all_meta_solutions_membership.csv.gz",index=False,compression="gzip")
    csum_all.to_csv(out / "all_meta_solutions_cluster_summary.csv.gz",index=False,compression="gzip")
    primary.to_csv(out / "final_meta_module_membership.csv",index=False)
    psummary.to_csv(out / "stage2d_primary_meta_summary.csv",index=False)

    # Final meta-module summary from the primary memberships.
    mrows=[]
    for (panel,meta_id),g in primary[primary["meta_module_id"].notna()].groupby(["panel","meta_module_id"],sort=True):
        mrows.append({
            "panel":panel,"meta_module_id":meta_id,
            "n_root_modules":int(g["root_module_id"].nunique()),
            "n_prep_roots":int(g[ROOT_COL].nunique()),
            "prep_roots":";".join(sorted(g[ROOT_COL].astype(str).unique())),
            "member_root_modules":";".join(g["root_module_id"].astype(str).tolist()),
        })
    meta_summary=pd.DataFrame(mrows)
    meta_summary.to_csv(out / "final_meta_module_summary.csv",index=False)

    meta_scores, integrated_scores=score_primary_meta_modules(scores,primary,cfg)
    meta_scores.to_parquet(out / "all_discovery_meta_module_scores_long.parquet",index=False)
    meta_scores.to_csv(out / "all_discovery_meta_module_scores_long.csv.gz",index=False,compression="gzip")
    integrated_scores.to_parquet(out / "all_discovery_integrated_program_scores_long.parquet",index=False)
    if not meta_scores.empty:
        mw=meta_scores.pivot_table(index=["patient_id","cohort","panel"],columns="meta_module_id",values="score_meta_meanz",aggfunc="first").reset_index(); mw.columns.name=None
        mw.to_parquet(out / "all_discovery_meta_module_scores_wide.parquet",index=False)
    iw=integrated_scores.pivot_table(index=["patient_id","cohort","panel"],columns="integrated_program_id",values="score_integrated_z",aggfunc="first").reset_index(); iw.columns.name=None
    iw.to_parquet(out / "all_discovery_integrated_program_scores_wide.parquet",index=False)

    # Human-readable summary/review template.
    review=psummary.copy(); review["manual_keep_primary_threshold"]=""; review["manual_alternative_rho_threshold"]=""; review["manual_notes"]=""
    review.to_csv(out / "stage2d_manual_meta_review_template.csv",index=False)
    lines=["STAGE 2D META-MODULE DISCOVERY SUMMARY","="*60,"Outcome-blind; direct signed consensus of Stage2C root-module scores.",f"Pair support >= {min_support}; sign consistency >= {min_sign}; pairwise n >= {min_n}.",""]
    for _,r in psummary.iterrows():
        lines.append(f"{r['panel']}: threshold={r['primary_rho_threshold']:.2f}; meta-modules={int(r['n_meta_modules'])}; integrated root modules={int(r['n_root_modules_integrated'])}/{int(r['n_root_modules'])}; standalone={int(r['n_standalone_root_modules'])}")
    (out / "stage2d_summary.txt").write_text("\n".join(lines)+"\n")
    write_json({
        "primary_linkage_method":primary_linkage,"primary_rho_thresholds":primary_thresholds,
        "default_primary_rho_threshold":default_thr,"min_pairwise_n":min_n,
        "min_pair_support":min_support,"min_sign_consistency":min_sign,
        "n_meta_modules":int(meta_summary["meta_module_id"].nunique()) if not meta_summary.empty else 0,
    },out / "stage2d_run_summary.json")
    log(f"[DONE] meta-modules={len(meta_summary)} output={out}")


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("command",choices=["run"]); return ap.parse_args()


def main() -> None:
    args=parse_args(); cfg=read_json(args.config); command_run(cfg)


if __name__ == "__main__":
    main()
