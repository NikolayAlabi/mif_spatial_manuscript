#!/usr/bin/env python3
"""Stage 4B: DEG-like matched TURBT vs RC screen of frozen root/meta programs."""

from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_1samp

from delta_rc_common_v1 import *


def outroot(cfg): return Path(cfg["stage4_output_root"])
def cache_dir(cfg,c,p): return outroot(cfg)/"score_cache"/f"{safe_slug(c)}__{safe_slug(p)}"
def result_dir(cfg,c,p): return outroot(cfg)/"stage4b_matched_shift"/safe_slug(c)/safe_slug(p)


def bootstrap_median_ci(x, n_boot=1000, seed=42):
    a = safe_numeric(pd.Series(x)).dropna().to_numpy(float)
    if len(a) < 3: return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    vals=np.array([np.median(rng.choice(a,size=len(a),replace=True)) for _ in range(int(n_boot))])
    return tuple(np.quantile(vals,[0.025,0.975]))


def family_name(level, root): return str(root) if level=="root_module" else "meta_modules"


def setup(cfg):
    rows=[]; aid=0
    for cohort in cfg["cohorts"]:
        for panel in cfg["panels"]:
            p=cache_dir(cfg,cohort,panel)/"turbt_reference_program_scores_long.parquet"
            if not p.exists(): continue
            d=pd.read_parquet(p)
            if d.empty: continue
            t=set(d.loc[d.sample_type.eq("TURBT"),"patient_id"].astype(str))
            r=set(d.loc[d.sample_type.eq("RC"),"patient_id"].astype(str))
            n=len(t&r)
            aid+=1
            rows.append({"array_id":aid,"cohort":cohort,"panel":panel,"n_matched_any_program":n})
    pd.DataFrame(rows).to_csv(outroot(cfg)/"stage4b_shift_worker_index.csv",index=False)
    print(f"[SETUP 4B] workers={aid}",flush=True)


def plot_family(summary, matched, level, root_name, pdir, top_n):
    d=summary[(summary.program_level==level)&(summary[ROOT_COL].astype(str)==str(root_name))].copy()
    if d.empty:return
    d=d.sort_values("delta_median")
    fig,ax=plt.subplots(figsize=(8,max(4,.3*len(d))))
    y=np.arange(len(d))
    good=d["median_ci_low"].notna()&d["median_ci_high"].notna()
    if good.any():
        dd=d[good]; yy=y[good.to_numpy()]
        ax.errorbar(dd["delta_median"],yy,xerr=[dd["delta_median"]-dd["median_ci_low"],dd["median_ci_high"]-dd["delta_median"]],fmt="o",capsize=2)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(y);ax.set_yticklabels(d["program_id"],fontsize=7)
    ax.set_xlabel("Median change in score (RC - TURBT)")
    ax.set_title(f"{root_name}: matched TURBT→RC shifts")
    fig.tight_layout(); fig.savefig(pdir/"01_delta_forest.png",dpi=250,bbox_inches="tight");plt.close(fig)

    dv=d.dropna(subset=["paired_wilcoxon_p","delta_median"]).copy()
    if not dv.empty:
        fig,ax=plt.subplots(figsize=(6.5,5.5))
        ax.scatter(dv["delta_median"],-np.log10(dv["paired_wilcoxon_p"].clip(lower=1e-300)))
        for _,r in dv.nsmallest(min(top_n,len(dv)),"paired_wilcoxon_p").iterrows():
            ax.text(r["delta_median"],-np.log10(max(r["paired_wilcoxon_p"],1e-300)),str(r["program_id"]),fontsize=7)
        ax.axvline(0,linestyle="--",linewidth=1)
        ax.set_xlabel("Median delta (RC - TURBT)");ax.set_ylabel("-log10 paired Wilcoxon P")
        ax.set_title(f"{root_name}: magnitude vs paired evidence")
        fig.tight_layout();fig.savefig(pdir/"02_delta_volcano.png",dpi=250,bbox_inches="tight");plt.close(fig)

    programs=d.reindex(d["delta_median"].abs().sort_values(ascending=False).index)["program_id"].head(top_n).tolist()
    for pid in programs:
        z=matched[(matched.program_id==pid)&(matched.program_level==level)].dropna(subset=["TURBT","RC"])
        if len(z)<3:continue
        fig,ax=plt.subplots(figsize=(4,4.5))
        for _,r in z.iterrows():
            ax.plot([0,1],[r["TURBT"],r["RC"]],marker="o",linewidth=.7,alpha=.55)
        ax.set_xticks([0,1]);ax.set_xticklabels(["TURBT","RC"]);ax.set_ylabel("Frozen program score")
        ax.set_title(str(pid));fig.tight_layout()
        fig.savefig(pdir/f"paired_{safe_slug(pid)}.png",dpi=220,bbox_inches="tight");plt.close(fig)


def worker(cfg,array_id):
    row=load_index_row(outroot(cfg)/"stage4b_shift_worker_index.csv",array_id)
    cohort,panel=str(row.cohort),str(row.panel)
    p=cache_dir(cfg,cohort,panel)/"turbt_reference_program_scores_long.parquet"
    d=pd.read_parquet(p)
    tur=d[d.sample_type.eq("TURBT")].rename(columns={"score":"TURBT"})
    rc=d[d.sample_type.eq("RC")].rename(columns={"score":"RC"})
    keys=["patient_id","cohort","panel",ROOT_COL,"program_id","program_level"]
    m=tur[keys+["TURBT"]].merge(rc[keys+["RC"]],on=keys,how="inner")
    m["delta"]=safe_numeric(m["RC"])-safe_numeric(m["TURBT"])

    rows=[]
    for (level,root_name,pid),g in m.groupby(["program_level",ROOT_COL,"program_id"],dropna=False):
        z=g.dropna(subset=["TURBT","RC","delta"]).copy()
        n=len(z)
        r={"cohort":cohort,"panel":panel,"program_level":level,ROOT_COL:root_name,"program_id":pid,"n_pairs":n,
           "primary_pair_eligible":bool(n>=int(cfg.get("primary_min_matched_pairs",10)))}
        if n<3:
            r.update({"fit_status":"too_few_pairs"});rows.append(r);continue
        delta=z.delta
        try:p_w=float(wilcoxon(delta,zero_method="wilcox").pvalue) if (delta!=0).sum()>=3 else np.nan
        except Exception:p_w=np.nan
        try:p_t=float(ttest_1samp(delta,0,nan_policy="omit").pvalue)
        except Exception:p_t=np.nan
        lo,hi=bootstrap_median_ci(delta,cfg.get("bootstrap_reps",1000),cfg.get("random_state",42))
        sd=float(delta.std(ddof=1))
        r.update({
            "fit_status":"ok","turbt_mean":float(z.TURBT.mean()),"rc_mean":float(z.RC.mean()),
            "delta_mean":float(delta.mean()),"delta_median":float(delta.median()),"delta_sd":sd,
            "paired_effect_dz":float(delta.mean()/sd) if np.isfinite(sd) and sd>0 else np.nan,
            "fraction_increased":float((delta>0).mean()),"fraction_decreased":float((delta<0).mean()),
            "median_ci_low":lo,"median_ci_high":hi,
            "paired_wilcoxon_p":p_w,"paired_ttest_p":p_t,
        });rows.append(r)

    s=pd.DataFrame(rows)
    s["q_all_programs"]=bh_adjust(s["paired_wilcoxon_p"]) if "paired_wilcoxon_p" in s else np.nan
    s["q_within_family"]=np.nan
    for _,g in s.groupby(["program_level",ROOT_COL],dropna=False):
        s.loc[g.index,"q_within_family"]=bh_adjust(g["paired_wilcoxon_p"])

    od=ensure_dir(result_dir(cfg,cohort,panel));pd_path=od/"paired_program_delta_long.parquet"
    m.to_parquet(pd_path,index=False);s.to_csv(od/"paired_shift_summary.csv",index=False)
    top_n=int(cfg.get("top_n_paired_plots",8))
    for (level,root_name),_ in s.groupby(["program_level",ROOT_COL],dropna=False):
        pdir=ensure_dir(od/"plots"/safe_slug(root_name))
        plot_family(s,m,str(level),str(root_name),pdir,top_n)
    (od/".done").write_text("complete\n")
    print(f"[DONE 4B] {cohort}/{panel} matched rows={m.patient_id.nunique()} programs={s.program_id.nunique()}",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["setup","worker"]);ap.add_argument("--config",required=True);ap.add_argument("--array-id",type=int)
    a=ap.parse_args();cfg=load_json(a.config)
    setup(cfg) if a.command=="setup" else worker(cfg,resolve_array_id(a.array_id))
if __name__=="__main__":main()
