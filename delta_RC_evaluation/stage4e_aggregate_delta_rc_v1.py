#!/usr/bin/env python3
"""Stage 4E: aggregate matched-shift, delta-outcome, and RC-only results + collaborator-review plots."""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delta_rc_common_v1 import *


def outroot(cfg):return Path(cfg["stage4_output_root"])


def modkey(s):
    m=re.search(r"(?:META|M)(\d+)",str(s));return int(m.group(1)) if m else 9999


def effect_dotplot(d,panel,level,root_name,endpoints,path,title):
    x=d[(d.panel==panel)&(d.program_level==level)&d.endpoint.isin(endpoints)].copy()
    if root_name is not None:x=x[x[ROOT_COL].astype(str)==str(root_name)]
    if x.empty:return
    # Red = favorable: response +coef; survival -coef.
    x["fav_coef"]=np.where(x.endpoint.isin(SURVIVAL_ENDPOINTS),-safe_numeric(x.coef),safe_numeric(x.coef))
    x["context"]=x.endpoint.astype(str)+" | "+x.cohort.astype(str)+np.where(x.patient_subset.astype(str).ne("all")," | "+x.patient_subset.astype(str),"")
    ctx=[]
    for e in endpoints:
        for c in ["NAC2020","PURE01","BLASST","No-NAC","NAC2015"]:
            for s in ["all","no_adj_chemo"]:
                lab=f"{e} | {c}" if s=="all" else f"{e} | {c} | {s}"
                if ((x.endpoint==e)&(x.cohort==c)&(x.patient_subset.astype(str)==s)).any():ctx.append(lab)
    progs=sorted(x.program_id.astype(str).unique(),key=modkey);xp={c:i for i,c in enumerate(ctx)};yp={m:i for i,m in enumerate(progs)}
    x["xx"]=x.context.map(xp);x["yy"]=x.program_id.astype(str).map(yp)
    p=safe_numeric(x.p_value).clip(lower=1e-300);nl=-np.log10(p);den=np.nanquantile(nl,.95) if nl.notna().any() else 1;den=max(float(den) if np.isfinite(den) else 1,1e-6)
    sizes=(35+180*nl/den).clip(35,260)
    lim=np.nanquantile(np.abs(x.fav_coef),.95) if x.fav_coef.notna().any() else 1;lim=max(float(lim),.1)
    fig,ax=plt.subplots(figsize=(max(9,.75*len(ctx)),max(5,.26*len(progs))))
    sc=ax.scatter(x.xx,x.yy,c=x.fav_coef,s=sizes,cmap="coolwarm",vmin=-lim,vmax=lim,edgecolor="black",linewidth=.3)
    ax.set_xticks(range(len(ctx)));ax.set_xticklabels(ctx,rotation=45,ha="right",fontsize=8);ax.set_yticks(range(len(progs)));ax.set_yticklabels(progs,fontsize=6);ax.invert_yaxis();ax.set_title(title)
    cb=fig.colorbar(sc,ax=ax,fraction=.025,pad=.02);cb.set_label("Favorable signed coefficient\nred = better outcome")
    fig.tight_layout();fig.savefig(path,dpi=250,bbox_inches="tight");plt.close(fig)


def shift_dotplot(d,panel,level,root_name,path,title):
    x=d[(d.panel==panel)&(d.program_level==level)].copy()
    if root_name is not None:x=x[x[ROOT_COL].astype(str)==str(root_name)]
    if x.empty:return
    progs=sorted(x.program_id.astype(str).unique(),key=modkey);coh=[c for c in ["NAC2020","PURE01","BLASST","No-NAC","NAC2015"] if c in x.cohort.unique()]
    xp={c:i for i,c in enumerate(coh)};yp={p:i for i,p in enumerate(progs)}
    lim=np.nanquantile(np.abs(x.delta_median),.95) if x.delta_median.notna().any() else 1;lim=max(float(lim),.1)
    q=safe_numeric(x.q_within_family).clip(lower=1e-300);nl=-np.log10(q);den=np.nanquantile(nl,.95) if nl.notna().any() else 1;den=max(float(den) if np.isfinite(den) else 1,1e-6)
    fig,ax=plt.subplots(figsize=(max(7,.7*len(coh)),max(5,.26*len(progs))))
    sc=ax.scatter(x.cohort.map(xp),x.program_id.astype(str).map(yp),c=x.delta_median,s=(35+180*nl/den).clip(35,260),cmap="coolwarm",vmin=-lim,vmax=lim,edgecolor="black",linewidth=.3)
    ax.set_xticks(range(len(coh)));ax.set_xticklabels(coh,rotation=30,ha="right");ax.set_yticks(range(len(progs)));ax.set_yticklabels(progs,fontsize=6);ax.invert_yaxis();ax.set_title(title)
    cb=fig.colorbar(sc,ax=ax,fraction=.025,pad=.02);cb.set_label("Median RC - TURBT score\nred = increased at RC")
    fig.tight_layout();fig.savefig(path,dpi=250,bbox_inches="tight");plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();cfg=load_json(a.config)
    root=outroot(cfg);out=ensure_dir(root/"stage4e_aggregate");plots=ensure_dir(out/"plots")

    shifts=[];delta=[];rc=[]
    for p in (root/"stage4b_matched_shift").glob("*/*/paired_shift_summary.csv"):
        shifts.append(pd.read_csv(p))
    for p in (root/"stage4c_delta_outcomes").glob("*/*/*/*/delta_univariate_metrics.csv"):
        delta.append(pd.read_csv(p))
    for p in (root/"stage4d_rc_only").glob("*/*/*/*/rc_univariate_metrics.csv"):
        rc.append(pd.read_csv(p))

    S=pd.concat(shifts,ignore_index=True,sort=False) if shifts else pd.DataFrame()
    D=pd.concat(delta,ignore_index=True,sort=False) if delta else pd.DataFrame()
    R=pd.concat(rc,ignore_index=True,sort=False) if rc else pd.DataFrame()
    S.to_csv(out/"all_matched_shift_metrics.csv",index=False);D.to_csv(out/"all_delta_outcome_metrics.csv",index=False);R.to_csv(out/"all_rc_only_metrics.csv",index=False)

    if not S.empty:
        cs=(S[S.primary_pair_eligible.astype(str).str.lower().isin(["true","1"])]
            .groupby(["panel","program_level",ROOT_COL,"program_id"],dropna=False)
            .agg(n_cohorts=("cohort","nunique"),median_of_cohort_median_delta=("delta_median","median"),
                 mean_of_cohort_median_delta=("delta_median","mean"),
                 n_increased=("delta_median",lambda x:int((x>0).sum())),
                 n_decreased=("delta_median",lambda x:int((x<0).sum())),
                 min_family_q=("q_within_family","min"))
            .reset_index())
        cs["direction_consistency"]=cs[["n_increased","n_decreased"]].max(axis=1)/cs["n_cohorts"].replace(0,np.nan)
        cs.to_csv(out/"matched_shift_cross_cohort_summary.csv",index=False)

    for panel in ["AR","BT"]:
        if not S.empty:
            roots=sorted(S[(S.panel==panel)&(S.program_level=="root_module")][ROOT_COL].dropna().astype(str).unique())
            for rn in roots:
                pdir=ensure_dir(plots/"matched_shift"/panel/"roots"/safe_slug(rn))
                shift_dotplot(S,panel,"root_module",rn,pdir/"01_cross_cohort_shift_dotplot.png",f"{panel}/{rn}: matched TURBT→RC changes")
            pdir=ensure_dir(plots/"matched_shift"/panel/"meta_modules")
            shift_dotplot(S,panel,"meta_module",None,pdir/"01_cross_cohort_shift_dotplot.png",f"{panel}: meta-module TURBT→RC changes")

        if not D.empty:
            roots=sorted(D[(D.panel==panel)&(D.program_level=="root_module")][ROOT_COL].dropna().astype(str).unique())
            for rn in roots:
                pdir=ensure_dir(plots/"delta_outcomes"/panel/"roots"/safe_slug(rn))
                effect_dotplot(D,panel,"root_module",rn,["any_response","complete_response"],pdir/"01_response_effect_dotplot.png",f"{panel}/{rn}: delta-response associations")
                effect_dotplot(D,panel,"root_module",rn,["OS","RFS"],pdir/"02_survival_effect_dotplot.png",f"{panel}/{rn}: delta survival associations")
            pdir=ensure_dir(plots/"delta_outcomes"/panel/"meta_modules")
            effect_dotplot(D,panel,"meta_module",None,["any_response","complete_response"],pdir/"01_response_effect_dotplot.png",f"{panel}: meta-delta response associations")
            effect_dotplot(D,panel,"meta_module",None,["OS","RFS"],pdir/"02_survival_effect_dotplot.png",f"{panel}: meta-delta survival associations")

        if not R.empty:
            roots=sorted(R[(R.panel==panel)&(R.program_level=="root_module")][ROOT_COL].dropna().astype(str).unique())
            for rn in roots:
                pdir=ensure_dir(plots/"rc_only"/panel/"roots"/safe_slug(rn))
                effect_dotplot(R,panel,"root_module",rn,["OS","RFS"],pdir/"01_survival_effect_dotplot.png",f"{panel}/{rn}: RC-only survival associations")
            pdir=ensure_dir(plots/"rc_only"/panel/"meta_modules")
            effect_dotplot(R,panel,"meta_module",None,["OS","RFS"],pdir/"01_survival_effect_dotplot.png",f"{panel}: RC-only meta-module survival associations")
            pdir=ensure_dir(plots/"rc_only"/panel/"clinical_variables")
            effect_dotplot(R,panel,"clinical_variable",None,["OS","RFS"],pdir/"01_survival_effect_dotplot.png",f"{panel}: RC clinical-variable survival associations")

    (out/"stage4e_summary.txt").write_text(
        f"matched_shift_rows={len(S)}\ndelta_outcome_rows={len(D)}\nrc_only_rows={len(R)}\n"
        "Delta/RC survival uses RC as the time origin.\n"
        "Red in outcome dotplots = favorable (response or longer survival).\n"
    )
    print(f"[DONE 4E] {out}",flush=True)

if __name__=="__main__":main()
