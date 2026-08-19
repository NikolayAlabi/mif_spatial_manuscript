#!/usr/bin/env python3
"""Stage 4C: univariate association of matched RC-TURBT deltas with response and RC-origin survival."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delta_rc_common_v1 import *


def outroot(cfg):return Path(cfg["stage4_output_root"])
def shift_dir(cfg,c,p):return outroot(cfg)/"stage4b_matched_shift"/safe_slug(c)/safe_slug(p)
def result_dir(cfg,c,p,e,s):return outroot(cfg)/"stage4c_delta_outcomes"/safe_slug(c)/safe_slug(p)/safe_slug(e)/safe_slug(s)


def setup(cfg):
    rows=[];audit=[];aid=0
    subset_map=cfg.get("cohort_patient_subsets",{"_default":["all"],"No-NAC":["all","no_adj_chemo"]})
    for cohort in cfg["cohorts"]:
        for panel in cfg["panels"]:
            pp=shift_dir(cfg,cohort,panel)/"paired_program_delta_long.parquet"
            if not pp.exists():continue
            d=pd.read_parquet(pp)
            ids=set(d.patient_id.astype(str))
            for endpoint in cfg["delta_endpoints"]:
                for subset in subset_map.get(cohort,subset_map.get("_default",["all"])):
                    if subset!="all" and endpoint in RESPONSE_ENDPOINTS:continue
                    ep=endpoint_table(cfg,cohort,endpoint,subset)
                    ep=ep[ep.patient_id.astype(str).isin(ids)].copy()
                    q=context_quality(endpoint,ep,cfg)
                    r={"cohort":cohort,"panel":panel,"endpoint":endpoint,"patient_subset":subset,"n_matched_endpoint":len(ep),**q}
                    audit.append(r)
                    if q["can_fit"]:
                        aid+=1;rows.append({"array_id":aid,**r})
    pd.DataFrame(audit).to_csv(outroot(cfg)/"stage4c_delta_context_audit.csv",index=False)
    pd.DataFrame(rows).to_csv(outroot(cfg)/"stage4c_delta_worker_index.csv",index=False)
    print(f"[SETUP 4C] evaluable contexts={aid}",flush=True)


def plot_forest(metrics,endpoint,p):
    d=metrics.dropna(subset=["effect"]).sort_values(["program_level",ROOT_COL,"p_value"],na_position="last")
    if d.empty:return
    y=np.arange(len(d));fig,ax=plt.subplots(figsize=(9,max(4,.26*len(d))))
    h=d.effect_ci_low.notna()&d.effect_ci_high.notna()
    if h.any():
        z=d[h];yy=y[h.to_numpy()]
        ax.errorbar(z.effect,yy,xerr=[z.effect-z.effect_ci_low,z.effect_ci_high-z.effect],fmt="o",capsize=2)
    ax.axvline(1,linestyle="--",linewidth=1);ax.set_xscale("log");ax.set_yticks(y);ax.set_yticklabels(d.program_id,fontsize=6)
    ax.invert_yaxis();ax.set_xlabel("OR per SD delta" if endpoint in RESPONSE_ENDPOINTS else "HR per SD delta")
    ax.set_title("Delta univariate associations");fig.tight_layout();fig.savefig(p,dpi=250,bbox_inches="tight");plt.close(fig)


def worker(cfg,array_id):
    row=load_index_row(outroot(cfg)/"stage4c_delta_worker_index.csv",array_id)
    cohort,panel,endpoint,subset=map(str,[row.cohort,row.panel,row.endpoint,row.patient_subset])
    d=pd.read_parquet(shift_dir(cfg,cohort,panel)/"paired_program_delta_long.parquet")
    ep=endpoint_table(cfg,cohort,endpoint,subset)
    data=d.merge(ep,on="patient_id",how="inner")
    rows=[];fold_parts=[];pred_parts=[]
    nrep=int(cfg.get("cv_n_repeats",5));ns=int(cfg.get("cv_n_splits",5));seed=int(cfg.get("random_state",42))
    for ii,((level,root_name,pid),g) in enumerate(data.groupby(["program_level",ROOT_COL,"program_id"],dropna=False)):
        if endpoint in RESPONSE_ENDPOINTS:
            full=fullfit_response(g.delta,g.y);oof,pred,folds=repeated_oof_response(g.delta,g.y,g.patient_id,ns,nrep,seed+ii*100)
        else:
            full=fullfit_survival(g.delta,g.time,g.event);oof,pred,folds=repeated_oof_survival(g.delta,g.time,g.event,g.patient_id,ns,nrep,seed+ii*100)
        coef=full.get("coef",np.nan)
        dc=np.nan;fsd=np.nan;vf=0
        if not folds.empty:
            vf=int(folds.fold_metric.notna().sum());fsd=float(folds.fold_metric.std(ddof=1))
            fc=folds.fold_coef.dropna()
            if len(fc) and np.isfinite(coef):dc=float((np.sign(fc)==np.sign(coef)).mean())
            folds=folds.copy();folds["program_id"]=pid;folds["program_level"]=level;folds[ROOT_COL]=root_name;fold_parts.append(folds)
        if not pred.empty:
            pred=pred.copy();pred["program_id"]=pid;pred["program_level"]=level;pred[ROOT_COL]=root_name;pred_parts.append(pred)
        rows.append({
            "cohort":cohort,"panel":panel,"endpoint":endpoint,"patient_subset":subset,
            "program_level":level,ROOT_COL:root_name,"program_id":pid,
            **full,"oof_metric":oof,"fold_sd":fsd,"direction_consistency":dc,"valid_folds":vf,
            "context_primary_eligible":bool(row.primary_eligible),"context_n_signal":row.n_signal,
            "analysis_role":"response_association_of_post_treatment_change" if endpoint in RESPONSE_ENDPOINTS else "RC_landmark_prognostic_delta",
        })
    m=pd.DataFrame(rows);m["q_within_family"]=np.nan;m["q_all_programs"]=bh_adjust(m.p_value)
    for _,g in m.groupby(["program_level",ROOT_COL],dropna=False):m.loc[g.index,"q_within_family"]=bh_adjust(g.p_value)
    od=ensure_dir(result_dir(cfg,cohort,panel,endpoint,subset))
    m.to_csv(od/"delta_univariate_metrics.csv",index=False)
    (pd.concat(fold_parts,ignore_index=True) if fold_parts else pd.DataFrame()).to_csv(od/"delta_fold_metrics.csv",index=False)
    (pd.concat(pred_parts,ignore_index=True) if pred_parts else pd.DataFrame()).to_csv(od/"delta_oof_predictions.csv.gz",index=False,compression="gzip")
    plot_forest(m,endpoint,od/"delta_univariate_forest.png")
    for (level,root_name),g in m.groupby(["program_level",ROOT_COL],dropna=False):
        plot_forest(g,endpoint,ensure_dir(od/"plots")/f"forest_{safe_slug(root_name)}.png")
    (od/".done").write_text("complete\n")
    print(f"[DONE 4C] {cohort}/{panel}/{endpoint}/{subset}: {len(m)} programs",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["setup","worker"]);ap.add_argument("--config",required=True);ap.add_argument("--array-id",type=int)
    a=ap.parse_args();cfg=load_json(a.config)
    setup(cfg) if a.command=="setup" else worker(cfg,resolve_array_id(a.array_id))
if __name__=="__main__":main()
