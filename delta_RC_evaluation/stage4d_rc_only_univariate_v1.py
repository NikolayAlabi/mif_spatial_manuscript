#!/usr/bin/env python3
"""Stage 4D: RC-only OS/RFS univariate screen of clinical variables + frozen root/meta programs."""

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
def cache_dir(cfg,c,p):return outroot(cfg)/"score_cache"/f"{safe_slug(c)}__{safe_slug(p)}"
def result_dir(cfg,c,p,e,s):return outroot(cfg)/"stage4d_rc_only"/safe_slug(c)/safe_slug(p)/safe_slug(e)/safe_slug(s)


def setup(cfg):
    rows=[];audit=[];aid=0
    subset_map=cfg.get("cohort_patient_subsets",{"_default":["all"],"No-NAC":["all","no_adj_chemo"]})
    for cohort in cfg["cohorts"]:
        for panel in cfg["panels"]:
            p=cache_dir(cfg,cohort,panel)/"rc_internal_program_scores_long.parquet"
            if not p.exists():continue
            d=pd.read_parquet(p)
            ids=set(d.patient_id.astype(str)) if not d.empty else set()
            for endpoint in ["OS","RFS"]:
                for subset in subset_map.get(cohort,subset_map.get("_default",["all"])):
                    ep=endpoint_table(cfg,cohort,endpoint,subset)
                    ep=ep[ep.patient_id.astype(str).isin(ids)].copy()
                    q=context_quality(endpoint,ep,cfg)
                    r={"cohort":cohort,"panel":panel,"endpoint":endpoint,"patient_subset":subset,"n_rc_endpoint":len(ep),**q}
                    audit.append(r)
                    if q["can_fit"]:
                        aid+=1;rows.append({"array_id":aid,**r})
    pd.DataFrame(audit).to_csv(outroot(cfg)/"stage4d_rc_context_audit.csv",index=False)
    pd.DataFrame(rows).to_csv(outroot(cfg)/"stage4d_rc_worker_index.csv",index=False)
    print(f"[SETUP 4D] evaluable RC contexts={aid}",flush=True)


def plot_forest(metrics,p):
    d=metrics.dropna(subset=["effect"]).sort_values(["program_level",ROOT_COL,"p_value"],na_position="last")
    if d.empty:return
    y=np.arange(len(d));fig,ax=plt.subplots(figsize=(9,max(4,.26*len(d))))
    h=d.effect_ci_low.notna()&d.effect_ci_high.notna()
    if h.any():
        z=d[h];yy=y[h.to_numpy()]
        ax.errorbar(z.effect,yy,xerr=[z.effect-z.effect_ci_low,z.effect_ci_high-z.effect],fmt="o",capsize=2)
    ax.axvline(1,linestyle="--",linewidth=1);ax.set_xscale("log");ax.set_yticks(y);ax.set_yticklabels(d.program_id,fontsize=6)
    ax.invert_yaxis();ax.set_xlabel("Hazard ratio per SD");ax.set_title("RC-only univariate survival associations")
    fig.tight_layout();fig.savefig(p,dpi=250,bbox_inches="tight");plt.close(fig)


def worker(cfg,array_id):
    row=load_index_row(outroot(cfg)/"stage4d_rc_worker_index.csv",array_id)
    cohort,panel,endpoint,subset=map(str,[row.cohort,row.panel,row.endpoint,row.patient_subset])
    p=cache_dir(cfg,cohort,panel)/"rc_internal_program_scores_long.parquet"
    scores=pd.read_parquet(p)
    scores=scores[scores.sample_type.eq("RC")].copy()
    ep=endpoint_table(cfg,cohort,endpoint,subset)

    # Wide frozen spatial scores.
    X=scores.pivot_table(index="patient_id",columns=["program_level",ROOT_COL,"program_id"],values="score",aggfunc="first")
    X.columns=["|||".join(map(str,c)) for c in X.columns];X=X.reset_index()
    data=X.merge(ep,on="patient_id",how="inner")

    # RC clinical variables.
    clin,audit=load_rc_clinical_variables(cfg,cohort);data=data.merge(clin,on="patient_id",how="left")

    evals=[]
    for c in ["Age","Sex","ypT","ypN"]:
        if c in data.columns:
            evals.append(("clinical_variable","clinical",c,c))
    for c in [x for x in data.columns if "|||" in x]:
        level,root_name,pid=c.split("|||",2);evals.append((level,root_name,pid,c))

    rows=[];fold_parts=[];pred_parts=[];nrep=int(cfg.get("cv_n_repeats",5));ns=int(cfg.get("cv_n_splits",5));seed=int(cfg.get("random_state",42))
    for ii,(level,root_name,pid,col) in enumerate(evals):
        full=fullfit_survival(data[col],data.time,data.event)
        oof,pred,folds=repeated_oof_survival(data[col],data.time,data.event,data.patient_id,ns,nrep,seed+ii*100)
        coef=full.get("coef",np.nan);dc=np.nan;fsd=np.nan;vf=0
        if not folds.empty:
            vf=int(folds.fold_metric.notna().sum());fsd=float(folds.fold_metric.std(ddof=1));fc=folds.fold_coef.dropna()
            if len(fc) and np.isfinite(coef):dc=float((np.sign(fc)==np.sign(coef)).mean())
            folds=folds.copy();folds["program_id"]=pid;folds["program_level"]=level;folds[ROOT_COL]=root_name;fold_parts.append(folds)
        if not pred.empty:
            pred=pred.copy();pred["program_id"]=pid;pred["program_level"]=level;pred[ROOT_COL]=root_name;pred_parts.append(pred)
        rows.append({
            "cohort":cohort,"panel":panel,"endpoint":endpoint,"patient_subset":subset,
            "program_level":level,ROOT_COL:root_name,"program_id":pid,
            **full,"oof_metric":oof,"fold_sd":fsd,"direction_consistency":dc,"valid_folds":vf,
            "context_primary_eligible":bool(row.primary_eligible),"context_n_signal":row.n_signal,
            "analysis_role":"RC_landmark_prognostic",
        })

    m=pd.DataFrame(rows);m["q_all_programs"]=bh_adjust(m.p_value);m["q_within_family"]=np.nan
    for _,g in m.groupby(["program_level",ROOT_COL],dropna=False):m.loc[g.index,"q_within_family"]=bh_adjust(g.p_value)

    od=ensure_dir(result_dir(cfg,cohort,panel,endpoint,subset))
    m.to_csv(od/"rc_univariate_metrics.csv",index=False);audit.to_csv(od/"rc_clinical_variable_availability.csv",index=False)
    (pd.concat(fold_parts,ignore_index=True) if fold_parts else pd.DataFrame()).to_csv(od/"rc_fold_metrics.csv",index=False)
    (pd.concat(pred_parts,ignore_index=True) if pred_parts else pd.DataFrame()).to_csv(od/"rc_oof_predictions.csv.gz",index=False,compression="gzip")
    plot_forest(m,od/"01_all_univariate_forest.png")
    for (level,root_name),g in m.groupby(["program_level",ROOT_COL],dropna=False):
        plot_forest(g,ensure_dir(od/"plots")/f"forest_{safe_slug(root_name)}.png")
    (od/".done").write_text("complete\n")
    print(f"[DONE 4D] {cohort}/{panel}/{endpoint}/{subset}: {len(m)} variables/programs",flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["setup","worker"]);ap.add_argument("--config",required=True);ap.add_argument("--array-id",type=int)
    a=ap.parse_args();cfg=load_json(a.config)
    setup(cfg) if a.command=="setup" else worker(cfg,resolve_array_id(a.array_id))
if __name__=="__main__":main()
