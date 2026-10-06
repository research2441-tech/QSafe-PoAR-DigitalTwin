from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats


def confusion_metrics(tp, tn, fp, fn):
    total=max(1,tp+tn+fp+fn)
    acc=(tp+tn)/total
    precision=tp/max(1,tp+fp)
    recall=tp/max(1,tp+fn)
    f1=2*precision*recall/max(1e-12,precision+recall)
    return {"accuracy":acc,"precision":precision,"recall":recall,"f1":f1}


def ci95(values):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    if n<2:
        return (float(x[0]) if n else np.nan, float(x[0]) if n else np.nan)
    m=float(np.mean(x)); se=stats.sem(x)
    h=float(stats.t.ppf(0.975,n-1)*se)
    return m-h,m+h


def summarize_runs(df: pd.DataFrame, group_cols=("method","scenario")):
    metrics=[
        "warning_accuracy","synchronization_fidelity","recovery_latency_s",
        "finality_s","reachability","validator_energy_efficiency","cri",
        "communication_bytes","pqc_overhead_ms"
    ]
    rows=[]
    for keys,g in df.groupby(list(group_cols)):
        keys=(keys,) if not isinstance(keys,tuple) else keys
        base=dict(zip(group_cols,keys))
        base["n"]=len(g)
        for m in metrics:
            vals=g[m].astype(float).replace([np.inf,-np.inf],np.nan).dropna()
            if len(vals):
                lo,hi=ci95(vals)
                base[m+"_mean"]=float(vals.mean())
                base[m+"_sd"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
                base[m+"_ci95_lo"]=float(lo); base[m+"_ci95_hi"]=float(hi)
        rows.append(base)
    return pd.DataFrame(rows)


def cohens_d_paired(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    sd=d.std(ddof=1) if len(d)>1 else 0
    return float(d.mean()/sd) if sd>1e-12 else 0.0


def holm_adjust(pvals):
    p=np.asarray(pvals,dtype=float)
    n=len(p); order=np.argsort(p); out=np.empty(n)
    running=0.0
    for rank,idx in enumerate(order):
        adj=(n-rank)*p[idx]
        running=max(running,adj)
        out[idx]=min(1.0,running)
    return out


def compare_to_proposed(df: pd.DataFrame, metric="cri"):
    proposed=df[df.method=="proposed"][["scenario","seed",metric]].rename(columns={metric:"proposed"})
    rows=[]
    for method in sorted(set(df.method)-{"proposed"}):
        sub=df[df.method==method][["scenario","seed",metric]].rename(columns={metric:"baseline"})
        m=proposed.merge(sub,on=["scenario","seed"]).dropna()
        if len(m)<2:
            continue
        t,p=stats.ttest_rel(m["proposed"],m["baseline"])
        rows.append({
            "metric":metric,"baseline":method,"n_pairs":len(m),
            "t_stat":float(t),"df":int(len(m)-1),"p_value":float(p),
            "effect_dz":cohens_d_paired(m["proposed"],m["baseline"]),
            "mean_difference":float((m["proposed"]-m["baseline"]).mean())
        })
    if rows:
        adj=holm_adjust([r["p_value"] for r in rows])
        for r,a in zip(rows,adj):
            r["p_holm"]=float(a)
    return pd.DataFrame(rows)


def carbon_resilience_index(config, warning_quality, sync_fidelity, recovery_utility,
                            reachability, finality_s, pqc_ms, carbon_cost):
    w=config["carbon_resilience_index"]["weights"]
    finality_eff=1/(1+max(0,finality_s))
    pqc_eff=1/(1+max(0,pqc_ms)/10.0)
    carbon_eff=1-np.clip(carbon_cost,0,1)
    score=(
        w["warning_quality"]*warning_quality +
        w["synchronization_fidelity"]*sync_fidelity +
        w["recovery_utility"]*recovery_utility +
        w["reachability"]*reachability +
        w["finality_efficiency"]*finality_eff +
        w["pqc_efficiency"]*pqc_eff +
        w["carbon_efficiency"]*carbon_eff
    )
    return float(np.clip(score,0,1))
