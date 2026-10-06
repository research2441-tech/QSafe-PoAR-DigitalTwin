from __future__ import annotations
from copy import deepcopy
from itertools import product
import pandas as pd
from simulation import run_once


ABLATIONS = {
    "full": {},
    "no_twin_correction": {"twin_correction":False},
    "centralized_recovery": {"federated":False},
    "no_pqc": {"pqc":False},
    "no_energy_awareness": {"poar_zero":"energy"},
    "no_carbon_awareness": {"poar_zero":"carbon_cost"},
    "no_dynamic_routing": {"dynamic_routing":False},
    "round_robin_selection": {"selection":"round_robin"},
}


def run_ablations(config: dict, quick=False):
    seeds=list(config["randomization"]["seeds"])
    if quick: seeds=seeds[:2]
    rows=[]
    for scenario in ["S1","S4","S6","S8"]:
        for seed in seeds:
            for name,cfg in ABLATIONS.items():
                r=run_once(config,scenario,seed,cfg,f"ablation_{name}")
                rows.append(r.__dict__)
    return pd.DataFrame(rows)


def run_sensitivity(config: dict, quick=False):
    seeds=list(config["randomization"]["seeds"])
    if quick: seeds=seeds[:2]
    score_thresholds=[0.40,0.50,0.60]
    comm_thresholds=[0.25,0.35,0.45]
    carbon_weights=[0.05,0.10,0.15]
    rows=[]
    for st,ct,cw in product(score_thresholds,comm_thresholds,carbon_weights):
        cfg=deepcopy(config)
        cfg["poar"]["thresholds"]["poar_score"]=st
        cfg["poar"]["thresholds"]["communication"]=ct
        w=cfg["poar"]["weights"]
        old=w["carbon_cost"]
        rest=sum(v for k,v in w.items() if k!="carbon_cost")
        scale=(1-cw)/rest
        for k in w:
            if k!="carbon_cost": w[k]*=scale
        w["carbon_cost"]=cw
        for seed in seeds:
            r=run_once(cfg,"S8",seed,{},"sensitivity")
            d=r.__dict__
            d.update({"score_threshold":st,"communication_threshold":ct,"carbon_weight":cw})
            rows.append(d)
    return pd.DataFrame(rows)
