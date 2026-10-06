from __future__ import annotations
from pathlib import Path
import pandas as pd
from baselines import configured_names, get_baseline
from simulation import run_once
from evaluation import summarize_runs, compare_to_proposed


SCENARIOS=[f"S{i}" for i in range(1,9)]


def run_experiments(config: dict, quick: bool=False, include_extended: bool=True,
                    attacks: bool=True):
    seeds=list(config["randomization"]["seeds"])
    if quick:
        seeds=seeds[:2]
    methods=configured_names(include_extended=include_extended)
    rows=[]
    for scenario in SCENARIOS:
        for seed in seeds:
            for name in methods:
                rows.append(run_once(config,scenario,seed,get_baseline(name),name).__dict__)
    if attacks:
        for scenario in ["S6","S8"]:
            for seed in seeds:
                for attack in ["spoof_sensors","partition","model_poison","dos","score_manipulation"]:
                    r=run_once(config,scenario,seed,{},f"proposed_attack_{attack}",attack)
                    rows.append(r.__dict__)
    df=pd.DataFrame(rows)
    return df


def export_results(df: pd.DataFrame, outdir: str | Path):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"run_level_results.csv",index=False)
    main=df[~df.method.str.startswith("proposed_attack_")].copy()
    summarize_runs(main).to_csv(out/"summary_mean_sd_ci.csv",index=False)
    for metric in ["cri","warning_accuracy","synchronization_fidelity","recovery_latency_s",
                   "finality_s","reachability","validator_energy_efficiency"]:
        compare_to_proposed(main,metric).to_csv(out/f"stats_{metric}.csv",index=False)
    return out
