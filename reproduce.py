from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
from data_pipeline import load_yaml
from experiments import run_experiments, export_results
from ablation_sensitivity import run_ablations, run_sensitivity


def main():
    p=argparse.ArgumentParser(description="Reproduce QSafe-PoAR-DigitalTwin experiments")
    p.add_argument("--config",default="config.yaml")
    p.add_argument("--outdir",default="results")
    p.add_argument("--quick",action="store_true",help="two seeds for smoke testing")
    p.add_argument("--no-attacks",action="store_true")
    args=p.parse_args()

    cfg=load_yaml(args.config)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    runs=run_experiments(cfg,quick=args.quick,include_extended=True,attacks=not args.no_attacks)
    export_results(runs,out)

    ab=run_ablations(cfg,quick=args.quick)
    ab.to_csv(out/"ablation_run_level.csv",index=False)

    sens=run_sensitivity(cfg,quick=args.quick)
    sens.to_csv(out/"poar_sensitivity_run_level.csv",index=False)

    metadata={
        "project":cfg["project"],
        "quick":args.quick,
        "n_main_rows":int(len(runs)),
        "n_ablation_rows":int(len(ab)),
        "n_sensitivity_rows":int(len(sens)),
        "note":"All aggregate results must be derived from run-level output; no manuscript result is hard-coded."
    }
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

    print(f"Completed. Results written to: {out.resolve()}")
    print(runs[runs.method=="proposed"].groupby("scenario")[
        ["warning_accuracy","synchronization_fidelity","recovery_latency_s",
         "finality_s","reachability","validator_energy_efficiency","cri"]
    ].mean().round(4))


if __name__=="__main__":
    main()
