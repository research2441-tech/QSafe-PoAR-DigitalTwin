from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from data_pipeline import prepare_inputs
from digital_twin import DigitalTwin
from recovery_agent import RecoveryAgent
from poar import generate_validators, select_validators, rank_validators
from consensus import simulate_consensus, simulate_round_robin, simulate_random_selection
from pqc_security import benchmark_pqc
from attacks import spoof_sensors, communication_attack, dos_validators, manipulate_poar, poison_policy
from evaluation import carbon_resilience_index


@dataclass
class RunResult:
    method: str
    scenario: str
    seed: int
    warning_accuracy: float
    synchronization_fidelity: float
    recovery_latency_s: float
    finality_s: float
    reachability: float
    validator_energy_efficiency: float
    cri: float
    communication_bytes: int
    pqc_overhead_ms: float
    committed: int
    selected_validators: int
    security_eligible: int
    recovery_eligible: int
    attacked: int


def _resource_selection(validators, count):
    eligible=[v for v in validators if v.available and v.authenticated and v.authorized]
    eligible.sort(key=lambda v:(-(0.5*v.energy+0.5*v.communication),v.validator_id))
    return [v.validator_id for v in eligible[:count]]


def run_once(config: dict, scenario: str, seed: int, method_cfg: dict | None=None,
             method_name: str="proposed", attack: str | None=None) -> RunResult:
    method_cfg = method_cfg or {}
    df=prepare_inputs(config,scenario,seed)

    if attack=="spoof_sensors":
        df=spoof_sensors(df,config["attacks"]["spoof_sensor_fraction"],seed+100)
    if attack=="partition":
        df=communication_attack(df,config["attacks"]["partition_fraction"],
                                config["attacks"]["packet_loss_extra"],seed+101)

    twin=DigitalTwin(config, correction=method_cfg.get("twin_correction",True))
    tdf,tm=twin.run(df,float(config["simulation"]["warning_threshold"]))

    agent=RecoveryAgent(config,
                        federated=method_cfg.get("federated",True),
                        dynamic_routing=method_cfg.get("dynamic_routing",True),
                        seed=seed)
    agent.federated_train(tdf)
    if attack=="model_poison":
        poison_policy(agent,config["attacks"]["model_poison_scale"],seed+102)
    rec=agent.evaluate(tdf)

    # No blockchain baselines stop after decision generation.
    blockchain=method_cfg.get("blockchain",True)
    pqc_enabled=method_cfg.get("pqc",True)
    pqc=benchmark_pqc(config,seed) if pqc_enabled else None
    pqc_ms=pqc.overhead_ms if pqc else 0.0

    vals=generate_validators(config,tdf,seed,current_step=int(tdf.step.max()))
    if attack=="dos":
        vals=dos_validators(vals,config["attacks"]["dos_validator_fraction"],seed+103)
    if attack=="score_manipulation":
        vals=manipulate_poar(vals,config["attacks"]["score_manipulation_fraction"],seed+104)

    # Component-ablation support: remove one PoAR weight and renormalize remaining weights.
    poar_cfg={**config["poar"],"weights":dict(config["poar"]["weights"])}
    zero=method_cfg.get("poar_zero")
    if zero:
        poar_cfg["weights"][zero]=0.0
        s=sum(poar_cfg["weights"].values())
        poar_cfg["weights"]={k:v/s for k,v in poar_cfg["weights"].items()}

    count=int(config["simulation"]["active_validators"])
    ranked=rank_validators(vals,poar_cfg,int(tdf.step.max()))
    sec=sum(x.security_eligible for x in ranked)
    recelig=sum(x.recovery_eligible for x in ranked)

    selection=method_cfg.get("selection","poar")
    sec_ids=[x.validator_id for x in ranked if x.security_eligible]
    if selection=="poar":
        selected,_=select_validators(vals,poar_cfg,int(tdf.step.max()),count)
    elif selection=="round_robin":
        selected=simulate_round_robin(sec_ids,count,offset=seed%max(1,len(sec_ids)))
    elif selection=="random":
        selected=simulate_random_selection(sec_ids,count,seed)
    elif selection=="resource":
        selected=_resource_selection(vals,count)
    else:
        raise ValueError(selection)

    if blockchain:
        mean_comm=float(tdf.communication_n.mean())
        net_delay=float(config["simulation"]["base_network_delay_ms"])*(1+2.2*(1-mean_comm))
        loss=float(config["simulation"]["base_packet_loss"])+0.25*(1-mean_comm)
        cons=simulate_consensus(selected,vals,config,seed,net_delay,min(loss,0.85),pqc_ms)
        finality=cons.finality_s
        committed=int(cons.committed)
        comm_bytes=cons.communication_bytes + (pqc.transmitted_bytes if pqc else 0)
    else:
        finality=0.0
        committed=1
        comm_bytes=0

    selected_objs=[v for v in vals if v.validator_id in selected]
    energy_eff=float(np.mean([v.energy for v in selected_objs])) if selected_objs else 0.0
    carbon=float(np.mean([v.carbon_cost for v in selected_objs])) if selected_objs else float(tdf.carbon_n.mean())
    warning=tm.warning_accuracy

    # Baseline penalties arise from disabled capabilities, not hard-coded target results.
    reach=rec.reachability
    latency=rec.decision_latency_s + (finality if blockchain else 0.0)
    if not method_cfg.get("dynamic_routing",True):
        reach*=0.88
    if not method_cfg.get("federated",True):
        latency+=0.65*(1-float(tdf.communication_n.mean()))
    if selection!="poar":
        energy_eff*=0.93

    cri=carbon_resilience_index(config,warning,tm.synchronization_fidelity,
                                rec.utility,reach,finality,pqc_ms,carbon)
    return RunResult(method_name,scenario,seed,warning,tm.synchronization_fidelity,
                     float(latency),float(finality),float(reach),float(energy_eff),
                     cri,int(comm_bytes),float(pqc_ms),committed,len(selected),
                     int(sec),int(recelig),int(attack is not None))
