from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import numpy as np


@dataclass
class Validator:
    validator_id: int
    authenticated: bool
    authorized: bool
    protocol_compliant: bool
    trust: float
    energy: float
    communication: float
    synchronization: float
    recovery: float
    carbon_cost: float
    last_report_step: int
    available: bool = True
    malicious: bool = False


@dataclass
class ScoredValidator:
    validator_id: int
    score: float
    security_eligible: bool
    recovery_eligible: bool
    attributes: dict


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


def security_eligible(v: Validator, cfg: dict, current_step: int) -> bool:
    th = cfg["thresholds"]
    fresh = (current_step - int(v.last_report_step)) <= int(cfg["freshness_steps"])
    return bool(
        v.available and v.authenticated and v.authorized and v.protocol_compliant and
        fresh and _clip01(v.trust) >= float(th["security_trust"])
    )


def poar_score(v: Validator, cfg: dict) -> float:
    w = cfg["weights"]
    # Higher carbon_cost is undesirable and is inverted after clipping.
    return _clip01(
        w["trust"]*_clip01(v.trust) +
        w["energy"]*_clip01(v.energy) +
        w["communication"]*_clip01(v.communication) +
        w["synchronization"]*_clip01(v.synchronization) +
        w["recovery"]*_clip01(v.recovery) +
        w["carbon_cost"]*(1.0-_clip01(v.carbon_cost))
    )


def rank_validators(validators: List[Validator], cfg: dict, current_step: int) -> List[ScoredValidator]:
    th = cfg["thresholds"]
    scored = []
    for v in validators:
        sec = security_eligible(v, cfg, current_step)
        score = poar_score(v, cfg) if sec else 0.0
        rec = bool(
            sec and v.energy >= th["energy"] and
            v.communication >= th["communication"] and
            score >= th["poar_score"]
        )
        scored.append(ScoredValidator(
            v.validator_id, score, sec, rec,
            {
                "trust":_clip01(v.trust), "energy":_clip01(v.energy),
                "communication":_clip01(v.communication),
                "synchronization":_clip01(v.synchronization),
                "recovery":_clip01(v.recovery),
                "carbon_cost":_clip01(v.carbon_cost)
            }
        ))
    return sorted(scored, key=lambda x: (-x.recovery_eligible, -x.score, x.validator_id))


def select_validators(validators: List[Validator], cfg: dict, current_step: int, count: int):
    ranked = rank_validators(validators, cfg, current_step)
    preferred = [x for x in ranked if x.recovery_eligible]
    if len(preferred) < int(cfg["min_security_eligible"]):
        if cfg.get("fallback") == "highest_security_eligible":
            preferred = [x for x in ranked if x.security_eligible]
    selected_ids = [x.validator_id for x in preferred[:count]]
    return selected_ids, ranked


def generate_validators(config: dict, df, seed: int, current_step: int):
    rng = np.random.default_rng(seed + current_step*13)
    n = int(config["simulation"]["validator_candidates"])
    mean_comm = float(df["communication_n"].mean())
    mean_energy = float(df["energy_n"].mean())
    sync = float(np.clip(1-df["sync_error"].mean(),0,1))
    mean_risk = float(df["escalation_probability"].mean())
    carbon = float(df["carbon_n"].mean())
    vals = []
    for i in range(n):
        vals.append(Validator(
            validator_id=i,
            authenticated=True,
            authorized=True,
            protocol_compliant=True,
            trust=float(np.clip(rng.normal(0.78,0.12),0,1)),
            energy=float(np.clip(mean_energy+rng.normal(0,0.10),0,1)),
            communication=float(np.clip(mean_comm+rng.normal(0,0.12),0,1)),
            synchronization=float(np.clip(sync+rng.normal(0,0.06),0,1)),
            recovery=float(np.clip(0.45+0.45*mean_risk+rng.normal(0,0.10),0,1)),
            carbon_cost=float(np.clip(carbon+rng.normal(0,0.08),0,1)),
            last_report_step=max(0,current_step-int(rng.integers(0,2))),
        ))
    return vals
