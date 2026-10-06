from __future__ import annotations
import numpy as np


def spoof_sensors(df, fraction: float, seed: int):
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = int(round(len(out)*fraction))
    if n <= 0:
        return out
    idx = rng.choice(out.index.to_numpy(), min(n,len(out)), replace=False)
    out.loc[idx,"rainfall_n"] = np.clip(out.loc[idx,"rainfall_n"] + rng.normal(0.35,0.15,len(idx)),0,1)
    out.loc[idx,"flood_n"] = np.clip(out.loc[idx,"flood_n"] + rng.normal(0.30,0.12,len(idx)),0,1)
    return out


def communication_attack(df, partition_fraction: float, packet_loss_extra: float, seed: int):
    out = df.copy()
    rng = np.random.default_rng(seed)
    zones = out["zone_id"].unique()
    k = max(1,int(round(len(zones)*partition_fraction)))
    attacked = rng.choice(zones,min(k,len(zones)),replace=False)
    mask = out.zone_id.isin(attacked)
    out.loc[mask,"communication_n"] *= np.clip(1.0-packet_loss_extra-0.35,0.05,1.0)
    return out


def dos_validators(validators, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    k = int(round(len(validators)*fraction))
    for i in (rng.choice(len(validators),min(k,len(validators)),replace=False) if k else []):
        validators[int(i)].available = False
    return validators


def manipulate_poar(validators, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    k = int(round(len(validators)*fraction))
    for i in (rng.choice(len(validators),min(k,len(validators)),replace=False) if k else []):
        v = validators[int(i)]
        v.energy = 1.0
        v.communication = 1.0
        v.recovery = 1.0
        v.carbon_cost = 0.0
        v.malicious = True
        # Security stage can still exclude nodes if trust/authorization/freshness fails.
        v.trust = min(v.trust,0.45)
    return validators


def poison_policy(agent, scale: float, seed: int):
    rng = np.random.default_rng(seed)
    agent.W += rng.normal(0,scale,size=agent.W.shape)
    return agent
