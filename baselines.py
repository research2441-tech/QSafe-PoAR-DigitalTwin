from __future__ import annotations
from copy import deepcopy


BASELINES = {
    "proposed": {},
    "centralized_monitoring": {"twin_correction":False,"federated":False,"blockchain":False,"dynamic_routing":False},
    "digital_twin_only": {"twin_correction":True,"federated":False,"blockchain":False,"dynamic_routing":False},
    "conventional_blockchain": {"selection":"round_robin","pqc":False,"federated":False},
    "classical_security_blockchain": {"pqc":False},
    "centralized_rl": {"federated":False},
    "energy_unaware": {"poar_zero":"energy"},
    "carbon_unaware": {"poar_zero":"carbon_cost"},
    "static_routing": {"dynamic_routing":False},
    "round_robin": {"selection":"round_robin"},
    "random_eligible": {"selection":"random"},
    "resource_aware": {"selection":"resource"},
    "pbft_fixed_membership": {"selection":"round_robin"},
}


def get_baseline(name: str) -> dict:
    if name not in BASELINES:
        raise KeyError(name)
    return deepcopy(BASELINES[name])


def configured_names(include_extended: bool = True):
    if include_extended:
        return list(BASELINES.keys())
    return [
        "proposed","centralized_monitoring","digital_twin_only",
        "conventional_blockchain","classical_security_blockchain",
        "centralized_rl","energy_unaware","carbon_unaware","static_routing"
    ]
