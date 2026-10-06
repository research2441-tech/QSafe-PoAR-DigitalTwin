from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ConsensusResult:
    committed: bool
    finality_s: float
    quorum_required: int
    votes_received: int
    byzantine_count: int
    messages: int
    communication_bytes: int
    view_changes: int


def pbft_quorum(n: int) -> int:
    if n <= 0:
        return 0
    f = (n-1)//3
    return 2*f + 1


def simulate_consensus(selected_ids, validators, config: dict, seed: int,
                       network_delay_ms: float, packet_loss: float,
                       pqc_overhead_ms: float = 0.0,
                       force_byzantine_fraction: float | None = None) -> ConsensusResult:
    rng = np.random.default_rng(seed)
    n = len(selected_ids)
    if n == 0:
        return ConsensusResult(False, float("inf"), 0, 0, 0, 0, 0, 0)

    c = config["consensus"]
    quorum = pbft_quorum(n)
    valmap = {v.validator_id:v for v in validators}
    byz_fraction = float(c["byzantine_fraction"] if force_byzantine_fraction is None else force_byzantine_fraction)
    byz_target = min(n, int(round(byz_fraction*n)))
    byz_ids = set(rng.choice(selected_ids, byz_target, replace=False).tolist()) if byz_target else set()

    responsive = []
    for vid in selected_ids:
        v = valmap[vid]
        p_success = np.clip((1-packet_loss)*v.communication, 0, 1)
        if vid in byz_ids:
            p_success *= 0.20
        if rng.random() < p_success:
            responsive.append(vid)

    votes = len(responsive)
    committed = votes >= quorum
    view_changes = 0 if committed else 1

    # PBFT-style message accounting over the selected validator set.
    messages = n + n*max(n-1,0) + n*max(n-1,0)
    meta = int(c["score_metadata_bytes_per_validator"])*n
    tx = int(c["transaction_bytes"])
    communication_bytes = messages*96 + meta + tx

    jitter = float(rng.lognormal(mean=0.0, sigma=0.20))
    processing_ms = float(c["base_processing_ms"])*n
    round_factor = 1.0 if committed else 1.8
    finality_ms = round_factor*(2.4*network_delay_ms*jitter + processing_ms + pqc_overhead_ms)
    return ConsensusResult(
        committed, float(finality_ms/1000.0), quorum, votes, byz_target,
        int(messages), int(communication_bytes), view_changes
    )


def simulate_round_robin(candidate_ids, count, offset=0):
    if not candidate_ids:
        return []
    ordered = sorted(candidate_ids)
    return [ordered[(offset+i)%len(ordered)] for i in range(min(count,len(ordered)))]


def simulate_random_selection(candidate_ids, count, seed):
    rng = np.random.default_rng(seed)
    if not candidate_ids:
        return []
    return rng.choice(candidate_ids, min(count,len(candidate_ids)), replace=False).tolist()
