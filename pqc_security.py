from __future__ import annotations
from dataclasses import dataclass
import time
import os
import numpy as np


@dataclass
class PQCResult:
    mode: str
    kem: str
    signature: str
    overhead_ms: float
    transmitted_bytes: int
    executed_real_crypto: bool


def benchmark_pqc(config: dict, seed: int = 0) -> PQCResult:
    cfg = config["pqc"]
    mode = str(cfg.get("mode","modeled")).lower()

    if mode == "oqs":
        try:
            import oqs  # optional external dependency
            t0 = time.perf_counter()
            with oqs.KeyEncapsulation(cfg["kem"]) as kem:
                pk = kem.generate_keypair()
                ct, shared = kem.encap_secret(pk)
                _ = kem.decap_secret(ct)
            with oqs.Signature(cfg["signature"]) as sig:
                pk2 = sig.generate_keypair()
                msg = b"recovery-command"
                signature = sig.sign(msg)
                ok = sig.verify(msg, signature, pk2)
                if not ok:
                    raise RuntimeError("PQC signature verification failed")
            elapsed = (time.perf_counter()-t0)*1000.0
            return PQCResult("oqs",cfg["kem"],cfg["signature"],elapsed,len(ct)+len(signature),True)
        except Exception:
            # Fall through to explicit modeled mode rather than failing the experiment.
            pass

    m = cfg["modeled_ms"]
    b = cfg["modeled_bytes"]
    # Four operations are modeled in the emergency confirmation path.
    overhead = float(m["encapsulate"]+m["decapsulate"]+m["sign"]+m["verify"])
    return PQCResult("modeled",cfg["kem"],cfg["signature"],overhead,
                     int(b["kem_ciphertext"]+b["signature"]),False)
