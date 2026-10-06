from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class TwinMetrics:
    synchronization_fidelity: float
    mean_sync_error: float
    warning_accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int


class DigitalTwin:
    def __init__(self, config: dict, correction: bool = True):
        self.cfg = config["digital_twin"]
        self.correction = correction

    def run(self, df: pd.DataFrame, warning_threshold: float) -> tuple[pd.DataFrame, TwinMetrics]:
        out = df.copy().sort_values(["zone_id", "step"]).reset_index(drop=True)
        alpha = float(self.cfg["observation_weight"])
        hw = self.cfg["hazard_weights"]
        ec = self.cfg["escalation_coefficients"]
        twin_hazard, sync_err, escalation = [], [], []
        last_by_zone = {}

        for row in out.itertuples(index=False):
            obs_vec = np.array([
                row.rainfall_n, row.flood_n, row.seismic_n, row.cyclone_n,
                1-row.road_mobility_n, 1-row.communication_n,
                1-row.energy_n, row.exposure_n, 1-row.resources_n
            ], dtype=float)
            prev = last_by_zone.get(row.zone_id, obs_vec.copy())
            if self.correction:
                stale = max(0.0, 1.0-row.communication_n)
                a_eff = np.clip(alpha * (1.0-0.35*stale), 0.25, 0.90)
                twin = a_eff*obs_vec + (1-a_eff)*prev
            else:
                twin = obs_vec.copy()
            last_by_zone[row.zone_id] = twin

            err = float(np.mean(np.abs(twin-obs_vec)))
            infrastructure = np.clip(np.mean([1-twin[4], 1-twin[5], 1-twin[6], 1-twin[8]]), 0, 1)
            hazard = (
                hw["rainfall"]*twin[0] + hw["flood"]*twin[1] +
                hw["seismic"]*twin[2] + hw["cyclone"]*twin[3] +
                hw["infrastructure"]*infrastructure
            )
            logit = (
                ec["bias"] + ec["hazard"]*hazard +
                ec["exposure"]*twin[7] +
                ec["mobility_loss"]*(1-row.road_mobility_n) +
                ec["communication_loss"]*(1-row.communication_n) +
                ec["resource_shortage"]*(1-row.resources_n)
            )
            twin_hazard.append(float(np.clip(hazard,0,1)))
            sync_err.append(err)
            escalation.append(float(sigmoid(logit)))

        out["twin_hazard"] = twin_hazard
        out["sync_error"] = sync_err
        out["escalation_probability"] = escalation
        out["pred_warning"] = (out["escalation_probability"] >= warning_threshold).astype(int)

        # Reference "unsafe" state is based only on latent physical/scenario state.
        physical_risk = (
            0.35*out["rainfall_n"] + 0.35*out["flood_n"] + 0.10*out["seismic_n"] +
            0.10*out["cyclone_n"] + 0.10*(1-out["road_mobility_n"])
        )
        out["true_warning"] = (physical_risk >= 0.52).astype(int)
        tp = int(((out.pred_warning==1)&(out.true_warning==1)).sum())
        tn = int(((out.pred_warning==0)&(out.true_warning==0)).sum())
        fp = int(((out.pred_warning==1)&(out.true_warning==0)).sum())
        fn = int(((out.pred_warning==0)&(out.true_warning==1)).sum())
        accuracy = (tp+tn)/max(1,tp+tn+fp+fn)
        mean_err = float(out["sync_error"].mean())
        fidelity = float(np.clip(1.0-mean_err,0,1))
        return out, TwinMetrics(fidelity, mean_err, accuracy, tp, tn, fp, fn)
