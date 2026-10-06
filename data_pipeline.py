from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def minmax_clip(x, lo, hi):
    x = np.asarray(x, dtype=float)
    if hi <= lo:
        raise ValueError("invalid normalization range")
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class ZoneFrame:
    zone_id: int
    step: int
    rainfall: float
    flood: float
    seismic: float
    cyclone: float
    road_mobility: float
    communication: float
    residual_energy: float
    exposure: float
    resources: float
    carbon_intensity: float


def _scenario_profile(scenario_id: str, t: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(t)
    zeros = np.zeros(n)
    bell = np.exp(-((t - 0.60) / 0.22) ** 2)
    late = 1.0 / (1.0 + np.exp(-(t - 0.52) * 15.0))
    prof = {
        "rain": zeros.copy(), "cyclone": zeros.copy(), "seismic": zeros.copy(),
        "road_loss": zeros.copy(), "energy_loss": zeros.copy(),
        "comm_loss": zeros.copy(), "carbon": zeros.copy()
    }
    if scenario_id == "S1":
        prof["rain"] = 0.85 * bell
        prof["road_loss"] = 0.35 * late
    elif scenario_id == "S2":
        prof["rain"] = 0.65 * bell
        prof["cyclone"] = 0.75 * bell
        prof["comm_loss"] = 0.25 * late
    elif scenario_id == "S3":
        prof["seismic"] = np.clip(0.90 * np.exp(-((t - 0.52) / 0.08) ** 2), 0, 1)
        prof["road_loss"] = 0.35 * late
    elif scenario_id == "S4":
        prof["road_loss"] = 0.72 * late
    elif scenario_id == "S5":
        prof["energy_loss"] = 0.70 * late
        prof["comm_loss"] = 0.18 * late
    elif scenario_id == "S6":
        prof["comm_loss"] = 0.78 * late
    elif scenario_id == "S7":
        prof["carbon"] = 0.80 * late
        prof["energy_loss"] = 0.25 * late
    elif scenario_id == "S8":
        prof["rain"] = 0.90 * bell
        prof["cyclone"] = 0.65 * bell
        prof["seismic"] = 0.25 * np.exp(-((t - 0.70) / 0.10) ** 2)
        prof["road_loss"] = 0.72 * late
        prof["energy_loss"] = 0.55 * late
        prof["comm_loss"] = 0.60 * late
        prof["carbon"] = 0.45 * late
    else:
        raise ValueError(f"unknown scenario: {scenario_id}")
    return prof


def generate_scenario_inputs(config: dict, scenario_id: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim = config["simulation"]
    ranges = config["synthetic_ranges"]
    zones = int(sim["zones"])
    steps = int(sim["duration_steps"])
    t = np.linspace(0.0, 1.0, steps)
    prof = _scenario_profile(scenario_id, t)

    rows = []
    for z in range(zones):
        vulnerability = np.clip(rng.normal(0.52, 0.15), 0.15, 0.92)
        elevation_suscept = np.clip(rng.normal(0.50, 0.18), 0.05, 0.95)
        exposure = np.clip(rng.beta(2.2, 1.8), 0.05, 0.98)
        for k in range(steps):
            noise = rng.normal(0.0, 0.025, 7)
            rainfall_n = np.clip(0.05 + prof["rain"][k] * (0.75 + 0.35*vulnerability) + noise[0], 0, 1)
            cyclone_n = np.clip(prof["cyclone"][k] * (0.70 + 0.30*vulnerability) + noise[1], 0, 1)
            seismic_n = np.clip(prof["seismic"][k] * rng.uniform(0.65, 1.0) + noise[2], 0, 1)
            flood_n = np.clip(
                0.05 + 0.52*rainfall_n + 0.18*cyclone_n + 0.22*elevation_suscept +
                0.18*prof["road_loss"][k] + noise[3], 0, 1
            )
            road = np.clip(1.0 - 0.58*flood_n - prof["road_loss"][k] + noise[4], 0, 1)
            communication = np.clip(0.96 - prof["comm_loss"][k] - 0.14*flood_n + noise[5], 0, 1)
            energy = np.clip(0.92 - prof["energy_loss"][k] - 0.10*flood_n + noise[6], 0, 1)
            resources = np.clip(0.92 - 0.38*flood_n - 0.20*cyclone_n - 0.18*(1-road), 0, 1)
            ci_lo, ci_hi = ranges["carbon_intensity_gco2_kwh"]
            carbon_n = np.clip(0.28 + prof["carbon"][k] + rng.normal(0, 0.04), 0, 1)
            carbon = ci_lo + carbon_n * (ci_hi-ci_lo)
            rain_lo, rain_hi = ranges["rainfall_mm_h"]
            flood_lo, flood_hi = ranges["flood_depth_m"]
            rows.append(ZoneFrame(
                z, k,
                rain_lo + rainfall_n*(rain_hi-rain_lo),
                flood_lo + flood_n*(flood_hi-flood_lo),
                seismic_n, cyclone_n, road, communication, energy,
                exposure, resources, carbon
            ).__dict__)
    return pd.DataFrame(rows)


def normalize_zone_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    r = config["synthetic_ranges"]
    out["rainfall_n"] = minmax_clip(out["rainfall"], *r["rainfall_mm_h"])
    out["flood_n"] = minmax_clip(out["flood"], *r["flood_depth_m"])
    out["seismic_n"] = np.clip(out["seismic"], 0, 1)
    out["cyclone_n"] = np.clip(out["cyclone"], 0, 1)
    out["road_mobility_n"] = np.clip(out["road_mobility"], 0, 1)
    out["communication_n"] = np.clip(out["communication"], 0, 1)
    out["energy_n"] = np.clip(out["residual_energy"], 0, 1)
    out["exposure_n"] = np.clip(out["exposure"], 0, 1)
    out["resources_n"] = np.clip(out["resources"], 0, 1)
    out["carbon_n"] = minmax_clip(out["carbon_intensity"], *r["carbon_intensity_gco2_kwh"])
    return out


def prepare_inputs(config: dict, scenario_id: str, seed: int) -> pd.DataFrame:
    return normalize_zone_frame(generate_scenario_inputs(config, scenario_id, seed), config)
