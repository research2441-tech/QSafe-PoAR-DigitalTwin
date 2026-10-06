# Reproducibility Protocol

## 1. Evidence boundary

This repository implements a simulation-based evaluation informed by public environmental and geospatial data sources. Operational edge-IoT, communication, energy, validator, and emergency-resource variables are scenario generated unless an external measured record is explicitly supplied.

Simulation evidence must not be described as field validation.

## 2. Experimental unit

The independent simulation run is the replication unit. Repeated time steps, sensor samples, transactions, and zone observations inside one run are not treated as statistically independent replications.

The default configuration uses ten deterministic seeds:

`101, 211, 307, 401, 503, 601, 701, 809, 907, 1009`

## 3. Public-data provenance

The manuscript framework refers to:

- NASA GPM IMERG precipitation;
- NOAA IBTrACS tropical-cyclone tracks;
- USGS earthquake records;
- OpenStreetMap road topology through OSMnx;
- SRTM or Copernicus DEM;
- GHSL or WorldPop exposure data.

A reproducible manuscript submission should report the exact version, access date, spatial range, temporal range, coordinate system, selected variables, inclusion/exclusion criteria, and preprocessing for every external source. `data_manifest.yaml` provides the required provenance fields.

The default source package does not silently invent missing download dates or dataset versions.

## 4. Synthetic operational variables

Scenario-generated variables include communication stability, gateway availability, residual energy, packet loss, communication delay, microgrid availability, emergency-resource availability, traffic pressure, selected flood-depth values, validator recovery contribution, and carbon conditions.

All generated values are bounded and deterministic for a fixed random seed.

## 5. Spatial and temporal alignment

The simulation uses zone-level state vectors and a default 30-minute update interval. Every state transition is associated with a zone identifier and simulation step.

Normalization is explicit and uses bounded min-max transformations or native [0,1] variables.

## 6. Digital-twin synchronization

The digital twin combines the latest normalized observation with the previous synchronized state. Under degraded communication, the effective observation weight is reduced so that delayed or noisy observations do not cause abrupt state discontinuities.

Synchronization fidelity is reported as:

`1 - mean absolute synchronization error`

with clipping to `[0,1]`.

Warning labels and warning predictions are computed separately so that confusion-matrix statistics can be reproduced.

## 7. Agentic recovery interpretation

The term “agentic” is operationalized as a closed feedback loop:

state observation → policy-based action selection → simulated feedback → parameter update.

It does not imply unrestricted autonomous authority over a real emergency system.

The lightweight policy model is provided so that the artifact can be executed without GPU-specific ML frameworks. Federated mode trains local zone policies and aggregates their parameters by weighted averaging.

## 8. PoAR validator selection

PoAR consists of two stages.

### Stage 1: security eligibility

A validator must satisfy:

- availability;
- authenticated identity;
- authorization;
- protocol compliance;
- report freshness;
- minimum security-trust threshold.

Operational suitability does not establish Byzantine trustworthiness.

### Stage 2: recovery-aware ranking

Only Stage-1 candidates are scored using:

- trust;
- residual energy;
- communication stability;
- digital-twin synchronization fidelity;
- predicted recovery contribution;
- carbon-aware validation cost.

Carbon cost is a negative preference and is therefore inverted in the weighted score.

The final validator set must also satisfy energy, communication, and minimum PoAR score thresholds. If too few candidates satisfy recovery eligibility, the documented fallback chooses the highest-ranked security-eligible nodes.

## 9. Consensus layer

PoAR is not treated as an independent consensus protocol.

The repository uses a PBFT-style simulator to model:

- proposal dissemination;
- validator voting;
- quorum;
- commit;
- packet loss;
- Byzantine non-cooperation;
- timeouts;
- view change;
- communication overhead.

For `n` selected validators, the quorum is derived from the classical `f=(n-1)//3`, `q=2f+1` rule.

The simulator is an experimental abstraction, not a formally verified production consensus implementation.

## 10. Post-quantum cryptography

Two execution modes are supported:

- actual liboqs-backed execution where available;
- declared deterministic overhead modeling otherwise.

When modeled mode is used, the paper must describe it as simulation of cryptographic overhead, not implementation-level proof of quantum-safe security.

## 11. Baseline fairness

All methods use the same scenario inputs, seeds, and evaluation opportunities where their design permits.

Conceptual reference categories are separated from executable comparisons.

## 12. Controlled ablation

Ablation variants modify one principal component while keeping the common implementation unchanged. Implemented variants include:

- no twin correction;
- centralized rather than federated recovery;
- no PQC overhead;
- no energy awareness;
- no carbon awareness;
- no dynamic routing;
- round-robin rather than PoAR validator selection.

The code also performs a PoAR sensitivity sweep over score threshold, communication threshold, and carbon weight.

## 13. Adversarial evaluation

Implemented stressors include:

- sensor spoofing;
- communication partition;
- model poisoning;
- validator denial-of-service;
- manipulation of PoAR operational attributes.

Security conclusions must be restricted to these implemented attack models.

## 14. Statistical analysis

For each method, scenario, and metric, the repository exports:

- sample size;
- mean;
- sample standard deviation;
- 95% confidence interval.

Matched scenario/seed comparisons against the proposed method use paired t-tests and paired Cohen effect size when the pairing assumptions are satisfied. Holm adjustment is applied across multiple baseline comparisons.

## 15. Main command

```bash
python reproduce.py
```

Quick deterministic verification:

```bash
python reproduce.py --quick
python test_framework.py
```

## 16. Result traceability

The following runtime outputs are generated:

- `run_level_results.csv`
- `summary_mean_sd_ci.csv`
- `stats_<metric>.csv`
- `ablation_run_level.csv`
- `poar_sensitivity_run_level.csv`
- `run_metadata.json`

Any manuscript table should be derived from these auditable outputs instead of manually entered target numbers.
