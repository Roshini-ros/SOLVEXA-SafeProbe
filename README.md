# SOLVEXA — SafeProbe

**A transparent, software-only prototype for choosing informative diagnostic tests when industrial motor faults may overlap.**

SafeProbe models three candidate fault types:
- mechanical overload
- supply-voltage abnormality
- bearing friction

It represents all eight combinations (including a healthy state), maintains a probability distribution over those hypotheses, and recommends a low-risk virtual test using an information-gain heuristic and illustrative cost values.

## Important scope and safety note

This repository is an educational simulation. All telemetry and test outcomes are synthetic; the motor model and likelihood assumptions are illustrative, not field-calibrated. It is not a certified diagnostic system and must not be used to operate, repair, or test real machinery. Physical testing requires qualified personnel and appropriate procedures.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Run tests

```bash
pytest -q
```

## Repository layout

- `app.py` — interactive Streamlit dashboard
- `engine/model.py` — synthetic motor model, Bayesian-style hypothesis update, and test-selection heuristic
- `tests/` — basic reproducibility and safety-gate checks

## Method overview

1. Generate synthetic telemetry from a known scenario for repeatable evaluation.
2. Score each of the eight fault combinations against observed sensor values under a simple Gaussian likelihood model.
3. Normalize scores into a posterior distribution.
4. Estimate the information gain of remaining candidate tests using a discretized, model-based approximation.
5. Exclude high-risk tests from automatic recommendations.

## Current limitations

- The model assumes additive fault effects, which is a simplification; real faults can interact nonlinearly.
- Sensor errors are approximated as independent Gaussian noise.
- Priors are uniform and not learned from field data.
- The test selector uses a simplified information-gain estimate, not a validated industrial diagnostic planner.
- The scenario selector reveals the hidden fault to the demo operator; hide or replace it with a blind evaluation interface for a judge-facing demo.
- Probability values are model-relative, not real-world calibrated confidence.

## Planned next steps

- Add blind scenario evaluation and seeded test runs.
- Compare SafeProbe against random and fixed-checklist baselines.
- Report test count, accumulated cost, fault-set exact-match rate, and unresolved-case rate.
- Add plots and a reproducible experiment report.


## Current demo flow

The initial observation set contains current and temperature. Virtual tests add supply voltage, vibration, or phase-imbalance measurements. A higher-risk load test is intentionally excluded from automatic recommendations.
