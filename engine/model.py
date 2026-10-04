
"""SafeProbe diagnostic engine.
Synthetic demonstration only — not validated for real machinery.
"""
from itertools import product
import math
import random

FAULTS = ("overload", "voltage", "bearing")

BASELINE = {
    "current_a": 5.0,
    "temperature_c": 55.0,
    "vibration_mm_s": 1.8,
    "voltage_v": 230.0,
    "phase_imbalance_pct": 0.8,
}

EFFECTS = {
    "overload": {
        "current_a": 2.1, "temperature_c": 7.0,
        "vibration_mm_s": 0.5, "voltage_v": 0.0,
        "phase_imbalance_pct": 1.0,
    },
    "voltage": {
        "current_a": 1.0, "temperature_c": 3.0,
        "vibration_mm_s": 0.1, "voltage_v": -18.0,
        "phase_imbalance_pct": 7.0,
    },
    "bearing": {
        "current_a": 0.8, "temperature_c": 9.0,
        "vibration_mm_s": 3.4, "voltage_v": 0.0,
        "phase_imbalance_pct": 0.3,
    },
}

SIGMA = {
    "current_a": 1.0,
    "temperature_c": 4.0,
    "vibration_mm_s": 1.0,
    "voltage_v": 7.0,
    "phase_imbalance_pct": 2.0,
}

TESTS = [
    {
        "id": "voltage_check",
        "name": "Measure supply voltage",
        "observation_key": "voltage_v",
        "unit": "V", "cost": 1, "risk": "Low",
        "why": "Provides direct evidence about a possible supply-voltage abnormality.",
    },
    {
        "id": "vibration_check",
        "name": "Measure vibration",
        "observation_key": "vibration_mm_s",
        "unit": "mm/s", "cost": 2, "risk": "Low",
        "why": "Provides evidence that may help distinguish bearing friction.",
    },
    {
        "id": "phase_check",
        "name": "Measure phase imbalance",
        "observation_key": "phase_imbalance_pct",
        "unit": "%", "cost": 2, "risk": "Low",
        "why": "Adds an electrical clue to help distinguish fault hypotheses.",
    },
    {
        "id": "loaded_run",
        "name": "Controlled load test (virtual only)",
        "observation_key": "load_response",
        "unit": "index", "cost": 3, "risk": "High",
        "why": "Excluded from automatic recommendations because of its illustrative high-risk classification.",
    },
]


def all_states():
    return [
        tuple(f for f, active in zip(FAULTS, bits) if active)
        for bits in product((0, 1), repeat=len(FAULTS))
    ]


def state_label(state):
    if not state:
        return "Healthy"
    names = {
        "overload": "Overload",
        "voltage": "Voltage abnormality",
        "bearing": "Bearing friction",
    }
    return " + ".join(names[f] for f in state)


def state_from_label(label):
    names = {
        "Overload": "overload",
        "Voltage abnormality": "voltage",
        "Bearing friction": "bearing",
    }
    if label == "Healthy":
        return ()
    return tuple(names[p.strip()] for p in label.split(" + "))


def expected_signals(faults):
    result = dict(BASELINE)
    for fault in faults:
        for key, effect in EFFECTS[fault].items():
            result[key] += effect
    return result


def expected_test_value(test, faults):
    if test["observation_key"] == "load_response":
        return (
            1.0
            + (0.9 if "overload" in faults else 0.0)
            + (0.35 if "bearing" in faults else 0.0)
        )
    return expected_signals(faults)[test["observation_key"]]


def simulate_readings(faults=(), noise=0.25, seed=42, keys=None):
    rng = random.Random(seed)
    expected = expected_signals(faults)
    keys = keys or ("current_a", "temperature_c")
    return {
        key: rng.gauss(
            expected[key],
            SIGMA[key] * max(noise, 0.08),
        )
        for key in keys
    }


def simulate_test(test, faults=(), noise=0.25, seed=42):
    rng = random.Random(seed)
    mean = expected_test_value(test, faults)
    key = test["observation_key"]
    spread = (
        0.15 if key == "load_response" else SIGMA[key]
    ) * max(noise, 0.08)
    return rng.gauss(mean, spread)


for _test in TESTS:
    _test["simulate"] = (
        lambda state, noise=0.25, seed=42, test=_test:
        simulate_test(test, state, noise, seed)
    )


def log_likelihood(value, mean, sigma):
    sigma = max(sigma, 1e-6)
    z = (value - mean) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2 * math.pi))


def infer_faults(observations, noise=0.25):
    """Estimate probabilities across all eight fault combinations."""
    states = all_states()
    scores = []

    for state in states:
        expected = expected_signals(state)
        score = -len(FAULTS) * math.log(2)

        for key, value in observations.items():
            if key not in expected:
                continue
            sigma = SIGMA[key] * max(noise, 0.08)
            score += log_likelihood(value, expected[key], sigma)

        scores.append(score)

    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)

    return {
        state_label(state): weight / total
        for state, weight in zip(states, weights)
    }


def entropy(distribution):
    """Shannon entropy in bits."""
    return -sum(
        p * math.log(p, 2)
        for p in distribution.values()
        if p > 0
    )


def posterior_after_test(posterior, test, outcome, noise):
    """Bayesian update for one hypothetical test outcome."""
    updated = {}
    key = test["observation_key"]
    sigma = (
        0.15 if key == "load_response" else SIGMA[key]
    ) * max(noise, 0.08)

    for label, prior in posterior.items():
        state = state_from_label(label)
        mean = expected_test_value(test, state)
        updated[label] = math.log(max(prior, 1e-300)) + log_likelihood(
            outcome, mean, sigma
        )

    peak = max(updated.values())
    weights = {
        label: math.exp(value - peak)
        for label, value in updated.items()
    }
    total = sum(weights.values())
    return {label: value / total for label, value in weights.items()}


def estimate_information_gain(posterior, test, noise=0.25):
    """Approximate expected information gain by deterministic outcome sampling."""
    current_entropy = entropy(posterior)
    key = test["observation_key"]
    sigma = (
        0.15 if key == "load_response" else SIGMA[key]
    ) * max(noise, 0.08)

    # Use a fixed set of standard-normal samples for reproducibility.
    z_samples = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    expected_entropy = 0.0

    for label, probability in posterior.items():
        if probability <= 0:
            continue

        state = state_from_label(label)
        mean = expected_test_value(test, state)

        for z in z_samples:
            outcome = mean + z * sigma
            updated = posterior_after_test(
                posterior, test, outcome, noise
            )
            expected_entropy += (
                probability / len(z_samples)
            ) * entropy(updated)

    return max(0.0, current_entropy - expected_entropy)


def choose_next_test(posterior, completed_ids=(), noise=0.35):
    """Choose an eligible test by information gain per cost unit."""
    eligible = [
        test for test in TESTS
        if test["id"] not in completed_ids
        and test["risk"] != "High"
    ]

    if not eligible:
        return None

    scored = []
    for test in eligible:
        gain = estimate_information_gain(
            posterior, test, noise=noise
        )
        score = gain / max(test["cost"], 1)

        result = dict(test)
        result["gain"] = gain
        result["score"] = score
        scored.append(result)

    scored.sort(
        key=lambda test: (test["score"], test["gain"]),
        reverse=True,
    )
    return scored[0]