
from engine.model import (
    infer_faults,
    simulate_readings,
    choose_next_test,
    TESTS,
)


def test_robust_selector_never_recommends_high_risk():
    posterior = infer_faults(
        simulate_readings(("overload", "bearing"), noise=0.3, seed=42),
        noise=0.3,
    )

    # Check the existing selector's safety invariant.
    selected = choose_next_test(posterior, [])

    assert selected is None or selected["risk"] != "High"


def test_all_automatically_eligible_tests_have_known_costs():
    eligible = [
        test for test in TESTS
        if test["risk"] != "High"
    ]

    assert eligible
    assert all(test["cost"] > 0 for test in eligible)


def test_selector_does_not_repeat_completed_test():
    posterior = infer_faults(
        simulate_readings(("bearing",), noise=0.2, seed=7),
        noise=0.2,
    )

    completed = ["voltage_check"]
    selected = choose_next_test(posterior, completed)

    assert selected is None or selected["id"] not in completed