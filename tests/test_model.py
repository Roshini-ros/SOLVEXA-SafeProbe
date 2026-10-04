from engine.model import infer_faults, simulate_readings, choose_next_test, entropy, TESTS

def test_posterior_sums_to_one():
    obs = simulate_readings(("bearing",), noise=0.2, seed=2)
    posterior = infer_faults(obs, noise=0.2)
    assert abs(sum(posterior.values()) - 1.0) < 1e-9

def test_all_fault_combinations_are_present():
    posterior = infer_faults(simulate_readings((), noise=0.2), noise=0.2)
    assert len(posterior) == 8

def test_selector_avoids_high_risk_test():
    posterior = infer_faults(simulate_readings(("overload", "bearing"), noise=0.3), noise=0.3)
    selected = choose_next_test(posterior, [])
    assert selected is not None
    assert selected["risk"] != "High"

def test_entropy_nonnegative():
    assert entropy({"a": 0.5, "b": 0.5}) > 0

def test_high_risk_test_is_not_automatically_recommended():
    assert all(t["risk"] != "High" for t in TESTS if t["id"] != "loaded_run")
