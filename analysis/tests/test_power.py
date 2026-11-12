"""Block 7: thresholds, bounds and the handling of the realized number of correct starts."""
import numpy as np
import pytest

from analysis import power as W
from analysis.binomial import (boole_lower_bound, confirm_threshold, cp_lower, cp_upper, exceed_threshold,
                               h1_decision, independent_product, power_confirm, power_exceed)


def nominal(spec):
    return W.PowerInput(spec, {D: spec["n_nominal"] for D in W.DENSITIES})


def test_thresholds_of_the_plan_reproduce():
    assert confirm_threshold(1000) == 20          # H1a: confirmed with <= 20 events of 1000
    assert exceed_threshold(1500) == 57           # H1b: exceedance with >= 57 events of 1500
    assert confirm_threshold(1500) == 33          # H1b: bound confirmed with <= 33 events of 1500
    assert (confirm_threshold(500), exceed_threshold(500)) == (8, 23)
    assert cp_upper(20, 1000) <= 0.03 < cp_upper(21, 1000)
    assert cp_lower(57, 1500) > 0.03 >= cp_lower(56, 1500)
    assert [h1_decision(x, 1500) for x in (33, 34, 56, 57)] == ["confirmed", "insufficient", "insufficient", "exceeded"]
    assert cp_upper(0, 1000) == pytest.approx(0.0030, abs=1e-4) and cp_upper(0, 1500) == pytest.approx(0.0020, abs=1e-4)


def test_joint_power_is_a_bound_and_the_product_is_only_a_naive_approximation():
    a, b = W.hypothesis_power(nominal(W.H1A)), W.hypothesis_power(nominal(W.H1B))
    assert power_confirm(1000, 0.01) == pytest.approx(0.9985, abs=1e-4)
    assert power_exceed(1500, 0.05) == pytest.approx(0.988, abs=1e-3)
    assert (a["checks"], b["checks"]) == (60, 15)
    assert a["bound"] == pytest.approx(0.910, abs=2e-3) and b["bound"] == pytest.approx(0.825, abs=2e-3)
    assert a["product"] == pytest.approx(0.914, abs=2e-3) and b["product"] == pytest.approx(0.839, abs=2e-3)
    assert a["bound"] < a["product"] and b["bound"] < b["product"]      # the bound assumes less
    assert a["bound"] >= W.TARGET and b["bound"] >= W.TARGET            # 0.8 per hypothesis holds
    # simultaneous confirmation is a different, separately parameterised condition
    assert W.simultaneous_bound(a, b) == pytest.approx(a["bound"] + b["bound"] - 1.0)
    assert W.simultaneous_bound(a, b) < W.TARGET
    assert boole_lower_bound([0.5, 0.4]) == 0.0 and independent_product([0.5, 0.4]) == pytest.approx(0.2)


def test_power_uses_the_realized_correct_start_count_not_the_nominal_budget():
    real = W.PowerInput(W.H1A, {10: 1000, 25: 985, 40: 940}, "realized")
    r = W.hypothesis_power(real)
    assert r["thresholds"] == {10: 20, 25: confirm_threshold(985), 40: confirm_threshold(940)}
    assert r["thresholds"][40] < 20
    assert r["single"] == {D: power_confirm(n, 0.01) for D, n in real.n_valid.items()}
    assert r["bound"] != W.hypothesis_power(nominal(W.H1A))["bound"]
    assert r["n_valid"] == {10: 1000, 25: 985, 40: 940} and r["source"] == "realized"


@pytest.mark.parametrize("n_valid", [{10: 1001, 25: 1000, 40: 1000},      # more than the design
                                     {10: 1000, 25: 1000},                # a density is missing
                                     {10: 1000, 25: 999.5, 40: 1000},     # not a count
                                     {10: -1, 25: 1000, 40: 1000}])
def test_inconsistent_correct_start_counts_fail_the_computation(n_valid):
    with pytest.raises(W.DesignError):
        W.hypothesis_power(W.PowerInput(W.H1A, n_valid))
    with pytest.raises(W.DesignError):
        W.simulate_joint_power(W.PowerInput(W.H1A, n_valid), 10, np.random.default_rng(0))


def test_shared_seed_consistency_between_h1a_and_h1b():
    a = W.PowerInput(W.H1A, {10: 990, 25: 1000, 40: 1000})
    W.check_shared_seeds(a, W.PowerInput(W.H1B, {10: 1480, 25: 1500, 40: 1500}))
    with pytest.raises(W.DesignError):                    # fewer correct starts among 1500 than among the first 1000
        W.check_shared_seeds(a, W.PowerInput(W.H1B, {10: 980, 25: 1500, 40: 1500}))
    with pytest.raises(W.DesignError):                    # more than 500 additional starts
        W.check_shared_seeds(W.PowerInput(W.H1A, {10: 900, 25: 1000, 40: 1000}),
                             W.PowerInput(W.H1B, {10: 1500, 25: 1500, 40: 1500}))


def test_simulator_draws_only_correct_start_episodes():
    rng = np.random.default_rng(3)
    small = W.PowerInput(W.H1B, {10: 300, 25: 300, 40: 300})
    # with 300 correct starts the exceedance threshold is far lower than 57
    assert W.threshold(W.H1B, 300) == exceed_threshold(300) < 20
    sim = W.simulate_joint_power(small, 300, rng, icc=0.0)
    assert sim == pytest.approx(W.hypothesis_power(small)["product"], abs=0.08)
    assert sim >= W.hypothesis_power(small)["bound"] - 0.08


def test_report_states_target_count_bound_and_assumption(capsys, tmp_path):
    starts = tmp_path / "start_states.csv"
    starts.write_text("D,unsafe_b_1.0\n" + "".join(f"{D},{int(i == 0 and D == 40)}\n"
                                                    for D in (10, 25, 40) for i in range(10)))
    assert W.read_start_counts(starts) == {10: (10, 10), 25: (10, 10), 40: (10, 9)}
    assert W.main(["--n-sim", "50", "--starts", str(starts), "--out", str(tmp_path / "power.md")]) == 0
    text = capsys.readouterr().out
    for name in ("## H1a", "## H1b"):
        assert name in text
    for needle in ("- target: 0.8", "realized correct-start episodes per check", "joint power, lower bound",
                   "sampling assumption: none about the dependence", "naive independent approximation",
                   "Simultaneous confirmation of H1a and H1b (a separate condition)",
                   "D = 40: 900", "D = 40: 1350"):
        assert needle in text, needle
    assert (tmp_path / "power.md").read_text() == text
