"""A1 (acceleration limits by kinematics, R2.2) and S1-S3 (standstill and release, R2.1)."""
import numpy as np
import pytest
from rss_shield.params import A_MAX_ACCEL, A_MIN_BRAKE, A_MAX_BRAKE, DT, K_TAU
from rss_shield.scenarios import scenario_random, scenario_stop, scenario_release, scenario1A

TOL = 1e-9


def _check_limits(rows):
    for row in rows:
        a, v0, v1 = row["a"], row["v_start"], row["v_end"]
        assert abs(row["a_integrated"] - a) < TOL, row                  # kinematics = applied a
        assert v1 >= -TOL, row                                           # no reversing
        assert -A_MAX_BRAKE - TOL <= a <= A_MAX_ACCEL + TOL, row
        if row["brake_mode"]:
            stopping = v1 <= TOL and a == pytest.approx(-v0 / DT)
            assert a <= -A_MIN_BRAKE + TOL or stopping, row


@pytest.mark.parametrize("seed", range(5))
def test_A1_random_command_stress(seed):
    r = scenario_random(seed)
    _check_limits(r.rows)
    a_ctrl = np.array([row["a_ctrl"] for row in r.rows])
    assert a_ctrl.max() > A_MAX_ACCEL and a_ctrl.min() < -A_MAX_BRAKE    # limiter was exercised


@pytest.mark.parametrize("rho", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("phase", ["sync", "worst"])
def test_A1_limits_in_table2_runs(rho, phase):
    r = scenario1A(rho, phase)
    rows = [row for row in r.rows if not row["crashed"]]     # after contact highway-env overrides a
    _check_limits(rows)


def test_S1_braking_to_standstill():
    r = scenario_stop(v0=10.0)
    _check_limits(r.rows)
    v = [row["v_end"] for row in r.rows]
    assert min(v) == 0.0
    k0 = next(i for i, x in enumerate(v) if x == 0.0)
    a = [row["a"] for row in r.rows]
    assert all(x == pytest.approx(-A_MIN_BRAKE) for x in a[K_TAU:k0])   # weakest admissible braking
    assert a[k0] >= -A_MIN_BRAKE - TOL                                    # last frame: a = -v/dt


def test_S2_holding_standstill_under_takeover():
    r = scenario_stop(v0=10.0)
    stopped = [row for row in r.rows if row["v_start"] == 0.0]
    assert len(stopped) >= 15
    x = {round(row["x_end"], 12) for row in stopped}
    assert len(x) == 1                                                    # no creeping
    assert all(row["v_end"] == 0.0 and row["a"] == 0.0 for row in stopped)


def test_S3_release_moving_no_acceleration_jump():
    r = scenario_release(v0=20.0, release_tick=5, then="IDLE")
    f_rel = 2 * 5 + K_TAU
    v_rel = r.rows[f_rel - 1]["v_end"]
    for row in r.rows[:f_rel]:
        assert row["a"] <= -A_MIN_BRAKE + TOL                           # still takeover
    for row in r.rows[f_rel:]:
        assert row["target_speed"] <= v_rel + TOL                         # synchronised, not 30
        assert row["a"] <= TOL                                            # no acceleration
    _check_limits(r.rows)


def test_S3_release_from_standstill_faster():
    r = scenario_release(standstill=True, release_tick=5, then="FASTER")
    f_rel = 2 * 5 + K_TAU
    assert all(row["v_end"] == 0.0 for row in r.rows[:f_rel])
    assert r.rows[f_rel]["a_ctrl"] == pytest.approx(5.0 / 0.6)
    assert r.rows[f_rel]["a"] == pytest.approx(A_MAX_ACCEL)
    _check_limits(r.rows)
