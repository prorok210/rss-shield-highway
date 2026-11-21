"""Regression tests for remark 3: density deficit, route progress, cut-in events, S_conv and the
validation schedule; and for the three-valued pilot acceptance of remark 4."""
import pytest

from experiments.metrics import (CutInTracker, DensityTracker, RouteProgressTracker, VehSnap,
                                 ValidationSchedule, compute_s_conv, pair_is_dangerous,
                                 target_distance_L)
from experiments.pilot import evaluate_pilot_status


# ---------------------------------------------------------------- density
def test_density_tracker_accounts_for_the_deficit():
    t = DensityTracker(D_nominal=40, n_removed=2)
    assert t.n_bg_total == 158
    t.record(151, is_start=True)
    t.record(158)
    t.record(155)
    s = t.summary()
    assert s["n_window_start"] == 151 and s["n_outside_start"] == 7
    assert s["density_start"] == pytest.approx(37.75)
    assert s["density_mean"] == pytest.approx((151 + 158 + 155) / 3 / 4)
    assert s["density_min"] == pytest.approx(151 / 4)
    assert s["n_outside_mean"] == pytest.approx((7 + 0 + 3) / 3)
    assert s["n_outside_max"] == 7
    assert s["n_deficit_mean"] == pytest.approx(2 + (7 + 0 + 3) / 3)


def test_density_tracker_rejects_inconsistent_counts():
    t = DensityTracker(D_nominal=10, n_removed=1)
    with pytest.raises(ValueError):
        t.record(40)                      # n_bg_total = 39
    with pytest.raises(ValueError):
        t.summary()                       # no start sample


# ---------------------------------------------------------------- route
def test_route_progress_separates_survival_and_completion():
    r = RouteProgressTracker(x_start=0.0, L_target=100.0, T_max=40.0)
    r.update_frame(5.0, 50.0, False)
    assert r.t_reach_L is None
    r.update_frame(10.0, 100.0, False)
    assert r.t_reach_L == 10.0
    s = r.summary(crashed=False, x_end=120.0)
    assert s["completed"] and s["reached_L"] and s["survived"]
    assert s["t_travel_pen"] == 10.0 and s["delta_x"] == 120.0 and s["v_eff"] == pytest.approx(3.0)

    r2 = RouteProgressTracker(x_start=0.0, L_target=100.0)
    r2.update_frame(1.0, 200.0, True)     # a collision frame never completes the route
    s2 = r2.summary(crashed=True, x_end=200.0)
    assert not s2["completed"] and not s2["survived"] and s2["t_travel_pen"] == 40.0


def test_target_distance_values():
    assert target_distance_L(10) == pytest.approx(755.15, abs=0.01)
    assert target_distance_L(25) == pytest.approx(543.60, abs=0.01)
    assert target_distance_L(40) == pytest.approx(313.66, abs=0.01)


# ---------------------------------------------------------------- cut-in
def _snap(vid, lane, y, vy=0.0, x=0.0, target=None, vx=20.0):
    return VehSnap(vid=vid, lane=lane, target_lane=lane if target is None else target, x=x, y=y,
                   vx=vx, vy=vy)


def _ego(**kw):
    return _snap(0, lane=1, y=4.0, **kw)


def test_cutin_adjacent_lane_moving_towards_ego():
    c = CutInTracker(rho_eval=0.6, pair_is_dangerous_fn=lambda e, n, r: True)
    c.update_frame(1, _ego(), [])
    c.update_frame(2, _ego(), [_snap(1, lane=2, y=7.5, vy=-1.0)])
    assert c.n_cutin == 1 and c.n_cutin_frames == 1


def test_cutin_crossing_and_completing_after_the_move():
    c = CutInTracker(rho_eval=0.6, pair_is_dangerous_fn=lambda e, n, r: n.y < 7.5)
    c.update_frame(1, _ego(), [_snap(1, lane=2, y=8.0)])
    c.update_frame(2, _ego(), [_snap(1, lane=1, y=7.0)])
    assert c.n_cutin == 1

    c2 = CutInTracker(rho_eval=0.6, pair_is_dangerous_fn=lambda e, n, r: n.y < 7.0)
    c2.update_frame(1, _ego(), [_snap(2, lane=2, y=8.0)])
    c2.update_frame(2, _ego(), [_snap(2, lane=1, y=6.0, vy=-1.0)])
    assert c2.n_cutin == 1


def test_cutin_not_counted_while_ego_changes_lane_or_returns():
    danger = lambda e, n, r: True
    c = CutInTracker(rho_eval=0.6, pair_is_dangerous_fn=danger)
    c.update_frame(1, _ego(), [])
    c.update_frame(2, _snap(0, lane=1, y=4.0, target=2), [_snap(1, lane=2, y=7.5, vy=-1.0)])
    assert c.n_cutin == 0
    c.update_frame(3, _ego(), [_snap(1, lane=2, y=7.5, vy=-1.0)], return_active=True)
    assert c.n_cutin == 0


def test_cutin_history_resets_on_recirculation_and_drops_missing_vehicles():
    c = CutInTracker(rho_eval=0.6, pair_is_dangerous_fn=lambda e, n, r: True)
    c.update_frame(1, _ego(), [_snap(1, lane=2, y=7.5, vy=-1.0)])
    assert c._prev_x[1] == 0.0
    c.update_frame(2, _ego(), [_snap(1, lane=2, y=8.0, vy=0.0, x=500.0)])
    assert c._cutin_active[1] is False and c._prev_x[1] == 500.0 and c.n_cutin == 1
    c.update_frame(3, _ego(), [])
    assert 1 not in c._prev_x and c.n_cutin == 1


def test_pair_danger_uses_the_approach_rule():
    ego = _snap(0, 1, 4.0)
    same_lane = _snap(1, 1, 4.0, x=18.0)
    assert pair_is_dangerous(ego, same_lane, 0.6)
    far = _snap(1, 1, 4.0, x=100.0)
    assert not pair_is_dangerous(ego, far, 0.6)


# ---------------------------------------------------------------- S_conv / schedule
def test_compute_s_conv_requires_a_full_window_and_insufficient_is_explicit():
    assert compute_s_conv([5000, 10000, 15000], [10, 11, 12])["status"] == "insufficient_checkpoints"
    r = compute_s_conv([5000, 10000, 15000, 20000], [100, 100, 100, 100])
    assert r["s_conv"] == 5000 and r["budget_sufficient"] and r["status"] == "converged"
    late = compute_s_conv([80000, 85000, 90000, 95000], [1, 1, 100, 100])
    assert late["s_conv"] is None and late["status"] == "not_converged"
    assert compute_s_conv([5000, 10000, 15000, 20000], [0, 0, 0, 0])["status"] == "non_positive_progress"
    with pytest.raises(ValueError):
        compute_s_conv([5000, 9000, 13000, 17000], [1, 2, 3, 4])


def test_validation_schedule_checkpoints():
    v = ValidationSchedule()
    assert v.next_due(4999) is None
    assert v.next_due(5003) == 5000
    v.record(5000, [1.0, 2.0, 3.0])
    assert v.next_due(9000) is None and v.next_due(10000) == 10000
    s = v.s_conv()
    assert s["status"] == "insufficient_checkpoints"


# ---------------------------------------------------------------- pilot acceptance
def _checks(**statuses):
    return {k: {"status": v} for k, v in statuses.items()}


def test_pilot_status_strict_and_diagnostic():
    ok = evaluate_pilot_status(_checks(a="pass", b="pass"))
    assert ok["verdict"] == "pilot:OK" and ok["exit_ok"]
    unv = evaluate_pilot_status(_checks(a="pass", b="unverified"))
    assert unv["verdict"] == "pilot:UNVERIFIED" and not unv["exit_ok"]
    fail = evaluate_pilot_status(_checks(a="pass", b="fail"))
    assert fail["verdict"] == "pilot:FAIL" and not fail["exit_ok"]
    d_unv = evaluate_pilot_status(_checks(a="pass", b="unverified"), "diagnostic")
    assert d_unv["verdict"] == "pilot:DIAGNOSTIC_UNVERIFIED" and d_unv["exit_ok"]
    d_ok = evaluate_pilot_status(_checks(a="pass"), "diagnostic")
    assert d_ok["verdict"] == "pilot:DIAGNOSTIC_OK" and not d_ok["strict_ok"]
    with pytest.raises(ValueError):
        evaluate_pilot_status(_checks(a="maybe"))
