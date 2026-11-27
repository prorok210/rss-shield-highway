"""Regression tests for remark 3: density deficit, route progress, cut-in events, S_conv and the
validation schedule (including exact-step mid-episode validation on the real DQN training path);
and for the three-valued pilot acceptance of remark 4."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.metrics import (VAL_HISTORY_COLUMNS, CutInTracker, DensityTracker,
                                 RouteProgressTracker, VehSnap, ValidationSchedule, compute_s_conv,
                                 pair_is_dangerous, target_distance_L)
from experiments.pilot import evaluate_pilot_status, train


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


def test_validation_history_keeps_ap_and_outcome_aggregates():
    v = ValidationSchedule()
    row = v.record(5000, [dict(dx=1.0, N_att=2, N_blocked=1, v_eff=10.0, completed=True),
                          dict(dx=3.0, N_att=4, N_blocked=2, v_eff=14.0, completed=False)])
    assert row["delta_x_mean"] == 2.0
    assert row["n_att_mean"] == 3.0 and row["n_blocked_mean"] == 1.5
    assert row["v_eff_mean"] == 12.0 and row["comp_rate"] == 0.5
    assert tuple(row.keys()) == VAL_HISTORY_COLUMNS
    assert row["step"] == 5000 and row["nominal_step"] == 5000


def test_validation_schedule_and_s_conv_guardrails():
    with pytest.raises(ValueError):
        ValidationSchedule(every=0)
    with pytest.raises(ValueError):
        ValidationSchedule(every=-100)
    with pytest.raises(ValueError):
        ValidationSchedule(scenario_ids=())

    v = ValidationSchedule(every=5000)
    with pytest.raises(ValueError):
        v.record(5040, [100.0])                     # overshoot rejected in exact-step mode
    with pytest.raises(ValueError):
        v.record(5000, [100.0], nominal_step=10000) # mismatched nominal mark rejected
    assert v.history == []

    # compute_s_conv must reject invalid spacing / monotonicity / NaN even with < window points
    with pytest.raises(ValueError):
        compute_s_conv([5040, 10080], [100.0, 110.0])
    with pytest.raises(ValueError):
        compute_s_conv([5000, 5000], [100.0, 110.0])
    with pytest.raises(ValueError):
        compute_s_conv([5000], [float("nan")])
    one = compute_s_conv([5000], [420.0])
    assert one["status"] == "insufficient_checkpoints" and one["x_max"] == pytest.approx(420.0)


def test_train_exact_step_validation_mid_episode_and_bit_exact_invariance(tmp_path):
    """Real DQN training path (Proposed, Mode B): checkpoints occur strictly mid-episode when
    `every` does not divide episode length (including two checkpoints inside one episode), record
    exact step/nominal/updates context, compute a full 4-point S_conv, and leave training state and
    RNG 100% bit-identical to a run without validation."""
    args = SimpleNamespace(D=10, rho=0.6, seed=42, steps=150, max_steps=150, learning_starts=20)
    seeds = [900_040, 900_041, 900_042]
    cache = str(tmp_path / "cache")
    env_cfg = dict(warmup=2.0, duration=10)         # 50 decision steps per full episode
    dqn_kw = dict(batch_size=16)
    val = ValidationSchedule(every=35, scenario_ids=(10_000, 10_001))

    agent_v, slog_v, eps_v, _ = train(
        args, seeds, cache, lambda _msg: None,
        shield_on=True, reward_kind="Rv", validation=val,
        env_config=env_cfg, dqn_kwargs=dqn_kw,
    )
    agent_0, slog_0, eps_0, _ = train(
        args, seeds, cache, lambda _msg: None,
        shield_on=True, reward_kind="Rv", validation=None,
        env_config=env_cfg, dqn_kwargs=dqn_kw,
    )

    assert len(val.history) == 4
    assert [r["step"] for r in val.history] == [35, 70, 105, 140]
    assert [r["nominal_step"] for r in val.history] == [35, 70, 105, 140]
    assert [r["episode"] for r in val.history] == [0, 1, 2, 2]
    assert [r["episode_step"] for r in val.history] == [35, 20, 5, 40]
    assert [r["updates"] for r in val.history] == [16, 51, 86, 121]
    for r in val.history:
        assert tuple(r.keys()) == VAL_HISTORY_COLUMNS
        assert r["n"] == 2
        assert np.isfinite(r["delta_x_mean"]) and r["delta_x_mean"] > 0.0
        assert np.isfinite(r["v_eff_mean"]) and np.isfinite(r["n_att_mean"])
        assert np.isfinite(r["n_blocked_mean"]) and 0.0 <= r["comp_rate"] <= 1.0

    sc = val.s_conv()
    assert sc["status"] in {"converged", "not_converged", "late_convergence"}
    assert sc["x_max"] > 0.0 and sc["threshold"] == pytest.approx(0.95 * sc["x_max"])

    # Bit-exact invariance: mid-episode validation does not perturb training weights, buffer, or RNG
    assert agent_v.steps == agent_0.steps == 150
    assert agent_v.updates == agent_0.updates == 131
    assert all(np.array_equal(a, b) for a, b in zip(agent_v.params, agent_0.params))
    assert all(np.array_equal(a, b) for a, b in zip(agent_v.target, agent_0.target))
    assert agent_v.buffer.size == agent_0.buffer.size
    assert np.array_equal(agent_v.buffer.R[:agent_v.buffer.size],
                          agent_0.buffer.R[:agent_0.buffer.size])
    assert agent_v.rng.bit_generator.state == agent_0.rng.bit_generator.state
    assert [e["ret"] for e in eps_v] == [e["ret"] for e in eps_0]
    assert len(slog_v) == len(slog_0)
    # Validation time must stay outside per-step t_learn
    assert sum(e["t_learn"] for e in eps_v) < 0.5


def test_train_exact_step_validation_with_crashing_baseline2_episodes(tmp_path):
    """Unshielded Baseline 2 episodes terminate at variable crash lengths, yet validation triggers
    at exact nominal steps without overshoot."""
    args = SimpleNamespace(D=25, rho=0.6, seed=42, steps=120, max_steps=120, learning_starts=20)
    seeds = list(range(900_040, 900_060))
    cache = str(tmp_path / "cache")
    val = ValidationSchedule(every=30, scenario_ids=(10_000, 10_001))

    agent, _, eps, _ = train(
        args, seeds, cache, lambda _msg: None,
        shield_on=False, reward_kind="RS", validation=val,
        env_config=dict(warmup=2.0, duration=10), dqn_kwargs=dict(batch_size=16),
    )

    assert any(e["crashed"] and e["steps"] < 50 for e in eps)
    assert [r["step"] for r in val.history] == [30, 60, 90, 120]
    assert [r["nominal_step"] for r in val.history] == [30, 60, 90, 120]
    assert [r["updates"] for r in val.history] == [11, 41, 71, 101]
    assert agent.steps >= 120
    sc = val.s_conv()
    assert sc["status"] != "insufficient_checkpoints" and sc["x_max"] > 0.0


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
