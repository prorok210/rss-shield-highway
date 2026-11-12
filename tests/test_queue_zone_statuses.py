"""Regression tests: FIFO-consistent mask, ABORT delay, neighbour zone, journal statuses
(protocol 7.2, 7.4, 7.5).

Q1 - a lane change chosen while the previous one is still waiting in the FIFO;
Q2 - the lane marking is crossed while an ABORT waits in the FIFO;
Q3 - completeness of the neighbour zone (no fixed 150 m cut);
Q4 - journal: supervisory intervention, active command and forced braking are separate statuses.
"""
import dataclasses

import pytest
from highway_env.vehicle.kinematics import Vehicle

import queue_reference as QR
from rss_shield.accounting import (COUNTERS, count_step, counters_from_journal, new_counters,
                                   step_kind)
from rss_shield.params import DT, ABORT, TAKEOVER, FALLBACK, ACTION_INDEX, SUPERVISORY, V_MAX
from rss_shield.rss import d_min_long
from rss_shield import scenarios as S
from rss_shield.shield import (MAIN_SERIES, Decision, RSSShield, ShieldConfig, ZoneError,
                               complete_zone, envelope_zone, reach, require_zone)
from rss_shield.sim import Runner, add_ego, make_road, scripted

LL, LR = ACTION_INDEX["LANE_LEFT"], ACTION_INDEX["LANE_RIGHT"]
IDLE = ACTION_INDEX["IDLE"]


class InstantStateShield(RSSShield):
    """Control: the behaviour before the fix - the mask and the supervision read the current
    maneuver / target lane instead of the state after the FIFO."""

    def plan_state(self):
        return self.ego.ghost()


def _tick(road, ego, cmd):
    if cmd is not None:
        ego.act(cmd)
    for _ in range(2):
        road.act()
        road.step(DT)


def _first(rows, cond):
    return next((row["frame"] for row in rows if cond(row)), None)


# ---------------------------------------------------------------- Q1
def test_Q1_actual_target_of_second_lane_change_violates_rss():
    """Tick 1: the first LANE_RIGHT is queued, not active. The lane a second LANE_RIGHT really
    leads to (3) violates the shield's own criterion; it must not be checked as entering lane 2."""
    road, ego, sh = S.scenario_q1_scene()
    assert sh.decide("LANE_RIGHT").command == "LANE_RIGHT"
    _tick(road, ego, "LANE_RIGHT")
    assert ego.maneuver is None and "LANE_RIGHT" in list(ego.fifo)
    v3 = sh.check_plan(["LANE_RIGHT"], sh.lc_frames, {3})
    assert v3["rear"]["vehicle"] == "R" and v3["rear"]["gap"] < v3["rear"]["d_min_long"]
    d1 = sh.decide("LANE_RIGHT")
    assert sh.plan["maneuver"]["kind"] == "lane_change" and sh.plan["target_lane"] == 2
    assert d1.mask[LR] == 0 and d1.mask[LL] == 0 and d1.command != "LANE_RIGHT"
    assert d1.violations["LANE_RIGHT"] == {"busy": True, "maneuver": "lane_change", "queued": True}


def test_Q1_repeated_lane_change_before_activation_is_masked():
    r = S.scenario_q1_double_lane_change()
    d0, d1 = r.decisions[0][2], r.decisions[1][2]
    assert d0.command == "LANE_RIGHT"
    assert _first(r.rows, lambda row: row["active_cmd"] == "LANE_RIGHT") == 2   # queued at tick 1
    assert d1.mask[LR] == 0 and d1.mask[LL] == 0 and d1.command != "LANE_RIGHT"
    assert {row["target_lane_index"] for row in r.rows} == {1, 2}
    assert r.rows[-1]["y_end"] == pytest.approx(8.0, abs=0.1) and r.t_contact is None


def test_Q1_review_counterexample_gap_1_661_vs_97_685_is_forbidden():
    """R in lane 3, 5 m behind at 28 m/s: the second LANE_RIGHT (0.2 s) really leads to lane 3,
    rear gap 1.661 m against d_min_long = 97.685 m. The mask must not offer it."""
    road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0)
    assert sh.decide("LANE_RIGHT").command == "LANE_RIGHT"
    _tick(road, ego, "LANE_RIGHT")
    rear = QR.command_violations(ego, sh.cfg, "LANE_RIGHT")["rear"]
    assert rear["vehicle"] == "R"
    assert rear["gap"] == pytest.approx(1.661, abs=5e-4)
    assert rear["d_min_long"] == pytest.approx(97.685, abs=5e-4)
    d1 = sh.decide("LANE_RIGHT")
    assert d1.mask[LR] == 0 and d1.command != "LANE_RIGHT"
    assert list(d1.mask) == list(QR.reference_mask(ego, sh.cfg))


def test_Q1_control_instantaneous_state_admits_the_counterexample():
    """Control of the test above: a mask that reads the current maneuver (behaviour before the
    fix) allows the second LANE_RIGHT although the lane it leads to violates the criterion."""
    road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0, shield=InstantStateShield)
    _tick(road, ego, sh.decide("LANE_RIGHT").command)
    d1 = sh.decide("LANE_RIGHT")
    assert d1.mask[LR] == 1 and d1.command == "LANE_RIGHT"
    assert QR.command_violations(ego, sh.cfg, "LANE_RIGHT")["rear"]["gap"] < 2.0
    assert QR.reference_mask(ego, sh.cfg)[LR] == 0


def test_Q1c_two_queued_lane_changes_are_checked_as_entry_into_lane_3():
    """FIFO of 4 frames: two LANE_RIGHT (issued without the filter) wait in it. After the queue
    the maneuver is 1 -> 3; every action is checked with the vehicles behind in lanes 2 and 3."""
    road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0, k_tau=4)
    _tick(road, ego, "LANE_RIGHT")
    _tick(road, ego, "LANE_RIGHT")
    assert ego.maneuver is None and [c for c in ego.fifo if c] == ["LANE_RIGHT", "LANE_RIGHT"]
    m, info = sh.mask()
    assert m.sum() == 0 and info["IDLE"]["rear"]["vehicle"] == "R"
    assert info["LANE_RIGHT"]["busy"] and info["LANE_LEFT"]["busy"]
    assert list(m) == list(QR.reference_mask(ego, sh.cfg))
    assert sh.plan["target_lane"] == 3 and sh.plan["lanes"] == {2, 3}
    assert sh.plan["maneuver"]["origin"][2] == 1 and sh.plan["maneuver"]["target"][2] == 3


def test_Q1c_maneuver_over_two_lanes_checks_the_lane_in_between():
    """Two LANE_RIGHT without the filter (ticks 0 and 1): at tick 2 the maneuver 1 -> 3 is active.
    R drives in lane 2 (the lane in between), close behind: IDLE must be checked against it,
    not only against lane 3."""
    road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0, lane_r=2)
    _tick(road, ego, "LANE_RIGHT")
    _tick(road, ego, "LANE_RIGHT")
    m, info = sh.mask()
    assert m[IDLE] == 0 and info["IDLE"]["rear"]["vehicle"] == "R"
    assert list(m) == list(QR.reference_mask(ego, sh.cfg))
    man = sh.plan["maneuver"]
    assert (man["origin"][2], man["target"][2]) == (1, 3) and sh.plan["lanes"] == {2, 3}


QUEUE_SWEEP = [(rho, cfg) for rho in (0.4, 1.0) for cfg in ({}, MAIN_SERIES)]
# 220, 273, 363: a maneuver over two lanes with a vehicle close behind in the lane in between
QUEUE_SEEDS = (*range(60), 220, 273, 363)


def _sweep(rho, kw, shield=RSSShield):
    """(scenes where the mask differs from the unlimited reference, coverage of the sweep)."""
    seen = dict(queued_lc=0, two_lanes=0, busy=0, lc_allowed=0, lc_refused=0)
    diff = []
    for seed in QUEUE_SEEDS:
        road, ego = S.scenario_queue_random(seed)
        cfg = ShieldConfig(rho=rho, **kw)
        m, info = shield(ego, cfg).mask()
        ref = QR.reference_mask(ego, cfg)
        if list(m) != list(ref):
            diff.append((seed, list(ego.fifo), list(m), list(ref)))
        lanes = QR.lanes_after_queue(ego)
        seen["queued_lc"] += any(c in ("LANE_LEFT", "LANE_RIGHT") for c in ego.fifo)
        seen["two_lanes"] += len(lanes) > 1
        seen["busy"] += bool(lanes)
        seen["lc_allowed"] += int(ref[LL] + ref[LR] > 0)
        seen["lc_refused"] += any("front" in info[a] or "rear" in info[a]
                                  for a in ("LANE_LEFT", "LANE_RIGHT"))
    return diff, seen


@pytest.mark.parametrize("rho, kw", QUEUE_SWEEP, ids=lambda x: "main" if x == MAIN_SERIES else str(x))
def test_Q1d_mask_equals_unlimited_reference_on_queued_command_sweep(rho, kw):
    """Random scenes with lane changes / supervisory commands waiting in the FIFO (2 or 4
    frames): the mask (default neighbour zone) equals the independent unlimited reference."""
    diff, seen = _sweep(rho, kw)
    assert diff == []
    assert all(seen.values()), seen                 # the sweep exercises every branch


def test_Q1d_control_instantaneous_state_fails_the_sweep():
    """Control of the sweep: the mask that reads the current maneuver (behaviour before the fix)
    differs from the reference on scenes with a queued lane change."""
    diff, _ = _sweep(0.4, MAIN_SERIES, shield=InstantStateShield)
    assert len(diff) >= 5
    assert all(any(c in ("LANE_LEFT", "LANE_RIGHT", ABORT, FALLBACK) for c in fifo)
               for _, fifo, _, _ in diff)


# ---------------------------------------------------------------- Q2
@pytest.mark.parametrize("v, gap_n, t_b", S.Q2_LATE_ABORT_CASES)
def test_Q2_no_abort_becomes_active_after_crossing(v, gap_n, t_b):
    r = S.scenario_q2_abort_crossing(v, gap_n, t_b)
    rows = r.rows
    cross = _first(rows, lambda row: row["lane_index"] == 2)
    act = _first(rows, lambda row: row["active_cmd"] == ABORT)
    assert act is None or cross is None or act <= cross
    assert r.ego.abort_ignored == 0
    if cross is not None:                                  # no return after the crossing
        assert all(row["lane_index"] == 2 for row in rows[cross:])
        assert any(d.violations.get("ABORT") == {"after_crossing": True} for _, _, d in r.decisions)
    else:                                                  # the return started before crossing
        assert act is not None and all(row["lane_index"] == 1 for row in rows)
    assert r.t_contact is None


def test_Q2_abort_decided_while_lane_change_queued():
    """Without the queue check: ABORT at 0.4 s (before the crossing on frame 5), active at 0.6 s
    (after it). With it: at tick 1 the queued lane change is already supervised; ABORT at 0.2 s becomes active
    on frame 4, before the crossing; the ego stays in lane 1."""
    r = S.scenario_q2_abort_crossing(10.0, 15.0, 0.0)
    cmds = [(round(t, 1), d.command) for _, t, d in r.decisions if d.command != "IDLE"]
    assert cmds == [(0.0, "LANE_RIGHT"), (0.2, ABORT)]
    assert _first(r.rows, lambda row: row["active_cmd"] == ABORT) == 4
    assert all(row["lane_index"] == 1 for row in r.rows) and r.ego.abort_ignored == 0


def test_Q2_crossing_while_abort_would_wait_gives_no_abort():
    """(8, 15, 0): at tick 2 (start of frame 4) the marking is not crossed, but it is crossed on
    frame 5, before a command decided now becomes active (frame 6): ABORT is not offered."""
    r = S.scenario_q2_abort_crossing(8.0, 15.0, 0.0)
    assert r.rows[3]["lane_index"] == 1 and _first(r.rows, lambda row: row["lane_index"] == 2) == 5
    d2 = r.decisions[2][2]
    assert d2.violations["ABORT"] == {"after_crossing": True}
    assert d2.command == TAKEOVER and d2.reason == "complete"
    assert ABORT not in [row["active_cmd"] for row in r.rows]


def test_Q2b_vehicle_ignores_abort_arriving_after_crossing():
    r = S.scenario_q2_scripted_abort()
    cross = _first(r.rows, lambda row: row["lane_index"] == 2)
    act = _first(r.rows, lambda row: row["active_cmd"] == ABORT)
    assert cross == 5 and act == 6
    assert r.ego.abort_ignored == 1 and r.rows[-1]["abort_ignored"] == 1
    assert all(row["lane_index"] == 2 and row["return_active"] == 0 for row in r.rows[cross:])
    assert r.rows[-1]["target_lane_index"] == 2


def test_Q2_control_instantaneous_state_issues_abort_that_lands_after_crossing():
    """Control of Q2: a supervision that reads the current maneuver state (behaviour before the
    fix) schedules the return at 0.4 s, before the crossing (frame 5); it leaves the FIFO at
    0.6 s, after it. The vehicle refuses it: abort_ignored, no return."""
    r = S.scenario_q2_abort_crossing(10.0, 15.0, 0.0, shield=InstantStateShield)
    aborts = [round(t, 1) for _, t, d in r.decisions if d.command == ABORT]
    cross = _first(r.rows, lambda row: row["lane_index"] == 2)
    act = _first(r.rows, lambda row: row["active_cmd"] == ABORT)
    assert aborts[0] == 0.4 and cross == 5 and act == 6
    assert r.ego.abort_ignored >= 1 and r.rows[act]["abort_ignored"] == 1
    assert all(row["return_active"] == 0 for row in r.rows)
    assert all(row["lane_index"] == 2 for row in r.rows[cross:])


def test_Q2b_abort_without_a_lane_change_is_recorded_not_dropped():
    """An ABORT that leaves the FIFO after the maneuver is completed (or without any maneuver)
    starts no return and is counted in abort_ignored."""
    road, ego, _ = S.scenario_q1_scene()
    _tick(road, ego, ABORT)
    _tick(road, ego, None)
    assert ego.active_cmd == ABORT and ego.abort_ignored == 1
    assert ego.maneuver is None and not ego.return_active and ego.target_lane_index[2] == 1


# ---------------------------------------------------------------- Q3
def test_Q3_stationary_leader_beyond_150m_is_checked():
    assert d_min_long(30.0, 0.0, 1.0) == pytest.approx(167.625)
    road, ego, sh = S.scenario_q3_far_leader()
    m0, info0 = sh.mask()
    f0 = info0["IDLE"]["front"]
    assert m0.sum() == 0 and (f0["vehicle"], f0["frame"]) == ("L", 2)
    assert f0["gap"] == pytest.approx(167.0) and f0["d_min_long"] == pytest.approx(167.625)
    _tick(road, ego, "IDLE")                       # one IDLE step
    m1, info1 = sh.mask()
    f1 = info1["IDLE"]["front"]
    assert m1.sum() == 0 and f1["frame"] == 1
    assert f1["gap"] == pytest.approx(164.0) and f1["d_min_long"] == pytest.approx(167.625)
    assert sh.decide("IDLE").command == TAKEOVER


class FixedRadiusShield(RSSShield):
    """Control: the neighbour zone before the fix - a fixed radius that is not validated."""
    radius = 150.0

    def _others(self, n_frames):
        x0 = self.ego.position[0]
        return [v for v in self.road.vehicles
                if v is not self.ego and abs(v.position[0] - x0) <= self.radius]


def test_Q3_control_fixed_150m_zone_admits_idle_and_300m_detects_the_leader():
    """Control of Q3: with the zone of 150 m the standing leader (173 m, then 170 m) is discarded
    before the prediction and IDLE stays allowed; 300 m shows 164 m against 167.625 m on the
    first predicted frame."""
    road, ego, sh = S.scenario_q3_far_leader(shield=FixedRadiusShield)
    assert sh.mask()[0].sum() == 5
    _tick(road, ego, "IDLE")
    m1, _ = sh.mask()
    assert m1[IDLE] == 1 and m1.sum() == 5
    sh.radius = 300.0
    m, info = sh.mask()
    f = info["IDLE"]["front"]
    assert m[IDLE] == 0 and (f["vehicle"], f["frame"]) == ("L", 1)
    assert f["gap"] == pytest.approx(164.0) and f["d_min_long"] == pytest.approx(167.625)


@pytest.mark.parametrize("rho", [0.1, 0.4, 1.0])
def test_Q3b_zone_filter_equals_full_check(rho):
    dropped = 0
    for seed in range(12):
        road, ego = S.scenario_q3_random(seed, ticks=1 + seed % 2)
        full = RSSShield(ego, ShieldConfig(rho=rho))
        full.filter_on = False
        fixed = RSSShield(ego, ShieldConfig(rho=rho, zone=complete_zone(rho)))
        filt = RSSShield(ego, ShieldConfig(rho=rho))
        for a_pi in ("LANE_LEFT", "FASTER"):
            ref = full.decide(a_pi)
            for sh in (filt, fixed):
                d = sh.decide(a_pi)
                assert list(d.mask) == list(ref.mask)
                assert (d.command, d.reason, d.violations) == (ref.command, ref.reason, ref.violations)
        dropped += len(road.vehicles) - 1 - len(filt._others(filt.lc_frames))
    assert dropped > 0                              # the filter is not vacuous


def test_Q3c_fixed_zone_must_be_complete():
    assert reach(30.0, 4, 1.0) == pytest.approx(196.085, abs=1e-3)
    assert reach(30.0, 24, 1.0) == pytest.approx(325.985, abs=1e-3)
    assert complete_zone(1.0) == pytest.approx(472.985, abs=1e-3)
    assert reach(V_MAX, 24, 1.0) == pytest.approx(complete_zone(1.0))
    with pytest.raises(ValueError):
        ShieldConfig(rho=1.0, zone=150.0)
    with pytest.raises(ValueError):
        ShieldConfig(rho=0.1, zone=150.0)
    assert ShieldConfig(rho=1.0, zone=complete_zone(1.0)).zone == pytest.approx(472.985, abs=1e-3)


# ---------------------------------------------------------------- Q4
def test_Q4_journal_abort_statuses():
    r = S.scenario2b("i", model="observed")
    rows = r.rows
    assert rows[0]["policy_step"] == 1 and rows[0]["intervention"] == 0
    for f in (4, 5):                                 # ABORT decision interval: intercept step
        assert rows[f]["intervention"] == 1 and rows[f]["sup_cmd"] == ABORT
    assert rows[4]["policy_step"] == 0
    act = [row for row in rows if row["active_cmd"] == ABORT]
    assert [row["frame"] for row in act] == [6, 7]   # active after the FIFO delay
    assert all(row["active_sup"] == 1 and row["brake_mode"] == 0 and row["return_active"] == 1
               for row in act)
    ticks = [row for row in rows if row["decision_tick"]]
    assert all(row["policy_step"] + row["intervention"] == 1 for row in ticks)
    # both flags are set per decision interval: on the tick frame and on the next frame
    assert rows[1]["policy_step"] == 1 and rows[1]["intervention"] == 0
    assert rows[5]["policy_step"] == 0 and rows[5]["intervention"] == 1
    assert all(row["policy_step"] + row["intervention"] == 1 for row in rows)


def test_Q4b_journal_takeover_and_fallback_statuses():
    r = S.scenario1A(0.4, "sync")
    rows = r.rows
    assert all(row["intervention"] == 1 and row["sup_cmd"] == TAKEOVER and row["policy_step"] == 0
               for row in rows)
    assert [row["brake_mode"] for row in rows[:3]] == [0, 0, 1]
    assert all(row["active_sup"] == 1 and row["brake_mode"] == 1 and row["return_active"] == 0
               for row in rows[2:])
    r = S.scenario2b("iii", model="observed", T=8.0)
    rows = r.rows
    assert rows[4]["sup_cmd"] == FALLBACK and rows[4]["intervention"] == 1
    assert rows[6]["active_cmd"] == FALLBACK and rows[6]["brake_mode"] == 1 and rows[6]["fallback"] == 1
    assert rows[5]["brake_mode"] == 0


def test_Q3c_zone_is_a_function_of_the_scenario_envelope():
    """V_MAX is the speed bound of highway-env; a window that truncates the zone is accepted only
    for the declared speed envelope (main generator: ego <= 30, traffic <= 27 m/s)."""
    assert V_MAX == Vehicle.MAX_SPEED == 40.0
    assert envelope_zone(1.0, V_MAX, V_MAX) == (complete_zone(1.0), complete_zone(1.0))
    ahead, behind = envelope_zone(1.0, v_ego_max=30.0, v_other_max=27.0)
    assert ahead == pytest.approx(325.985, abs=1e-3) and behind == pytest.approx(286.76, abs=1e-3)
    assert require_zone(700.0, 300.0, 1.0, 30.0, 27.0) == (ahead, behind)
    with pytest.raises(ZoneError):
        require_zone(700.0, 250.0, 1.0, 30.0, 27.0)       # too short behind
    with pytest.raises(ZoneError):
        require_zone(300.0, 300.0, 1.0, 30.0, 27.0)       # too short ahead
    with pytest.raises(ZoneError):
        require_zone(700.0, 300.0, 1.0, 30.0, 30.0)       # faster traffic needs 325.985 m behind


def test_Q3c_zone_cannot_be_shrunk_after_validation():
    cfg = ShieldConfig(rho=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.zone = 150.0
    with pytest.raises(ZoneError):
        dataclasses.replace(cfg, zone=150.0)
    assert issubclass(ZoneError, ValueError)


# ---------------------------------------------------------------- Q4: decision set
def _journals():
    yield S.scenario1A(0.2, "sync")
    yield S.scenario1A(0.2, "worst", "once")
    yield S.scenario1B(0.2, "nominal")
    yield S.scenario_stop()
    yield S.scenario_release()
    yield S.scenario_random(0, T=20.0)
    yield S.scenario2a()
    yield S.scenario2b("i", model="observed")
    yield S.scenario2b("iii", model="observed", T=8.0)
    yield S.scenario_q1_double_lane_change()
    yield S.scenario_q2_abort_crossing()
    yield S.scenario_q2_scripted_abort()
    road = make_road()                                   # ticks without a new command
    ego = add_ego(road, 1, 100.0, 20.0, target_speed=20.0, initial_active="IDLE")
    yield Runner(road, ego, scripted(lambda k, t: "FASTER" if k == 2 else None), 0.4).run(2.0)


def test_Q4c_exactly_one_of_policy_step_and_intervention_on_every_frame():
    n_int = n_pol = 0
    for r in _journals():
        for row in r.rows:
            assert row["policy_step"] + row["intervention"] == 1, (r.name, row["frame"])
            assert row["intervention"] == int(row["sup_cmd"] in SUPERVISORY)
            assert "takeover" not in row                 # removed from the journal schema
        for a, b in zip(r.rows[0::2], r.rows[1::2]):     # both frames of a decision interval
            assert (a["policy_step"], a["sup_cmd"]) == (b["policy_step"], b["sup_cmd"])
        n_int += sum(row["intervention"] for row in r.rows)
        n_pol += sum(row["policy_step"] for row in r.rows)
    assert n_int > 100 and n_pol > 100


def test_Q4d_decision_takeover_is_a_deprecated_alias_of_intervention():
    mask = [0] * 5
    for cmd, brake in ((TAKEOVER, True), (FALLBACK, True), (ABORT, False), ("IDLE", False)):
        d = Decision(cmd, "IDLE", None, mask)
        assert d.intervention == (cmd in SUPERVISORY) and d.brake == brake
        assert step_kind(cmd) == ("intervention" if d.intervention else "policy")
        with pytest.warns(DeprecationWarning):
            assert d.takeover == d.intervention
    assert step_kind(None) == "policy"


def _count(decisions):
    c = new_counters()
    kinds = [count_step(c, d.a_pi, d.command, d.mask, d.violations.get(d.a_pi)) for d in decisions]
    return c, kinds


@pytest.mark.parametrize("variant, sup", [("i", ABORT), ("iii", FALLBACK)])
def test_Q4e_requests_while_a_maneuver_is_completed_on_an_intercept_segment(variant, sup):
    """Scenario 2b, the policy asks LANE_RIGHT on every tick. The request of a tick at which the
    supervisor decides goes to N_att_int; the request of a policy tick during the same maneuver
    is an attempt blocked as busy (N_att, N_blocked, N_busy). Never both, never outside E."""
    r = S.scenario2b(variant, model="observed", T=8.0, policy=lambda k, t: "LANE_RIGHT")
    D = [d for _, _, d in r.decisions]
    c, kinds = _count(D)
    assert [d.command for d in D[:3]] == ["LANE_RIGHT", "IDLE", sup]
    assert kinds[:3] == ["policy", "policy", "intervention"]
    assert D[1].violations["LANE_RIGHT"]["busy"]             # queued lane change: busy, not RSS
    n_int = sum(d.intervention for d in D)
    assert c["intercept"] == n_int >= 1 and c["E"] == len(D) - n_int
    assert c["N_att_int"] == n_int and c["N_att"] == c["E"]  # every request counted exactly once
    assert c["N_att"] + c["N_att_int"] == c["steps"] == len(D)
    assert c["N_busy"] >= 1 and c["N_blocked"] >= c["N_busy"]
    for d, kind in zip(D, kinds):                            # the tick decides, not the active command
        assert (kind == "intervention") == (d.command in SUPERVISORY)
    # policy ticks while the supervisory command is still active or the maneuver is in progress
    busy = [d for d in D if not d.intervention and (d.violations.get("LANE_RIGHT") or {}).get("busy")]
    assert len(busy) == c["N_busy"] >= 2


def test_Q4f_requests_are_counted_on_the_exposure_set_only():
    """N_att and N_blocked are evaluated on the same steps as E; adding intercept steps changes
    N_att_int (and the intercept counters) only."""
    allow, block = [1, 1, 1, 1, 1], [0, 1, 0, 1, 1]
    steps = [("LANE_LEFT", "LANE_LEFT", allow, None), ("LANE_RIGHT", "IDLE", block, {"rear": {}}),
             ("IDLE", "IDLE", allow, None), ("LANE_RIGHT", "IDLE", block, {"busy": True}),
             ("LANE_LEFT", "FASTER", block, {"no_lane": True}), ("FASTER", "FASTER", allow, None)]
    base = new_counters()
    for a_pi, cmd, mask, why in steps:
        assert count_step(base, a_pi, cmd, mask, why) == "policy"
    assert (base["E"], base["N_att"], base["N_blocked"], base["N_busy"], base["N_no_lane"],
            base["masked"], base["N_att_int"], base["intercept"]) == (6, 4, 3, 1, 1, 3, 0, 0)
    more = new_counters()
    inserted = [("LANE_RIGHT", TAKEOVER), ("LANE_LEFT", ABORT), ("IDLE", FALLBACK), ("LANE_LEFT", TAKEOVER)]
    for i, (a_pi, cmd, mask, why) in enumerate(steps):
        count_step(more, a_pi, cmd, mask, why)
        if i < len(inserted):
            assert count_step(more, inserted[i][0], inserted[i][1], [0] * 5, {"rear": {}}) == "intervention"
    changed = {k for k in COUNTERS if more[k] != base[k]}
    assert changed == {"steps", "intercept", "takeover", "abort", "fallback", "N_att_int"}
    assert more["N_att_int"] == 3 and more["intercept"] == 4 and more["steps"] == base["steps"] + 4
    assert (more["E"], more["N_att"], more["N_blocked"]) == (base["E"], base["N_att"], base["N_blocked"])
    # the same counters are recovered from the journal flags alone
    rows = [dict(policy_step=1, intervention=0, a_pi=a, command=c, block_reason=(
                 "busy" if (w or {}).get("busy") else "no_lane" if (w or {}).get("no_lane")
                 else "rss" if not m[ACTION_INDEX.get(a, 1)] else "")) for a, c, m, w in steps]
    rows += [dict(policy_step=0, intervention=1, a_pi=a, command=c, block_reason="") for a, c in inserted]
    assert counters_from_journal(rows) == more
