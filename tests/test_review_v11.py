"""Regression tests of the v11 review (reviewer comments 1-3; protocol v11, 7.2, 7.4, 7.5).

Q1 - a lane change chosen while the previous one is still waiting in the FIFO;
Q2 - the lane marking is crossed while an ABORT waits in the FIFO;
Q3 - completeness of the neighbour zone (no fixed 150 m cut);
Q4 - journal: supervisory intervention, active command and forced braking are separate statuses.
"""
import pytest

from rss_shield.params import DT, ABORT, TAKEOVER, FALLBACK, ACTION_INDEX, V_MAX
from rss_shield.rss import d_min_long
from rss_shield import scenarios as S
from rss_shield.shield import RSSShield, ShieldConfig, complete_zone, reach

LL, LR = ACTION_INDEX["LANE_LEFT"], ACTION_INDEX["LANE_RIGHT"]


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
    leads to (3) violates the shield's own criterion; v10 checked it as entering lane 2 only."""
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


def test_Q2_reviewer_case_abort_decided_while_lane_change_queued():
    """v10: ABORT at 0.4 s (before the crossing on frame 5), active at 0.6 s (after it).
    v11: at tick 1 the queued lane change is already supervised; ABORT at 0.2 s becomes active
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


# ---------------------------------------------------------------- Q3
def test_Q3_stationary_leader_beyond_150m_is_checked():
    assert d_min_long(30.0, 0.0, 1.0) == pytest.approx(167.625)
    road, ego, sh = S.scenario_q3_far_leader()
    m0, info0 = sh.mask()
    f0 = info0["IDLE"]["front"]
    assert m0.sum() == 0 and (f0["vehicle"], f0["frame"]) == ("L", 2)
    assert f0["gap"] == pytest.approx(167.0) and f0["d_min_long"] == pytest.approx(167.625)
    _tick(road, ego, "IDLE")                       # the reviewer's step: one IDLE
    m1, info1 = sh.mask()
    f1 = info1["IDLE"]["front"]
    assert m1.sum() == 0 and f1["frame"] == 1
    assert f1["gap"] == pytest.approx(164.0) and f1["d_min_long"] == pytest.approx(167.625)
    assert sh.decide("IDLE").command == TAKEOVER


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
