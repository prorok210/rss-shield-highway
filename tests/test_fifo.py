"""F1-F6: FIFO command delay inside DelayedRSSVehicle (protocol 7.5, list R1.5)."""
import pytest
from rss_shield.params import TAKEOVER, K_TAU, A_MIN_BRAKE, A_MAX_ACCEL
from rss_shield.sim import make_road, add_ego, Runner, scripted
from rss_shield.vehicle import DelayedRSSVehicle


def run(schedule, v0=20.0, target=20.0, initial="IDLE", T=2.0):
    road = make_road()
    ego = add_ego(road, 1, 100.0, v0, target_speed=target, initial_active=initial)
    activations = []
    orig = ego._activate

    def rec(cmd):
        activations.append((ego.frame, cmd))
        orig(cmd)
    ego._activate = rec
    r = Runner(road, ego, scripted(schedule), 0.4).run(T)
    return r, activations


def test_F1_command_active_exactly_k_tau_frames_after_issue():
    assert K_TAU == 2
    for issue_tick in range(4):
        sched = lambda k, t: "FASTER" if k == issue_tick else None
        r, act = run(sched)
        assert act == [(2 * issue_tick + K_TAU, "FASTER")]
        tgt = [row["target_speed"] for row in r.rows]
        f = 2 * issue_tick + K_TAU
        assert all(x == 20.0 for x in tgt[:f]) and tgt[f] == 25.0


def test_F2_order_preserved_for_different_commands():
    seq = ["FASTER", "SLOWER", "SLOWER", "FASTER", "LANE_LEFT", "IDLE"]
    r, act = run(lambda k, t: seq[k] if k < len(seq) else None, T=3.0)
    assert [c for _, c in act] == seq
    assert [f for f, _ in act] == [2 * k + K_TAU for k in range(len(seq))]


def test_F3_empty_slots_at_start_keep_preset_command():
    # t = 0: FASTER preset (target 30) active, FIFO has K_TAU empty slots
    r, act = run(lambda k, t: "IDLE", v0=25.0, target=30.0, initial="FASTER", T=1.0)
    assert act[0] == (K_TAU, "IDLE")
    for row in r.rows[:K_TAU]:
        assert row["active_cmd"] == "FASTER"
        assert row["a"] == pytest.approx(A_MAX_ACCEL)          # 8.33 clipped to +3
        assert row["a_ctrl"] > A_MAX_ACCEL


def test_F4_takeover_uses_same_queue_and_delay():
    r, act = run(lambda k, t: TAKEOVER if k >= 1 else "IDLE", v0=25.0, target=30.0,
                 initial="FASTER", T=1.2)
    first = [f for f, c in act if c == TAKEOVER][0]
    assert first == 2 * 1 + K_TAU
    for row in r.rows:
        if row["frame"] < first:
            assert row["brake_mode"] == 0 and row["a"] > -A_MIN_BRAKE
        else:
            assert row["brake_mode"] == 1 and row["a"] == pytest.approx(-A_MIN_BRAKE)


def test_F5_release_policy_command_arrives_after_tau():
    rel = 5
    r, act = run(lambda k, t: TAKEOVER if k < rel else "IDLE", v0=20.0, T=2.0)
    f_rel = 2 * rel + K_TAU
    for row in r.rows:
        if row["frame"] in (2 * rel, 2 * rel + 1):          # release tick: TAKEOVER still queued
            assert row["brake_mode"] == 1 and row["active_cmd"] == TAKEOVER
        if row["frame"] >= f_rel:
            assert row["brake_mode"] == 0 and row["active_cmd"] == "IDLE"
    assert (f_rel, "IDLE") in act


def test_F6_act_without_argument_neither_adds_nor_drops_commands():
    road = make_road()
    ego = add_ego(road, 1, 100.0, 20.0, target_speed=20.0, initial_active="IDLE")
    ego.act("FASTER")
    snapshot = list(ego.fifo) + [ego.pending]
    assert snapshot.count("FASTER") == 1
    for _ in range(10):                                     # frames without a decision
        ego.act()
        ego.step(0.1)
        assert sum(c is not None for c in ego.fifo) + (ego.pending is not None) <= 1
    assert list(ego.fifo) == [None] * K_TAU and ego.pending is None
    assert ego.active_cmd == "FASTER" and ego.target_speed == 25.0
