"""Block 5: the H1 event and the correct-start rule are single definitions (experiments/h1.py) and
give the documented label for both arms on the boundary cases."""
import numpy as np
import pytest

from experiments import h1
from experiments.dqn import DQN
from experiments.env import make_env
from experiments.generator import RHO_INSERT, RHOS, initially_unsafe
from experiments.runner import run_episode
from experiments.h1 import (CATEGORIES, CR_EGO, CR_EGO_MANEUVER, CR_EGO_REACTION, CR_ENV_BOUNDS, CR_ENV_NEIGHBOUR,
                            CR_ENV_REAR, CR_INIT, CR_UNC, RHO_REF, RHO_START, Trace,
                            classification_rho, classify_collision, correct_start, danger_onset,
                            h1_summary, is_h1_event, reaction_late, reaction_start, start_violation)
from rss_shield.params import DT, LENGTH, TAU_EFF
from rss_shield.rss import d_min_long
from rss_shield.sim import add_ego, add_scripted, make_road

# the two arms of H1: (name, filter on, rho of the cell)
ARMS = [("Proposed", True, 0.6), ("Proposed", True, 1.0), ("Baseline 2", False, None)]


# ---------------------------------------------------------------------- trace builders
def _trace(ego, partner, lc=None, crashed=None, sup=(), lc_ticks=()):
    n = len(ego["x"])
    for d in (ego, partner):
        for f in ("y", "vy"):
            d.setdefault(f, np.zeros(n))
    return Trace(ego=ego, partner=partner, lc=np.zeros(n, bool) if lc is None else lc,
                 partner_crashed=np.zeros(n, bool) if crashed is None else crashed,
                 sup_ticks=tuple(sup), lc_ticks=tuple(lc_ticks))


def follow(rho, delay_frames, v0=25.0, margin=0.5, a_lead=-8.0, a_before=3.0, a_brake=-4.0,
           n_max=400):
    """Ego behind a leader in one lane, equal speeds v0, gap d_min_long(rho) + margin (safe at
    t = 0). The leader brakes with a_lead from the first frame. The ego accelerates with a_before
    (the worst case assumed by RSS) until `delay_frames` frames after the danger onset t_b, then
    brakes with a_brake. Explicit Euler, position first (as highway-env). Returns the trace up to
    the contact, or None if there is no contact."""
    xe, ve, xl, vl = 0.0, v0, LENGTH + d_min_long(v0, v0, rho) + margin, v0
    E = dict(x=[xe], v=[ve], a=[0.0])
    P = dict(x=[xl], v=[vl], a=[0.0])
    k_b = None
    for k in range(n_max):
        if k_b is None and xl - xe - LENGTH < d_min_long(ve, vl, rho) - 1e-9:
            k_b = k
        ae = a_brake if (k_b is not None and k >= k_b + delay_frames) else a_before
        al = a_lead
        if ve + ae * DT < 0:
            ae = -ve / DT
        if vl + al * DT < 0:
            al = -vl / DT
        xe, xl = xe + ve * DT, xl + vl * DT
        ve, vl = ve + ae * DT, vl + al * DT
        for d, x, v, a in ((E, xe, ve, ae), (P, xl, vl, al)):
            d["x"].append(x); d["v"].append(v); d["a"].append(a)
        if xl - xe < LENGTH:
            return _trace({f: np.array(v) for f, v in E.items()}, {f: np.array(v) for f, v in P.items()})
        if ve == 0.0 and vl == 0.0:
            return None
    return None


# ---------------------------------------------------------------------- single definitions
def test_reference_rho_and_start_rho_are_written_once():
    assert RHO_START == RHO_INSERT == max(RHOS) == 1.0 and RHO_REF == RHO_START
    assert classification_rho(True, 0.6) == 0.6 and classification_rho(True, 1.0) == 1.0
    # Baseline 2 has no rho of its own: the reference rho, whatever cell it is compared with
    assert classification_rho(False) == classification_rho(False, 0.6) == RHO_REF
    with pytest.raises(ValueError):
        classification_rho(True)


@pytest.mark.parametrize("arm, shield_on, rho", ARMS)
def test_borderline_start_validity_gives_the_same_label_in_both_arms(arm, shield_on, rho):
    """Gap exactly at d_min_long(RHO_START): correct; 1e-6 m below: incorrect; a gap that is safe
    only at rho = 0.6: incorrect in every arm - the start rule does not depend on the arm."""
    v = 20.0
    exact = d_min_long(v, v, RHO_START)
    assert exact == pytest.approx(62.625)
    cases = [(exact, False), (exact - 1e-6, True), (exact + 1e-6, False),
             (d_min_long(v, v, 0.6) + 0.01, True)]
    for gap, violated in cases:
        assert start_violation(gap_front=gap, v_front=v, v_ego=v) is violated
        assert start_violation(gap_rear=gap, v_rear=v, v_ego=v) is violated
        # the same label from the generator (ego-lane rule) on a real scene
        road = make_road()
        ego = add_ego(road, 1, 100.0, v, target_speed=v, initial_active="IDLE")
        add_scripted(road, 1, 100.0 + LENGTH + gap, v, "L")
        add_scripted(road, 1, 100.0 - LENGTH - 2.0 * exact, v, "F")
        label = {r: initially_unsafe(ego, road, r, adjacent=False) for r in RHOS}
        assert correct_start(dict(unsafe_lane=label)) is (not violated)
    # the label of the cell rho is descriptive only: safe at 0.6, incorrect start all the same
    assert label[0.6] is False and label[1.0] is True
    assert h1.classification_rho(shield_on, rho) == (rho if shield_on else RHO_REF)


# ---------------------------------------------------------------------- event at the rho threshold
def test_reaction_exactly_at_t_b_plus_rho_is_in_time():
    for rho in (0.6, 1.0):
        n = int(round(rho / DT))
        assert reaction_late(10, 10 + n, rho, 99) is False        # exactly at t_b + rho
        assert reaction_late(10, 10 + n + 1, rho, 99) is True     # one frame later
        assert reaction_late(10, None, rho, 10 + n) is False      # contact before the deadline
        assert reaction_late(10, None, rho, 10 + n + 1) is True   # no reaction at all


@pytest.mark.parametrize("arm, shield_on, rho", ARMS)
def test_event_exactly_at_the_rho_threshold(arm, shield_on, rho):
    """Leader brakes with -8 m/s^2, the ego accelerates until it reacts. Reaction exactly at
    t_b + rho_cls: not CR_ego_op (documented label: CR_unc_op, no H1 event). One frame later:
    CR_ego_op (reaction), an H1 event. rho_cls = rho of the filter / RHO_REF for Baseline 2."""
    rho_cls = classification_rho(shield_on, rho)
    n = int(round(rho_cls / DT))
    on_time, late = follow(rho_cls, n), follow(rho_cls, n + 1)
    assert on_time is not None and late is not None
    for tr, delay in ((on_time, n), (late, n + 1)):
        k_b = danger_onset(tr, rho_cls)
        assert k_b == 1 and reaction_start(tr, k_b) == k_b + delay
    assert classify_collision(on_time, rho_cls) == CR_UNC
    assert classify_collision(late, rho_cls) == CR_EGO_REACTION
    assert is_h1_event(True, True, classify_collision(on_time, rho_cls)) is False
    assert is_h1_event(True, True, classify_collision(late, rho_cls)) is True
    assert is_h1_event(False, True, CR_EGO_REACTION) is False     # incorrect start: no H1 event


def test_baseline_2_is_classified_at_the_reference_rho_only():
    """The same collision (reaction 0.8 s after the onset at rho = 0.6) is late for a filter with
    rho = 0.6. For Baseline 2 the label comes from RHO_REF alone and does not change with the
    Proposed cell it is compared with."""
    tr = follow(0.6, 8)
    assert classify_collision(tr, classification_rho(True, 0.6)) == CR_EGO_REACTION
    labels = {classify_collision(tr, classification_rho(False, cell)) for cell in (0.6, 1.0)}
    assert len(labels) == 1


# ---------------------------------------------------------------------- order of the rules
def test_rules_are_applied_in_the_documented_order():
    rho = 1.0
    # 1: danger from t = 0
    assert classify_collision(follow(rho, 10_000, margin=-1.0), rho) == CR_INIT
    # 3: the leader did not brake, the ego ran into it
    assert classify_collision(follow(rho, 10_000, a_lead=0.0, a_before=3.0), rho) == CR_EGO_MANEUVER
    # 3: a lane change of the ego is in progress at t_b (rule 2 needs the supervisor)
    tr = follow(rho, 10)
    tr.lc[:] = True
    assert classify_collision(tr, rho) == CR_EGO_MANEUVER
    # 2: the partner started to move laterally after the ego lane change was issued and the
    # supervisor reacted by the next tick
    tr = follow(rho, 10)
    tr.lc[:] = True
    tr.partner["vy"][1:] = 1.0
    tr.lc_ticks, tr.sup_ticks = (0,), (2,)
    assert classify_collision(tr, rho) == CR_ENV_NEIGHBOUR
    tr.sup_ticks = (8,)                                   # too late: stays with the ego
    assert classify_collision(tr, rho) == CR_EGO_MANEUVER
    # 4: the leader brakes harder than -8 m/s^2
    assert classify_collision(follow(rho, 10, a_lead=-12.0), rho) == CR_ENV_BOUNDS
    # 4: the partner had crashed before
    tr = follow(rho, 10)
    tr.partner_crashed[3:-1] = True
    assert classify_collision(tr, rho) == CR_ENV_BOUNDS
    # 5: weak braking after a reaction in time
    assert classify_collision(follow(rho, 5, a_brake=-2.0), rho) == CR_EGO_REACTION
    # 5: rho below tau_eff
    assert TAU_EFF == pytest.approx(0.4)
    assert classify_collision(follow(0.2, 4), 0.2) == CR_EGO_REACTION   # reaction after tau_eff
    # 6: hit from behind; the ego (leader of the pair) brakes within the bounds
    front = follow(rho, 10_000, a_lead=-6.0)
    rear = _trace(dict(front.partner), dict(front.ego))
    assert classify_collision(rear, rho) == CR_ENV_REAR


def test_cr_total_does_not_depend_on_the_categories():
    eps = [dict(crashed=True, correct_start=True, crash_category=CR_EGO_REACTION, h1_event=True, rho_cls=1.0),
           dict(crashed=True, correct_start=True, crash_category=CR_ENV_REAR, h1_event=False, rho_cls=1.0),
           dict(crashed=True, correct_start=False, crash_category=CR_INIT, h1_event=False, rho_cls=1.0),
           dict(crashed=False, correct_start=True, crash_category=None, h1_event=False, rho_cls=1.0),
           dict(crashed=False, correct_start=False, crash_category=None, h1_event=False, rho_cls=1.0)]
    s = h1_summary(eps)
    assert (s["n"], s["n_valid"], s["crashes"], s["crashes_valid"], s["h1_events"]) == (5, 3, 3, 2, 1)
    assert s["CR_total"] == pytest.approx(0.6) and s["CR_total_valid"] == pytest.approx(2 / 3)
    assert s["CR_ego_op"] == pytest.approx(1 / 3) and s["rho_cls"] == 1.0
    assert sum(s["categories"].values()) == s["crashes"]
    relabeled = [dict(e, crash_category=CR_UNC if e["crashed"] else None, h1_event=False) for e in eps]
    again = h1_summary(relabeled)
    assert (again["CR_total"], again["CR_total_valid"]) == (s["CR_total"], s["CR_total_valid"])
    assert set(CR_EGO) == {CR_EGO_MANEUVER, CR_EGO_REACTION}


def test_runner_applies_one_start_rule_and_the_classification_rho_of_the_arm():
    """The same start states for Baseline 2 (no filter) and Proposed (rho = 0.6): the same
    correct-start label, classification at RHO_REF / at the rho of the filter, a category for
    every collision, and CR_total available without the classification."""
    env = make_env(D=25, duration=20, warmup=2.0)
    agent = DQN(seed=3)
    crashes = 0
    for seed in range(21, 27):
        b = run_episode(env, agent, shield_on=False, reward_kind="RS", seed=seed, classify=True)
        start_b = correct_start(env.start_info)
        p = run_episode(env, agent, rho=0.6, seed=seed, classify=True)
        assert b["rho_cls"] == RHO_REF == 1.0 and p["rho_cls"] == 0.6
        assert b["correct_start"] == p["correct_start"] == start_b == correct_start(env.start_info)
        for c in (b, p):
            assert (c["crash_category"] in CATEGORIES) == c["crashed"]
            assert c["h1_event"] == is_h1_event(c["correct_start"], c["crashed"], c["crash_category"])
        crashes += b["crashed"]
    assert crashes >= 1                                   # the classification was exercised
    c = run_episode(env, agent, shield_on=False, seed=21)
    assert c["crash_category"] is None and c["h1_event"] is None and c["rho_cls"] == RHO_REF
    assert isinstance(c["crashed"], bool) and isinstance(c["correct_start"], bool)
