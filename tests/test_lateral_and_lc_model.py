"""Regression tests for the lateral rule shared by the filter and
the danger label, and the prediction model of vehicles in the lane-change lanes (protocol 7.7).

LR1 - neighbour centred in the adjacent lane inside d_min_long, no lateral motion: at rho = 1.0
      d_min_lat(0, 0) = 2.75 m > 2.0 m (side gap of adjacent lanes), but there is no lateral
      approach -> no dangerous situation: IDLE admissible, danger label 0 (the strict RSS rule
      lateral_rule="rss" is kept as a diagnostic only);
LR2 - the same neighbour moving towards the ego -> IDLE forbidden, danger label 1;
LR3 - rear vehicle in the target lane (case D = 10, rho = 0.1): the worst-case
      prediction forbids the lane change, the nominal prediction of the main series admits it;
LR4 - nominal prediction refuted by observation (the rear vehicle accelerates after the start):
      re-check every tick -> ABORT before the crossing, no contact;
LR5 - the same, but the violation becomes visible only after the crossing: FALLBACK; an
      RSS-compliant follower avoids contact, a non-reacting one hits the ego from behind
      (contact by the rear vehicle, CR_env): documented limitation of the nominal model;
LR6 - T2b-i under the nominal prediction: no ABORT is needed (gap 40 m >= d_min_long 39.42 m),
      the lane change is completed without a dangerous situation;
LR7 - ShieldConfig rejects unknown model names.
"""
import pytest

from rss_shield.params import LENGTH, ABORT, FALLBACK, ACTION_INDEX
from rss_shield.rss import d_min_long, d_min_lat
from rss_shield import scenarios as S
from rss_shield.sim import make_road, add_ego, add_scripted, Runner, shielded, _danger
from rss_shield.shield import RSSShield, ShieldConfig, MAIN_SERIES

LL, IDLE, LR = ACTION_INDEX["LANE_LEFT"], ACTION_INDEX["IDLE"], ACTION_INDEX["LANE_RIGHT"]


def _first(rows, cond):
    return next((row["frame"] for row in rows if cond(row)), None)


def _adjacent_scene(rho, heading, v=25.0):
    """Ego in lane 1; N centred in lane 2, ahead at half of d_min_long, same speed."""
    road = make_road()
    ego = add_ego(road, 1, S.X0, v, target_speed=v, initial_active="IDLE")
    n = add_scripted(road, 2, S.X0 + LENGTH + 0.5 * d_min_long(v, v, rho), v, "N")
    n.heading = heading                     # heading < 0: lateral speed towards the ego
    return road, ego, n


def _rear_accelerating(gap, acc, lc_prediction="nominal", follow=None, rho=0.4, T=6.0):
    """Ego lane 1 -> lane 2 (LANE_RIGHT at tick 0), 20 m/s; R in lane 2, `gap` m behind (bumper
    gap), 20 m/s, accelerating with `acc` from t = 0; follow: None (non-reacting) or the response
    time of an RSS-compliant follower."""
    road = make_road()
    ego = add_ego(road, 1, S.X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 2, S.X0 - LENGTH - gap, 20.0, "R", accel=lambda t: acc, rss_follow=follow)
    sh = RSSShield(ego, ShieldConfig(rho=rho, **dict(MAIN_SERIES, lc_prediction=lc_prediction)))
    m0 = sh.mask()[0]
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_RIGHT" if k == 0 else "IDLE"), rho,
               name=f"L_{lc_prediction}")
    r.run(T, stop_after_crash=0.5)
    return m0, r


# -------------------------------------------------------------- LR1, LR2: lateral rule
def test_LR1_centred_adjacent_neighbour_is_not_dangerous():
    rho = 1.0
    assert d_min_lat(0.0, 0.0, rho) == pytest.approx(2.75)          # > 2.0 m: strict RSS flags it
    road, ego, n = _adjacent_scene(rho, 0.0)
    m, info = RSSShield(ego, ShieldConfig(rho=rho, **MAIN_SERIES)).mask()
    assert m[IDLE] == 1 and info["IDLE"] == {}
    lab = _danger(ego, [n], rho)
    assert lab["side_min"] == pytest.approx(2.0) and not lab["danger"]
    road, ego, n = _adjacent_scene(rho, 0.0)
    m, info = RSSShield(ego, ShieldConfig(rho=rho, **dict(MAIN_SERIES, lateral_rule="rss"))).mask()
    assert m[IDLE] == 0 and info["IDLE"]["front"]["side"] == pytest.approx(2.0)


@pytest.mark.parametrize("model", ["observed", "occupancy"])
def test_LR2_approaching_adjacent_neighbour_forbids_idle(model):
    rho = 1.0
    road, ego, n = _adjacent_scene(rho, -0.05)                      # ~1.25 m/s towards the ego
    cfg = ShieldConfig(rho=rho, adjacent_lane_model=model, lc_prediction="nominal")
    m, info = RSSShield(ego, cfg).mask()
    assert m[IDLE] == 0 and info["IDLE"]["front"]["vehicle"] == "N"
    assert _danger(ego, [n], rho)["danger"]


# ------------------------------------------------------------ LR3-LR6: lc prediction
def test_LR3_rear_vehicle_in_target_lane_worst_case_vs_nominal():
    v, rho, gap = 23.6, 0.1, 54.0
    assert gap > d_min_long(v, v, rho) + 10.0                       # 38.97 m: the gap is safe now
    res = {}
    for lcp in ("worst_case", "nominal"):
        road = make_road()
        ego = add_ego(road, 1, S.X0, v, target_speed=v, initial_active="IDLE")
        add_scripted(road, 0, S.X0 - LENGTH - gap, v, "R")
        m, info = RSSShield(ego, ShieldConfig(rho=rho, **dict(MAIN_SERIES, lc_prediction=lcp))).mask()
        res[lcp] = (int(m[LL]), info["LANE_LEFT"])
    assert res["worst_case"][0] == 0 and res["worst_case"][1]["rear"]["vehicle"] == "R"
    assert res["nominal"] == (1, {})


def test_LR4_nominal_prediction_refuted_by_observation_gives_abort():
    m0, r = _rear_accelerating(44.0, 2.0, follow=0.4)
    rows = r.rows
    assert m0[LR] == 1                                               # admissible at t = 0
    sup = [(row["frame"], row["sup_cmd"]) for row in rows if row["decision_tick"] and row["sup_cmd"]]
    assert sup == [(4, ABORT)]
    assert all(row["lane_index"] == 1 for row in rows)               # the marking is never crossed
    assert not any(row["crashed"] for row in rows)
    assert rows[-1]["y_end"] == pytest.approx(4.0, abs=0.05)
    m0_wc, _ = _rear_accelerating(44.0, 2.0, lc_prediction="worst_case", follow=0.4, T=0.2)
    assert m0_wc[LR] == 0                                            # worst case: not admissible


def test_LR5_nominal_prediction_relies_on_rss_compliant_follower():
    for follow, contact in ((0.4, False), (None, True)):
        m0, r = _rear_accelerating(45.0, 2.0, follow=follow, T=8.0)
        rows = r.rows
        assert m0[LR] == 1
        cross = _first(rows, lambda row: row["lane_index"] == 2)
        fb = _first(rows, lambda row: row["sup_cmd"] == FALLBACK)
        assert cross is not None and fb is not None
        assert ABORT not in {row["sup_cmd"] for row in rows}
        hit = _first(rows, lambda row: row["crashed"])
        assert (hit is not None) == contact
        if contact:                                                  # rear-end by R: CR_env
            row = rows[hit]
            assert row["R_lane"] == row["lane_index"] == 2 and row["R_x"] < row["x_end"]
    m0_wc, _ = _rear_accelerating(45.0, 2.0, lc_prediction="worst_case", T=0.2)
    assert m0_wc[LR] == 0


def test_LR6_T2b_i_under_nominal_prediction():
    assert d_min_long(20.0, 20.0, 0.4) == pytest.approx(39.42, abs=1e-2)
    rows = S.scenario2b("i", model="observed", lc_prediction="nominal").rows
    assert not any(row["sup_cmd"] for row in rows)                  # gap 40 m >= 39.42 m: no ABORT
    assert _first(rows, lambda row: row["lane_index"] == 2) is not None
    assert not any(row["crashed"] for row in rows) and not any(row["danger"] for row in rows)


def test_LR7_shield_config_rejects_unknown_models():
    for kw in (dict(lc_prediction="optimistic"), dict(lateral_rule="none"),
               dict(adjacent_lane_model="all"), dict(long_prediction="x")):
        with pytest.raises(ValueError):
            ShieldConfig(rho=0.4, **kw)
    cfg = ShieldConfig(rho=0.4, **MAIN_SERIES)
    assert (cfg.adjacent_lane_model, cfg.lc_prediction, cfg.lateral_rule) == (
        "observed", "nominal", "approach")
