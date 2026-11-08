"""Rewards of the protocol (7.1): R_v for Baseline 1 and Proposed, R_RS for Baseline 2."""
import numpy as np

from rss_shield.params import LENGTH

V_REF = 30.0          # R_v = clip(v_x / 30, 0, 1), v_x at the end of the decision step
TTC_MIN = 2.0         # s
P_TTC = 5.0
P_COLLISION = 20.0


def speed_reward(ego):
    return float(np.clip(ego.speed * np.cos(ego.heading) / V_REF, 0.0, 1.0))


def ttc(ego, road):
    """Minimal time to collision with the nearest vehicles ahead and behind in the ego lane:
    bumper-to-bumper gap / closing speed, inf if the vehicles do not close in."""
    front, rear = road.neighbour_vehicles(ego, ego.lane_index)
    lane = road.network.get_lane(ego.lane_index)
    s_e = lane.local_coordinates(ego.position)[0]
    vx_e = ego.speed * np.cos(ego.heading)
    out = np.inf
    if front is not None:
        gap = lane.local_coordinates(front.position)[0] - s_e - LENGTH
        closing = vx_e - front.speed * np.cos(front.heading)
        if closing > 0:
            out = min(out, max(gap, 0.0) / closing)
    if rear is not None:
        gap = s_e - lane.local_coordinates(rear.position)[0] - LENGTH
        closing = rear.speed * np.cos(rear.heading) - vx_e
        if closing > 0:
            out = min(out, max(gap, 0.0) / closing)
    return float(out)


def step_reward(kind, ego, road):
    """kind 'Rv': R_v; kind 'RS': R_v - 5 * 1(TTC < 2 s) - 20 * 1(collision). Returns (r, parts)."""
    rv = speed_reward(ego)
    if kind == "Rv":
        return rv, dict(rv=rv, ttc_pen=0.0, col_pen=0.0)
    if kind != "RS":
        raise ValueError(kind)
    t = ttc(ego, road)
    pen_ttc = P_TTC if t < TTC_MIN else 0.0
    pen_col = P_COLLISION if ego.crashed else 0.0
    return rv - pen_ttc - pen_col, dict(rv=rv, ttc=t, ttc_pen=pen_ttc, col_pen=pen_col)
