"""Scenario runner with a full per-physics-frame log (protocol 7.5, "Состав журнала").

Time convention: physics frame k covers [k*dt, (k+1)*dt). Columns t, a_ctrl, a refer to the
frame start / the acceleration applied during the frame; *_end columns are the state at the
end of the frame. Decision ticks are frames with k % 2 == 0 (t_k = 0.0, 0.2, ...).

Status columns (protocol 7.5): supervisory intervention, active command and forced
braking are logged separately. The decision set is defined once, in rss_shield.accounting.
  intervention  - 1 on both frames of a decision interval governed by a supervisory decision
                  (TAKEOVER / ABORT / FALLBACK): intercept step, not part of the exposure E;
  policy_step   - 1 on both frames of every other decision interval (policy command after the
                  mask, or no new command): a policy step, part of the exposure E;
                  both flags are set per decision interval (by the decision tick), and
                  policy_step + intervention = 1 on every frame;
  sup_cmd       - that supervisory command ("" otherwise);
  active_cmd    - command that left the FIFO last (acts now); active_sup - it is supervisory;
  brake_mode    - forced braking limiter [-8, -4] is on (TAKEOVER / FALLBACK active);
  return_active - a return to the origin lane (after ABORT) is in progress;
  abort_ignored - cumulative number of ABORT commands that started no return (after the
                  crossing, or without a lane change to return from).
There is no `takeover` column: Decision.takeover is a deprecated alias of Decision.intervention.
"""
import csv
import numpy as np
from highway_env.road.road import Road, RoadNetwork

from .accounting import INTERVENTION, step_kind
from .params import DT, FRAMES_PER_DECISION, K_TAU, LENGTH, WIDTH, ACTION_INDEX, SUPERVISORY
from .rss import d_min_long, d_min_lat
from .shield import lateral_danger
from .vehicle import DelayedRSSVehicle, ScriptedVehicle


def make_road(lanes=4, length=10000.0, seed=0):
    net = RoadNetwork.straight_road_network(lanes, length=length, speed_limit=40.0)
    return Road(network=net, np_random=np.random.RandomState(seed), record_history=False)


def lane(road, i):
    return road.network.get_lane(("0", "1", i))


def add_ego(road, lane_id, x, v, target_speed=None, initial_active=None, k_tau=K_TAU):
    ln = lane(road, lane_id)
    ego = DelayedRSSVehicle(road, ln.position(x, 0.0), heading=ln.heading_at(x), speed=v,
                            target_speed=target_speed, initial_active=initial_active, k_tau=k_tau)
    if target_speed is not None:
        ego.speed_index = int(np.argmin(np.abs(ego.target_speeds - target_speed)))
        ego.target_speed = float(ego.target_speeds[ego.speed_index])
    if initial_active in SUPERVISORY:          # takeover already in progress at t = 0
        ego.takeover = True
        ego.fallback = initial_active == "FALLBACK"
    road.vehicles.append(ego)
    return ego


def add_scripted(road, lane_id, x, v, name, accel=None, lane_change=None, rss_follow=None):
    ln = lane(road, lane_id)
    veh = ScriptedVehicle(road, ln.position(x, 0.0), heading=ln.heading_at(x), speed=v,
                          accel=accel, lane_change=lane_change, name=name, rss_follow=rss_follow)
    road.vehicles.append(veh)
    return veh


def _danger(ego, others, rho):
    """Actual RSS quantities at the end of a frame (nearest front / rear / lateral). danger: both
    thresholds violated, the lateral one with the operational condition of protocol 7.7
    (lateral overlap or lateral approach, shield.lateral_danger)."""
    out = dict(gap_front=np.nan, d_min_long_front=np.nan, gap_rear=np.nan,
               d_min_long_rear=np.nan, side_min=np.nan, d_min_lat_min=np.nan,
               danger=False, danger_with="")
    xe, ye = ego.position
    vye = ego.speed * np.sin(ego.heading)
    best_f, best_r, best_s = None, None, None
    for o in others:
        dx = o.position[0] - xe
        side = abs(o.position[1] - ye) - WIDTH
        sgn = np.sign(o.position[1] - ye) or 1.0
        vyo = o.speed * np.sin(o.heading)
        dlat = d_min_lat(vye * sgn, vyo * sgn, rho)
        if dx >= 0:
            gap, need = dx - LENGTH, d_min_long(ego.speed, o.speed, rho)
        else:
            gap, need = -dx - LENGTH, d_min_long(o.speed, ego.speed, rho)
        same_lane = side < 0
        if same_lane and dx >= 0 and (best_f is None or gap < best_f[0]):
            best_f = (gap, need)
        if same_lane and dx < 0 and (best_r is None or gap < best_r[0]):
            best_r = (gap, need)
        if abs(dx) < 50 and (best_s is None or side < best_s[0]):
            best_s = (side, dlat)
        # lateral part as in the filter and t_danger (protocol 7.7): overlap or lateral approach
        if gap < need and side < dlat and lateral_danger(side, (vye - vyo) * sgn):
            out["danger"] = True
            out["danger_with"] += getattr(o, "name", "?") + ("F" if dx >= 0 else "R") + ";"
    if best_f:
        out["gap_front"], out["d_min_long_front"] = best_f
    if best_r:
        out["gap_rear"], out["d_min_long_rear"] = best_r
    if best_s:
        out["side_min"], out["d_min_lat_min"] = best_s
    return out


class Runner:
    """controller(tick, t, runner) -> dict(command, a_pi, a_exec, mask, reason)."""

    def __init__(self, road, ego, controller, rho, name="run"):
        self.road, self.ego, self.controller, self.rho, self.name = road, ego, controller, rho, name
        self.rows = []
        self.t_contact = None
        self.last = dict(command="", a_pi="", a_exec="", mask="", reason="")
        self.sup = ""                        # supervisory command of the current decision interval
        self.pol = 1                         # 1 if the current decision interval is a policy step

    def run(self, T, stop=None, stop_after_crash=0.0):
        n = int(round(T / DT))
        ego = self.ego
        crash_frame = None
        for k in range(n):
            t = round(k * DT, 10)
            tick = k % FRAMES_PER_DECISION == 0
            dec = None
            if tick:
                dec = self.controller(k // FRAMES_PER_DECISION, t, self)
                if dec["command"] is not None:
                    ego.act(dec["command"])
                self.last = dec
                self.sup = dec["command"] if step_kind(dec["command"]) == INTERVENTION else ""
                self.pol = int(not self.sup)
            v_start = float(ego.speed)
            pending = ego.pending
            self.road.act()
            fifo_after = "|".join("" if c is None else str(c) for c in ego.fifo)
            self.road.step(DT)
            others = [v for v in self.road.vehicles if v is not ego]
            row = dict(frame=k, t=t, t_end=round(t + DT, 10), decision_tick=int(tick),
                       cmd_issued=(dec["command"] if dec else ""), pending=pending or "",
                       fifo_after_shift=fifo_after, active_cmd=ego.active_cmd or "",
                       a_pi=(dec.get("a_pi") or "") if dec else "",
                       a_exec=(dec.get("a_exec") or "") if dec else "",
                       mask=_mask_str(dec.get("mask")) if dec else "",
                       reason=(dec.get("reason") or "") if dec else "",
                       policy_step=self.pol, intervention=int(bool(self.sup)),
                       sup_cmd=self.sup, active_sup=int(ego.sup_active),
                       brake_mode=int(ego.brake_mode), return_active=int(ego.return_active),
                       fallback=int(ego.fallback), abort_ignored=int(ego.abort_ignored),
                       a_ctrl=ego.a_ctrl, a=ego.a_applied, limiter_active=int(ego.limiter_active),
                       v_start=v_start, v_end=float(ego.speed),
                       a_integrated=(float(ego.speed) - v_start) / DT,
                       x_end=float(ego.position[0]), y_end=float(ego.position[1]),
                       vlat_end=float(ego.speed * np.sin(ego.heading)), heading=float(ego.heading),
                       lane_index=ego.lane_index[2], target_lane_index=ego.target_lane_index[2],
                       target_speed=float(ego.target_speed),
                       maneuver=(ego.maneuver["kind"] if ego.maneuver else ""),
                       crashed=int(ego.crashed))
            for o in others:
                p = getattr(o, "name", "V")
                row.update({f"{p}_lane": o.lane_index[2], f"{p}_x": float(o.position[0]),
                            f"{p}_y": float(o.position[1]), f"{p}_v": float(o.speed),
                            f"{p}_a": float(o.action.get("acceleration", 0.0)),
                            f"{p}_vlat": float(o.speed * np.sin(o.heading)),
                            f"{p}_crashed": int(o.crashed)})
            row.update(_danger(ego, others, self.rho))
            self.rows.append(row)
            if ego.crashed and crash_frame is None:
                crash_frame = k
                self.t_contact = row["t_end"]
            if crash_frame is not None and (k - crash_frame) * DT >= stop_after_crash - 1e-9:
                break
            if stop is not None and stop(self):
                break
        return self

    def write_csv(self, path):
        keys = []
        for r in self.rows:
            for key in r:
                if key not in keys:
                    keys.append(key)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow({key: _fmt(r.get(key, "")) for key in keys})
        return path


def _mask_str(m):
    if m is None or (isinstance(m, str) and m == ""):
        return ""
    return "".join(str(int(x)) for x in m)


def _fmt(v):
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.6f}"
    if isinstance(v, np.ndarray):
        return "".join(str(int(x)) for x in v)
    return v


def scripted(schedule):
    """Controller issuing schedule(tick, t) -> command (no filter)."""
    def ctrl(tick, t, runner):
        c = schedule(tick, t)
        return dict(command=c, a_pi="", a_exec=c if c not in SUPERVISORY else "", mask="",
                    reason="scripted")
    return ctrl


def shielded(shield, policy, q_values=None):
    """Controller: policy(tick, t) -> a_pi (str); the shield returns the executed command."""
    def ctrl(tick, t, runner):
        a_pi = policy(tick, t)
        d = shield.decide(a_pi, q_values)
        runner.decisions = getattr(runner, "decisions", []) + [(tick, t, d)]
        return dict(command=d.command, a_pi=a_pi, a_exec=d.a_exec or "", mask=d.mask,
                    reason=d.reason)
    return ctrl
