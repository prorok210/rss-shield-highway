"""RSS-distance safety filter (mask, TAKEOVER, lane-change supervision).

Definition of admissibility (protocol 7.2): an action a chosen at decision tick t_k becomes
active on [t_{k+1}, t_{k+2}) because of the FIFO delay; on [t_k, t_{k+1}) the already queued
command acts. a is admissible if on every predicted physics frame of the horizon no RSS
dangerous situation is created, i.e. there is no neighbour for which BOTH
    longitudinal gap < d_min_long   and   lateral clearance < d_min_lat.
Vehicles ahead are always checked; vehicles behind only in lanes the ego is entering.
Horizon: 2 decision periods for lane keeping; 2 decision periods + LC_HORIZON for lane
changes and while a maneuver is in progress.

Queue consistency (protocol v11, 7.2). The lateral state used by the mask and by the
supervision is the state at the activation moment of the command chosen now, i.e. after every
command already waiting in the FIFO has left it (plan_state(): ghost run for k_tau frames
without a new command). Consequences:
  * a lane change that is queued but not yet active is a maneuver in progress: LANE_LEFT /
    LANE_RIGHT are not offered again until it is completed (no second lane change before the
    first one is active), and the other actions are checked with the lane-change horizon and
    the vehicles behind in its target lane;
  * the target lane of a new lane change is counted from the target lane after the queue;
  * ABORT is offered only if the return becomes active before the lane marking is crossed:
    the command chosen now leaves the FIFO after k_tau frames, so the check uses the maneuver
    state at that moment, not the current one. The vehicle ignores an ABORT that arrives
    after crossing (DelayedRSSVehicle.abort_ignored), protocol 7.4, item 3.

Neighbour zone (protocol v11, 7.4). A neighbour is discarded before prediction only if an RSS
violation with it is impossible on the horizon of N frames (necessary condition; complete for
the prediction model below and the actual ego code: ego acceleration <= A_MAX_ACCEL, every
predicted neighbour acceleration <= A_MAX_ACCEL, v >= 0):
    ahead  (dx0 >= 0): violation possible only if  dx0 <= reach(v_ego, N)
    behind (dx0 <  0): violation possible only if -dx0 <= reach(v_neighbour, N)
    reach(v, N) = L + d_min_long(v + A_MAX_ACCEL*N*dt, 0, rho) + N*dt*v + A_MAX_ACCEL*dt^2*N(N-1)/2
(dx0 - current centre distance; d_min_long grows with the rear speed and is largest for a
standing front vehicle; the last two terms bound the distance the rear vehicle of the pair can
gain in N frames). A fixed zone (ShieldConfig.zone) is accepted only if it is not smaller than
reach(V_MAX, N_lc) for the longest horizon, V_MAX = highway-env Vehicle.MAX_SPEED.

Neighbour prediction model (ShieldConfig):
  long_prediction = "nominal"    - vehicles in the ego's own lane keep their current speed.
                                   RSS reactive scheme: leader braking is covered by d_min,
                                   reaction comes after tau_eff = tau + dt_decision
                                   (this is the model behind table 2 of the protocol).
                  = "worst_case" - own-lane leader brakes with -A_MAX_BRAKE on the horizon.
  Vehicles relevant to a lane change (target lane and lanes adjacent to it) are always
  predicted with worst-case bounds: ahead -A_MAX_BRAKE, behind +A_MAX_ACCEL (v >= 0).
  adjacent_lane_model = "occupancy" - every vehicle in a lane adjacent to the target lane is
                                      treated as occupying the target lane (v6 text, 7.4);
                      = "observed"  - only if it already moves laterally towards the target
                                      lane (|v_lat| > V_LAT_OBS).

Supervision of an ongoing maneuver (hierarchy of section 7.4), checked at every tick:
  if some action is admissible -> normal choice;
  else (1) before crossing (at the activation moment of the return): ABORT if the return is
           admissible (origin lane leader AND rear);
       (2) TAKEOVER (brake, keep lateral target) if completion with braking is admissible;
       (4) FALLBACK: brake in [-8,-4] and hold the lane containing the ego centre.
In lane keeping an empty safe set gives TAKEOVER. Takeover is released at the first tick
with a non-empty safe set.

Decision.intervention (TAKEOVER, ABORT, FALLBACK) marks a supervisory decision (intercept
step, excluded from the exposure E); Decision.brake marks the commands that switch the vehicle
to forced braking (TAKEOVER, FALLBACK). ABORT is an intervention without forced braking.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .params import (DT, FRAMES_PER_DECISION, A_MAX_ACCEL, A_MAX_BRAKE, ACTIONS, ACTION_INDEX,
                     TAKEOVER, ABORT, FALLBACK, SUPERVISORY, LENGTH, WIDTH, LANE_WIDTH, V_MAX)
from .rss import d_min_long, d_min_lat

V_LAT_OBS = 0.1   # m/s, lateral speed treated as an observed lane change


def lc_frames_for(lc_horizon):
    """Prediction horizon (physics frames) of lane changes and of an ongoing maneuver."""
    return 2 * FRAMES_PER_DECISION + int(round(lc_horizon / DT))


def path_bound(v, n):
    """Upper bound of the distance covered in n frames from speed v >= 0 with acceleration
    <= A_MAX_ACCEL (explicit Euler, position first, as in highway-env)."""
    return n * DT * v + 0.5 * A_MAX_ACCEL * DT ** 2 * n * (n - 1)


def reach(v, n, rho):
    """Centre distance beyond which a pair cannot violate RSS within n frames.
    v - current speed of the rear vehicle of the pair (ego for a neighbour ahead, the
    neighbour for a vehicle behind)."""
    v = max(0.0, float(v))
    return LENGTH + d_min_long(v + A_MAX_ACCEL * n * DT, 0.0, rho) + path_bound(v, n)


def complete_zone(rho, lc_horizon=2.0, v_max=V_MAX):
    """Smallest fixed zone that is complete for all speeds <= v_max on the longest horizon."""
    return reach(v_max, lc_frames_for(lc_horizon), rho)


@dataclass
class ShieldConfig:
    rho: float
    long_prediction: str = "nominal"
    adjacent_lane_model: str = "occupancy"
    lc_horizon: float = 2.0
    # None: per-vehicle complete radius reach() (default); a number: fixed radius, m, which must
    # not be smaller than complete_zone(rho, lc_horizon) (v11: 150 m was incomplete)
    zone: Optional[float] = None
    eps: float = 1e-9

    def __post_init__(self):
        if self.zone is not None:
            need = complete_zone(self.rho, self.lc_horizon)
            if self.zone < need - 1e-9:
                raise ValueError(f"zone = {self.zone} m is incomplete: rho = {self.rho} s and "
                                 f"v <= {V_MAX} m/s need >= {need:.3f} m (or zone=None)")


@dataclass
class Decision:
    command: object
    a_pi: object
    a_exec: object
    mask: np.ndarray
    takeover: bool = False        # legacy name: True for every supervisory decision
    fallback: bool = False
    reason: str = ""
    violations: dict = field(default_factory=dict)

    @property
    def intervention(self):
        """Decision made by the supervisor (TAKEOVER / ABORT / FALLBACK): intercept step."""
        return self.command in SUPERVISORY

    @property
    def brake(self):
        """Command switches the vehicle to forced braking [-A_MAX_BRAKE, -A_MIN_BRAKE]."""
        return self.command in (TAKEOVER, FALLBACK)


class _NeighbourPrediction:
    """Neighbour prediction: constant longitudinal acceleration a (v >= 0, explicit Euler,
    position first, as in highway-env) and lateral motion towards y_goal.

    y_mode = "const"  - keeps current y;
           = "vel"    - current lateral speed, stopped at the centre of the lane it moves to;
           = "jump"   - worst case of the occupancy model: already at the target lane centre.
    """

    def __init__(self, veh, accel, y_mode="vel", y_goal=None):
        self.v0 = float(veh.speed)
        self.x0 = float(veh.position[0])
        self.y0 = float(veh.position[1])
        self.vy = float(veh.speed * np.sin(veh.heading))
        self.a = accel
        self.y_mode = y_mode
        self.y_goal = y_goal

    def at(self, n):
        x, v = self.x0, self.v0
        for _ in range(n):
            x += v * DT
            v = max(0.0, v + self.a * DT)
        if self.y_mode == "jump":
            return x, self.y_goal, v, 0.0
        if self.y_mode == "const" or abs(self.vy) <= V_LAT_OBS:
            return x, self.y0, v, 0.0
        y = self.y0 + self.vy * n * DT
        goal = self.y_goal
        if goal is not None and (y - goal) * np.sign(self.vy) > 0:
            return x, goal, v, 0.0
        return x, y, v, self.vy


class RSSShield:
    def __init__(self, ego, cfg: ShieldConfig):
        self.ego = ego
        self.cfg = cfg
        self.plan = None          # lateral state after the queue, set by mask()
        self.filter_on = True     # False only in tests: disables the zone pre-filter

    # ------------------------------------------------------------- helpers
    @property
    def road(self):
        return self.ego.road

    @property
    def lc_frames(self):
        return lc_frames_for(self.cfg.lc_horizon)

    def _lane_id(self, lane_index):
        return lane_index[2]

    def _n_lanes(self):
        return len(self.road.network.graph[self.ego.lane_index[0]][self.ego.lane_index[1]])

    def _lanes_overlapped(self, y):
        """Lane ids whose area intersects the ego body (straight road, lane i centred at 4*i)."""
        ids = set()
        for i in range(self._n_lanes()):
            if abs(y - LANE_WIDTH * i) < LANE_WIDTH / 2 + WIDTH / 2 - 1e-6:
                ids.add(i)
        return ids

    def _others(self, n_frames):
        """Neighbours for which a violation within n_frames is possible (complete filter)."""
        x0 = self.ego.position[0]
        others = [v for v in self.road.vehicles if v is not self.ego]
        if not self.filter_on:
            return others
        if self.cfg.zone is not None:
            return [v for v in others if abs(v.position[0] - x0) <= self.cfg.zone]
        r_front = reach(self.ego.speed, n_frames, self.cfg.rho) + self.cfg.eps
        out = []
        for v in others:
            dx = v.position[0] - x0
            if dx >= 0:
                if dx <= r_front:
                    out.append(v)
            elif -dx <= reach(v.speed, n_frames, self.cfg.rho) + self.cfg.eps:
                out.append(v)
        return out

    def plan_state(self):
        """Ego ghost at the activation moment of a command chosen now: every command already
        in the FIFO has left it (k_tau frames), the new command has not acted yet."""
        g = self.ego.ghost()
        for _ in range(len(g.fifo)):
            g.act()
            g.step(DT)
        return g

    def _simulate_ego(self, commands, n_frames):
        """Predict ego with the real vehicle code (FIFO + controller + limiter)."""
        g = self.ego.ghost()
        states = []
        for f in range(n_frames):
            if f % FRAMES_PER_DECISION == 0:
                k = f // FRAMES_PER_DECISION
                if k < len(commands) and commands[k] is not None:
                    g.act(commands[k])
            g.act()
            g.step(DT)
            states.append((float(g.position[0]), float(g.position[1]), float(g.speed),
                           float(g.speed * np.sin(g.heading)), self._lane_id(g.lane_index),
                           self._lane_id(g.target_lane_index)))
        return states

    # ------------------------------------------------------------- check
    def check_plan(self, commands, n_frames, lc_lanes):
        """Violations of the ego command sequence on n_frames (empty dict = admissible).

        A predicted frame violates RSS w.r.t. a neighbour when BOTH the longitudinal and the
        lateral distances are below d_min (RSS dangerous situation, protocol 7.7):
          * neighbour ahead  -> always checked (ego must keep d_min to the front);
          * neighbour behind -> checked only if it is in / predicted to occupy a lane of
            lc_lanes, i.e. a lane the ego is entering (lane change target or return lane);
            in lane keeping ego is not responsible for the vehicle behind it.
        lc_lanes - lanes of the lane change / return; vehicles occupying them are predicted
                   with worst-case bounds (ahead -A_MAX_BRAKE, behind +A_MAX_ACCEL).
        """
        cfg = self.cfg
        ego_states = self._simulate_ego(commands, n_frames)
        lc_lanes = set(lc_lanes)
        preds = []
        for o in self._others(n_frames):
            lane_o = self._lane_id(o.lane_index)
            vy = o.speed * np.sin(o.heading)
            moving = abs(vy) > V_LAT_OBS
            ahead = o.position[0] >= self.ego.position[0]
            occ = {lane_o}
            y_mode, y_goal = "vel", None
            if moving:
                y_goal = LANE_WIDTH * round((o.position[1] + np.sign(vy) * (LANE_WIDTH / 2 - 1e-6)) / LANE_WIDTH)
                if abs(o.position[1] - y_goal) < 1e-6:
                    y_goal = o.position[1] + np.sign(vy) * LANE_WIDTH
            goal_lane = int(round(y_goal / LANE_WIDTH)) if moving else None
            if moving:
                occ.add(goal_lane)
            for tl in lc_lanes:
                if tl in occ:
                    continue
                if abs(lane_o - tl) == 1 and cfg.adjacent_lane_model == "occupancy" and not moving:
                    occ.add(tl)
                    y_mode, y_goal = "jump", LANE_WIDTH * tl
            if occ & lc_lanes:
                a = -A_MAX_BRAKE if ahead else A_MAX_ACCEL
            elif cfg.long_prediction == "worst_case" and ahead:
                a = -A_MAX_BRAKE
            else:
                a = 0.0
            preds.append((o, _NeighbourPrediction(o, a, y_mode, y_goal), occ, ahead))
        viol = {}
        for n, (xe, ye, ve, vye, _, _) in enumerate(ego_states, start=1):
            for o, pred, occ, ahead in preds:
                if not ahead and not (occ & lc_lanes):
                    continue
                xo, yo, vo, vyo = pred.at(n)
                name = getattr(o, "name", str(id(o)))
                dx = xo - xe
                if dx >= 0:
                    gap, need = dx - LENGTH, d_min_long(ve, vo, cfg.rho)
                else:
                    gap, need = -dx - LENGTH, d_min_long(vo, ve, cfg.rho)
                if gap >= need - cfg.eps:
                    continue
                side = abs(yo - ye) - WIDTH
                sgn = np.sign(yo - ye) or 1.0
                need_lat = d_min_lat(vye * sgn, vyo * sgn, cfg.rho)
                if side < need_lat - cfg.eps:
                    key = "front" if dx >= 0 else "rear"
                    viol.setdefault(key, dict(frame=n, vehicle=name, gap=round(gap, 3),
                                              d_min_long=round(need, 3), side=round(side, 3),
                                              d_min_lat=round(need_lat, 3)))
        return viol

    # ------------------------------------------------------------- mask
    def _lc_target(self, action, lane):
        """Target lane of a lane change counted from `lane` (target lane after the queue)."""
        if action == "LANE_LEFT":
            return lane - 1 if lane - 1 >= 0 else None
        if action == "LANE_RIGHT":
            return lane + 1 if lane + 1 < self._n_lanes() else None
        return None

    def mask(self):
        m = np.zeros(5, dtype=int)
        info = {}
        lc_frames = self.lc_frames
        keep_frames = 2 * FRAMES_PER_DECISION
        g = self.plan_state()
        man = g.maneuver                      # maneuver in progress after the queue
        lane = self._lane_id(g.target_lane_index)
        self.plan = dict(maneuver=None if man is None else dict(man), target_lane=lane,
                         queued=[c for c in self.ego.fifo if c is not None])
        for idx, name in ACTIONS.items():
            if name in ("LANE_LEFT", "LANE_RIGHT"):
                if man is not None:
                    info[name] = {"busy": True, "maneuver": man["kind"],
                                  "queued": self.ego.maneuver is None}
                    continue
                tl = self._lc_target(name, lane)
                if tl is None:
                    info[name] = {"no_lane": True}
                    continue
                v = self.check_plan([name], lc_frames, {tl})
            else:
                if man is not None:
                    tl = self._lane_id(man["target"])
                    v = self.check_plan([name], lc_frames, {tl})
                else:
                    v = self.check_plan([name], keep_frames, set())
            m[idx] = int(not v)
            info[name] = v
        return m, info

    def decide(self, a_pi, q_values=None):
        """a_pi: unmasked policy choice (str). q_values: optional dict name->Q for argmax over A_safe."""
        m, info = self.mask()
        safe = [ACTIONS[i] for i in range(5) if m[i]]
        if safe:
            if a_pi in safe:
                a_exec = a_pi
            elif q_values:
                a_exec = max(safe, key=lambda n: q_values[n])
            else:
                a_exec = "IDLE" if "IDLE" in safe else safe[0]
            return Decision(a_exec, a_pi, a_exec, m, reason="safe", violations=info)
        man = self.plan["maneuver"]           # state at the activation moment of this decision
        lc_frames = self.lc_frames
        if man is not None and man["kind"] in ("lane_change", "return"):
            origin = self._lane_id(man["origin"])
            target = self._lane_id(man["target"])
            if man["kind"] == "lane_change":
                if man["crossed"]:
                    # the return would become active after the marking is crossed (7.4, item 3)
                    info["ABORT"] = {"after_crossing": True}
                else:
                    v = self.check_plan([ABORT], lc_frames, {origin})
                    info["ABORT"] = v
                    if not v:
                        return Decision(ABORT, a_pi, None, m, takeover=True, reason="return",
                                        violations=info)
            v = self.check_plan([TAKEOVER], lc_frames, {target})
            info["COMPLETE_BRAKE"] = v
            if not v:
                return Decision(TAKEOVER, a_pi, None, m, takeover=True, reason="complete",
                                violations=info)
            return Decision(FALLBACK, a_pi, None, m, takeover=True, fallback=True,
                            reason="fallback", violations=info)
        return Decision(TAKEOVER, a_pi, None, m, takeover=True, reason="empty_safe_set",
                        violations=info)
