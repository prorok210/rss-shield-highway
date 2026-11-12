"""RSS-distance safety filter (mask, TAKEOVER, lane-change supervision).

Definition of admissibility (protocol 7.2): an action a chosen at decision tick t_k becomes
active on [t_{k+1}, t_{k+2}) because of the FIFO delay; on [t_k, t_{k+1}) the already queued
command acts. a is admissible if on every predicted physics frame of the horizon no RSS
dangerous situation is created, i.e. there is no neighbour for which BOTH
    longitudinal gap < d_min_long   and   lateral clearance < d_min_lat,
where the lateral part follows the operational definition of protocol 7.7 (the same one as
t_danger and the start label): for a neighbour whose body overlaps the ego laterally (clearance
< 0, e.g. the ego lane) the longitudinal threshold alone decides; for any other neighbour the
lateral threshold counts only while the lateral clearance decreases (closing speed > V_LAT_OBS).
Without this condition d_min_lat(0, 0, rho) > LANE_WIDTH - WIDTH = 2 m for rho > ~0.82 s and two
vehicles driving side by side in the centres of adjacent lanes would be "dangerous" (IDLE
forbidden next to any adjacent-lane neighbour at rho = 1.0). lateral_rule="rss" restores the
strict RSS threshold (diagnostic only).
Vehicles ahead are always checked; vehicles behind only in lanes the ego is entering.
Horizon: 2 decision periods for lane keeping; 2 decision periods + LC_HORIZON for lane
changes and while a maneuver is in progress.

Queue consistency (protocol 7.2). The lateral state used by the mask and by the
supervision is the state at the activation moment of the command chosen now, i.e. after every
command already waiting in the FIFO has left it (plan_state(): ghost run for k_tau frames
without a new command). Consequences:
  * a lane change that is queued but not yet active is a maneuver in progress: LANE_LEFT /
    LANE_RIGHT are not offered again until it is completed (no second lane change before the
    first one is active), and the other actions are checked with the lane-change horizon and
    the vehicles behind in its target lane;
  * the target lane of a new lane change is counted from the target lane after the queue, and
    the lanes it enters are all lanes between the lane of the ego centre and that final target
    (maneuver_lanes(): the full traversed geometry, not only the first lane entered). The same
    holds for a maneuver that already spans several lanes (lane changes chained in a FIFO longer
    than one decision period, or issued without the filter): vehicles behind are checked in every
    lane it still has to enter;
  * ABORT is offered only if the return becomes active before the lane marking is crossed:
    the command chosen now leaves the FIFO after k_tau frames, so the check uses the maneuver
    state at that moment, not the current one. The vehicle ignores an ABORT that arrives
    after crossing (DelayedRSSVehicle.abort_ignored), protocol 7.4, item 3.

Neighbour zone (protocol 7.4). A neighbour is discarded before prediction only if an RSS
violation with it is impossible on the horizon of N frames (necessary condition; complete for
the prediction model below and the actual ego code: ego acceleration <= A_MAX_ACCEL, every
predicted neighbour acceleration <= A_MAX_ACCEL, v >= 0):
    ahead  (dx0 >= 0): violation possible only if  dx0 <= reach(v_ego, N)
    behind (dx0 <  0): violation possible only if -dx0 <= reach(v_neighbour, N)
    reach(v, N) = L + d_min_long(v + A_MAX_ACCEL*N*dt, 0, rho) + N*dt*v + A_MAX_ACCEL*dt^2*N(N-1)/2
(dx0 - current centre distance; d_min_long grows with the rear speed and is largest for a
standing front vehicle; the last two terms bound the distance the rear vehicle of the pair can
gain in N frames). The zone is therefore a function of the scenario envelope, not a constant:
  * default (zone=None): the per-vehicle radius reach() with the actual speeds - complete for
    every scene;
  * a fixed zone (ShieldConfig.zone) is accepted only if it is not smaller than
    reach(V_MAX, N_lc) for the longest horizon, V_MAX = highway-env Vehicle.MAX_SPEED
    (complete_zone(); 472.985 m at rho = 1.0 s). A smaller value raises ZoneError; ShieldConfig is
    frozen, so the zone cannot be shrunk after the validation;
  * a scenario generator that keeps traffic only in a window around the ego truncates the zone
    physically. Such a window is sound only for the declared envelope of speeds:
    envelope_zone(rho, v_ego_max, v_other_max) gives the required extent ahead / behind and
    require_zone() raises ZoneError if the window is smaller (experiments.generator validates
    its window on every reset and checks the speeds on every step).

Neighbour prediction model (ShieldConfig):
  long_prediction = "nominal"    - vehicles in the ego's own lane keep their current speed.
                                   RSS reactive scheme: leader braking is covered by d_min,
                                   reaction comes after tau_eff = tau + dt_decision
                                   (this is the model behind table 2 of the protocol).
                  = "worst_case" - own-lane leader brakes with -A_MAX_BRAKE on the horizon.
  Vehicles relevant to a lane change (in / predicted to occupy the target or return lane):
  lc_prediction = "worst_case" - bounds on the whole horizon: ahead -A_MAX_BRAKE, behind
                                 +A_MAX_ACCEL (v >= 0); default of ShieldConfig, the model behind
                                 rows T2a-Q2 of table 3 (conservative control variant);
                = "nominal"    - current speed, as the own-lane leader: braking of the vehicle
                                 ahead and acceleration of the vehicle behind within rho are
                                 already in d_min_long; the maneuver is re-checked at every tick
                                 (hierarchy below). Model of the main series (MAIN_SERIES), chosen
                                 before it: with "worst_case" +3 m/s^2 of
                                 the follower over 2.4 s on top of d_min blocks almost every lane
                                 change on the main generator.
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
Decision.takeover is a deprecated alias of Decision.intervention (DeprecationWarning on access);
it is not the forced braking of the vehicle (brake_mode).
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .params import (DT, FRAMES_PER_DECISION, A_MAX_ACCEL, A_MAX_BRAKE, ACTIONS, ACTION_INDEX,
                     TAKEOVER, ABORT, FALLBACK, SUPERVISORY, LENGTH, WIDTH, LANE_WIDTH, V_MAX)
from .rss import d_min_long, d_min_lat

V_LAT_OBS = 0.1   # m/s, lateral speed treated as an observed lane change / lateral approach

# Configuration of the main series and of the pilot (protocol 7.4). ShieldConfig defaults are the
# verification defaults of table 3 (scenarios set their models explicitly where needed).
MAIN_SERIES = dict(adjacent_lane_model="observed", lc_prediction="nominal", lateral_rule="approach")


def is_main_series(cfg):
    """cfg (ShieldConfig) is the configuration of the main series: MAIN_SERIES, nominal own-lane
    prediction, complete per-vehicle neighbour zone, default lane-change horizon."""
    return (all(getattr(cfg, k) == v for k, v in MAIN_SERIES.items())
            and cfg.long_prediction == "nominal" and cfg.zone is None and cfg.lc_horizon == 2.0)


def describe(cfg):
    """Short text of a ShieldConfig for logs: adjacent model / lane-change prediction / lateral rule."""
    zone = "reach" if cfg.zone is None else f"{cfg.zone:g}"
    return f"{cfg.adjacent_lane_model}/{cfg.lc_prediction}/{cfg.lateral_rule}/zone={zone}"


def lateral_danger(side, closing, rule="approach"):
    """Lateral part of an RSS dangerous situation, given side < d_min_lat (protocol 7.7).
    side    - lateral clearance between the bodies, m (< 0: lateral overlap, e.g. the same lane);
    closing - rate of decrease of the clearance, m/s (> 0: the vehicles approach laterally).
    rule = "approach": overlap, or approach faster than V_LAT_OBS (operational definition shared by
           the filter, t_danger and the start label); "rss": strict RSS, any side < d_min_lat."""
    if rule == "rss":
        return True
    return side < 0.0 or closing > V_LAT_OBS


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


class ZoneError(ValueError):
    """The neighbour zone (or the traffic window that truncates it) is smaller than the bound."""


def envelope_zone(rho, v_ego_max, v_other_max, lc_horizon=2.0):
    """Smallest sound neighbour zone of a scenario envelope: (ahead, behind) centre distances, m,
    for ego speeds <= v_ego_max and speeds of every other vehicle <= v_other_max."""
    n = lc_frames_for(lc_horizon)
    return reach(v_ego_max, n, rho), reach(v_other_max, n, rho)


def require_zone(ahead, behind, rho, v_ego_max, v_other_max, lc_horizon=2.0, what="zone"):
    """Raise ZoneError unless the extent (ahead, behind) covers envelope_zone(); returns the bound."""
    need_a, need_b = envelope_zone(rho, v_ego_max, v_other_max, lc_horizon)
    if ahead < need_a - 1e-9 or behind < need_b - 1e-9:
        raise ZoneError(f"{what} [-{behind:g}; +{ahead:g}] m is incomplete: rho = {rho} s, ego <= "
                        f"{v_ego_max} m/s, others <= {v_other_max} m/s need "
                        f"[-{need_b:.3f}; +{need_a:.3f}] m")
    return need_a, need_b


def lanes_entered(lane_from, lane_to):
    """Lane ids entered on the way from lane_from to lane_to: every lane after lane_from up to and
    including lane_to; {lane_to} when they coincide (reversal / return into the same lane)."""
    step = 1 if lane_to >= lane_from else -1
    return set(range(lane_from + step, lane_to + step, step)) or {lane_to}


def maneuver_lanes(man):
    """Lanes a maneuver (dict of DelayedRSSVehicle.maneuver) enters: full traversed geometry."""
    return lanes_entered(man["origin"][2], man["target"][2])


def complete_zone(rho, lc_horizon=2.0, v_max=V_MAX):
    """Smallest fixed zone that is complete for all speeds <= v_max on the longest horizon."""
    return reach(v_max, lc_frames_for(lc_horizon), rho)


@dataclass(frozen=True)
class ShieldConfig:
    rho: float
    long_prediction: str = "nominal"
    adjacent_lane_model: str = "occupancy"
    lc_prediction: str = "worst_case"
    lateral_rule: str = "approach"
    lc_horizon: float = 2.0
    # None: per-vehicle complete radius reach() (default); a number: fixed radius, m, which must
    # not be smaller than complete_zone(rho, lc_horizon) (a fixed 150 m is incomplete)
    zone: Optional[float] = None
    eps: float = 1e-9

    def __post_init__(self):
        for name, allowed in (("long_prediction", ("nominal", "worst_case")),
                              ("adjacent_lane_model", ("occupancy", "observed")),
                              ("lc_prediction", ("nominal", "worst_case")),
                              ("lateral_rule", ("approach", "rss"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} = {getattr(self, name)!r}, expected one of {allowed}")
        if self.zone is not None:
            need = complete_zone(self.rho, self.lc_horizon)
            if self.zone < need - 1e-9:
                raise ZoneError(f"zone = {self.zone} m is incomplete: rho = {self.rho} s and "
                                f"v <= {V_MAX} m/s need >= {need:.3f} m (or zone=None)")


@dataclass
class Decision:
    command: object
    a_pi: object
    a_exec: object
    mask: np.ndarray
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

    @property
    def takeover(self):
        """Deprecated alias of `intervention`: True for TAKEOVER, ABORT and FALLBACK, not only for
        the forced braking of TAKEOVER. Use `intervention` (intercept step) or `brake` (forced
        braking, brake_mode of the vehicle)."""
        warnings.warn("Decision.takeover is deprecated: use Decision.intervention (supervisory "
                      "decision) or Decision.brake (forced braking)", DeprecationWarning, stacklevel=2)
        return self.intervention


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
                   by cfg.lc_prediction (worst_case: ahead -A_MAX_BRAKE, behind +A_MAX_ACCEL;
                   nominal: current speed).
        Lateral part: lateral_danger(side, closing, cfg.lateral_rule).
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
                if cfg.lc_prediction == "worst_case":
                    a = -A_MAX_BRAKE if ahead else A_MAX_ACCEL
                else:
                    a = 0.0
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
                closing = (vye - vyo) * sgn
                if side < need_lat - cfg.eps and lateral_danger(side, closing, cfg.lateral_rule):
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
        centre = self._lane_id(g.lane_index)  # lane of the ego centre after the queue
        self.plan = dict(maneuver=None if man is None else dict(man), target_lane=lane,
                         lane=centre, queued=[c for c in self.ego.fifo if c is not None],
                         lanes=set() if man is None else maneuver_lanes(man))
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
                v = self.check_plan([name], lc_frames, lanes_entered(centre, tl))
            else:
                if man is not None:
                    v = self.check_plan([name], lc_frames, self.plan["lanes"])
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
            if man["kind"] == "lane_change":
                if man["crossed"]:
                    # the return would become active after the marking is crossed (7.4, item 3)
                    info["ABORT"] = {"after_crossing": True}
                else:
                    v = self.check_plan([ABORT], lc_frames, {origin})
                    info["ABORT"] = v
                    if not v:
                        return Decision(ABORT, a_pi, None, m, reason="return",
                                        violations=info)
            v = self.check_plan([TAKEOVER], lc_frames, maneuver_lanes(man))
            info["COMPLETE_BRAKE"] = v
            if not v:
                return Decision(TAKEOVER, a_pi, None, m, reason="complete",
                                violations=info)
            return Decision(FALLBACK, a_pi, None, m, fallback=True,
                            reason="fallback", violations=info)
        return Decision(TAKEOVER, a_pi, None, m, reason="empty_safe_set",
                        violations=info)
