"""Ego vehicle with FIFO actuator delay and acceleration limiter; scripted neighbours.

Why inside the vehicle and not in a wrapper of env.step():
highway-env calls action_type.act(meta) only on the first physics frame of env.step(),
then road.act() calls vehicle.act() WITHOUT arguments on every frame, which re-runs the
P speed controller from target_speed and overwrites any acceleration set by a wrapper.
Therefore both the FIFO and the limiter live in act(): the limiter is applied after every
controller call on every physics frame.
"""
from collections import deque
import copy
import numpy as np
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle

from .params import (DT, K_TAU, A_MAX_ACCEL, A_MIN_BRAKE, A_MAX_BRAKE, TARGET_SPEEDS,
                     TAKEOVER, ABORT, FALLBACK, SUPERVISORY, MANEUVER_DONE_TOL, DELTA_V)


class DelayedRSSVehicle(MDPVehicle):
    """MDPVehicle whose commands pass through a FIFO of K_TAU physics frames.

    act(cmd)  - called once per decision (frame 0 of env.step): only stores the command;
    act()     - called by road.act() on every physics frame: shifts the FIFO, applies the
                command that leaves it, runs the controllers, then clips the acceleration:
                  normal:   a = clip(a_ctrl, -A_MAX_BRAKE, +A_MAX_ACCEL)
                  takeover: a = clip(a_ctrl, -A_MAX_BRAKE, -A_MIN_BRAKE)
                  v + a*dt < 0  ->  a = -v/dt   (no reversing; Vehicle.MIN_SPEED = -40 m/s)
    TAKEOVER does not change target_speed: when target_speed >= v the controller output is
    positive and the applied braking is exactly -A_MIN_BRAKE (the weakest admissible one).
    """

    def __init__(self, road, position, heading=0.0, speed=0.0, target_lane_index=None,
                 target_speed=None, target_speeds=None, route=None,
                 k_tau: int = K_TAU, dt: float = DT, initial_active=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed,
                         TARGET_SPEEDS if target_speeds is None else target_speeds, route)
        self.k_tau = k_tau
        self.dt = dt
        self.fifo = deque([None] * k_tau)      # explicit empty slots at t = 0
        self.pending = None                    # command stored by act(cmd) this frame
        self.active_cmd = initial_active       # last command that left the FIFO
        self.takeover = False
        self.fallback = False
        self.a_ctrl = 0.0                      # controller output before limiting
        self.a_applied = 0.0                   # acceleration actually integrated
        self.frame = 0
        self.maneuver = None                   # dict(origin, target, crossed, kind)
        self.limiter_active = False
        self.released_frame = None             # frame at which takeover was released
        self.abort_ignored = 0                 # ABORT commands that started no return

    # ------------------------------------------------------------------ commands
    def act(self, action=None):
        if action is not None:
            self.pending = action
            return
        self.fifo.append(self.pending)
        self.pending = None
        cmd = self.fifo.popleft()
        if cmd is not None:
            self._activate(cmd)
        else:
            MDPVehicle.act(self)               # keep last command, controllers recomputed
        self._limit()
        self.frame += 1

    def _activate(self, cmd):
        self.active_cmd = cmd
        if cmd == TAKEOVER:
            self.takeover, self.fallback = True, False
            MDPVehicle.act(self)
            return
        if cmd == FALLBACK:
            self.takeover, self.fallback = True, True
            self.target_lane_index = self.lane_index          # hold lane containing ego centre
            self.maneuver = None
            MDPVehicle.act(self)
            return
        if cmd == ABORT:
            # idempotent: a repeated ABORT during the return must not swap origin/target back;
            # after the marking is crossed (or when no lane change is left to return from) the
            # return is not executed (7.4, item 3): the lateral target is kept and the command is
            # counted in abort_ignored (the shield never issues such an ABORT)
            m = self.maneuver
            if m is not None and m["kind"] == "lane_change" and not m["crossed"]:
                origin = m["origin"]
                self.maneuver = dict(origin=m["target"], target=origin,
                                     crossed=False, kind="return")
                self.target_lane_index = origin
            elif m is None or m["kind"] == "lane_change":
                self.abort_ignored += 1
            MDPVehicle.act(self)
            return
        # ordinary meta-action
        if self.takeover:                      # release: sync target speed, no jump up
            self.takeover, self.fallback = False, False
            self.released_frame = self.frame
            idx = int(np.searchsorted(self.target_speeds, self.speed + 1e-9, side="right") - 1)
            self.speed_index = max(0, idx)
            self.target_speed = self.index_to_speed(self.speed_index)
        lane_before = self.target_lane_index
        MDPVehicle.act(self, cmd)
        if cmd in ("LANE_LEFT", "LANE_RIGHT") and self.target_lane_index != lane_before:
            self.maneuver = dict(origin=self.lane_index, target=self.target_lane_index,
                                 crossed=False, kind="lane_change")

    def _limit(self):
        a_ctrl = float(self.action["acceleration"])
        if self.takeover:
            a = min(max(a_ctrl, -A_MAX_BRAKE), -A_MIN_BRAKE)
        else:
            a = min(max(a_ctrl, -A_MAX_BRAKE), A_MAX_ACCEL)
        if self.speed + a * self.dt < 0.0:
            a = -self.speed / self.dt
        self.a_ctrl = a_ctrl
        self.a_applied = a
        self.limiter_active = abs(a - a_ctrl) > 1e-12
        self.action["acceleration"] = a

    # ------------------------------------------------------------------ state
    def on_state_update(self):
        super().on_state_update()
        m = getattr(self, "maneuver", None)
        if m is not None and self.road is not None:
            # crossed: the ego centre left the origin lane (first marking of a maneuver that spans
            # several lanes) or is in the target lane (reversal / return: origin lane = target)
            if self.lane_index == m["target"] or self.lane_index != m["origin"]:
                m["crossed"] = True
            lat = self.road.network.get_lane(m["target"]).local_coordinates(self.position)[1]
            if m["crossed"] and abs(lat) < MANEUVER_DONE_TOL:
                self.maneuver = None

    @property
    def lateral_speed(self):
        return float(self.speed * np.sin(self.heading))

    # status flags of the journal (protocol 7.5): three different things
    @property
    def brake_mode(self):
        """Forced braking limiter [-A_MAX_BRAKE, -A_MIN_BRAKE] is on (TAKEOVER / FALLBACK)."""
        return bool(self.takeover)

    @property
    def sup_active(self):
        """The active command (last one that left the FIFO) is supervisory."""
        return self.active_cmd in SUPERVISORY

    @property
    def return_active(self):
        """A return to the origin lane (after ABORT) is in progress."""
        return self.maneuver is not None and self.maneuver["kind"] == "return"

    def ghost(self):
        """Light copy for prediction: shares road/network, copies all mutable state."""
        g = copy.copy(self)
        g.position = np.array(self.position, dtype=float)
        g.fifo = deque(self.fifo)
        g.action = dict(self.action)
        g.maneuver = copy.deepcopy(self.maneuver)
        g.history = deque(maxlen=1)
        return g


class ScriptedVehicle(ControlledVehicle):
    """Neighbour with scripted longitudinal acceleration and optional scripted lane change.

    accel(t) -> m/s^2 is applied without IDM's +-6 m/s^2 clip (only v >= 0 is enforced).
    lane_change = (t_start, target_lane_index): from t_start the vehicle steers to the target
    lane with the standard highway-env cascade controller (same as MOBIL lane changes).
    """

    def __init__(self, road, position, heading=0.0, speed=0.0, accel=None, lane_change=None,
                 dt: float = DT, name: str = "N", rss_follow=None):
        super().__init__(road, position, heading, speed)
        self.accel = accel or (lambda t: 0.0)
        self.rss_follow = rss_follow          # None or response time rho of an RSS-compliant follower
        self._danger_since = None
        self.lane_change = lane_change
        self.dt = dt
        self.frame = 0
        self.name = name
        self.a_applied = 0.0

    def act(self, action=None):
        t = self.frame * self.dt
        if self.lane_change is not None and t >= self.lane_change[0] - 1e-9:
            self.target_lane_index = self.lane_change[1]
        a = float(self.accel(t))
        if self.rss_follow is not None:
            a = self._rss_response(t, a)
        if self.speed + a * self.dt < 0.0:
            a = -self.speed / self.dt
        self.a_applied = a
        steering = self.steering_control(self.target_lane_index)
        self.action = {"steering": float(np.clip(steering, -self.MAX_STEERING_ANGLE,
                                                 self.MAX_STEERING_ANGLE)),
                       "acceleration": a}
        self.frame += 1

    @property
    def lateral_speed(self):
        return float(self.speed * np.sin(self.heading))

    def _rss_response(self, t, a):
        """RSS proper response of a follower (Shalev-Shwartz et al., Def. 4): if the gap to the
        nearest vehicle ahead with lateral overlap is below d_min_long, keep the scripted
        acceleration for rho seconds, then brake with A_MIN_BRAKE while faster than the leader."""
        from .rss import d_min_long
        from .params import LENGTH, WIDTH
        front, gap = None, None
        for o in self.road.vehicles:
            if o is self:
                continue
            dx = o.position[0] - self.position[0]
            if dx > 0 and abs(o.position[1] - self.position[1]) < WIDTH and (gap is None or dx - LENGTH < gap):
                front, gap = o, dx - LENGTH
        if front is None or gap >= d_min_long(self.speed, front.speed, self.rss_follow):
            self._danger_since = None
            return a
        if self._danger_since is None:
            self._danger_since = t
        if t - self._danger_since >= self.rss_follow - 1e-9 and self.speed > front.speed:
            return -A_MIN_BRAKE
        return min(a, A_MAX_ACCEL)
