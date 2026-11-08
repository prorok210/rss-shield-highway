"""Main traffic generator of the protocol (section 7.6) and the 86-feature observation (7.2).

Window traffic: 4 lanes, D vehicles per km and lane in the window [x - 300 m, x + 700 m] around
the anchor x (the ego after insertion). Background: IDMVehicle, DELTA ~ U(3.5, 4.5), desired speed
~ U(23, 27) m/s, initial speed v_eq(s) + U(-1, 1). Warm-up 20 s without the ego (the anchor moves
with the mean speed of the flow in the window), then the ego is inserted where the longitudinal RSS
conditions at rho_max = 1.0 s hold for the leader and the follower; otherwise the nearest vehicle of
the chosen lane is removed (at most m_max = 2).

This module does NOT enable fast_road: sim/equiv_runner.py imports it in the original and in the
accelerated process. Experiments create environments through experiments.env, which enables
fast_road on import (experiments/README.md, item 1).
"""
import functools
import pickle

import numpy as np
from highway_env.envs.common.observation import KinematicObservation
from highway_env.vehicle.behavior import IDMVehicle

from rss_shield.env import RSSHighwayEnv, RSSMetaAction
from rss_shield.params import ABORT, DT, FALLBACK, FRAMES_PER_DECISION, LENGTH, TAKEOVER, WIDTH
from rss_shield.rss import d_min_lat, d_min_long
from rss_shield.vehicle import DelayedRSSVehicle

RHOS = (0.1, 0.2, 0.4, 0.6, 1.0)           # rho grid of the main series, s
RHO_INSERT = 1.0                            # rho_max of the insertion rule, s
S_MIN = 2.0                                 # minimal bumper-to-bumper gap of the generator, m
WINDOW_BACK, WINDOW_FRONT = 300.0, 700.0    # window [x - 300, x + 700] m around the anchor
X0 = 1000.0                                 # initial anchor on the 10 km road, m
DESIRED_SPEED = (23.0, 27.0)                # desired speed of the IDM vehicles, m/s
QUEUE_CODES = ("LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER", TAKEOVER, ABORT, FALLBACK)
N_VEH, N_FEAT = 15, 5
N_KIN = N_VEH * N_FEAT                      # 75 kinematic features
N_EXTRA = len(QUEUE_CODES) + 3              # one-hot of the queue + brake flag + speed + lane
N_OBS = N_KIN + N_EXTRA                     # 86
I_BRAKE = len(QUEUE_CODES)                  # index of the flag inside the extra features
I_SPEED, I_LANE = I_BRAKE + 1, I_BRAKE + 2


def idm_equilibrium_speed(s, v0=25.0, delta=4.0, d0=None, T=None):
    """Speed of the homogeneous IDM flow with centre spacing s (equal speeds, zero acceleration):
    1 - (v / v0)^delta = ((d0 + v T) / s)^2."""
    d0 = IDMVehicle.DISTANCE_WANTED if d0 is None else d0
    T = IDMVehicle.TIME_WANTED if T is None else T
    lo, hi = 0.0, v0
    for _ in range(100):
        v = 0.5 * (lo + hi)
        if 1.0 - (v / v0) ** delta - ((d0 + v * T) / s) ** 2 > 0.0:
            lo = v
        else:
            hi = v
    return 0.5 * (lo + hi)


def queued_command(ego):
    """Command waiting in the FIFO at the decision moment (it activates on the next frame)."""
    for c in ego.fifo:
        if c is not None:
            return c
    return None


def mode_a_code(ego, cmd):
    """Mode A mapping (7.2): TAKEOVER / FALLBACK -> SLOWER, ABORT -> lane change in the direction
    of the return."""
    if cmd in (TAKEOVER, FALLBACK):
        return "SLOWER"
    if cmd == ABORT:
        m = ego.maneuver
        if m is None:
            return "IDLE"
        right = m["target"][2] > m["origin"][2]      # direction of the current lateral movement
        if m["kind"] == "lane_change":
            right = not right                           # the return goes the other way
        return "LANE_RIGHT" if right else "LANE_LEFT"
    return cmd


def initially_unsafe(ego, road, rho, adjacent=True, eps=1e-9):
    """Start-state label (7.6): d_min_long(rho) is violated to the leader or to the follower in the
    ego lane, or (adjacent=True) both the longitudinal and the lateral thresholds are violated for a
    vehicle of an adjacent lane (same sign convention of d_min_lat as in the shield)."""
    lane_e = ego.lane_index[2]
    xe, ye, ve = float(ego.position[0]), float(ego.position[1]), float(ego.speed)
    vye = ego.speed * np.sin(ego.heading)
    same = [v for v in road.vehicles if v is not ego and v.lane_index[2] == lane_e]
    ahead = [v for v in same if v.position[0] >= xe]
    behind = [v for v in same if v.position[0] < xe]
    if ahead:
        lead = min(ahead, key=lambda v: v.position[0])
        if lead.position[0] - xe - LENGTH < d_min_long(ve, lead.speed, rho) - eps:
            return True
    if behind:
        foll = max(behind, key=lambda v: v.position[0])
        if xe - foll.position[0] - LENGTH < d_min_long(foll.speed, ve, rho) - eps:
            return True
    if not adjacent:
        return False
    for o in road.vehicles:
        if o is ego or abs(o.lane_index[2] - lane_e) != 1:
            continue
        dx = o.position[0] - xe
        if dx >= 0:
            gap, need = dx - LENGTH, d_min_long(ve, o.speed, rho)
        else:
            gap, need = -dx - LENGTH, d_min_long(o.speed, ve, rho)
        if gap >= need - eps:
            continue
        sgn = np.sign(o.position[1] - ye) or 1.0
        need_lat = d_min_lat(vye * sgn, o.speed * np.sin(o.heading) * sgn, rho)
        if abs(o.position[1] - ye) - WIDTH < need_lat - eps:
            return True
    return False


class ProtocolObservation(KinematicObservation):
    """15 x 5 Kinematics (presence, x, y, vx, vy; see_behind, relative to the ego, standard
    normalisation with clipping), flattened, plus 11 features (7.2):
      [75:83] one-hot of the command waiting in the FIFO, u_{k-1}, over QUEUE_CODES (0 if empty);
      [83]    brake_mode: the forced braking limiter of TAKEOVER / FALLBACK is on;
      [84]    target_speed index / (number of levels - 1);
      [85]    target_lane_index - lane_index (-1, 0, +1).
    mode_a=True: Mode A mapping, supervisory codes replaced by policy actions and [83] = 0."""

    def __init__(self, env, mode_a=False, **kwargs):
        super().__init__(env, **kwargs)
        self.mode_a = mode_a

    def space(self):
        box = super().space()
        return type(box)(shape=(N_OBS,), low=-np.inf, high=np.inf, dtype=np.float32)

    def extras(self):
        ego = self.observer_vehicle
        x = np.zeros(N_EXTRA, dtype=np.float32)
        cmd = queued_command(ego)
        if cmd is not None:
            x[QUEUE_CODES.index(mode_a_code(ego, cmd) if self.mode_a else cmd)] = 1.0
        x[I_BRAKE] = 0.0 if self.mode_a else float(ego.brake_mode)
        x[I_SPEED] = ego.speed_index / max(len(ego.target_speeds) - 1, 1)
        x[I_LANE] = float(np.clip(ego.target_lane_index[2] - ego.lane_index[2], -1, 1))
        return x

    def observe(self):
        kin = np.asarray(super().observe(), dtype=np.float32)
        if kin.shape != (self.vehicles_count, len(self.features)):
            return np.zeros(N_OBS, dtype=np.float32)
        return np.concatenate([kin.reshape(-1), self.extras()])


class LoggedRSSVehicle(DelayedRSSVehicle):
    """DelayedRSSVehicle that keeps the acceleration applied on every physics frame (JerkRMS, 7.7)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_log = []

    def act(self, action=None):
        super().act(action)
        if action is None:
            self.acc_log.append(float(self.a_applied))

    def ghost(self):
        g = super().ghost()
        g.acc_log = []                        # predictions of the shield must not touch the log
        return g


class ProtocolMetaAction(RSSMetaAction):
    @property
    def vehicle_class(self):
        return functools.partial(LoggedRSSVehicle, target_speeds=self.target_speeds)


class ProtocolHighwayEnv(RSSHighwayEnv):
    """4-lane straight highway with window traffic of density D (7.6), the delayed and limited ego
    of rss_shield and the 86-feature observation. reset(seed) builds the start state (placement,
    warm-up, insertion); start_info describes it. env.step() accepts 0..4 or a supervisory command."""

    def __init__(self, config=None, render_mode=None):
        self._constructing = True              # the constructor's own reset() builds no traffic
        super().__init__(config=config, render_mode=render_mode)
        self._constructing = False

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "D": 25,                  # vehicles per km and lane in the window
            "lanes_count": 4,
            "vehicles_count": 0,      # the standard generator of HighwayEnv is not used
            "duration": 40,           # T_max, s (200 decision steps)
            "warmup": 20.0,           # warm-up without the ego, s
            "insert_zone": 100.0,     # the insertion point is within +-100 m of the anchor
            "max_removed": 2,         # m_max
            "obs_mode_a": False,
            "observation": {"type": "Kinematics", "vehicles_count": N_VEH,
                            "features": ["presence", "x", "y", "vx", "vy"], "see_behind": True,
                            "absolute": False, "normalize": True, "clip": True, "order": "sorted"},
        })
        return cfg

    def define_spaces(self):
        super().define_spaces()
        kw = {k: v for k, v in self.config["observation"].items() if k != "type"}
        self.observation_type = ProtocolObservation(self, mode_a=bool(self.config["obs_mode_a"]), **kw)
        self.observation_space = self.observation_type.space()
        akw = {k: v for k, v in self.config["action"].items() if k != "type"}
        self.action_type = ProtocolMetaAction(self, **akw)

    def step(self, action):
        out = super().step(action)
        if self.vehicle is not None:
            self._recycle(float(self.vehicle.position[0]))
        return out

    def _is_truncated(self):
        """T_max is reached after exactly duration * policy_frequency decision steps (frame counter).
        The float sum of 1 / policy_frequency in AbstractEnv.time gives the same 200 steps for
        duration = 40 s, but 51 steps for 10 s."""
        return self.steps >= int(round(self.config["duration"] * self.config["simulation_frequency"]))

    # ------------------------------------------------------------------ scene
    def _lane(self, i):
        return self.road.network.get_lane(("0", "1", int(i)))

    def _window(self, x_ref, exclude=None):
        return [v for v in self.road.vehicles if v is not exclude
                and -WINDOW_BACK <= v.position[0] - x_ref <= WINDOW_FRONT]

    def _lane_vehicles(self, lane_id):
        return sorted((v for v in self.road.vehicles if v.lane_index[2] == lane_id),
                      key=lambda v: v.position[0])

    def _create_vehicles(self):
        self.controlled_vehicles = []
        self.recycled = 0
        self.warmup_speed = []
        if getattr(self, "_constructing", False):
            self._add_ego(1, X0, 20.0)
            self.anchor, self.start_info = X0, {}
            return
        self._place_background()
        self._warmup()
        self._insert_ego()

    def _add_ego(self, lane_id, x, v0):
        lane = self._lane(lane_id)
        ego = self.action_type.vehicle_class(self.road, lane.position(x, 0.0), lane.heading_at(x), v0)
        self.road.vehicles.insert(0, ego)
        self.controlled_vehicles = [ego]
        return ego

    def _place_background(self):
        rng = self.np_random
        D = float(self.config["D"])
        s = 1000.0 / D
        jitter = min(20.0, (s - LENGTH - S_MIN) / 2.0)
        self.v_eq = idm_equilibrium_speed(s)
        n = int(round(D * (WINDOW_BACK + WINDOW_FRONT) / 1000.0))
        for i in range(int(self.config["lanes_count"])):
            lane = self._lane(i)
            for k in range(n):
                x = X0 - WINDOW_BACK + s * (k + 0.5) + rng.uniform(-jitter, jitter)
                v = IDMVehicle(self.road, lane.position(x, 0.0), lane.heading_at(x),
                               self.v_eq + rng.uniform(-1.0, 1.0),
                               target_speed=rng.uniform(*DESIRED_SPEED))
                v.randomize_behavior()
                self.road.vehicles.append(v)
        self.anchor = X0

    def _warmup(self):
        frames = int(round(float(self.config["warmup"]) / DT))
        for f in range(frames):
            self.road.act()
            self.road.step(DT)
            w = self._window(self.anchor)
            vbar = float(np.mean([v.speed for v in w])) if w else 0.0
            self.anchor += vbar * DT                     # virtual observer moving with the flow
            self.warmup_speed.append(vbar)
            if (f + 1) % FRAMES_PER_DECISION == 0:
                self._recycle(self.anchor)

    def _recycle(self, x_ref):
        """Vehicles that left the window go to its opposite edge in the same lane (7.6): behind
        x - 300 m -> x + 700 m and vice versa, if the centre distance to every vehicle of that lane
        is at least L + s_min (otherwise retried later); speed = mean speed of the lane in the window
        (ego excluded)."""
        ego = self.vehicle
        for v in list(self.road.vehicles):
            if v is ego:
                continue
            dx = v.position[0] - x_ref
            if -WINDOW_BACK <= dx <= WINDOW_FRONT:
                continue
            x_new = x_ref + (WINDOW_FRONT if dx < -WINDOW_BACK else -WINDOW_BACK)
            lane_id = v.lane_index[2]
            same = [o for o in self.road.vehicles if o is not v and o.lane_index[2] == lane_id]
            if any(abs(o.position[0] - x_new) < LENGTH + S_MIN for o in same):
                continue
            speeds = [o.speed for o in same if o is not ego
                      and -WINDOW_BACK <= o.position[0] - x_ref <= WINDOW_FRONT]
            lane = self._lane(lane_id)
            v.position = np.array(lane.position(x_new, 0.0), dtype=float)
            v.heading = float(lane.heading_at(x_new))
            if speeds:
                v.speed = float(np.mean(speeds))
            v.target_lane_index = v.lane_index
            v.on_state_update()
            self.recycled += 1

    def _insertion_points(self, lane_id, xa, rho):
        """For every gap of the lane where the longitudinal RSS conditions at rho hold for the
        leader and the follower with the ego speed v0 = mean speed of the two (7.6): the admissible
        point closest to xa within insert_zone. Returns [(x, v0)]."""
        zone = float(self.config["insert_zone"])
        vs = self._lane_vehicles(lane_id)
        out = []
        for f, l in zip([None] + vs, vs + [None]):
            speeds = [u.speed for u in (f, l) if u is not None]
            if not speeds:
                continue
            v0 = float(np.mean(speeds))
            lo, hi = xa - zone, xa + zone
            if f is not None:
                lo = max(lo, f.position[0] + LENGTH + d_min_long(f.speed, v0, rho))
            if l is not None:
                hi = min(hi, l.position[0] - LENGTH - d_min_long(v0, l.speed, rho))
            if lo <= hi:
                out.append((float(min(max(xa, lo), hi)), v0))
        return out

    def _gap_midpoint(self, lane_id, xa):
        vs = self._lane_vehicles(lane_id)
        f = max((v for v in vs if v.position[0] <= xa), key=lambda v: v.position[0], default=None)
        l = min((v for v in vs if v.position[0] > xa), key=lambda v: v.position[0], default=None)
        if f is None or l is None:
            return xa, (float(np.mean([v.speed for v in vs])) if vs else self.v_eq)
        return 0.5 * (f.position[0] + l.position[0]), 0.5 * (f.speed + l.speed)

    def _insert_ego(self):
        rng = self.np_random
        xa, n_lanes = self.anchor, int(self.config["lanes_count"])
        cands = [(i, x, v0) for i in range(n_lanes)
                 for x, v0 in self._insertion_points(i, xa, RHO_INSERT)]
        removed, ok = 0, bool(cands)
        if ok:
            lane_id, x, v0 = cands[int(rng.integers(len(cands)))]
        else:
            lane_id = int(rng.integers(n_lanes))
            pts = []
            while not pts and removed < int(self.config["max_removed"]):
                vs = self._lane_vehicles(lane_id)
                if not vs:
                    break
                self.road.vehicles.remove(min(vs, key=lambda v: abs(v.position[0] - xa)))
                removed += 1
                pts = self._insertion_points(lane_id, xa, RHO_INSERT)
            ok = bool(pts)
            if ok:
                x, v0 = pts[int(rng.integers(len(pts)))]
            else:
                x, v0 = self._gap_midpoint(lane_id, xa)
        ego = self._add_ego(lane_id, x, v0)
        self.anchor = float(x)
        self._recycle(self.anchor)
        self.start_info = self._start_info(ego, lane_id, v0, ok, removed)

    def _start_info(self, ego, lane_id, v0, ok, removed):
        x = float(ego.position[0])
        w = self._window(x, exclude=ego)
        n_lanes = int(self.config["lanes_count"])
        ws, k = self.warmup_speed, int(round(5.0 / DT))
        late = (float(np.mean(ws[-2 * k:-k])), float(np.mean(ws[-k:]))) if len(ws) >= 2 * k else (None, None)
        return dict(D=self.config["D"], lane=int(lane_id), x=x, v0=float(v0),
                    target_speed=float(ego.target_speed), insert_ok=bool(ok), removed=int(removed),
                    n_window=len(w),
                    density=len(w) / (n_lanes * (WINDOW_BACK + WINDOW_FRONT) / 1000.0),
                    mean_speed=float(np.mean([v.speed for v in w])) if w else 0.0,
                    warmup_10_15=late[0], warmup_15_20=late[1],
                    unsafe={r: initially_unsafe(ego, self.road, r) for r in RHOS},
                    unsafe_lane={r: initially_unsafe(ego, self.road, r, adjacent=False) for r in RHOS})

    # ------------------------------------------------------------------ start states (README, 2)
    GEN_KEYS = ("D", "lanes_count", "warmup", "insert_zone", "max_removed")

    def save_start_state(self, path):
        """State at the ego insertion moment (call right after reset): the road with all vehicles,
        the random generator of the env (the road holds the same generator) and start_info."""
        blob = dict(version=1, road=self.road, anchor=self.anchor, start_info=self.start_info,
                    warmup_speed=list(self.warmup_speed),
                    config={k: self.config[k] for k in self.GEN_KEYS})
        with open(path, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_start_state(self, path):
        """Equivalent of reset(seed) from a saved start state (bit-identical episode)."""
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        for k, v in blob["config"].items():
            if self.config[k] != v:
                raise ValueError(f"start state {path}: {k} = {v}, env has {self.config[k]}")
        self.time = self.steps = 0
        self.done = False
        self.road = blob["road"]
        self._np_random = self.road.np_random
        self.controlled_vehicles = [self.road.vehicles[0]]
        self.anchor, self.start_info = blob["anchor"], blob["start_info"]
        self.warmup_speed, self.recycled = blob["warmup_speed"], 0
        self.define_spaces()
        obs = self.observation_type.observe()
        return obs, {"start_info": self.start_info}
