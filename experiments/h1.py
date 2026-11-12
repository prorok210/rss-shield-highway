"""H1 event and correct-start rules (protocol 6, 7.6, 7.7, 7.9), written once.

Every arm - Proposed in modes A and B, Baseline 1, Baseline 2 - goes through the same procedure;
the classification (classify_collision), the runner and the pilot import the rules from here.

Start rule (denominator). An episode has a correct start iff at t = 0 the longitudinal RSS
threshold d_min_long(RHO_START) holds to the leader and to the follower in the ego lane, with
RHO_START = rho_max = 1.0 s, the threshold of the ego insertion (generator.RHO_INSERT). The rule
depends neither on the arm nor on the rho of the cell: the start state is a function of
(D, episode_seed), so every arm that shares the episode seeds has the same correct starts and the
same n_valid. A gap exactly equal to the threshold is correct (violation: gap < d_min - 1e-9).
The per-rho label of the generator (start_info["unsafe_lane"][rho]) stays in the logs as a
descriptive column and defines no denominator.

Classification rho. The operational category of a collision depends on a response time through
the thresholds of the dangerous situation (t_b), the window [t_b - rho, t_b] of the leader test
and the admissible reaction time t_b + rho:
    arms with the filter      rho_cls = rho of the filter (its own response time);
    arms without the filter   rho_cls = RHO_REF = 1.0 s (Baseline 1 and Baseline 2 have no rho).
RHO_REF = RHO_START: the reference thresholds are those of the insertion, so a correct start is
never dangerous at t = 0 under the reference classification. Baseline 2 is classified once, at
RHO_REF; it is not re-classified at the rho of a Proposed cell, neither for the 15 checks of H1b
nor for the descriptive difference CR_ego(Proposed, rho) - CR_ego(Baseline 2).

H1 event. An episode is an H1 event iff it has a correct start and ends with a collision of the
ego whose category at rho_cls is CR_ego_op (manoeuvre or reaction). H1a: share of H1 events among
the correct-start episodes of a filtered policy; H1b: the same share for Baseline 2.
CR_total - the share of episodes with a collision - is computed for every arm from the crash
flag alone, over all episodes and over the correct-start episodes, independently of the
categories.

Operational categories (7.7), applied in this order to the pair ego - collision partner; the
first rule that fires decides. State index k = state at the end of physics frame k - 1 (k = 0:
t = 0); a[k] is the acceleration applied during the frame that ends at state k.
  danger(k)  longitudinal gap < d_min_long(rho_cls) and the lateral part of shield.lateral_danger:
             lateral overlap, or lateral clearance < d_min_lat while it shrinks faster than
             V_LAT_OBS;
  t_b        start of the last uninterrupted danger interval that reaches the contact;
  1 CR_init_op            danger holds from t = 0 to the contact (t_b = 0);
  2 CR_env_op (neighbour) at t_b the partner moves laterally (|v_lat| > V_LAT_OBS), its lateral
                          movement started after the ego lane change that is in progress at t_b
                          was issued, and the supervisor issued TAKEOVER / ABORT / FALLBACK not
                          later than the first decision tick after t_b;
  3 CR_ego_op (manoeuvre) a lane change of the ego is in progress at t_b; or the partner is the
                          leader (ahead, lateral overlap at t_b) and did not brake on [t_b - rho,
                          t_b] (no frame with a < -BRAKE_EPS);
  4 CR_env_op (bounds)    on [t_b - rho, t_contact) the partner braked harder than -A_MAX_BRAKE,
                          accelerated harder than +A_MAX_ACCEL, moved laterally into the lane of
                          the ego, or had crashed before the contact (highway-env: a = -v);
  5 CR_ego_op (reaction)  rho_cls < tau_eff; or the reaction of the ego (first frame from t_b on
                          with a <= -A_MIN_BRAKE, or standstill) started later than t_b + rho -
                          a start exactly at t_b + rho is in time; or after it started the ego
                          braked weaker than A_MIN_BRAKE while moving;
  6 CR_env_op (rear-end)  the partner hit the ego from behind (lateral overlap at the contact) and
                          the ego never braked harder than -A_MAX_BRAKE on [t_b - rho, t_contact);
  7 CR_unc_op             everything else (listed and reviewed by hand).
The procedure is operational; it is not a formal attribution of responsibility under RSS.
"""
from dataclasses import dataclass, field

import numpy as np

from experiments.generator import RHO_INSERT
from rss_shield.params import (A_MAX_ACCEL, A_MAX_BRAKE, A_MIN_BRAKE, DT, FRAMES_PER_DECISION,
                               LANE_WIDTH, LENGTH, SUPERVISORY, TAU_EFF, WIDTH)
from rss_shield.rss import d_min_lat, d_min_long
from rss_shield.shield import V_LAT_OBS, lateral_danger

RHO_START = RHO_INSERT   # s, start rule: rho_max of the ego insertion
RHO_REF = RHO_START      # s, classification rho of arms without the filter (Baseline 1 / 2)
BRAKE_EPS = 0.1          # m/s^2: a leader "braked" if a < -BRAKE_EPS on some frame
TOL = 1e-9

CR_INIT = "CR_init_op"
CR_ENV_NEIGHBOUR = "CR_env_op (neighbour manoeuvre)"
CR_EGO_MANEUVER = "CR_ego_op (manoeuvre)"
CR_ENV_BOUNDS = "CR_env_op (bounds)"
CR_EGO_REACTION = "CR_ego_op (reaction)"
CR_ENV_REAR = "CR_env_op (rear-end)"
CR_UNC = "CR_unc_op"
CATEGORIES = (CR_INIT, CR_ENV_NEIGHBOUR, CR_EGO_MANEUVER, CR_ENV_BOUNDS, CR_EGO_REACTION,
              CR_ENV_REAR, CR_UNC)
CR_EGO = (CR_EGO_MANEUVER, CR_EGO_REACTION)


# ---------------------------------------------------------------------- rules shared by all arms
def classification_rho(shield_on, rho=None):
    """rho used to classify the collisions of an arm: the rho of its filter, RHO_REF without one."""
    if shield_on:
        if rho is None:
            raise ValueError("an arm with the filter needs the rho of the filter")
        return float(rho)
    return RHO_REF


def start_violation(gap_front=None, v_front=None, gap_rear=None, v_rear=None, v_ego=0.0,
                    rho=RHO_START):
    """True if a bumper gap of the ego lane violates d_min_long(rho) at t = 0 (None: no vehicle)."""
    if gap_front is not None and gap_front < d_min_long(v_ego, v_front, rho) - TOL:
        return True
    return gap_rear is not None and gap_rear < d_min_long(v_rear, v_ego, rho) - TOL


def correct_start(start_info):
    """Start rule on the start_info of the generator: its ego-lane label at RHO_START."""
    return not bool(start_info["unsafe_lane"][RHO_START])


def is_h1_event(valid_start, crashed, category):
    return bool(valid_start and crashed and category in CR_EGO)


def h1_summary(episodes):
    """H1 numbers of one check (a policy in a cell) from the per-episode counters of the runner:
    n, n_valid (correct starts), CR_total over all and over correct-start episodes - from the
    crash flag alone -, H1 events and their share among the correct starts, category counts."""
    n = len(episodes)
    valid = [c for c in episodes if c["correct_start"]]
    crashes = [c for c in episodes if c["crashed"]]
    cats = {}
    for c in crashes:
        cats[c["crash_category"]] = cats.get(c["crash_category"], 0) + 1
    events = sum(bool(c["h1_event"]) for c in episodes)
    rhos = sorted({c["rho_cls"] for c in episodes})
    return dict(n=n, n_valid=len(valid), crashes=len(crashes),
                crashes_valid=sum(bool(c["crashed"]) for c in valid),
                CR_total=len(crashes) / n if n else None,
                CR_total_valid=sum(bool(c["crashed"]) for c in valid) / len(valid) if valid else None,
                h1_events=events, CR_ego_op=events / len(valid) if valid else None,
                categories=cats, rho_cls=rhos[0] if len(rhos) == 1 else rhos)


# ---------------------------------------------------------------------- trace of a collision
@dataclass
class Trace:
    """Ego and collision partner on states k = 0..K (the contact is the last state).
    ego / partner: dicts of float arrays x, y, v, vy, a; partner values are NaN while the vehicle
    is outside the recorded range. lc: a lane change of the ego is in progress at state k.
    partner_crashed[k]: the partner had already crashed at state k (before the contact).
    sup_ticks / lc_ticks: state indices of the decision ticks at which a supervisory command /
    a lane change of the ego was issued."""
    ego: dict
    partner: dict
    lc: np.ndarray
    partner_crashed: np.ndarray
    sup_ticks: tuple = ()
    lc_ticks: tuple = ()
    dt: float = DT
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.ego["x"]) - 1


def _pair(tr, k, rho):
    """(danger, ahead, overlap) of the pair at state k."""
    e, p = tr.ego, tr.partner
    if np.isnan(p["x"][k]):
        return False, False, False
    dx = p["x"][k] - e["x"][k]
    if dx >= 0:
        gap, need = dx - LENGTH, d_min_long(e["v"][k], p["v"][k], rho)
    else:
        gap, need = -dx - LENGTH, d_min_long(p["v"][k], e["v"][k], rho)
    side = abs(p["y"][k] - e["y"][k]) - WIDTH
    sgn = np.sign(p["y"][k] - e["y"][k]) or 1.0
    closing = (e["vy"][k] - p["vy"][k]) * sgn
    lat = side < 0.0 or (side < d_min_lat(e["vy"][k] * sgn, p["vy"][k] * sgn, rho) - TOL
                         and lateral_danger(side, closing))
    return bool(gap < need - TOL and lat), bool(dx >= 0), bool(side < 0.0)


def danger_onset(tr, rho):
    """State index k_b of t_b: start of the last uninterrupted danger interval before the contact."""
    k = tr.K
    while k > 0 and _pair(tr, k - 1, rho)[0]:
        k -= 1
    return k


def reaction_start(tr, k_b):
    """First state from k_b + 1 on reached by a frame with a <= -A_MIN_BRAKE or at standstill;
    None if the ego never reacted before the contact."""
    e = tr.ego
    for k in range(k_b + 1, tr.K + 1):
        if e["a"][k] <= -A_MIN_BRAKE + 1e-6 or e["v"][k] <= TOL:
            return k - 1                      # the braking frame starts at state k - 1
    return None


def reaction_late(k_b, k_r, rho, k_contact, dt=DT):
    """The reaction started later than t_b + rho. A start exactly at t_b + rho is in time; no
    reaction at all is late only if the contact itself is later than t_b + rho."""
    limit = k_b + rho / dt
    k = k_contact if k_r is None else k_r
    return bool(k > limit + 1e-6)


def classify_collision(tr, rho):
    """Operational category (7.7) of the collision recorded in the trace, at classification rho."""
    e, p, K = tr.ego, tr.partner, tr.K
    k_b = danger_onset(tr, rho)
    n_rho = int(np.ceil(rho / tr.dt - 1e-6))
    w0 = max(0, k_b - n_rho)
    _, ahead_b, overlap_b = _pair(tr, k_b, rho)
    lc_b = bool(tr.lc[k_b])
    # 1
    if k_b == 0 and _pair(tr, 0, rho)[0]:
        return CR_INIT
    # 2
    p_lat = np.abs(np.nan_to_num(p["vy"])) > V_LAT_OBS
    if lc_b and p_lat[k_b]:
        k_lc = k_b
        while k_lc > 0 and tr.lc[k_lc - 1]:
            k_lc -= 1
        issued = max([t for t in tr.lc_ticks if t <= k_lc], default=k_lc)
        k_n = k_b
        while k_n > 0 and p_lat[k_n - 1]:
            k_n -= 1
        next_tick = (k_b // FRAMES_PER_DECISION + 1) * FRAMES_PER_DECISION
        if k_n > issued and any(issued <= t <= next_tick for t in tr.sup_ticks):
            return CR_ENV_NEIGHBOUR
    # 3
    if lc_b:
        return CR_EGO_MANEUVER
    if ahead_b and overlap_b and not np.any(p["a"][w0 + 1:k_b + 1] < -BRAKE_EPS):
        return CR_EGO_MANEUVER
    # 4
    pa = p["a"][w0 + 1:K]
    into_lane = np.any(p_lat[w0:K] & (np.abs(p["y"][w0:K] - e["y"][w0:K]) < LANE_WIDTH)
                       & (np.sign(np.nan_to_num(p["vy"][w0:K])) == np.sign(e["y"][w0:K] - p["y"][w0:K])))
    if (np.any(pa < -A_MAX_BRAKE - 1e-6) or np.any(pa > A_MAX_ACCEL + 1e-6) or into_lane
            or np.any(tr.partner_crashed[:K])):
        return CR_ENV_BOUNDS
    # 5
    if rho < TAU_EFF - TOL:
        return CR_EGO_REACTION
    k_r = reaction_start(tr, k_b)
    if reaction_late(k_b, k_r, rho, K, tr.dt):
        return CR_EGO_REACTION
    if k_r is not None:
        moving = e["v"][k_r + 1:K + 1] > TOL
        if np.any(moving & (e["a"][k_r + 1:K + 1] > -A_MIN_BRAKE + 1e-6)):
            return CR_EGO_REACTION
    # 6
    _, ahead_c, overlap_c = _pair(tr, K, rho)
    if not ahead_c and overlap_c and not np.any(e["a"][w0 + 1:K] < -A_MAX_BRAKE - 1e-6):
        return CR_ENV_REAR
    return CR_UNC


# ---------------------------------------------------------------------- recording in the env
FIELDS = ("x", "y", "v", "vy", "a")
TRACE_RANGE = 250.0      # m: vehicles within this longitudinal distance of the ego are recorded


class FrameRecorder:
    """Per-physics-frame record of the ego and of the vehicles around it (hook on road.step).
    Memory: only the vehicles within TRACE_RANGE; trace() builds the Trace of a collision."""

    def __init__(self, env):
        self.ego, self.road = env.vehicle, env.road
        self.frames, self.sup_ticks, self.lc_ticks = [], [], []
        self._step = self.road.step
        self.road.step = self._hook
        self._snap()

    def close(self):
        self.road.step = self._step

    def _hook(self, dt):
        self._step(dt)
        self._snap()

    @staticmethod
    def _state(v):
        return (float(v.position[0]), float(v.position[1]), float(v.speed),
                float(v.speed * np.sin(v.heading)), float(v.action.get("acceleration", 0.0)))

    def _snap(self):
        ego = self.ego
        xe = float(ego.position[0])
        m = getattr(ego, "maneuver", None)
        others = {id(v): self._state(v) + (bool(v.crashed),) for v in self.road.vehicles
                  if v is not ego and abs(float(v.position[0]) - xe) <= TRACE_RANGE}
        self.frames.append((self._state(ego), bool(m is not None and m["kind"] == "lane_change"),
                            bool(ego.crashed), others))

    def decision(self, command):
        """Call at a decision tick, before env.step(command)."""
        k = len(self.frames) - 1
        if command in SUPERVISORY:
            self.sup_ticks.append(k)
        elif command in ("LANE_LEFT", "LANE_RIGHT"):
            self.lc_ticks.append(k)

    def trace(self):
        """Trace of the collision of the ego, or None if it did not crash."""
        K = next((k for k, f in enumerate(self.frames) if f[2]), None)
        if K is None:
            return None
        ex, ey = self.frames[K][0][0], self.frames[K][0][1]
        crashed = {i: s for i, s in self.frames[K][3].items() if s[5]}
        if not crashed:
            return None
        pid = min(crashed, key=lambda i: (crashed[i][0] - ex) ** 2 + (crashed[i][1] - ey) ** 2)
        ego = {f: np.array([fr[0][j] for fr in self.frames[:K + 1]]) for j, f in enumerate(FIELDS)}
        rows = [fr[3].get(pid) for fr in self.frames[:K + 1]]
        partner = {f: np.array([np.nan if r is None else r[j] for r in rows])
                   for j, f in enumerate(FIELDS)}
        before = np.array([bool(r is not None and r[5]) for r in rows])
        before[K] = False
        return Trace(ego=ego, partner=partner,
                     lc=np.array([fr[1] for fr in self.frames[:K + 1]]), partner_crashed=before,
                     sup_ticks=tuple(self.sup_ticks), lc_ticks=tuple(self.lc_ticks))
