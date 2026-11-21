"""Episode metrics of protocol 7.7 that are not decision counters: density of the traffic window
(delayed recirculation), route progress L(D), cut-in events, convergence of the validation curve
and the validation schedule of the main training run.

All definitions are written once here and used by the runner, the pilot and the analysis so that
the protocol text, the code and the stored episode tables use the same numbers.
"""
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rss_shield.params import LENGTH, WIDTH
from rss_shield.rss import d_min_lat, d_min_long
from rss_shield.shield import lateral_danger

VAL_SCENARIOS = tuple(range(10_000, 10_020))   # fixed validation scenarios (protocol 7.7)
VAL_EVERY = 5_000                              # decision steps between validation runs
VAL_WINDOW = 4                                 # checks in the convergence window (4 x 5000 steps)
MAX_SUFFICIENT_STEP = 85_000                   # S_conv <= 85 000 is a sufficient budget


def idm_eq_speed(D: float) -> float:
    """Equilibrium speed of the homogeneous IDM flow at D vehicles/km/lane."""
    from experiments.generator import idm_equilibrium_speed
    if D <= 0:
        raise ValueError(f"D must be positive, got {D!r}")
    return float(idm_equilibrium_speed(1000.0 / float(D)))


def target_distance_L(D: float, T_max: float = 40.0) -> float:
    """L(D) = 0.8 * v_eq(D) * T_max (protocol 7.7)."""
    return float(0.8 * idm_eq_speed(D) * T_max)


@dataclass
class DensityTracker:
    """Density of the moving window [x - 300, x + 700] m.  The generator recirculates a vehicle
    only when its destination slot is free; until then the vehicle stays outside the window, so
    n_window + n_outside is constant (4D - removed) but n_window is not.  Every sample records the
    deficit explicitly instead of assuming n_window = 4D - removed."""
    D_nominal: int
    n_removed: int
    n_lanes: int = 4
    window_length_km: float = 1.0
    n_window_start: int | None = None
    n_outside_start: int | None = None
    n_window_samples: list[int] = field(default_factory=list)
    n_outside_samples: list[int] = field(default_factory=list)

    @property
    def n_bg_total(self) -> int:
        return self.n_lanes * self.D_nominal - self.n_removed

    def record(self, n_window: int, *, is_start: bool = False) -> None:
        n_window = int(n_window)
        n_out = self.n_bg_total - n_window
        if n_window < 0 or n_out < 0:
            raise ValueError(f"inconsistent window count {n_window} vs n_bg_total={self.n_bg_total}")
        if is_start:
            self.n_window_start = n_window
            self.n_outside_start = n_out
        self.n_window_samples.append(n_window)
        self.n_outside_samples.append(n_out)

    def summary(self) -> dict[str, Any]:
        if not self.n_window_samples or self.n_window_start is None:
            raise ValueError("DensityTracker.record(is_start=True) was never called")
        denom = self.n_lanes * self.window_length_km
        nw = np.asarray(self.n_window_samples, dtype=float)
        no = np.asarray(self.n_outside_samples, dtype=float)
        return {
            "D_nominal": int(self.D_nominal),
            "n_removed": int(self.n_removed),
            "n_window_start": int(self.n_window_start),
            "n_outside_start": int(self.n_outside_start),
            "density_start": float(self.n_window_start / denom),
            "density_mean": float(nw.mean() / denom),
            "density_min": float(nw.min() / denom),
            "n_outside_mean": float(no.mean()),
            "n_outside_max": int(no.max()),
            "n_deficit_mean": float(self.n_removed + no.mean()),
        }


@dataclass
class RouteProgressTracker:
    """L(D) progress at physics-frame resolution.  survived (no collision to T_max) and completed
    (reached L(D) before a collision or the end of the episode) are different fields."""
    x_start: float
    L_target: float
    T_max: float = 40.0
    t_reach_L: float | None = None

    def update_frame(self, t_sec: float, x_ego: float, crashed: bool) -> None:
        if self.t_reach_L is None and not crashed and (x_ego - self.x_start) >= self.L_target:
            self.t_reach_L = float(t_sec)

    def summary(self, crashed: bool, x_end: float) -> dict[str, Any]:
        reached = self.t_reach_L is not None
        delta_x = float(x_end - self.x_start)
        return {
            "L_target": float(self.L_target),
            "delta_x": delta_x,
            "v_eff": float(delta_x / self.T_max),
            "survived": bool(not crashed),
            "reached_L": reached,
            "completed": reached,
            "t_reach_L": self.t_reach_L,
            "t_travel_pen": float(self.t_reach_L) if reached else float(self.T_max),
        }


@dataclass
class VehSnap:
    vid: int
    lane: int
    target_lane: int
    x: float
    y: float
    vx: float
    vy: float


def pair_is_dangerous(ego: VehSnap, nb: VehSnap, rho: float) -> bool:
    """RSS dangerous pair at rho with the operational lateral rule of the main series: the
    longitudinal threshold is violated and the lateral condition is overlap or approach."""
    dx = nb.x - ego.x
    if dx >= 0:
        gap, need = dx - LENGTH, d_min_long(ego.vx, nb.vx, rho)
    else:
        gap, need = -dx - LENGTH, d_min_long(nb.vx, ego.vx, rho)
    side = abs(nb.y - ego.y) - WIDTH
    sgn = np.sign(nb.y - ego.y) or 1.0
    closing = (ego.vy - nb.vy) * sgn
    lateral = side < 0.0 or (side < d_min_lat(ego.vy * sgn, nb.vy * sgn, rho)
                             and lateral_danger(side, closing))
    return bool(gap < need and lateral)


@dataclass
class CutInTracker:
    """Frame-wise detector of a neighbour cutting into the ego lane while the ego keeps its lane.
    A cut-in is counted on the transition from safe to dangerous when the neighbour moves towards
    the ego lane centre (adjacent lane, lane crossing, or completing the manoeuvre after the
    lateral move).  A recirculation teleport resets the history of the vehicle.  The danger
    predicate must use exactly the RSS approach rule of the filter."""
    rho_eval: float
    pair_is_dangerous_fn: Any = pair_is_dangerous
    lane_width: float = 4.0
    lat_vel_thresh: float = 0.1
    lane_center_tol: float = 0.1
    teleport_jump_m: float = 100.0
    n_cutin: int = 0
    n_cutin_frames: int = 0
    _prev_x: dict[int, float] = field(default_factory=dict)
    _prev_lane: dict[int, int] = field(default_factory=dict)
    _origin_lane: dict[int, int] = field(default_factory=dict)
    _in_danger: dict[int, bool] = field(default_factory=dict)
    _cutin_active: dict[int, bool] = field(default_factory=dict)

    def _lane_center(self, lane: int) -> float:
        return float(lane) * self.lane_width

    def update_frame(self, frame_idx: int, ego: VehSnap, neighbors: Sequence[VehSnap], *,
                     return_active: bool = False) -> None:
        ego_center = self._lane_center(ego.lane)
        ego_keeping_lane = ((ego.target_lane == ego.lane)
                            and (abs(ego.y - ego_center) < self.lane_center_tol)
                            and (not return_active))
        seen: set[int] = set()
        for nb in neighbors:
            vid = nb.vid
            seen.add(vid)
            prev_x = self._prev_x.get(vid)
            if prev_x is not None and abs(nb.x - prev_x) > self.teleport_jump_m:
                self._in_danger[vid] = False
                self._cutin_active[vid] = False
                self._prev_lane[vid] = nb.lane
                self._origin_lane[vid] = nb.lane

            prev_lane = self._prev_lane.get(vid, nb.lane)
            origin_lane = self._origin_lane.get(vid, nb.lane)
            nb_center = self._lane_center(nb.lane)
            if abs(nb.y - nb_center) < self.lane_center_tol and abs(nb.vy) <= self.lat_vel_thresh:
                origin_lane = nb.lane

            danger_now = bool(self.pair_is_dangerous_fn(ego, nb, self.rho_eval))
            was_danger = self._in_danger.get(vid, False)
            if not danger_now:
                self._in_danger[vid] = False
                self._cutin_active[vid] = False
            else:
                if frame_idx >= 1 and not was_danger and ego_keeping_lane:
                    moving_toward = ((ego_center - nb.y) * nb.vy > 0.0
                                     and abs(nb.vy) > self.lat_vel_thresh)
                    cond_adjacent = (nb.lane != ego.lane) and moving_toward
                    cond_crossed = (prev_lane != ego.lane) and (nb.lane == ego.lane)
                    cond_completing = ((nb.lane == ego.lane) and (origin_lane != ego.lane)
                                       and (abs(nb.y - ego_center) >= self.lane_center_tol)
                                       and moving_toward)
                    if cond_adjacent or cond_crossed or cond_completing:
                        self.n_cutin += 1
                        self._cutin_active[vid] = True
                self._in_danger[vid] = True
                if self._cutin_active.get(vid, False):
                    self.n_cutin_frames += 1

            self._prev_x[vid] = nb.x
            self._prev_lane[vid] = nb.lane
            self._origin_lane[vid] = origin_lane

        for old in list(self._prev_x):
            if old not in seen:
                for store in (self._prev_x, self._prev_lane, self._origin_lane, self._in_danger,
                              self._cutin_active):
                    store.pop(old, None)

    def summary(self) -> dict[str, Any]:
        return {"n_cutin": int(self.n_cutin), "n_cutin_frames": int(self.n_cutin_frames),
                "has_cutin": bool(self.n_cutin > 0), "rho_cutin_eval": float(self.rho_eval)}


def compute_s_conv(steps: Sequence[int], delta_x_vals: Sequence[float], *, window: int = VAL_WINDOW,
                   rel_threshold: float = 0.95, max_sufficient_step: int = MAX_SUFFICIENT_STEP,
                   expected_interval: int | None = VAL_EVERY) -> dict[str, Any]:
    """S_conv: first validation check whose value and the next `window - 1` checks are at least
    rel_threshold of their maximum (protocol 7.7).  Fewer than window checkpoints is explicitly
    insufficient, never a silent convergence."""
    s_arr = np.asarray(steps, dtype=int)
    x_arr = np.asarray(delta_x_vals, dtype=float)
    if len(s_arr) != len(x_arr):
        raise ValueError("steps and delta_x_vals must have equal length")
    if len(s_arr) < window:
        return {"s_conv": None, "budget_sufficient": False, "status": "insufficient_checkpoints",
                "x_max": float(np.max(x_arr)) if len(x_arr) else None}
    if np.any(np.diff(s_arr) <= 0):
        raise ValueError("steps must be strictly increasing")
    if expected_interval is not None and np.any(np.diff(s_arr) != expected_interval):
        raise ValueError(f"steps must be spaced by {expected_interval}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("delta_x_vals contains non-finite values")
    x_max = float(np.max(x_arr))
    if x_max <= 0.0:
        return {"s_conv": None, "budget_sufficient": False, "status": "non_positive_progress",
                "x_max": x_max}
    threshold = rel_threshold * x_max
    s_conv = None
    for i in range(0, len(x_arr) - window + 1):
        if np.all(x_arr[i:i + window] >= threshold):
            s_conv = int(s_arr[i])
            break
    sufficient = (s_conv is not None) and (s_conv <= max_sufficient_step)
    return {"s_conv": s_conv, "budget_sufficient": sufficient,
            "status": "converged" if sufficient else
                      ("late_convergence" if s_conv is not None else "not_converged"),
            "x_max": x_max, "threshold": float(threshold)}


@dataclass
class ValidationSchedule:
    """Fixed validation runs of the main training budget: 20 scenarios every 5000 decision steps,
    epsilon = 0, recorded as (step, mean delta_x).  The pilot never reaches the first checkpoint and
    writes status = insufficient_checkpoints."""
    every: int = VAL_EVERY
    scenario_ids: tuple[int, ...] = VAL_SCENARIOS
    history: list[dict[str, Any]] = field(default_factory=list)

    def next_due(self, step: int) -> int | None:
        """Scheduled checkpoint crossed by `step` (the first one when several were skipped)."""
        if step < self.every:
            return None
        target = (self.history[-1]["step"] if self.history else 0) + self.every
        return target if target <= step else None

    def record(self, step: int, delta_x: Sequence[float]) -> dict[str, Any]:
        values = np.asarray(delta_x, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("validation delta_x must be a non-empty finite sample")
        row = {"step": int(step), "n": int(values.size),
               "delta_x_mean": float(values.mean()), "delta_x_min": float(values.min()),
               "delta_x_max": float(values.max())}
        self.history.append(row)
        return row

    def s_conv(self) -> dict[str, Any]:
        return compute_s_conv([r["step"] for r in self.history],
                              [r["delta_x_mean"] for r in self.history], expected_interval=self.every)
