"""Control scenarios of protocol section 7.5 (v6) with the fixes of the v7 instruction."""
import numpy as np
from .params import DT, LENGTH, TAKEOVER, ABORT, A_MAX_BRAKE
from .rss import d_min_long
from .sim import make_road, add_ego, add_scripted, Runner, scripted, shielded
from .shield import RSSShield, ShieldConfig

X0 = 100.0


# ---------------------------------------------------------------- scenario 1A
def scenario1A(rho, phase, slower_instead=None, T=12.0):
    """Execution chain. Ego and leader at 25 m/s, d0 = d_min_long(25,25,rho), FASTER active
    before t=0 (target 30 m/s, controller 8.33 m/s^2 clipped to +3). Leader brakes -8 m/s^2
    from the first physics frame. Detection tick is scripted: sync -> tick 0, worst -> tick 1.
    slower_instead: None | "once" (one SLOWER at the detection tick, then IDLE) |
                    "every_tick" (SLOWER on every tick from the detection tick)."""
    road = make_road()
    d0 = d_min_long(25.0, 25.0, rho)
    ego = add_ego(road, 1, X0, 25.0, target_speed=30.0, initial_active="FASTER")
    lead = add_scripted(road, 1, X0 + LENGTH + d0, 25.0, "L", accel=lambda t: -A_MAX_BRAKE)
    det = 0 if phase == "sync" else 1
    if slower_instead is None:
        sched = lambda k, t: TAKEOVER if k >= det else "IDLE"
    elif slower_instead == "once":
        sched = lambda k, t: "SLOWER" if k == det else "IDLE"
    else:
        sched = lambda k, t: "SLOWER" if k >= det else "IDLE"
    r = Runner(road, ego, scripted(sched), rho,
               name=f"S1A_rho{rho}_{phase}" + (f"_SLOWER_{slower_instead}" if slower_instead else ""))
    r.run(T, stop=lambda rr: rr.ego.speed <= 1e-12 and lead.speed <= 1e-12)
    last = r.rows[-1]
    r.result = dict(rho=rho, phase=phase, d0=d0, detect_tick=det, crashed=bool(ego.crashed),
                    t_contact=r.t_contact,
                    final_gap=None if ego.crashed else float(lead.position[0] - ego.position[0] - LENGTH),
                    ego_path=float(ego.position[0] - X0), lead_path=float(lead.position[0] - (X0 + LENGTH + d0)))
    return r


# ---------------------------------------------------------------- scenario 1B
def scenario1B(rho, long_prediction, margin=0.5, T=1.2):
    """Detection logic. Equal speeds 25 m/s, ego IDLE (target 25), d0 = d_min_long + margin;
    leader brakes -8 m/s^2 from the first frame after t=0. Policy proposes IDLE every tick.
    Returns the first tick with A_safe = empty (TAKEOVER)."""
    road = make_road()
    d0 = d_min_long(25.0, 25.0, rho) + margin
    ego = add_ego(road, 1, X0, 25.0, target_speed=25.0, initial_active="IDLE")
    lead = add_scripted(road, 1, X0 + LENGTH + d0, 25.0, "L", accel=lambda t: -A_MAX_BRAKE)
    sh = RSSShield(ego, ShieldConfig(rho=rho, long_prediction=long_prediction))
    r = Runner(road, ego, shielded(sh, lambda k, t: "IDLE"), rho,
               name=f"S1B_rho{rho}_{long_prediction}")
    r.run(T)
    ticks = [tick for tick, t, d in r.decisions if d.command == TAKEOVER]
    r.result = dict(rho=rho, long_prediction=long_prediction, d0=d0,
                    detect_tick=ticks[0] if ticks else None)
    return r


# ---------------------------------------------------------------- stop / release
def scenario_stop(v0=10.0, T_brake=4.0, T_hold=2.0):
    """S1 braking to standstill in takeover, S2 holding v = 0 under takeover."""
    road = make_road()
    ego = add_ego(road, 1, X0, v0, target_speed=v0, initial_active="IDLE")
    r = Runner(road, ego, scripted(lambda k, t: TAKEOVER), 0.4, name="S1S2_stop_hold")
    r.run(T_brake + T_hold)
    return r


def scenario_release(v0=20.0, release_tick=5, then="IDLE", T=4.0, standstill=False):
    """S3 release of takeover: takeover in progress at t = 0 (target speed 30 m/s left from an
    earlier FASTER), TAKEOVER on ticks < release_tick, then `then` every tick."""
    road = make_road()
    ego = add_ego(road, 1, X0, 0.0 if standstill else v0, target_speed=30.0,
                  initial_active=TAKEOVER)
    r = Runner(road, ego, scripted(lambda k, t: TAKEOVER if k < release_tick else then), 0.4,
               name=f"S3_release_{'standstill' if standstill else 'moving'}_{then}")
    r.run(T)
    r.release_tick = release_tick
    return r


# ---------------------------------------------------------------- random limiter stress
def scenario_random(seed=0, T=60.0):
    rng = np.random.RandomState(seed)
    cmds = ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT", TAKEOVER]
    p = [0.25, 0.25, 0.2, 0.05, 0.05, 0.2]
    road = make_road()
    ego = add_ego(road, 1, X0, 15.0, target_speed=15.0, initial_active="IDLE")
    seq = rng.choice(cmds, size=int(T / 0.2) + 1, p=p)
    r = Runner(road, ego, scripted(lambda k, t: str(seq[k])), 0.4, name=f"A1_random_seed{seed}")
    r.run(T)
    return r


# ---------------------------------------------------------------- scenario 2
EGO_Y_LANE = 1


def scenario2a(rho=0.4, T=3.0):
    """Ego lane 1, 20 m/s; rear vehicle R in target lane 0, 18 m behind (bumper gap), 28 m/s,
    constant speed (scripted). Policy asks LANE_LEFT every tick."""
    road = make_road()
    ego = add_ego(road, 1, X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 0, X0 - LENGTH - 18.0, 28.0, "R")
    sh = RSSShield(ego, ShieldConfig(rho=rho))
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_LEFT"), rho, name="S2a")
    r.run(T)
    return r


def scenario2b(variant="i", rho=0.4, t_b=0.2, model="observed", T=5.0, gap_n=40.0,
               rear_margin=1.0, rear_reactive=True):
    """Ego lane 1 -> lane 2 (LANE_RIGHT at tick 0) at 20 m/s. Neighbour N in lane 3, gap_n m
    ahead (bumper gap), 20 m/s; at t_b it starts a lane change into lane 2.
    variant "i":   origin lane empty -> return (ABORT) expected;
    variant "iii": rear vehicle R in origin lane 1 at 28 m/s, bumper gap
                   d_min_long(28, 20, rho) + rear_margin behind the ego (R complies with RSS at
                   t = 0). With worst-case prediction of R (+A_MAX_ACCEL) the return is
                   inadmissible, completion too -> FALLBACK expected. rear_reactive=True: R is an
                   RSS-compliant follower (brakes -A_MIN_BRAKE after rho when its gap < d_min);
                   False: R keeps 28 m/s (non-reacting, any contact is CR_env by the rear car).
    model: adjacent_lane_model of the shield ("observed" / "occupancy")."""
    road = make_road()
    ego = add_ego(road, 1, X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 3, X0 + LENGTH + gap_n, 20.0, "N", lane_change=(t_b, ("0", "1", 2)))
    if variant == "iii":
        gap_r = d_min_long(28.0, 20.0, rho) + rear_margin
        add_scripted(road, 1, X0 - LENGTH - gap_r, 28.0, "R",
                     rss_follow=rho if rear_reactive else None)
    sh = RSSShield(ego, ShieldConfig(rho=rho, adjacent_lane_model=model))
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_RIGHT" if k == 0 else "IDLE"), rho,
               name=f"S2b_{variant}_{model}")
    r.run(T, stop_after_crash=0.5)
    return r


# ---------------------------------------------------------------- P0 lane change profile
def profile_lane_change(v=20.0, T=4.0):
    """Lateral profile of the standard highway-env lane change (ControlledVehicle cascade,
    same controller as MOBIL lane changes of background traffic)."""
    road = make_road()
    veh = add_scripted(road, 1, X0, v, "P", lane_change=(0.0, ("0", "1", 2)))
    ys, vys = [], []
    for k in range(int(T / DT)):
        road.act(); road.step(DT)
        ys.append(float(veh.position[1])); vys.append(float(veh.speed * np.sin(veh.heading)))
    ys, vys = np.array(ys), np.array(vys)
    a_lat = np.diff(vys) / DT
    done = np.argmax(np.abs(ys - 8.0) < 0.1) if np.any(np.abs(ys - 8.0) < 0.1) else None
    return dict(v=v, duration=None if done is None else (done + 1) * DT,
                vlat_peak=float(np.max(np.abs(vys))), alat_peak=float(np.max(np.abs(a_lat))))


# ---------------------------------------------------------------- v11 review checks (Q1-Q4)
def scenario_q1_scene(rho=0.4):
    """Q1 (v11, item 1): ego lane 1, 20 m/s (target 20); R in lane 3, 10 m behind (bumper gap),
    25 m/s, no reaction. Returns (road, ego, shield) before tick 0."""
    road = make_road()
    ego = add_ego(road, 1, X0, 20.0, target_speed=20.0, initial_active="IDLE")
    add_scripted(road, 3, X0 - LENGTH - 10.0, 25.0, "R")
    return road, ego, RSSShield(ego, ShieldConfig(rho=rho, adjacent_lane_model="observed"))


def scenario_q1_double_lane_change(rho=0.4, T=4.0):
    """Q1: the policy asks LANE_RIGHT on every tick. At tick 1 the first LANE_RIGHT is still in
    the FIFO (maneuver not active); it must count as a maneuver in progress, so a second lane
    change (lane 2 -> 3, R closing in from behind) is not admissible."""
    road, ego, sh = scenario_q1_scene(rho)
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_RIGHT"), rho, name="Q1_double_lane_right")
    r.run(T)
    return r


# (v, gap_n, t_b) for which the v10 shield issued ABORT on a tick before the crossing that became
# active after it (decision tick + 0.2 s); found by a sweep around scenario 2b-i
Q2_LATE_ABORT_CASES = ((8.0, 15.0, 0.0), (10.0, 15.0, 0.0), (10.0, 15.0, 0.2), (14.0, 30.0, 0.0),
                       (14.0, 30.0, 0.2), (16.0, 25.0, 0.4), (16.0, 30.0, 0.4), (20.0, 40.0, 0.4))


def scenario_q2_abort_crossing(v=10.0, gap_n=15.0, t_b=0.0, rho=0.4, T=3.0):
    """Q2 (v11, item 1): scenario 2b-i at speed v. Ego lane 1 -> 2 (LANE_RIGHT at tick 0, then
    IDLE); N in lane 3, bumper gap gap_n ahead, speed v, starts a lane change into lane 2 at t_b.
    v10, (10, 15, 0): ABORT decided at 0.4 s, active at 0.6 s, crossing on frame 5."""
    road = make_road()
    ego = add_ego(road, 1, X0, v, target_speed=v, initial_active="IDLE")
    add_scripted(road, 3, X0 + LENGTH + gap_n, v, "N", lane_change=(t_b, ("0", "1", 2)))
    sh = RSSShield(ego, ShieldConfig(rho=rho, adjacent_lane_model="observed"))
    r = Runner(road, ego, shielded(sh, lambda k, t: "LANE_RIGHT" if k == 0 else "IDLE"), rho,
               name=f"Q2_abort_v{v:g}_gap{gap_n:g}_tb{t_b:g}")
    r.run(T, stop_after_crash=0.5)
    return r


def scenario_q2_scripted_abort(v=10.0, T=3.0):
    """Q2b: no shield; scripted LANE_RIGHT at tick 0 and ABORT at tick 2 (t = 0.4 s, before the
    crossing on frame 5). The ABORT leaves the FIFO at 0.6 s, after the crossing: the vehicle
    must not return (abort_ignored)."""
    road = make_road()
    ego = add_ego(road, 1, X0, v, target_speed=v, initial_active="IDLE")
    sched = {0: "LANE_RIGHT", 2: ABORT}
    r = Runner(road, ego, scripted(lambda k, t: sched.get(k, "IDLE")), 0.4, name="Q2b_scripted_abort")
    r.run(T)
    return r


def scenario_q3_far_leader(rho=1.0, v=30.0, gap=173.0):
    """Q3 (v11, item 2): ego 30 m/s (target 30), standing leader in the ego lane, bumper gap
    173 m > d_min_long(30, 0, 1.0) = 167.625 m, i.e. beyond the v10 fixed zone of 150 m.
    Returns (road, ego, shield) before tick 0."""
    road = make_road()
    ego = add_ego(road, 1, X0, v, target_speed=v, initial_active="IDLE")
    add_scripted(road, 1, X0 + LENGTH + gap, 0.0, "L")
    return road, ego, RSSShield(ego, ShieldConfig(rho=rho))


def scenario_q3_random(seed, n=40, ticks=1):
    """Q3: random scene for the zone completeness check: up to n vehicles within 600 m of the ego
    in 4 lanes, 0-40 m/s, some changing lanes or with constant acceleration; ego 0-30 m/s with a
    random command issued `ticks` ticks before the check. Same-lane centre spacing >= 30 m (40 m
    to the ego), so no contact occurs in the <= 0.4 s simulated before the check. Returns
    (road, ego)."""
    rng = np.random.RandomState(seed)
    road = make_road()
    xe = X0 + 1000.0
    lane_e = 1 + int(rng.randint(2))
    ego = add_ego(road, lane_e, xe, float(rng.uniform(0.0, 30.0)),
                  target_speed=float(5 * rng.randint(7)), initial_active="IDLE")
    placed = {lane: [] for lane in range(4)}
    for i in range(n):
        lane = int(rng.randint(4))
        x = xe + float(rng.uniform(-600.0, 600.0))
        v = float(rng.uniform(0.0, 40.0))
        lc = None
        if rng.rand() < 0.25:
            to = lane + 1 if lane == 0 or (lane < 3 and rng.rand() < 0.5) else lane - 1
            lc = (float(rng.choice([0.0, 0.2, 0.4])), ("0", "1", to))
        acc = float(rng.uniform(-8.0, 3.0))
        const_acc = rng.rand() < 0.3
        if (lane == lane_e and abs(x - xe) < 40.0) or any(abs(x - xp) < 30.0 for xp in placed[lane]):
            continue
        placed[lane].append(x)
        add_scripted(road, lane, x, v, f"V{i}", accel=(lambda t, a=acc: a) if const_acc else None,
                     lane_change=lc)
    for _ in range(ticks):
        ego.act(str(rng.choice(["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"])))
        for _ in range(2):
            road.act()
            road.step(DT)
    return road, ego
