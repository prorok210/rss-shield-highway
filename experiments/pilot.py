"""Short end-to-end pilot of the training chain (protocol, section 8).

The filter runs with the main-series configuration MAIN_SERIES (rss_shield.shield: adjacent-lane
model "observed", nominal prediction of the vehicles in the lanes of a lane change, lateral rule
"approach") unless stated otherwise.

1. warm-up check (7.6): mean flow speed over 10-15 s and 15-20 s of the warm-up, D = 40; the
   criterion is checked on the pooled seeds, the per-seed values are descriptive;
2. start states for D = 10 / 25 / 40: removed vehicles, insertion, start labels for every rho
   (a: rule of 7.6 including adjacent lanes, lateral overlap or approach as in the filter and in
   t_danger, for reference; b: leader and follower in the ego lane only, the main label);
3. short training of Proposed (mode B), rho = 0.6 s, D = 25, network seed 42;
4. replay records before, during and after the first intercept with the reward of every covered
   decision step (a record that starts from a moving state is preferred); check of all stored
   transitions against the step log;
5. saving and loading the model;
6. evaluation episodes of the loaded model (epsilon = 0) with the step journal and the counters;
7. Baselines: short training of Baseline 1 (R_v) and Baseline 2 (R_RS) without the filter on the
   training seeds (every transition has n = 1), evaluation on the evaluation seeds; mode A: the
   frozen Baseline 1 with the filter, with the mode-A mapping of the queue features and without
   it (A-raw);
8. scripted lane-change seeker (prefers a lane change, otherwise IDLE; not trained) on the main
   generator, D = 10 / 25, rho = 0.1 and 0.6 s: executed lane changes, ABORT and the journal
   around it;
9. diagnostic of the lateral rule for adjacent lanes: a fixed IDLE policy at rho = 0.6 and 1.0 on
   the same start states, rule "approach" (main series) and strict RSS ("rss");
10. admissibility of lane changes on the main generator: share of decision ticks with LANE_LEFT or
    LANE_RIGHT in A_safe over the (D, rho) grid; the main configuration drives (fixed policy IDLE,
    otherwise SLOWER), the masks of four configurations (adjacent-lane model x prediction in the
    lanes of a lane change) are evaluated on the same states, with the blocking reasons.
H1-H3 are not tested by the pilot.

    python -m experiments.pilot --out runs/pilot_ci        (CI)
    python -m experiments.pilot --out /tmp/pilot --quick   (smoke run)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from experiments.dqn import DQN  # noqa: E402
from experiments import h1  # noqa: E402
from experiments.env import FAST_ROAD, THREAD_VARS, make_env, reset_cached  # noqa: E402
from experiments.generator import (I_BRAKE, I_LANE, I_SPEED, N_KIN, QUEUE_CODES, RHOS,  # noqa: E402
                                   SPEED_SEEN, initially_unsafe, window_bounds)
from experiments.runner import JERK_JUMP, NAMES, mask_str, run_episode  # noqa: E402
from rss_shield.accounting import LANE_CHANGES  # noqa: E402
from rss_shield.params import ABORT, ACTION_INDEX, LANE_WIDTH, WIDTH  # noqa: E402
from rss_shield.rss import d_min_lat  # noqa: E402
from rss_shield.shield import MAIN_SERIES, RSSShield, ShieldConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.99
SEEDS = {
    "warmup": list(range(900_000, 900_020)),    # D = 40: warm-up check and start states
    "start10": list(range(900_020, 900_030)),   # D = 10
    "start25": list(range(900_030, 900_040)),   # D = 25
    "train": list(range(900_040, 900_090)),     # training episodes (cycled; Proposed, Baselines 1 / 2)
    "diag": list(range(900_092, 900_095)),      # lateral-rule diagnostic, rho = 0.6 vs 1.0
    "eval": list(range(900_100, 900_110)),      # evaluation: Proposed, Baselines 1 / 2, mode A, A-raw
    "seek10": list(range(900_110, 900_120)),    # lane-change seeker, D = 10
    "seek25": list(range(900_120, 900_130)),    # lane-change seeker, D = 25
}                                               # 900 090-091 and 900 095-099 are not used
QUICK_SEEDS = {"warmup": 3, "start10": 2, "start25": 2, "train": 50, "diag": 1, "eval": 2, "seek10": 2,
               "seek25": 2}
LC_GRID_STEPS = 30          # decision steps per start state in the lane-change grid
LC_GRID_STARTS = 10         # cached start states per D in the lane-change grid
LC_PREF = {"LANE_LEFT": 0.0, "IDLE": 1.0, "LANE_RIGHT": 0.0, "FASTER": 0.0, "SLOWER": 0.5}
# configurations of the lane-change grid, (adjacent_lane_model, lc_prediction): the first one is
# the main series (MAIN_SERIES) and drives the episodes, the last one is the former default
LC_CONFIGS = (("observed", "nominal"), ("observed", "worst_case"), ("occupancy", "nominal"),
              ("occupancy", "worst_case"))
LC_CONFIG_NAMES = {("observed", "nominal"): "observed, номинальный (основная серия)",
                   ("observed", "worst_case"): "observed, худший случай (контроль)",
                   ("occupancy", "nominal"): "occupancy, номинальный",
                   ("occupancy", "worst_case"): "occupancy, худший случай (прежняя по умолчанию)"}
BASE_STEPS = 1500           # short training of Baselines 1 / 2 without the filter, decision steps
BASE_LEARNING_STARTS = 500
A_RAW_EPISODES = 2          # A-raw evaluation episodes (descriptive)
SEEK_RHOS = (0.1, 0.6)      # rho of the lane-change seeker, s
V_MOVING = 5.0              # replay record from a moving state: v and v_target >= 5 m/s
REWARD_NAMES = {"Rv": "R_v", "RS": "R_RS"}
RULE_NAMES = {"approach": "перекрытие или сближение (основная)", "rss": "строгое RSS"}


# ---------------------------------------------------------------------- formatting
def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "да" if x else "нет"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------------- journal (README, 4)
def machine_journal():
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "not installed"

    cpu = platform.processor() or "unknown"
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        elif sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, check=False).stdout.strip() or cpu
    except OSError:
        pass
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO, check=False).stdout.strip()
    except OSError:
        head = ""
    j = {"machine": platform.node(), "os": platform.platform(), "cpu": cpu, "cores": os.cpu_count(),
         "python": platform.python_version(), "numpy": ver("numpy"),
         "highway-env": ver("highway-env"), "gymnasium": ver("gymnasium"),
         "commit (HEAD)": head or "unknown", "PR head": os.environ.get("PR_HEAD_SHA", ""),
         "fast_road": "enabled" if FAST_ROAD else "disabled (RSS_FAST_ROAD=0)"}
    j.update({v: os.environ.get(v, "") for v in THREAD_VARS})
    return j


# ---------------------------------------------------------------------- phases
def start_states(D, seeds, cache):
    env = make_env(D=D)
    infos = []
    for s in seeds:
        t0 = time.perf_counter()
        reset_cached(env, s, cache)
        infos.append(dict(env.start_info, seed=s, t_reset=time.perf_counter() - t0))
    return infos


def warmup_check(infos):
    v1 = np.array([i["warmup_10_15"] for i in infos])
    v2 = np.array([i["warmup_15_20"] for i in infos])
    rel = np.abs(v2 - v1) / v1
    pooled = abs(v2.mean() - v1.mean()) / v1.mean()
    return dict(n=len(infos), v10_15=float(v1.mean()), v15_20=float(v2.mean()), pooled=float(pooled),
                max_seed=float(rel.max()), n_seed_ok=int((rel < 0.02).sum()), passed=bool(pooled < 0.02))


def start_summary(D, infos):
    return dict(D=D, n=len(infos),
                removed=[sum(1 for i in infos if i["removed"] == m) for m in (0, 1, 2)],
                ok=sum(1 for i in infos if i["insert_ok"]),
                unsafe_a={r: sum(1 for i in infos if i["unsafe"][r]) for r in RHOS},
                unsafe_b={r: sum(1 for i in infos if i["unsafe_lane"][r]) for r in RHOS},
                density=float(np.mean([i["density"] for i in infos])),
                speed=float(np.mean([i["mean_speed"] for i in infos])),
                t_reset=float(np.mean([i["t_reset"] for i in infos])))


def train(args, seeds, cache, log, shield_on=True, reward_kind="Rv", steps=None, max_steps=None,
          learning_starts=None, tag="Proposed"):
    """Training on the cycled training seeds. Proposed (shield_on=True): the budget is extended up
    to max_steps while the buffer holds no transition with n > 1. Baselines (no filter, every
    transition has n = 1): exactly the budget."""
    env = make_env(D=args.D)
    steps = args.steps if steps is None else steps
    max_steps = args.max_steps if max_steps is None else max_steps
    agent = DQN(seed=args.seed,
                learning_starts=args.learning_starts if learning_starts is None else learning_starts)
    steps_log, episodes = [], []
    ep = 0
    t0 = time.perf_counter()
    while True:
        seed = seeds[ep % len(seeds)]
        c = run_episode(env, agent, rho=args.rho, shield_on=shield_on, reward_kind=reward_kind,
                        train=True, explore=True, episode=ep,
                        reset=lambda s=seed: reset_cached(env, s, cache), steps_log=steps_log)
        c.update(episode=ep, seed=seed, total_steps=agent.steps, epsilon=agent.epsilon(),
                 buffer=agent.buffer.size, loss=agent.last_loss)
        episodes.append(c)
        log(f"  {tag} episode {ep:3d} seed {seed} steps {c['steps']:3d} policy {c['E']:3d} "
            f"intercept {c['intercept']:2d} crash {int(c['crashed'])} return {c['ret']:7.2f} | "
            f"total {agent.steps:5d} eps {agent.epsilon():.3f} buffer {agent.buffer.size:5d} "
            f"loss {fmt(agent.last_loss, 4)}")
        ep += 1
        multi = bool((agent.buffer.n[:agent.buffer.size] > 1).any())
        if agent.steps >= steps and (multi or not shield_on or agent.steps >= max_steps):
            break
    return agent, steps_log, episodes, time.perf_counter() - t0


def check_transitions(agent, steps_log):
    """Every stored transition against the step log: starts at a policy step, covers only
    intercept steps after it, ends before the next policy step, R = sum gamma^j r. Also counts the
    supervisory commands of the intercept steps covered by transitions with n > 1."""
    b = agent.buffer
    rows = {(r["episode"], r["k"]): r for r in steps_log}
    bad = dict(start=0, inner=0, next=0)
    inner_cmds = {}
    max_dr = 0.0
    for i in range(b.size):
        e, k0 = b.meta[i]
        n = int(b.n[i])
        bad["start"] += int(rows[(e, k0)]["policy_step"] != 1)
        bad["inner"] += int(any(rows[(e, k0 + j)]["policy_step"] != 0 for j in range(1, n)))
        for j in range(1, n):
            cmd = rows[(e, k0 + j)]["command"]
            inner_cmds[cmd] = inner_cmds.get(cmd, 0) + 1
        nxt = rows.get((e, k0 + n))
        bad["next"] += int(nxt is not None and nxt["policy_step"] != 1)
        r = sum(GAMMA ** j * rows[(e, k0 + j)]["r"] for j in range(n))
        max_dr = max(max_dr, abs(r - float(b.R[i])))
    ns = b.n[:b.size]
    d = b.d[:b.size]
    return rows, dict(bad=bad, max_dR=max_dr, hist={int(v): int((ns == v).sum()) for v in np.unique(ns)},
                      d1=int(d.sum()), d1_multi=int((d & (ns > 1)).sum()), inner_cmds=inner_cmds)


def decode(s):
    e = s[N_KIN:]
    q = [QUEUE_CODES[i] for i in range(len(QUEUE_CODES)) if e[i] > 0.5]
    return (f"FIFO {q[0] if q else '—'}, brake_mode {int(e[I_BRAKE] > 0.5)}, "
            f"v_target {5 * int(round(e[I_SPEED] * 6))} м/с, "
            f"целевая − текущая полоса {int(round(e[I_LANE])):+d}, "
            f"v ≈ {float(s[3]) * 80.0:.2f} м/с")


def moving(s):
    """The ego speed and the target speed decoded from an observation are both >= V_MOVING."""
    e = s[N_KIN:]
    return float(s[3]) * 80.0 >= V_MOVING and 5 * int(round(e[I_SPEED] * 6)) >= V_MOVING


def records_section(agent, rows):
    b = agent.buffer
    cands = [i for i in range(b.size) if b.n[i] > 1]
    # an intercept transition with a non-zero discounted sum that starts from a moving state is
    # preferred; otherwise the conditions are relaxed in this order
    nz = [i for i in cands if abs(b.R[i]) > 1e-9]
    good = [i for i in nz if moving(b.s[i])]
    first = good[0] if good else nz[0] if nz else cands[0] if cands else None
    if first is None:
        return None, ("Переходов с n > 1 в буфере нет: "
                      "перехват за бюджет не встретился.")
    if good:
        how = (f"показан первый из них с R ≠ 0, у которого в s скорость ego и целевая скорость "
               f"не ниже {V_MOVING:g} м/с")
    else:
        how = (f"переходов с R ≠ 0 из состояния со скоростью и целевой скоростью не ниже "
               f"{V_MOVING:g} м/с нет; показан первый переход с n > 1" + (" и R ≠ 0" if nz else ""))
    parts = [f"Переходов с n > 1: {len(cands)}, из них с R ≠ 0: {len(nz)}, из них из движущегося "
             f"состояния: {len(good)}; {how}."]
    ep0 = b.meta[first][0]
    idxs = [i for i in (first - 1, first, first + 1) if 0 <= i < b.size and b.meta[i][0] == ep0]
    names = {-1: "до перехвата", 0: "с перехватом",
             1: "после перехвата"}
    for i in idxs:
        e, k0 = b.meta[i]
        n = int(b.n[i])
        parts.append(f"**Запись {i} ({names[i - first]})**: эпизод {e}, s = s_{k0}, "
                     f"a = {NAMES[int(b.a[i])]}, n = {n}, R = {b.R[i]:.6f}, d = {int(b.d[i])}, "
                     f"маска(s′) = {mask_str(b.m2[i])}\n\n"
                     f"- s: {decode(b.s[i])}\n- s′: {decode(b.s2[i])}")
        steps, acc = [], 0.0
        for j in range(n):
            r = rows[(e, k0 + j)]
            acc += GAMMA ** j * r["r"]
            steps.append([k0 + j, f"{r['t']:.1f}",
                          "политика" if r["policy_step"] else "перехват",
                          r["command"], r["reason"], r["mask"], r["queued"] or "—",
                          r["active_cmd"] or "—", r["brake_mode"], f"{r['r']:.5f}",
                          f"{GAMMA ** j * r['r']:.5f}"])
        parts.append(table(["такт", "t, с", "шаг", "команда", "причина", "маска",
                            "в FIFO", "активна после шага", "brake_mode", "r",
                            "γ^j·r"], steps))
        nxt = rows.get((e, k0 + n))
        tail = ("конец эпизода" if nxt is None else
                "шаг политики, переход закрыт на нём" if nxt["policy_step"]
                else "перехват")
        parts.append(f"Σ γ^j·r = {acc:.6f}; R в буфере = {b.R[i]:.6f}; "
                     f"|Δ| = {abs(acc - b.R[i]):.1e}. Такт {k0 + n}: {tail}.")
    return first, "\n\n".join(parts)


def save_load(agent, out):
    path = out / "model.npz"
    agent.save(path)
    loaded = DQN.load(path)
    S = agent.buffer.s[:min(256, agent.buffer.size)]
    qa, qb = agent.q_batch(S), loaded.q_batch(S)
    ta, tb = agent.q_batch(S, agent.target), loaded.q_batch(S, loaded.target)
    return loaded, dict(path=str(path), size=path.stat().st_size, n_states=len(S),
                        dq=float(np.max(np.abs(qa - qb))), dqt=float(np.max(np.abs(ta - tb))),
                        same_greedy=bool(np.array_equal(qa.argmax(1), qb.argmax(1))),
                        steps=loaded.steps, updates=loaded.updates)


def evaluate(agent, args, seeds, cache, shield_on=True, reward_kind="Rv", env_config=None,
             journal=True):
    """Evaluation episodes (epsilon = 0) on the cached start states; env_config: extra environment
    configuration, e.g. obs_mode_a=True for mode A."""
    env = make_env(D=args.D, **(env_config or {}))
    rows, stats = [], []
    for i, s in enumerate(seeds):
        c = run_episode(env, agent, rho=args.rho, shield_on=shield_on, reward_kind=reward_kind,
                        explore=False, episode=i, reset=lambda s=s: reset_cached(env, s, cache),
                        journal=rows if journal else None, classify=True)
        c["seed"] = s
        stats.append(c)
    return rows, stats


class IdlePolicy:
    """Fixed policy that always prefers IDLE (diagnostic of the filter, not trained)."""

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.cfg = {"gamma": GAMMA}

    def q_values(self, obs):
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    def epsilon(self):
        return 0.0


class LaneChangeSeeker:
    """Scripted pilot policy (not trained): prefers a lane change - the direction is drawn anew on
    every decision step - otherwise IDLE, then SLOWER. Exercises executed lane changes and ABORT on
    the main generator; the shield executes the preferred admissible
    action."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.cfg = {"gamma": GAMMA}

    def q_values(self, obs):
        u = float(self.rng.random())
        return np.array([0.6 + 0.4 * u, 0.5, 1.0 - 0.4 * u, 0.0, 0.1])

    def epsilon(self):
        return 0.0


def seek(seeds_by_D, cache, rhos, log):
    """Lane-change seeker on the cached start states (D: seeds) for every rho of rhos; the same start
    states for every rho; every episode runs to its end."""
    stats, journal = [], []
    envs = {D: make_env(D=D) for D in seeds_by_D}
    ep = 0
    for rho in rhos:
        for D, seeds in seeds_by_D.items():
            env = envs[D]
            for s in seeds:
                j = []
                c = run_episode(env, LaneChangeSeeker(s), rho=rho, episode=ep,
                                reset=lambda s=s, env=env: reset_cached(env, s, cache), journal=j)
                for r in j:
                    r["D"], r["rho"], r["seed"] = D, rho, s
                journal += j
                c.update(D=D, rho=rho, seed=s, episode=ep)
                stats.append(c)
                log(f"  seeker D {D} rho {rho} seed {s}: steps {c['steps']} N_att_rss {c['N_att_rss']} "
                    f"N_blocked_rss {c['N_blocked_rss']} N_exec {c['N_exec']} ABORT {c['abort']} "
                    f"TAKEOVER {c['takeover']} FALLBACK {c['fallback']} crash {int(c['crashed'])}")
                ep += 1
    return stats, journal


def abort_fragment(journal, before=4, after=6):
    """Journal rows of the same episode around the first ABORT of the seeker."""
    hit = next((r for r in journal if r["command"] == ABORT), None)
    if hit is None:
        return None
    rows = [r for r in journal if r["episode"] == hit["episode"]
            and hit["k"] - before <= r["k"] <= hit["k"] + after]
    return dict(D=hit["D"], rho=hit["rho"], seed=hit["seed"], episode=hit["episode"], k=hit["k"],
                rows=rows)


def seek_lc_counts(journal):
    """Per (D, rho) of the seeker: lane-change commands executed on policy steps, of them in the
    direction opposite to a_pi (the chosen direction is blocked, the other one is admissible), and of
    them without a crossing before the end of the episode."""
    crossed = {(r["episode"], r["exec_from"]) for r in journal if r["exec_from"] != ""}
    out = {}
    for r in journal:
        if r["policy_step"] and r["command"] in LANE_CHANGES:
            o = out.setdefault((r["D"], r["rho"]), dict(D=r["D"], rho=r["rho"], lc_cmds=0, opposite=0,
                                                         no_cross=0))
            o["lc_cmds"] += 1
            o["opposite"] += int(r["a_pi"] in LANE_CHANGES and r["command"] != r["a_pi"])
            o["no_cross"] += int((r["episode"], r["k"]) not in crossed)
    return list(out.values())


def diagnostic(args, seeds, cache, rules=("approach", "rss")):
    """Fixed IDLE policy at rho = 0.6 and 1.0 on the same start states with the lateral rule of the
    main series (overlap or approach) and with strict RSS; start label (a) by the same rule."""
    env = make_env(D=args.D)
    out = []
    for s in seeds:
        for rho in (0.6, 1.0):
            for rule in rules:
                reset_cached(env, s, cache)
                label = initially_unsafe(env.vehicle, env.road, rho, rule=rule)
                j = []
                c = run_episode(env, IdlePolicy(), rho=rho, journal=j,
                                reset=lambda s=s: reset_cached(env, s, cache),
                                shield_cfg=dict(MAIN_SERIES, lateral_rule=rule))
                reasons = {}
                for r in j:
                    if r["intervention"]:
                        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
                out.append(dict(seed=s, rho=rho, rule=rule, steps=c["steps"], intercept=c["intercept"],
                                share=c["intercept"] / c["steps"],
                                empty=reasons.get("empty_safe_set", 0), masked=c["masked"],
                                crashed=c["crashed"], unsafe_rule=label, unsafe=c["unsafe_start"],
                                unsafe_lane=c["unsafe_start_lane"]))
    return out


def lane_change_grid(seeds_by_D, cache, n_steps=LC_GRID_STEPS, configs=LC_CONFIGS):
    """Admissibility of lane changes on the main generator: share of
    decision ticks with LANE_LEFT or LANE_RIGHT in A_safe over the (D, rho) grid. Fixed policy
    IDLE, otherwise SLOWER (LC_PREF, not trained); the same cached start states for every rho; at
    most n_steps decision steps per start state. The first configuration (main series) drives the
    episode, the masks of all configurations (adjacent_lane_model, lc_prediction) are evaluated on
    the same states. Blocks of each lane change are split by reason: front / rear (RSS violation
    with a vehicle ahead / behind; one block can have both), busy (maneuver in progress or
    queued), no_lane (no lane that way)."""
    assert dict(MAIN_SERIES, adjacent_lane_model=configs[0][0], lc_prediction=configs[0][1]) == MAIN_SERIES
    out = []
    for D, seeds in seeds_by_D.items():
        env = make_env(D=D)
        for rho in RHOS:
            G = {cf: dict(D=D, model=cf[0], lc_prediction=cf[1], rho=rho, episodes=len(seeds), ticks=0,
                          left=0, right=0, any=0, front=0, rear=0, busy=0, no_lane=0, intercept=0)
                 for cf in configs}
            for s in seeds:
                reset_cached(env, s, cache)
                shields = {cf: RSSShield(env.vehicle, ShieldConfig(
                    rho=rho, **dict(MAIN_SERIES, adjacent_lane_model=cf[0], lc_prediction=cf[1])))
                    for cf in configs}
                for _ in range(n_steps):
                    decs = {cf: sh.decide("IDLE", LC_PREF) for cf, sh in shields.items()}
                    for cf, dec in decs.items():
                        g = G[cf]
                        ok = {n: bool(dec.mask[ACTION_INDEX[n]]) for n in LANE_CHANGES}
                        g["ticks"] += 1
                        g["left"] += int(ok["LANE_LEFT"])
                        g["right"] += int(ok["LANE_RIGHT"])
                        g["any"] += int(ok["LANE_LEFT"] or ok["LANE_RIGHT"])
                        g["intercept"] += int(dec.intervention)
                        for n in LANE_CHANGES:
                            if ok[n]:
                                continue
                            why = dec.violations.get(n) or {}
                            if why.get("busy"):
                                g["busy"] += 1
                            elif why.get("no_lane"):
                                g["no_lane"] += 1
                            else:
                                g["front"] += int("front" in why)
                                g["rear"] += int("rear" in why)
                    _, _, term, trunc, _ = env.step(decs[configs[0]].command)
                    if term or trunc:
                        break
            for g in G.values():
                g["share"] = g["any"] / g["ticks"] if g["ticks"] else None
                out.append(g)
    return out


def totals(stats):
    def s(k):
        return sum(c[k] for c in stats)
    steps = s("steps")
    n_rss = s("N_att_rss")
    return dict(episodes=len(stats), steps=steps, E=s("E"), intercept=s("intercept"),
                takeover=s("takeover"), abort=s("abort"), fallback=s("fallback"), N_att=s("N_att"),
                N_blocked=s("N_blocked"), BLCR=(s("N_blocked") / s("N_att") if s("N_att") else None),
                N_busy=s("N_busy"), N_no_lane=s("N_no_lane"), N_att_rss=n_rss,
                N_blocked_rss=s("N_blocked_rss"), BLCR_rss=(s("N_blocked_rss") / n_rss if n_rss else None),
                N_att_int=s("N_att_int"), N_exec=s("N_exec"), N_exec_int=s("N_exec_int"),
                N_exec_other=s("N_exec_other"), masked=s("masked"),
                R_int=(s("masked") + s("intercept")) / steps if steps else None,
                crashes=s("crashed"), CR=s("crashed") / len(stats) if stats else None,
                crash_intercept=s("crash_intercept"), crash_sup_active=s("crash_sup_active"),
                crash_rear=s("crash_rear"), completed=s("completed"), abort_ignored=s("abort_ignored"),
                ttc_pen=s("ttc_pen"), col_pen=s("col_pen"),
                v_eff=float(np.mean([c["v_eff"] for c in stats])),
                jerk_rms=float(np.mean([c["jerk_rms"] for c in stats])),
                jerk_rms_smooth=float(np.mean([c["jerk_rms_smooth"] for c in stats])),
                jerk_jump_share=float(np.mean([c["jerk_jump_share"] for c in stats])),
                jerk_jumps=s("jerk_jumps"), frames=s("frames"),
                shield_cfg="; ".join(sorted({c["shield_cfg"] for c in stats})),
                main_series=(None if all(c["main_series"] is None for c in stats)
                             else all(c["main_series"] for c in stats)))


FINAL_COUNTERS = ("steps", "E", "intercept", "takeover", "abort", "fallback", "N_att", "N_blocked",
                  "N_busy", "N_no_lane", "N_att_int", "N_exec", "N_exec_int", "N_exec_other",
                  "masked", "abort_ignored", "crashes")


def final_counters(t):
    """One line with the counters of protocol 7.7 (rss_shield.accounting) of an arm."""
    return " ".join(f"{k}={t[k]}" for k in FINAL_COUNTERS) + f" | filter {t['shield_cfg']}"


def arm(name, filt, reward, tot, stats=None):
    """Totals of an evaluation arm; with the episodes - also its H1 numbers (experiments.h1)."""
    return dict(tot, arm=name, filter=filt, reward=reward,
                h1=None if stats is None else h1.h1_summary(stats))


# ---------------------------------------------------------------------- report
COUNTER_COLS = [("steps", "шагов"), ("E", "E"), ("intercept", "перехват"), ("takeover", "TAKEOVER"),
                ("abort", "ABORT"), ("fallback", "FALLBACK"), ("N_att", "N_att"), ("N_blocked", "N_blocked"),
                ("BLCR", "BLCR"), ("N_busy", "N_busy"), ("N_no_lane", "N_no_lane"),
                ("N_att_rss", "N_att_rss"), ("N_blocked_rss", "N_blocked_rss"), ("BLCR_rss", "BLCR_rss"),
                ("N_att_int", "N_att_int"), ("N_exec", "N_exec"), ("N_exec_int", "N_exec_int"),
                ("N_exec_other", "N_exec_other"), ("masked", "замена aπ"), ("R_int", "R_int"),
                ("crashed", "столкновение"), ("completed", "до T_max"), ("dx", "Δx, м"),
                ("v_eff", "v_eff, м/с"), ("jerk_rms", "JerkRMS"), ("jerk_rms_smooth", "JerkRMS без скачков"),
                ("jerk_jumps", "кадров со скачком")]


def report(args, R):
    L = []
    a = L.append
    a("# Пилот сквозной цепочки обучения (раздел 8 протокола)")
    a("")
    a(f"Proposed, Режим Б, D = {args.D} авт/км/полосу, ρ = {args.rho} с, R_v, сид сети {args.seed}, "
      f"бюджет {args.steps} шагов решения (продление до {args.max_steps}, пока в буфере "
      f"нет перехода с n > 1). Фильтр везде, где не оговорено иное, — в конфигурации основной серии "
      f"(MAIN_SERIES): модель соседней полосы observed, номинальный прогноз автомобилей в полосах "
      f"манёвра, боковое правило «перекрытие или сближение» (п. 7.3, 7.7). Сиды пилота "
      f"900 000–900 129 (900 090–900 091 и 900 095–900 099 не используются). Проверяется "
      f"работоспособность цепочки; H1–H3 пилотом не проверяются." + (" **Режим --quick.**" if args.quick else ""))
    a("")
    a(f"Фактическая конфигурация фильтра по счётчикам эпизодов (adjacent_lane_model / lc_prediction / "
      f"lateral_rule / зона): обучение — `{R['train_total']['shield_cfg']}`; оценочные плечи с фильтром — "
      + "; ".join(f"{m['arm']}: `{m['shield_cfg']}`" for m in R["arms"] if m["main_series"] is not None)
      + ". Пилот считается пройденным, только если во всех этих плечах использована конфигурация MAIN_SERIES.")
    a("")
    a(f"Событие и корректный старт H1 (п. 6, 7.6, 7.7; единое определение — `experiments/h1.py`). Старт корректен, "
      f"если при t = 0 выполнен продольный порог RSS при ρ = {h1.RHO_START:g} с к лидеру и замыкающему в полосе ego — "
      f"одно правило для всех плеч и всех ρ. Столкновения плеч с фильтром классифицируются при ρ фильтра, плеч без "
      f"фильтра (Baseline 1, Baseline 2) — при ρref = {h1.RHO_REF:g} с. Событие H1 — столкновение категории CR_ego_op в "
      f"эпизоде с корректным стартом. CR_total считается по факту столкновения, независимо от категорий.")
    a("")
    a(table(["плечо", "эпизодов", "корректных стартов", "ρ классификации, с", "столкновений", "CR_total",
             "CR_total (корректные старты)", "событий H1 (CR_ego_op)", "категории столкновений"],
            [[m["arm"], m["h1"]["n"], m["h1"]["n_valid"], m["h1"]["rho_cls"], m["h1"]["crashes"],
              m["h1"]["CR_total"], m["h1"]["CR_total_valid"], m["h1"]["h1_events"],
              "; ".join(f"{k}: {v}" for k, v in sorted(m["h1"]["categories"].items())) or "—"]
             for m in R["arms"]]))
    z = window_bounds()
    a("")
    a(f"Зона проверки соседей (п. 7.4). Фильтр отбирает соседей по полной зоне reach(v, N) для "
      f"каждого автомобиля (ShieldConfig.zone = None, без фиксированного радиуса). Физически зону "
      f"ограничивает окно трафика генератора [−{z['back']:g}; +{z['front']:g}] м — это явное "
      f"приближение с областью применимости ρ ≤ {z['rho_max']:g} с, скорость ego ≤ {z['v_ego']:g} м/с, "
      f"скорость остальных автомобилей ≤ {z['v_traffic']:g} м/с: при этих границах требуется "
      f"[−{z['need_back']:.3f}; +{z['need_front']:.3f}] м. Окно проверяется при каждом reset() "
      f"(ZoneError, если оно меньше границы), скорости — на каждом шаге (EnvelopeError); в этом "
      f"прогоне наибольшая скорость ego {R['vmax']['ego']:.2f} м/с, остальных автомобилей "
      f"{R['vmax']['traffic']:.2f} м/с.")
    w = R["warmup"]
    a("")
    a(f"## 1. Прогрев (п. 7.6): D = 40, {w['n']} сидов")
    a("")
    a(table(["средняя скорость 10–15 с, м/с", "15–20 с, м/с", "разница по объединённой выборке",
             "максимум по сиду", "сидов < 2 %", "критерий < 2 % (объединённая выборка)"],
            [[w["v10_15"], w["v15_20"], f"{100 * w['pooled']:.2f} %", f"{100 * w['max_seed']:.2f} %",
              f"{w['n_seed_ok']} из {w['n']}", "выполнен" if w["passed"] else "НЕ выполнен"]]))
    a("")
    a("Критерий п. 7.6 проверяется по объединённой выборке: средние скорости потока за 10–15 и 15–20 с "
      "по всем сидам. Разница по отдельным сидам приводится описательно и в критерий не входит.")
    a("")
    a("## 2. Стартовые состояния")
    a("")
    a("(б) — основная метка п. 7.6: лидер и догоняющий в полосе ego. (а) — справочная метка с "
      "соседними полосами: продольный и боковой пороги нарушены одновременно при боковом перекрытии "
      "или боковом сближении (тот же критерий, что у фильтра и у t_danger, п. 7.7). Число исходно "
      "некорректных стартов по ρ.")
    a("")
    rows = []
    for s in R["starts"]:
        rows.append([s["D"], s["n"], "/".join(str(x) for x in s["removed"]), f"{s['ok']} из {s['n']}",
                     " / ".join(str(s["unsafe_a"][r]) for r in RHOS),
                     " / ".join(str(s["unsafe_b"][r]) for r in RHOS), s["density"], s["speed"],
                     s["t_reset"]])
    a(table(["D", "эпизодов", "удалено 0/1/2", "вставка при ρ = 1,0",
             "(а) ρ = " + "/".join(str(r) for r in RHOS), "(б) ρ = " + "/".join(str(r) for r in RHOS),
             "плотность в окне", "скорость в окне, м/с", "reset, с"], rows))
    T = R["train"]
    a("")
    a("## 3. Обучение")
    a("")
    a(table(["шагов решения", "эпизодов", "шагов политики", "шагов перехвата",
             "столкновений", "переходов в буфере", "отброшено при усечении",
             "градиентных шагов", "ε в конце", "последний loss", "время, с"],
            [[T["steps"], T["episodes"], T["E"], T["intercept"], T["crashes"], T["stored"], T["dropped"],
              T["updates"], T["epsilon"], T["loss"], T["time"]]]))
    a("")
    chk = R["check"]
    inner = chk["inner_cmds"]
    a("Распределение n в буфере: " + ", ".join(f"n = {k}: {v}" for k, v in chk["hist"].items())
      + f"; d = 1: {chk['d1']} (из них с n > 1: {chk['d1_multi']}). Проверка всех записей "
      f"по журналу шагов: начало не на шаге политики — {chk['bad']['start']}, шаг политики "
      f"внутри перехода — {chk['bad']['inner']}, переход не закрыт на следующем шаге "
      f"политики — {chk['bad']['next']}, max |R − Σ γ^j r| = {chk['max_dR']:.1e}. Команды на "
      f"шагах перехвата внутри переходов с n > 1: "
      + (", ".join(f"{k} — {v}" for k, v in sorted(inner.items())) if inner else "нет") + ".")
    a("")
    a("## 4. Записи буфера до, во время и после перехвата")
    a("")
    a(R["records"])
    s = R["saveload"]
    a("")
    a("## 5. Сохранение и загрузка модели")
    a("")
    a(f"`{Path(s['path']).name}` ({s['size'] / 1e6:.2f} МБ: веса Q, Q_target, состояние Adam, счётчики, "
      f"состояние ГСЧ). На {s['n_states']} состояниях буфера: max |ΔQ| = {s['dq']:.1e}, "
      f"max |ΔQ_target| = {s['dqt']:.1e}, жадные действия совпадают: {fmt(s['same_greedy'])}; "
      f"шагов {s['steps']}, обновлений {s['updates']}.")
    a("")
    a(f"## 6. Оценочные эпизоды загруженной модели (ε = 0, {len(R['eval'])} эпизодов)")
    a("")
    rows = [[c["seed"]] + [c[k] for k, _ in COUNTER_COLS] for c in R["eval"]]
    tot = R["eval_total"]
    rows.append(["итого"] + [tot.get(k if k != "crashed" else "crashes") if k != "dx" else None
                          for k, _ in COUNTER_COLS])
    a(table(["сид"] + [h for _, h in COUNTER_COLS], rows))
    a("")
    a("Итого: BLCR = ΣN_blocked / ΣN_att; N_att_rss и N_blocked_rss — попытки и блокировки без "
      "блокировок busy (манёвр в процессе или в очереди) и «нет полосы», по ним строятся модели H3; "
      "BLCR_rss = ΣN_blocked_rss / ΣN_att_rss. N_exec считается по шагу политики, команда которого "
      "действительно сдвинула целевую полосу (столбец exec_from журнала; разворот незавершённого "
      "манёвра, ABORT и FALLBACK снимают запись), поэтому каждое выполненное перестроение входит в "
      "экспозицию E; N_exec_other — пересечения разметки без такой записи (ожидается 0). "
      "R_int = (Σзамен + Σшагов перехвата) / Σшагов, v_eff и JerkRMS — средние по эпизодам. "
      "Журнал шагов: `eval_journal.csv`.")
    ep_pick = next((i for i, c in enumerate(R["eval"]) if c["intercept"] or c["N_exec"]), 0)
    J = [r for r in R["journal"] if r["episode"] == ep_pick]
    pick = [r for r in J if r["k"] < 6 or r["intervention"] or r["crossed"]][:24]
    a("")
    a(f"Фрагмент эпизода {ep_pick} (сид {R['eval'][ep_pick]['seed']}; первые шаги, все шаги перехвата "
      f"и пересечения разметки):")
    a("")
    a(table(["k", "aπ", "команда", "маска", "policy_step", "intervention", "active_cmd", "brake_mode",
             "return_active", "полоса", "пересечение", "exec_from", "v, м/с", "r"],
            [[r["k"], r["a_pi"], r["command"], r["mask"], r["policy_step"], r["intervention"],
              r["active_cmd"] or "—", r["brake_mode"], r["return_active"], r["lane"], r["crossed"],
              r["exec_from"] if r["exec_from"] != "" else "—", r["v"], r["r"]]
             for r in pick]))
    a("")
    rev = slow = 0
    prev = None
    for r in R["journal"]:
        if prev is not None and prev["episode"] == r["episode"] and \
                {prev["command"], r["command"]} == {"FASTER", "SLOWER"}:
            rev += 1
        if r["policy_step"] and r["command"] == "SLOWER" and r["a_pi"] != "SLOWER":
            slow += 1
        prev = r
    share = tot["jerk_jump_share"]
    a(f"JerkRMS считается по ускорению, применённому на каждом кадре 10 Гц (п. 7.7). Регулятор скорости "
      f"highway-env с ограничителями [−8; +3] м/с² при смене целевой скорости переключается за один кадр: "
      f"переход от +3 к −8 м/с² даёт |Δa|/Δt = 110 м/с³. Кадров с |Δa|/Δt > {JERK_JUMP:g} м/с³: "
      f"{tot['jerk_jumps']} из {tot['frames']}; их доля в Σ(Δa/Δt)² — {fmt(share)} (среднее по "
      f"эпизодам); JerkRMS без этих кадров — {fmt(tot['jerk_rms_smooth'])} м/с³ при "
      f"{fmt(tot['jerk_rms'])} м/с³ по всем кадрам; смен FASTER↔SLOWER на соседних тактах: {rev}; "
      f"замен aπ на SLOWER фильтром: {slow}. "
      + ("Большая часть JerkRMS приходится на скачки при переключении целевой скорости, поэтому сам по "
         "себе JerkRMS характеризует частоту переключений, а не плавность движения между ними; в "
         "основной серии вместе с JerkRMS приводятся число скачков и JerkRMS без них (п. 7.7)."
         if share > 0.5 else
         "Скачки дают меньшую часть JerkRMS; число скачков и JerkRMS без них приводятся рядом с "
         "JerkRMS (п. 7.7).")
      + " Для почти необученной политики большой JerkRMS ожидаем и к проверке цепочки не относится.")
    a("")
    a("## 7. Baseline 1, Baseline 2 и Режим А")
    a("")
    a(f"Короткое обучение без фильтра на тех же сидах обучения, сид сети {args.seed}: Baseline 1 — "
      f"награда R_v, Baseline 2 — R_RS (−5 за шаг решения с TTC < 2 с, −20 за столкновение, п. 7.2). "
      f"Бюджет {R['base_steps']} шагов решения, обучение с {R['base_learning_starts']}-го шага. Без "
      f"фильтра каждый шаг — шаг политики, поэтому все переходы должны иметь n = 1.")
    a("")
    rows = []
    for name, b in R["baselines"].items():
        ch = b["check"]
        rows.append([name, b["reward"], b["steps"], b["episodes"], b["stored"],
                     ", ".join(f"n = {k}: {v}" for k, v in ch["hist"].items()),
                     f"{ch['bad']['start']} / {ch['bad']['inner']} / {ch['bad']['next']}",
                     f"{ch['max_dR']:.1e}", b["crashes"], b["ttc_pen"], b["col_pen"], b["time"]])
    a(table(["конфигурация", "награда", "шагов решения", "эпизодов", "переходов", "распределение n",
             "ошибок: начало / внутри / закрытие", "max |R − Σγ^j r|", "столкновений",
             "Σ штрафа TTC", "Σ штрафа за столкновения", "время, с"], rows))
    a("")
    a("Оценка (ε = 0) на тех же оценочных сидах, что у Proposed. Режим А — замороженная Baseline 1 с "
      "фильтром основной серии; коды команд фильтра в признаках очереди отображаются в действия "
      "политики (TAKEOVER / FALLBACK → SLOWER, ABORT → смена полосы в сторону возврата), "
      f"brake_mode = 0 (п. 7.2). А-raw — та же политика и тот же фильтр без отображения, "
      f"{R['a_raw_episodes']} эпизода (описательно).")
    a("")
    a(table(["конфигурация", "фильтр", "награда", "эпизодов", "шагов", "E", "перехват",
             "TAKEOVER / ABORT / FALLBACK", "замена aπ", "R_int", "N_att", "N_att_rss", "N_blocked_rss",
             "BLCR_rss", "N_exec", "столкновений", "CR", "из них удар сзади", "v_eff, м/с", "JerkRMS",
             "Σ штрафа TTC", "Σ штрафа за столкновения"],
            [[m["arm"], m["filter"], m["reward"], m["episodes"], m["steps"], m["E"], m["intercept"],
              f"{m['takeover']} / {m['abort']} / {m['fallback']}", m["masked"], m["R_int"], m["N_att"],
              m["N_att_rss"], m["N_blocked_rss"], m["BLCR_rss"], m["N_exec"], m["crashes"], m["CR"],
              m["crash_rear"], m["v_eff"], m["jerk_rms"],
              m["ttc_pen"] if m["reward"] == "R_RS" else None,
              m["col_pen"] if m["reward"] == "R_RS" else None] for m in R["arms"]]))
    a("")
    a("«Удар сзади» — описательно: ближайший другой столкнувшийся автомобиль с боковым перекрытием "
      "находится позади ego. Операциональная классификация столкновений п. 7.7 пилотом не "
      "выполняется. Суммы штрафов R_RS приводятся для конфигураций с наградой R_RS.")
    a("")
    a("## 8. Смена полосы и ABORT на основном генераторе")
    a("")
    n_seek = {}
    for t in R["seek_total"]:
        n_seek.setdefault(t["D"], t["episodes"])
    a("Сценарная политика «искатель перестроений» (не обучается): на каждом шаге предпочитает смену "
      "полосы, направление выбирается случайно на каждом шаге, иначе IDLE; фильтр основной серии, "
      "ρ = " + " и ".join(f"{r:g}" for r in R["seek_rhos"]) + " с (доля тактов с допустимой сменой "
      "полосы по ρ — раздел 10); " + ", ".join(f"D = {D}: {n} стартов" for D, n in n_seek.items())
      + " из кэша (сиды 900 110–900 129), одни и те же старты при каждом ρ. Все эпизоды прогоняются "
      "целиком.")
    a("")
    rows = []
    for t in R["seek_total"]:
        rows.append([t["D"], str(t["rho"]), t["episodes"], t["steps"], t["E"], t["N_att_rss"],
                     t["N_blocked_rss"], t["BLCR_rss"], t["N_busy"], t["N_no_lane"], t["N_exec"],
                     t["N_exec_int"], t["N_exec_other"], t["abort"], t["takeover"], t["fallback"],
                     t["abort_ignored"], t["crashes"], t["crash_rear"], t["completed"]])
    a(table(["D", "ρ, с", "эпизодов", "шагов", "E", "N_att_rss", "N_blocked_rss", "BLCR_rss", "N_busy",
             "нет полосы", "N_exec", "N_exec_int", "N_exec_other", "ABORT", "TAKEOVER", "FALLBACK",
             "ABORT проигнорирован (после пересечения)", "столкновений", "из них удар сзади",
             "до T_max"], rows))
    a("")
    lc = [x for x in R.get("seek_lc", []) if x["lc_cmds"]]
    if lc:
        a("N_exec может быть больше N_att_rss − N_blocked_rss: у искателя Q обеих смен полосы выше IDLE, "
          "поэтому, когда выбранное направление заблокировано, а противоположное допустимо, фильтр "
          "исполняет смену полосы в другую сторону (попытка учтена в N_blocked_rss, исполненная смена "
          "полосы — в N_exec). Команды смены полосы на шагах политики: "
          + "; ".join(f"D = {x['D']}, ρ = {x['rho']:g} с — {x['lc_cmds']}, из них в сторону, "
                      f"противоположную aπ, {x['opposite']}, без пересечения до конца эпизода "
                      f"{x['no_cross']}" for x in lc) + ".")
        a("")
    frag = R.get("seek_abort")
    if frag:
        a(f"Фрагмент журнала вокруг первого ABORT (D = {frag['D']}, ρ = {frag['rho']} с, сид {frag['seed']}, "
          f"такт {frag['k']}). "
          "Журнал всех эпизодов: `seeker_journal.csv`.")
        a("")
        a(table(["k", "t, с", "aπ", "команда", "причина", "маска", "в FIFO", "активна после шага",
                 "return_active", "полоса", "целевая полоса", "пересечение", "exec_from", "v, м/с"],
                [[r["k"], r["t"], r["a_pi"], r["command"], r["reason"], r["mask"], r["queued"] or "—",
                  r["active_cmd"] or "—", r["return_active"], r["lane"], r["target_lane"], r["crossed"],
                  r["exec_from"] if r["exec_from"] != "" else "—", r["v"]] for r in frag["rows"]]))
    else:
        a("ABORT в прогонах искателя не встретился, поэтому ветвь ABORT на основном генераторе пилотом "
          "не показана. Она проверяется сценарием с разгоняющимся автомобилем сзади в целевой полосе "
          "(tests/test_lateral_and_lc_model.py, LR4) и строками T2b и Q2 верификации.")
    a("")
    a("## 9. Диагностика: боковое правило для соседней полосы")
    a("")
    gap = LANE_WIDTH - WIDTH
    a(f"Боковой зазор между автомобилями в центрах соседних полос: {LANE_WIDTH:g} − {WIDTH:g} = {gap:g} м. "
      "При нулевых боковых скоростях d_min_lat = μ + 2,25·ρ²:")
    a("")
    a(table(["ρ, с"] + [str(r) for r in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)],
            [["d_min_lat(0, 0), м"] + [d_min_lat(0.0, 0.0, r) for r in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)],
             ["превышает " + f"{gap:g} м"] + [d_min_lat(0.0, 0.0, r) > gap for r in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]]))
    a("")
    a("Фиксированная политика «всегда IDLE», одни и те же стартовые состояния (кэш start_states), "
      f"D = {args.D}. Правило основной серии: нарушение бокового порога соседом в соседней полосе "
      "опасно только при боковом перекрытии или боковом сближении — так же, как у t_danger и метки "
      "старта (п. 7.7). «Строгое RSS» — опасен любой боковой зазор меньше d_min_lat (прежнее "
      "поведение фильтра, только диагностика).")
    a("")
    a(table(["сид", "ρ, с", "боковое правило", "шагов", "шагов перехвата", "доля", "из них empty_safe_set",
             "замен IDLE", "столкновение", "старт (а) по тому же правилу", "старт (б)"],
            [[d["seed"], d["rho"], RULE_NAMES[d["rule"]], d["steps"], d["intercept"], d["share"], d["empty"],
              d["masked"], d["crashed"], d["unsafe_rule"], d["unsafe_lane"]] for d in R["diag"]]))
    a("")
    G = R.get("lc_grid") or []
    if G:
        starts = {}
        for g in G:
            starts.setdefault(g["D"], g["episodes"])
        a("## 10. Допустимость смены полосы на основном генераторе (сетка D × ρ)")
        a("")
        a(f"Фиксированная политика «IDLE, иначе SLOWER» (не обучается), до {R['lc_grid_steps']} шагов "
          "решения на старт; стартов из кэша start_states: " + ", ".join(f"D = {D}: {n}" for D, n in starts.items())
          + ". Эпизод ведёт фильтр основной "
          "серии; маски всех четырёх конфигураций (модель соседней полосы × прогноз автомобилей в "
          "полосах манёвра) вычисляются на тех же состояниях. Доля — такты, на которых LANE_LEFT или "
          "LANE_RIGHT входит в A_safe. Прежняя конфигурация по умолчанию (occupancy, худший случай: "
          "впереди −8, сзади +3 м/с² на всём горизонте 2,4 с) — та, на которой ранее получено 0/300 "
          "и 0/180.")
        a("")
        piv = {}
        for g in G:
            piv.setdefault((g["D"], g["rho"]), {})[(g["model"], g["lc_prediction"])] = g
        rows = []
        for (D, rho), m in piv.items():
            main, old = m[LC_CONFIGS[0]], m[LC_CONFIGS[-1]]
            rows.append([D, str(rho)] + [m[cf]["share"] for cf in LC_CONFIGS]
                        + [f"{main['front']} / {main['rear']} / {main['busy']} / {main['no_lane']}",
                           f"{old['front']} / {old['rear']}", main["ticks"], main["intercept"]])
        a(table(["D", "ρ, с"] + ["доля: " + LC_CONFIG_NAMES[cf] for cf in LC_CONFIGS]
                + ["основная: front / rear / busy / нет полосы", "прежняя: front / rear", "тактов",
                   "шагов перехвата (основная)"], rows))
        a("")
        a("Причины блокировки считаются по каждому из двух действий: front / rear — нарушение RSS с ТС "
          "впереди / сзади (у одной блокировки могут быть обе), busy — манёвр в процессе или в очереди, "
          "нет полосы — смена полосы за край дороги. Полная таблица по всем конфигурациям: `lc_grid.csv`.")
        a("")
    a("## Время")
    a("")
    tm = R["timing"]
    a(table(["этап", "с"], [[k, v] for k, v in tm["phases"].items()]))
    a("")
    a(f"На шаг решения в обучении Proposed: среда {tm['env_ms']:.1f} мс, фильтр {tm['shield_ms']:.1f} мс, "
      f"прямой проход сети {tm['policy_ms']:.2f} мс, обучение {tm['learn_ms']:.1f} мс. "
      "Журнал машины: `journal.md`.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description="Short end-to-end pilot (protocol, section 8)")
    p.add_argument("--out", required=True)
    p.add_argument("--D", type=int, default=25)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--base-steps", type=int, default=BASE_STEPS)
    p.add_argument("--base-learning-starts", type=int, default=BASE_LEARNING_STARTS)
    p.add_argument("--quick", action="store_true", help="smoke run: fewer seeds and steps")
    args = p.parse_args(argv)
    seeds = {k: list(v) for k, v in SEEDS.items()}
    if args.quick:
        seeds = {k: v[:QUICK_SEEDS[k]] for k, v in seeds.items()}
        args.steps, args.max_steps, args.learning_starts = 300, 1500, 100
        args.base_steps, args.base_learning_starts = 200, 100
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = str(out / "start_states")

    def log(msg):
        print(msg, flush=True)

    journal = machine_journal()
    (out / "journal.md").write_text(
        "| параметр | значение |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in journal.items()),
        encoding="utf-8")
    log("journal: " + to_json(journal))
    R, phases = {}, {}

    t0 = time.perf_counter()
    log("[1/10] start states and warm-up check")
    infos = {40: start_states(40, seeds["warmup"], cache), 10: start_states(10, seeds["start10"], cache),
             25: start_states(25, seeds["start25"], cache)}
    R["warmup"] = warmup_check(infos[40])
    R["starts"] = [start_summary(D, infos[D]) for D in (10, 25, 40)]
    flat = []
    for D in (10, 25, 40):
        for i in infos[D]:
            row = {k: v for k, v in i.items() if k not in ("unsafe", "unsafe_lane")}
            row.update({f"unsafe_a_{r}": int(i["unsafe"][r]) for r in RHOS})
            row.update({f"unsafe_b_{r}": int(i["unsafe_lane"][r]) for r in RHOS})
            flat.append(row)
    write_csv(out / "start_states.csv", flat)
    log("  warm-up: " + to_json(R["warmup"]))
    phases["стартовые состояния и прогрев"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[2/10] training (Proposed)")
    agent, steps_log, episodes, t_train = train(args, seeds["train"], cache, log)
    write_csv(out / "train_episodes.csv", episodes)
    tot = totals(episodes)
    n_steps = max(tot["steps"], 1)
    R["train_total"] = tot
    log(f"  filter of the training episodes: {tot['shield_cfg']} (MAIN_SERIES: {tot['main_series']})")
    R["train"] = dict(steps=agent.steps, episodes=len(episodes), E=tot["E"], intercept=tot["intercept"],
                      crashes=tot["crashes"], stored=agent.buffer.size,
                      dropped=sum(c["dropped"] for c in episodes), updates=agent.updates,
                      epsilon=agent.epsilon(), loss=agent.last_loss, time=t_train)
    timing = dict(env_ms=1e3 * sum(c["t_env"] for c in episodes) / n_steps,
                  shield_ms=1e3 * sum(c["t_shield"] for c in episodes) / n_steps,
                  policy_ms=1e3 * sum(c["t_policy"] for c in episodes) / n_steps,
                  learn_ms=1e3 * sum(c["t_learn"] for c in episodes) / n_steps)
    phases["обучение Proposed"] = time.perf_counter() - t0

    log("[3/10] replay records")
    rows, R["check"] = check_transitions(agent, steps_log)
    R["first_multi"], R["records"] = records_section(agent, rows)
    log("  check: " + to_json(R["check"]))

    log("[4/10] save / load")
    loaded, R["saveload"] = save_load(agent, out)
    log("  " + to_json(R["saveload"]))

    t0 = time.perf_counter()
    log("[5/10] evaluation of the loaded model")
    R["journal"], R["eval"] = evaluate(loaded, args, seeds["eval"], cache)
    R["eval_total"] = totals(R["eval"])
    write_csv(out / "eval_journal.csv", R["journal"])
    log("  totals: " + to_json(R["eval_total"]))
    phases["оценка Proposed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[6/10] Baselines 1 / 2 and mode A")
    R["base_steps"], R["base_learning_starts"] = args.base_steps, args.base_learning_starts
    R["a_raw_episodes"] = min(A_RAW_EPISODES, len(seeds["eval"]))
    R["baselines"] = {}
    R["arms"] = [arm("Proposed (Режим Б)", "основная серия", "R_v", R["eval_total"], R["eval"])]
    base_agents = {}
    for name, kind in (("Baseline 1", "Rv"), ("Baseline 2", "RS")):
        ag, slog, eps, tt = train(args, seeds["train"], cache, log, shield_on=False, reward_kind=kind,
                                  steps=args.base_steps, max_steps=args.base_steps,
                                  learning_starts=args.base_learning_starts, tag=name)
        _, chk = check_transitions(ag, slog)
        tb = totals(eps)
        R["baselines"][name] = dict(reward=REWARD_NAMES[kind], steps=ag.steps, episodes=len(eps),
                                    stored=ag.buffer.size, check=chk, crashes=tb["crashes"],
                                    ttc_pen=tb["ttc_pen"], col_pen=tb["col_pen"], time=tt)
        write_csv(out / f"train_episodes_{name.replace(' ', '_').lower()}.csv", eps)
        _, st = evaluate(ag, args, seeds["eval"], cache, shield_on=False, reward_kind=kind, journal=False)
        R["arms"].append(arm(name, "нет", REWARD_NAMES[kind], totals(st), st))
        base_agents[name] = ag
        log(f"  {name}: " + to_json(R["baselines"][name]).replace("\n", ""))
    b1 = base_agents["Baseline 1"]
    _, st = evaluate(b1, args, seeds["eval"], cache, env_config=dict(obs_mode_a=True), journal=False)
    R["arms"].append(arm("Режим А (Baseline 1 + фильтр)", "основная серия", "R_v", totals(st), st))
    _, st = evaluate(b1, args, seeds["eval"][:R["a_raw_episodes"]], cache,
                     env_config=dict(obs_mode_a=False), journal=False)
    R["arms"].append(arm("А-raw (без отображения)", "основная серия", "R_v", totals(st), st))
    for m in R["arms"]:
        log("  " + to_json(m).replace("\n", ""))
    phases["Baseline 1 / 2 и Режим А"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[7/10] lane-change seeker (executed lane changes, ABORT)")
    R["seek_rhos"] = list(SEEK_RHOS)
    R["seek"], seek_journal = seek({10: seeds["seek10"], 25: seeds["seek25"]}, cache, SEEK_RHOS, log)
    write_csv(out / "seeker_journal.csv", seek_journal)
    R["seek_total"] = [dict(D=D, rho=rho, **totals([c for c in R["seek"] if c["D"] == D and c["rho"] == rho]))
                       for rho in SEEK_RHOS for D in (10, 25)]
    R["seek_abort"] = abort_fragment(seek_journal)
    R["seek_lc"] = seek_lc_counts(seek_journal)
    for t in R["seek_total"]:
        log(f"  D {t['D']} rho {t['rho']}: " + to_json(t).replace("\n", ""))
    fa = R["seek_abort"]
    log("  first ABORT: " + (f"D {fa['D']} rho {fa['rho']} seed {fa['seed']} k {fa['k']}" if fa else "none"))
    phases["искатель перестроений"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[8/10] adjacent-lane diagnostic (rho = 0.6 vs 1.0, lateral rules)")
    R["diag"] = diagnostic(args, seeds["diag"], cache)
    for d in R["diag"]:
        log("  " + to_json(d).replace("\n", ""))
    phases["диагностика бокового правила"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[9/10] lane-change admissibility over the (D, rho) grid")
    R["lc_grid_steps"] = 20 if args.quick else LC_GRID_STEPS
    n_grid = LC_GRID_STARTS
    R["lc_grid"] = lane_change_grid({10: seeds["start10"][:n_grid], 25: seeds["start25"][:n_grid],
                                     40: seeds["warmup"][:n_grid]}, cache, R["lc_grid_steps"])
    write_csv(out / "lc_grid.csv", R["lc_grid"])
    for g in R["lc_grid"]:
        log("  " + to_json(g).replace("\n", ""))
    phases["допустимость смены полосы"] = time.perf_counter() - t0

    log("[10/10] report")
    timing["phases"] = phases
    R["timing"] = timing
    R["vmax"] = dict(SPEED_SEEN)
    (out / "report.md").write_text(report(args, R), encoding="utf-8")
    summary = {k: v for k, v in R.items() if k not in ("journal", "records")}
    summary["args"] = vars(args)
    summary["journal_machine"] = journal
    (out / "summary.json").write_text(to_json(summary), encoding="utf-8")
    clean = dict(start=0, inner=0, next=0)
    ok = (R["check"]["bad"] == clean and R["check"]["max_dR"] < 1e-9
          and R["saveload"]["dq"] == 0.0 and R["saveload"]["dqt"] == 0.0
          and all(b["check"]["bad"] == clean and set(b["check"]["hist"]) == {1}
                  and b["check"]["max_dR"] < 1e-9 for b in R["baselines"].values())
          and all(m["N_exec_other"] == 0 for m in R["arms"])
          and all(t["N_exec_other"] == 0 for t in R["seek_total"]))
    # the filter of the training and of every filtered evaluation arm is the main-series one
    filtered = [R["train_total"]] + [m for m in R["arms"] if m["main_series"] is not None]
    main_series = all(m["main_series"] for m in filtered)
    log("final counters (protocol 7.7; decision set of rss_shield.accounting):")
    log("  training Proposed: " + final_counters(R["train_total"]))
    for m in R["arms"]:
        log(f"  evaluation {m['arm']}: " + final_counters(m))
    log(f"filter configuration of all filtered arms is MAIN_SERIES: {main_series}")
    log(f"H1 (experiments.h1): correct start at rho = {h1.RHO_START} s for every arm; classification rho = "
        f"rho of the filter, {h1.RHO_REF} s without a filter; CR_total from the crash flag alone")
    for m in R["arms"]:
        s1 = m["h1"]
        log(f"  {m['arm']}: episodes {s1['n']} correct starts {s1['n_valid']} crashes {s1['crashes']} "
            f"(CR_total {fmt(s1['CR_total'])}) rho_cls {s1['rho_cls']} H1 events {s1['h1_events']} "
            f"categories {to_json(s1['categories']).replace(chr(10), '')}")
    ok = ok and main_series
    log("pilot: " + ("OK" if ok else "CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
