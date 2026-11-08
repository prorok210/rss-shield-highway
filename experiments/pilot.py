"""Short end-to-end pilot of the training chain (protocol v11, section 8).

1. warm-up check (7.6): mean flow speed over 10-15 s and 15-20 s of the warm-up, D = 40;
2. start states for D = 10 / 25 / 40: removed vehicles, insertion, start labels for every rho
   (a: rule of 7.6 including adjacent lanes; b: leader and follower in the ego lane only);
3. short training of Proposed (mode B), rho = 0.6 s, D = 25, network seed 42;
4. replay records before, during and after the first intercept with the reward of every covered
   decision step; check of all stored transitions against the step log;
5. saving and loading the model;
6. evaluation episodes of the loaded model (epsilon = 0) with the step journal and the counters;
7. diagnostic of the lateral threshold for adjacent lanes: a fixed IDLE policy at rho = 0.6 and
   1.0 on the same start states.
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
from experiments.env import FAST_ROAD, THREAD_VARS, make_env, reset_cached  # noqa: E402
from experiments.generator import I_BRAKE, I_LANE, I_SPEED, N_KIN, QUEUE_CODES, RHOS  # noqa: E402
from experiments.runner import JERK_JUMP, NAMES, mask_str, run_episode  # noqa: E402
from rss_shield.params import LANE_WIDTH, WIDTH  # noqa: E402
from rss_shield.rss import d_min_lat  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.99
SEEDS = {
    "warmup": list(range(900_000, 900_020)),    # D = 40: warm-up check and start states
    "start10": list(range(900_020, 900_030)),   # D = 10
    "start25": list(range(900_030, 900_040)),   # D = 25
    "train": list(range(900_040, 900_090)),     # training episodes (cycled)
    "eval": list(range(900_090, 900_092)),      # evaluation of the loaded model
    "diag": list(range(900_092, 900_095)),      # rho = 0.6 vs 1.0 diagnostic
}
QUICK_SEEDS = {"warmup": 3, "start10": 2, "start25": 2, "train": 50, "eval": 1, "diag": 1}


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


def train(args, seeds, cache, log):
    env = make_env(D=args.D)
    agent = DQN(seed=args.seed, learning_starts=args.learning_starts)
    steps_log, episodes = [], []
    ep = 0
    t0 = time.perf_counter()
    while True:
        seed = seeds[ep % len(seeds)]
        c = run_episode(env, agent, rho=args.rho, train=True, explore=True, episode=ep,
                        reset=lambda s=seed: reset_cached(env, s, cache), steps_log=steps_log)
        c.update(episode=ep, seed=seed, total_steps=agent.steps, epsilon=agent.epsilon(),
                 buffer=agent.buffer.size, loss=agent.last_loss)
        episodes.append(c)
        log(f"  episode {ep:3d} seed {seed} steps {c['steps']:3d} policy {c['E']:3d} "
            f"intercept {c['intercept']:2d} crash {int(c['crashed'])} return {c['ret']:7.2f} | "
            f"total {agent.steps:5d} eps {agent.epsilon():.3f} buffer {agent.buffer.size:5d} "
            f"loss {fmt(agent.last_loss, 4)}")
        ep += 1
        multi = bool((agent.buffer.n[:agent.buffer.size] > 1).any())
        if agent.steps >= args.steps and (multi or agent.steps >= args.max_steps):
            break
    return agent, steps_log, episodes, time.perf_counter() - t0


def check_transitions(agent, steps_log):
    """Every stored transition against the step log: starts at a policy step, covers only
    intercept steps after it, ends before the next policy step, R = sum gamma^j r."""
    b = agent.buffer
    rows = {(r["episode"], r["k"]): r for r in steps_log}
    bad = dict(start=0, inner=0, next=0)
    max_dr = 0.0
    for i in range(b.size):
        e, k0 = b.meta[i]
        n = int(b.n[i])
        bad["start"] += int(rows[(e, k0)]["policy_step"] != 1)
        bad["inner"] += int(any(rows[(e, k0 + j)]["policy_step"] != 0 for j in range(1, n)))
        nxt = rows.get((e, k0 + n))
        bad["next"] += int(nxt is not None and nxt["policy_step"] != 1)
        r = sum(GAMMA ** j * rows[(e, k0 + j)]["r"] for j in range(n))
        max_dr = max(max_dr, abs(r - float(b.R[i])))
    ns = b.n[:b.size]
    d = b.d[:b.size]
    return rows, dict(bad=bad, max_dR=max_dr, hist={int(v): int((ns == v).sum()) for v in np.unique(ns)},
                      d1=int(d.sum()), d1_multi=int((d & (ns > 1)).sum()))


def decode(s):
    e = s[N_KIN:]
    q = [QUEUE_CODES[i] for i in range(len(QUEUE_CODES)) if e[i] > 0.5]
    return (f"FIFO {q[0] if q else '—'}, brake_mode {int(e[I_BRAKE] > 0.5)}, "
            f"v_target {5 * int(round(e[I_SPEED] * 6))} м/с, "
            f"целевая − текущая полоса {int(round(e[I_LANE])):+d}, "
            f"v ≈ {float(s[3]) * 80.0:.2f} м/с")


def records_section(agent, rows):
    b = agent.buffer
    cands = [i for i in range(b.size) if b.n[i] > 1]
    # prefer an intercept transition with a non-zero discounted sum, so that the check is informative
    first = next((i for i in cands if abs(b.R[i]) > 1e-9), cands[0] if cands else None)
    if first is None:
        return None, ("Переходов с n > 1 в буфере нет: "
                      "перехват за бюджет не встретился.")
    ep0 = b.meta[first][0]
    idxs = [i for i in (first - 1, first, first + 1) if 0 <= i < b.size and b.meta[i][0] == ep0]
    names = {-1: "до перехвата", 0: "с перехватом",
             1: "после перехвата"}
    parts = []
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
                          r["active"] or "—", r["brake_mode"], f"{r['r']:.5f}",
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


def evaluate(agent, args, seeds, cache):
    env = make_env(D=args.D)
    journal, stats = [], []
    for i, s in enumerate(seeds):
        c = run_episode(env, agent, rho=args.rho, explore=False, episode=i,
                        reset=lambda s=s: reset_cached(env, s, cache), journal=journal)
        c["seed"] = s
        stats.append(c)
    return journal, stats


class IdlePolicy:
    """Fixed policy that always prefers IDLE (diagnostic of the filter, not trained)."""

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.cfg = {"gamma": GAMMA}

    def q_values(self, obs):
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    def epsilon(self):
        return 0.0


def diagnostic(args, seeds, cache):
    env = make_env(D=args.D)
    out = []
    for s in seeds:
        for rho in (0.6, 1.0):
            j = []
            c = run_episode(env, IdlePolicy(), rho=rho, reset=lambda s=s: reset_cached(env, s, cache),
                            journal=j)
            reasons = {}
            for r in j:
                if r["intervention"]:
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            out.append(dict(seed=s, rho=rho, steps=c["steps"], intercept=c["intercept"],
                            share=c["intercept"] / c["steps"], empty=reasons.get("empty_safe_set", 0),
                            masked=c["masked"], crashed=c["crashed"], unsafe=c["unsafe_start"],
                            unsafe_lane=c["unsafe_start_lane"]))
    return out


def totals(stats):
    def s(k):
        return sum(c[k] for c in stats)
    steps = s("steps")
    return dict(episodes=len(stats), steps=steps, E=s("E"), intercept=s("intercept"),
                takeover=s("takeover"), abort=s("abort"), fallback=s("fallback"), N_att=s("N_att"),
                N_blocked=s("N_blocked"), BLCR=(s("N_blocked") / s("N_att") if s("N_att") else None),
                N_att_int=s("N_att_int"), N_exec=s("N_exec"), N_exec_int=s("N_exec_int"),
                masked=s("masked"), R_int=(s("masked") + s("intercept")) / steps,
                crashes=s("crashed"), CR=s("crashed") / len(stats), completed=s("completed"),
                v_eff=float(np.mean([c["v_eff"] for c in stats])),
                jerk_rms=float(np.mean([c["jerk_rms"] for c in stats])))


# ---------------------------------------------------------------------- report
COUNTER_COLS = [("steps", "шагов"), ("E", "E"), ("intercept", "перехват"), ("takeover", "TAKEOVER"),
                ("abort", "ABORT"), ("fallback", "FALLBACK"), ("N_att", "N_att"), ("N_blocked", "N_blocked"),
                ("BLCR", "BLCR"), ("N_att_int", "N_att_int"), ("N_exec", "N_exec"),
                ("N_exec_int", "N_exec_int"), ("masked", "замена aπ"), ("R_int", "R_int"),
                ("crashed", "столкновение"), ("completed", "до T_max"), ("dx", "Δx, м"),
                ("v_eff", "v_eff, м/с"), ("jerk_rms", "JerkRMS")]


def report(args, R):
    L = []
    a = L.append
    a("# Пилот сквозной цепочки обучения (протокол v11, раздел 8)")
    a("")
    a(f"Proposed, Режим Б, D = {args.D} авт/км/полосу, ρ = {args.rho} с, R_v, сид сети {args.seed}, "
      f"бюджет {args.steps} шагов решения (продление до {args.max_steps}, пока в буфере "
      f"нет перехода с n > 1). Сиды пилота 900 000–900 094. Проверяется работоспособность "
      f"цепочки; H1–H3 пилотом не проверяются." + (" **Режим --quick.**" if args.quick else ""))
    w = R["warmup"]
    a("")
    a(f"## 1. Прогрев (п. 7.6): D = 40, {w['n']} сидов")
    a("")
    a(table(["средняя скорость 10–15 с, м/с", "15–20 с, м/с", "разница (по всем сидам)",
                "максимум по сиду", "сидов < 2 %", "критерий < 2 %"],
            [[w["v10_15"], w["v15_20"], f"{100 * w['pooled']:.2f} %", f"{100 * w['max_seed']:.2f} %",
              f"{w['n_seed_ok']} из {w['n']}", "выполнен" if w["passed"] else "НЕ выполнен"]]))
    a("")
    a("## 2. Стартовые состояния")
    a("")
    a("(а) — правило п. 7.6 с соседними полосами (продольный и боковой пороги одновременно), "
      "(б) — только лидер и догоняющий в полосе ego. Число исходно некорректных стартов по ρ.")
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
    a("Распределение n в буфере: " + ", ".join(f"n = {k}: {v}" for k, v in chk["hist"].items())
      + f"; d = 1: {chk['d1']} (из них с n > 1: {chk['d1_multi']}). Проверка всех записей "
      f"по журналу шагов: начало не на шаге политики — {chk['bad']['start']}, шаг политики "
      f"внутри перехода — {chk['bad']['inner']}, переход не закрыт на следующем шаге "
      f"политики — {chk['bad']['next']}, max |R − Σ γ^j r| = {chk['max_dR']:.1e}.")
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
    a("## 6. Оценочные эпизоды загруженной модели (ε = 0)")
    a("")
    rows = [[c["seed"]] + [c[k] for k, _ in COUNTER_COLS] for c in R["eval"]]
    tot = R["eval_total"]
    rows.append(["итого"] + [tot.get(k if k != "crashed" else "crashes") if k != "dx" else None
                          for k, _ in COUNTER_COLS])
    a(table(["сид"] + [h for _, h in COUNTER_COLS], rows))
    a("")
    a("Итого: BLCR = ΣN_blocked / ΣN_att, R_int = (Σзамен + Σшагов перехвата) / Σшагов, "
      "v_eff и JerkRMS — средние по эпизодам. Журнал шагов: `eval_journal.csv`; фрагмент первого "
      "эпизода (первые шаги и все шаги перехвата):")
    a("")
    J = [r for r in R["journal"] if r["episode"] == 0]
    pick = [r for r in J if r["k"] < 6 or r["intervention"] or r["crossed"]][:24]
    a(table(["k", "aπ", "команда", "маска", "policy_step", "intervention", "active", "brake_mode",
             "return_active", "полоса", "v, м/с", "r"],
            [[r["k"], r["a_pi"], r["command"], r["mask"], r["policy_step"], r["intervention"],
              r["active"] or "—", r["brake_mode"], r["return_active"], r["lane"], r["v"], r["r"]]
             for r in pick]))
    a("")
    jumps = sum(int(c.get("jerk_jumps", 0)) for c in R["eval"])
    frames = sum(int(c.get("frames", 0)) for c in R["eval"])
    rev = slow = 0
    prev = None
    for r in R["journal"]:
        if prev is not None and prev["episode"] == r["episode"] and \
                {prev["command"], r["command"]} == {"FASTER", "SLOWER"}:
            rev += 1
        if r["policy_step"] and r["command"] == "SLOWER" and r["a_pi"] != "SLOWER":
            slow += 1
        prev = r
    a(f"JerkRMS считается по ускорению, применённому на каждом кадре 10 Гц (п. 7.7). Регулятор скорости "
      f"highway-env с ограничителями [−8; +3] м/с² при смене целевой скорости переключается за один кадр: "
      f"переход от +3 к −8 м/с² даёт |Δa|/Δt = 110 м/с³. Кадров с |Δa|/Δt > {JERK_JUMP:g} м/с³: {jumps} "
      f"из {frames}; смен FASTER↔SLOWER на соседних тактах: {rev}; замен aπ на SLOWER фильтром: {slow}. "
      f"Для почти необученной политики большой JerkRMS ожидаем и к проверке цепочки не относится.")
    a("")
    a("## 7. Диагностика: боковой порог для соседней полосы")
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
      f"D = {args.D}:")
    a("")
    a(table(["сид", "ρ, с", "шагов", "шагов перехвата", "доля", "из них empty_safe_set", "замен IDLE",
             "столкновение", "старт (а) некорр.", "старт (б) некорр."],
            [[d["seed"], d["rho"], d["steps"], d["intercept"], d["share"], d["empty"], d["masked"],
              d["crashed"], d["unsafe"], d["unsafe_lane"]] for d in R["diag"]]))
    a("")
    a("## Время")
    a("")
    tm = R["timing"]
    a(table(["этап", "с"], [[k, v] for k, v in tm["phases"].items()]))
    a("")
    a(f"На шаг решения в обучении: среда {tm['env_ms']:.1f} мс, фильтр {tm['shield_ms']:.1f} мс, "
      f"прямой проход сети {tm['policy_ms']:.2f} мс, обучение {tm['learn_ms']:.1f} мс. "
      "Журнал машины: `journal.md`.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description="Short end-to-end pilot (protocol v11, section 8)")
    p.add_argument("--out", required=True)
    p.add_argument("--D", type=int, default=25)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="smoke run: fewer seeds and steps")
    args = p.parse_args(argv)
    seeds = {k: list(v) for k, v in SEEDS.items()}
    if args.quick:
        seeds = {k: v[:QUICK_SEEDS[k]] for k, v in seeds.items()}
        args.steps, args.max_steps, args.learning_starts = 300, 1500, 100
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
    log("[1/7] start states and warm-up check")
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
    log("[2/7] training")
    agent, steps_log, episodes, t_train = train(args, seeds["train"], cache, log)
    write_csv(out / "train_episodes.csv", episodes)
    tot = totals(episodes)
    n_steps = max(tot["steps"], 1)
    R["train"] = dict(steps=agent.steps, episodes=len(episodes), E=tot["E"], intercept=tot["intercept"],
                      crashes=tot["crashes"], stored=agent.buffer.size,
                      dropped=sum(c["dropped"] for c in episodes), updates=agent.updates,
                      epsilon=agent.epsilon(), loss=agent.last_loss, time=t_train)
    timing = dict(env_ms=1e3 * sum(c["t_env"] for c in episodes) / n_steps,
                  shield_ms=1e3 * sum(c["t_shield"] for c in episodes) / n_steps,
                  policy_ms=1e3 * sum(c["t_policy"] for c in episodes) / n_steps,
                  learn_ms=1e3 * sum(c["t_learn"] for c in episodes) / n_steps)
    phases["обучение"] = time.perf_counter() - t0

    log("[3/7] replay records")
    rows, R["check"] = check_transitions(agent, steps_log)
    R["first_multi"], R["records"] = records_section(agent, rows)
    log("  check: " + to_json(R["check"]))

    log("[4/7] save / load")
    loaded, R["saveload"] = save_load(agent, out)
    log("  " + to_json(R["saveload"]))

    t0 = time.perf_counter()
    log("[5/7] evaluation of the loaded model")
    R["journal"], R["eval"] = evaluate(loaded, args, seeds["eval"], cache)
    R["eval_total"] = totals(R["eval"])
    write_csv(out / "eval_journal.csv", R["journal"])
    log("  totals: " + to_json(R["eval_total"]))
    phases["оценка"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    log("[6/7] adjacent-lane diagnostic (rho = 0.6 vs 1.0)")
    R["diag"] = diagnostic(args, seeds["diag"], cache)
    for d in R["diag"]:
        log("  " + to_json(d).replace("\n", ""))
    phases["диагностика ρ"] = time.perf_counter() - t0

    log("[7/7] report")
    timing["phases"] = phases
    R["timing"] = timing
    (out / "report.md").write_text(report(args, R), encoding="utf-8")
    summary = {k: v for k, v in R.items() if k not in ("journal", "records")}
    summary["args"] = vars(args)
    summary["journal_machine"] = journal
    (out / "summary.json").write_text(to_json(summary), encoding="utf-8")
    ok = (R["check"]["bad"] == dict(start=0, inner=0, next=0) and R["check"]["max_dR"] < 1e-9
          and R["saveload"]["dq"] == 0.0 and R["saveload"]["dqt"] == 0.0)
    log("pilot: " + ("OK" if ok else "CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
