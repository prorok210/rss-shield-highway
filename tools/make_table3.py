"""Runs pytest + all control scenarios, writes CSV logs and the tables for the protocol text.

Usage (from the package root):  python tools/make_table3.py
Outputs:
  results/pytest.txt, results/junit.xml   - pytest -q output ("N passed") and per-test status
  results/table3.md / table3.csv          - table 3 of the protocol (pass/fail per row)
  results/table2_check.md                 - table 2: v6 vs independent reference vs simulation
  results/detection_1B.md                 - L8 detection ticks for both prediction models
  results/scenario2_setup.md              - initial conditions of scenario 2 (for R1.7)
  results/p0_profile.md                   - P0 lane change profile
  results/env_versions.md                 - table 1 (versions, hashes)
  logs/*.csv                              - per-physics-frame logs of every control run
"""
import hashlib, os, platform, re, subprocess, sys, xml.etree.ElementTree as ET
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

import numpy as np
from reference import table2_row, TABLE2_V6
from rss_shield.params import LENGTH, ACTION_INDEX, K_TAU, ABORT, FALLBACK, TAKEOVER
from rss_shield.rss import d_min_long
from rss_shield import scenarios as S
from rss_shield.shield import V_LAT_OBS


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def f(x, n=4):
    return "—" if x is None else (f"{x:+.{n}f}" if isinstance(x, float) else str(x))


# ------------------------------------------------------------------ 1. pytest
proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--junitxml=results/junit.xml", "tests"],
                      capture_output=True, text=True)
open("results/pytest.txt", "w").write(proc.stdout + proc.stderr)
status = {}
for tc in ET.parse("results/junit.xml").getroot().iter("testcase"):
    bad = any(ch.tag in ("failure", "error") for ch in tc)
    skip = any(ch.tag == "skipped" for ch in tc)
    status[tc.get("name")] = "fail" if bad else ("skip" if skip else "pass")
summary_line = (proc.stdout.strip().splitlines() or ["?"])[-1]


def verdict(pattern):
    names = [n for n in status if re.search(pattern, n)]
    if not names:
        return "нет теста"
    return "pass" if all(status[n] == "pass" for n in names) else "fail"


# ------------------------------------------------------------------ 2. scenario 1A (table 2)
t2 = []
meas = {}
for code, (rho, phase) in zip(["L1", "L2", "L3", "L4", "L5", "L6"],
                              [(0.1, "sync"), (0.1, "worst"), (0.2, "sync"), (0.2, "worst"),
                               (0.4, "sync"), (0.4, "worst")]):
    r = S.scenario1A(rho, phase)
    r.write_csv(f"logs/{r.name}.csv")
    res, ref, v6 = r.result, table2_row(rho, phase), TABLE2_V6[(rho, phase)]
    a = [row["a"] for row in r.rows if not row["crashed"]]
    t2.append([code, rho, "синхр." if phase == "sync" else "худш.", f"{res['d0']:.4f}",
               "да, t = %.1f с" % v6["t_contact"] if v6["crashed"] else f"нет, {v6['final_gap']:+.3f} м",
               "да, t = %.1f с" % ref["t_contact"] if ref["crashed"] else f"нет, {ref['final_gap']:+.4f} м",
               "да, t = %.1f с" % res["t_contact"] if res["crashed"] else f"нет, {res['final_gap']:+.4f} м",
               f"{res['ego_path']:.3f}" + ("*" if res["crashed"] else ""),
               f"{res['lead_path']:.3f}", f"[{min(a):.3f}; {max(a):.3f}]",
               verdict(rf"\[{rho}-{phase}-{code}\]")])
    meas[code] = (f"контакт t = {res['t_contact']:.1f} с" if res["crashed"]
                  else f"зазор {res['final_gap']:+.4f} м")
    meas[code + "_ref"] = (f"контакт t = {ref['t_contact']:.1f} с" if ref["crashed"]
                           else f"зазор {ref['final_gap']:+.4f} м")
with open("results/table2_check.md", "w") as fh:
    fh.write("# Таблица 2: v6 / независимый эталон (tests/reference.py) / симуляция highway-env\n\n")
    fh.write(md_table(["Строка", "ρ, с", "Фаза", "d0, м", "v6", "Эталон", "Симуляция",
                       "Путь ego, м", "Путь лидера, м", "a по кадрам до контакта", "Итог"], t2))
    fh.write("\n(*) путь до момента контакта и после него: после контакта highway-env меняет траекторию.\n")

r_once = S.scenario1A(0.2, "sync", slower_instead="once"); r_once.write_csv(f"logs/{r_once.name}.csv")
r_every = S.scenario1A(0.2, "sync", slower_instead="every_tick"); r_every.write_csv(f"logs/{r_every.name}.csv")
meas["L7"] = (f"однократный SLOWER: контакт t = {r_once.result['t_contact']:.1f} с; "
              f"SLOWER на каждом такте: " +
              (f"контакт t = {r_every.result['t_contact']:.1f} с" if r_every.result["crashed"]
               else f"контакта нет, зазор {r_every.result['final_gap']:+.2f} м"))

# ------------------------------------------------------------------ 3. scenario 1B (L8)
det_rows, det = [], {}
for rho in (0.1, 0.2, 0.4):
    row = [rho]
    for model in ("nominal", "worst_case"):
        r = S.scenario1B(rho, model)
        r.write_csv(f"logs/{r.name}.csv")
        det[(rho, model)] = r.result["detect_tick"]
        row.append(f"такт {r.result['detect_tick']} (t = {0.2 * r.result['detect_tick']:.1f} с)")
    det_rows.append(row)
with open("results/detection_1B.md", "w") as fh:
    fh.write("# Сценарий 1B: такт, на котором A_safe = ∅ (лидер −8 м/с² с первого кадра, d0 = d_min + 0.5 м)\n\n")
    fh.write(md_table(["ρ, с", "long_prediction = nominal", "long_prediction = worst_case"], det_rows))
meas["L8"] = "; ".join(f"ρ={rho}: nominal→{det[(rho, 'nominal')]}, worst_case→{det[(rho, 'worst_case')]}"
                       for rho in (0.1, 0.2, 0.4))

# ------------------------------------------------------------------ 4. stop / release / random
r = S.scenario_stop(); r.write_csv(f"logs/{r.name}.csv")
v = [row["v_end"] for row in r.rows]
k0 = next(i for i, x in enumerate(v) if x == 0.0)
xs = {round(row["x_end"], 9) for row in r.rows if row["v_start"] == 0.0}
meas["S1"] = (f"v = 0 на кадре {k0} (t_end = {r.rows[k0]['t_end']:.1f} с), min v = {min(v):.3g}, "
              f"a ∈ [{min(row['a'] for row in r.rows):.3f}; {max(row['a'] for row in r.rows[K_TAU:]):.3f}]")
meas["S2"] = f"кадров с v = 0: {sum(1 for row in r.rows if row['v_start'] == 0.0)}, различных x: {len(xs)}"
r1 = S.scenario_release(); r1.write_csv(f"logs/{r1.name}.csv")
r2 = S.scenario_release(standstill=True, then="FASTER"); r2.write_csv(f"logs/{r2.name}.csv")
f_rel = 2 * 5 + K_TAU
meas["S3"] = (f"движение: target после снятия = {r1.rows[f_rel]['target_speed']:.0f} м/с при v = "
              f"{r1.rows[f_rel - 1]['v_end']:.2f}, max a после снятия = {max(row['a'] for row in r1.rows[f_rel:]):+.3f}; "
              f"с места: a = {r2.rows[f_rel]['a']:.1f} (a_ctrl = {r2.rows[f_rel]['a_ctrl']:.2f}) с кадра {f_rel}")
err, amin, amax, cmin, cmax = 0.0, 9, -9, 0, 0
for seed in range(5):
    rr = S.scenario_random(seed)
    if seed == 0:
        rr.write_csv(f"logs/{rr.name}.csv")
    for row in rr.rows:
        err = max(err, abs(row["a_integrated"] - row["a"]))
        amin, amax = min(amin, row["a"]), max(amax, row["a"])
        cmin, cmax = min(cmin, row["a_ctrl"]), max(cmax, row["a_ctrl"])
meas["A1"] = (f"max abs(Δv/dt − a) = {err:.1e}; a ∈ [{amin:.2f}; {amax:.2f}] при a_ctrl ∈ [{cmin:.2f}; {cmax:.2f}] "
              f"(5 случайных последовательностей × 300 кадров + строки таблицы 2)")

# ------------------------------------------------------------------ 5. scenario 2
r = S.scenario2a(T=4.0); r.write_csv(f"logs/{r.name}.csv")
ys = [row["y_end"] for row in r.rows]
passed = next(row["frame"] for row in r.rows if row["R_x"] - row["x_end"] >= LENGTH)
m0 = [int(d.mask[ACTION_INDEX["LANE_LEFT"]]) for tick, t, d in r.decisions if 2 * tick <= passed]
meas["T2a"] = (f"M0 = {sorted(set(m0))} на {len(m0)} тактах до выхода R вперёд на L (кадр {passed}); "
               f"a_exec = {sorted({d.a_exec for tick, t, d in r.decisions if 2 * tick <= passed})}; "
               f"y ∈ [{min(ys[:passed + 1]):.3f}; {max(ys[:passed + 1]):.3f}]; контакт: {r.t_contact or 'нет'}")


def cmds(rr):
    return [(round(t, 1), d.command) for _, t, d in rr.decisions if d.command != "IDLE"]


r = S.scenario2b("i", model="observed"); r.write_csv(f"logs/{r.name}.csv")
tl = [row["target_lane_index"] for row in r.rows]
ret = next(i for i in range(3, len(tl)) if tl[i] == 1)
meas["T2b-i"] = (f"команды {cmds(r)}; target_lane_index → 1 на кадре {ret} (t = {0.1 * ret:.1f} с); "
                 f"итог y = {r.rows[-1]['y_end']:.3f}; контакт: {r.t_contact or 'нет'}")
r = S.scenario2b("iii", model="observed", T=8.0); r.write_csv(f"logs/{r.name}_reactiveR.csv")
fb = [row["frame"] for row in r.rows if row["fallback"]]
meas["T2b-iii"] = (f"команды {cmds(r)}; fallback = 1 на кадрах {fb[0]}–{fb[-1]}; "
                   f"контакт (R по RSS): {r.t_contact or 'нет'}")
r = S.scenario2b("iii", model="observed", T=10.0, rear_reactive=False); r.write_csv(f"logs/{r.name}_nonreactiveR.csv")
meas["T2b-iii"] += f"; контакт (R без реакции, 28 м/с): t = {r.t_contact} с, удар сзади → CR_env"
r = S.scenario2b("i", model="occupancy"); r.write_csv(f"logs/{r.name}.csv")
meas["T2b-occ"] = f"маска на такте 0: {''.join(map(str, r.decisions[0][2].mask))}; команды {cmds(r) or 'только IDLE'}"

# ------------------------------------------------------------------ 5b. regression checks Q1-Q4
from rss_shield.shield import RSSShield, ShieldConfig, complete_zone


def first_frame(rows, cond):
    return next((row["frame"] for row in rows if cond(row)), None)


r = S.scenario_q1_double_lane_change(); r.write_csv(f"logs/{r.name}.csv")
road, ego, sh = S.scenario_q1_scene()
ego.act(sh.decide("LANE_RIGHT").command)
for _ in range(2):
    road.act(); road.step(0.1)
v3 = sh.check_plan(["LANE_RIGHT"], sh.lc_frames, {3})["rear"]
meas["Q1"] = (f"такт 1: LANE_RIGHT в очереди (активен с кадра {first_frame(r.rows, lambda row: row['active_cmd'] == 'LANE_RIGHT')}), "
              f"маска {''.join(map(str, r.decisions[1][2].mask))}; команды {cmds(r)}; "
              f"target_lane_index ∈ {sorted({int(row['target_lane_index']) for row in r.rows})}; итог y = {r.rows[-1]['y_end']:.3f}; "
              f"проверка полосы 3 на такте 1: зазор сзади {v3['gap']:.3f} < {v3['d_min_long']:.3f} м (кадр {v3['frame']})")
import queue_reference as QR
road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0)       # counterexample of the review
ego.act(sh.decide("LANE_RIGHT").command)
for _ in range(2):
    road.act(); road.step(0.1)
cx = QR.command_violations(ego, sh.cfg, "LANE_RIGHT")["rear"]
cx_mask = sh.decide("LANE_RIGHT").mask
road, ego, sh = S.scenario_q1_scene(gap_r=5.0, v_r=28.0, k_tau=4)   # two lane changes in the FIFO
for _ in range(2):
    ego.act("LANE_RIGHT")
    for _ in range(2):
        road.act(); road.step(0.1)
m4 = sh.mask()[0]
meas["Q1"] += (f"; контрпример (R в полосе 3, 5 м сзади, 28 м/с): вторая LANE_RIGHT ведёт в полосу 3, зазор "
               f"{cx['gap']:.3f} < {cx['d_min_long']:.3f} м, маска {''.join(map(str, cx_mask))}; две смены полосы в FIFO "
               f"из 4 кадров: цель после очереди {sh.plan['target_lane']}, полосы манёвра {sorted(sh.plan['lanes'])}, "
               f"маска {''.join(map(str, m4))}")
late = early = after = crashes = ignored = 0
for case in S.Q2_LATE_ABORT_CASES:
    rr = S.scenario_q2_abort_crossing(*case)
    cr = first_frame(rr.rows, lambda row: row["lane_index"] == 2)
    ac = first_frame(rr.rows, lambda row: row["active_cmd"] == ABORT)
    late += int(ac is not None and cr is not None and ac > cr)
    early += int(ac is not None and cr is None)
    after += int(any(d.violations.get("ABORT") == {"after_crossing": True} for _, _, d in rr.decisions))
    crashes += int(rr.t_contact is not None)
    ignored += rr.ego.abort_ignored
    if case == (10.0, 15.0, 0.0):
        rr.write_csv(f"logs/{rr.name}.csv")
r = S.scenario_q2_scripted_abort(); r.write_csv(f"logs/{r.name}.csv")
meas["Q2"] = (f"{len(S.Q2_LATE_ABORT_CASES)} конфигураций: активаций ABORT после пересечения {late}; ABORT до пересечения "
              f"(без пересечения) {early}; ABORT не предложен после пересечения (after_crossing) {after}; "
              f"abort_ignored {ignored}; контактов {crashes}; без фильтра ABORT на 0.4 с: пересечение на кадре "
              f"{first_frame(r.rows, lambda row: row['lane_index'] == 2)}, активация на кадре "
              f"{first_frame(r.rows, lambda row: row['active_cmd'] == ABORT)}, abort_ignored = {r.ego.abort_ignored}, "
              f"итог полоса {r.rows[-1]['lane_index']}")
road, ego, sh = S.scenario_q3_far_leader()
m0, i0 = sh.mask()
ego.act("IDLE")
for _ in range(2):
    road.act(); road.step(0.1)
m1, i1 = sh.mask()
n_eq = n_sc = dropped = 0
for rho in (0.1, 0.4, 1.0):
    for seed in range(12):
        road, ego = S.scenario_q3_random(seed, ticks=1 + seed % 2)
        full = RSSShield(ego, ShieldConfig(rho=rho)); full.filter_on = False
        filt = RSSShield(ego, ShieldConfig(rho=rho))
        same = True
        for a_pi in ("LANE_LEFT", "FASTER"):
            d_full, d_filt = full.decide(a_pi), filt.decide(a_pi)
            same &= (list(d_full.mask) == list(d_filt.mask) and d_full.command == d_filt.command
                     and d_full.violations == d_filt.violations)
        n_eq += int(same); n_sc += 1
        dropped += len(road.vehicles) - 1 - len(filt._others(filt.lc_frames))
try:
    ShieldConfig(rho=1.0, zone=150.0)
    z150 = "принята"
except ValueError:
    z150 = "отклонена"
meas["Q3"] = (f"такт 0: маска {''.join(map(str, m0))}, IDLE: кадр {i0['IDLE']['front']['frame']}, зазор "
              f"{i0['IDLE']['front']['gap']:.3f} < {i0['IDLE']['front']['d_min_long']:.3f} м; после IDLE такт 1: маска "
              f"{''.join(map(str, m1))}, кадр {i1['IDLE']['front']['frame']}, зазор {i1['IDLE']['front']['gap']:.3f} м; "
              f"zone = 150 м {z150} (полная зона при ρ = 1.0: {complete_zone(1.0):.3f} м); фильтр = полная проверка в "
              f"{n_eq} из {n_sc} случайных сцен, отброшено {dropped} ТС")
from rss_shield.shield import ZoneError, envelope_zone, require_zone
env_a, env_b = envelope_zone(1.0, 30.0, 27.0)
try:
    require_zone(700.0, 280.0, 1.0, 30.0, 27.0)
    w280 = "принято"
except ZoneError:
    w280 = "отклонено"
meas["Q3"] += (f"; огибающая основной серии (ego ≤ 30, трафик ≤ 27 м/с, ρ = 1.0): требуется "
               f"[−{env_b:.3f}; +{env_a:.3f}] м, окно [−300; +700] м достаточно, окно [−280; +700] м {w280}")
r = S.scenario2b("i", model="observed")
r1a = S.scenario1A(0.4, "sync")
meas["Q4"] = (f"2b-i: intervention = 1 на кадрах {[row['frame'] for row in r.rows if row['intervention']]}, "
              f"ABORT активен на кадрах {[row['frame'] for row in r.rows if row['active_cmd'] == ABORT]} "
              f"(active_sup = 1, brake_mode = {max(row['brake_mode'] for row in r.rows if row['active_cmd'] == ABORT)}), "
              f"return_active на кадрах {first_frame(r.rows, lambda row: row['return_active'])}–"
              f"{max(row['frame'] for row in r.rows if row['return_active'])}; 1A sync: brake_mode = 1 с кадра "
              f"{first_frame(r1a.rows, lambda row: row['brake_mode'])}")
from rss_shield.accounting import count_step, new_counters
rq = S.scenario2b("i", model="observed", T=8.0, policy=lambda k, t: "LANE_RIGHT")
cq = new_counters()
for _, _, dq in rq.decisions:
    count_step(cq, dq.a_pi, dq.command, dq.mask, dq.violations.get(dq.a_pi))
one = all(row["policy_step"] + row["intervention"] == 1 for row in rq.rows + r.rows + r1a.rows)
meas["Q4"] += (f"; policy_step + intervention = 1 на каждом кадре: {'да' if one else 'НЕТ'}; 2b-i с запросом LANE_RIGHT на "
               f"каждом такте: шагов {cq['steps']}, E = {cq['E']}, перехват {cq['intercept']}, N_att = {cq['N_att']}, "
               f"N_blocked = {cq['N_blocked']} (busy {cq['N_busy']}), N_att_int = {cq['N_att_int']}")

with open("results/scenario2_setup.md", "w") as fh:
    fh.write("# Начальные условия сценария 2 (для R1.7). Дорога: 4 полосы, ширина 4 м, центр полосы i: y = 4i.\n\n")
    g = d_min_long(28.0, 20.0, 0.4) + 1.0
    fh.write(md_table(["Подтест", "Участник", "Полоса", "x (центр), м", "y, м", "v, м/с", "Поведение"], [
        ["2a", "ego", 1, 100.0, 4.0, 20.0, "политика LANE_LEFT на каждом такте, ρ = 0.4"],
        ["2a", "R", 0, f"{100 - LENGTH - 18:.1f} (зазор 18.0)", 0.0, 28.0, "скрипт, v = const"],
        ["2b", "ego", 1, 100.0, 4.0, 20.0, "LANE_RIGHT на такте 0, далее IDLE"],
        ["2b", "N", 3, f"{100 + LENGTH + 40:.1f} (зазор 40.0)", 12.0, 20.0,
         "с t_b = 0.2 с перестроение в полосу 2 (штатный регулятор highway-env)"],
        ["2b-iii", "R", 1, f"{100 - LENGTH - g:.2f} (зазор d_min(28,20,0.4)+1 = {g:.2f})", 4.0, 28.0,
         "RSS-совместимый: тормозит −4 через ρ, если зазор < d_min; вариант без реакции — v = const"],
    ]))
    fh.write(f"\nНаблюдаемое перестроение соседа: |v_lat| > {V_LAT_OBS} м/с (adjacent_lane_model = observed).\n")

# ------------------------------------------------------------------ 6. P0
p0 = [S.profile_lane_change(vv) for vv in (15.0, 20.0, 25.0, 30.0)]
with open("results/p0_profile.md", "w") as fh:
    fh.write("# P0: профиль штатного перестроения highway-env (каскадный регулятор, dt = 0.1 с)\n\n")
    fh.write(md_table(["v, м/с", "Длительность (до отклонения < 0.1 м от центра целевой полосы), с", "Пик v_lat, м/с", "Пик a_lat, м/с²"],
                      [[p["v"], f"{p['duration']:.1f}", f"{p['vlat_peak']:.2f}", f"{p['alat_peak']:.2f}"] for p in p0]))
meas["P0"] = (f"длительность {min(p['duration'] for p in p0):.1f}–{max(p['duration'] for p in p0):.1f} с; "
              f"пик v_lat {min(p['vlat_peak'] for p in p0):.2f}–{max(p['vlat_peak'] for p in p0):.2f} м/с; "
              f"пик a_lat {min(p['alat_peak'] for p in p0):.2f}–{max(p['alat_peak'] for p in p0):.2f} м/с²")

# ------------------------------------------------------------------ 7. table 3
rows3 = [
    ["F1–F6", "pytest FIFO (R1.5)", "—", "порядок, задержка ровно 2 кадра, пустые слоты, TAKEOVER, снятие, act() без аргумента",
     "точное", "см. results/pytest.txt", verdict(r"^test_F\d")],
]
for code, rho, ph in [("L1", 0.1, "синхр."), ("L2", 0.1, "худш."), ("L3", 0.2, "синхр."),
                      ("L4", 0.2, "худш."), ("L5", 0.4, "синхр."), ("L6", 0.4, "худш.")]:
    tol = "±0.1 с" if "контакт" in meas[code + "_ref"] else "±0.01 м"
    rows3.append([code, f"Сценарий 1A, {ph}", rho, meas[code + "_ref"], tol, meas[code],
                  verdict(rf"-{code}\]")])
rows3 += [
    ["L7", "SLOWER вместо перехвата", 0.2, "контакт, t ≈ 4.2 с (однократный SLOWER)", "±0.1 с", meas["L7"], verdict(r"^test_L7")],
    ["L8", "Сценарий 1B", "0.1/0.2/0.4", "nominal → такт 1, worst_case → такт 0", "точное", meas["L8"], verdict(r"^test_L8")],
    ["S1", "Торможение до остановки", 0.4, "v ≥ 0, a = −4 до остановки, последний кадр a = −v/dt", "точное", meas["S1"], verdict(r"^test_S1")],
    ["S2", "Удержание v = 0", 0.4, "v = 0, x = const", "точное", meas["S2"], verdict(r"^test_S2")],
    ["S3", "Снятие перехвата", 0.4, "нет разгона до прихода команды; target синхронизирован", "точное", meas["S3"], verdict(r"^test_S3")],
    ["A1", "Пределы ускорения по кинематике", "все", "max abs(Δv/dt − a) ≤ 1e−9; a ∈ [−8; 3], перехват [−8; −4]", "1e−9", meas["A1"], verdict(r"^test_A1")],
    ["T2a", "2a", 0.4, "M0 = 0 при d_long < d_min, a_exec = IDLE, y = const, нет контакта", "точное", meas["T2a"], verdict(r"^test_T2a")],
    ["T2b-i", "2b-i", 0.4, "ABORT на такте 2 (t = 0.4 с), возврат target на кадре 6", "точное", meas["T2b-i"], verdict(r"^test_T2b_i_")],
    ["T2b-iii", "2b-iii", 0.4, "FALLBACK на такте 2, флаг fallback с кадра 6", "точное", meas["T2b-iii"], verdict(r"^test_T2b_iii")],
    ["T2b-occ", "2b, модель occupancy", 0.4, "LANE_RIGHT запрещён на такте 0", "точное", meas["T2b-occ"], verdict(r"^test_T2b_occ")],
    ["Q1", "Повторная смена полосы до активации предыдущей", 0.4,
     "такт 1: LANE_LEFT/LANE_RIGHT = 0 (манёвр в очереди), цель не дальше полосы 2; контрпример 1.661 / 97.685 м запрещён; "
     "маска совпадает с эталоном без ограничения зоны на случайных сценах с очередью", "точное", meas["Q1"], verdict(r"^test_Q1")],
    ["Q2", "Пересечение разметки за время ожидания ABORT", 0.4,
     "нет активации ABORT после пересечения; после пересечения полоса сохраняется", "точное", meas["Q2"], verdict(r"^test_Q2")],
    ["Q3", "Полнота зоны проверки соседей", "0.1/0.4/1.0",
     "лидер v = 0 в 173 м при 30 м/с: IDLE запрещён (кадр 1: 164 < 167.625 м); фильтр = полная проверка", "точное",
     meas["Q3"], verdict(r"^test_Q3")],
    ["Q4", "Статусы журнала и множество решений", 0.4,
     "intervention, active_sup, brake_mode, return_active разведены; на каждом кадре ровно один из policy_step / "
     "intervention; N_att и N_blocked на шагах политики (E), запросы на шагах перехвата только в N_att_int", "точное",
     meas["Q4"], verdict(r"^test_Q4")],
    ["E1–E3", "Интеграция с HighwayEnv.step()", 0.4, "задержка и ограничитель внутри _simulate()", "1e−9", "см. results/pytest.txt", verdict(r"^test_E\d")],
    ["R0", "Формулы RSS", "—", "43.4638 / 47.9175 / 56.9825 / 93.02; d_lat(0,0,0.4) = 0.86", "5e−3", "см. results/pytest.txt", verdict(r"^test_R0")],
    ["P0", "Профиль манёвра highway-env", "—", "описательно", "—", meas["P0"], verdict(r"^test_P0")],
]
header = ["Строка", "Тест", "ρ", "Эталон", "Допуск", "Результат", "Итог"]
all_pass = all(r[-1] == "pass" for r in rows3)
with open("results/table3.md", "w") as fh:
    fh.write("# Таблица 3. Результаты контрольных тестов\n\n")
    fh.write(md_table(header, rows3))
    fh.write(f"\npytest: {summary_line}\n\nВсе строки pass: {'да' if all_pass else 'НЕТ'}. "
             "Основная серия запускается только при pass во всех строках.\n")
import csv
with open("results/table3.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(rows3)

# ------------------------------------------------------------------ 8. table 1 (versions)
def ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return "не установлен"


def sha(path):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(path)):
        dn.sort()
        for name in sorted(fn):
            if name.endswith(".py"):
                h.update(open(os.path.join(dp, name), "rb").read())
    return h.hexdigest()[:16]


try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "нет git"
except Exception:
    commit = "нет git"
with open("results/env_versions.md", "w") as fh:
    fh.write("# Таблица 1. Программное окружение\n\n")
    fh.write(md_table(["Компонент", "Версия"], [
        ["Python", platform.python_version()], ["ОС", platform.platform()],
        ["highway-env", ver("highway-env")], ["gymnasium", ver("gymnasium")],
        ["numpy", ver("numpy")], ["pytest", ver("pytest")], ["pygame / pygame-ce", f"{ver('pygame')} / {ver('pygame-ce')}"],
        ["torch (для DQN)", ver("torch")], ["stable-baselines3 (если используется)", ver("stable-baselines3")],
        ["git commit", commit],
        ["sha256 (rss_shield/*.py)", sha("rss_shield")], ["sha256 (tests/*.py)", sha("tests")],
    ]))

print(summary_line)
print("table 3:", "ALL PASS" if all_pass else "FAILURES: " + ", ".join(r[0] for r in rows3 if r[-1] != "pass"))
sys.exit(0 if all_pass and proc.returncode == 0 else 1)
