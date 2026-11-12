"""Runs the statistical plan of protocol 7.9 on a per-episode table (pilot outputs).

    python -m analysis.run --episodes runs/pilot_run/stats/episodes.csv --out runs/pilot_run/stats

Emits every number the plan mentions: H1a / H1b checks with Clopper-Pearson bounds and the IUT
outcome, CR_total of every arm, the descriptive difference Proposed - Baseline 2 with its
two-level cluster bootstrap interval, the seven primary checks of H2 / H3 (estimate, standard
error, signed root, one-sided bootstrap p, Holm, sensitivity analysis with shared scenario ids,
reconciled outcome), the control model, the contrast of differences, adjacent contrasts, Page's
tests and the dispersion diagnostics. On pilot data the numbers only show that the chain runs;
they are not evidence for or against H1-H3.
"""
import argparse
import csv
import json
import sys

import numpy as np

from analysis import binomial as B
from analysis import plan as P

H1_RHOS = (0.6, 1.0)           # rho levels of H1a; 0.4 is the boundary level (descriptive)


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def flag(v):
    return str(v) in ("1", "True", "true")


def h1_checks(rows):
    """One check per policy x mode x D x rho (filtered arms) or per policy x D (Baseline 2)."""
    groups = {}
    for r in rows:
        key = (r["mode"], float(r["D"]), r["rho"], r["unit"])
        groups.setdefault(key, []).append(r)
    out = []
    for (mode, D, rho, unit), ep in sorted(groups.items()):
        valid = [r for r in ep if flag(r["correct_start"])]
        x = sum(flag(r["h1_event"]) for r in valid)
        n = len(valid)
        out.append(dict(mode=mode, D=D, rho=rho, unit=unit, n=len(ep), n_valid=n, events=x,
                        rho_cls=sorted({r["rho_cls"] for r in ep}),
                        crashes=sum(flag(r["crashed"]) for r in ep), CR_total=sum(flag(r["crashed"]) for r in ep) / len(ep),
                        CR_total_valid=(sum(flag(r["crashed"]) for r in valid) / n) if n else None,
                        share=(x / n) if n else None, upper=B.cp_upper(x, n) if n else None,
                        lower=B.cp_lower(x, n) if n else None, decision=B.h1_decision(x, n)))
    return out


def h1_summary(checks):
    a = [c for c in checks if c["mode"] in "AB" and float(c["rho"]) in H1_RHOS]
    b = [c for c in checks if c["mode"] == "B2"]
    return dict(H1a=dict(checks=len(a), confirmed=sum(c["decision"] == B.CONFIRMED for c in a),
                         exceeded=sum(c["decision"] == B.EXCEEDED for c in a),
                         outcome="confirmed" if a and all(c["decision"] == B.CONFIRMED for c in a) else "not confirmed"),
                H1b=dict(checks=len(b), exceeded=sum(c["decision"] == B.EXCEEDED for c in b),
                         confirmed=sum(c["decision"] == B.CONFIRMED for c in b),
                         outcome="confirmed" if b and all(c["decision"] == B.EXCEEDED for c in b) else "not confirmed"))


def h1_difference(rows, n_boot, rng):
    """CR_ego(Proposed, mode B, rho) - CR_ego(Baseline 2) per D and rho in H1_RHOS on correct
    starts; Baseline 2 events are the ones classified at the reference rho."""
    def events(mode, D, rho=None):
        g = {}
        for r in rows:
            if r["mode"] == mode and float(r["D"]) == D and (rho is None or float(r["rho"]) == rho) \
                    and flag(r["correct_start"]):
                g.setdefault(r["unit"], []).append(float(flag(r["h1_event"])))
        return [np.array(v) for v in g.values()]
    out = []
    for D in sorted({float(r["D"]) for r in rows}):
        base = events("B2", D)
        for rho in H1_RHOS:
            prop = events("B", D, rho)
            if prop and base:
                d, lo, hi = B.cluster_bootstrap_difference(prop, base, n_boot, rng)
                out.append(dict(D=D, rho=rho, difference=d, lower=lo, upper=hi, policies=(len(prop), len(base))))
    return out


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "nan" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows])


def main(argv=None):
    p = argparse.ArgumentParser(description="Statistical plan of protocol 7.9 on a per-episode table")
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--boot", type=int, default=1000, help="parametric bootstrap replicates (plan: >= 1000)")
    p.add_argument("--boot-sens", type=int, default=2000, help="two-way cluster bootstrap replicates")
    p.add_argument("--boot-diff", type=int, default=10000, help="two-level bootstrap of the H1 difference")
    p.add_argument("--seed", type=int, default=20261110)
    args = p.parse_args(argv)
    rows = read(args.episodes)
    ab = [r for r in rows if r["mode"] in "AB"]
    rng = np.random.default_rng(args.seed)
    R = dict(args=vars(args), n_rows=len(rows))

    checks = h1_checks(rows)
    R["h1_checks"], R["h1"] = checks, h1_summary(checks)
    R["h1_difference"] = h1_difference(rows, args.boot_diff, rng)

    R["excluded"] = dict(E0=sum(float(r["E"]) <= 0 for r in ab), N_att0=sum(float(r["N_att"]) <= 0 for r in ab),
                         episodes=len(ab))
    R["primary"] = P.run_primary(ab, args.boot, args.boot_sens, args.seed)
    R["control"] = P.CONTROL.function(ab, args.boot, np.random.default_rng([args.seed, 50]))
    R["dispersion"] = dict(N_exec=P.dispersion(ab, "N_exec", P.H2_TERMS, "B"),
                           N_blocked=P.dispersion(ab, "N_blocked", P.H3_TERMS, "AB"),
                           N_att=P.dispersion(ab, "N_att", P.H3_TERMS, "AB"),
                           blocked_share=P.dispersion(ab, "N_blocked", P.H3_TERMS, "AB", trials="N_att"))
    means, reps = P.bootstrap_cells(ab, "v_eff", args.boot_sens, np.random.default_rng([args.seed, 60]))
    con = P.contrast_of_differences(means)
    lo, hi = P.interval([P.contrast_of_differences(m) for m in reps])
    R["contrast"] = dict(estimate=con, lower=lo, upper=hi)
    R["adjacent"], R["page"] = {}, {}
    for outcome in ("v_eff", "N_exec"):
        m0, rp = (means, reps) if outcome == "v_eff" else P.bootstrap_cells(
            ab, outcome, args.boot_sens, np.random.default_rng([args.seed, 61]))
        adj = P.adjacent_contrasts(m0)
        R["adjacent"][outcome] = {k: dict(estimate=v, interval=P.interval([P.adjacent_contrasts(m)[k] for m in rp]))
                                  for k, v in adj.items()}
        for over in ("rho", "D"):
            levels, tab = P.trend_table(ab, outcome, over)
            if tab.size:
                L, pv, how = P.page_test(tab)
                R["page"][f"{outcome} over {over}"] = dict(levels=levels, blocks=len(tab), L=L, p=pv, method=how)

    L = ["# Statistical plan on pilot data (protocol 7.9)", "",
         f"Input: `{args.episodes}`, {len(rows)} episodes. Parametric bootstrap: {args.boot} replicates "
         f"(the plan requires at least 1000 for the main series); two-way cluster bootstrap: {args.boot_sens}; "
         f"H1 difference: {args.boot_diff}. Pilot policies are not trained: the numbers show that every step of "
         "the plan is computed, not that H1-H3 hold.", "",
         "## H1: checks on episodes with a correct start", "",
         "Start rule and classification rho: `experiments/h1.py` (correct start at rho = 1.0 s for every arm; "
         "Baseline 2 classified at the reference rho = 1.0 s). `rho_cls` is the rho used for the categories.", "",
         table(["arm", "D", "rho", "unit", "n", "n_valid", "CR_total", "events CR_ego_op", "rho_cls", "upper 95 %",
                "lower 95 %", "decision"],
               [[c["mode"], c["D"], c["rho"] or "—", c["unit"], c["n"], c["n_valid"], c["CR_total"], c["events"],
                 "/".join(c["rho_cls"]), c["upper"], c["lower"], c["decision"]] for c in checks]), "",
         f"H1a (IUT over the {R['h1']['H1a']['checks']} checks with rho in {{0.6; 1.0}}): **{R['h1']['H1a']['outcome']}** "
         f"({R['h1']['H1a']['confirmed']} confirmed, {R['h1']['H1a']['exceeded']} exceeded). "
         f"H1b (IUT over the {R['h1']['H1b']['checks']} Baseline 2 checks): **{R['h1']['H1b']['outcome']}** "
         f"({R['h1']['H1b']['exceeded']} exceeded, {R['h1']['H1b']['confirmed']} confirmed). "
         "Checks with rho = 0.4 are the boundary level: descriptive, outside the decision.", "",
         "Descriptive difference CR_ego_op(Proposed, mode B) − CR_ego_op(Baseline 2), two-level cluster bootstrap:", "",
         table(["D", "rho", "difference", "lower 95 %", "upper 95 %", "policies (Proposed, Baseline 2)"],
               [[d["D"], d["rho"], d["difference"], d["lower"], d["upper"], f"{d['policies'][0]}, {d['policies'][1]}"]
                for d in R["h1_difference"]]), "",
         "## H2 / H3: seven primary checks", "",
         "Covariates: Dc = (D − 25) / 10, rhoc = rho − 2/3 (grid means), mode = 1 for mode B. One-sided p: "
         "parametric bootstrap of the signed likelihood root under the null model; Holm over the seven checks in "
         "each analysis; sensitivity: two-way cluster bootstrap that keeps shared scenario ids.", "",
         table(["check", "function", "model", "estimate", "SE", "signed root", "p (bootstrap)", "p Holm",
                "converged", "sens. estimate", "sens. 95 % interval", "sens. p", "sens. p Holm", "outcome"],
               [[r["check"], f"`{r['function']}`", r["model"], r["primary"]["estimate"], r["primary"]["se"],
                 r["primary"]["root"], r["primary"]["p_boot"], r["p_holm_primary"],
                 "yes" if r["primary"]["converged"] else "no: " + r["primary"]["message"],
                 r["sensitivity"]["estimate"], f"[{fmt(r['sensitivity']['lower'])}; {fmt(r['sensitivity']['upper'])}]",
                 r["sensitivity"]["p"], r["p_holm_sensitivity"], r["outcome"]] for r in R["primary"]]), "",
         f"Control (outside the family) `{P.CONTROL.function.__name__}`, M5: estimate {fmt(R['control']['estimate'])}, "
         f"SE {fmt(R['control']['se'])}, one-sided bootstrap p {fmt(R['control']['p_boot'])}, converged: "
         f"{R['control']['converged']}{'' if R['control']['converged'] else ' (' + R['control']['message'] + ')'}.", "",
         f"Excluded episodes: E = 0: {R['excluded']['E0']}; N_att = 0 (only from M4): {R['excluded']['N_att0']} of "
         f"{R['excluded']['episodes']}.", "",
         "Pearson dispersion without random effect and overdispersion (Poisson / binomial): "
         + ", ".join(f"{k} {fmt(v, 2)}" for k, v in R["dispersion"].items()) + ".", "",
         "## Descriptive contrasts and monotonicity (secondary)", "",
         f"Contrast of differences [v_eff(40, 1.0) − v_eff(40, 0.4)] − [v_eff(10, 1.0) − v_eff(10, 0.4)]: "
         f"{fmt(con)} [{fmt(lo)}; {fmt(hi)}] m/s.", "",
         table(["outcome", "adjacent levels", "difference", "95 % interval"],
               [[o, k, v["estimate"], f"[{fmt(v['interval'][0])}; {fmt(v['interval'][1])}]"]
                for o, d in R["adjacent"].items() for k, v in d.items()]), "",
         table(["Page's test (decreasing)", "levels", "blocks", "L", "p", "method"],
               [[k, "/".join(f"{x:g}" for x in v["levels"]), v["blocks"], v["L"], v["p"], v["method"]]
                for k, v in R["page"].items()]), ""]
    text = "\n".join(L)
    with open(f"{args.out}/analysis.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(f"{args.out}/analysis.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
