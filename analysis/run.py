"""Runs the statistical plan of protocol 7.9 on a per-episode table (pilot outputs).

    python -m analysis.run --episodes runs/pilot_run/stats/episodes.csv --out runs/pilot_run/stats

Emits every number the plan mentions: H1a / H1b checks with Clopper-Pearson bounds and the IUT
outcome, CR_total of every arm, the descriptive difference Proposed - Baseline 2 with its
two-factor paired cluster-bootstrap interval, the seven primary checks of H2 / H3 (estimate, standard
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
from analysis.design import (DesignCompletenessError, DesignSpec, normalize_mode, parse_bool,
                             parse_rho, select_descriptive_rows, select_h1_diff_rows,
                             select_h2_rows, select_h3_rows, validate_design_completeness)

SPEC = DesignSpec()
H1_RHOS = SPEC.h1a_rhos           # rho levels of H1a; 0.4 is boundary, 0.1 / 0.2 are descriptive
H23_RHOS = SPEC.h23_rhos          # rho levels of H2 and H3 (rho >= tau_eff)
DESCRIPTIVE_RHOS = SPEC.descriptive_rhos


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _rho_str(rho):
    return "" if rho is None else str(rho)


def _rho_role(mode, rho, spec=SPEC):
    if mode == spec.b2_arm:
        return "H1b"
    if rho in set(spec.descriptive_rhos):
        return "descriptive (rho < tau_eff)"
    if rho in set(spec.h1a_rhos):
        return "primary (H1a, H2/H3)"
    if rho in set(spec.h23_rhos):
        return "primary (H2/H3) / H1 boundary"
    return "descriptive"


def h1_checks(rows):
    """One check per policy x mode x D x rho (filtered arms) or per policy x D (Baseline 2).
    Includes descriptive rho levels (0.1, 0.2, 0.4 s) alongside H1a levels (0.6, 1.0 s)."""
    groups = {}
    for r in rows:
        mode = normalize_mode(r["mode"])
        D = float(r["D"])
        rho = None if mode == SPEC.b2_arm else parse_rho(r.get("rho"))
        unit = str(r["unit"]).strip()
        key = (mode, D, rho, unit)
        groups.setdefault(key, []).append(r)
    out = []
    ordered = sorted(groups.items(),
                     key=lambda kv: (kv[0][0], kv[0][1], -1.0 if kv[0][2] is None else kv[0][2], kv[0][3]))
    for (mode, D, rho, unit), ep in ordered:
        valid = [r for r in ep if parse_bool(r["correct_start"], "correct_start")]
        x = sum(parse_bool(r["h1_event"], "h1_event") for r in valid)
        n = len(valid)
        crashes = sum(parse_bool(r["crashed"], "crashed") for r in ep)
        crashes_valid = sum(parse_bool(r["crashed"], "crashed") for r in valid)
        out.append(dict(mode=mode, D=D, rho=_rho_str(rho), unit=unit, n=len(ep), n_valid=n, events=x,
                        rho_cls=sorted({str(r["rho_cls"]).strip() for r in ep}),
                        crashes=crashes, CR_total=crashes / len(ep),
                        CR_total_valid=(crashes_valid / n) if n else None,
                        share=(x / n) if n else None, upper=B.cp_upper(x, n) if n else None,
                        lower=B.cp_lower(x, n) if n else None, decision=B.h1_decision(x, n)))
    return out


def h1_summary(checks):
    arms = set(SPEC.h1a_arms)
    h1_rhos = {round(float(r), 6) for r in H1_RHOS}
    a = [c for c in checks
         if normalize_mode(c["mode"]) in arms and parse_rho(c.get("rho")) in h1_rhos]
    b = [c for c in checks if normalize_mode(c["mode"]) == SPEC.b2_arm]
    return dict(H1a=dict(checks=len(a), confirmed=sum(c["decision"] == B.CONFIRMED for c in a),
                         exceeded=sum(c["decision"] == B.EXCEEDED for c in a),
                         outcome="confirmed" if a and all(c["decision"] == B.CONFIRMED for c in a) else "not confirmed"),
                H1b=dict(checks=len(b), exceeded=sum(c["decision"] == B.EXCEEDED for c in b),
                         confirmed=sum(c["decision"] == B.CONFIRMED for c in b),
                         outcome="confirmed" if b and all(c["decision"] == B.EXCEEDED for c in b) else "not confirmed"))


def h1_difference(rows, n_boot, rng):
    """CR_ego(Proposed, mode B, rho) - CR_ego(Baseline 2) per D and rho in H1_RHOS on correct
    starts, with the two-way cluster bootstrap that keeps the shared episode_seed.  Baseline 2
    events are the ones classified at the reference rho; extra Baseline 2 scenarios are excluded
    from the pair and reported."""
    diff_rows = select_h1_diff_rows(rows, SPEC)

    def events(mode, D, rho=None):
        g = {}
        for r in diff_rows:
            if (normalize_mode(r["mode"]) == mode and float(r["D"]) == D
                    and (rho is None or parse_rho(r.get("rho")) == rho)
                    and parse_bool(r["correct_start"], "correct_start")):
                g.setdefault(str(r["unit"]).strip(), {})[int(float(r["episode_seed"]))] = int(
                    parse_bool(r["h1_event"], "h1_event"))
        return g

    out = []
    for D in sorted({float(r["D"]) for r in diff_rows}):
        base = events(SPEC.b2_arm, D)
        for rho in H1_RHOS:
            prop = events("B", D, round(float(rho), 6))
            if prop and base:
                b = B.bootstrap_h1_diff_twoway(prop, base, n_boot=n_boot,
                                               seed=int(rng.integers(2**31 - 1)))
                out.append(dict(D=D, rho=rho, difference=b["diff"], lower=b["ci_low"],
                                upper=b["ci_high"], policies=(b["n_units_proposed"],
                                                              b["n_units_baseline2"]),
                                n_common_scenarios=b["n_common_scenarios"],
                                n_extra_b2_excluded=b["n_extra_b2_excluded"]))
    return out


def descriptive_cells(rows, spec=SPEC):
    """Descriptive cell summary across every valid arm and rho level present in the table,
    including descriptive rho in {0.1, 0.2} s and boundary rho = 0.4 s."""
    desc_rows = select_descriptive_rows(rows, spec)
    groups = {}
    for r in desc_rows:
        mode = normalize_mode(r["mode"])
        D = float(r["D"])
        rho = None if mode == spec.b2_arm else parse_rho(r.get("rho"))
        groups.setdefault((mode, D, rho), []).append(r)
    ordered = sorted(groups.items(),
                     key=lambda kv: (kv[0][0], kv[0][1], -1.0 if kv[0][2] is None else kv[0][2]))
    out = []
    for (mode, D, rho), ep in ordered:
        units = {str(r["unit"]).strip() for r in ep}
        valid = [r for r in ep if parse_bool(r["correct_start"], "correct_start")]
        n_ep, n_valid = len(ep), len(valid)
        crashes = sum(parse_bool(r["crashed"], "crashed") for r in ep)
        h1_events = sum(parse_bool(r["h1_event"], "h1_event") for r in valid)
        e_sum = float(sum(float(r["E"]) for r in ep))
        n_exec_sum = float(sum(float(r["N_exec"]) for r in ep))
        n_blocked_sum = float(sum(float(r["N_blocked"]) for r in ep))
        n_att_sum = float(sum(float(r["N_att"]) for r in ep))
        out.append(dict(
            mode=mode, D=D, rho=_rho_str(rho), role=_rho_role(mode, rho, spec),
            n_units=len(units), n_episodes=n_ep, n_valid=n_valid,
            CR_total=(crashes / n_ep) if n_ep else None,
            CR_ego_op=(h1_events / n_valid) if n_valid else None,
            v_eff=float(np.mean([float(r["v_eff"]) for r in ep])) if ep else None,
            N_exec_rate=(n_exec_sum / e_sum) if e_sum > 0 else None,
            N_blocked_rate=(n_blocked_sum / e_sum) if e_sum > 0 else None,
            N_att_rate=(n_att_sum / e_sum) if e_sum > 0 else None,
            BLCR=(n_blocked_sum / n_att_sum) if n_att_sum > 0 else None,
        ))
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
    p.add_argument("--boot-diff", type=int, default=10000, help="paired bootstrap of the H1 difference")
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--design", choices=("main", "pilot"), default="main",
                   help="main: enforce the full pre-registered design before IUT/models; "
                        "pilot: diagnostic run, completeness check skipped")
    args = p.parse_args(argv)
    rows = read(args.episodes)
    if args.design == "main":
        try:
            validate_design_completeness(rows)
        except DesignCompletenessError as exc:
            sys.stderr.write(f"design completeness error: {exc}\n")
            return 2
    desc_rows = select_descriptive_rows(rows, SPEC)
    h2_rows = select_h2_rows(rows, SPEC)
    h3_rows = select_h3_rows(rows, SPEC)
    rng = np.random.default_rng(args.seed)
    R = dict(args=vars(args), n_rows=len(rows), design=args.design)

    checks = h1_checks(desc_rows)
    R["h1_checks"], R["h1"] = checks, h1_summary(checks)
    R["h1_difference"] = h1_difference(rows, args.boot_diff, rng)
    R["descriptive"] = descriptive_cells(desc_rows, SPEC)

    R["excluded"] = dict(E0=sum(float(r["E"]) <= 0 for r in h3_rows),
                         N_att0=sum(float(r["N_att"]) <= 0 for r in h3_rows),
                         episodes=len(h3_rows))
    R["primary"] = P.run_primary(h3_rows, args.boot, args.boot_sens, args.seed)
    R["control"] = P.CONTROL.function(h3_rows, args.boot, np.random.default_rng([args.seed, 50]))
    R["dispersion"] = dict(N_exec=P.dispersion(h2_rows, "N_exec", P.H2_TERMS, "B"),
                           N_blocked=P.dispersion(h3_rows, "N_blocked", P.H3_TERMS, "AB"),
                           N_att=P.dispersion(h3_rows, "N_att", P.H3_TERMS, "AB"),
                           blocked_share=P.dispersion(h3_rows, "N_blocked", P.H3_TERMS, "AB", trials="N_att"))
    means, reps = P.bootstrap_cells(h2_rows, "v_eff", args.boot_sens, np.random.default_rng([args.seed, 60]))
    con = P.contrast_of_differences(means)
    lo, hi = P.interval([P.contrast_of_differences(m) for m in reps])
    R["contrast"] = dict(estimate=con, lower=lo, upper=hi)
    R["adjacent"], R["page"] = {}, {}
    for outcome in ("v_eff", "N_exec"):
        m0, rp = (means, reps) if outcome == "v_eff" else P.bootstrap_cells(
            h2_rows, outcome, args.boot_sens, np.random.default_rng([args.seed, 61]))
        adj = P.adjacent_contrasts(m0)
        R["adjacent"][outcome] = {k: dict(estimate=v, interval=P.interval([P.adjacent_contrasts(m)[k] for m in rp]))
                                  for k, v in adj.items()}
        for over in ("rho", "D"):
            levels, tab = P.trend_table(h2_rows, outcome, over)
            if tab.size:
                L, pv, how = P.page_test(tab)
                R["page"][f"{outcome} over {over}"] = dict(levels=levels, blocks=len(tab), L=L, p=pv, method=how)

    title = ("# Statistical plan (protocol 7.9)" if args.design == "main"
             else "# Statistical plan on pilot data (protocol 7.9)")
    intro = (f"Input: `{args.episodes}`, {len(rows)} episodes. Parametric bootstrap: {args.boot} "
             f"replicates; two-way cluster bootstrap: {args.boot_sens}; H1 difference: {args.boot_diff}.")
    if args.design != "main":
        intro += (" Pilot policies are not trained: the numbers show that every step of the plan is "
                  "computed, not that H1-H3 hold.")
    L = [title, "",
         f"Design mode: **{args.design}**"
         + ("" if args.design == "main" else
            " (diagnostic: the pre-registered design completeness is not enforced)") + ".", "",
         intro, "",
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
         "Checks with rho = 0.4 are the boundary level and checks with rho in {0.1; 0.2} violate rho >= tau_eff: "
         "both are descriptive and outside the H1a decision.", "",
         "Descriptive difference CR_ego_op(Proposed, mode B) − CR_ego_op(Baseline 2) on the common "
         "correct-start scenarios, two-factor cluster bootstrap (shared scenario multiplicities):", "",
         table(["D", "rho", "difference", "lower 95 %", "upper 95 %", "common scenarios",
                "extra B2 excluded", "policies (Proposed, Baseline 2)"],
               [[d["D"], d["rho"], d["difference"], d["lower"], d["upper"], d["n_common_scenarios"],
                 d["n_extra_b2_excluded"], f"{d['policies'][0]}, {d['policies'][1]}"]
                for d in R["h1_difference"]]), "",
         "## Descriptive cell summary (all rho levels)", "",
         "All present arms and rho levels (including descriptive rho in {0.1; 0.2} s and boundary rho = 0.4 s). "
         "Descriptive levels rho in {0.1; 0.2} s are reported here and in the H1 table above, and excluded from "
         "primary H1a IUT, H2/H3 models, control M5, sensitivity, dispersion, contrasts and Page's test.", "",
         table(["arm", "D", "rho", "role", "units", "episodes", "n_valid", "CR_total", "CR_ego_op",
                "v_eff", "N_exec/E", "N_blocked/E", "N_att/E", "BLCR"],
               [[d["mode"], d["D"], d["rho"] or "—", d["role"], d["n_units"], d["n_episodes"], d["n_valid"],
                 d["CR_total"], d["CR_ego_op"], d["v_eff"], d["N_exec_rate"], d["N_blocked_rate"],
                 d["N_att_rate"], d["BLCR"]] for d in R["descriptive"]]), "",
         "## H2 / H3: seven primary checks", "",
         "Selected rows: rho in {0.4; 0.6; 1.0} s (mode B for H2, modes A and B for H3 and M5; descriptive "
         "rho in {0.1; 0.2} s excluded). Covariates: Dc = (D − 25) / 10, rhoc = rho − 2/3 (grid means), "
         "mode = 1 for mode B. One-sided p: parametric bootstrap of the signed likelihood root under the null "
         "model; Holm over the seven checks in each analysis; sensitivity: two-way cluster bootstrap that keeps "
         "shared scenario ids.", "",
         table(["check", "function", "model", "estimate", "SE", "signed root", "p (bootstrap)",
                "B / B_conv", "stationary share", "p Holm", "converged (via)", "sens. estimate", "sens. 95 % interval",
                "sens. p", "sens. p Holm", "outcome"],
               [[r["check"], f"`{r['function']}`", r["model"], r["primary"]["estimate"], r["primary"]["se"],
                 r["primary"]["root"], r["primary"]["p_boot"],
                 f"{r['primary'].get('B')} / {r['primary'].get('B_conv')}",
                 fmt(r["primary"].get("stationary_share"), 3), r["p_holm_primary"],
                 ("yes" if r["primary"]["converged"] else "no: " + r["primary"]["message"])
                 + f" ({r['primary'].get('converged_via', '')})",
                 r["sensitivity"]["estimate"], f"[{fmt(r['sensitivity']['lower'])}; {fmt(r['sensitivity']['upper'])}]",
                 r["sensitivity"]["p"], r["p_holm_sensitivity"], r["outcome"]] for r in R["primary"]]), "",
         f"Control (outside the family) `{P.CONTROL.function.__name__}`, M5: estimate {fmt(R['control']['estimate'])}, "
         f"SE {fmt(R['control']['se'])}, one-sided bootstrap p {fmt(R['control']['p_boot'])}, "
         f"B/B_conv {R['control'].get('B')}/{R['control'].get('B_conv')}, "
         f"converged: {R['control']['converged']} ({R['control'].get('converged_via', '')}), "
         f"stationary share {fmt(R['control'].get('stationary_share'), 3)}"
         f"{'' if R['control']['converged'] else ' (' + R['control']['message'] + ')'}.", "",
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
