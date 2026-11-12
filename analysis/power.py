"""Power statements of H1a / H1b (protocol 7.9), block 7 of the review.

    python -m analysis.power [--starts runs/pilot_run/start_states.csv] [--out power.md]

Design: H1a - 60 checks (5 policies x 2 modes x 3 D x 2 rho), 1000 test episodes each, planned
true risk 1 %; H1b - 15 checks (5 Baseline 2 policies x 3 D), 1500 episodes each, planned risk
5 %. The target power is 0.8 for each hypothesis separately.

What is and is not claimed:
  * the checks of a hypothesis share the test scenarios, so they are not independent. The joint
    power is reported as a lower bound that assumes nothing about the dependence (Boole):
    1 - K (1 - p1);
  * the product p1^K is printed only as a naive approximation for independent checks; it is not
    the joint power under shared scenarios;
  * a requirement that H1a and H1b are confirmed simultaneously is a different condition with its
    own target; it is computed separately (simultaneous_bound) and is not part of the plan;
  * every threshold and power is computed from the number of episodes with a correct start
    (n_valid), never from the nominal budget. PowerInput.validate() fails if the reported count
    contradicts the design.
"""
import argparse
import csv
import sys
from dataclasses import dataclass

import numpy as np

from analysis.binomial import (ALPHA, DELTA, boole_lower_bound, confirm_threshold, exceed_threshold,
                               independent_product, power_confirm, power_exceed)

TARGET = 0.8
DENSITIES = (10, 25, 40)
H1A = dict(name="H1a", kind="confirm", n_nominal=1000, risk=0.01, checks_per_D=20,
           label="Proposed, rho in {0.6; 1.0}: 5 policies x 2 modes x 2 rho per D")
H1B = dict(name="H1b", kind="exceed", n_nominal=1500, risk=0.05, checks_per_D=5,
           label="Baseline 2: 5 policies per D")


class DesignError(ValueError):
    """The reported number of correct-start episodes contradicts the design."""


@dataclass
class PowerInput:
    """Episodes with a correct start per density for one hypothesis. The start state depends on
    (D, episode_seed) only, so all checks of a density share one n_valid."""
    spec: dict
    n_valid: dict                     # D -> episodes with a correct start
    source: str = "nominal budget"

    def validate(self):
        n_nom = self.spec["n_nominal"]
        if set(self.n_valid) != set(DENSITIES):
            raise DesignError(f"{self.spec['name']}: n_valid is needed for D = {DENSITIES}, got {sorted(self.n_valid)}")
        for D, n in self.n_valid.items():
            if int(n) != n or n < 0:
                raise DesignError(f"{self.spec['name']}, D = {D}: n_valid = {n} is not an episode count")
            if n > n_nom:
                raise DesignError(f"{self.spec['name']}, D = {D}: n_valid = {n} exceeds the "
                                  f"{n_nom} episodes of the design")
        return self


def check_shared_seeds(h1a, h1b):
    """The 1000 seeds of H1a are the first 1000 of the 1500 seeds of H1b: per density the correct
    starts of H1a are a subset of those of H1b, and H1b cannot add more than 500 starts."""
    for D in DENSITIES:
        a, b = h1a.n_valid[D], h1b.n_valid[D]
        if a > b or b - a > H1B["n_nominal"] - H1A["n_nominal"]:
            raise DesignError(f"D = {D}: n_valid of H1a ({a}) and of H1b ({b}) are inconsistent with "
                              f"shared seeds (first 1000 of 1500)")


def single_power(spec, n, risk=None):
    risk = spec["risk"] if risk is None else risk
    return power_confirm(n, risk) if spec["kind"] == "confirm" else power_exceed(n, risk)


def threshold(spec, n):
    return confirm_threshold(n) if spec["kind"] == "confirm" else exceed_threshold(n)


def hypothesis_power(inp, risk=None):
    """Powers of one hypothesis from the realized correct-start counts."""
    inp.validate()
    spec = inp.spec
    per_D = {D: single_power(spec, n, risk) for D, n in inp.n_valid.items()}
    powers = [per_D[D] for D in DENSITIES for _ in range(spec["checks_per_D"])]
    return dict(name=spec["name"], target=TARGET, risk=spec["risk"] if risk is None else risk,
                n_nominal=spec["n_nominal"], n_valid=dict(inp.n_valid), source=inp.source,
                thresholds={D: threshold(spec, n) for D, n in inp.n_valid.items()},
                single=per_D, checks=len(powers), bound=boole_lower_bound(powers),
                product=independent_product(powers))


def simultaneous_bound(*results):
    """Lower bound of P(every listed hypothesis is confirmed), free of any dependence assumption:
    Boole over the hypotheses. A separate condition - not the per-hypothesis target."""
    return max(0.0, 1.0 - sum(1.0 - r["bound"] for r in results))


def simulate_joint_power(inp, n_sim, rng, icc=0.0, risk=None):
    """Monte-Carlo joint power of one hypothesis under shared scenarios. Episodes are drawn only
    for correct-start scenarios (n_valid per density, validated against the design). Dependence
    model (an assumption, stated in the output): a scenario s has its own event probability
    q_s ~ Beta with mean `risk` and intra-scenario correlation icc; given q_s the checks of the
    density are independent. icc = 0: independent checks; icc -> 1: identical checks."""
    inp.validate()
    spec = inp.spec
    risk = spec["risk"] if risk is None else risk
    ok = np.ones(n_sim, bool)
    for D in DENSITIES:
        n, K = int(inp.n_valid[D]), spec["checks_per_D"]
        thr = threshold(spec, n)
        for i in range(n_sim):
            if not ok[i]:
                continue
            if icc <= 0.0:
                q = np.full(n, risk)
            else:
                k = (1.0 - icc) / icc
                q = rng.beta(risk * k, (1.0 - risk) * k, n)
            x = (rng.random((K, n)) < q[None, :]).sum(axis=1)
            ok[i] = bool(np.all(x <= thr)) if spec["kind"] == "confirm" else bool(np.all(x >= thr))
    return float(ok.mean())


def read_start_counts(path, rho="1.0"):
    """Correct starts per density from a start-state table of the pilot (column unsafe_b_<rho>:
    the ego-lane label at RHO_START). Returns {D: (n, n_valid)}."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            D = int(float(row["D"]))
            n, v = out.get(D, (0, 0))
            out[D] = (n + 1, v + int(not int(float(row[f"unsafe_b_{rho}"]))))
    return out


def report(results, sims, extra):
    L = ["# Power of H1a and H1b (protocol 7.9)", "",
         f"Risk threshold delta = {DELTA:.0%}, one-sided level alpha = {ALPHA}. Target power: {TARGET} for "
         "each hypothesis separately.", ""]
    for r in results:
        L += [f"## {r['name']}: {r['checks']} checks, planned true risk {r['risk']:.1%}", "",
              f"- target: {r['target']}",
              f"- nominal budget: {r['n_nominal']} episodes per check",
              f"- realized correct-start episodes per check (source: {r['source']}): "
              + ", ".join(f"D = {D}: {r['n_valid'][D]}" for D in DENSITIES),
              "- threshold from n_valid: " + ", ".join(f"D = {D}: {r['thresholds'][D]} events" for D in DENSITIES),
              "- power of one check: " + ", ".join(f"D = {D}: {r['single'][D]:.4f}" for D in DENSITIES),
              f"- **joint power, lower bound: {r['bound']:.3f}** - sampling assumption: none about the "
              "dependence between checks (Boole's inequality; the checks share scenarios)",
              f"- naive independent approximation (product of the powers of the checks): {r['product']:.3f} - "
              "assumes independent checks, which shared scenarios violate; not the joint power",
              f"- bound against the target: {'met' if r['bound'] >= r['target'] else 'NOT met'}"]
        for icc, value in sims.get(r["name"], []):
            L.append(f"- simulation, shared-scenario model with intra-scenario correlation {icc}: {value:.3f} - "
                     "sampling assumption: exchangeable scenario effect, episodes drawn only for correct starts")
        L.append("")
    L += extra
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Power of H1a / H1b from correct-start episode counts")
    p.add_argument("--starts", help="start-state table of the pilot (start_states.csv)")
    p.add_argument("--n-valid-h1a", type=int, nargs=3, metavar=("D10", "D25", "D40"),
                   help="realized correct-start episodes of the H1a checks per density")
    p.add_argument("--n-valid-h1b", type=int, nargs=3, metavar=("D10", "D25", "D40"))
    p.add_argument("--simultaneous-target", type=float, default=TARGET)
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out")
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    inputs = [(PowerInput(H1A, {D: H1A["n_nominal"] for D in DENSITIES}, "design: every start correct"),
               PowerInput(H1B, {D: H1B["n_nominal"] for D in DENSITIES}, "design: every start correct"))]
    if args.n_valid_h1a and args.n_valid_h1b:
        inputs.append((PowerInput(H1A, dict(zip(DENSITIES, args.n_valid_h1a)), "realized counts (command line)"),
                       PowerInput(H1B, dict(zip(DENSITIES, args.n_valid_h1b)), "realized counts (command line)")))
    extra = []
    if args.starts:
        counts = read_start_counts(args.starts)
        share = {D: counts[D][1] / counts[D][0] for D in DENSITIES}
        src = (f"projection from the pilot start states {args.starts}: "
               + ", ".join(f"D = {D}: {counts[D][1]} of {counts[D][0]} correct" for D in DENSITIES))
        inputs.append((PowerInput(H1A, {D: int(np.floor(H1A["n_nominal"] * share[D])) for D in DENSITIES}, src),
                       PowerInput(H1B, {D: int(np.floor(H1B["n_nominal"] * share[D])) for D in DENSITIES}, src)))
    blocks = []
    for a, b in inputs:
        a.validate(), b.validate()
        check_shared_seeds(a, b)
        ra, rb = hypothesis_power(a), hypothesis_power(b)
        sims = {r["name"]: [(icc, simulate_joint_power(inp, args.n_sim, rng, icc))
                            for icc in (0.0, 0.2)] for r, inp in ((ra, a), (rb, b))}
        sim = simultaneous_bound(ra, rb)
        extra_block = [
            "## Simultaneous confirmation of H1a and H1b (a separate condition)", "",
            f"- parameters: target {args.simultaneous_target}, risks {ra['risk']:.1%} (H1a) and {rb['risk']:.1%} (H1b), "
            "the same n_valid as above",
            f"- lower bound without any dependence assumption: 1 - (1 - {ra['bound']:.4f}) - (1 - {rb['bound']:.4f}) = "
            f"**{sim:.4f}** ({'meets' if sim >= args.simultaneous_target else 'does not meet'} {args.simultaneous_target})",
            "- this condition is not part of the plan: the target of 0.8 applies to each hypothesis separately, and the "
            "sample sizes are not changed because of shared scenarios", ""]
        blocks.append(report([ra, rb], sims, extra_block))
    text = "\n".join(blocks)
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
