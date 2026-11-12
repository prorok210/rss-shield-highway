"""Statistical pilot (protocol 8.1): a per-episode table over the design grid for the executable
statistical plan (python -m analysis.run).

    python -m experiments.stats_pilot --out runs/pilot_run/stats

Grid: D in {10, 25, 40}, rho in {0.4, 0.6, 1.0}; the episode seeds are shared by every arm
(scenario id = (D, episode_seed), protocol 7.6.1).
    mode B   one unit per (D, rho, unit seed): a policy evaluated with the filter at that rho;
    mode A   one unit per (D, unit seed): a policy evaluated with the filter at every rho
             (observation of mode A);
    B2       one unit per (D, unit seed): no filter (the arm of H1b), classified at the reference rho.
The policies do not learn: a unit is a seeded random policy with its own action propensities
(PilotPolicy), so that attempts, blocks and executed lane changes occur in every cell (untrained
networks choose one action all the time). The table exercises every step of the analysis on real
outputs of the generator, the filter and the runner; it carries no evidence about H1-H3: a mode
effect cannot exist, because nothing was trained. Counters, start rule and H1 event are those of
experiments.runner / experiments.h1.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from experiments.env import make_env, reset_cached
from experiments.runner import run_episode

DENSITIES = (10, 25, 40)
RHOS = (0.4, 0.6, 1.0)
SEED0 = 900_300                 # episode seeds of the statistical pilot (pilot block 900 000+ of protocol table 4)
COLUMNS = ("mode", "D", "rho", "unit", "unit_seed", "episode_seed", "correct_start", "rho_cls", "crashed",
           "crash_category", "h1_event", "steps", "E", "intercept", "N_att", "N_blocked", "N_busy",
           "N_no_lane", "N_att_int", "N_exec", "N_exec_int", "N_exec_other", "v_eff", "shield_cfg")


class PilotPolicy:
    """Random policy of one unit: a_pi = argmax(b + Gumbel noise), i.e. a_pi ~ softmax(b), with
    the propensities b ~ N(0, 0.5^2) drawn once per unit (the unit effect). No learning."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.b = self.rng.normal(0.0, 0.5, 5)

    def q_values(self, obs):
        return self.b + self.rng.gumbel(size=5)


def row(mode, D, rho, unit, unit_seed, seed, c):
    r = dict(mode=mode, D=D, rho="" if rho is None else rho, unit=unit, unit_seed=unit_seed, episode_seed=seed)
    r.update({k: c[k] for k in COLUMNS if k in c})
    for k in ("correct_start", "crashed", "h1_event"):
        r[k] = int(bool(c[k]))
    r["crash_category"] = c["crash_category"] or ""
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-episode table for the statistical plan (pilot)")
    p.add_argument("--out", required=True)
    p.add_argument("--units", type=int, default=2, help="units (unit seeds) per cell")
    p.add_argument("--episodes", type=int, default=4, help="shared episode seeds per density")
    p.add_argument("--duration", type=int, default=40)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = str(out.parent / "start_states")
    seeds = [SEED0 + i for i in range(args.episodes)]
    rows, t0 = [], time.perf_counter()

    def run(env, agent, mode, D, rho, unit, us):
        for s in seeds:
            c = run_episode(env, agent, rho=0.6 if rho is None else rho, shield_on=mode != "B2",
                            reward_kind="RS" if mode == "B2" else "Rv", explore=False,
                            reset=lambda s=s: reset_cached(env, s, cache), classify=True)
            rows.append(row(mode, D, rho, unit, us, s, c))
        last = rows[-len(seeds):]
        print(f"  {unit}: episodes {len(last)} E {sum(r['E'] for r in last)} N_att {sum(r['N_att'] for r in last)} "
              f"N_blocked {sum(r['N_blocked'] for r in last)} N_exec {sum(r['N_exec'] for r in last)} "
              f"crashes {sum(r['crashed'] for r in last)} H1 events {sum(r['h1_event'] for r in last)} "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)

    for D in DENSITIES:
        env_b = make_env(D=D, duration=args.duration)
        env_a = make_env(D=D, duration=args.duration, obs_mode_a=True)
        for us in range(args.units):
            agent = PilotPolicy(1000 + 10 * D + us)
            for rho in RHOS:
                run(env_a, agent, "A", D, rho, f"A:{D}:{us}", us)
            for k, rho in enumerate(RHOS):
                run(env_b, PilotPolicy(2000 + 100 * D + 10 * k + us), "B", D, rho, f"B:{D}:{rho}:{us}", us)
            run(env_b, PilotPolicy(3000 + 10 * D + us), "B2", D, None, f"B2:{D}:{us}", us)
    with open(out / "episodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"statistical pilot: {len(rows)} episodes, {out / 'episodes.csv'}, {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
