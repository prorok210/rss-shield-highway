"""Usage (from the repo root): python sim/bench.py orig|fast [steps]
Time per decision step (env.step = 2 physics frames + obs + reward + window recycling), per
shield.decide() and per reset() (placement, 20 s warm-up, insertion) of the main traffic generator
(experiments/generator.py, section 7.6), original highway-env vs fast_road."""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]   # sim/ and the repo root (rss_shield, experiments)
import numpy as np
import fast_road
from experiments.generator import ProtocolHighwayEnv
from rss_shield.shield import RSSShield, ShieldConfig
from rss_shield.params import ACTIONS

mode = sys.argv[1]
if mode == "fast":
    fast_road.enable()
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out = {}
for D in (10, 25, 40):
    t_env, t_mask, t_reset, n_veh = [], [], [], []
    for ep in range(3):
        env = ProtocolHighwayEnv(config={"D": D})
        t0 = time.perf_counter()
        env.reset(seed=500 + ep)
        t_reset.append(time.perf_counter() - t0)
        n_veh.append(len(env.road.vehicles))
        sh = RSSShield(env.vehicle, ShieldConfig(rho=0.6))
        rng = np.random.default_rng(ep)
        for k in range(n_steps):
            t0 = time.perf_counter()
            d = sh.decide(ACTIONS[int(rng.choice(5, p=[0.1, 0.5, 0.1, 0.2, 0.1]))])
            t1 = time.perf_counter()
            env.step(d.command)
            t2 = time.perf_counter()
            t_mask.append(t1 - t0); t_env.append(t2 - t1)
        env.close()
    out[D] = dict(env_ms=1000 * float(np.mean(t_env)), mask_ms=1000 * float(np.mean(t_mask)),
                  reset_s=float(np.mean(t_reset)), n_veh=float(np.mean(n_veh)), n=len(t_env))
    print(mode, D, "env %.1f ms/step, shield %.1f ms/step, reset %.2f s, %.0f vehicles (n=%d)"
          % (out[D]["env_ms"], out[D]["mask_ms"], out[D]["reset_s"], out[D]["n_veh"], len(t_env)), flush=True)
json.dump(out, open(os.path.join(HERE, "bench_%s.json" % mode), "w"))
