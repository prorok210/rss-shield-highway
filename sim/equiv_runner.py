"""Run fixed scenarios with the original (mode=orig) or accelerated (mode=fast) highway-env and
write a digest of the full state of every vehicle after every physics frame + obs/reward per step.
Scenarios "gen" and "shield" use the main traffic generator of the protocol (section 7.6,
experiments/generator.py): the frames of reset() (warm-up without the ego, insertion) are hashed
too, the recycling of the window runs inside step()."""
import os, sys, json, hashlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]   # sim/ and the repo root (rss_shield, experiments)
import numpy as np
import highway_env  # noqa
from highway_env.road.road import Road
import fast_road
from experiments.generator import ProtocolHighwayEnv
from rss_shield.shield import RSSShield, ShieldConfig
from rss_shield.params import ACTIONS

mode, out = sys.argv[1], sys.argv[2]
quick = "--quick" in sys.argv
if mode == "fast":
    fast_road.enable()
FR = []
_inner = Road.step


def _hashed_step(self, dt):
    _inner(self, dt)
    h = hashlib.sha256()
    for v in self.vehicles:
        h.update(np.asarray(v.position, dtype=float).tobytes())
        h.update(np.array([v.heading, v.speed, getattr(v, "timer", 0.0) or 0.0,
                           getattr(v, "target_speed", 0.0) or 0.0], dtype=float).tobytes())
        a = v.action or {}
        h.update(np.array([a.get("steering", 0.0), a.get("acceleration", 0.0)], dtype=float).tobytes())
        h.update(b"N" if v.impact is None else np.asarray(v.impact, dtype=float).tobytes())
        h.update(repr((bool(v.crashed), v.lane_index, getattr(v, "target_lane_index", None))).encode())
    FR.append(h.hexdigest()[:16])


Road.step = _hashed_step
res, t0 = [], time.time()
# (kind, D or N, episodes, steps); gen/shield: main generator, D vehicles per km and lane
# termination is ignored: simulation continues after crashes (more interactions per run)
plan = [("gen", 10, 6, 150), ("gen", 25, 4, 120), ("gen", 40, 3, 100), ("std", 100, 3, 100),
        ("std", 160, 2, 60), ("shield", 10, 2, 100), ("shield", 25, 2, 100), ("shield", 40, 2, 100)]
if quick:
    plan = [("gen", 10, 2, 100), ("gen", 25, 1, 80), ("gen", 40, 1, 60), ("std", 100, 1, 60),
            ("shield", 25, 1, 60)]
for kind, p, n_ep, n_st in plan:
    for ep in range(n_ep):
        if kind in ("gen", "shield"):
            env = ProtocolHighwayEnv(config={"D": p})
        else:
            import gymnasium as gym
            env = gym.make("highway-v0", config={"lanes_count": 4, "vehicles_count": p - 1,
                           "vehicles_density": 3.0, "simulation_frequency": 10, "policy_frequency": 5,
                           "duration": 10000, "observation": {"type": "Kinematics", "vehicles_count": 15,
                           "see_behind": True}}).unwrapped
        seed = 1000 * p + ep
        FR.clear()                     # frames of reset() are compared as well
        obs, _ = env.reset(seed=seed)
        reset_frames = len(FR)
        rng = np.random.default_rng(seed)
        shield = RSSShield(env.vehicle, ShieldConfig(rho=0.6)) if kind == "shield" else None
        n_int = 0
        steps, sh = 0, hashlib.sha256(obs.tobytes())
        for k in range(n_st):
            a = int(rng.choice(5, p=[0.2, 0.15, 0.4, 0.15, 0.1]))
            if shield is not None:
                d = shield.decide(ACTIONS[a])
                a = d.command
                n_int += bool(d.intervention)
                sh.update(repr((d.command, d.mask.tolist())).encode())
            obs, r, term, trunc, info = env.step(a)
            sh.update(obs.tobytes()); sh.update(np.float64(r).tobytes())
            steps += 1
        bg_crash = sum(bool(v.crashed) for v in env.road.vehicles if v is not env.vehicle)
        res.append(dict(kind=kind, p=p, ep=ep, steps=steps, interventions=n_int, reset_frames=reset_frames,
                        frames=len(FR), n_veh=len(env.road.vehicles), bg_crashed=bg_crash,
                        ego_crashed=bool(env.vehicle.crashed),
                        frames_digest=hashlib.sha256("".join(FR).encode()).hexdigest()[:16],
                        frame_list=list(FR), steps_digest=sh.hexdigest()[:16]))
        env.close()
json.dump(dict(mode=mode, seconds=round(time.time() - t0, 1), runs=res), open(out, "w"))
print(mode, "done in %.1f s" % (time.time() - t0))
