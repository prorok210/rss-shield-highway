"""The single module that creates environments for experiments (experiments/README.md, item 1).
Importing it enables fast_road in this process, including workers started with spawn."""
import os

from sim import fast_road
from experiments.generator import ProtocolHighwayEnv

FAST_ROAD = os.environ.get("RSS_FAST_ROAD", "1") != "0"     # "0" only for debugging
if FAST_ROAD:
    fast_road.enable()

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def make_env(D=25, **config):
    return ProtocolHighwayEnv(config=dict(config, D=D))


def start_state_path(root, D, seed):
    return os.path.join(root, f"D{int(D)}_seed{int(seed)}.pkl")


def reset_cached(env, seed, root):
    """reset(seed) through the start-state cache (README, item 2): the state after the warm-up and
    the insertion is computed once per (D, seed) and reused by every policy, rho and mode."""
    path = start_state_path(root, env.config["D"], seed)
    if os.path.exists(path):
        return env.restore_start_state(path)
    obs, info = env.reset(seed=int(seed))
    os.makedirs(root, exist_ok=True)
    env.save_start_state(path)
    return obs, info
