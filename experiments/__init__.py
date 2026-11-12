"""Experiment code of the protocol: main traffic generator and 86-feature observation
(generator.py), environment factory with fast_road (env.py), rewards, numpy DQN, semi-MDP runner
with the RSS shield and per-episode counters (runner.py), short end-to-end pilot (pilot.py).

Public names (resolved lazily: importing experiments.env switches fast_road on, and
sim/equiv_runner.py must be able to import the generator without it):
    ProtocolHighwayEnv, N_OBS   main-series generator and the size of its observation (86);
    make_env, ENV_ID            environment factory and the gymnasium id of the generator;
    run_episode                 semi-MDP episode loop with the counters of protocol 7.7;
    MAIN_SERIES                 filter configuration of the main series.
"""
_EXPORTS = {"ProtocolHighwayEnv": "experiments.generator", "N_OBS": "experiments.generator",
            "make_env": "experiments.env", "ENV_ID": "experiments.env",
            "run_episode": "experiments.runner", "MAIN_SERIES": "rss_shield.shield"}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")
