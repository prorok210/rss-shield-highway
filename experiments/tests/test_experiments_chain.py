"""Tests of the experiment chain (protocol v11: 7.2, 7.6, 7.7, 8): generator, 86-feature
observation, start states, semi-MDP transitions, numpy DQN, runner counters."""
import os
import tempfile

import numpy as np

from experiments.env import make_env, reset_cached, start_state_path
from experiments.dqn import DQN, backward, forward, init_mlp
from experiments.generator import (I_BRAKE, I_SPEED, N_KIN, N_OBS, QUEUE_CODES, RHO_INSERT, RHOS,
                                   S_MIN, WINDOW_BACK, WINDOW_FRONT, idm_equilibrium_speed)
from experiments.runner import TransitionAssembler, run_episode
from rss_shield.params import LENGTH, TAKEOVER

SHORT = dict(warmup=2.0)          # short warm-up for unit tests (the protocol uses 20 s)


def _env(D=10, **kw):
    return make_env(D=D, **dict(SHORT, **kw))


def test_idm_equilibrium_speed():
    for s in (100.0, 40.0, 25.0):
        v = idm_equilibrium_speed(s)
        assert abs(1 - (v / 25.0) ** 4 - ((10.0 + 1.5 * v) / s) ** 2) < 1e-9
    assert idm_equilibrium_speed(100.0) > idm_equilibrium_speed(40.0) > idm_equilibrium_speed(25.0)


def test_initial_placement():
    env = _env(D=25, warmup=0.0)
    env.reset(seed=1)
    ego = env.vehicle
    bg = [v for v in env.road.vehicles if v is not ego]
    assert len(bg) == 4 * 25 - env.start_info["removed"]
    for lane in range(4):
        xs = sorted(v.position[0] for v in bg if v.lane_index[2] == lane)
        assert np.min(np.diff(xs)) >= LENGTH + S_MIN - 1e-9
    v_eq = idm_equilibrium_speed(40.0)
    assert all(abs(v.speed - v_eq) <= 1.0 + 1e-9 for v in bg)
    assert all(23.0 <= v.target_speed <= 27.0 for v in bg)
    assert all(3.5 <= v.DELTA <= 4.5 for v in bg)


def test_insertion_rule_and_start_info():
    for D in (10, 25):
        env = _env(D=D)
        for seed in range(3):
            env.reset(seed=seed)
            info, ego = env.start_info, env.vehicle
            assert env.road.vehicles[0] is ego and env.controlled_vehicles == [ego]
            assert 0 <= info["removed"] <= 2
            assert set(info["unsafe"]) == set(RHOS) == set(info["unsafe_lane"])
            if info["insert_ok"]:
                # leader and follower in the ego lane satisfy d_min_long at rho_max, hence at all rho
                assert not info["unsafe_lane"][RHO_INSERT]
                assert not any(info["unsafe_lane"].values())
            levels = np.asarray(ego.target_speeds, float)
            assert ego.target_speed == levels[np.argmin(np.abs(levels - info["v0"]))]
            assert abs(ego.speed - info["v0"]) < 1e-9


def test_recycling_keeps_the_window_around_the_ego():
    env = _env(D=10)
    env.reset(seed=5)
    n0 = len(env.road.vehicles)
    for _ in range(25):
        _, _, term, _, _ = env.step("FASTER")
        if term:
            break
    x = env.vehicle.position[0]
    assert len(env.road.vehicles) == n0
    margin = 2 * 0.2 * 40.0 + LENGTH + S_MIN
    for v in env.road.vehicles[1:]:
        assert -WINDOW_BACK - margin <= v.position[0] - x <= WINDOW_FRONT + margin


def test_observation_86_features_and_queue_state():
    env = _env(D=10)
    obs, _ = env.reset(seed=2)
    assert N_OBS == 86 and obs.shape == (N_OBS,) and obs.dtype == np.float32
    assert env.observation_space.shape == (N_OBS,)
    assert not obs[N_KIN:N_KIN + len(QUEUE_CODES)].any()
    obs, *_ = env.step(TAKEOVER)
    assert obs[N_KIN + QUEUE_CODES.index(TAKEOVER)] == 1.0      # waiting in the FIFO
    assert obs[N_KIN + I_BRAKE] == 0.0
    obs, *_ = env.step("IDLE")
    assert obs[N_KIN + QUEUE_CODES.index("IDLE")] == 1.0
    assert obs[N_KIN + I_BRAKE] == 1.0                          # TAKEOVER active: braking mode
    assert 0.0 <= obs[N_KIN + I_SPEED] <= 1.0
    env_a = _env(D=10, obs_mode_a=True)
    env_a.reset(seed=2)
    obs, *_ = env_a.step(TAKEOVER)
    assert obs[N_KIN + QUEUE_CODES.index("SLOWER")] == 1.0      # Mode A mapping
    assert obs[N_KIN + QUEUE_CODES.index(TAKEOVER)] == 0.0
    obs, *_ = env_a.step("IDLE")
    assert obs[N_KIN + I_BRAKE] == 0.0


def test_start_state_cache_is_bit_identical():
    actions = ["IDLE", "LANE_LEFT", "FASTER", "IDLE", "SLOWER"] * 4

    def run(env, root):
        obs, _ = reset_cached(env, 7, root) if root else env.reset(seed=7)
        trace = [obs.copy(), np.array(env.np_random.bit_generator.state["state"]["state"], dtype=object)]
        for a in actions:
            obs, _, term, _, _ = env.step(a)
            trace.append(obs.copy())
            trace.append(np.array([[v.position[0], v.position[1], v.speed, v.heading]
                                   for v in env.road.vehicles]).ravel())
            if term:
                break
        return trace

    with tempfile.TemporaryDirectory() as root:
        first = run(_env(D=10), root)            # computes and saves the start state
        assert os.path.exists(start_state_path(root, 10, 7))
        restored = run(_env(D=10), root)         # restores it
    direct = run(_env(D=10), None)
    assert len(first) == len(restored) == len(direct)
    for x, y, z in zip(first, restored, direct):
        assert np.array_equal(x, y) and np.array_equal(x, z)


def test_semi_mdp_assembler():
    out = []
    asm = TransitionAssembler(0.5, lambda *t: out.append(t))
    m = np.array([1, 1, 0, 1, 1], bool)
    s = [np.full(3, float(i)) for i in range(6)]
    asm.policy_step(s[0], m, 1, meta=(0, 0))
    asm.reward(1.0)
    asm.policy_step(s[1], m, 3, meta=(0, 1))
    asm.reward(1.0)
    asm.reward(2.0)                               # intercept steps: no new transition
    asm.reward(4.0)
    asm.policy_step(s[4], m, 0, meta=(0, 4))
    asm.reward(8.0)
    asm.end(s[5], None, terminated=True)
    assert [(t[1], t[2], t[4], t[5]) for t in out] == [(1, 1.0, False, 1), (3, 3.0, False, 3),
                                                       (0, 8.0, True, 1)]
    assert np.array_equal(out[1][3], s[4]) and np.array_equal(out[1][6], m)
    assert not out[2][6].any()
    dropped, kept = [], []
    a1 = TransitionAssembler(0.9, lambda *t: dropped.append(t))
    a1.policy_step(s[0], m, 1)
    a1.reward(1.0)
    a1.reward(1.0)
    a1.end(s[2], np.zeros(5, bool), terminated=False)   # truncation with empty A_safe: dropped
    assert dropped == [] and a1.dropped == 1
    a2 = TransitionAssembler(0.9, lambda *t: kept.append(t))
    a2.policy_step(s[0], m, 1)
    a2.reward(1.0)
    a2.reward(1.0)
    a2.end(s[2], m, terminated=False)                  # truncation with A_safe: bootstrap
    assert len(kept) == 1 and kept[0][4] is False and kept[0][5] == 2
    assert abs(kept[0][2] - 1.9) < 1e-12


def test_dqn_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    params = init_mlp((4, 6, 3), rng)
    x, w = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))

    def f(ps):
        return float((forward(ps, x)[0] * w).sum())

    _, acts = forward(params, x)
    grads = backward(params, acts, w)
    for p, g in zip(params, grads):
        for flat in rng.choice(p.size, size=min(5, p.size), replace=False):
            idx = np.unravel_index(flat, p.shape)
            old = p[idx]
            p[idx] = old + 1e-6
            up = f(params)
            p[idx] = old - 1e-6
            down = f(params)
            p[idx] = old
            assert abs((up - down) / 2e-6 - g[idx]) < 1e-6 * max(1.0, abs(g[idx]))


def test_dqn_masked_target_training_and_save_load():
    rng = np.random.default_rng(1)
    agent = DQN(n_obs=6, hidden=(8, 8), seed=3, buffer_size=100, batch_size=4, learning_starts=4,
                target_update=5)
    s2 = rng.normal(size=(3, 6))
    q2 = agent.q_batch(s2, agent.target)
    m2 = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], bool)
    y = agent.targets(np.array([1.0, 2.0, 3.0]), s2, np.array([0.0, 0.0, 1.0]), np.array([1, 3, 2]), m2)
    assert np.isclose(y[0], 1.0 + 0.99 * q2[0, 0])
    assert np.isclose(y[1], 2.0 + 0.99 ** 3 * q2[1, 4])
    assert y[2] == 3.0
    for i in range(12):
        agent.buffer.add(rng.normal(size=6), i % 5, 1.0, rng.normal(size=6), False, 1, np.ones(5, bool))
        agent.on_env_step()
    assert agent.updates == 9 and agent.last_loss is not None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.npz")
        agent.save(path)
        loaded = DQN.load(path)
    x = rng.normal(size=(7, 6))
    assert np.array_equal(agent.q_batch(x), loaded.q_batch(x))
    assert np.array_equal(agent.q_batch(x, agent.target), loaded.q_batch(x, loaded.target))
    assert (loaded.steps, loaded.updates, loaded.opt.t) == (agent.steps, agent.updates, agent.opt.t)
    loaded.buffer = agent.buffer
    agent.train_step()
    loaded.train_step()                            # same Adam state and RNG: same update
    assert np.array_equal(agent.q_batch(x), loaded.q_batch(x))


def test_runner_counters_and_semi_mdp_records():
    env = _env(D=25, duration=10)
    agent = DQN(seed=1, learning_starts=20, batch_size=16, buffer_size=1000)
    log = []
    stats = [run_episode(env, agent, rho=0.6, train=True, explore=True, seed=s, episode=i,
                         steps_log=log) for i, s in enumerate((11, 12))]
    for c in stats:
        assert c["E"] + c["intercept"] == c["steps"]
        assert c["takeover"] + c["abort"] + c["fallback"] == c["intercept"]
        assert c["N_blocked"] <= c["N_att"] <= c["E"]
        assert c["N_exec_int"] <= c["N_exec"]
        assert c["steps"] <= 50
    rows = {(r["episode"], r["k"]): r for r in log}
    b = agent.buffer
    assert b.size == sum(c["stored"] for c in stats) > 0
    for i in range(b.size):
        e, k0 = b.meta[i]
        n = int(b.n[i])
        assert rows[(e, k0)]["policy_step"] == 1
        assert all(rows[(e, k0 + j)]["policy_step"] == 0 for j in range(1, n))
        if (e, k0 + n) in rows:
            assert rows[(e, k0 + n)]["policy_step"] == 1
        assert abs(sum(0.99 ** j * rows[(e, k0 + j)]["r"] for j in range(n)) - b.R[i]) < 1e-12
        assert b.d[i] or b.m2[i].any()
    base = run_episode(env, agent, shield_on=False, reward_kind="RS", seed=13)
    assert base["E"] == base["steps"] and base["intercept"] == 0 and base["masked"] == 0
