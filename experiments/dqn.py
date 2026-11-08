"""DQN in numpy (protocol 8, Table 5): MLP 86 -> 256 -> 256 -> 5 with ReLU, Adam 5e-4, Huber loss,
gradient-norm clipping 10, replay buffer 50 000, batch 64, gamma 0.99, epsilon 1 -> 0.05 linearly
over 20 000 decision steps, hard target update every 1000 steps, learning starts after 1000 steps,
one gradient step per decision step. Semi-MDP target (7.2):
    y = R + gamma^n * (1 - d) * max_{a' in A_safe(s')} Q_target(s', a').
"""
import json

import numpy as np

DEFAULTS = dict(n_obs=86, n_act=5, hidden=(256, 256), lr=5e-4, gamma=0.99, buffer_size=50_000,
                batch_size=64, eps_start=1.0, eps_end=0.05, eps_steps=20_000, target_update=1000,
                learning_starts=1000, train_freq=1, huber_delta=1.0, max_grad_norm=10.0, seed=42)


def init_mlp(sizes, rng):
    """Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases (as torch.nn.Linear)."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / np.sqrt(n_in)
        params.append(rng.uniform(-bound, bound, size=(n_in, n_out)))
        params.append(rng.uniform(-bound, bound, size=n_out))
    return params


def forward(params, x):
    """Returns (output, activations); activations[i] is the input of layer i."""
    acts = [x]
    h = x
    n_layers = len(params) // 2
    for i in range(n_layers):
        z = h @ params[2 * i] + params[2 * i + 1]
        h = np.maximum(z, 0.0) if i < n_layers - 1 else z
        acts.append(h)
    return h, acts


def backward(params, acts, dout):
    grads = [None] * len(params)
    g = dout
    for i in reversed(range(len(params) // 2)):
        grads[2 * i] = acts[i].T @ g
        grads[2 * i + 1] = g.sum(axis=0)
        if i > 0:
            g = (g @ params[2 * i].T) * (acts[i] > 0)
    return grads


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity, n_obs, n_act):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.s = np.zeros((capacity, n_obs), np.float32)
        self.s2 = np.zeros((capacity, n_obs), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.R = np.zeros(capacity, np.float64)
        self.d = np.zeros(capacity, bool)
        self.n = np.zeros(capacity, np.int64)
        self.m2 = np.zeros((capacity, n_act), bool)
        self.meta = [None] * capacity          # (episode, tick of s) for reports
        self.added = 0

    def add(self, s, a, R, s2, d, n, m2, meta=None):
        i = self.pos
        self.s[i], self.a[i], self.R[i], self.s2[i] = s, a, R, s2
        self.d[i], self.n[i], self.m2[i], self.meta[i] = d, n, m2, meta
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.added += 1

    def record(self, i):
        return dict(index=i, s=self.s[i], a=int(self.a[i]), R=float(self.R[i]), s2=self.s2[i],
                    d=bool(self.d[i]), n=int(self.n[i]), m2=self.m2[i].copy(), meta=self.meta[i])


class DQN:
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown DQN parameters {sorted(unknown)}")
        self.cfg = dict(DEFAULTS, **cfg)
        self.cfg["hidden"] = tuple(self.cfg["hidden"])
        c = self.cfg
        self.rng = np.random.default_rng(c["seed"])
        self.sizes = (c["n_obs"], *c["hidden"], c["n_act"])
        self.params = init_mlp(self.sizes, self.rng)
        self.target = [p.copy() for p in self.params]
        self.opt = Adam(self.params, c["lr"])
        self.buffer = ReplayBuffer(c["buffer_size"], c["n_obs"], c["n_act"])
        self.steps = 0                  # decision steps seen (epsilon, schedule)
        self.updates = 0
        self.last_loss = None

    # ------------------------------------------------------------------ acting
    def epsilon(self):
        c = self.cfg
        frac = min(1.0, self.steps / float(c["eps_steps"]))
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    def q_values(self, obs):
        return forward(self.params, np.asarray(obs, np.float64)[None])[0][0]

    def q_batch(self, obs, params=None):
        return forward(self.params if params is None else params, np.asarray(obs, np.float64))[0]

    # ------------------------------------------------------------------ learning
    def targets(self, R, s2, d, n, m2):
        q2 = self.q_batch(s2, self.target)
        best = np.where(m2, q2, -np.inf).max(axis=1)
        best = np.where(np.isfinite(best), best, 0.0)      # empty mask only with d = 1
        return R + self.cfg["gamma"] ** n * (1.0 - d) * best

    def train_step(self):
        c, b = self.cfg, self.buffer
        idx = self.rng.integers(0, b.size, size=c["batch_size"])
        y = self.targets(b.R[idx], b.s2[idx], b.d[idx].astype(np.float64), b.n[idx], b.m2[idx])
        q, acts = forward(self.params, b.s[idx].astype(np.float64))
        rows = np.arange(idx.size)
        delta = q[rows, b.a[idx]] - y
        k = c["huber_delta"]
        loss = np.where(np.abs(delta) <= k, 0.5 * delta ** 2, k * (np.abs(delta) - 0.5 * k)).mean()
        dq = np.zeros_like(q)
        dq[rows, b.a[idx]] = np.clip(delta, -k, k) / idx.size
        grads = backward(self.params, acts, dq)
        norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
        if norm > c["max_grad_norm"]:
            scale = c["max_grad_norm"] / (norm + 1e-6)
            grads = [g * scale for g in grads]
        self.opt.step(self.params, grads)
        self.updates += 1
        self.last_loss = float(loss)
        return float(loss), norm

    def on_env_step(self):
        """Call after every decision step of a training episode."""
        c = self.cfg
        self.steps += 1
        if (self.steps >= c["learning_starts"] and self.steps % c["train_freq"] == 0
                and self.buffer.size >= c["batch_size"]):
            self.train_step()
        if self.steps % c["target_update"] == 0:
            self.target = [p.copy() for p in self.params]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        arrays = {}
        for name, group in (("q", self.params), ("t", self.target), ("m", self.opt.m), ("v", self.opt.v)):
            for i, p in enumerate(group):
                arrays[f"{name}{i}"] = p
        meta = dict(cfg=self.cfg, steps=self.steps, updates=self.updates, adam_t=self.opt.t,
                    rng=self.rng.bit_generator.state)
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        agent = cls(**meta["cfg"])
        n = len(agent.params)
        agent.params = [z[f"q{i}"].copy() for i in range(n)]
        agent.target = [z[f"t{i}"].copy() for i in range(n)]
        agent.opt.m = [z[f"m{i}"].copy() for i in range(n)]
        agent.opt.v = [z[f"v{i}"].copy() for i in range(n)]
        agent.opt.t = meta["adam_t"]
        agent.steps, agent.updates = meta["steps"], meta["updates"]
        agent.rng.bit_generator.state = meta["rng"]
        return agent
