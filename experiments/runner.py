"""Episode loop with the RSS shield (Proposed, mode B or A) or without it (Baseline 1 / 2):
semi-MDP transitions (7.2) and per-episode counters (7.7).

Accounting (protocol v11, 7.7):
  policy step      A_safe(s_k) is not empty; the executed command is a policy action;
  intercept step   A_safe(s_k) is empty; the shield issues TAKEOVER, ABORT or FALLBACK;
  E                number of policy steps (exposure);
  N_att, N_blocked policy steps whose unmasked greedy action a_pi is a lane change, and those of
                   them with a_pi outside A_safe; BLCR = N_blocked / N_att, both counted on E;
  N_att_int        intercept steps whose a_pi is a lane change (reported separately, not in BLCR);
  N_exec           lane markings crossed by the ego (lane_index changed during a step);
  N_exec_int       of them, crossings while a supervisory command is active (maneuver completed
                   under TAKEOVER / FALLBACK); they remain in N_exec;
  masked           policy steps where the executed command differs from a_pi;
  R_int            (masked + intercept steps) / all steps.
"""
import time

import numpy as np

from rss_shield.params import (ABORT, ACTION_INDEX, ACTIONS, DT, DT_DECISION, FALLBACK, SUPERVISORY,
                               TAKEOVER)
from rss_shield.shield import RSSShield, ShieldConfig
from experiments.generator import queued_command
from experiments.rewards import step_reward

JERK_JUMP = 50.0      # m/s^3: a frame with |da|/dt above this is counted as a jump (diagnostics)
NAMES = tuple(ACTIONS[i] for i in range(len(ACTIONS)))
LANE_CHANGES = ("LANE_LEFT", "LANE_RIGHT")
SUP_KEY = {TAKEOVER: "takeover", ABORT: "abort", FALLBACK: "fallback"}


def mask_str(mask):
    return "".join("1" if m else "0" for m in mask)


class TransitionAssembler:
    """Semi-MDP transitions (7.2). A transition starts at a policy step and ends at the next policy
    step, at termination or at truncation. Intercept steps start no transition; their rewards are
    added with discount: R = sum_{j<n} gamma^j r_{t0+j}, n = number of decision steps covered.
    Termination (collision), also during an intercept: d = 1. Truncation: d = 0 and the target uses
    mask(s_T) if A_safe(s_T) is not empty; otherwise the transition is dropped (no admissible action
    for the maximum of the target)."""

    def __init__(self, gamma, sink):
        self.gamma, self.sink = gamma, sink
        self.open = None
        self.stored = self.dropped = 0

    def policy_step(self, s, mask, a, meta=None):
        if self.open is not None:
            self._store(s, mask, False)
        self.open = dict(s=np.array(s, copy=True), a=int(a), R=0.0, n=0, meta=meta)

    def reward(self, r):
        if self.open is not None:
            self.open["R"] += self.gamma ** self.open["n"] * r
            self.open["n"] += 1

    def end(self, s_last, mask_last, terminated):
        if self.open is None:
            return
        if terminated:
            self._store(s_last, np.zeros(len(NAMES), bool), True)
        elif mask_last is not None and np.any(mask_last):
            self._store(s_last, mask_last, False)
        else:
            self.dropped += 1
            self.open = None

    def _store(self, s2, m2, d):
        o = self.open
        self.sink(o["s"], o["a"], o["R"], np.array(s2, copy=True), bool(d), o["n"],
                  np.asarray(m2, bool), o["meta"])
        self.stored += 1
        self.open = None


def run_episode(env, agent, rho=0.6, shield_on=True, reward_kind="Rv", train=False, explore=False,
                seed=None, reset=None, episode=0, journal=None, steps_log=None):
    """One episode. agent: object with q_values(obs), rng, epsilon() (a DQN for training).
    shield_on=False: Baseline (no filter, every step is a policy step with the full action set).
    explore: epsilon-greedy over A_safe - a random preference vector is passed to the shield, so the
    executed action is uniform over A_safe; a_pi for the counters stays the greedy action.
    reset: optional callable returning (obs, info), e.g. the start-state cache; else reset(seed).
    journal / steps_log: lists that receive one row per decision step."""
    obs, _ = reset() if reset is not None else env.reset(seed=seed)
    ego = env.vehicle
    shield = RSSShield(ego, ShieldConfig(rho=rho)) if shield_on else None
    asm = TransitionAssembler(agent.cfg["gamma"], agent.buffer.add) if train else None
    c = dict(steps=0, E=0, intercept=0, takeover=0, abort=0, fallback=0, N_att=0, N_blocked=0,
             N_att_int=0, N_exec=0, N_exec_int=0, masked=0, ret=0.0, ttc_events=0, crashed=False,
             crash_intercept=False, crash_sup_active=False,
             t_policy=0.0, t_shield=0.0, t_env=0.0, t_learn=0.0)
    x0, lane_prev = float(ego.position[0]), ego.lane_index[2]
    del ego.acc_log[:]
    k, term, trunc = 0, False, False
    while not (term or trunc):
        t0 = time.perf_counter()
        q = np.asarray(agent.q_values(obs), dtype=float)
        a_pi = NAMES[int(np.argmax(q))]
        pref = agent.rng.random(len(NAMES)) if explore and agent.rng.random() < agent.epsilon() else q
        a_pref = NAMES[int(np.argmax(pref))]
        t1 = time.perf_counter()
        if shield is not None:
            dec = shield.decide(a_pref, {n: float(pref[i]) for i, n in enumerate(NAMES)})
            cmd, mask, reason = dec.command, np.asarray(dec.mask, bool), dec.reason
        else:
            cmd, mask, reason = a_pref, np.ones(len(NAMES), bool), "no_filter"
        t2 = time.perf_counter()
        policy = cmd not in SUPERVISORY
        if policy:
            c["E"] += 1
            if a_pi in LANE_CHANGES:
                c["N_att"] += 1
                c["N_blocked"] += int(not mask[ACTION_INDEX[a_pi]])
            c["masked"] += int(cmd != a_pi)
            if asm is not None:
                asm.policy_step(obs, mask, ACTION_INDEX[cmd], meta=(episode, k))
        else:
            c["intercept"] += 1
            c[SUP_KEY[cmd]] += 1
            c["N_att_int"] += int(a_pi in LANE_CHANGES)
        queued = queued_command(ego)
        obs2, _, term, trunc, _ = env.step(cmd)
        t3 = time.perf_counter()
        r, parts = step_reward(reward_kind, ego, env.road)
        c["ret"] += r
        c["ttc_events"] += int(parts.get("ttc_pen", 0.0) > 0)
        if asm is not None:
            asm.reward(r)
        lane_now = ego.lane_index[2]
        crossed = lane_now != lane_prev
        if crossed:
            c["N_exec"] += 1
            c["N_exec_int"] += int(ego.sup_active)
        lane_prev = lane_now
        if term and ego.crashed:
            c["crashed"] = True
            c["crash_intercept"] = not policy
            c["crash_sup_active"] = bool(ego.sup_active)
        row = dict(episode=episode, k=k, t=round(k * DT_DECISION, 1), a_pi=a_pi, command=cmd,
                   policy_step=int(policy), intervention=int(not policy), reason=reason,
                   mask=mask_str(mask), queued=queued or "", active=ego.active_cmd or "",
                   brake_mode=int(ego.brake_mode), sup_active=int(ego.sup_active),
                   return_active=int(ego.return_active), fallback=int(bool(ego.fallback)),
                   abort_ignored=int(ego.abort_ignored), lane=int(lane_now),
                   target_lane=int(ego.target_lane_index[2]), crossed=int(crossed),
                   x=round(float(ego.position[0]), 3), v=round(float(ego.speed), 3), r=r,
                   crashed=int(ego.crashed))
        if steps_log is not None:
            steps_log.append(row)
        if journal is not None:
            journal.append(row)
        t4 = time.perf_counter()
        if train:
            agent.on_env_step()
        t5 = time.perf_counter()
        c["t_policy"] += t1 - t0
        c["t_shield"] += t2 - t1
        c["t_env"] += t3 - t2
        c["t_learn"] += t5 - t4
        c["steps"] += 1
        k += 1
        obs = obs2
    if asm is not None:
        mask_T = None
        if not term:
            mask_T = (shield.mask()[0].astype(bool) if shield is not None
                      else np.ones(len(NAMES), bool))
        asm.end(obs, mask_T, terminated=term)
        c["stored"], c["dropped"] = asm.stored, asm.dropped
    a = np.asarray(ego.acc_log, dtype=float)
    c["jerk_rms"] = float(np.sqrt(np.mean((np.diff(a) / DT) ** 2))) if a.size > 1 else 0.0
    c["jerk_jumps"] = int(np.sum(np.abs(np.diff(a)) / DT > JERK_JUMP)) if a.size > 1 else 0
    c["frames"] = int(a.size)
    c["dx"] = float(ego.position[0] - x0)
    c["v_eff"] = c["dx"] / float(env.config["duration"])
    c["BLCR"] = c["N_blocked"] / c["N_att"] if c["N_att"] else None
    c["R_int"] = (c["masked"] + c["intercept"]) / c["steps"] if c["steps"] else 0.0
    c["completed"] = bool(trunc and not term)
    c["abort_ignored"] = int(ego.abort_ignored)
    info = getattr(env, "start_info", None) or {}
    c["removed"], c["insert_ok"] = info.get("removed"), info.get("insert_ok")
    c["unsafe_start"] = info.get("unsafe", {}).get(rho)
    c["unsafe_start_lane"] = info.get("unsafe_lane", {}).get(rho)
    c["recycled"] = getattr(env, "recycled", 0)
    return c
