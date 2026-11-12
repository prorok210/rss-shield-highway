"""Episode loop with the RSS shield (Proposed, mode B or A) or without it (Baseline 1 / 2):
semi-MDP transitions (7.2) and per-episode counters (7.7).

Accounting (protocol 7.7). The decision set - which steps are policy steps (exposure E) and
which are intercept steps, and where a lane-change request is counted - is defined once in
rss_shield.accounting and used here through count_step():
  policy step      A_safe(s_k) is not empty; the executed command is a policy action;
  intercept step   A_safe(s_k) is empty; the shield issues TAKEOVER, ABORT or FALLBACK. The type of
                   a step is set by its decision tick (who issued the command at s_k), not by the
                   command active during the step: a tick at which the policy decides while an
                   ABORT is still active is a policy step and belongs to E;
  E                number of policy steps (exposure);
  N_att, N_blocked policy steps whose unmasked greedy action a_pi is a lane change, and those of
                   them with a_pi outside A_safe; BLCR = N_blocked / N_att, both counted on E;
  N_busy           of N_blocked: lane changes masked because a maneuver is in progress or waiting
                   in the FIFO (not an RSS block); N_no_lane - masked because there is no lane in
                   that direction; N_att_rss = N_att - N_busy - N_no_lane and
                   N_blocked_rss = N_blocked - N_busy - N_no_lane are reported with
                   BLCR_rss = N_blocked_rss / N_att_rss (descriptive);
  N_att_int        intercept steps whose a_pi is a lane change (reported separately, not in BLCR).
                   A request made while a maneuver is completed under a supervisory command is
                   counted here as long as the supervisor decides, and in N_att / N_blocked /
                   N_busy from the first policy step on - never in both;
  N_exec           executed lane changes, counted by the step that started the maneuver. vehicle.py
                   executes LANE_LEFT / LANE_RIGHT from target_lane_index when the command leaves
                   the FIFO, so a maneuver is registered when the command actually moves
                   target_lane_index away from the lane of the ego, with the decision step that
                   issued it; a return of the target to the current lane before the crossing (the
                   opposite lane change, i.e. a reversal, ABORT or FALLBACK) cancels the record. A
                   crossing of a lane marking (lane_index changed) takes the oldest open record of
                   its direction (journal column exec_from), so every counted lane change belongs to
                   the exposure E even if the crossing itself happens on an intercept step (offset
                   log E of the N_exec model is consistent);
  N_exec_int       of them, crossings while a supervisory command is active (maneuver completed
                   under TAKEOVER / FALLBACK); they remain in N_exec;
  N_exec_other     crossings without an open record of a policy-issued lane change (expected 0,
                   checked);
  masked           policy steps where the executed command differs from a_pi;
  R_int            (masked + intercept steps) / all steps;
  ttc_pen, col_pen sums of the TTC and collision penalties of R_RS over the episode (0 for R_v);
  crash_rear       descriptive: the ego was hit from behind (the nearest other crashed vehicle is
                   behind the ego and overlaps it laterally); not the operational classification;
  correct_start, rho_cls, crash_category, h1_event
                   H1 (experiments.h1, the single definition): start rule at rho = 1.0 s for every
                   arm, classification rho of the arm (rho of the filter, 1.0 s without a filter),
                   operational category of the collision (classify=True) and the H1 event;
                   `crashed` alone gives CR_total, independently of the categories;
  jerk_*           JerkRMS over physics frames; jerk_jumps - frames with |da|/dt > 50 m/s^3,
                   jerk_jump_share - their share of the sum of squared jerk, jerk_rms_smooth -
                   JerkRMS without them (informativeness of JerkRMS, pilot section 6).
The shield runs with MAIN_SERIES (rss_shield.shield) unless shield_cfg is given; the counters
shield_cfg / main_series record which configuration an episode really used.
"""
import time

import numpy as np

from rss_shield.accounting import (LANE_CHANGES, POLICY, block_reason, count_step,
                                   new_counters)
from rss_shield.params import ACTION_INDEX, ACTIONS, DT, DT_DECISION, LENGTH, WIDTH
from rss_shield.shield import MAIN_SERIES, RSSShield, ShieldConfig, describe, is_main_series
from experiments import h1
from experiments.generator import queued_command
from experiments.rewards import step_reward

JERK_JUMP = 50.0      # m/s^3: a frame with |da|/dt above this is counted as a jump (diagnostics)
NAMES = tuple(ACTIONS[i] for i in range(len(ACTIONS)))


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


def rear_contact(ego, road):
    """Descriptive crash attribution: the nearest other crashed vehicle that overlaps the ego
    laterally is behind it (the ego was hit from behind)."""
    xe, ye = float(ego.position[0]), float(ego.position[1])
    hit = [v for v in road.vehicles if v is not ego and v.crashed
           and abs(float(v.position[1]) - ye) < WIDTH and abs(float(v.position[0]) - xe) < 2 * LENGTH]
    if not hit:
        return False
    other = min(hit, key=lambda v: abs(float(v.position[0]) - xe))
    return bool(float(other.position[0]) < xe)


def run_episode(env, agent, rho=0.6, shield_on=True, reward_kind="Rv", train=False, explore=False,
                seed=None, reset=None, episode=0, journal=None, steps_log=None, shield_cfg=None,
                classify=False):
    """One episode. agent: object with q_values(obs), rng, epsilon() (a DQN for training).
    shield_on=False: Baseline (no filter, every step is a policy step with the full action set).
    explore: epsilon-greedy over A_safe - a random preference vector is passed to the shield, so the
    executed action is uniform over A_safe; a_pi for the counters stays the greedy action.
    reset: optional callable returning (obs, info), e.g. the start-state cache; else reset(seed).
    journal / steps_log: lists that receive one row per decision step.
    shield_cfg: ShieldConfig keyword arguments besides rho (default MAIN_SERIES).
    classify: record the physics frames and give a collision its operational category (7.7) at
    the classification rho of the arm (experiments.h1); needed for the H1 event of the episode."""
    obs, _ = reset() if reset is not None else env.reset(seed=seed)
    ego = env.vehicle
    cfg = dict(MAIN_SERIES if shield_cfg is None else shield_cfg)
    shield = RSSShield(ego, ShieldConfig(rho=rho, **cfg)) if shield_on else None
    asm = TransitionAssembler(agent.cfg["gamma"], agent.buffer.add) if train else None
    c = new_counters()
    c.update(N_exec=0, N_exec_int=0, N_exec_other=0, ret=0.0,
             ttc_events=0, ttc_pen=0.0, col_pen=0.0, crashed=False,
             crash_intercept=False, crash_sup_active=False, crash_rear=False,
             t_policy=0.0, t_shield=0.0, t_env=0.0, t_learn=0.0)
    x0, lane_prev = float(ego.position[0]), ego.lane_index[2]
    frames = h1.FrameRecorder(env) if classify else None
    in_fifo = {}                    # FIFO exit frame -> (decision step, policy step) of the command
    open_lc = []                    # lane changes started, not crossed yet: (step or None, direction)
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
        why = dec.violations.get(a_pi) if shield is not None else None
        policy = count_step(c, a_pi, cmd, mask, why) == POLICY
        if policy and asm is not None:
            asm.policy_step(obs, mask, ACTION_INDEX[cmd], meta=(episode, k))
        queued = queued_command(ego)
        in_fifo[ego.frame + ego.k_tau] = (k, policy)
        target_prev = ego.target_lane_index[2]
        if frames is not None:
            frames.decision(cmd)
        obs2, _, term, trunc, _ = env.step(cmd)
        t3 = time.perf_counter()
        r, parts = step_reward(reward_kind, ego, env.road)
        c["ret"] += r
        c["ttc_events"] += int(parts.get("ttc_pen", 0.0) > 0)
        c["ttc_pen"] += float(parts.get("ttc_pen", 0.0))
        c["col_pen"] += float(parts.get("col_pen", 0.0))
        if asm is not None:
            asm.reward(r)
        # commands that left the FIFO during this step (K_TAU = FRAMES_PER_DECISION: the command of
        # the previous decision step, on the first frame, i.e. before any crossing of this step)
        left = [in_fifo.pop(f) for f in sorted(in_fifo) if f < ego.frame]
        target_now = ego.target_lane_index[2]
        if target_now != target_prev:
            d, n = int(np.sign(target_now - lane_prev)), abs(target_now - lane_prev)
            if open_lc and open_lc[-1][1] != d:
                del open_lc[:]          # reversal before the crossing: cancelled, as by ABORT
            del open_lc[n:]
            if len(open_lc) < n:
                src = left[-1] if left else (None, False)
                open_lc.append((src[0] if src[1] else None, d))
        lane_now = ego.lane_index[2]
        crossed = lane_now != lane_prev
        exec_from = ""
        if crossed:
            rec = (open_lc.pop(0) if open_lc and open_lc[0][1] == np.sign(lane_now - lane_prev)
                   else None)
            if rec is not None and rec[0] is not None:
                exec_from = rec[0]
                c["N_exec"] += 1
                c["N_exec_int"] += int(ego.sup_active)
            else:
                c["N_exec_other"] += 1
        lane_prev = lane_now
        if term and ego.crashed:
            c["crashed"] = True
            c["crash_intercept"] = not policy
            c["crash_sup_active"] = bool(ego.sup_active)
            c["crash_rear"] = rear_contact(ego, env.road)
        row = dict(episode=episode, k=k, t=round(k * DT_DECISION, 1), a_pi=a_pi, command=cmd,
                   policy_step=int(policy), intervention=int(not policy),
                   sup_cmd="" if policy else cmd, reason=reason,
                   mask=mask_str(mask), block_reason=block_reason(a_pi, mask, why) if policy else "",
                   queued=queued or "", active_cmd=ego.active_cmd or "",
                   active_sup=int(ego.sup_active), brake_mode=int(ego.brake_mode),
                   return_active=int(ego.return_active), fallback=int(bool(ego.fallback)),
                   abort_ignored=int(ego.abort_ignored), lane=int(lane_now),
                   target_lane=int(ego.target_lane_index[2]), crossed=int(crossed),
                   exec_from=exec_from,
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
    if a.size > 1:
        j2 = (np.diff(a) / DT) ** 2
        jump = np.sqrt(j2) > JERK_JUMP
        c["jerk_jump_share"] = float(j2[jump].sum() / j2.sum()) if j2.sum() > 0 else 0.0
        c["jerk_rms_smooth"] = float(np.sqrt(np.mean(j2[~jump]))) if (~jump).any() else 0.0
    else:
        c["jerk_jump_share"], c["jerk_rms_smooth"] = 0.0, 0.0
    c["frames"] = int(a.size)
    c["dx"] = float(ego.position[0] - x0)
    c["v_eff"] = c["dx"] / float(env.config["duration"])
    c["BLCR"] = c["N_blocked"] / c["N_att"] if c["N_att"] else None
    c["N_att_rss"] = c["N_att"] - c["N_busy"] - c["N_no_lane"]
    c["N_blocked_rss"] = c["N_blocked"] - c["N_busy"] - c["N_no_lane"]
    c["BLCR_rss"] = c["N_blocked_rss"] / c["N_att_rss"] if c["N_att_rss"] else None
    c["R_int"] = (c["masked"] + c["intercept"]) / c["steps"] if c["steps"] else 0.0
    c["completed"] = bool(trunc and not term)
    c["abort_ignored"] = int(ego.abort_ignored)
    info = getattr(env, "start_info", None) or {}
    c["removed"], c["insert_ok"] = info.get("removed"), info.get("insert_ok")
    c["unsafe_start"] = info.get("unsafe", {}).get(rho)
    c["unsafe_start_lane"] = info.get("unsafe_lane", {}).get(rho)
    c["recycled"] = getattr(env, "recycled", 0)
    # H1 (experiments.h1): one start rule and one classification procedure for every arm
    c["rho_cls"] = h1.classification_rho(shield is not None, rho)
    c["correct_start"] = h1.correct_start(info) if "unsafe_lane" in info else None
    c["crash_category"] = None
    if frames is not None:
        frames.close()
        if c["crashed"]:
            trace = frames.trace()
            c["crash_category"] = h1.CR_UNC if trace is None else h1.classify_collision(trace, c["rho_cls"])
    c["h1_event"] = (h1.is_h1_event(c["correct_start"], c["crashed"], c["crash_category"])
                     if classify else None)
    # filter configuration of the episode: main_series is None without the filter
    c["shield_cfg"] = describe(shield.cfg) if shield is not None else "none"
    c["main_series"] = is_main_series(shield.cfg) if shield is not None else None
    return c
