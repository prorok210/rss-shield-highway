"""Decision set of the primary rates (protocol 7.7), defined once.

Used by the journal of the verification package (rss_shield.sim) and by the counters of the
experiments (experiments.runner); the analysis reads the same definition.

A decision step is the tick t_k together with the interval [t_k, t_k+1) it governs.
  intercept step ("intervention")  the command issued at t_k is supervisory - TAKEOVER, ABORT or
                                   FALLBACK (the safe set was empty);
  policy step                      every other step: the command is a policy action that passed
                                   the mask (or no new command was issued).
Exactly one of the two holds on every step and on every physics frame of it. The type is set by
the tick, not by the command that is active during the step: a tick at which the policy decides
while an ABORT or a TAKEOVER is still active is a policy step.

  E          number of policy steps - the exposure, denominator of every rate;
  N_att      policy steps whose unmasked greedy action a_pi is a lane change (attempts);
  N_blocked  those of them with a_pi outside A_safe; N_busy / N_no_lane - of them, blocked because
             a maneuver is in progress or queued / because there is no lane on that side;
  N_att_int  intercept steps whose a_pi is a lane change. They are not attempts: not in N_att,
             N_blocked, BLCR or E.
Requests made while a maneuver is being completed on an intercept segment follow the same rule
and nothing else: as long as the supervisor decides (TAKEOVER that completes the maneuver, ABORT,
FALLBACK) the request goes to N_att_int; from the first tick with a non-empty safe set the step is
a policy step, the request is an attempt (N_att) and, because the maneuver is still in progress,
a blocked one (N_blocked, N_busy) - also while the supervisory command is still active. No request
is counted twice and none is counted on a step outside E.
Executed lane changes: N_exec counts marking crossings by the policy step whose command moved the
lateral target (experiments.runner); N_exec_int - those of them that happened while a supervisory
command was active; they stay in N_exec.
"""
from .params import ABORT, ACTION_INDEX, FALLBACK, SUPERVISORY, TAKEOVER

POLICY, INTERVENTION = "policy", "intervention"
LANE_CHANGES = ("LANE_LEFT", "LANE_RIGHT")
SUP_KEY = {TAKEOVER: "takeover", ABORT: "abort", FALLBACK: "fallback"}
COUNTERS = ("steps", "E", "intercept", "takeover", "abort", "fallback", "N_att", "N_blocked",
            "N_busy", "N_no_lane", "N_att_int", "masked")


def step_kind(command):
    """Type of the decision step whose tick issued `command` (None: no new command)."""
    return INTERVENTION if command in SUPERVISORY else POLICY


def block_reason(a_pi, mask, why=None):
    """"" if the lane change a_pi is admissible, else "busy" / "no_lane" / "rss".
    mask: sequence of 5 flags (or a "01011" string); why: violations of a_pi from the shield."""
    if a_pi not in LANE_CHANGES or mask is None or int(mask[ACTION_INDEX[a_pi]]):
        return ""
    why = why or {}
    return "busy" if why.get("busy") else "no_lane" if why.get("no_lane") else "rss"


def new_counters():
    return {k: 0 for k in COUNTERS}


def count_step(c, a_pi, command, mask=None, why=None):
    """Add one decision step to the counters c (new_counters()). Returns its type."""
    kind = step_kind(command)
    c["steps"] += 1
    if kind == POLICY:
        c["E"] += 1
        if a_pi in LANE_CHANGES:
            c["N_att"] += 1
            reason = block_reason(a_pi, mask, why)
            c["N_blocked"] += int(bool(reason))
            c["N_busy"] += int(reason == "busy")
            c["N_no_lane"] += int(reason == "no_lane")
        c["masked"] += int(command is not None and command != a_pi)
    else:
        c["intercept"] += 1
        c[SUP_KEY[command]] += 1
        c["N_att_int"] += int(a_pi in LANE_CHANGES)
    return kind


def counters_from_journal(rows):
    """Recount from journal rows (a_pi, command, mask, block_reason): the counters must be
    recoverable from the policy_step / intervention flags of the journal alone."""
    c = new_counters()
    for r in rows:
        policy, lc = bool(r["policy_step"]), r["a_pi"] in LANE_CHANGES
        c["steps"] += 1
        c["E"] += int(policy)
        c["intercept"] += int(not policy)
        c["N_att"] += int(policy and lc)
        c["N_att_int"] += int(not policy and lc)
        if policy:
            reason = r.get("block_reason", "")
            c["N_blocked"] += int(lc and bool(reason))
            c["N_busy"] += int(lc and reason == "busy")
            c["N_no_lane"] += int(lc and reason == "no_lane")
            c["masked"] += int(r["command"] != r["a_pi"])
        else:
            c[SUP_KEY[r["command"]]] += 1
    return c
