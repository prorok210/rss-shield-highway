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
import math

import numpy as np

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


def is_action_masked(policy_step, a_pi, command):
    """A policy step is masked iff the policy asked for a lane change and the executed command
    differs from it.  A missing request or a missing command is not a masking: the same rule is
    used by count_step and by counters_from_journal (protocol 7.7, R_int)."""
    return bool(policy_step and a_pi is not None and command is not None and command != a_pi)


def _parse_bool(val, field_name):
    """Strict boolean reader for journal fields.  Python's bool("0") is True, so a CSV journal
    must never be read through bool(); only the strings / numbers listed here are accepted."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        if int(val) in (0, 1):
            return bool(int(val))
        raise ValueError(f"invalid integer boolean for {field_name}: {val!r}")
    if isinstance(val, (float, np.floating)):
        fv = float(val)
        if math.isfinite(fv) and fv in (0.0, 1.0):
            return bool(int(fv))
        raise ValueError(f"invalid float boolean for {field_name}: {val!r}")
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes"):
            return True
        if s in ("0", "false", "no"):
            return False
        raise ValueError(f"invalid string boolean for {field_name}: {val!r}")
    raise ValueError(f"cannot parse boolean for {field_name}: {val!r}")


def _parse_optional_str(val):
    if val is None:
        return None
    s = str(val).strip()
    return None if s == "" or s.lower() in ("none", "null") else s


def _parse_mask(val):
    """5-bit mask from a CSV cell: "01011", "[0, 1, 0, 1, 1]" or a sequence."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() == "none":
            return None
        if len(s) == 5 and set(s).issubset({"0", "1"}):
            return tuple(ch == "1" for ch in s)
        parts = [p.strip() for p in s.strip("[]()").split(",") if p.strip() != ""]
        if len(parts) == 5:
            return tuple(_parse_bool(p, "mask") for p in parts)
        raise ValueError(f"cannot parse 5-bit mask: {val!r}")
    seq = tuple(_parse_bool(x, "mask") for x in val)
    if len(seq) != 5:
        raise ValueError(f"mask must have length 5, got {len(seq)}")
    return seq


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
        c["masked"] += int(is_action_masked(True, a_pi, command))
    else:
        c["intercept"] += 1
        c[SUP_KEY[command]] += 1
        c["N_att_int"] += int(a_pi in LANE_CHANGES)
    return kind


def _journal_command(row, policy):
    """Executed / supervisory command of a journal row, from either schema:
    the runner writes `command`; the per-frame journal of rss_shield.sim writes `a_exec`
    (policy steps) and `sup_cmd` (intercept steps)."""
    if "command" in row:
        return _parse_optional_str(row.get("command"))
    if policy:
        return _parse_optional_str(row.get("a_exec"))
    return _parse_optional_str(row.get("sup_cmd"))


def counters_from_journal(rows):
    """Recount decision steps from journal rows.  Every flag is parsed strictly (no implicit
    bool() on CSV strings); frames with decision_tick = 0 are skipped, and the step is delegated
    to count_step, so online counters and a CSV round-trip agree by construction."""
    c = new_counters()
    for r in rows:
        if "decision_tick" in r and not _parse_bool(r["decision_tick"], "decision_tick"):
            continue
        policy = _parse_bool(r["policy_step"], "policy_step")
        intervention = (_parse_bool(r["intervention"], "intervention")
                        if "intervention" in r else not policy)
        if policy == intervention:
            raise ValueError("exactly one of policy_step and intervention must be True")
        a_pi = _parse_optional_str(r.get("a_pi"))
        command = _journal_command(r, policy)
        mask = _parse_mask(r.get("mask"))
        reason = _parse_optional_str(r.get("block_reason"))
        if reason is None and not policy:
            reason = _parse_optional_str(r.get("reason"))
        if mask is None and reason and a_pi in LANE_CHANGES:
            # a journal may store the block reason without the mask; make the target action
            # blocked so block_reason() returns exactly the stored reason
            mask = tuple(0 if n == a_pi else 1 for n in ("LANE_LEFT", "IDLE", "LANE_RIGHT",
                                                         "FASTER", "SLOWER"))
        count_step(c, a_pi, command, mask, {"busy": reason == "busy",
                                            "no_lane": reason == "no_lane"} if reason else None)
    return c
