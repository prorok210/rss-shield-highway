"""Regression test for remark 1: a CSV journal must round-trip exactly.

The old counters_from_journal read flags through bool(), so the string "0" (every intercept
step) became True after a CSV write/read cycle.  The step counters also disagreed on
command=None.  This test builds the online counters, writes the same steps to CSV, reads them
with csv.DictReader and requires full equality, including a decision_tick=0 frame, an ABORT,
TAKEOVER, FALLBACK, a blocked lane change and a policy step without a command.
"""
import csv

import numpy as np
import pytest

from rss_shield.accounting import (COUNTERS, LANE_CHANGES, count_step, counters_from_journal,
                                   is_action_masked, new_counters)
from rss_shield.params import ABORT, ACTION_INDEX, FALLBACK, TAKEOVER

ALLOW = [1, 1, 1, 1, 1]
BLOCK_RIGHT = [1, 1, 0, 1, 1]


def _mask(mask):
    return "".join(str(int(x)) for x in mask)


def _row(*, decision_tick=1, policy_step=1, intervention=0, a_pi="IDLE", command="IDLE",
         mask=ALLOW, block_reason="", sup_cmd="", brake_mode=0):
    return dict(decision_tick=decision_tick, policy_step=policy_step,
                intervention=intervention, a_pi=a_pi, command=command, mask=_mask(mask),
                block_reason=block_reason, sup_cmd=sup_cmd, brake_mode=brake_mode)


def test_csv_roundtrip_agrees_with_count_step(tmp_path):
    steps = [
        # ordinary policy steps
        dict(a_pi="IDLE", command="IDLE", mask=ALLOW, why=None),
        dict(a_pi="LANE_RIGHT", command="LANE_RIGHT", mask=ALLOW, why=None),
        # blocked lane change (rss), then busy, then no_lane
        dict(a_pi="LANE_RIGHT", command="IDLE", mask=BLOCK_RIGHT, why={"rear": {}}),
        dict(a_pi="LANE_LEFT", command="IDLE", mask=[0, 1, 1, 1, 1], why={"busy": True}),
        dict(a_pi="LANE_LEFT", command="IDLE", mask=[0, 1, 1, 1, 1], why={"no_lane": True}),
        # policy step without a new command: not masked
        dict(a_pi="LANE_LEFT", command=None, mask=ALLOW, why=None),
        # intercept steps
        dict(a_pi="LANE_RIGHT", command=ABORT, mask=[0] * 5, why=None),
        dict(a_pi="IDLE", command=TAKEOVER, mask=[0] * 5, why=None),
        dict(a_pi="LANE_LEFT", command=FALLBACK, mask=[0] * 5, why=None),
        # a non-decision frame of the per-frame journal must be skipped
        dict(a_pi="LANE_LEFT", command="LANE_LEFT", mask=ALLOW, why=None, decision_tick=0),
    ]
    online = new_counters()
    rows = []
    for s in steps:
        row = _row(a_pi=s["a_pi"], command="" if s["command"] is None else s["command"],
                   intervention=int(s["command"] in (TAKEOVER, ABORT, FALLBACK)),
                   policy_step=int(s["command"] not in (TAKEOVER, ABORT, FALLBACK)),
                   mask=s["mask"], block_reason="" if s["command"] in (TAKEOVER, ABORT, FALLBACK)
                   else ("busy" if (s["why"] or {}).get("busy") else
                         "no_lane" if (s["why"] or {}).get("no_lane") else
                         "rss" if s["command"] != s["a_pi"] and s["a_pi"] in LANE_CHANGES else ""),
                   sup_cmd=s["command"] if s["command"] in (TAKEOVER, ABORT, FALLBACK) else "",
                   brake_mode=int(s["command"] in (TAKEOVER, FALLBACK)),
                   decision_tick=s.get("decision_tick", 1))
        if s.get("decision_tick", 1):
            count_step(online, s["a_pi"], s["command"], s["mask"], s["why"])
        rows.append(row)

    path = tmp_path / "journal.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(path, newline="") as fh:
        restored = counters_from_journal(list(csv.DictReader(fh)))
    assert restored == {k: online[k] for k in COUNTERS}
    assert restored["steps"] == 9
    assert restored["E"] == 6 and restored["intercept"] == 3
    assert restored["N_att"] == 5 and restored["N_att_int"] == 2
    assert restored["N_blocked"] == 3 and restored["N_busy"] == 1 and restored["N_no_lane"] == 1
    assert restored["abort"] == 1 and restored["takeover"] == 1 and restored["fallback"] == 1
    # LANE_RIGHT -> IDLE, LANE_LEFT -> IDLE twice, LANE_LEFT with no command is NOT masked
    assert restored["masked"] == 3


def test_bool_string_zero_is_false_and_invalid_flags_are_rejected():
    rows = [_row(policy_step="1", intervention="0", a_pi="IDLE", command="IDLE")]
    c = counters_from_journal(rows)
    assert c["E"] == 1 and c["intercept"] == 0
    bad = [_row(policy_step="0", intervention="0")]
    with pytest.raises(ValueError):
        counters_from_journal(bad)
    with pytest.raises(ValueError):
        counters_from_journal([_row(policy_step="2")])


def test_is_action_masked_requires_both_actions():
    assert is_action_masked(True, "LANE_LEFT", "IDLE")
    assert not is_action_masked(True, "LANE_LEFT", None)
    assert not is_action_masked(True, None, "IDLE")
    assert not is_action_masked(False, "LANE_LEFT", "IDLE")
    assert np.bool_(is_action_masked(np.bool_(True), "LANE_LEFT", "IDLE"))


def test_mask_parser_accepts_string_and_sequence():
    base = dict(decision_tick=1, policy_step=1, intervention=0, a_pi="LANE_RIGHT",
                command="IDLE", block_reason="rss")
    c1 = counters_from_journal([dict(base, mask="11011")])
    c2 = counters_from_journal([dict(base, mask=[1, 1, 0, 1, 1])])
    c3 = counters_from_journal([dict(base, mask="1,1,0,1,1")])
    assert c1["N_blocked"] == c2["N_blocked"] == c3["N_blocked"] == 1
    assert ACTION_INDEX["LANE_RIGHT"] == 2
