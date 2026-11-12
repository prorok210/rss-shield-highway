"""Regression test of the lane-change accounting of the runner (N_exec, exec_from, N_exec_other).

vehicle.py executes LANE_LEFT / LANE_RIGHT from target_lane_index when the command leaves the FIFO
(one decision step after it was issued), so in the edge lane the outward lane change issued in the
middle of an unfinished maneuver reverses it. The policy alternates the outward and the inward lane
change starting in the edge lane: every crossing must take the decision step whose command moved the
lateral target to the lane being entered, a reversal must cancel the maneuver, and no crossing may
stay without a record (N_exec_other = 0).
"""
import numpy as np
import pytest

from experiments.env import make_env
from experiments.runner import NAMES, run_episode
from rss_shield.params import ACTION_INDEX, DT, DT_DECISION, K_TAU

N_ALT = 30                        # decision steps of alternation, then IDLE


class Alternating:
    """Scripted policy (no learning): outward, inward, outward, ... for N_ALT steps, then IDLE."""

    def __init__(self, outward, inward):
        self.seq, self.k = (outward, inward), 0

    def q_values(self, obs):
        name = self.seq[self.k % 2] if self.k < N_ALT else "IDLE"
        self.k += 1
        q = np.zeros(len(NAMES))
        q[ACTION_INDEX[name]] = 1.0
        return q


def _edge_episode(edge, seed=3):
    """Baseline episode (no filter) on an empty road, the ego starting in lane `edge`."""
    env = make_env(D=10, warmup=2.0, duration=12)

    def reset():
        obs, info = env.reset(seed=seed)
        ego = env.vehicle
        env.road.vehicles[:] = [ego]          # only the lateral dynamics of the ego matter here
        lane_index = ego.lane_index[:2] + (edge,)
        lane = env.road.network.get_lane(lane_index)
        x = float(ego.position[0])
        ego.position = lane.position(x, 0.0)
        ego.heading = lane.heading_at(x)
        ego.lane_index = ego.target_lane_index = lane_index
        ego.lane = lane
        return obs, info

    agent = Alternating(*(("LANE_RIGHT", "LANE_LEFT") if edge else ("LANE_LEFT", "LANE_RIGHT")))
    journal = []
    c = run_episode(env, agent, shield_on=False, reset=reset, journal=journal)
    return c, journal


@pytest.mark.parametrize("edge", [3, 0])
def test_alternating_lane_changes_from_the_edge_lane(edge):
    # the command of decision step k leaves the FIFO on the first frame of step k + 1
    assert K_TAU == int(round(DT_DECISION / DT))
    c, J = _edge_episode(edge)
    lane, target, src = edge, edge, None
    reversals = crossings = 0
    for r in J:
        if r["target_lane"] != target:
            if r["target_lane"] == lane:      # back to the lane of the ego before the crossing
                reversals += 1
                src = None
            else:                             # a maneuver started by the command of step k - 1
                src = r["k"] - 1
                step = r["target_lane"] - lane
                assert J[src]["command"] == ("LANE_LEFT" if step < 0 else "LANE_RIGHT")
        if r["crossed"]:
            crossings += 1
            assert r["lane"] == r["target_lane"]
            assert src is not None and r["exec_from"] == src, (r["k"], r["exec_from"], src)
            src = None
        lane, target = r["lane"], r["target_lane"]
    assert reversals >= 1 and crossings >= 2      # the scenario exercises the reversal case
    assert c["N_exec"] == crossings and c["N_exec_other"] == 0
    assert not c["crashed"]
