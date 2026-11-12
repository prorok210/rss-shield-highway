"""Independent reference of the action mask for commands waiting in the FIFO (protocol 7.2).

The reference does not use RSSShield.mask() / plan_state(). For every action it lets a copy of the
real vehicle (FIFO, controllers, limiter) run and reads where the command really leads:
  * state at the activation moment: the copy runs k_tau frames without a new command;
  * a lane change is admissible only if no maneuver is in progress at that moment, the command
    really moves the lateral target, and the predictive criterion of the filter holds for every
    lane between the lane of the ego centre and the target the command really produces;
  * any other action is checked with the lanes the maneuver in progress still enters (lane-change
    horizon) or, in lane keeping, on the two decision periods.
The criterion itself is RSSShield.check_plan() of a shield without the neighbour zone
(filter_on = False): an unlimited check.
"""
import numpy as np

from rss_shield.params import ACTIONS, DT, FRAMES_PER_DECISION
from rss_shield.shield import RSSShield, lc_frames_for

LANE_CHANGES = ("LANE_LEFT", "LANE_RIGHT")


def _run(ego, n_frames, cmd=None):
    g = ego.ghost()
    if cmd is not None:
        g.act(cmd)
    for _ in range(n_frames):
        g.act()
        g.step(DT)
    return g


def _between(a, b):
    """Lanes after a up to and including b; {b} if a == b."""
    if a == b:
        return {b}
    step = 1 if b > a else -1
    return set(range(a + step, b + step, step))


def unlimited(ego, cfg):
    sh = RSSShield(ego, cfg)
    sh.filter_on = False
    return sh


def command_violations(ego, cfg, action):
    """Violations of the predictive criterion for the lanes the lane change `action` really enters
    when it is issued now (after everything already waiting in the FIFO)."""
    after_queue = _run(ego, ego.k_tau)
    final = _run(ego, ego.k_tau + 1, action).target_lane_index[2]
    lanes = _between(after_queue.lane_index[2], final)
    return unlimited(ego, cfg).check_plan([action], lc_frames_for(cfg.lc_horizon), lanes)


def reference_mask(ego, cfg):
    sh = unlimited(ego, cfg)
    n_lc, n_keep = lc_frames_for(cfg.lc_horizon), 2 * FRAMES_PER_DECISION
    after_queue = _run(ego, ego.k_tau)
    man = after_queue.maneuver
    mask = np.zeros(len(ACTIONS), dtype=int)
    for idx, action in ACTIONS.items():
        if action in LANE_CHANGES:
            final = _run(ego, ego.k_tau + 1, action).target_lane_index[2]
            if man is not None or final == after_queue.target_lane_index[2]:
                continue
            lanes = _between(after_queue.lane_index[2], final)
            mask[idx] = int(not sh.check_plan([action], n_lc, lanes))
        elif man is not None:
            lanes = _between(man["origin"][2], man["target"][2])
            mask[idx] = int(not sh.check_plan([action], n_lc, lanes))
        else:
            mask[idx] = int(not sh.check_plan([action], n_keep, set()))
    return mask


def lanes_after_queue(ego):
    """Lanes the maneuver in progress at the activation moment still enters (empty: none)."""
    man = _run(ego, ego.k_tau).maneuver
    return set() if man is None else _between(man["origin"][2], man["target"][2])
