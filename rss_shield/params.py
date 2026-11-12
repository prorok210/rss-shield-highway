"""Protocol constants (v6, sections 7.3-7.4). Change here only together with the protocol text."""
import numpy as np

DT = 0.1                 # physics step, s (simulation_frequency = 10 Hz)
DT_DECISION = 0.2        # decision period, s (policy_frequency = 5 Hz)
FRAMES_PER_DECISION = int(round(DT_DECISION / DT))   # = 2
TAU = 0.2                # actuator delay, s
K_TAU = int(round(TAU / DT))                          # = 2 physics frames in FIFO
TAU_EFF = TAU + DT_DECISION                           # = 0.4 s

A_MAX_ACCEL = 3.0        # ego acceleration upper bound, m/s^2
A_MIN_BRAKE = 4.0        # guaranteed braking in takeover, m/s^2
A_MAX_BRAKE = 8.0        # never brake harder (assumption of the vehicle behind ego), m/s^2

A_LAT_MAX_ACCEL = 1.5    # RSS lateral assumption, m/s^2 (not respected by highway-env traffic)
A_LAT_MIN_BRAKE = 3.0    # m/s^2
MU_LAT = 0.5             # lateral fluctuation margin, m

TARGET_SPEEDS = np.arange(0.0, 30.0 + 1e-9, 5.0)      # {0, 5, ..., 30} m/s
DELTA_V = 5.0

LENGTH = 5.0             # highway-env Vehicle.LENGTH
WIDTH = 2.0              # highway-env Vehicle.WIDTH
LANE_WIDTH = 4.0
V_MAX = 40.0             # highway-env Vehicle.MAX_SPEED: upper bound of every vehicle speed, m/s (zone, 7.4)

ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTION_INDEX = {v: k for k, v in ACTIONS.items()}

# Supervisory commands (not policy actions); they pass through the same FIFO as meta-actions.
TAKEOVER = "TAKEOVER"    # brake in [-A_MAX_BRAKE, -A_MIN_BRAKE], lateral target unchanged
ABORT = "ABORT"          # return target_lane_index to the maneuver origin lane
FALLBACK = "FALLBACK"    # brake in [-A_MAX_BRAKE, -A_MIN_BRAKE] and hold the lane containing ego centre
SUPERVISORY = (TAKEOVER, ABORT, FALLBACK)

MANEUVER_DONE_TOL = 0.1  # |y - y_target| < 0.1 m -> maneuver completed
