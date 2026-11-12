"""RSS-distance filter with delayed actuation for highway-env (verification package).

Modules:
    params   - constants of the protocol (section 7.3-7.4);
    rss      - RSS distance formulas (section 7.4);
    vehicle  - DelayedRSSVehicle: FIFO command delay + acceleration limiter inside act();
               ScriptedVehicle: scripted neighbours for control scenarios;
    shield   - RSSShield: safety mask, TAKEOVER, lane-change supervision hierarchy;
    accounting - decision set of the primary rates: policy / intercept steps, E, N_att, N_blocked;
    sim      - Scenario runner: decision every 2 frames, full per-frame CSV log;
    env      - integration with highway-env HighwayEnv (vehicle_class of DiscreteMetaAction).
"""
