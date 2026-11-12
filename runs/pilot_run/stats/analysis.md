# Statistical plan on pilot data (protocol 7.9)

Input: `runs/pilot_run/stats/episodes.csv`, 168 episodes. Parametric bootstrap: 199 replicates (the plan requires at least 1000 for the main series); two-way cluster bootstrap: 999; H1 difference: 2000. Pilot policies are not trained: the numbers show that every step of the plan is computed, not that H1-H3 hold.

## H1: checks on episodes with a correct start

Start rule and classification rho: `experiments/h1.py` (correct start at rho = 1.0 s for every arm; Baseline 2 classified at the reference rho = 1.0 s). `rho_cls` is the rho used for the categories.

| arm | D | rho | unit | n | n_valid | CR_total | events CR_ego_op | rho_cls | upper 95 % | lower 95 % | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 10.0000 | 0.4 | A:10:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 10.0000 | 0.4 | A:10:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 10.0000 | 0.6 | A:10:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 10.0000 | 0.6 | A:10:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 10.0000 | 1.0 | A:10:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| A | 10.0000 | 1.0 | A:10:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 0.4 | A:25:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 0.4 | A:25:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 0.6 | A:25:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 0.6 | A:25:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 1.0 | A:25:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| A | 25.0000 | 1.0 | A:25:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 0.4 | A:40:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 0.4 | A:40:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 0.6 | A:40:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 0.6 | A:40:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 1.0 | A:40:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| A | 40.0000 | 1.0 | A:40:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 0.4 | B:10:0.4:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 0.4 | B:10:0.4:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 0.6 | B:10:0.6:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 0.6 | B:10:0.6:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 1.0 | B:10:1.0:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 10.0000 | 1.0 | B:10:1.0:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 0.4 | B:25:0.4:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 0.4 | B:25:0.4:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 0.6 | B:25:0.6:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 0.6 | B:25:0.6:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 1.0 | B:25:1.0:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 25.0000 | 1.0 | B:25:1.0:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 0.4 | B:40:0.4:0 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 0.4 | B:40:0.4:1 | 4 | 4 | 0.0000 | 0 | 0.4 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 0.6 | B:40:0.6:0 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 0.6 | B:40:0.6:1 | 4 | 4 | 0.0000 | 0 | 0.6 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 1.0 | B:40:1.0:0 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B | 40.0000 | 1.0 | B:40:1.0:1 | 4 | 4 | 0.0000 | 0 | 1.0 | 0.5271 | 0.0000 | insufficient |
| B2 | 10.0000 | — | B2:10:0 | 4 | 4 | 1.0000 | 4 | 1.0 | 1.0000 | 0.4729 | exceeded |
| B2 | 10.0000 | — | B2:10:1 | 4 | 4 | 1.0000 | 4 | 1.0 | 1.0000 | 0.4729 | exceeded |
| B2 | 25.0000 | — | B2:25:0 | 4 | 4 | 1.0000 | 4 | 1.0 | 1.0000 | 0.4729 | exceeded |
| B2 | 25.0000 | — | B2:25:1 | 4 | 4 | 1.0000 | 4 | 1.0 | 1.0000 | 0.4729 | exceeded |
| B2 | 40.0000 | — | B2:40:0 | 4 | 4 | 1.0000 | 4 | 1.0 | 1.0000 | 0.4729 | exceeded |
| B2 | 40.0000 | — | B2:40:1 | 4 | 4 | 0.7500 | 3 | 1.0 | 0.9873 | 0.2486 | exceeded |

H1a (IUT over the 24 checks with rho in {0.6; 1.0}): **not confirmed** (0 confirmed, 0 exceeded). H1b (IUT over the 6 Baseline 2 checks): **confirmed** (6 exceeded, 0 confirmed). Checks with rho = 0.4 are the boundary level: descriptive, outside the decision.

Descriptive difference CR_ego_op(Proposed, mode B) − CR_ego_op(Baseline 2), two-level cluster bootstrap:

| D | rho | difference | lower 95 % | upper 95 % | policies (Proposed, Baseline 2) |
|---|---|---|---|---|---|
| 10.0000 | 0.6000 | -1.0000 | -1.0000 | -1.0000 | 2, 2 |
| 10.0000 | 1.0000 | -1.0000 | -1.0000 | -1.0000 | 2, 2 |
| 25.0000 | 0.6000 | -1.0000 | -1.0000 | -1.0000 | 2, 2 |
| 25.0000 | 1.0000 | -1.0000 | -1.0000 | -1.0000 | 2, 2 |
| 40.0000 | 0.6000 | -0.8750 | -1.0000 | -0.5000 | 2, 2 |
| 40.0000 | 1.0000 | -0.8750 | -1.0000 | -0.5000 | 2, 2 |

## H2 / H3: seven primary checks

Covariates: Dc = (D − 25) / 10, rhoc = rho − 2/3 (grid means), mode = 1 for mode B. One-sided p: parametric bootstrap of the signed likelihood root under the null model; Holm over the seven checks in each analysis; sensitivity: two-way cluster bootstrap that keeps shared scenario ids.

| check | function | model | estimate | SE | signed root | p (bootstrap) | p Holm | converged | sens. estimate | sens. 95 % interval | sens. p | sens. p Holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2: v_eff ~ rho | `h2_veff_rho` | M1 LMM | -1.6751 | 4.9762 | -0.3361 | 0.3800 | 1.0000 | yes | -1.6751 | [-11.1639; 7.8986] | 0.3320 | 1.0000 | not established (primary analysis) |
| H2: v_eff ~ D | `h2_veff_D` | M1 LMM | -3.6376 | 1.0135 | -3.1171 | 0.0050 | 0.0350 | yes | -3.6376 | [-5.7231; -1.3280] | 0.0010 | 0.0070 | established |
| H2: v_eff ~ D x rho | `h2_veff_Dxrho` | M1 LMM | -0.9502 | 4.0630 | -0.2337 | 0.4100 | 1.0000 | yes | -0.9502 | [-10.8560; 8.9446] | 0.3600 | 1.0000 | not established (primary analysis) |
| H2: N_exec ~ rho | `h2_nexec_rho` | M2 NB2-GLMM | -0.9782 | 0.8307 | -1.2281 | 0.1354 | 0.8125 | yes | -0.4468 | [-1.7182; 0.9050] | 0.2480 | 1.0000 | not established (primary analysis) |
| H2: N_exec ~ D | `h2_nexec_D` | M2 NB2-GLMM | 0.3436 | 0.1558 | 2.2624 | 0.9898 | 1.0000 | yes | 0.4216 | [0.1023; 0.7531] | 0.9900 | 1.0000 | not established (direction) |
| H3a: N_blocked ~ mode | `h3a_nblocked_mode` | M3 NB2-GLMM | 0.1555 | 0.0824 | 1.8155 | 0.9447 | 1.0000 | yes | 0.1552 | [-0.0045; 0.3187] | 0.9690 | 1.0000 | not established (direction) |
| H3b: P(block ∣ attempt) ~ mode | `h3b_pblock_mode` | M4 beta-binomial GLMM | 0.4640 | 0.2464 | 1.8102 | 0.9548 | 1.0000 | yes | 0.5047 | [-0.3551; 1.1873] | 0.8990 | 1.0000 | not established (direction) |

Control (outside the family) `control_natt_mode`, M5: estimate 0.1482, SE 0.0804, one-sided bootstrap p 0.9497, converged: True.

Excluded episodes: E = 0: 0; N_att = 0 (only from M4): 0 of 144.

Pearson dispersion without random effect and overdispersion (Poisson / binomial): N_exec 0.85, N_blocked 2.87, N_att 2.76, blocked_share 0.77.

## Descriptive contrasts and monotonicity (secondary)

Contrast of differences [v_eff(40, 1.0) − v_eff(40, 0.4)] − [v_eff(10, 1.0) − v_eff(10, 0.4)]: -0.0306 [-18.7144; 18.0883] m/s.

| outcome | adjacent levels | difference | 95 % interval |
|---|---|---|---|
| v_eff | rho 0.4 -> 0.6 | -1.0701 | [-6.1500; 3.7242] |
| v_eff | rho 0.6 -> 1 | -0.0820 | [-5.1355; 4.5656] |
| v_eff | D 10 -> 25 | -9.7478 | [-15.9811; -3.5786] |
| v_eff | D 25 -> 40 | -1.1650 | [-2.4257; 0.3885] |
| N_exec | rho 0.4 -> 0.6 | -0.0002 | [-0.0021; 0.0017] |
| N_exec | rho 0.6 -> 1 | -0.0004 | [-0.0019; 0.0010] |
| N_exec | D 10 -> 25 | 0.0033 | [0.0002; 0.0061] |
| N_exec | D 25 -> 40 | -0.0013 | [-0.0040; 0.0019] |

| Page's test (decreasing) | levels | blocks | L | p | method |
|---|---|---|---|---|---|
| v_eff over rho | 0.4/0.6/1 | 6 | 77.0000 | 0.0996 | exact |
| v_eff over D | 10/25/40 | 6 | 80.0000 | 0.0131 | exact |
| N_exec over rho | 0.4/0.6/1 | 6 | 77.0000 | 0.0996 | exact |
| N_exec over D | 10/25/40 | 6 | 67.0000 | 0.9428 | exact |
