# Power of H1a and H1b (protocol 7.9)

Risk threshold delta = 3%, one-sided level alpha = 0.05. Target power: 0.8 for each hypothesis separately.

## H1a: 60 checks, planned true risk 1.0%

- target: 0.8
- nominal budget: 1000 episodes per check
- realized correct-start episodes per check (source: projection from the pilot start states runs/pilot_run/start_states.csv: D = 10: 10 of 10 correct, D = 25: 10 of 10 correct, D = 40: 20 of 20 correct): D = 10: 1000, D = 25: 1000, D = 40: 1000
- threshold from n_valid: D = 10: 20 events, D = 25: 20 events, D = 40: 20 events
- power of one check: D = 10: 0.9985, D = 25: 0.9985, D = 40: 0.9985
- **joint power, lower bound: 0.910** - sampling assumption: none about the dependence between checks (Boole's inequality; the checks share scenarios)
- naive independent approximation (product of the powers of the checks): 0.914 - assumes independent checks, which shared scenarios violate; not the joint power
- bound against the target: met
- simulation, shared-scenario model with intra-scenario correlation 0.0: 0.928 - sampling assumption: exchangeable scenario effect, episodes drawn only for correct starts
- simulation, shared-scenario model with intra-scenario correlation 0.2: 0.921 - sampling assumption: exchangeable scenario effect, episodes drawn only for correct starts

## H1b: 15 checks, planned true risk 5.0%

- target: 0.8
- nominal budget: 1500 episodes per check
- realized correct-start episodes per check (source: projection from the pilot start states runs/pilot_run/start_states.csv: D = 10: 10 of 10 correct, D = 25: 10 of 10 correct, D = 40: 20 of 20 correct): D = 10: 1500, D = 25: 1500, D = 40: 1500
- threshold from n_valid: D = 10: 57 events, D = 25: 57 events, D = 40: 57 events
- power of one check: D = 10: 0.9884, D = 25: 0.9884, D = 40: 0.9884
- **joint power, lower bound: 0.825** - sampling assumption: none about the dependence between checks (Boole's inequality; the checks share scenarios)
- naive independent approximation (product of the powers of the checks): 0.839 - assumes independent checks, which shared scenarios violate; not the joint power
- bound against the target: met
- simulation, shared-scenario model with intra-scenario correlation 0.0: 0.844 - sampling assumption: exchangeable scenario effect, episodes drawn only for correct starts
- simulation, shared-scenario model with intra-scenario correlation 0.2: 0.851 - sampling assumption: exchangeable scenario effect, episodes drawn only for correct starts

## Simultaneous confirmation of H1a and H1b (a separate condition)

- parameters: target 0.8, risks 1.0% (H1a) and 5.0% (H1b), the same n_valid as above
- lower bound without any dependence assumption: 1 - (1 - 0.9102) - (1 - 0.8253) = **0.7355** (does not meet 0.8)
- this condition is not part of the plan: the target of 0.8 applies to each hypothesis separately, and the sample sizes are not changed because of shared scenarios

