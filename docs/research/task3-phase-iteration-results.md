# Task 3 phase-reward iteration results

## Decision

The phase-aware Task 3 iteration failed its preregistered cross-seed gates. No
Task 3 candidate was created and Task 4 was not started.

The best failed configuration was `continuous-phase-v1` with
`r7_safe_credit_sparse` and `survival-mask-v2`. It improved seed 11 combat and
reduced the worst suicide rate relative to the same v1-mask arm, but it did not
meet the 5% safety gate and severely reduced Task 2 resource retention.

## Round 1

All eight seed/arm runs completed 500 rounds on source commit `125124f` and
were evaluated on seeds 13000–13019. Every arm had at least one training seed
above the 10% suicide elimination threshold.

| Arm | Seed | Task 3 score gain | Kill gain | Suicide | Task 2 coin retention | Task 2 crate retention |
|---|---:|---:|---:|---:|---:|---:|
| phase + r7 | 11 | +0.75 | −0.05 | 20% | 85% | 83% |
| phase + r7 | 22 | −0.45 | −0.10 | 20% | 104% | 103% |
| phase + r9 resource | 11 | +0.95 | +0.25 | 40% | 85% | 85% |
| phase + r9 resource | 22 | −1.15 | −0.05 | 10% | 76% | 80% |
| phase + r9 combat | 11 | +0.95 | +0.05 | 15% | 94% | 93% |
| phase + r9 combat | 22 | −1.35 | −0.15 | 25% | 89% | 91% |
| phase + r9 full | 11 | +0.90 | +0.05 | 20% | 81% | 80% |
| phase + r9 full | 22 | −1.75 | −0.10 | 15% | 87% | 88% |

The frozen ranking selected phase+r7 for the conditional mask-v2 control. The
r9 variants did not provide cross-seed evidence that phase-weighted event
rewards or potentials improved the official game objective.

## Round 2

Both mask-v2 controls were rebuilt from their own Task 2 parent rather than
continued from a failed Task 3 child.

| Seed | Task 3 score gain | Kill gain | Suicide | Bomb survival | Task 2 coin retention | Task 2 crate retention |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | +1.85 | +0.15 | 10% | 99.8% | 60% | 58% |
| 22 | −0.75 | −0.20 | 15% | 99.4% | 67% | 75% |

Mask-v2 reduced suicide frequency, but “survived bomb” and “did not die in the
round” remain different quantities: an agent can survive most individual bomb
resolutions and still eventually die to its own bomb. The results therefore do
not justify treating bomb survival as a substitute for the episode suicide
gate.

## Stopping rationale

The opponent curriculum was preregistered only for a model that passed safety,
resource, and retention gates and failed combat alone. No model met that
condition. Round 3 required two independently successful Round 2
interventions; mask-v2 did not fully solve safety and no curriculum control was
eligible. Continuing would have changed the experiment after seeing results.

Machine-readable per-seed evidence is in
`experiments/task3_phase_evaluations.csv`; gate decisions, hashes, and training
run identities are in `experiments/task3_phase_results.json`. Confirmation,
main-validation, and final-test seed ranges were not used.
