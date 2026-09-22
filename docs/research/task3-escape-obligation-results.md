# Task 3 own-bomb escape-obligation result

## Outcome

The experiment stopped at its first preregistered gate. On the seven known
seed-22 self-death worlds, `survival-mask-v3` reduced self-deaths from 7/7 to
1/7, but the gate required 0/7. The full 16000–16019 frozen comparison and all
long training, confirmation, and main-validation stages were therefore not
run.

The tested learner was unchanged: Double DQN, 84-dimensional
`continuous-v2`, `r7_safe_credit_sparse`, and the seed-22 1500-round Task 3
checkpoint from the `814173b` lineage. Only the frozen inference safety mask
changed.

| Metric (7 fixed worlds) | v1 | v3 |
|---|---:|---:|
| Self-deaths | 7 (100%) | 1 (14.3%) |
| Mean game score | 2.29 | 3.71 |
| Mean coins | 2.29 | 3.00 |
| Mean crates | 41.57 | 33.14 |
| Mean kills | 0.00 | 0.14 |
| Bomb survival | 93.75% | 99.43% |
| Avoidable escape collapses | 7 | 1 |
| Invalid-action rate | 0.32% | 0.07% |
| Worst per-run act P95 | 6.82 ms | 15.62 ms |
| Maximum act latency | 15.55 ms | 27.97 ms |

These seven worlds were selected because v1 died in them, so they are a
targeted regression set rather than an unbiased estimate of general Task 3
performance. The score and resource changes above are diagnostic only.

## Remaining failure

Seed 12017 survived five of six own bombs and died to the sixth at step 289.
At step 285 the hypothetical bomb graph reported two internally
node-disjoint escape routes and allowed `BOMB`. After the environment advanced,
the first post-bomb state at step 286 had no H=7 survivable action, forcing the
v1-to-physical fallback. A one-route v1 recovery briefly appeared at step 287;
the agent then returned to no-safe-action fallback and its own bomb exploded.

The evidence isolates a modeling boundary: v3 treats opponents at their
current positions as obstacles but does not model their next movement. Two
routes in that static graph therefore do not guarantee two usable routes after
the simultaneous environment transition. The trace establishes that the
predicted guarantee was invalidated; it does not yet prove which opponent move
or board update was individually causal.

There is also a diagnostic attribution defect. The avoidable-collapse metric
was first raised at step 288, not at the initial step-286 fallback, because the
pre-bomb safe-alternative fact is not retained across the bomb obligation.
This does not change the action, death, or failed gate, but it should be fixed
before this diagnostic is used as a training-sample selector.

## Decision

Per the preregistration, the route threshold is not tuned after seeing these
results. No Task 3 training or safety-replay fallback is started: the fallback
was permitted only after frozen admission passed and a trained model failed a
safety gate. No Task 3 winner is published, and the final-test seeds remain
unused.

Machine-readable metrics and artifact hashes are in
`experiments/task3_escape_obligation_results.json`. Full replays and run
snapshots remain under `runs/` and are not tracked by Git.
