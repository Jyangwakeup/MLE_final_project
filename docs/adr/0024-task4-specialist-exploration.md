# Task4 specialist exploration with policy-only initialization

## Decision

A separate `task4-exploration-v1` protocol abandons old-task retention and selects
by Task4 mean official score, then first-place rate. The historical curriculum
protocols and their failed outcomes remain unchanged. E1 copies only the fixed
Task3 policy, synchronizes target, and resets Adam and all Replay. E2 changes
only rewards to coin +1 / opponent kill +5; E3 changes only gamma to .99.

## Consequences

This trades curriculum retention for unrestricted Task4 adaptation. E1 changes
several initialization and learning settings together, so its effect cannot be
attributed solely to removing retention. Explicit Task4-only sampling rejects
foreign partitions; no parent Replay is inherited. Versioned effective learning
parameters govern both n-step accumulation and bootstrap, frozen evaluation and
strict resume. Safety v9, features, network and game rules remain unchanged.

All three arms and seeds 22/11 are screened on 100 development worlds. The winning
arm adds seed33, then one candidate receives a single 200-world paired final
assessment. An observed positive mean is not curriculum qualification; intervals
crossing zero remain uncertain. Engineering failure or unexplained self-death
stops the experiment. Checkpoint behavior failures only eliminate that checkpoint.
The 18/20/23/24-hour deadlines do not authorize reduced screening requirements.
