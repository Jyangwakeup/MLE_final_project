# ADR 0004: Use controllable survival for own-bomb safety

## Status

Accepted for the Task 3 safety experiment.

## Context

The 84-dimensional Double DQN with `r7_safe_credit_sparse` retained Task 1/2
ability, but some Task 3 seeds still entered a trap after their own bomb. The
v3 two-route check and v4 one-transition opponent check judged each move using
static continuations. They could therefore punish the last escape step while
allowing the earlier bomb placement that created the unavoidable trap.

A literal tree containing every future learner action, every opponent action,
and every official execution order exceeded the 400 ms internal budget. It
failed closed, but a timeout is not a completed safety proof and made full
rounds impractically slow.

## Decision

`survival-mask-v5` treats bomb safety as a finite feedback viability problem
over the complete own-bomb danger interval. The first simultaneous transition
is modeled exactly, including legal opponent moves, bombs, execution order,
world progression, crates, and active explosions. Later histories are
represented by a conservative union of opponent-reachable occupancy and bomb
danger, followed by backward reachability for the learner.

This abstraction may reject an action that would survive the actual opponent
choices, but it cannot approve an action merely because one favorable future
path exists. Search is fail-closed. Before placement, an unproved `BOMB` is
vetoed whenever a v1-safe non-bomb action exists. During the resulting own-bomb
obligation, an empty robust set falls back to v1 and records a robust guarantee
loss. The mask only removes actions; Q values retain their ordering among all
remaining actions.

The feature vector, network, reward, n-step length, opponents, and learning
parameters remain fixed. Thus the experiment isolates the safety boundary.

## Consequences

- Placement receives the safety credit that was previously delayed until the
  final escape steps.
- Frozen inference, exploration, greedy behavior, replay next masks, and
  Double DQN bootstrap share one mask contract.
- Conservative rejection can reduce bomb use and combat. Frozen admission
  therefore checks zero-bomb rounds and 90% resource/combat retention.
- Runtime is a correctness condition: P95 must remain below 250 ms, every
  action below 480 ms, and no robust search may time out.
- `training-resume-v11` is required for exact continuation; older snapshots
  remain frozen-evaluation inputs only.
