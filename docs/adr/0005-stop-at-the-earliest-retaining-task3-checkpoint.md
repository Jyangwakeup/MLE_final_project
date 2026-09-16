# ADR 0005: Stop at the earliest Task 3 checkpoint that retains prior abilities

## Status

Accepted for the Task 3 retention experiment.

## Context

The first 500-round v5 child improved Task 3 score, kills, coins, and crates and
kept frozen suicide at zero, but retained only about 82% of its Task 2 coin and
crate ability. Diagnosis also found that the original transfer accidentally
kept the Task 1 teacher; correcting the teacher restored much of the lost
ability but did not meet the 90% gate.

## Decision

Task 3 always distils from the direct Task 2 parent policy. Before changing
Replay, seed 22 deterministically reproduces only the first 250 rounds of the
corrected 500-round run and evaluates that checkpoint on the same development
seeds. The earliest checkpoint that passes every capability, retention, safety,
and engineering gate is selected; longer training is not preferred by default.

If 250 rounds still fail prior-task retention, a separate experiment changes
only the 64-row Replay allocation from an approximately uniform parent pool to
16 Task 1, 32 Task 2, and 16 Task 3 rows. It checks 250 rounds before allowing a
single extension to 500. Training length and Replay allocation are never
changed in the same first comparison.

## Consequences

- A 500-round checkpoint with stronger Task 3 metrics may lose to an earlier
  checkpoint that satisfies the complete curriculum contract.
- Seeds 11 and 33 use the same fixed 250-round budget only after seed 22 passes.
- Development seeds 19200–19219 may guide this iteration; confirmation and main
  validation seeds remain untouched.
