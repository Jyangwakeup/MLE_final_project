# Implement and train R20

Type: task
Status: resolved

Implement a versioned outcome-credit reward, verify that predicted crate and
opponent coverage do not positively reward BOMB, and start the V6 Task 4
training lineage from the Task 3 r370 checkpoint.

## Comments

Implemented `r20_outcome_credit` with zero predicted crate/opponent BOMB
bonuses. Reward and V6 tests pass. Warm-started V6 independently from the
Task 3 r370 snapshot, completed 100 Task 4 rounds as
`rainbow_lite_v6_r20_s11_task4_r370_c0100`, and completed the matching frozen
10-seed diagnostic evaluation.

## Answer

The first R20 checkpoint and its fixed-seed evaluation are complete. See the
specification and the run metadata for the frozen reward and lineage contracts.
