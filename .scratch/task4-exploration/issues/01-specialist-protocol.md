# Specialist migration, controller and verification
Type: task
Status: resolved

Implement policy-only transfer, current-task Replay, effective gamma/LR, pure
score rewards, strict identity, screening and one final paired evaluation.
Run full unit tests and integration resume checks before background launch.

## Comments

World registration scanned 27,259 distinct JSON artifacts across all worktrees.

## Answer

Implemented isolated specialist contracts and controller. 493 full-suite tests
passed (one skipped), followed by 12 specialist tests after controller changes.
Real 2-round versus 1+1-round learner and action equality verified; frozen
engineering-world25800 evaluation passed. See registration verification.json.
