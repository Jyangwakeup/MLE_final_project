# Train V10 with the frozen R21 reward

Type: task
Status: resolved
Blocked by: 01

Train the zero-extended V10 policy for 50 Task 3 rounds with the unchanged R21
reward.  Evaluate on the same ten development seeds as the V7 `c0400` parent
and report whether each new quadrant input acquired non-zero first-layer weight.

## Comments

- Existing action features already encode progress toward the nearest crate
  frontier and toward/away from opponents; quadrant density supplies regional
  context without adding a rule-selected action.

## Answer

All eight zero-initialized input columns acquired non-zero weights after 50
Task 3 rounds, confirming that the learner consumed the new information.
However, the frozen 10-seed screen regressed from the V7 parent: mean score
7.4 to 5.7, mean kills 0.7 to 0.5, first-place rate 70% to 50%, and suicide
rate 10% to 20%.  Preserve V7 `c0400`; the V10 checkpoint is diagnostic only
and is not promoted.
