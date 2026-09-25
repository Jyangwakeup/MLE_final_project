# Implement agent-centred quadrant densities

Type: task
Status: resolved

Create a new feature and Rainbow-lite contract without modifying continuous-v7
or its checkpoints.  Add deterministic tests and a zero-extension migration
from the selected V7 checkpoint.

## Comments

- Eight inputs are ordered crate NW/NE/SW/SE then opponent NW/NE/SW/SE.

## Answer

Implemented `continuous-v10-agent-quadrant-density` as a 170-dimensional
extension of continuous-v7.  Added the V10 Rainbow runtime, registry and
experiment contracts, deterministic quadrant tests, and a zero-extension
migration.  The selected V7 Task 3 `c0400` policy was migrated with all eight
new first-layer columns initialized to zero.
