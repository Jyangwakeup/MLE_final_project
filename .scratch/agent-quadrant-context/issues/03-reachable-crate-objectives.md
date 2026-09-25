# Add reachable crate objectives by Agent quadrant

Type: task
Status: resolved

Preserve the V10 density experiment and add a new contract whose quadrants are
recomputed around the Agent on every state.  Represent the share of all
reachable crates and the nearest reachable crate-frontier distance in each
quadrant.

## Answer

Implemented continuous-v11 with four reachable-crate shares and four normalized
nearest-frontier distances.  Axis-aligned crates split evenly between adjacent
Agent-centred quadrants.  Tests cover runtime parity, share normalization,
unreachable sentinels, and near-versus-far crate groups.
