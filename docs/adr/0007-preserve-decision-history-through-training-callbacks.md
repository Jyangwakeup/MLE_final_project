# Preserve decision history through training callbacks

Accepted. Training previously queried live history using the pre-BOMB state,
clearing the own-bomb obligation that act had just recorded. Each actual decision
now captures immutable before/after history, features and masks; transition
construction projects from that snapshot without changing live history. This
also prevents terminal callbacks from reconstructing old features using newer
history. Missing or mismatched decisions fail explicitly.

Snapshots are episode-local and cleared before round-boundary checkpoints, so
the v11 storage schema remains unchanged. A lifecycle marker prevents old Task 3
training continuation; explicit transfer from the registered Task 2 parents is
retained. Parent Replay remains an acknowledged historical limitation, not a
claim that old training transitions can be repaired without raw states.
