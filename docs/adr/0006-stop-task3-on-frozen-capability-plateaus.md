# ADR 0006: Stop Task 3 on frozen capability plateaus

## Status

Accepted for the Task 3 plateau experiment.

## Context

Task 3 seed 33 retained resource and safety ability after 250 rounds but lost
four kills over twenty development games. Its exploratory training statistics
had peaked earlier, while training reward and loss did not identify which
frozen checkpoint best preserved the full curriculum. The existing rolling
reward stopper therefore observes the wrong signal, and the Task 1 score
stopper cannot express Task 3's multi-task constraints.

It is impossible to know that the next update will reduce frozen performance.
The recoverable decision is to preserve immutable checkpoints, observe a
decline or plateau afterwards, and select the best earlier checkpoint.

## Decision

Task 3 trains in immutable 50-round child runs and performs an
exploration-free Task 1/2/3 assessment after each child. A checkpoint is
qualified only when every existing capability, retention, safety, and
engineering gate passes.

The first qualified checkpoint starts a score anchor. An improvement of at
least 0.25 mean Task 3 points resets a two-assessment patience window. Smaller
changes may replace the selected checkpoint under the preregistered ranking,
but do not reset patience. Before qualification, a score improvement of 0.25
or fewer failed gates resets a three-assessment patience window. Training is
capped at 300 rounds.

Development, confirmation, and main-validation worlds remain separate. The
controller never uses training reward, loss, or exploratory episode score as a
stopping signal.

## Consequences

- Different training seeds may stop at different round counts under one rule.
- A decline is detected after it occurs, but the selected model rolls back to
  the best qualified checkpoint before that decline.
- A qualified model still improving at 300 rounds may enter confirmation as
  budget-truncated, never as plateau-converged.
- Full v11 learner snapshots remain unchanged; the stopping history belongs to
  the experiment controller and its immutable run lineage.
