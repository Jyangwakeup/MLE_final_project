# Three-step score comparison with retained observed-kill fragments

## Decision

Use a separate `task4-score-improvement-v1` campaign from `312cc82d`.
L lowers learning rate to 3e-5; LP adds the existing coin potential weight .5;
LPK reserves 4,000 of 20,000 unique Replay slots for observed-kill fragments and
preferentially samples 16 of 64 from that pool. Every arm initializes from the
original Task3 policy, with new optimizer, Replay and learning streams.
Rules-only opponents and the existing v9 proof implementation remain fixed.

## Consequences

Two seed screens evaluate 20k/40k/60k action snapshots; the winner adds seed33
and a matching rules control. Archived C22/C11 snapshots receive the same new
checkpoint-selection opportunity. This supersedes fixed-endpoint selection only
for the new experiment; the old failed experiment is not reclassified.

Kill labels OR actual events across the n-step window, never duplicate reward,
and include simultaneous negative outcomes. Two disjoint FIFO pools change both
retention and sampling distribution. Empty reserved space reduces effective
occupancy; fallback sampling fills batches without duplication. This is not PER
or an unbiased estimate of the old uniform Replay objective.

A unique candidate is compared on 400 fresh worlds with the original parent
and its same-seed development-selected control. Both score differences must be
positive to report an observed improvement. Intervals spanning zero remain
uncertain. Safety failures stop the campaign; the 18/20/23/24-hour cutoffs do not
permit reduced sample counts. No opponent-pool updates, push or packaging occur.
