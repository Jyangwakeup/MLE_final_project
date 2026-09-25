# Measure safe crate-bomb opportunities

Type: research
Status: resolved

Add post-action, read-only diagnostics and summary metrics for useful and
horizon-survivable crate-hitting BOMB opportunities in frozen evaluations.

## Acceptance criteria

- Diagnostics do not change policy inputs, masks, chosen actions, or RNG.
- Unit tests cover useful, survivable-useful, and selected opportunity flags.
- A fresh frozen C400 evaluation reports the new aggregate metrics.

## Comments

- Existing C400 fixed evaluation: 2.08 crates/bomb, 2.6% zero-utility bombs,
  15.2 bombs/round, 22.2% WAIT actions, and 1.2 coins/round.

## Answer

The repeated frozen C400 evaluation exactly reproduced the existing outcomes.
Across 10 rounds, V7 observed 392 horizon-survivable, crate-hitting BOMB
opportunities and selected BOMB in 120 of them (30.6%). In the remaining
opportunities it selected WAIT 92 times and moved 180 times. Of those WAITs,
82 occurred with no reachable visible coin. Opportunity scarcity is therefore
not the primary bottleneck; the next experiment should expose the conjunction
of bomb survivability and crate utility more directly to the learner while
leaving the veto-only survival mask unchanged.
