# Fixed placement proof deadline: v8 validation in progress

World 24003 step 162 reproduces v7's rolling-horizon guarantee loss; retaining the original placement deadline admits a proven RIGHT. The separate world 24266 step 182 regression still rejects the dangerous DOWN under the opponent-rearming envelope.

The placement step is recorded only after actual BOMB selection, propagated through immutable history and pure transition projections, and cleared on capacity release or round reset. Pending proofs reject missing, expired or timer-inconsistent origins. The initial seven-transition obligation is unchanged; subsequent observations consume its remaining interval. No runtime timing or failure gate is relaxed.

Six targeted tests cover both historical failures, countdown and invisible flame intervals, malformed clocks, history serialization and the real training callback chain. The full suite passed 449 tests (one skipped); both historical whole-game regressions passed at source 703d59a; this is not safety admission or Task 4 qualification. Previously submitted ZIP and registered parent weights are unchanged.

Reproduce targeted checks with `python -m unittest tests.test_fixed_safety_deadline` from the repository root using the mle environment and CPU thread counts set to one.

The two regressions completed 800 actions and 85 placements. Independent per-step clock and fallback auditing found no issues; world 24003 P95/max were 40.740/58.764ms and world 24266 17.396/28.279ms. These reused historical worlds are regression evidence only. Next, compare v5 and v8 over the registered 60 engineering worlds for each of Tasks 1–3; stop the candidate on any safety failure, and require all existing retention and engineering gates.
