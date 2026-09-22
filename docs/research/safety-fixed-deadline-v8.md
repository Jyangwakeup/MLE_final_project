# Fixed placement proof deadline: v8 validation in progress

World 24003 step 162 reproduces v7's rolling-horizon guarantee loss; retaining the original placement deadline admits a proven RIGHT. The separate world 24266 step 182 regression still rejects the dangerous DOWN under the opponent-rearming envelope.

The placement step is recorded only after actual BOMB selection, propagated through immutable history and pure transition projections, and cleared on capacity release or round reset. Pending proofs reject missing, expired or timer-inconsistent origins. The initial seven-transition obligation is unchanged; subsequent observations consume its remaining interval. No runtime timing or failure gate is relaxed.

Six targeted tests cover both historical failures, countdown and invisible flame intervals, malformed clocks, history serialization and the real training callback chain. The full suite passed 449 tests (one skipped); both historical whole-game regressions passed at source 703d59a; this is not safety admission or Task 4 qualification. Previously submitted ZIP and registered parent weights are unchanged.

Reproduce targeted checks with `python -m unittest tests.test_fixed_safety_deadline` from the repository root using the mle environment and CPU thread counts set to one.

The two regressions completed 800 actions and 85 placements. Independent per-step clock and fallback auditing found no issues; world 24003 P95/max were 40.740/58.764ms and world 24266 17.396/28.279ms. These reused historical worlds are regression evidence only. Next, compare v5 and v8 over the registered 60 engineering worlds for each of Tasks 1–3; stop the candidate on any safety failure, and require all existing retention and engineering gates.

## Completed paired retention at 5ff8d4c

All 360 games completed and all existing gates passed. Task 1 score and Task 2 coins/crates retained 100%; Task 3 score/coins/crates retained 96.53%/99.65%/99.56%. Task 3 score difference was -0.2667 (paired 95% bootstrap interval [-0.5833, 0.0000]), kills -0.0500 ([-0.1167, 0.0000]); this is a small observed combat regression within the retention gate, not evidence of an improvement.

The raw audit covered 5,398 candidate placements and 32,155 pending actions without clock/fallback/safety issues. Candidate Task 1/2/3 complete-act P95 was 8.455/7.252/20.812ms, maximum 11.466/151.887/59.335ms. Task 2/3 had zero self deaths and 100% bomb survival; all Task 3 games used bombs. Evidence and 10,000-sample paired bootstrap outputs are under `experiments/results/safety_fixed_deadline_v8_20260917/retention`.

Next is a frozen Task 4 engineering sweep on reused diagnostic worlds 24000–24059, against three rule-based agents. It is neither formal training nor independent qualification. Keep the original parent checkpoint and submitted archive unchanged.

## Completed strong-opponent engineering sweep at 0dc8dba

All 60 worlds completed and the registered gates passed: score 3.5333, coins 2.7833, crates 34.1167, kills 0.1500, zero self deaths, 100% bomb survival and no zero-bomb games. Raw audit covered 21734 actions, 2264 placements and 13481 pending actions without clock or safety engineering findings; complete-act P95/max were 73.092/354.178ms.

There were 9 opponent-bomb deaths outside own responsibility, all preserved separately. World 24025 step 189 exposed a specific movement protection gap: v8 admitted RIGHT while the same solver proved UP/LEFT/WAIT. Passing the registered own-responsibility engineering gates does not resolve this gap or establish Task 4 qualification. A separate v9 experiment attempts movement preference; no changes are made to this frozen run or the submitted archive.
