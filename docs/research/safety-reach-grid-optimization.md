# Equivalent opponent reach grid optimization

The terminal v9 regression at world 24025 step 75 exceeded the unchanged 400ms budget. Its complete shared search required 452–475ms in three offline runs; profiling found abstract opponent reach expansion dominated the cost.

The optimization replaces per-name Python set expansion with two boolean grid frontiers. Union distributes over the common blocked-cell transition, so this preserves the original opponent occupancy union and armed occupancy union at every future step. Initialization retains name deduplication semantics, and shifts do not wrap at board edges. No opponent, action or execution order is removed; the fixed placement clock and all safety/fallback contracts remain unchanged.

The preserved reference is `tests/reference_named_reach_survival.py` from d205cfd. Tests compare completed proofs, first failure evidence and state/scenario counts for all 20 timeout-prefix states, plus 256 randomized time-varying frontiers for both rearming modes, including duplicate names and edge positions. Existing historical/random proof tests and movement tests passed (13 targeted tests in total). The full suite passed 456 tests with one skip; repeated fixed-core performance verification also passed.

The original 008a99f run remains terminal. A future regression must use a newly frozen source identity and independent run directory; passing historical tests alone is not admission or Task 4 qualification. Submitted files are unchanged.

Fixed-core CPU 3 timing used one warmup followed by ten alternating old/new samples per historical state, keeping the runtime 400ms budget. Across 200 measured calls per implementation, the named-set reference timed out ten times (P95 255.745ms, max 403.061ms); the grid implementation had zero timeouts (P95 128.727ms, max 185.085ms). These are complete safety-entry timings, not complete-act admission; full correctness comparisons used separate completed offline proofs, not the reference's timed-out partial outputs.

## Whole-game regression at ca68c8a

World 24025 completed all 400 steps with score 7, one kill, 44 bombs and no learner death. All registered gates passed, including zero invalid actions, zero safety-search/guarantee/collapse events and complete-act P95/max 82.631/222.514ms. A separate raw audit checked 44 placements and 263 pending actions without clock or fallback findings. This resolves the recorded whole-game regression, not general safety admission or Task 4 qualification; next compare v5/v9 on the registered 60 engineering worlds for Tasks 1–3 before further strong-opponent sweeps.

## Completed paired retention at 933de65

All 360 games and all existing gates passed. Task1 score and Task2 coins/crates retained 100%; Task3 score/coins/crates retained 94.79%/98.60%/99.12%. Task3 score difference versus v5 was -0.4000 (10,000-sample paired 95% interval [-0.9833, 0.1667]), kills -0.0667 ([-0.1667, 0.0167]); this is not evidence of improved combat performance.

The candidate raw audit covered 5,408 placements and 32,208 pending actions without clock, placement or selected-action proof inconsistencies. Candidate complete-act P95/max for Tasks1/2/3 were 9.135/13.931ms, 7.696/17.829ms and 23.515/54.831ms. All 180 pairs had matching world/opponent seeds, parent checkpoint and model identity; the registered parent SHA was rechecked. Task2/3 had zero self deaths and 100% bomb survival, and every Task3 game used bombs.

The next frozen test reuses the registered 60 engineering worlds against three rule-based opponents. It is not formal training or independent confirmation/main validation, and any new engineering failure stops that run with evidence preserved.

## Completed strong-opponent sweep at 3f67cbc

All 60 registered engineering worlds passed the existing gates: mean score 3.7833, coins 2.7000, crates 33.8500, kills 0.2167, first-place rate 41.67%, zero self deaths, 100% bomb survival and no zero-bomb games. Complete-act P95/max were 71.946/363.268ms. Auditing 22,748 actions, 2,441 placements and 14,526 pending actions found no clock, selected-action proof or engineering inconsistencies.

Opponent deaths fell from nine in v8 to four (24013, 24022, 24023, 24058), all outside own responsibility and after the last recorded proof endpoint. This does not prove those deaths globally unavoidable. Compared with v8 on the same worlds/opponent seeds, score changed by +0.2500 (paired 95% interval [-0.3333, 0.8500]), kills by +0.0667 ([-0.0500, 0.1833]), and survival by +0.0833 ([-0.0167, 0.1833]); these are engineering observations, not Task 4 qualification.

Next test Tasks1–4 on the previously unused 24300–24399 engineering range, with four distinct physical cores and the same fail-fast gates. The scan covered all 12 worktrees and 37,307 JSON files with no read errors. The range remains engineering data, not an independent confirmation/main-validation claim, and 20000–20099 stays sealed.

## Fresh-world terminal failure at 5dbb4e7

The 24300–24399 sweep stopped at Task4 world 24384, step79, on `robust_search_timed_out`. Tasks1–3 completed 100 games each; Task4 completed 84 games, then aborted its next game. The full act took 408.911ms, with 573,224 logical states evaluated before the 400ms search deadline. This is a terminal engineering failure, not a passed 400-game evaluation. The submitted ZIP and parent weights are unchanged.

The original state window, worker failure, metadata, controller result and raw completed-game audit are preserved in `experiments/results/safety_reach_grid_20260917/fresh_failure`. A fixed-core CPU3 replay of the exact safety call completed five times in 268.824–295.020ms. A further 100 sequential repetitions had no timeout but ranged from 257.736 to 399.860ms. These successful isolated calls do not invalidate the original failure; the timing issue remains unresolved and the reproduction needs the relevant runtime context. No algorithm or safety threshold has been changed in response.

Reproduce from this worktree with `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 3 /export/data/sfan/miniforge3/envs/mle/bin/python experiments/results/safety_reach_grid_20260917/fresh_failure/repro_24384_repeat.py`. This diagnostic still references the preserved raw run directory; it asserts the actual safety timeout flag. Next reproduce the recorded sequence or whole game to capture the timing context before choosing a fix.

## Final diagnostic replay and user-requested stop

The user requested completion of the running experiment, a progress report, and then termination of goal mode with no additional experiments. The observational whole-worker replay at source `7efa03b` completed all 85 worlds (24300–24384), returned zero, and is preserved under `experiments/results/safety_reach_grid_20260917/fresh_failure/context85`. It does not replace the failed original sweep or establish Task4 qualification.

Across 31,155 actions, complete-act P95/max were 72.699/328.708ms, with zero decision timeouts/skips, search timeouts, guarantee losses, avoidable escape collapses or self deaths. Bomb survival was 100%; invalid action rate was 0.2215%. Raw auditing found no inconsistencies across 3,315 placements and 19,750 pending actions. Eight opponent deaths occurred outside the last recorded proof interval. Mean score was 4.1647, kills 0.2353 and first-place rate 49.41%; these are diagnostic observations without a new paired performance claim.

All 96,433 original actor action records matched the replay, including the original incomplete game's prefix. World24384 step79 completed in 274.914ms rather than the original 408.911ms. GC observation captured substantial pauses, but this run did not reproduce the timeout, so the cause remains unconfirmed and no fix is claimed. The safety algorithm and thresholds were not changed during this investigation.

The submitted archive and parent checkpoint were rehashed and remain unchanged (`b4f96841e573eed9d083b0f21dea9eb4b9941a0722dc850ac015327afa4cb6a4` and `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`). The worker has exited. Work stops here at the user's request; unresolved timeout risk is explicitly retained for any future resumption.
