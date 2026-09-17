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
