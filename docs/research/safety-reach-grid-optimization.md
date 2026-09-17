# Equivalent opponent reach grid optimization

The terminal v9 regression at world 24025 step 75 exceeded the unchanged 400ms budget. Its complete shared search required 452–475ms in three offline runs; profiling found abstract opponent reach expansion dominated the cost.

The optimization replaces per-name Python set expansion with two boolean grid frontiers. Union distributes over the common blocked-cell transition, so this preserves the original opponent occupancy union and armed occupancy union at every future step. Initialization retains name deduplication semantics, and shifts do not wrap at board edges. No opponent, action or execution order is removed; the fixed placement clock and all safety/fallback contracts remain unchanged.

The preserved reference is `tests/reference_named_reach_survival.py` from d205cfd. Tests compare completed proofs, first failure evidence and state/scenario counts for all 20 timeout-prefix states, plus 256 randomized time-varying frontiers for both rearming modes, including duplicate names and edge positions. Existing historical/random proof tests and movement tests passed (13 targeted tests in total). The full suite passed 456 tests with one skip; repeated fixed-core performance verification also passed.

The original 008a99f run remains terminal. A future regression must use a newly frozen source identity and independent run directory; passing historical tests alone is not admission or Task 4 qualification. Submitted files are unchanged.

Fixed-core CPU 3 timing used one warmup followed by ten alternating old/new samples per historical state, keeping the runtime 400ms budget. Across 200 measured calls per implementation, the named-set reference timed out ten times (P95 255.745ms, max 403.061ms); the grid implementation had zero timeouts (P95 128.727ms, max 185.085ms). These are complete safety-entry timings, not complete-act admission; full correctness comparisons used separate completed offline proofs, not the reference's timed-out partial outputs.
