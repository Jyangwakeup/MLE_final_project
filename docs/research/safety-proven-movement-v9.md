# v9 movement preference — implementation under test

Recorded v8 world 24025 step 189 selected RIGHT. The real v8 safety entry point admitted all five non-bomb actions; a separate seven-step opponent-rearming proof admitted only UP, LEFT and WAIT. An expected-red regression reproduced this discrepancy before implementation, without changing the original v8 engineering failure counters.

The proposed v9 contract shares one search across all candidates outside own responsibility, preserves Q-value ranking within the proved set, and records an explicit fallback if no proof succeeds. It never admits an unproved BOMB and leaves pending fixed-deadline behavior unchanged. A failure to prove is conservative rejection, not proof that death is inevitable. The preference is not a persistent non-bomb survival guarantee.

Four targeted movement tests passed, covering the recorded trap, unchanged v8 behavior, explicit empty-proof fallback and preservation of the active placement deadline. Full-suite and initial historical performance checks passed; whole-game validation is pending. No claim of safety admission, Task 4 qualification or improved score is made.

Exploratory timing on 80 recorded death-prefix states found no search timeouts (v8/v9 maximum 23.009/46.209ms). The nonpending subset of 100 older failure states contained 48 eligible states, with v9 maximum 249.493ms and no timeout; 52 old pending records were excluded because they lack a recorded placement origin. These are single-sample safety-entry timings, not repeated benchmarks or complete-act admission.

The full unittest suite completed 453 tests with one skip. The only new behavior is in the explicit v9 safety contract; the v8 frozen study remains in its original worktree.

## Terminal regression failure at 008a99f

The first whole-game regression stopped at world 24025 step 75 with `robust_search_timed_out`, complete act 409.827ms, before a completed game. The search recorded 377,648 evaluated states and 530 scenarios while processing RIGHT/LEFT/WAIT/BOMB. The initial historical benchmarks did not cover this newly reached state; v9 is not admitted and the failed run remains terminal.

Next diagnosis separates movement-only, bomb-only and shared combined search at this exact state, using an explicitly offline enlarged proof budget only to obtain complete reference results. Runtime remains 400ms, and no official training or submission update is authorized by this experiment.

The enlarged-budget complete reference reproduced combined-search cost at 451.718/475.283/467.258ms (576 scenarios), versus 213.985–227.081ms for BOMB only and 234.682–248.024ms for movement only. All four actions eventually proved; this is computation cost rather than an empty safe set. Profiling attributed 0.587/0.633s cumulative time to abstract feedback computation and only 0.037s to first-transition enumeration; most self-time is Python set-based opponent reach expansion. The next bounded optimization should replace that reach expansion by an equivalent boolean-grid recurrence, checking complete proof outputs against an independent preserved implementation before any new run.
