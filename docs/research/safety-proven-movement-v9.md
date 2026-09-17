# v9 movement preference — implementation under test

Recorded v8 world 24025 step 189 selected RIGHT. The real v8 safety entry point admitted all five non-bomb actions; a separate seven-step opponent-rearming proof admitted only UP, LEFT and WAIT. An expected-red regression reproduced this discrepancy before implementation, without changing the original v8 engineering failure counters.

The proposed v9 contract shares one search across all candidates outside own responsibility, preserves Q-value ranking within the proved set, and records an explicit fallback if no proof succeeds. It never admits an unproved BOMB and leaves pending fixed-deadline behavior unchanged. A failure to prove is conservative rejection, not proof that death is inevitable. The preference is not a persistent non-bomb survival guarantee.

Four targeted movement tests passed, covering the recorded trap, unchanged v8 behavior, explicit empty-proof fallback and preservation of the active placement deadline. Full-suite and initial historical performance checks passed; whole-game validation is pending. No claim of safety admission, Task 4 qualification or improved score is made.

Exploratory timing on 80 recorded death-prefix states found no search timeouts (v8/v9 maximum 23.009/46.209ms). The nonpending subset of 100 older failure states contained 48 eligible states, with v9 maximum 249.493ms and no timeout; 52 old pending records were excluded because they lack a recorded placement origin. These are single-sample safety-entry timings, not repeated benchmarks or complete-act admission.

The full unittest suite completed 453 tests with one skip. The only new behavior is in the explicit v9 safety contract; the v8 frozen study remains in its original worktree.
