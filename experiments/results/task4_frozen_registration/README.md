# Frozen-opponent implementation admission

Base: 182f516d. Implementation clock: 2026-09-19T18:10:22+02:00.

- Full unittest: 504 tests, one pre-existing optional skip; full-unittest.txt.
- After controller changes, focused suite: 15 tests; frozen-unittest.txt.
- Real round-boundary replay: 12 continuous versus 6+6 restored rounds, 4,800 actions, 2,801 updates. Full checkpoints, Replay, optimizer, global/world random streams and 15,631 decision records identical.
- Separate round-5 rule to round-6 historical restoration: 1,600 decision records and full checkpoints identical.
- Each of the five completed implementation runs passed learner and all frozen-seat engineering audits; runtime_verification.json. They are reusable engineering data, not independent score evidence.
- Controller audit merging and completed-snapshot adoption were tested against these real artifacts; controller_resume_verification.json.
- World registration scanned 31,540 unique JSON artifacts from all worktrees. Training/diagnostic/schedule streams are separately registered.
- All historical weights remain immutable; parent and seven registered model identities are in the campaign manifest.

Runtime replay used the exact agent-runtime-v2 source hash recorded in each test checkpoint; controller-only refinements do not alter it. The final source commit binds worker/controller gate behavior and budgets. Archived scripts reproduce the checks from the retained runs in this worktree.

Background command and PID are recorded under runs/task4_frozen_20260919. Formal training remains conditional on the 20-world engineering gate, both fresh 20-round diagnostics and measured throughput.
