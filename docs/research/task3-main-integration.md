# Task 3 validated checkpoint integration

The frozen seed22/c150 checkpoint is published unchanged at
`agent_code/double_dqn_continuous_v2_agent/task3_validated.pt`.
Its release manifest is `experiments/task3_validated_release.json`.
`final.pt` continues to identify the remote Task 2 baseline; Task 3 loading is
explicit, as documented in the agent README.

The merge preserves remote main's other agents, reward identifiers, phase and
distillation experiments, context reuse and historical evidence. Conflict
resolution retains current v11 resume metadata and both explicit Task 3 transfer
APIs. Resume source hashes remain mandatory even when an old record lacks a
scope. Relative Python imports are included in runtime hashing so shared feature
implementations cannot silently escape the provenance contract.

History reads remain pure. Actual decisions advance history; position history
needed by the newer remote feature sets is projected without mutating old-state
queries. Versioned safety v1 through v5 coexist. Double DQN's registered five-step
return, safety replay and immutable decision snapshots are retained alongside
remote phase distillation. No reward version, trained tensor or gate is retuned.

Integration validation:

- Full unittest: 388 tests, 1 skipped, all remaining tests passed.
- World 19489 replay: all 1012 recorded actions across all agents match the
  corrected frozen-source replay. Every previously recorded episode-agent
  metric matches. The remote runner additionally records `zero_utility_bombs`.
- The portable false-positive and true-positive collapse regressions, lifecycle,
  resume, safety, packaging and controller checks are included in the suite.
- Weight SHA-256 is unchanged:
  `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`.

Logs: `/tmp/task3-merge-tests2.log`, `/tmp/task3-merge-equivalence.log`;
replay artifacts: `runs/task3_merge_equivalence_19489/`.
These are integration checks, not a rerun of the 100-world selection protocol.
The published frozen statistics continue to refer to source `7cc6f02`; the merge
and its compatibility checks are separate provenance. Task 4 is not launched.
