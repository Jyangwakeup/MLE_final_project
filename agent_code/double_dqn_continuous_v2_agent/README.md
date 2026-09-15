# Continuous-v2 Double DQN

Pre-registered capability fallback. It is used only after the standard DQN
fails the Task 1/2 joint gate.

The `task3-escape-obligation` experiment keeps this Agent's 84-dimensional
feature vector and `r7_safe_credit_sparse` reward unchanged. Its
`survival-mask-v3` runtime contract vetoes non-robust bomb placement and, while
the Agent remains responsible for its own bomb, prefers the subset retaining
two independent time-expanded escape routes. It never supplies a preferred
direction. The experiment contract is recorded in
`experiments/task3_escape_obligation.json`; v7 Task 2 state enters only through
`--transfer-task3-safety-from` and subsequent exact snapshots use v9.
