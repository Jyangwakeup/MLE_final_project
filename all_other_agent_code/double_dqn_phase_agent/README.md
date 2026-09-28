# Phase-aware Double DQN

Archived source: the runner and submission builder do not load this agent from
`all_other_agent_code/`. Its experiment contract remains available for
historical metadata resolution.

This training-only curriculum agent uses the 117-dimensional
`continuous-phase-v1` contract. It keeps the 84 parent inputs as an exact
prefix, adds factual safety and match-phase inputs, and chooses actions with
Double DQN inside a versioned survival mask. Task 3 must start through
`--transfer-task3-from`; ordinary v7 resume is intentionally rejected.
