# Keep standalone Expected SARSA and Rainbow Lite active

Keep `agent_code/expected_sarsa/` and `agent_code/rainbow_lite/` as the active
Expected SARSA and Rainbow Lite packages. These directories include their
selected weights and vendored runtime dependencies, so the package builder
reuses those dependencies instead of injecting a second copy.

Keep the lambda and versioned experimental variants under
`all_other_agent_code/`. The experiment registry may read their metadata for
historical configuration and result inspection, but the current runner and
submission builder do not treat them as active agents.

This boundary makes each model family have one current standalone package
while keeping historical variants available for research. It does not rewrite
existing manifests or result records.
