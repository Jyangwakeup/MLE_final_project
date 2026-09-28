# Keep continuous Double DQN v2 on the active surface

Keep `double_dqn_continuous_v2_agent` as the active continuous-feature Double
DQN development agent. Move the unversioned baseline and the v3/v4 variants to
`all_other_agent_code/` as retained historical source. The current experiment
runner and package builder operate on active `agent_code/` packages; archived
variants are not implicitly runnable or packageable.

Keep `die_hardest` as the separately selected frozen competition package.
Phase-aware, CNN, hybrid, and non-Double-DQN agents remain in their existing
locations. This decision narrows the active continuous line without changing
historical experiment manifests or result records.

The active inventory and current training examples identify v2 as the
continuous Double DQN entry point. Archived callbacks remain available for
metadata inspection, while reproducibility records preserve the agent names
and source identities used when those experiments ran.
