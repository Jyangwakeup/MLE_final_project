# Archive the Double Q agent family

Move `double_q_agent`, `double_q_compact_agent`, `double_q_lambda_agent`, and
`optimized_double_q_lambda_agent` out of the active `agent_code/` surface and
retain them under `all_other_agent_code/`. The active and archived copies of
the first two were duplicates; keep the latest `double_q_agent` callback when
consolidating them.

The experiment registry retains metadata resolution for historical contracts.
The current runner and submission builder do not treat these variants as
active or packageable agents. Existing configurations, manifests, and results
remain unchanged as historical evidence.
