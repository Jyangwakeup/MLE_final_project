# Frozen historical-opponent control at a fixed training endpoint

## Decision

Use `task4-frozen-opponents-v1` from `182f516d` to compare C (three official
rule agents) with S (five rule games and five historical games per shuffled
ten-game block). Only the learner updates. Each historical cycle covers all
24 ordered triples from a fixed four-policy pool. Leave out the three final
E1 policies for transfer diagnostics, while acknowledging their shared lineage.

Both arms inherit only the original Task3 policy, with fresh Adam and empty
Task4-only Replay. All other E1 parameters and v9 decisions are identical.
The single eligible endpoint is the first round boundary at or beyond 60,000
stage actions. Earlier snapshots cannot rescue an unsuccessful endpoint.

## Consequences

A fixed endpoint controls checkpoint-selection opportunity and avoids repeating
the previous best-checkpoint comparison. Stable framework seats delegate to
frozen policies or the official rule callback at round boundaries, preserving
bomb-owner references. Each seat isolates Python, NumPy and PyTorch streams.
Pool, schedule, role, source, configuration and manifest identities govern
loading, caching and exact resume. No production safety algorithm is forked.

The primary result remains performance against three rule agents. Initial
screening requires both S seeds to exceed the parent and their average to
exceed C. Only then run the third matched pair. A unique S endpoint receives
400 new paired rule worlds alongside its same-seed C endpoint and the parent.
Both final score differences must be positive to report an observed benefit;
confidence intervals containing zero remain uncertain. Pool promotion is a
separate future decision. All owned neural seats receive engineering audits.

The implementation's 24-hour clock includes tests and diagnostics. The
18/20/23/24-hour cutoffs and measured-throughput gate cannot be satisfied by
reducing seeds, world counts or training budgets. Engineering failure or
unexplained self-death ends this campaign; previous artifacts remain unchanged.
