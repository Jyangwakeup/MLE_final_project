# Rainbow Lite index

This is the stable entry point for the Rainbow Lite research family. The
family uses a shared Dueling Double DQN implementation with proportional
prioritized replay and four-step returns. Individual experiments vary the
feature representation, reward contract, and survival-mask boundary.

The directories and configuration paths below are preserved because they are
part of the public experiment history. A listed variant is not necessarily a
qualified or selected competition model.

## Implementation lineage

| Agent | Location | Feature contract | Reward and safety contracts | Role |
|---|---|---|---|---|
| `rainbow_lite_agent` | `agent_code/rainbow_lite_agent/` | `continuous-v4` | r7/r9/r10; survival-mask-v1 | Shared model, replay, training, and callback implementation |
| `rainbow_lite_continuous_v2_agent` | `agent_code/rainbow_lite_continuous_v2_agent/` | `continuous-v2` | r7; survival-mask-v5 | Compact-feature Task 1 experiment |
| `rainbow_lite_no_safety_agent` | `agent_code/rainbow_lite_no_safety_agent/` | `continuous-v4` | r10/r13-r16; safety off | Safety-ablation family |
| `rainbow_lite_v5_agent` | `agent_code/rainbow_lite_v5_agent/` | `continuous-v5` | r17/r18; survival-mask-v1 or v5 | Crate-discipline and WAIT-attractor experiments |
| `rainbow_lite_v6_agent` | `all_other_agent_code/rainbow_lite_v6_agent/` | `continuous-v6-opponent-tracking` | r19/r20; survival-mask-v1 or v5 | Historical opponent-tracking Task 4 variant |
| `rainbow_lite_v6_stable_agent` | `all_other_agent_code/rainbow_lite_v6_stable_agent/` | `continuous-v6-opponent-tracking` | r19/r19a/r19b; survival-mask-v1 | Historical stability ablations |
| `rainbow_lite_v7_agent` | `all_other_agent_code/rainbow_lite_v7_agent/` | `continuous-v7-phase-aware` | r20-r22; survival-mask-v1 | Historical phase-aware variant |
| `rainbow_lite_v8_agent` | `all_other_agent_code/rainbow_lite_v8_agent/` | `continuous-v8-safe-crate-opportunity` | r21/r22; survival-mask-v1 | Historical safe-crate variant |
| `rainbow_lite_v9_agent` | `all_other_agent_code/rainbow_lite_v9_agent/` | `continuous-v9-compact` | r22; survival-mask-v1 | Historical compact variant |
| `rainbow_lite_v10_agent` | `all_other_agent_code/rainbow_lite_v10_agent/` | `continuous-v10-agent-quadrant-density` | r21; survival-mask-v1 | Historical quadrant-density Task 3 variant |
| `rainbow_lite_v11_agent` | `all_other_agent_code/rainbow_lite_v11_agent/` | `continuous-v11-agent-quadrant-crate-objectives` | r20; survival-mask-v1 | Historical quadrant-objective Task 3 variant |
| `rainbow_lite_spatial_v6_agent` | `agent_code/rainbow_lite_spatial_v6_agent/` | `spatial-v6-board12` | r20; survival-mask-v5 | Experimental vector-plus-board CNN; no shipped final checkpoint |

All configurations are kept flat under `experiments/configs/` so recorded
commands and manifests retain their original paths. Search for
`rainbow_lite*.json` and `final_rainbow*.json` to enumerate them.

## Migrations and tests

The explicit migration chain is implemented by
`experiments/migrate_rainbow_v4_to_v5.py`, `migrate_rainbow_v5_to_v6.py`,
`migrate_rainbow_v6_to_v7.py`, `migrate_rainbow_v7_to_v8.py`,
`migrate_rainbow_v8_to_v9.py`, `migrate_rainbow_v7_to_v10.py`, and
`migrate_rainbow_v7_to_v11.py`. The spatial branch uses
`migrate_rainbow_v6_to_spatial.py`. These are explicit transfers, not ordinary
checkpoint resumes.

The focused regression surface is:

- `tests/test_recommended_training_configs.py` and
  `tests/test_experiment_run.py` for configuration and runner contracts;
- `tests/test_continuous_v5.py`, `tests/test_continuous_v6.py`, and
  `tests/test_phase_v7.py` for feature lineage;
- `tests/test_safe_crate_v8.py`, `tests/test_compact_v9.py`,
  `tests/test_quadrant_v10.py`, and `tests/test_quadrant_v11.py` for later
  variants;
- `tests/test_competitive_learners.py` for shared learner behavior.

## Historical launch helpers

One-off launch, evaluation, and monitoring helpers for completed experiments
live under `scripts/archive/rainbow_lite/`. They remain available for audit and
reconstruction, but they depend on machine-local data under the ignored
`runs/` tree and are not current recommended entry points. New experiments
should use `python -m experiments.run` with an explicit configuration.

Detailed experiment narrative and checkpoint evidence remain in
`thesis/members/Zhu Yuyan/` and the versioned research records under
`experiments/results/`.
