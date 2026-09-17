# Continuous-v2 Double DQN — Task 2 winner

This Agent uses an 84-dimensional `continuous-v2` state representation,
Double DQN, the `r7_safe_credit_sparse` training reward, and
`survival-mask-v1` in `all` mode with a seven-step horizon. The bundled
`final.pt` is the seed 22 checkpoint that passed the three-training-seed
development gate and the one permitted 100-seed main validation.

The full selection contract, hashes, and evaluation results are recorded in
[`experiments/task2_winner.json`](../../experiments/task2_winner.json).

## Historical safety experiments

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

## Frozen Task 3 candidate (2026-09-17)

`task3_validated.pt` is the selected seed22/c150 Task 3 checkpoint. The existing
`final.pt` remains the Task 2 winner; select Task 3 explicitly with
`BOMBERMAN_CHECKPOINT` or the runner's `--checkpoint` flag. Both are frozen
inference checkpoints, not an authorization to resume historical training.

The release contract, SHA-256 and evaluation configuration are in
[`experiments/task3_validated_release.json`](../../experiments/task3_validated_release.json).
The three-seed confirmation and unique 100-world main validation passed under
source `7cc6f0249c0ecb09a539614965c6ee3f758b26e3`; see
[the results](../../docs/research/task3-counter-validation-results.md).

A development replay with the integrated source can be run from repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/run.py \
  --config experiments/configs/task3_lifecycle_A.json --mode evaluate --device cpu \
  --task 3 --agent double_dqn_continuous_v2_agent --seeds 19489 --n-rounds 1 \
  --checkpoint agent_code/double_dqn_continuous_v2_agent/task3_validated.pt \
  --run-id task3_published_replay_19489
```

Use a fresh run ID for subsequent replays; never overwrite frozen evidence.
