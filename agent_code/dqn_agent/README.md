# DQN agent

This agent uses the same versioned representation and objective base reward as
`agent_code/team_agent`, so comparisons with `q_learning_agent` hold features
and rewards constant:

- feature version: `v1` (40-dimensional one-hot vector)
- reward version: `base-v1`
- action order and physical legal-action mask: shared team contract

There are no hand-authored best-action labels, direction recommendations, or
rewards for following a human-selected action. The 40→64→64→6 network learns
with replay memory, a target network, Huber loss, and epsilon-greedy exploration
over physically legal actions.

Train from the repository root:

```bash
BOMBERMAN_TRAINING_TASK=task1 python3 main.py play --agents dqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --no-gui
```

Set `BOMBERMAN_TRAINING_TASK` to a stable curriculum-stage name. When the saved
name changes (for example, from `task1` to `task2`), the checkpoint weights and
optimizer are retained while epsilon exploration progress restarts. Continuing
with the same name also continues the saved exploration progress. A checkpoint
with a different or missing feature version is never loaded into this network.

The checkpoint is saved as `agent_code/dqn_agent/dqn-model.pt`.




BOMBERMAN_TRAINING_TASK=task2 \
python3 main.py play \
  --agents dqn_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 10000 \
  --no-gui


python3 experiments/run.py \
  --config experiments/configs/base.json \
  --mode train \
  --task 1 \
  --agent dqn_agent \
  --n-rounds 10000 \
  --seed 11 \
  --run-id dqn_task1_train
```

通过实验运行器选择 `--task 1` 时，训练和评估都会自动禁用 `BOMB`；Task 2–4
则允许放置炸弹。
