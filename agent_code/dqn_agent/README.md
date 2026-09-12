# DQN agent

This agent uses the same versioned representation and objective base reward as
`agent_code/team_agent`, so comparisons with `q_learning_agent` hold features
and rewards constant:

- feature version: `v1` (40-dimensional one-hot vector)
- reward version: `r1`
- action order and physical legal-action mask: shared team contract

There are no hand-authored best-action labels, direction recommendations, or
rewards for following a human-selected action. The 40→64→64→6 network learns
with replay memory, a target network, Huber loss, and epsilon-greedy exploration
over physically legal actions.

Formal runs receive their Agent/model initialization seed from the runner and
share `linear-v1` with Q-learning: epsilon decreases from 1.0 to 0.05 over
1,920,000 Agent decisions and continues across Tasks. Direct framework use
defaults to Agent seed 0.

Train from the repository root:

```bash
BOMBERMAN_TRAINING_TASK=task1 python3 main.py play --agents dqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --no-gui
```

For curriculum training, use `experiments/run.py --resume-from <parent-run>`.
The runner retains policy/target weights, optimizer, replay buffer, sampler RNG,
Torch RNG, Agent RNG, update counters, and the global epsilon progress. A direct
next Task rebuilds the environment/opponent streams but does not restart epsilon.
A checkpoint with a different or missing feature version is never loaded into
this network.

Exact resume uses `training-resume-v3` and also validates the Agent seed and
complete exploration specification. Earlier v1/v2 resume snapshots and legacy
final checkpoints remain frozen-evaluation inputs only.

The experiment runner accepts `--device cuda` for single-GPU training. Direct
framework use and every frozen evaluation default to CPU, matching the official
runtime. GPU checkpoints include CUDA RNG state and remain loadable on CPU.

Direct official-run training saves `agent_code/dqn_agent/final.pt`; experiment
runs save their own `checkpoints/final.pt` plus private resume generations.

The six formal curriculum chains run on CPU because the retained Task 1 smoke
measured about 120 seconds for the first 20 CPU rounds versus about 130 seconds
for the same GPU prefix. GPU remains supported for engineering checks; smoke
runs are excluded from formal lineage and model selection.




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
