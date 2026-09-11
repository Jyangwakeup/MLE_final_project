# Basic DQN agent

This agent learns from real game outcomes plus objective reward shaping:

- `COIN_COLLECTED`: `settings.REWARD_COIN`
- `KILLED_OPPONENT`: `settings.REWARD_KILL`
- `CRATE_DESTROYED`: `+0.2`
- `KILLED_SELF` or `GOT_KILLED`: `-10` once per transition
- `INVALID_ACTION`: `-0.1`
- each transition: `-0.01` time cost

There are no hand-authored best-action labels, direction recommendations, or
rewards for following a human-selected action. The shaping above evaluates only
the observed result of an action. The network observes seven board channels plus
bomb availability and learns with replay memory, a target network, Huber loss,
and epsilon-greedy exploration over physically legal actions.

Train from the repository root:

```bash
BOMBERMAN_TRAINING_TASK=task1 python3 main.py play --agents dqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --no-gui
```

Set `BOMBERMAN_TRAINING_TASK` to a stable curriculum-stage name. When the saved
name changes (for example, from `task1` to `task2`), the checkpoint weights and
optimizer are retained while epsilon exploration progress restarts. Continuing
with the same name also continues the saved exploration progress.

The checkpoint is saved as `agent_code/dqn_agent/dqn-model.pt`.



BOMBERMAN_TRAINING_TASK=task2 \
python3 main.py play \
  --agents dqn_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 10000 \
  --no-gui