# DQN Agent（冻结基线）

## 中文说明

该 Agent 与 Q-learning 使用相同的 `discrete-q-v2` 和 `r4_anti_oscillation`。50 维
one-hot 输入经过 `50 → 64 → 64 → 6` MLP。

- Algorithm：`dqn`
- Feature：`discrete-q-v2` 的 `(50,) float32` vector
- Reward：`r4_anti_oscillation`
- Checkpoint：`final.pt`
- 训练组件：replay buffer、target network、Huber loss、epsilon-greedy
- 推理：CPU 单线程，只 hard-mask 物理非法动作

<<<<<<< HEAD
Formal runs receive their Agent/model initialization seed from the runner and
share `linear-v1` with Q-learning: epsilon decreases from 1.0 to 0.05 over
1,920,000 Agent decisions and continues across Tasks. Direct framework use
defaults to Agent seed 0.

Train from the repository root:
=======
这是冻结基线；新的 Double DQN Agent 不会改变本模型结构或更新语义。

## 用 experiments.run 训练、恢复和测试
>>>>>>> e6253fd1 (add more feature id, reward id, and model)

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent dqn_agent --seed 11 --n-rounds 1000 \
  --run-id dqn_t1_train
```

模型位于 `runs/dqn_t1_train/checkpoints/final.pt`。断点续训：

<<<<<<< HEAD
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
=======
```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent dqn_agent --seed 11 --n-rounds 1000 \
  --resume-from runs/dqn_t1_train --run-id dqn_t1_resume
>>>>>>> e6253fd1 (add more feature id, reward id, and model)
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode evaluate --task 1 \
  --agent dqn_agent --checkpoint runs/dqn_t1_train/checkpoints/final.pt \
  --seed 10001 --n-rounds 20 --run-id dqn_t1_eval
```

## 用 main.py 快速训练和测试

```bash
mkdir -p runs/manual_dqn/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_dqn/checkpoints/final.pt" \
BOMBERMAN_REWARD_ID=r4_anti_oscillation \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents dqn_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

无 GUI 测试：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_dqn/checkpoints/final.pt" \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 是课程学习动作约束，不会改写 Feature 的物理 `legal_mask`。
`experiments.run --task 1` 会自动设置为 `false`；Task 2、3、4 自动设置为 `true`。
直接调用 `main.py` 时必须显式设置，上面的 Task 1 命令就是禁用炸弹的示例。

直接训练 Task 2 并启用炸弹：

```bash
mkdir -p runs/manual_dqn_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_dqn_task2/checkpoints/final.pt" \
BOMBERMAN_REWARD_ID=r4_anti_oscillation \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents dqn_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

统一使用 `--agents` 显式指定参赛者数量。下面只列出一个 Agent，因此是 Task 1 单人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/dqn_t1_train/checkpoints/final.pt" \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

下面显式列出四个 Agent，因此是 Task 2 四人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_dqn_task2/checkpoints/final.pt" \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents dqn_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。不传 `--no-gui` 即打开界面。

## 单元测试

```bash
python3 -m unittest tests.test_dqn_agent -v
python3 -m unittest tests.test_experiment_run tests.test_resume -v
python3 -m unittest discover -s tests
```

训练 checkpoint 保存 policy、target、optimizer、replay、RNG、更新计数以及严格 Feature、
reward 和动作顺序契约。
