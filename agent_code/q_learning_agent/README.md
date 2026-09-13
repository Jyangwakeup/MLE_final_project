# Q-learning Agent

## 中文说明

<<<<<<< HEAD
正式训练由 Runner 传入 Agent seed，并与 DQN 共用 `linear-v1` 探索：ε 从 1.0 在
1,920,000 次自身动作内线性降到 0.05，跨 Task 保留累计动作步。直接使用官方框架时
Agent seed 默认为 0。

训练：
=======
这是表格型 Q-learning Agent，不依赖深度学习框架。它使用 `discrete-q-v2`：在冻结的
14 元状态后加入最近金币距离分桶和上一步移动，共 16 个离散字段。

- Algorithm：`q_learning`
- Feature：`discrete-q-v2`，16 state / 50 one-hot vector
- Reward：`r4_anti_oscillation`，距离势能加无进展连续反转惩罚
- Checkpoint：`final.pkl`
- 动作只使用公共物理合法 mask；不会用 Feature 规则直接给出最佳动作

## 用 experiments.run 训练、恢复和测试
>>>>>>> e6253fd1 (add more feature id, reward id, and model)

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent q_learning_agent --seed 11 --n-rounds 1000 \
  --run-id q_learning_t1_train
```

模型位于 `runs/q_learning_t1_train/checkpoints/final.pkl`。断点续训：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent q_learning_agent --seed 11 --n-rounds 1000 \
  --resume-from runs/q_learning_t1_train --run-id q_learning_t1_resume
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode evaluate --task 1 \
  --agent q_learning_agent \
  --checkpoint runs/q_learning_t1_train/checkpoints/final.pkl \
  --seed 10001 --n-rounds 20 --run-id q_learning_t1_eval
```

## 用 main.py 快速训练和测试

```bash
mkdir -p runs/manual_q_learning/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_q_learning/checkpoints/final.pkl" \
BOMBERMAN_REWARD_ID=r4_anti_oscillation \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents q_learning_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

无 GUI 测试：

<<<<<<< HEAD
跨 Task 训练请使用仓库级实验入口的 `--resume-from`。它保留 Q-table、动作步数和
Agent RNG，因此 epsilon 不会在新 Task 重新开始；同时会为下一 Task 重建环境和对手随机流。
不要通过复制 `final.pkl` 模拟精确恢复，因为冻结推理 checkpoint 不包含 Runner 状态。
当前精确恢复协议为 `training-resume-v3`，并校验 Agent seed 与完整探索配置；旧 v1/v2
resume 和旧 final checkpoint 只能冻结评估。

正式六链命令、每阶段预算和门槛见仓库根目录 `IMPLEMENTATION_GUIDE.md`；正式配置固定 CPU，
不使用上面的 10,000 局官方入口示例作为课程链。
=======
```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_q_learning/checkpoints/final.pkl" \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents q_learning_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```
>>>>>>> e6253fd1 (add more feature id, reward id, and model)

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 是课程学习动作约束，不会改写 Feature 的物理 `legal_mask`。
`experiments.run --task 1` 会自动设置为 `false`；Task 2、3、4 自动设置为 `true`。
直接调用 `main.py` 时必须显式设置，上面的 Task 1 命令就是禁用炸弹的示例。

直接训练 Task 2 并启用炸弹：

```bash
mkdir -p runs/manual_q_learning_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_q_learning_task2/checkpoints/final.pkl" \
BOMBERMAN_REWARD_ID=r4_anti_oscillation \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents q_learning_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

统一使用 `--agents` 显式指定参赛者数量。下面只列出一个 Agent，因此是 Task 1 单人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/q_learning_t1_train/checkpoints/final.pkl" \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents q_learning_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

下面显式列出四个 Agent，因此是 Task 2 四人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_q_learning_task2/checkpoints/final.pkl" \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents q_learning_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。GUI 命令不要添加 `--no-gui`；它用于行为观察，
不替代正式冻结评估。

## 单元测试

```bash
python3 -m unittest tests.test_q_learning_agent -v
python3 -m unittest tests.test_experiment_run tests.test_resume -v
python3 -m unittest discover -s tests
```
