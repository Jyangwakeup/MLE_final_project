# CNN Double DQN Agent

## 中文说明

该 Agent 使用 `board-v1` 原始空间通道和卷积网络，研究 CNN 能否从棋盘结构中学习出优于
手工摘要的表示。输入不会转置：`board[channel,x,y]` 与 `field[x,y]` 一致。

- Algorithm：`cnn_double_dqn`
- Feature：官方棋盘为 `(12,17,17) float32`
- Reward：新训练使用 `r3_potential`
- Checkpoint：`final.pt`
- 网络：三层卷积、MaxPool、AdaptiveAvgPool `(4,4)`、128 维全连接层和 6 个动作输出
- 超参数：gamma 0.95、learning rate 2e-4、batch 64、replay 50,000、warmup 5,000、
  target sync 2,000 updates、gradient clip 10
- 探索：epsilon 1.0 → 0.05，衰减 80,000 action steps

Replay 用 `packbits` 保存 11 个二值通道，用 `uint8` 精确保存炸弹倒计时。50,000 条
classic transition 约占 84 MiB；直接保存 float32 state/next-state 约需 1.29 GiB。

## 用 experiments.run 训练、恢复和测试

训练 Task 1：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode train --task 1 \
  --agent cnn_double_dqn_agent --seed 11 --n-rounds 1000 \
  --run-id cnn_ddqn_r3_t1_train
```

模型保存在 `runs/cnn_ddqn_r3_t1_train/checkpoints/final.pt`。

断点续训：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode train --task 1 \
  --agent cnn_double_dqn_agent --seed 11 --n-rounds 1000 \
  --resume-from runs/cnn_ddqn_r3_t1_train --run-id cnn_ddqn_r3_t1_resume
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode evaluate --task 1 \
  --agent cnn_double_dqn_agent \
  --checkpoint runs/cnn_ddqn_r3_t1_train/checkpoints/final.pt \
  --seed 10001 --n-rounds 20 --run-id cnn_ddqn_r3_t1_eval
```

## 用 main.py 快速训练和测试

神经网络 checkpoint 使用 PyTorch 的 `final.pt` 文件名：

```bash
mkdir -p runs/manual_cnn_ddqn/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_cnn_ddqn/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=board-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents cnn_double_dqn_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

加载模型进行无 GUI 测试：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_cnn_ddqn/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=board-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents cnn_double_dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 是课程学习动作约束，与 `board-v1` 的棋盘通道和 Feature 的物理
`legal_mask` 分离。它只控制 Agent 能否选择 `BOMB`，不会隐藏炸弹、危险通道或其他危险但
合法的动作。训练和评估同一 Task 时应保持设置一致。

- `experiments.run --task 1` 自动设置 `false`。
- `experiments.run --task 2/3/4` 自动设置 `true`。
- 直接调用 `main.py` 时需显式传入；上面的 Task 1 命令是禁用炸弹示例。

直接训练 Task 2、启用炸弹的示例：

```bash
mkdir -p runs/manual_cnn_ddqn_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_cnn_ddqn_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=board-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents cnn_double_dqn_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

统一使用 `--agents` 显式指定参赛者数量。下面只列出一个 Agent，因此是 Task 1 单人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/cnn_ddqn_r3_t1_train/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=board-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents cnn_double_dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

下面显式列出四个 Agent，因此是 Task 2 四人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_cnn_ddqn_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=board-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents cnn_double_dqn_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。GUI 命令不要添加 `--no-gui`。界面用于观察行为，
正式指标仍应来自固定 seeds 的冻结评估。

## 单元测试

```bash
python3 -m unittest tests.test_new_agents -v
python3 -m unittest tests.test_submission_package -v
python3 -m unittest discover -s tests
```

checkpoint 保存 policy、target、optimizer、压缩 replay、全部 RNG、输入 shape 和严格版本
契约。该 Agent 应与 `double_dqn_continuous_agent` 做棋盘通道对手工摘要的控制比较。
