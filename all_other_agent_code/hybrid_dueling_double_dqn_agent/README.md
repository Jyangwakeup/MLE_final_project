# Hybrid Dueling Double DQN Agent

> Archived source. The runner and submission builder do not load this agent
> from `all_other_agent_code/`; its experiment contract remains available for
> historical metadata resolution.

## 中文说明

该 Agent 一次调用 `hybrid-v1`，同时获得 `board-v1` 棋盘和 `continuous-v1` 向量，避免
重复执行危险预测与路径搜索。网络由棋盘分支、向量分支、融合层和 Dueling heads 组成。

- Algorithm：`hybrid_dueling_double_dqn`
- Feature：board `(12,17,17)`，vector `(70,)`
- Reward：新训练默认 `r2_balanced`
- Checkpoint：`final.pt`
- 超参数：gamma 0.95、learning rate 2e-4、batch 64、replay 20,000、warmup 5,000、
  target sync 2,000 updates、gradient clip 10
- 探索：epsilon 1.0 → 0.05，衰减 80,000 action steps

棋盘分支输出 128 维，向量分支输出 64 维，拼接后的 192 维表示经 128 维融合层进入 value
和 advantage heads：

```text
Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
```

算法仍是 Double DQN：policy 选择下一合法动作，target 评价。压缩棋盘和 float32 向量的
20,000 条 replay 原始存储约 44 MiB。

## 用 experiments.run 训练、恢复和测试

训练 Task 1：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent hybrid_dueling_double_dqn_agent --seed 11 --n-rounds 1000 \
  --run-id hybrid_dueling_t1_train
```

模型保存在 `runs/hybrid_dueling_t1_train/checkpoints/final.pt`。

断点续训：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent hybrid_dueling_double_dqn_agent --seed 11 --n-rounds 1000 \
  --resume-from runs/hybrid_dueling_t1_train \
  --run-id hybrid_dueling_t1_resume
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode evaluate --task 1 \
  --agent hybrid_dueling_double_dqn_agent \
  --checkpoint runs/hybrid_dueling_t1_train/checkpoints/final.pt \
  --seed 10001 --n-rounds 20 --run-id hybrid_dueling_t1_eval
```

## 用 main.py 快速训练和测试

神经网络 checkpoint 使用 PyTorch 的 `final.pt` 文件名：

```bash
mkdir -p runs/manual_hybrid_dueling/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_hybrid_dueling/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=hybrid-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents hybrid_dueling_double_dqn_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

加载模型进行无 GUI 测试：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_hybrid_dueling/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=hybrid-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents hybrid_dueling_double_dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 是独立于 `hybrid-v1` 和物理 `legal_mask` 的课程学习动作约束。
设为 `false` 只禁止 Agent 选择 `BOMB`，不会改变 board/vector 特征，也不会屏蔽危险但
物理合法的移动。训练和评估同一 Task 时应保持设置一致。

- `experiments.run --task 1` 自动设置 `false`。
- `experiments.run --task 2/3/4` 自动设置 `true`。
- 直接调用 `main.py` 时需显式设置；上面的 Task 1 命令是禁用炸弹示例。

直接训练 Task 2、启用炸弹的示例：

```bash
mkdir -p runs/manual_hybrid_dueling_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_hybrid_dueling_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=hybrid-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents hybrid_dueling_double_dqn_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

统一使用 `--agents` 显式指定参赛者数量。下面只列出一个 Agent，因此是 Task 1 单人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/hybrid_dueling_t1_train/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=hybrid-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents hybrid_dueling_double_dqn_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

下面显式列出四个 Agent，因此是 Task 2 四人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_hybrid_dueling_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=hybrid-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents hybrid_dueling_double_dqn_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。不传 `--no-gui` 即会打开图形界面。

## 单元测试

```bash
python3 -m unittest tests.test_new_agents -v
python3 -m unittest tests.test_submission_package -v
python3 -m unittest discover -s tests
```

checkpoint 保存 policy、target、optimizer、压缩 replay、全部 RNG、网络结构和严格契约。
该方案应分别与 continuous 和 CNN Agent 比较，以判断融合与 Dueling head 是否带来收益。
