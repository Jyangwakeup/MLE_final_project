# Continuous Double DQN Agent

## 中文说明

该 Agent 使用公共 `continuous-v1` 的 70 维连续手工摘要和 Double DQN。policy network
选择下一合法动作，target network 评价该动作，避免使用同一网络同时选择和评价造成的
过估计。Feature 的 `legal_mask` 只屏蔽物理非法动作。

- Algorithm：`double_dqn`
- Feature：`(70,) float32`
- 网络：`70 → 128 → 128 → 6`，隐藏层为 ReLU
- Reward：新训练默认 `r2_balanced`
- Checkpoint：`final.pt`
- 超参数：gamma 0.95、learning rate 3e-4、batch 64、replay 50,000、warmup 2,000、
  target sync 1,000 updates、gradient clip 10
- 探索：epsilon 1.0 → 0.05，衰减 80,000 action steps

训练使用 Adam、Huber loss、replay buffer 和硬 target 同步；推理固定使用 CPU 单线程。

## 用 experiments.run 训练、恢复和测试

训练 Task 1：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent double_dqn_continuous_agent --seed 11 --n-rounds 1000 \
  --run-id continuous_ddqn_t1_train
```

模型保存在 `runs/continuous_ddqn_t1_train/checkpoints/final.pt`。

断点续训：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent double_dqn_continuous_agent --seed 11 --n-rounds 1000 \
  --resume-from runs/continuous_ddqn_t1_train \
  --run-id continuous_ddqn_t1_resume
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode evaluate --task 1 \
  --agent double_dqn_continuous_agent \
  --checkpoint runs/continuous_ddqn_t1_train/checkpoints/final.pt \
  --seed 10001 --n-rounds 20 --run-id continuous_ddqn_t1_eval
```

`experiments.run` 会严格检查算法、Feature、reward、动作顺序和网络结构，并写出完整实验记录。

## 用 main.py 快速训练和测试

神经网络 checkpoint 使用 PyTorch 的 `final.pt` 文件名：

```bash
mkdir -p runs/manual_continuous_ddqn/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_continuous_ddqn/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=continuous-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_dqn_continuous_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

加载模型进行无 GUI 测试：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_continuous_ddqn/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=continuous-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_dqn_continuous_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 只控制课程阶段是否允许选择 `BOMB`，不会改写公共 Feature 的
物理 `legal_mask`，也不会按危险程度屏蔽动作。训练和评估同一 Task 时应使用相同设置。

- `experiments.run --task 1` 自动禁用炸弹（`false`）。
- `experiments.run --task 2/3/4` 自动启用炸弹（`true`）。
- 直接调用 `main.py` 时需要显式设置；上面的 Task 1 命令展示了禁用炸弹。

直接训练 Task 2、启用炸弹的示例：

```bash
mkdir -p runs/manual_continuous_ddqn_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_continuous_ddqn_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=continuous-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents double_dqn_continuous_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

统一使用 `--agents` 显式指定参赛者数量。下面只列出一个 Agent，因此是 Task 1 单人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/continuous_ddqn_t1_train/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=continuous-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_dqn_continuous_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

下面显式列出四个 Agent，因此是 Task 2 四人局：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_continuous_ddqn_task2/checkpoints/final.pt" \
BOMBERMAN_FEATURE_ID=continuous-v1 BOMBERMAN_REWARD_ID=r2_balanced \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents double_dqn_continuous_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。不传 `--no-gui` 即会打开界面。

## 单元测试

```bash
python3 -m unittest tests.test_new_agents -v
python3 -m unittest tests.test_submission_package -v
python3 -m unittest discover -s tests
```

checkpoint 包含 policy、target、optimizer、replay、所有 RNG 状态、更新计数、网络结构和
Feature/reward/action 契约。该方案主要与 40 维 `dqn_agent` 冻结基线比较。
