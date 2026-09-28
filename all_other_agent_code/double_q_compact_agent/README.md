# Double Q Compact Agent

> Archived source. The runner and submission builder do not load this agent
> from `all_other_agent_code/`; its experiment contract remains available for
> historical metadata resolution.

## 中文说明

该 Agent 使用公共 `discrete-compact-v1` 特征和 Double Q-learning，用来研究对称规范化、
紧凑状态空间和 Double Q 更新能否改善表格学习。

- Algorithm：`double_q_learning`
- Feature：12 个离散字段，理论状态数 622,080
- Reward：新训练使用 `r3_potential`
- Checkpoint：`final.pkl`
- 超参数：learning rate 0.1、gamma 0.95、epsilon 1.0 → 0.05，衰减 80,000 steps

Agent 维护 `q_table_a`、`q_table_b`。更新 A 时由 A 选择下一动作、B 评价，更新 B 时反过来；
推理使用两表之和。Q-table 的动作槽位是规范坐标，因此合法动作和实际动作都会通过
`world_to_canonical` / `canonical_to_world` 转换。危险但物理合法的动作不会被屏蔽。

## 用 experiments.run 训练、恢复和测试

推荐用该入口做正式实验，因为它会保存 metadata、训练 CSV、耗时、checkpoint 和恢复快照。

训练 Task 1：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode train --task 1 \
  --agent double_q_compact_agent --seed 11 --n-rounds 10000 \
  --run-id double_q_r3_t1_train
```

模型保存在 `runs/double_q_r3_t1_train/checkpoints/final.pkl`。

断点续训：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode train --task 1 \
  --agent double_q_compact_agent --seed 11 --n-rounds 10000 \
  --resume-from runs/double_q_r3_t1_train \
  --run-id double_q_r3_t1_resume
```

冻结评估：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json --mode evaluate --task 1 \
  --agent double_q_compact_agent \
  --checkpoint runs/double_q_r3_t1_train/checkpoints/final.pkl \
  --seed 10001 --n-rounds 20 --run-id double_q_r3_t1_eval
```

## 用 main.py 快速训练和测试

显式指定 checkpoint，可以避免在 Agent 源码目录覆盖 `final.pkl`。这里的最终文件名使用
`final.pkl`：

```bash
mkdir -p runs/manual_double_q/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_double_q/checkpoints/final.pkl" \
BOMBERMAN_FEATURE_ID=discrete-compact-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_q_compact_agent --train 1 \
  --scenario coin-heaven --seed 11 --n-rounds 3 --no-gui
```

加载模型进行无 GUI 测试：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_double_q/checkpoints/final.pkl" \
BOMBERMAN_FEATURE_ID=discrete-compact-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_q_compact_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 3 --no-gui
```

## 炸弹禁用与启用

`BOMBERMAN_ALLOW_BOMB` 是课程学习阶段的动作约束，不是 Feature 返回的物理
`legal_mask`。即使某处可以物理放弹，设为 `false` 后 Agent 也不会选择 `BOMB`；危险但
物理合法的移动仍不会被屏蔽。训练和评估同一个 Task 时应保持该值一致。

- 使用 `experiments.run --task 1` 时会自动设置为 `false`，无需手动添加环境变量。
- 使用 `experiments.run --task 2`、`3` 或 `4` 时会自动设置为 `true`。
- 直接调用 `main.py` 时必须自己明确设置。上面的 Task 1 命令就是禁用炸弹的完整示例。

直接运行 Task 2 并启用炸弹：

```bash
mkdir -p runs/manual_double_q_task2/checkpoints
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_double_q_task2/checkpoints/final.pkl" \
BOMBERMAN_FEATURE_ID=discrete-compact-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents double_q_compact_agent --train 1 \
  --scenario classic --seed 11 --n-rounds 3 --no-gui
```

## 打开 GUI

不要传 `--no-gui` 即可打开界面。

统一使用 `--agents` 显式指定参赛者数量。观察 Task 1 的单 Agent、禁用炸弹评估时只列出
一个 Agent。`coin-heaven` 只选择地图，不会自动限制玩家数量：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/double_q_r3_t1_train/checkpoints/final.pkl" \
BOMBERMAN_FEATURE_ID=discrete-compact-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=coin_navigation BOMBERMAN_ALLOW_BOMB=false \
python3 main.py play --agents double_q_compact_agent \
  --scenario coin-heaven --seed 10001 --n-rounds 1
```

观察 Task 2–4 的多人炸弹赛时，显式列出当前 Agent 和三个 `rule_based_agent`：

```bash
BOMBERMAN_CHECKPOINT="$PWD/runs/manual_double_q_task2/checkpoints/final.pkl" \
BOMBERMAN_FEATURE_ID=discrete-compact-v1 BOMBERMAN_REWARD_ID=r3_potential \
BOMBERMAN_TRAINING_TASK=crate_navigation BOMBERMAN_ALLOW_BOMB=true \
python3 main.py play --agents double_q_compact_agent \
  rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --seed 10001 --n-rounds 1
```

`--agents` 后写几个名称就启动几个参赛者。GUI 适合观察行为，不替代固定 seeds 的冻结评估。

## 单元测试

```bash
python3 -m unittest tests.test_new_agents -v
python3 -m unittest tests.test_submission_package -v
python3 -m unittest discover -s tests
```
