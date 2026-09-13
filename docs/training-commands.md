# 六个学习型 Agent 训练命令

> 2026-09-13：当前 Task 3 前入口已切换到
> [`experiments/pre_task3_iteration.json`](../experiments/pre_task3_iteration.json)。下方原六模型命令仅作历史参考，不得替代预注册三轮实验或接续 v4 run。

当前第一轮只使用 `q_learning_agent` 与 `dqn_agent`，以 seed 11 比较
`discrete-v1/discrete-objective-v1 × r1_coin3/r5_coin_potential` 共 8 条链。每条命令使用
`experiments/configs/pre_task3_task1.json`、`--n-rounds 500`、
`--target-stage-action-steps 100000 --min-rounds 1 --device cpu`，run ID 遵守
`iter_r<round>_<agent>_<feature>_<reward>_s<seed>_t<task>_a<target>_<commit>`。

各算法前两名才用 `--resume-from` 累计到 200k 动作；Task 1 mean coins ≥35 后才进入
Task 2。Task 2 使用 `pre_task3_task2*.json`，至少 500 局且 150k 动作、最多 2000 局。
第三轮能力失败才启用 `double_q_agent` 或 `double_dqn_continuous_v2_agent`。完整分支、门槛与
seed 隔离见实施指南第 8.6 节。当前完整恢复协议为 v5；v4 及更早 run 只可冻结评估。

所有命令都应从仓库根目录运行。`experiments.run` 会自动选择 Agent 的 `feature_id`，Task 1
自动禁止炸弹，Task 2–4 自动允许炸弹，并保存 metadata、CSV、checkpoint 和恢复快照。

## 训练加速说明

下列优化默认生效，不需要修改训练命令：

- 五个首动作的时间展开逃生搜索采用一次批量传播，输出与逐动作搜索一致；
- `board-v1` 只计算棋盘通道需要的危险图和合法动作，不再额外运行目标 BFS；
- `r3_potential` 只计算 potential 所需的金币/箱子距离和当前格危险；
- 神经网络按连续 batch 从 replay 采样，checkpoint 将 replay 保存为少量列式 Tensor；
- 实验 CSV 采用增量读取，神经 checkpoint 的两代恢复快照优先使用硬链接；不支持硬链接的
  文件系统会自动回退到复制；
- 逐步 timing 文件复用已打开的流，普通动作日志降为 debug，超时警告仍保留。

这些改动没有减少 replay 容量、batch size、更新频率、训练局数或 checkpoint 的回合边界，
也没有改变 Feature/Reward 数值、epsilon、网络、Q 更新和随机采样顺序。旧 replay checkpoint
仍可加载；新 checkpoint 使用更快的列式 replay 内部格式。

## 历史六模型组合（非当前 Task 1–2 入口）

| Agent | Feature | Reward | Checkpoint |
|---|---|---|---|
| `q_learning_agent` | `discrete-q-v2` | `r4_anti_oscillation` | `final.pkl` |
| `double_q_compact_agent` | `discrete-compact-v1` | `r3_potential` | `final.pkl` |
| `dqn_agent` | `discrete-q-v2` vector | `r4_anti_oscillation` | `final.pt` |
| `double_dqn_continuous_agent` | `continuous-v1` | `r2_balanced` | `final.pt` |
| `cnn_double_dqn_agent` | `board-v1` | `r3_potential` | `final.pt` |
| `hybrid_dueling_double_dqn_agent` | `hybrid-v1` | `r2_balanced` | `final.pt` |

`experiments/configs/reward_r2_balanced.json` 选择 `r2_balanced`；需要 `r3_potential` 时使用
`experiments/configs/reward_r3_potential.json`。
Q-learning 的防摆动训练使用 `experiments/configs/reward_r4_anti_oscillation.json`。

## Agent 1：Q-learning

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent q_learning_agent --seed 11 --n-rounds 1000 \
  --run-id q_learning_r4_t1_train
```

输出：`runs/q_learning_r4_t1_train/checkpoints/final.pkl`

## Agent 2：Compact Double Q-learning

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json \
  --mode train --task 1 --agent double_q_compact_agent \
  --seed 11 --n-rounds 10000 --run-id double_q_r3_t1_train
```

输出：`runs/double_q_r3_t1_train/checkpoints/final.pkl`

## Agent 3：DQN

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r4_anti_oscillation.json --mode train --task 1 \
  --agent dqn_agent --seed 11 --n-rounds 1000 \
  --run-id dqn_r4_t1_train
```

输出：`runs/dqn_r4_t1_train/checkpoints/final.pt`

## Agent 4：Continuous Double DQN

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent double_dqn_continuous_agent --seed 11 --n-rounds 1000 \
  --run-id continuous_ddqn_r2_t1_train
```

输出：`runs/continuous_ddqn_r2_t1_train/checkpoints/final.pt`

## Agent 5：CNN Double DQN

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json \
  --mode train --task 1 --agent cnn_double_dqn_agent \
  --seed 11 --n-rounds 1000 --run-id cnn_ddqn_r3_t1_train
```

输出：`runs/cnn_ddqn_r3_t1_train/checkpoints/final.pt`

## Agent 6：Hybrid Dueling Double DQN

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r2_balanced.json --mode train --task 1 \
  --agent hybrid_dueling_double_dqn_agent --seed 11 --n-rounds 1000 \
  --run-id hybrid_dueling_r2_t1_train
```

输出：`runs/hybrid_dueling_r2_t1_train/checkpoints/final.pt`

## 从 Task 1 继续到 Task 2

课程阶段之间必须保持 Agent、Feature 和 Reward 一致，并通过 `--resume-from` 指向父 run。
例如继续训练 Double Q：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json \
  --mode train --task 2 --agent double_q_compact_agent \
  --seed 11 --n-rounds 10000 \
  --resume-from runs/double_q_r3_t1_train \
  --run-id double_q_r3_t2_train
```

Task 2 输出：`runs/double_q_r3_t2_train/checkpoints/final.pkl`

其他 Agent 按同样方式替换 Agent 名、配置、父 run 和新 run ID。不能从 `r1` checkpoint
切换到 `r2_balanced` 或 `r3_potential` 后继续训练；Reward 不同必须建立全新的训练链。

## 冻结评估

评估必须使用与 checkpoint 相同的 Reward 配置。以 Double Q 为例：

```bash
python3 -m experiments.run \
  --config experiments/configs/reward_r3_potential.json \
  --mode evaluate --task 1 --agent double_q_compact_agent \
  --checkpoint runs/double_q_r3_t1_train/checkpoints/final.pkl \
  --seeds 10001 10002 10003 10004 10005 --n-rounds 20 \
  --run-id double_q_r3_t1_eval
```

不要用不同 Reward 的训练 reward 数值直接排名。最终比较应使用冻结评估中的官方得分、
金币数、击杀数、存活率、非法动作数和每步推理时间。

## 注意事项

- `--run-id` 必须唯一；Runner 不会覆盖已有 run。
- Task 1 自动设置 `BOMBERMAN_ALLOW_BOMB=false`，Task 2–4 自动设置为 `true`。
- 表格模型输出 `final.pkl`，PyTorch 模型输出 `final.pt`。
- 当前配置启用了早停；`--n-rounds` 是最大预算，不保证一定跑满。
- GUI 和直接 `main.py` 命令见各 Agent 目录下的 README。
