# 全局训练与评估框架参数

更新日期：2026-09-15

本文记录项目级实验协议，并单列当前推荐 Agent 训练链对全局默认值的覆盖。单次实验若有经过预注册的专用配置，以该配置及对应 ADR 为准。

## 1. 全局训练框架

### 独立训练重复

```text
training seeds = [11, 22, 33]
```

- 每个 training seed 是一条从 Task 1 开始的独立训练链，并产生独立 checkpoint。
- 训练入口一次只接受一个 `--seed`；三个 seed 需要运行三次，不能在同一次训练中途换 seed。
- training seed 同时控制 Agent RNG 和模型参数初始化。
- training seed 为 `S` 时，环境初始 seed 为 `1000 + S`，官方对手 RNG seed 为 `3000 + S`。
- 每个 training seed 单独报告结果，再报告三个 seed 间的均值与样本标准差。

### 四阶段课程预算

| Task | 场景与对手 | 每个 training seed 的新增训练局数 | 性能失败时最多追加 |
| --- | --- | ---: | ---: |
| 1 | `coin-heaven`，无对手，禁用 `BOMB` | 500 | 125 |
| 2 | `classic`，无对手 | 1,000 | 250 |
| 3 | `classic`，同时对抗 `peaceful_agent` 和 `coin_collector_agent` | 1,500 | 375 |
| 4 | `classic`，对抗三个 `rule_based_agent` | 3,000 | 750 |

一条完整训练链的基础预算为 `6,000 rounds/training seed`；三个 training seeds 的单模型族基础总预算为 `18,000 rounds/model family`。

链内必须按照 `Task 1 → Task 2 → Task 3 → Task 4` 串行训练。后续阶段通过 `--resume-from` 继承同一个 training seed 的父 checkpoint；不同 seed 之间不得交叉续训。

### 初始化与 Warm start

| 阶段 | 初始化方式 | 来源 |
| --- | --- | --- |
| Task 1 | Cold start（从零训练） | 使用当前 training seed 初始化模型、优化器和随机状态 |
| Task 2 | Warm start | 同一模型、同一 training seed 的 Task 1 checkpoint |
| Task 3 | Warm start | 同一模型、同一 training seed 的 Task 2 checkpoint |
| Task 4 | Warm start | 同一模型、同一 training seed 的 Task 3 checkpoint |

项目中的正式课程 warm start 使用 `--resume-from`。它不仅加载模型权重，还恢复完整训练状态，包括 optimizer、探索进度、全局动作数、Agent RNG，以及算法所需的 replay/trace 状态；进入下一 Task 时，再按照预注册的阶段配置重置允许重置的阶段状态。

`--init-from-checkpoint` 只表示用已有 checkpoint 初始化模型，不等同于可复现的完整续训。它适合明确登记的迁移或实验，不应替代正式课程链的 `--resume-from`。

Warm start 必须满足以下约束：

- 只能在同一 training seed 内进行，禁止 seed 11 的 Task 2 接 seed 22 的 Task 1 checkpoint。
- 只允许同一 Task 续训，或直接按 `1→2→3→4` 晋级，不能跳阶段。
- 算法、动作顺序、Feature、Reward、Safety、设备和 checkpoint schema 必须兼容。
- 源码或行为修复改变公平比较条件后，受影响的训练链必须从 Task 1 cold start，不能沿用旧 checkpoint。
- 三个 training seeds `11/22/33` 的 Task 1 都必须独立 cold start；warm start 不会把三个 seed 合并成一个模型。

### 通用训练参数

| 参数 | 正式值 |
| --- | --- |
| Device | CPU |
| 每个训练进程线程数 | 1 |
| 并行方式 | 不同训练链可并行；同一条链的阶段串行 |
| Exploration schedule | `linear-v1` |
| 初始 epsilon | `1.0` |
| 最终 epsilon | `0.05` |
| 衰减预算 | `80,000 action steps` |
| Reward-based early stopping | 关闭 |
| Replay 保存 | 第 1 局及约每 10% 进度保存 |
| Resume 保存 | 每回合提交，保留最新两代完整 generation |

训练局数是每个 training seed 的完整预算。例如 Task 1 的 500 局表示 seed 11、22、33 各训练 500 局，而不是三者合计500局。

### 当前推荐训练链的 Task 1 性能停止

以下专用配置不使用上表的固定 500 局 Task 1 默认预算，而使用
`task1-frozen-score-v1` 性能停止协议：

- `double_q_lambda_task1.json`
- `expected_sarsa_lambda_task1.json`
- `rainbow_lite_task1.json`
- `rainbow_lite_fallback_task1.json`

这不是基于 shaped training reward 的普通 early stopping。两种机制的当前状态为：

```text
training.early_stopping.enabled = false
training.performance_stopping.enabled = true
```

性能监控参数如下：

| 参数 | 当前值 |
| --- | --- |
| 监控 environment seeds | `9000–9019`，共 20 个 |
| 每个 seed 的评估局数 | 1 |
| 是否探索/学习 | 否；冻结 checkpoint，关闭探索和参数更新 |
| 首次评估 | 累计训练第 200 局 |
| 后续间隔 | 每新增 50 个训练回合 |
| 通过门槛 | 20 局 `mean_score >= 48` |
| 停止要求 | 连续通过 3 次 |
| 最早停止点 | 第 300 个训练回合 |
| 最大训练预算 | 1,000 个训练回合 |

一次监控失败会把连续通过次数归零。监控 seed 只用于判断是否停止，不得代替独立阶段验收；
阶段验收继续使用未参与停止的 `10000–10019`。该协议遵循
[`docs/adr/0002-use-frozen-score-for-task1-convergence.md`](../docs/adr/0002-use-frozen-score-for-task1-convergence.md)。

Task 2 不启用性能停止或 reward-based early stopping。当前推荐配置仍要求至少 500 个训练
回合且达到 150,000 个阶段动作，最多 2,000 回合。

### 当前 Reward 版本冻结

- no-safety 的最终新训练合同固定为 `r16_no_safety_locked`；它冻结复制
  `r15_no_safety_objective_credit` 的参数。r11–r15 仅保留用于旧 checkpoint 复现，
  不再作为新的迭代分支。
- Safety 箱区导航实验使用 `continuous-v5 + r17_global_crate_bomb_discipline`，
  不修改已有 `continuous-v4 + r10` checkpoint 合同。

## 2. 全局评估框架

所有评估必须冻结 checkpoint、关闭探索和参数更新，并在 CPU 上运行。不同候选必须复用相同的 environment seeds、对手、场景和局数。

### 阶段门槛评估

```text
environment seeds = [10000, 10001, 10002, 10003, 10004]
rounds per seed = 20
total = 5 × 20 = 100 rounds/checkpoint/task
```

- 每个 training seed 产生的 checkpoint 都要独立评估。
- 三个 training seeds 全部评估时，一个模型在一个 Task 上共有 `3 × 100 = 300` 局阶段门槛评估。
- 父 checkpoint 与子 checkpoint 必须使用完全相同的 seeds 和每-seed局数，以便配对比较。

### 阶段能力门槛

| Task | 主要能力门槛 |
| --- | --- |
| 1 | 平均金币至少比同-seed的 `legal_random_agent` 高 2 |
| 2 | 平均炸箱数至少比 Task 1 父模型高 0.5 |
| 3 | 平均击杀数至少比父模型高 0.1，或独占/并列第一率提高至少 5 个百分点 |
| 4 | 平均击杀数至少比父模型高 0.1，或独占/并列第一率提高至少 5 个百分点 |

### Task 2 严格质量门槛（新候选）

从 2026-09-15 起，新 Task 2 候选还必须通过
[`task2_quality_gate.json`](task2_quality_gate.json)。`classic` 每局共有 9 枚金币，
因此旧实验中的 `mean_coins >= 2` 只保留为历史安全消融合同，不再作为新候选的
能力合格线。

| 指标 | 新门槛 |
| --- | ---: |
| 平均金币 | `>= 6.0 / 9` |
| 零金币局率 | `<= 5%` |
| 9金币完成率 | `>= 10%` |
| 平均炸箱 | `>= 60` |
| 每100步金币 | `>= 1.5` |
| 跑满400步比例 | `<= 90%` |
| 长 WAIT 循环率 | `<= 10%` |
| 长往返循环率 | `<= 5%` |
| 自杀率 | `<= 5%` |
| 零放弹局率 | `<= 10%` |
| 安全但不命中箱子或对手的放弹率 | `<= 10%` |
| 每颗炸弹炸毁箱子数 | `>= 1.5` |
| 放弹存活率 | `>= 95%` |
| Task 1 分数保留率 | `>= 90%` |

这些门槛只使用关闭探索和学习的冻结评估计算，而且三个 training-seed checkpoint
必须分别通过。训练轨迹只能用于诊断，不能代替阶段验收。

旧 Task 保留要求：晋级模型在所有旧 Task 上的 `mean_score` 不得低于父模型的 90%。有炸弹阶段的自杀率不得比父模型高 10 个百分点，且绝对自杀率不得超过 35%。

### 通用工程硬门槛

- 评估 run 完整，实际局数与配置一致。
- 冻结 checkpoint 能成功加载。
- 最新两代 resume generation 的 hash 校验通过。
- 学习状态确实更新：Q 表状态数增长，或神经网络 optimizer updates 大于 0 且 loss 有限。
- 无异常、超时或框架跳过动作。
- 完整 `act` P95 小于 `50 ms`，最大值小于 `500 ms`。
- Task 1 不得执行 `BOMB`。
- 无效动作率不超过 `1%`。

### 主验证

完成 Task 4 后，对每个主候选执行：

```text
environment seeds = 10000–10099
rounds per seed = 1
total = 100 rounds/checkpoint
opponents = 3 × rule_based_agent
```

先对每个模型族的三个 training-seed checkpoints 汇总 `mean_score`，选择模型族；再在胜出模型族内选择单个 checkpoint。选择顺序以预注册协议为准，不能看完结果后更改。

### 最终测试

只对冻结的唯一胜者执行：

```text
environment seeds = 20000–20099
rounds per seed = 1
total = 100 rounds
```

这些 seeds 在最终选模前必须保持未使用。最终测试结果只用于报告，不得触发重新训练、调参或重新选模。

## 3. 评估指标

- `mean_score`
- coins、crates、kills、suicides
- 独占第一率、并列第一率、全员零分率
- 生存率和生存步数
- invalid-action rate
- 完整 `act` 的 median、P95、max
- 超过 0.5 秒次数和框架 skipped-action 次数

Task 1 等单智能体能力场景不报告“胜率”，而报告金币、完成率、效率与死亡情况。

## 4. 统计单位与总局数

- **Training seed** 是独立训练重复单位。
- **Environment seed** 是冻结模型的测试环境重复单位。
- 同一 checkpoint 的100个评估环境不能写成100次独立训练。
- 每个 training seed 的指标应先独立报告，再跨 training seeds 汇总均值和样本标准差。

以 Task 1 为例：

```text
训练：3 training seeds × 500 rounds = 1,500 training rounds
阶段评估：3 checkpoints × 5 environment seeds × 20 rounds = 300 evaluation rounds
```

## 5. 输出与追溯

每个实验进程独占 `runs/<run_id>/`，至少保存：

- `metadata.json`：展开配置、源码身份、依赖、硬件、seed、状态
- `episodes.jsonl`：逐局原始指标
- `training.csv` 与 `training_summary.json`：训练过程
- `timing.jsonl`：完整动作耗时
- `official_stats.json`：官方统计交叉检查
- `checkpoints/`：当前 run 的模型文件
- `resume/`：可恢复的训练快照

## 6. 全局配置来源

- `experiments/configs/formal_training.json`：正式训练 seeds、课程预算和通用训练参数
- `experiments/configs/stage_gate.json`：阶段门槛评估
- `experiments/configs/main_validation.json`：100-seed主验证
- `experiments/configs/final_test.json`：保留的100-seed最终测试
- `IMPLEMENTATION_GUIDE.md` 第 7–8 节：随机性、指标、门槛、选模与输出协议

专用实验配置可以覆盖上述默认值，但必须保留独立配置和决策记录，不能把专用参数默认为全局参数。
