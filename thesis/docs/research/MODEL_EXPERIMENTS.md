# 四类模型的已完成实验与实验思路

> 用途：为论文的 Methods、Training、Experiments and Results 提供可核验素材。
> 本文只记录仓库中已经实现或已有明确实验记录的内容；配置文件存在不等于实验已经完成。
> 课程要求的仓库内摘要见 [`PROJECT_REQUIREMENTS.md`](../../../PROJECT_REQUIREMENTS.md)。原始课程文件由课程平台提供。

## 1. 证据口径与共同评价原则

### 1.1 证据状态

| 状态 | 含义 |
|---|---|
| `verified_raw` | 有逐局数据、JSON/CSV、冻结 checkpoint、hash 或独立审计，可以回算主要结论。 |
| `verified_summary` | 有实验日志或研究总结，但本文未直接从完整逐局文件重算。 |
| `implemented_not_evaluated` | 代码或配置已经实现，未找到完成的性能评估。 |
| `planned_or_skipped` | 仅有实验计划，或流水线按预注册条件明确跳过。 |
| `not_comparable` | 场景、对手、训练预算、seed、checkpoint 或统计单位不同，不能直接排序。 |
| `not_run` | 没有执行该阶段；不得把空缺写成零。 |

证据优先级为：原始结果与 checkpoint 审计 > 模型实验日志 > 研究汇总 > 配置和代码。
本文中的训练 seed 与环境 seed 分开表述；同一 checkpoint 在 100 个环境世界上的表现不是
100 次独立训练。

### 1.2 模型合同

一个可比较的 Agent 不只是 learner 名称，而是完整合同：

```text
Agent = Feature + Learner/Network + Reward + Survival Mask + Training/Evaluation Protocol
```

因此，不同完整组合之间的差异不能自动归因于其中某一个组件。只有其余条件固定、单独改变
一个轴时，本文才称其为较强消融。

### 1.3 课程阶段与指标

- Task 1：无箱子、无对手的金币导航。主要看平均金币、全收集率、完成步数、WAIT/循环和延迟。
- Task 2：有箱子、无对手。增加炸箱、每弹炸箱、放弹存活、自杀、零放弹、Task 1 保留。
- Task 3：对 `peaceful_agent`、`coin_collector_agent` 等弱对手。增加正式得分、击杀、第一名和旧任务保留。
- Task 4：对 `rule_based_agent` 或自有强对手。重点是正式得分、击杀、第一名、生存和完整 `act` 延迟。

训练 loss 和 shaped reward 只用于诊断，不替代冻结、关闭探索后的正式游戏指标。

## 2. Q-learning 模型族

### 2.1 模型范围和研究思路

该模型族从最小离散 Q-learning 基线逐步扩展到 Double Q、eligibility trace 与 tile coding：

- `q_learning_agent`：单表离散 Q-learning，用于证明最小学习系统能够学习 Task 1。
- `double_q_agent`、`double_q_compact_agent`：以双估计器减少最大化偏差，并探索不同离散状态压缩。
- `double_q_lambda_agent`：加入 Watkins trace，使延迟结果沿近期状态传播。
- `optimized_double_q_lambda_agent`：主线版本；把 84 维 `continuous-v2` 按动作映射到多组 tile，
  使用 Watkins Double Q($\lambda$)。

主线超参数记录为 $\gamma=0.95$、$\lambda=0.8$。Task 1 早期确认使用
`r7_safe_credit_potential`；正式 Task 2 搜索转向 r20 的窄条件 WAIT 信号，并固定
`survival-mask-v1/all`。核心问题是：表格方法能否借局部泛化、历史和 trace 从高质量导航迁移到
安全炸箱及箱后目标重选。

主要来源：

- [`optimized_double_q_lambda_agent/EXPERIMENT_LOG.md`](../../../agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)
- [`q-cnn-experiment-report.md`](../../../docs/research/q-cnn-experiment-report.md)
- [`q-learning-task1-optimization.md`](../../../docs/research/q-learning-task1-optimization.md)

### 2.2 Task 1：单表 Q-learning 与 Watkins Double Q($\lambda$)

#### T1-E01：学习器比较

- **问题与假设**：双估计器、tile coding 和 eligibility trace 能否比单表 Q-learning 更稳定地完成 50 金币导航。
- **基线与改动**：单表 Q-learning 对比 Watkins Double Q($\lambda$)；训练 seed 11，各约 100k action steps。
- **组合**：主线为 `continuous-v2 + r7_safe_credit_potential + mask-v1/off`。
- **协议**：开发集 5 个环境 seed、每 seed 20 局；候选选定后在 `11000--11099` 做 100 局独立确认。
- **结果**：单表基线最佳快照全收集率 71%、平均金币 47.98；Double Q($\lambda$) 开发集最佳快照
  全收集率 96%、平均金币 49.51。固定该 checkpoint 后，独立确认仍为 96% 全收集、平均金币
  **49.58**、循环率 0%，`act` P95/max 为 7.83/11.63 ms。
- **结论**：在这一协议下，优化的 Double Q($\lambda$) 通过 Task 1；训练过程并非单调，50k 快照曾退化。
  这不能证明 Double Q($\lambda$) 在所有表示或任务上普遍优于 Q-learning。
- **状态**：`verified_summary`；实验日志给出 checkpoint SHA-256 和作业号。

注意：[`table2_task1_task2.csv`](../../report-assets/tables/table2_task1_task2.csv) 当前把该独立确认均值写为
48.95，而模型实验日志写 49.58。本文采用更接近原实验的模型日志口径；正式论文应在引用前从
原始 `selection.json`/逐局结果再核对并统一这两个数字。

#### T2-L01：为 Task 2 重建兼容父链

早期通过的 Task 1 checkpoint 使用 mask-off，不能直接作为 mask-all 的普通 resume 父模型。
因此从零重建相同 learner、`continuous-v2`、`r7_safe_credit_potential`，只把 safety 固定为
`survival-mask-v1/all`。训练在 300 局、99,845 actions 达到三次连续 50/50，独立 20 局 stage gate
仍为 50/50，非法动作 0%，P95/max 11.35/14.44 ms。该实验说明 checkpoint 迁移必须遵守完整合同，
而不是只看网络或 Q 表形状。状态为 `verified_summary`。

### 2.3 Task 2：安全炸箱成功，但长期目标重选失败

#### T2-P01：正式 Task 2 pilot

- **问题与假设**：Task 1 的 tile-coded 导航策略能否在保持旧能力的同时学习安全放弹和隐藏金币收集。
- **组合**：Double Q($\lambda$) + `continuous-v2` + `r7_safe_credit_potential` + `mask-v1/all`。
- **训练**：seed 11，从已审计 Task 1 父链继续，完成 500 局、200k Task 2 actions；每 25k actions 冻结评估。
- **最佳快照**：175.2k；20 局开发结果为 3.60/9 金币、54.2 箱、0% 自杀、100% Task 1 保留，
  但长 WAIT 70%、长往返 60%。
- **结论**：即时安全和炸箱已学到，但金币、箱子和循环质量门未通过，不能晋级。
- **状态**：`verified_summary`。

#### T2-R20：窄条件 WAIT 惩罚

- **问题与假设**：WAIT 并非 Q 值并列或 mask 大量否决造成；当状态安全且存在可行推进动作时，
  轻微惩罚 WAIT 可能减少吸引子。
- **唯一改动**：从 r7 potential 增加 `targeted_wait=-0.04`；feature、tile、$\lambda$、学习率和 mask 不变。
- **结果**：100k 后平均金币 3.70、炸箱 50.25、长 WAIT 45%、长往返 70%、自杀 0%。继续到最佳
  175.2k 后，整合报告记录 3.95 金币、53.45 箱、长 WAIT 30%、长往返 75%、自杀 0%、放弹存活 100%。
  19 个 Task 2 门槛只通过 12 个。
- **结论**：WAIT 有所缓解，但失败转移为往返，仍未解决箱后目标重选；不启动 seeds 22/33。
- **状态**：`verified_summary`。

#### Task 2 失败分支

| 分支 | 主要变化 | 20 局冻结观察 | 结论 | 状态 |
|---|---|---|---|---|
| crate quantized | 箱子三值量化 | 1.35 金币，11/19 gates | 未超过 r20 | `verified_summary` |
| history input | 加强历史输入 | 1.65 金币、29.70 箱、长往返 85% | 历史本身未解决目标重选 | `verified_summary` |
| continuous-v4/r12 | 显式循环历史与更强反循环 reward | 0.95 金币、18.40 箱、长 WAIT 0%、往返 35% | 停滞下降但任务能力同时下降 | `verified_summary` |
| grouped tiles | 改 tile 分组 | 0 金币、0 箱，5/19 gates | 明显失败 | `verified_summary` |
| team-demo 50k | 团队示范的一步离线 Double Q 更新 | 3.75 金币、56.90 箱、长 WAIT 65%、往返 75% | 炸箱提高但停滞恶化 | `verified_summary` |
| T2-R21 | r20 上仅加条件折返惩罚 | 日志状态为 submitted，未见完成结果 | 不能写成已完成 | `planned_or_skipped` |

### 2.4 阶段结论

Q-learning 路线证明了表格型方法在适当表示和 trace 下可成为强 Task 1 基线；Task 2 的主要失败不是
自杀，而是高维组合泛化与长期目标维持。Task 3/4 均为 `not_run`。报告中不能用其 Task 1 成绩推断
对战能力，也不能把已创建的 Task 2 配置写成已完成实验。

## 3. Continuous Double DQN 模型族

### 3.1 模型范围和研究思路

该路线以连续特征向量输入小型 MLP，使用经验回放、target network 和 Double DQN bootstrap。
相关变体包括：

- `double_dqn_continuous_agent`：70 维 continuous-v1 基线。
- `double_dqn_continuous_v2_agent`：84 维历史增强表示，形成 Task 2 winner 和 Task 3 validated lineage。
- `double_dqn_continuous_v3_agent`：显式安全特征实验。
- `double_dqn_continuous_v4_agent`：循环历史实验。
- `double_dqn_phase_agent`：117 维 phase-aware Task 3 分支。
- B33/Die Hardest：`continuous-v2 + r7_safe_credit_sparse + survival-mask-v9` 的最终交付候选。

这条路线的实验主线不是“网络越复杂越好”，而是逐阶段验证导航、炸箱、生存、旧任务保持和
对手压力下的正式得分。

主要来源：

- [`task2_winner.json`](../../../experiments/task2_winner.json)
- [`task2_winner_evaluations.csv`](../../../experiments/task2_winner_evaluations.csv)
- [`task3-counter-validation-results.md`](../../../docs/research/task3-counter-validation-results.md)
- [`task4-frozen-results.md`](../../../docs/research/task4-frozen-results.md)
- [`table4_task4_attempts.csv`](../../report-assets/tables/table4_task4_attempts.csv)
- [`die-hardest-submission/REPORT.md`](../../../docs/research/die-hardest-submission/REPORT.md)

### 3.2 Task 1：连续表示基线筛选

早期探索同时改变了表示、learner、reward 和训练预算，因此只能作为完整方案筛选。
Continuous Double DQN 历史 Task 1 结果达到约 49.52/50 和 86% 全收集率；Hybrid Dueling 分支
仅约 2.05/50，说明增加网络复杂度并不自动改善学习。由于这些不是严格单变量实验，不能把全部差异
归因于 Double DQN 或连续表示。状态为 `verified_summary`，来源为
[`agent-code-timeline-and-experiment-narrative.md`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md)。

### 3.3 Task 2：正式 winner 与 survival mask

#### Task 2 winner

- **问题与假设**：84 维历史特征、Double DQN 与统一 survival mask 能否从导航迁移到安全炸箱。
- **组合**：continuous-v2 主线，r7 family，mask 在探索、训练贪心、冻结推理和 bootstrap 中保持一致。
- **协议**：三个训练 seed 通过开发门后，只对固定候选执行一次独立主验证。
- **结果**：100 局主验证平均 **7.25/9 金币、100.26 个箱子、0% 自杀**，并保持 100% Task 1 能力。
- **结论**：该完整合同通过项目 Task 2 晋级门；不能仅归因于 mask、reward 或 v2 中任一项。
- **状态**：`verified_raw`，结构化结果见上述 winner JSON/CSV。

这一阶段还形成 continuous-v3/v4 等安全与历史消融。它们帮助诊断安全余量与循环，但现有汇总没有
提供一套覆盖全部变体、相同训练预算和相同世界的完整纯 Feature 消融，因此相关效果只能谨慎描述。

### 3.4 Task 3：生命周期修复与配对主验证

早期 pilot 暴露两个实现层问题：训练回调读取可变历史会破坏 transition 的决策快照；安全 fallback
计数曾把物理合法 WAIT 错当作安全替代动作。修复计数语义时没有修改权重、策略、reward、feature 或门槛，
并通过世界 19489 的逐动作等价回放和测试。

固定 seed22/c150 后，在相同世界中对 Task 2 父模型与 Task 3 子模型做配对比较：

| 指标 | 父模型 | 子模型 | 差值与区间 |
|---|---:|---:|---:|
| 正式得分 | 5.91 | 7.14 | +1.23，95% paired bootstrap CI [0.27, 2.14] |
| 金币 | 3.66 | 4.94 | +1.28，[0.87, 1.70] |
| 炸箱 | 43.78 | 58.97 | +15.19，[12.17, 18.16] |
| 击杀 | 0.45 | 0.44 | -0.01，[-0.16, 0.14] |
| 第一名率 | 41% | 53% | +12 个百分点 |

三个训练 seed 的确认集得分增量分别为 +1.92、+1.23、+2.56。结论是 Task 3 训练提高了总体得分，
主要来自金币和炸箱；**没有证据表明击杀提高**。状态为 `verified_raw`，每阶段 100 个配对世界，
bootstrap 10,000 次。

### 3.5 Task 4：专项训练、工程准入与停止

#### E1/E2/E3 specialist 探索

| 尝试 | 训练 seed | 端点评分 | 击杀 | 自杀/炸弹存活 | 完整 act P95 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| E1 | 11 | 4.06 | 0.29 | 0% / 100% | 19.98 ms | 无稳定增益 |
| E1 | 22 | 4.10 | 0.26 | 0% / 100% | 19.12 ms | 无稳定增益 |
| E1 | 33 | 3.27 | 0.16 | 0% / 100% | 22.25 ms | seed 间波动明显 |
| E2 | 22 | 2.34 | 0.09 | 0% / 100% | 19.61 ms | 更差端点 |
| E3 | 22 | 2.79 | 0.15 | 0% / 100% | 18.92 ms | 无稳定增益 |

这些实验表明局部安全能够保持，但更复杂的 replay/reward/specialist 方案未形成稳定得分提升。
Frozen C/S 使用另一协议，只能作工程观察；Score L/P/K 因 worker failure 未完成正式比较。
状态分别为 `verified_raw`、`not_comparable` 和 `not_run`。

#### Survival mask v6--v9 工程线

v6 严格要求新放 BOMB 必须有完整证明；v7 加入对手重新获得放弹能力；v8 固定放弹证明终点；
v9 偏好有完整对手条件证明的移动，并通过 reach-grid/compact exact 优化尝试满足 CPU 时限。
这些工作修复了登记反例并完成大量保持性和工程测试，但多次出现终止性工程失败：例如 v9 世界 24025
完整搜索超过预算，后续新世界 24384 再次出现 408.911 ms 搜索超时；compact admission 又以
262.603 ms 超过项目 250 ms margin gate。它们是安全工程证据，不是新的 Task 4 qualified model。

#### B33 / Die Hardest 回退选择

新 Task 4 方案没有稳定胜出，因此选择历史证据最完整的 Task4 B seed33/c200（B33）：

- 组合：Continuous Double DQN、84 维 `continuous-v2`、`r7_safe_credit_sparse`、`survival-mask-v9`。
- 历史 1,000-world benchmark：平均得分 4.231、含并列第一率 50.2%、独占第一率 37.8%、存活率 94.7%。
- 重命名为 Die Hardest 后：180 条 trace 等价检查和 6 局配对打包检查通过；完整 `act` P95/max
  为 14.826/44.601 ms，峰值 RSS 319.645 MiB。
- 已知限制：这不是重命名后的 1,000 局重跑；Docker/官方 CPU 未完成；世界 27155 的安全反例仍能复现。

因此 B33 是当前证据与交付约束下的工程选择，不是统一协议下已经证明的算法冠军。历史 benchmark
为 `verified_summary`，包等价性为 `packaging_verified`。

## 4. CNN Double DQN 模型族

### 4.1 模型范围和研究思路

CNN 路线使用 17 通道 `board-path-history-v2`，通道包括棋盘对象、炸弹/爆炸危险、到金币/箱子/对手
的距离场、上一位置和最近访问热图。网络为无降采样残差 CNN，使用 Double DQN。后期通过团队内部
冻结的 Continuous Double DQN teacher 做 masked-KL 蒸馏；student 正式推理不加载 teacher 或数据集。

主要来源：

- [`cnn-task1-task2-experiment-report.md`](../../../docs/research/cnn-task1-task2-experiment-report.md)
- [`cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md`](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)

### 4.2 Task 1：输入修复、结构筛选与蒸馏

早期 self channel 因变量复用标错自身位置，因此旧 feature-v1 结果不能作为可信奖励或结构比较。
修复为 17 通道 feature-v2 后：r3 最佳平均金币 13.35；r5 conditional-loop 在约 75k steps 达到
17% 全收集、41.78 金币；4-step r5 未改善后期退化。

T1-D01 改用团队 teacher 蒸馏：

- 数据为 seeds 6000--6099，共 12,825 rows；训练 seed 11。
- 比较 global、action-aligned、action-aligned+D4 三种 student head。
- global pretrained 在开发 100 局达到 97% 全收集、49.96 金币；reserved 100 局为 96%、49.84。
- 10k/25k/50k TD 微调后全收集率依次下降到 80%/72%/55%，WAIT 上升；较低 KL loss 也未保证闭环更优。
- 最佳 checkpoint CPU `act` P95/max 为 8.39/15.55 ms。

结论是蒸馏成功迁移了 Task 1 行为，但在线 TD 微调会破坏它；这不是 CNN 从零学习达到相同结果的证据。
状态为 `verified_summary`，日志同时保存数据集和 checkpoint hash。

### 4.3 Task 2：迁移、WAIT shaping 与 safety 反事实

#### T2-D01：带 Task 1 KL 的迁移

- **组合**：17 通道 global CNN、r7 safe credit sparse、4-step、`mask-v1/all`、KL=2。
- **协议**：seed 11，200k actions/500 局；另有相同预算的随机初始化对照。
- **结果**：迁移 200k 保持 Task 1 50.0，但 Task 2 仅 2.00 金币、34.35 箱、20% 零放弹、
  WAIT 55.20%、长 WAIT 75%；自杀 0%、炸弹存活 100%。从零模型 Task 2 略高，但 Task 1 只剩 2.4。
- **结论**：D01 没通过；迁移保护旧能力，但策略出现 WAIT 两极化。
- **状态**：`verified_summary`。

#### T2-D02：可避免 WAIT 的单变量实验

D02 是仓库中较清晰的 reward 消融：唯一训练变化是当 WAIT 安全且存在安全推进移动或有效 BOMB 时，
增加 `-0.04`；网络、feature、KL、mask、4-step、探索日程和预算保持不变。

| 指标 | D01 200k | D02 seed11 | D02 seed22 | D02 seed33 | D02 主验证 100 局 |
|---|---:|---:|---:|---:|---:|
| Task 1 平均分 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| Task 2 金币 | 2.00 | 2.80 | 4.20 | 3.65 | 2.60 |
| 炸箱 | 34.35 | 54.20 | 67.25 | 57.35 | 44.15 |
| 零放弹率 | 20% | 0% | 0% | 0% | 0% |
| WAIT 率 | 55.20% | 17.26% | 14.90% | 25.69% | 22.56% |
| 自杀/炸弹存活 | 0%/100% | 0%/100% | 0%/100% | 0%/100% | 0%/100% |

该改动显著改善炸箱与 WAIT，并在三训练 seed 后完成 100-seed 主验证；但主验证全金币率仍为 0%，
所以只能称为安全炸箱迁移通过，不能称 Task 2 完成。状态为 `verified_summary`。

#### T2-S01：相同权重的 safety-on/off 反事实

在 D01 同一 checkpoint、同一 20 seeds 上，mask-all 否决 7.47% 的物理可用 BOMB，否决 53.29%
的 raw-Q 首选 BOMB。关闭 mask 后，平均炸箱从 34.35 降到 3.55，自杀升到 90%，WAIT 升到 91.82%。
因此，在该 checkpoint 上 mask 是必要保护，少炸箱的主要原因不是 mask 过度 veto。由于 off 策略访问了
完全不同的状态分布，这一实验能支持运行时安全作用，但不能等同于“分别训练 mask-on/off”的学习消融。
状态为 `verified_summary`。

#### 条件跳过实验

- T2-D03：`r7 sparse -> r7 potential`，只有 D02 失败才运行；D02 通过后正常跳过。
- T2-D04：`n_step 4 -> 5`，只有 D02/D03 均失败才运行；同样正常跳过。

两者均为 `planned_or_skipped`，没有任何性能数字。Task 3/4 为 `not_run`。

## 5. Rainbow Lite 模型族

### 5.1 模型范围和研究思路

Rainbow Lite 在 Double DQN 上组合 dueling head、比例优先回放（PER）和固定四步回报；没有实现完整
Rainbow，因此必须保留 “Lite” 名称。相关分支包括基础/无 safety/旧 continuous-v2 对照、continuous-v5
资源主线、v6--v11 对手与区域目标分支，以及 spatial-v6。不同分支常同时改变 feature、reward、父模型
和预算，只能作为完整组合筛选。

主要来源：

- [`agent-code-timeline-and-experiment-narrative.md`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md)
- [`three-model-experiment-process-draft.md`](../../../docs/research/three-model-experiment-process-draft.md)
- `runs/final_rainbow_v5_r18_maskv5_s11_*`

### 5.2 Task 1--2：固定 V5/R18/Mask-v5 课程链

为停止多轴漂移，最终本地链固定为：

```text
Rainbow Lite learner
+ continuous-v5 (140-D)
+ r18_wait_attractor_escape
+ survival-mask-v5
+ training seed 11
```

- **Task 1**：训练 500 局；多个 20 局开发快照达到 50/50，c0300 因完成速度更好被保留。
- **Task 2**：c0800 在 20 局开发评估达到 8.45/9 金币、60% 全收集、长 WAIT/往返均 0%，被选作 Task 3 父模型。

这些结果说明该单 seed 链能够完成资源课程，但属于开发集选择，不是跨 seed 独立主验证。
状态为 `verified_summary`；原始训练、metadata、逐局记录与 console log 位于对应 `runs/` 目录。

### 5.3 Task 3：checkpoint 非单调与对抗分支

固定 V5 链中：c0300 的 20 局开发结果为 7.80 分、0.65 击杀、65% 第一率、0% 长 WAIT；
c250 为 7.25 分并观察到 20% 自杀，c350 为 5.65 分并观察到 30% 自杀。因此继续训练不保证更好，
最终选择平衡更好的 c0300 进入 Task 4。

v6--v11、spatial-v6 和 r19--r22 分支尝试加入对手跟踪、阶段状态、可生存炸箱机会、压缩表示、
象限密度或可达箱目标。它们的训练预算和评估协议不一致；例如 v11/r20 的 10 局 9.9 分不能直接与
V5/c0300 的 20 局结果排名。相关结果为 `not_comparable`，部分工作区/中断分支仅为
`implemented_not_evaluated` 或 `planned_or_skipped`。

### 5.4 Task 4：扩展评估暴露长期 WAIT

固定 V5 链的开发候选 c1200 在 20 局中得到 3.25 分、0.25 击杀、40% 第一率、35% 独占第一率，
但长 WAIT 已达 65%。随后对相同 checkpoint 做 50 环境 seeds × 每 seed 20 局的 1,000 局冻结诊断：

| 指标 | 结果 |
|---|---:|
| 平均正式得分 | 2.938 |
| 合并中位数 | 2 |
| 平均金币 | 2.098 |
| 平均击杀 | 0.168 |
| 第一名/独占第一 | 26.7% / 20.1% |
| 存活率 | 83.8% |
| 自杀/被对手击杀 | 0.7% / 15.5% |
| 长 WAIT/长往返 | 57.1% / 1.0% |

这 1,000 局提高了 checkpoint 性能估计的稳定性，但不是 1,000 个独立训练，也不是预注册的
Task 4 正式质量门协议。它支持的结论是：主要残余失败从明显自杀转为长期目标停滞；不支持
“r18 单独导致 WAIT”或“Rainbow Lite 普遍弱于 Double DQN”。状态为 `verified_summary`。

### 5.5 阶段结论

Rainbow Lite 展示了完整的本地 Task 1--4 课程链，但只有一个训练 seed，内部多个 Feature/Reward
分支协议不一致，Task 4 长 WAIT 严重。它不作为最终提交候选。其历史结果与 B33 的历史 benchmark
也不是同一协议，不能直接计算算法差异。

## 6. 跨模型可支持的结论

1. 四个模型族都能在各自协议下通过 Task 1，但证据强度和训练方式不同；CNN 的最佳 Task 1 依赖 teacher 蒸馏。
2. Continuous Double DQN 与 Rainbow Lite 推进至 Task 4；Q-learning 和 CNN 停在 Task 2，空缺不是零表现。
3. 即时安全、炸箱和完成隐藏金币任务是不同能力。Q/CNN 都能安全炸箱，却仍无法稳定收齐金币。
4. 更长训练、更高版本号和更复杂网络均不保证更好；多个模型出现后期退化或行为瓶颈转移。
5. Survival mask 能显著减少特定 checkpoint 的自杀，但不会自动解决箱后目标重选、击杀或强对手得分。
6. 当前最终 B33 选择基于证据链、历史表现、CPU 和打包约束；统一冻结横评尚未完成，不能声称算法普遍最优。

## 7. 尚不能从现有实验声称的内容

- 不能声称四种 learner 已在相同训练预算与相同 Task 4 世界上完成公平排名。
- 不能把 Feature v11 解释为在 v10 上递进，也不能按版本号推断性能单调上升。
- 不能把 Rainbow V5/R18/Mask-v5 与旧组合的差异归因于任意单一轴。
- 不能把 Q-learning/CNN 的 Task 2 停止写成 Task 3/4 得分为零。
- 不能把 0% 自杀写成全局安全保证。
- 不能把训练 reward、短开发集最高分或单局回放当作最终选模依据。
