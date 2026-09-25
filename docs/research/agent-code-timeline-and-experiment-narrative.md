# `agent_code` 时间线与实验主线

> 整理日期：2026-09-25。
>
> 本文用于把仓库中的 Agent 组织成可核验的实验叙事。时间依据优先级为：Git 首次提交、实验 metadata、原始结果、工作区文件。未提交目录无法仅凭文件时间确定严格开发顺序，因此标为“工作区分支”。

## 1. 先区分目录角色

`agent_code/` 不是一组可以直接按版本号排名的模型。它包含四类内容：

| 类型 | 目录 | 在实验中的作用 |
|---|---|---|
| 课程或诊断对手 | `random_agent`、`rule_based_agent`、`peaceful_agent`、`coin_collector_agent`、`stationary_target_agent`、`fail_agent`、`user_agent` | 环境基线、对手与调试工具，不参加学习模型排名 |
| 项目基线 | `legal_random_agent`、`no_bomb_random_agent` | 检验动作合法性和“是否真的学到东西” |
| 学习候选 | Q-learning、Double Q、DQN、Double DQN、CNN、Rainbow Lite 等目录 | 接受训练并产生 checkpoint 的模型族 |
| 共享实现 | `team_agent`、`learning_common`、`tpl_agent`、`cnn_distillation_teacher_agent` | 特征、奖励、安全、训练器、模板或教师，不应作为独立最终候选 |

每个学习方案实际是一个完整合同：

```text
Agent = Feature + Learner/Network + Reward + Safety mask + Training protocol
```

所以 `rainbow_lite_v5_agent` 优于某个旧模型时，不能自动把差异全部归因于 Rainbow、Dueling、PER 或 v5 特征中的任何单项。

## 2. 可核验的开发时间线

### 2026-09-02 至 09-08：框架与领域特征

项目先建立官方模板、游戏状态处理、炸弹与逃生预测、金币和对手特征。这一阶段形成后续所有模型共享的问题分解：导航、炸箱、生存和对抗。

**实验问题：** 能否把游戏状态转成可供学习器使用、同时不直接给出“最佳动作”的事实特征？

### 2026-09-09 至 09-12：最小学习基线

首次加入：

- `q_learning_agent`：离散状态的单表 Q-learning；
- `dqn_agent`：小型 MLP DQN；
- 可恢复的课程训练与统一训练/评估入口。

**研究目的：** 先证明机器学习策略能在 Task 1 学会收金币，并建立表格方法与神经网络的最低基线。

**观察：** 早期模型能获得大量金币，但部分局会长时间 WAIT 或往返，说明平均得分不足以诊断策略质量。

### 2026-09-13：多表示与多算法探索

同一天扩展出多个探索分支：

| Agent | 核心变化 | 实验角色 |
|---|---|---|
| `double_q_compact_agent` | 紧凑离散状态 + Double Q | 检验降低最大化偏差和状态压缩 |
| `double_q_agent` | objective 特征 + Double Q | 表格课程候选 |
| `double_dqn_continuous_agent` | 连续向量 + MLP Double DQN | 检验连续表示的泛化能力 |
| `double_dqn_continuous_v2_agent` | continuous-v2 与安全课程 | 后续 Task 2 主线 |
| `cnn_double_dqn_agent` | 棋盘张量 + CNN Double DQN | 检验端到端空间表示 |
| `hybrid_dueling_double_dqn_agent` | 棋盘和向量融合 + Dueling Double DQN | 检验更复杂网络是否值得训练成本 |

这一批属于**探索性筛选**，因为特征、reward、网络和实际训练预算并未全部固定。历史 Task 1 结果中，Continuous Double DQN 达到 49.52/50 金币和 86% 全收集率；Hybrid Dueling 仅为 2.05/50，成为应如实保留的失败分支。

### 2026-09-14：从“会导航”转向“能安全炸箱”

主要变化包括：

- `double_dqn_continuous_v3_agent`：显式安全特征消融；
- `cnn_path_double_dqn_agent`：加入路径感知的空间表示；
- 将 survival mask 一致应用于探索、贪心行为、冻结推理和 Double DQN bootstrap；
- 用冻结得分判断 Task 1 收敛；
- 发布 `double_dqn_continuous_v2_agent` 的 Task 2 winner。

**核心问题：** 高金币导航能力进入有箱子场景后，能否学会放弹并活下来？

**已验证结果：** Task 2 winner 的主验证为平均 7.25 枚金币、100.26 个箱子、0% 自杀，并保留 100% 的 Task 1 能力。该结论有三训练 seed 和独立主验证支撑。

### 2026-09-15：弱对手、阶段奖励与空间分支

加入或推进：

- `double_dqn_phase_agent`：Task 3 phase-aware reward；
- `cnn_path_double_dqn_agent` 的历史缓存、自身通道修正和四步回报；
- own-bomb escape obligation；
- opponent-robust bomb escape constraints。

**核心问题：** Task 2 的资源与安全能力能否迁移到有对手的 Task 3？

初始 Task 3 pilot 和 phase 分支并不稳定。这些失败促使项目把“当前状态特征”扩展为“实际决策时的不可变历史快照”，并区分生存 mask、fallback 与学习器的动作排序。

### 2026-09-16：Rainbow Lite 与更细的特征、奖励、安全版本

首次提交 Rainbow Lite 主线：

- `rainbow_lite_agent`：Dueling Double DQN + proportional PER + 固定四步回报；
- `rainbow_lite_no_safety_agent`：安全消融；
- `rainbow_lite_v5_agent`：continuous-v5，加入箱区方向和跨步炸箱目标；
- `double_dqn_continuous_v4_agent`、`double_q_lambda_agent`、Expected SARSA(λ) 系列：历史与 trace 方向的并行研究。

Rainbow Lite 的“lite”只表示上述组件组合，不能写成完整 Rainbow 实现。v5/r18/mask-v5 是三个独立版本轴的组合：

```text
continuous-v5（140维）
+ r18_wait_attractor_escape
+ survival-mask-v5
+ Rainbow Lite learner
```

### 2026-09-17 至 09-19：验证修复与三条路线并行收敛

这一阶段不是单一模型的线性升级，而是三条并行研究：

1. **Double DQN 主线**：修复 Task 3 决策历史和安全计数，形成 `double_dqn_continuous_v2_agent/task3_validated.pt`。它是已验证的 Task 3 候选，但 release manifest 明确记录 `task4_started=false`。
2. **CNN 主线**：加入 distilled CNN、教师数据和 n-step 流程。CNN 结果应作为表示学习分支记录，不能因为网络更复杂就假定更强。
3. **表格主线**：`optimized_double_q_lambda_agent` 使用 tile coding、Watkins trace 和独立确认集。Task 1 确认达到 49.58/50 和 96% 全收集率，证明表格方法仍是强基线。

同一时期还进行了 Q-learning demonstration、历史特征、箱距量化和 Task 4 specialist 等诊断。这些是针对失败现象的实验，不应全部包装成最终模型候选。

### 2026-09-17 至 09-20：Rainbow Lite 工作区分支

以下目录在整理时仍为未提交工作区分支，顺序来自特征依赖和本地实验记录，不等同于正式发布顺序：

| 分支 | 表示或目的 | 关系 |
|---|---|---|
| `rainbow_lite_v6_agent` | continuous-v6，对手跟踪、相对运动和火力线 | v5 的对抗扩展 |
| `rainbow_lite_v6_stable_agent` | v6 稳定性消融 | v6 平行分支 |
| `rainbow_lite_v7_agent` | continuous-v7，加入是否已首杀 | 从 v6 扩展 |
| `rainbow_lite_v8_agent` | 可生存炸箱机会 | 从 v7 扩展 |
| `rainbow_lite_v9_agent` | 删除冗余字段的紧凑投影 | v8 的压缩版本 |
| `rainbow_lite_v10_agent` | Agent 中心的象限密度 | 从 v7 分叉 |
| `rainbow_lite_v11_agent` | 象限内可达箱目标 | 从 v7 分叉，非 v10 后继 |
| `rainbow_lite_spatial_v6_agent` | 空间表示实验 | 独立表示分支 |
| `rainbow_lite_continuous_v2_agent` | 旧连续表示上的 Rainbow 消融 | 算法/表示对照 |

这些分支训练预算、父模型和评估 seed 不完全一致，只能用于局部诊断。现有证据不能支持“版本号越大越好”。

### 2026-09-20 至 09-25：固定 V5 完成 Task 1–4 课程链

为避免继续同时改变表示、reward 和安全合同，最终本地实验固定 `rainbow_lite_v5_agent + r18 + mask-v5`，训练 seed 11，并在每 100 局冻结评估：

| 阶段 | 选择 | 选择依据 |
|---|---|---|
| Task 1 | 完成 500 局 | 多个快照在 20 局开发评估达到 50/50；c0300 完成速度较快 |
| Task 2 → Task 3 | c0800 | 8.45/9 金币、60% 全收集，长 WAIT/往返均为 0%；综合门槛优于只看金币最高的 c0500 |
| Task 3 → Task 4 | c0300 | 7.80 分、0.65 击杀、65% 第一率、0% 长 WAIT；牺牲 0.10 分换取明显较低停滞 |
| Task 4 开发候选 | c1200 | 20 局中 3.25 分、0.25 击杀、40% 第一率和35%独占第一率；但长 WAIT 达 65% |

c1200 随后完成 1000 局扩展冻结评估（50 seeds × 20 rounds）：平均分 2.938，1000 局合并中位数 2，平均金币 2.098，平均击杀 0.168，第一率 26.7%，独占第一率 20.1%，存活率 83.8%，自杀率 0.7%，被对手击杀率 15.5%，长 WAIT 率 57.1%，长往返率 1.0%。框架 `AVERAGE` 行的 `median_score=2.875` 是 50 个 seed 内中位数的平均，不能当成全部 1000 局的合并中位数。

这 1000 局提高了性能估计的稳定性，但使用的 50-seed 扩展设计不是 `task4_quality_gate.json` 预注册的 5 seeds × 20 rounds 正式门槛设计，不能替代预注册判定。结果还表明 c1200 的主要失败模式是长 WAIT，而不是自杀或推理超时。

## 3. 建议用于报告的实验思路

### 研究总问题

> 在 CPU 决策时限下，怎样逐步扩展一个学习型 Bomberman Agent，使其从金币导航迁移到安全炸箱和多智能体对抗，同时避免停滞、自杀及旧任务能力遗忘？

### 实验链条

#### 实验 1：选择基础学习器与状态表示

比较 Q-learning、DQN、Double Q、Continuous Double DQN、CNN 和 Hybrid Dueling 在 Task 1 的冻结表现。

- 主要指标：平均金币、全收集率、完成步数；
- 诊断指标：长 WAIT、长往返、非法动作和 CPU `act` 时间；
- 结论形式：筛选“完整实现组合”，不做纯算法因果归因；
- 决策：Continuous Double DQN 提供强导航载体，Double Q(λ) 保留为强表格对照，Hybrid 作为失败记录。

#### 实验 2：定位导航停滞并改进历史可观测性

从“平均金币高但不能收齐”出发，引入上一动作、目标连续性、回访和周期历史，并比较反循环 reward。

- 假设：历史特征与条件惩罚能减少 WAIT 和往返，同时保持金币能力；
- 必须控制：算法、训练 seed、预算和评估 worlds；
- 注意：WAIT 降低但 ping-pong 上升不算完全改善。

#### 实验 3：在 Task 2 验证炸箱与生存

将 survival mask 一致用于行为和 bootstrap，并区分物理合法与 horizon-survivable action。

- 主要指标：金币、全金币率、炸箱、每弹炸箱、炸弹存活率和自杀率；
- 保留指标：Task 1 得分不得明显下降；
- 结果：Double DQN v2 的正式 Task 2 winner 证明安全约束能支持课程晋级；Rainbow V5 的 c0800 则作为最终 V5 链的局部父模型。

#### 实验 4：在 Task 3 研究对手压力与能力遗忘

加入弱对手后比较击杀、金币、第一率和循环；同时回测 Task 1/2。

- 假设：对手特征和延迟结果信用能增加击杀，而不会破坏资源能力；
- 失败学习：phase reward、历史快照错误和安全计数错误说明训练逻辑本身也是实验合同的一部分；
- 决策：最终 V5 链用 c0300 晋级，因为它在得分、击杀和停滞之间更平衡。

#### 实验 5：在 Task 4 选择对战 checkpoint

以相同三名 `rule_based_agent`、相同 worlds 和冻结权重比较每 100 局 checkpoint。

- 主要指标：平均正式得分；
- 次要指标：击杀、第一率、独占第一率、存活和对手击杀率；
- 硬门槛：自杀、非法动作、循环和推理延迟；
- 结果：开发集选出 c1200，但 1000 局扩展评估把平均分修正到 2.938，并确认 57.1% 长 WAIT 是主要限制。

#### 实验 6：三类最终模型的公平横评

最终报告不应只展示 Rainbow Lite 内部版本。应冻结三类代表：

1. 表格路线：`optimized_double_q_lambda_agent`；
2. MLP 路线：`double_dqn_continuous_v2_agent`；
3. Rainbow 路线：`rainbow_lite_v5_agent` c1200 或通过正式门槛后的替代 checkpoint。

三者必须在同一 Task 4 worlds、对手、局数和 CPU 条件下评估，并报告逐局配对差值及不确定性。如果某一类没有 Task 4 checkpoint，应明确写“无同阶段候选”，不能用 Task 1/2 成绩代替 Task 4 横评。

## 4. 报告章节的推荐叙事

```text
基础基线
  → 发现导航停滞
  → 引入历史和条件 reward
  → 进入 Task 2 后发现炸弹生存问题
  → 用 veto-only survival mask 约束可证明必死动作
  → 进入 Task 3 后发现对手压力、延迟信用和能力遗忘
  → Rainbow Lite 加强多步信用与 replay
  → Task 4 checkpoint 筛选
  → 大样本评估暴露长 WAIT 的残余失败模式
```

这一叙事的重点是“每次改动由上一轮失败现象驱动”。不要把所有 Agent 逐个介绍成互不相干的作品，也不要把 Git 提交顺序伪写成严格串行实验。

## 5. 可直接用于 Results 的结论边界

- 可以说：Continuous Double DQN、Double Q(λ) 和 Rainbow Lite 在各自已评估阶段表现出不同优势。
- 可以说：更长训练并不保证更好，Task 2/3/4 均出现后期快照退化。
- 可以说：冻结评估、历史一致性、安全合同和多指标选模对结果有实质影响。
- 可以说：c1200 的 1000 局扩展评估显示其自杀率低，但长 WAIT 严重。
- 不能说：v11 因版本号更高而优于 v5。
- 不能说：早期探索表中某个算法单独造成全部性能差异。
- 不能把 20 局开发筛选称为正式通过，也不能把 1000 局扩展评估自动称为预注册质量门槛评估。
- 不能把 `task3_validated.pt` 描述为 Task 4 模型。

## 6. 证据索引

| 内容 | 证据 |
|---|---|
| 课程任务与硬约束 | `PROJECT_REQUIREMENTS.md` |
| 训练与评估协议 | `experiments/CURRENT_TRAINING_EVALUATION_PARAMETERS.md` |
| 初期模型筛选 | `docs/experiment-log-runs-1-2.md` |
| Agent 算法与角色 | `docs/research/agent-schemes-guide.md` |
| 特征、reward、mask 版本 | `docs/version-comparison-v-features.md`、`docs/version-comparison-r-rewards.md`、`docs/version-comparison-survival-masks.md` |
| Double DQN Task 2 winner | `experiments/task2_winner.json`、`experiments/task2_winner_evaluations.csv` |
| Double DQN Task 3 验证 | `experiments/task3_validated_release.json`、`docs/research/task3-counter-validation-results.md` |
| 表格路线 | `agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md` |
| CNN 路线 | `agent_code/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md`、`agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md` |
| V5 最终本地链 | `runs/final_rainbow_v5_r18_maskv5_s11_*` |
| c1200 扩展评估 | `runs/final_rainbow_v5_r18_maskv5_s11_t4_c1200_formal1000/` |

## 附录：学习型 Agent 目录归并表

| 模型族 | `agent_code` 目录 | 建议写法 |
|---|---|---|
| 单表 Q-learning | `q_learning_agent` | 最小、可解释的表格基线 |
| Double Q | `double_q_agent`、`double_q_compact_agent` | 双估计与状态压缩探索 |
| Trace 表格方法 | `double_q_lambda_agent`、`optimized_double_q_lambda_agent` | Watkins Double Q(λ) 主线；报告优先使用有独立确认的 optimized 版本 |
| Expected SARSA(λ) | `expected_sarsa_lambda_agent`、`expected_sarsa_lambda_v5_agent`、`expected_sarsa_lambda_no_safety_agent` | on-policy expectation 与安全/特征消融，不与 Double Q(λ) 合并为同一算法 |
| 基础 DQN | `dqn_agent` | 离散向量神经基线 |
| Continuous Double DQN | `double_dqn_continuous_agent`、`double_dqn_continuous_v2_agent`、`double_dqn_continuous_v3_agent`、`double_dqn_continuous_v4_agent` | 连续表示主线及安全、历史特征消融；v2 包含正式 Task 2 winner |
| Phase Double DQN | `double_dqn_phase_agent` | Task 3 阶段奖励实验，保留失败证据 |
| CNN Double DQN | `cnn_double_dqn_agent`、`cnn_path_double_dqn_agent` | 原始棋盘表示与路径感知表示 |
| CNN distillation | `cnn_distillation_teacher_agent`、`cnn_distilled_double_dqn_agent` | 教师数据生成和蒸馏学生；教师目录不是最终参赛候选 |
| Hybrid Dueling | `hybrid_dueling_double_dqn_agent` | 棋盘与向量融合的早期复杂网络失败分支 |
| Rainbow Lite 基线/消融 | `rainbow_lite_agent`、`rainbow_lite_no_safety_agent`、`rainbow_lite_continuous_v2_agent`、`rainbow_lite_v3_eval_agent` | 基础实现、安全消融、旧表示对照和冻结评估适配器 |
| Rainbow Lite 资源主线 | `rainbow_lite_v5_agent` | continuous-v5/r18/mask-v5 最终本地课程链 |
| Rainbow Lite 对抗与表示分支 | `rainbow_lite_v6_agent`、`rainbow_lite_v6_stable_agent`、`rainbow_lite_v7_agent`、`rainbow_lite_v8_agent`、`rainbow_lite_v9_agent`、`rainbow_lite_v10_agent`、`rainbow_lite_v11_agent`、`rainbow_lite_spatial_v6_agent` | 工作区研究分支；按假设描述，不按编号宣称递进胜出 |
