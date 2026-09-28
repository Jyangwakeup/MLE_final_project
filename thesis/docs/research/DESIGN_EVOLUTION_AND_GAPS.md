# Feature、Reward、Survival Mask 演进、跨模型使用与待补实验

> 本文把三个相互独立的设计轴合并整理。Feature 描述状态，Reward 提供训练信号，Survival Mask
> 只否决未通过相应生存准入的动作；三者不能互相替代。模型实验结果见
> [`MODEL_EXPERIMENTS.md`](MODEL_EXPERIMENTS.md)。

## 1. 阅读规则

### 1.1 三个轴的职责

```text
game_state + decision-history snapshot
        │
        ├── Feature：形成 learner 输入，不直接给出“最佳动作”
        ├── Reward：训练时评价 transition，不在正式比赛中直接存在
        └── Survival Mask：从物理合法动作中做 veto-only 准入
                         │
                         └── learner 的 Q 值对剩余动作排序
```

Feature、Reward 和 Mask 的编号彼此独立。`continuous-v5 + r18 + mask-v5` 是三个合同的组合，
不是一个共同的“版本 5”。版本号也不代表性能顺序。

### 1.2 证据状态

本文使用 `verified_raw`、`verified_summary`、`implemented_not_evaluated`、
`planned_or_skipped`、`not_comparable` 和 `not_run`。定义见
[`MODEL_EXPERIMENTS.md`](MODEL_EXPERIMENTS.md#11-证据状态)。

## 2. Feature 演进

### 2.1 表示家族

- **离散表示**：`discrete-v1`、`discrete-q-v2`、`discrete-objective-v1`、`discrete-compact-v1`，
  用于单表 Q-learning 与 Double Q 基线。
- **连续向量**：`continuous-v1`--`continuous-v11`，用于 MLP、Rainbow Lite，也可经 tile coding
  提供给 Watkins Double Q($\lambda$)。
- **Phase 表示**：`continuous-phase-v1`，117 维，用于 Task 3 phase Double DQN。
- **空间表示**：`board-v1`、`spatial-v6` 和 CNN 的 17 通道 `board-path-history-v2`。
- **Hybrid 表示**：棋盘张量与连续向量融合，用于早期 Hybrid Dueling 筛选。

完整字段定义见 [`version-comparison-v-features.md`](../../../docs/version-comparison-v-features.md) 和
[`feature-principles-guide.md`](../../../docs/research/feature-principles-guide.md)。

### 2.2 continuous-v1--v11

| 版本 | 父版本/维度 | 主要变化与要解决的问题 | 主要使用模型/阶段 | 证据与限制 |
|---|---|---|---|---|
| v1 | 起点，70 | 每动作合法性、t1--t3 危险、safe horizon/area、金币/箱子/对手距离，以及全局放弹事实；解决基本导航与危险描述 | 早期 Continuous DDQN | 有 Task 1 探索结果；与后续版本常同时改变 reward/预算，`not_comparable` |
| v2 | v1，84 | 上一动作、金币目标连续性、是否返回上一位置；缓解折返与目标丢失 | Continuous DDQN 主线、B33、Q($\lambda$) tile coding、Rainbow 对照 | 支撑 Task 2 winner、Task 3 validated 和 Q Task 1；不是纯 Feature 消融 |
| v3 | v2，107 | 每动作生存、逃生余量、下一步安全分支；己方炸弹义务；改变 safe-area 标度 | Continuous v3 安全显式分支 | 已实现并参与安全诊断；缺统一单变量性能表 |
| v4 | v3，126 | WAIT streak、近期回访、2--8 步周期及重复数；让 learner 观察循环 | Continuous v4、Q 历史/反循环实验 | Q v4/r12 Task 2 仅 0.95 金币；同时改变 reward，不能单独归因 |
| v5 | v4，140 | 箱子半平面密度、跨步炸箱目标、剩余箱子比例 | Rainbow V5 资源主线 | 完成单 seed Task 1--4 链；与 r18/mask-v5 绑定，非单变量结论 |
| v6 | v5，160 | 持久对手跟踪、相对方向/运动、火力线、目标可移动空间 | Rainbow 对抗分支 | 有局部诊断；协议不统一，`not_comparable` |
| v7 | v6，162 | 本局击杀数和是否首杀，使 phase reward 可观察 | Rainbow v7、r21/r22 依赖 | 有分支训练/中断记录；没有统一胜出证据 |
| v8 | v7，163 | 单一“可生存炸箱机会”交互事实 | Rainbow v8 | 已实现；缺严格对照，`implemented_not_evaluated`/局部诊断 |
| v9 | v8，139 | 删除 24 个恒定或精确冗余字段；按 KEEP_INDICES 投影 | Rainbow compact 分支 | 目标是等价压缩，不是新增能力；缺统一性能消融 |
| v10 | 从 v7 分叉，170 | Agent 中心四象限箱子/对手密度 | Rainbow v10 | 不考虑可达性；不是 v9 后继，协议不同 |
| v11 | 从 v7 分叉，170 | 四象限可达箱份额和最近可达箱前沿 | Rainbow v11 | 不是 v10 后继；10 局局部高分不足以证明更优 |

另有 `continuous-v2-legacy78`，只用于旧 checkpoint 兼容，不属于正式 84 维 v2 演进。

### 2.3 Feature 演进得到的经验

1. 历史特征能让模型区分静态相同但行为上下文不同的状态，但不会自动产生长期目标规划。
2. 增加维度会扩大可表达信息，也可能扩大估计难度；Q tile coding 在 Task 2 即暴露组合泛化不足。
3. 含历史的 transition 必须保存实际决策快照，不能用之后已经变化的 live history 重建。
4. v3 改变旧字段数值尺度，v9 改变索引，均不能用简单补零作 ordinary resume。
5. v10/v11 是竞争分支；同为 170 维也不表示 checkpoint 兼容。

## 3. Reward 演进

完整参数表见 [`version-comparison-r-rewards.md`](../../../docs/version-comparison-r-rewards.md)，实现见
[`rewards.py`](../../../agent_code/team_agent/rewards.py)。下表按设计亲缘整理，不把编号解释为严格单链。

| Reward | 来源与主要变化 | 目标问题 | 主要模型/实验 | 结果状态与限制 |
|---|---|---|---|---|
| r1 | 稀疏事件：步长、金币、击杀、炸箱、死亡、无效动作；含 no-crate/coin3 变体 | 最小官方目标基线 | 早期 Q/DQN | 早期筛选；缺统一消融 |
| r2 | 从 coin3/no-crate 平衡化；区分自杀/被杀，加发现金币与存活 | 稀疏反馈与死亡类型 | 早期连续模型 | 设计记录为主 |
| r3 | r2 + 金币/箱子/安全势能差 | 稠密导航信用 | Continuous DDQN、早期 CNN r3 | CNN 约 13.35 金币；不能单独归因 |
| r4 | r3 + 粗粒度振荡/idle 惩罚 | WAIT 与往返 | 早期反循环实验 | 可能误罚必要等待，后被条件项替代 |
| r5 | `coin_potential` 简化支线；`conditional_loop` 条件循环/可避免 WAIT 支线 | 减少误罚并保持导航 | CNN path r5、连续模型 | CNN 达 41.78/17% 全收集但后期退化 |
| r6 | sparse/potential；自杀 -20、危险势能、不安全 BOMB -10 | 安全放弹 | Task 2 过渡 | 设计/组合证据，缺统一消融 |
| r7 | r6 对应支线中 unsafe BOMB -10→-20 | 强化不安全放弹信用 | Continuous Task 2/3、CNN D01/D02、Q Task 1、B33 | 成为稳定基线；不能由多组合结果单独证明参数值最优 |
| r8 | 从 r7 sparse 去预测炸箱正奖励；加 avoidable-fatal -20、自杀压过正事件 | 惩罚有替代时的致命选择 | safety 消融配置 | 配置与实现存在；完成结果需逐 run 核验 |
| r9 phase | resource/combat/full 连续 phase 混合；自杀 -30 | Task 3 资源—战斗—机动权衡 | Phase DDQN | 阶段实验未稳定晋级，`verified_summary` |
| r9 anti-loop | r7 + 无用炸弹、条件循环、可避免 WAIT/致命动作 | 合并安全与停滞控制 | 多路线 Task 2 | 与 phase r9 不是同一支线 |
| r9 score-aligned | r7 sparse，仅将击杀 +5→+15 | Task 3 官方得分对齐 | 生命周期实验 | 局部实验；不等于击杀实际提高 |
| r10 | 用 v4 的真实有界循环/WAIT 历史替代启发式项 | 奖励定义与可观察特征对齐 | Q/Expected SARSA/Rainbow no-safety 支线 | 多因素结果，缺纯 reward 横评 |
| r11 | 取消即时炸箱奖励；完整炸弹周期的死亡/存活/实际炸箱结算 | 延迟 BOMB 因果信用 | no-safety/Task 2 研究 | 实现与条件实验；总体效果未统一确认 |
| r12 | 金币势能 1.5、强循环/WAIT；有金币时抑制预测炸箱 | 箱后优先收币 | Q v4/r12 | Task 2 0.95 金币；与 feature 同时改变 |
| r13 | mask-off 支线；生存选项势能、自杀/致命 -40、因果炸弹信用 | 只靠 reward 学安全 | Rainbow no-safety | 已配置/局部实验，不能替代 mask 安全证据 |
| r14 | 加强有用/无用炸弹差异和成功结算 | mask-off 下学会有效放弹 | Rainbow no-safety | 多阶段筛选；结果需按原 run 解释 |
| r15 | r14 + 金币优先、无用/零用途炸弹更重、强反循环 | 可完成目标信用 | Rainbow no-safety | 研究支线 |
| r16 | r15 的逐字段冻结副本 | 防止合同漂移 | no-safety locked | 没有新增数值；新增的是版本身份 |
| r17 | 从 no-safety 激进支线回撤，建立稳健全局炸箱纪律 | 新资源主线基线 | Rainbow v5 前期 | 组合筛选 |
| r18 | r17，仅加强 WAIT 历史惩罚并让有效 BOMB 计作进展 | 逃离 WAIT attractor | Rainbow V5 最终链 | 完成 Task 1--4；Task 4 仍 57.1% 长 WAIT，不能称问题已解决 |
| r19/a/b | r18 + 三种强度的对手势能/安全攻击 BOMB 奖励 | Task 4 对手压力 | Rainbow v6 stable 分支 | 并列消融设计；协议/父模型不完全一致 |
| r20 | 从 r18 去预测性 BOMB 正奖励，以实际炸箱/击杀结算 | 减少为预测奖励频繁放弹 | Rainbow v10/v11、Q 另有同名 targeted-wait 合同 | 名称在不同路线语境需核对完整 ID；局部结果不可横比 |
| r21 | r20 + 箱密度和首杀阶段势能 | 资源、追击、保优势的连续切换 | Rainbow v7/v8/v10 | 依赖 v7 可观察首杀；多个运行中断或协议不同 |
| r22 | r20 + 简化为首杀前/后两阶段 | 降低 r21 非平稳性和超参数数 | Rainbow v7/v8/v9 | 已实现/局部分支，未形成统一胜出结论 |

### 3.1 较强 Reward 证据

- **CNN D01→D02**：只增加窄条件 WAIT `-0.04`，其他训练合同固定。三 seed 与主验证中炸箱和
  WAIT 改善，是当前最清晰的 reward 单变量实验；但全金币率仍为 0%。
- **Q r20**：设计上只增加窄条件 WAIT，100k 相对旧同预算有改善；继续训练后长往返仍高。
  由于现有汇总未给出所有旧基线的同世界配对差异，因果强度弱于 CNN D01/D02。
- **Rainbow r18/r20--r22**：多数与 Feature 或父 checkpoint 同时变化，只能用于筛选/诊断。

### 3.2 Reward 解释边界

- 训练 reward 的量纲随版本变化，不能跨版本直接画成“性能提升”。
- 势能、WAIT 和循环项可能减少一种停滞，却把失败转移为另一种循环。
- 预测性 BOMB 奖励可能激励放弹次数，但未必产生实际箱子或击杀。
- Reward 可以惩罚致命动作，但不能像 mask 一样提供动作准入证明。

## 4. Survival Mask 演进

### 4.1 共同行为

所有版本都先计算物理合法动作，再执行 veto-only 过滤，由 learner 的 Q 值在剩余集合中排序。
`off`、`exploration`、`all` 分别表示关闭、仅约束探索、约束探索和贪心/推理。基础安全集合为空时
回到 physical Q，并记录 no-safe-action fallback；这不代表回退动作安全。

v1--v5 的正式对照见
[`version-comparison-survival-masks.md`](../../../docs/version-comparison-survival-masks.md)；v6--v9 是后续安全工程线，
其版本含义由 ADR 和研究报告补充。

### 4.2 v1--v9

| Mask | 核心变化 | 要解决的问题 | 使用与实验 | 结果/限制 |
|---|---|---|---|---|
| v1 | H=7 时间展开图；动作后至少存在一条存活路径 | 可避免的有限时域死亡 | Continuous Task 2、Q Task 2、CNN D01/D02 | Task 2 多路线 0% 自杀；不考虑对手主动封路，也不保证后续 learner 沿证明路线行动 |
| v2 | own-bomb 期间保留接近最大 escape area 的动作 | 防止逐步耗尽逃生空间 | 安全边际实验 | 只在登记失败后启用；面积大不等于路径独立 |
| v3 | BOMB/义务期要求两条内部顶点不相交时空路线 | 单一路径瓶颈 | robust safety 实验 | refinement 空回 v1；仍是逐步静态证明 |
| v4 | v3 + 所有建模的一步对手动作、放弹和执行顺序 | 对手下一步堵路/放弹 | Task 3 opponent-robust 实验 | 只精确看一步，之后仍静态；计算成本上升 |
| v5 | 完整 own-bomb 危险期的反馈可控生存；第一步精确、后续保守并集 | 把责任前移到 BOMB，并覆盖后续反应 | Task 3 safety、Rainbow V5 链 | 能 fail-closed；可能保守拒绝，400 ms 搜索预算和 guarantee-loss 成为新门槛 |
| v6 | 即使 v1 为空，也只有完成可控证明才允许新 BOMB；物理回退仅在非 BOMB 间排序 | 修复“无证明仍可放弹”反例 | Continuous 冻结父模型工程验证 | 60-world 工程与 Task1--3 paired retention 通过；不保证非 BOMB 回退生存，不是 Task 4 qualification |
| v7 | 对手 rearming envelope；未来可恢复放弹资格，当前步仍尊重实际资格 | 修复把暂时无弹对手永久视为 disarmed | 安全工程线 | 修复 rearming 反例；随后暴露 rolling deadline 的保证损失 |
| v8 | placement 时固定证明终点；后续观察只消耗剩余义务，不滚动延长 | 修复 rolling admission 与原放弹合同不一致 | paired retention/强对手工程 | 保持性测试通过，但仍有历史世界与计算成本问题 |
| v9 | 有已完成对手条件证明的移动时优先这些 proven movements；保留显式 fallback；随后做 reach-grid/compact exact 等价优化 | 防止义务外也选择未证明移动，并压缩强语义计算成本 | B33 合同与后续 v9 工程线 | 多轮正确性/保持测试通过，但 24025、24384 和 compact admission 暴露尾延迟/准入失败；不能声称全局安全 |

### 4.3 关键实验与反例

1. **CNN safety-on/off**：同一 D01 权重关闭 v1 后，自杀从 0% 升到 90%，炸箱从 34.35 降到 3.55。
   这支持 mask 对该策略的运行时保护作用。
2. **v6 严格放弹**：Task1--3 各 60 对配对世界的能力指标与 v5 相同，工程异常为零；说明登记语料上
   修复未改变观察到的能力，但不外推所有世界。
3. **v9 reach-grid**：相对 named-set reference，历史完整搜索 P95 从 255.745 ms 降至 128.727 ms，
   且保留证明输出；随后新世界 24384 仍出现一次 408.911 ms 终止失败。
4. **compact exact admission**：预检历史搜索 P95 31.913 ms，fresh Task1--4 各 100 局通过；但诊断 seed33
   第 13 局单次完整 act 262.603 ms，超过项目 250 ms margin gate，冻结准入终止。
5. **世界 27155**：最终候选仍保留 fixed-deadline 与 rolling-admission 合同不一致的已知反例；零自杀历史
   不能覆盖该限制。

### 4.4 不能混淆的诊断

- `physical_fallback`：v1 为空，只能回物理动作。
- `robust_fallback`：基础 v1 非空，更强版本的 refinement 为空。
- `robust_guarantee_loss`：主动 own-bomb 义务中可控集合为空，只能降级。
- `robust_search_timed_out`：证明未在内部预算内完成；即使拒绝动作后没有死亡，也不是证明成功。
- `safety intervention`：raw-Q 最优物理动作被 veto；它不是规则层选择了“最佳动作”。

## 5. 各模型的 Feature/Reward/Mask 组合演进

### 5.1 总矩阵

| 模型 | Task/阶段 | Feature | Reward | Mask | 状态 | 主要结果 | 主要证据 |
|---|---|---|---|---|---|---|---|
| Q-learning | T1 最小基线 | discrete variants | early r1 family | off | `verified_summary` | 能收集大量金币，稳定性不足 | [`experiment-log-runs-1-2.md`](../../../docs/experiment-log-runs-1-2.md) |
| Q Double Q($\lambda$) | T1 T1-E01 | continuous-v2 tile coding | r7 potential | v1/off | `verified_summary` | 96% 全收集，49.58 金币 | [实验日志](../../../agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md) |
| Q Double Q($\lambda$) | T2 正式父链/pilot | continuous-v2 | r7 potential | v1/all | `verified_summary` | 安全炸箱但长 WAIT/往返，未晋级 | 同上 |
| Q Double Q($\lambda$) | T2 R20 | continuous-v2 | targeted-WAIT r20 | v1/all | `verified_summary` | WAIT 减少，往返仍高 | 同上 |
| Q 失败分支 | T2 | crate/history/v4/grouped | r12/r20 等 | v1/all | `not_comparable` | 0--3.75 金币，均未晋级 | [`q-cnn-experiment-report.md`](../../../docs/research/q-cnn-experiment-report.md) |
| Continuous DDQN | T1 探索 | continuous-v1/v2 | r3/r5 family | off/early safety | `verified_summary` | 约 49.52/50，筛出强导航载体 | [`agent-code-timeline...`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md) |
| Continuous DDQN | T2 winner | continuous-v2 | r7 family | v1/all | `verified_raw` | 7.25/9、100.26 箱、0% 自杀 | [`task2_winner.json`](../../../experiments/task2_winner.json) |
| Continuous DDQN | T3 validated | continuous-v2 | r7/生命周期合同 | v5 family | `verified_raw` | 5.91→7.14；击杀无提升证据 | [`task3-counter-validation-results.md`](../../../docs/research/task3-counter-validation-results.md) |
| Phase DDQN | T3 phase | continuous-phase-v1 | r9 phase | v2/v5 experiments | `verified_summary` | retention/safety 不稳定，停止 | [`task3-phase-iteration-results.md`](../../../docs/research/task3-phase-iteration-results.md) |
| Continuous DDQN | T4 specialist | continuous-v2 | E1/E2/E3 contracts | v5 | `verified_raw` | 安全保持，无稳定得分增益 | [`table4_task4_attempts.csv`](../../report-assets/tables/table4_task4_attempts.csv) |
| B33/Die Hardest | T4 历史候选 | continuous-v2 | r7 sparse | v9 | `verified_summary` | 4.231 分/1000 worlds；包等价通过 | [`table5_die_hardest.csv`](../../report-assets/tables/table5_die_hardest.csv) |
| CNN path | T1 | 17-channel v2 | r3/r5 | off | `verified_summary` | 最佳 41.78 金币、17% 全收集 | [CNN 报告](../../../docs/research/cnn-task1-task2-experiment-report.md) |
| CNN distilled | T1 | 17-channel v2 | teacher KL | off | `verified_summary` | reserved 96%、49.84 | [CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md) |
| CNN D01 | T2 | 17-channel v2 | r7 sparse + KL | v1/all | `verified_summary` | 安全但 WAIT 55.2%，未通过 | 同上 |
| CNN D02 | T2 | 同 D01 | D01 + avoidable-WAIT -0.04 | v1/all | `verified_summary` | 炸箱/WAIT 改善；全金币仍 0% | 同上 |
| CNN S01 | T2 反事实 | 同 D01 冻结权重 | 不变 | v1 all→off | `verified_summary` | off 自杀 90%，炸箱更差 | 同上 |
| Rainbow Lite | T1--T4 固定链 | continuous-v5 | r18 | v5 | `verified_summary` | 单 seed 完成课程；T4 长 WAIT 57.1% | [`agent-code-timeline...`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md) |
| Rainbow 分支 | T3/T4 | v6--v11/spatial-v6 | r19--r22 | 多为 v5 | `not_comparable` | 局部筛选/中断，未形成统一胜者 | 同上 |

### 5.2 组合演进解释

#### Q-learning

离散基线先回答“能否学习”；Double Q($\lambda$)+tile coding 再回答“局部泛化与 trace 是否改善 Task 1”。
Task 2 先固定 v2/r7/mask-v1，再把 WAIT reward 作为窄改动；结果从长 WAIT 转为长往返，说明 Reward
只移动了行为瓶颈。v4/r12、grouped tiles 和示范实验同时涉及表示或训练来源，只能作为失败诊断。

#### Continuous Double DQN

v1/v2 筛选后，v2/r7/mask-v1 形成 Task 2 winner。Task 3 的主要进展来自冻结父子配对、历史快照正确性
和更强安全合同，而不是简单扩大网络。v6--v9 安全演进大多冻结 Q 权重以隔离运行时准入语义；这使其
适合回答“安全层改变了什么”，但不能回答“重新训练后 learner 会如何适应”。

#### CNN Double DQN

CNN 从输入 bug 修复开始，随后 teacher 蒸馏与随机初始化不属于纯结构消融。D01→D02 是明确单轴 Reward
实验；S01 是固定权重的运行时 Mask 反事实。二者共同说明：Reward 可以减少 WAIT，Mask 可以防止当前
策略自杀，但两者都未解决 Task 2 的全金币目标。

#### Rainbow Lite

基础 Rainbow 组件与 continuous-v5/r18/mask-v5 共同构成固定主线。v6--v11 和 r19--r22 围绕对手跟踪、
首杀阶段、区域目标和 outcome credit 展开，但父模型、预算和评估世界不统一。它们应写成由失败驱动的
设计探索，而不是“v11/r22 优于 v5/r18”的线性升级。

## 6. 待补实验清单

课程 PDF 要求至少两个模型、与课程预置 Agent 的比较、明确指标，以及系统展示设计修改改善或失败。
下表把“论文核心缺口”与 PDF 的建议项分开；不是所有建议项都必须在截止前补跑。

### 6.1 P0：论文核心证据，优先补

#### P0-1 四模型统一冻结横评

- **目的**：补齐当前 `main.tex` 中仍为 TBD 的统一结果，避免用异协议历史数字排名。
- **候选**：每族预先冻结一个可加载 checkpoint；记录训练终点和 hash。Q/CNN 明确标注只训练到 Task 2。
- **对照**：Task 1/2 同时加入适用的 `coin_collector_agent`/诊断基线；学习候选相互比较。
- **固定变量**：相同 scenario、环境 seed 列表、每 seed 局数、无探索、无更新、单线程 CPU、计时范围。
- **最小样本量**：Task 1 与 Task 2 各至少 100 个共同世界；所有候选使用完全相同 worlds。
- **指标**：金币、全收集、完成步数、箱子、炸弹数/存活、自杀、WAIT、长循环、非法动作、P50/P95/max。
- **完成标准**：逐局结果、聚合表、配置、checkpoint hash 和命令均归档；缺 checkpoint 的格子写 `not_run`。
- **当前状态**：`not_run`。这是最直接的报告缺口。

#### P0-2 最终模型选择的共同 Task 4 验证

- **目的**：在共同协议下比较 B33/Die Hardest、Rainbow V5/c1200 和 `rule_based_agent` 基线。
- **固定变量**：三个相同 `rule_based_agent`、共同世界、固定 agent RNG、相同单线程 CPU 与 warm-up。
- **最小样本量**：至少 100 个配对世界用于基本比较；若时间允许扩至 1,000，但先预注册且不再选 checkpoint。
- **指标**：正式得分、金币、击杀、含并列/独占第一、生存、自杀、对手击杀、WAIT/循环、完整 `act` 延迟。
- **统计**：逐世界配对差异和 10,000 次 paired bootstrap 95% CI；不把环境重复称为独立训练。
- **完成标准**：未见世界只使用一次，候选不因结果再更换；形成能直接支持最终选模的表。
- **当前状态**：`not_run`。B33 历史 benchmark 与 Rainbow diagnostic 目前不可比。

#### P0-3 论文关键主张的单变量消融

不需要补齐所有 v/r/mask 组合；只补论文准备声称“有效”的关键变化：

1. **Feature**：建议固定 Continuous DDQN 或 Rainbow 父模型，比较 v2 与选定历史/目标 Feature；显式迁移新列置零，
   optimizer/replay 按合同重置。至少三训练 seeds，每 seed 20 个开发世界，固定候选后再 100 个确认世界。
2. **Reward**：优先 Rainbow r18 对 r20 或 r22，保持 Feature、Mask、父权重、训练 seed、预算、探索和 n-step 不变；
   重点验证长 WAIT 是否下降且正式得分不降。
3. **Mask**：相同冻结 Q 权重比较 mask-on/off 或相邻 mask；若研究“学习适应”，另立相同父权重的训练消融，
   不把冻结反事实与重训练混在一起。

- **完成标准**：一次只改变一个轴，并同时报告目标指标与副作用；否则论文将措辞降为“设计动机/相关性观察”。
- **当前状态**：CNN D01/D02 已满足一个 Reward 案例；其余主张仍不完整。

#### P0-4 课程预置 Agent 对照

- **Task 1/2**：使用适用的课程或诊断基线，报告相同游戏指标，而非只写“通过”。
- **Task 3**：分别对 `peaceful_agent` 与 `coin_collector_agent`；不要合并成一个不透明平均数。
- **Task 4**：三个 `rule_based_agent`；明确训练对手、开发对手和最终评估对手是否重合。
- **最小样本量**：每个冻结候选/对手设置至少 100 个共同世界；短 20 局只能标为开发筛选。
- **完成标准**：预置 Agent 版本、seed、场景和逐局结果可复现。
- **当前状态**：历史运行覆盖部分对手，但尚无覆盖四模型的统一对照。

### 6.2 P1：提高可信度，时间允许应补

#### P1-1 多训练 seed 重复

- **对象**：至少最终 B33 路线与 Rainbow 主要竞争路线。
- **协议**：三个真正独立训练 seeds；每 seed 使用相同训练预算和 stage gate，不以某一 seed 的最好 checkpoint
  代替全部结果。
- **指标**：逐 seed 正式指标及跨 seed 均值、样本标准差；环境世界只作为 seed 内重复。
- **完成标准**：能够区分训练不稳定性与环境随机性。

#### P1-2 Reward shaping 有效性与副作用

- **优先问题**：Rainbow 57.1% 长 WAIT 是否由 r18 未能维持长期目标；r20/r21/r22 是否改善正式指标。
- **固定变量**：learner、Feature、Mask、父 checkpoint、训练 seed、预算、探索、n-step。
- **最小样本量**：三训练 seeds；每 seed 至少 20 开发世界，选定后 100 held-out worlds。
- **指标**：正式得分、金币、箱子、击杀、WAIT、往返、放弹和死亡；不比较 shaped-return 绝对值。
- **完成标准**：减少 WAIT 时不得以更多自杀、循环或更低正式得分换取虚假改善。

#### P1-3 Survival Mask 的安全—能力权衡

- **对照**：同一权重的 mask-off/v1/v5/v9，或相邻版本；先做冻结反事实，再决定是否值得重训练。
- **指标**：自杀、avoidable fatal、炸弹存活、放弹/零放弹、金币、箱子、击杀、得分、干预、各类 fallback、
  guarantee loss、search timeout、P95/max。
- **最小样本量**：安全回归语料全部通过；Task 1--3 每项 60 个配对保持世界；强对手至少 100 新世界。
- **完成标准**：不能靠“一直不放弹”通过安全门；首个工程失败即保留并停止该 frozen admission。

#### P1-4 完整 CPU 推理延迟

- **对象**：所有最终横评候选及最终提交包。
- **方法**：单线程、固定 CPU、warm-up 后计完整 `act`，包括 Feature、Mask 搜索和网络；记录 framework skip。
- **样本量**：随 P0 横评覆盖全部 action；另包含登记的最坏安全状态。
- **指标**：P50/P95/max、超过 250/480/500 ms 次数、连续超时风险、RSS。
- **完成标准**：在接近课程 CPU 的环境重跑；本机快不等于官方 AMD Ryzen 5 2600 上已通过。

### 6.3 P2：课程建议项，不作为硬性完成条件

#### P2-1 超参数优化记录

汇总实际尝试的学习率、$\gamma$、$\lambda$、epsilon、replay、target sync、batch、n-step、PER 参数和训练预算。
若只是人工逐次选择，写作“人工筛选”，不要声称系统网格或贝叶斯优化。至少保留失败配置和停止理由。

#### P2-2 对称性实验

课程 PDF 把旋转/镜像增强列为建议。若补做，固定模型与训练预算，比较无增强和 D4 增强，先验证动作/棋盘
变换严格一致；至少三训练 seeds。CNN T1 中 action-aligned+D4 只做过结构筛选，不能替代完整受控实验。
时间不足时列入 future work 即可。

#### P2-3 训练过程图

为代表性模型绘制“训练进度—冻结正式指标”曲线，并同时显示 checkpoint 退化。Task 1 用金币/全收集，
Task 2 用金币/箱子/自杀/WAIT，Task 3/4 用正式得分/击杀/第一率。不要用 loss 或 shaped reward 替代。

## 7. 建议执行顺序

1. 冻结并登记四个代表 checkpoint、hash 和可加载性。
2. 完成 P0-1 Task 1/2 统一横评，先补论文最确定的共同证据。
3. 完成 P0-2 B33 与 Rainbow Task 4 共同验证，锁定最终选择措辞。
4. 根据论文准备保留的因果主张，只执行最少数量的 P0-3 单变量消融。
5. 补齐课程预置 Agent 对照及完整 CPU 延迟。
6. 时间剩余再做多训练 seed、超参数、对称性与训练曲线；不得为追求表格完整而改写已冻结历史结论。

## 8. 写入论文时的措辞边界

- “通过”必须说明通过哪个预注册门，不等于完成整个课程 Task。
- “提升”必须有共同协议对照；异协议结果只能说“观察到”。
- “安全”应写为相应有限模型下的 survival admission，不写全局保证。
- “最终最好”在 P0-2 完成前应写成“当前证据与交付约束下选择”。
- 跳过、未运行和失败都是实验结果的一部分，不用零或空表掩盖。
- 课程要求的核心是系统设计、优化和测试；版本数量本身不是贡献。

