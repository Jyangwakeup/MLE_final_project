# Die Hardest / Continuous Double DQN 设计过程证据档案

> 文档性质：阶段 1 本地证据归档 + 阶段 2 作者决策回填，不是最终论文正文。
> 最后核验：2026-09-26。
> 目标用途：作为后续英文课程报告写作与 LaTeX 内容分配的可追溯事实来源。
> 修改边界：本阶段未修改 LaTeX、模型、实验结果或项目领域词汇。

## 1. 范围与证据状态

本文整理 Continuous Double DQN 从早期导航基线、Task 2 安全炸箱、Task 3 弱对手验证、Task 4 强对手与安全工程，到 B33 被整理为 **Die Hardest** 的过程。重点是记录“问题—假设—改动—结果—决策—限制”，而不是证明所有修改都具有独立因果效果。

### 1.1 使用的状态标签

| 标签 | 本文中的含义 |
|---|---|
| `[VERIFIED RESULT]` | 有结构化结果、逐局数据或正式冻结报告支持的结果 |
| `[VERIFIED SUMMARY]` | 有项目总结或归档报告支持，但本文未重新计算全部原始轨迹 |
| `[PACKAGING VERIFIED]` | 仅证明重命名、打包或运行等价，不等同于重新评估性能 |
| `[ENGINEERING CHECK]` | 正确性、兼容性、延迟或安全工程检查，不是模型性能实验 |
| `[DIAGNOSTIC ONLY]` | 用于定位问题，不满足正式模型选择或晋级协议 |
| `[FAILED EXPERIMENT]` | 按预注册门槛失败或被工程门槛终止 |
| `[PLANNED, NOT RUN]` | 在协议中存在，但实际没有执行 |
| `[EVIDENCE GAP]` | 当前仓库不足以支持该陈述，需要补证据或作者确认 |
| `[AUTHOR CONFIRMED]` | 由范思卿在阶段 2 访谈中确认的个人职责、设计动机或判断；不是实验测量结果 |
| `[TEAM DECISION]` | 经作者确认的团队共同决定；用于说明决策过程，不升级其性能证据等级 |

### 1.2 证据生命周期

本文沿用 [`CONTEXT.md`](../../../CONTEXT.md) 的术语：当前 `docs/`、`experiments/` 和 `thesis/` 中的材料属于 active surface 或 versioned experiment evidence；`archive/` 中的旧工作树和结果属于 recovery archive。归档中的旧绝对路径只表示历史出处，不表示相同工作树仍处于活动状态。

冻结参赛 Agent 的唯一活动副本位于 [`agent_code/die_hardest/`](../../../agent_code/die_hardest/)，包含回调、依赖、提交清单和冻结权重。源码及权重身份可在该目录核验；历史评估仍以对应的报告和原始实验归档为准：

- [`archive/README.md`](../../../archive/README.md)
- [`docs/research/die-hardest-submission/REPORT.md`](../../../docs/research/die-hardest-submission/REPORT.md)
- [`docs/research/die-hardest-submission/verification.json`](../../../docs/research/die-hardest-submission/verification.json)
- `archive/recovery/remote-merge-20260923/die-hardest-validation/official-framework/agent_code/die_hardest/`（恢复前的历史副本）
- `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/`

### 1.3 不能混合的四类结论

1. Task 2/3 的正式晋级结果不能直接当作 Task 4 强对手性能。
2. B33 的历史 1,000-world benchmark 与 Die Hardest 的打包等价检查必须分开。
3. Safety v6–v9 的工程修复和保持性验证不能写成重新训练后的性能收益。
4. 不同模型、世界集合、父 checkpoint 或评估协议下的分数不能组成正式排行榜。

## 2. 一句话贡献

`[VERIFIED SUMMARY]` Continuous Double DQN 路线以 84 维 `continuous-v2`、MLP Double DQN、`r7_safe_credit_sparse` 和逐步增强的 Survival Mask 为主线，将一个高金币导航策略发展为可安全炸箱、能在弱对手环境中提高正式得分、并最终在 CPU 和交付约束下以 B33/Die Hardest 形式冻结交付的候选；该路线的主要贡献是分离学习策略与 veto-only 安全准入，并用冻结、配对和可追溯验证限制结论范围，而不是证明更复杂网络或更强安全规则必然提高最终得分。

主要依据：

- [`thesis/docs/research/MODEL_EXPERIMENTS.md`](../../docs/research/MODEL_EXPERIMENTS.md)
- [`thesis/docs/research/EXPERIMENT_STORY_BLUEPRINT.md`](../../docs/research/EXPERIMENT_STORY_BLUEPRINT.md)
- [`docs/research/agent-code-timeline-and-experiment-narrative.md`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md)

## 3. 责任范围与团队协作

`[AUTHOR CONFIRMED]` 范思卿主导 Continuous Double DQN 路线、`r7_safe_credit_sparse`、Survival Mask v1–v9，以及从训练协议、实验运行与审计、统计分析、候选选择支持到 Die Hardest 打包验证的全链路工作。这里的“主导”不等于独占完成：相关 Feature、Reward、Mask、评估工具和结果审查在团队路线之间持续交换。

`[AUTHOR CONFIRMED]` `continuous-v2` 由祝雨嫣主导开发，随后作为共享 Feature 被范思卿接入 Continuous Double DQN，并在 Task 2–4 课程中继续验证。本文不得将 `continuous-v2` 表述为范思卿个人原创。

`[TEAM DECISION]` 团队协作最适合描述为“共享组件的交叉复用与验证”：祝雨嫣主导的 `continuous-v2` 进入 Continuous DDQN；范思卿主导的 r7 与 Survival Mask 被其他路线复用；季嘉怡负责的 CNN safety-on/off 反事实又为 Mask 的运行时作用提供跨学习器证据。课程要求明确禁止把项目描述为三个人彼此隔离、各做一个模型，因此最终论文应写成问题驱动的成果流动，而不是三个孤立子项目。

来源：

- [`thesis/docs/research/EXPERIMENT_STORY_BLUEPRINT.md`](../../docs/research/EXPERIMENT_STORY_BLUEPRINT.md)，第 3 节与第 8 节
- [`PROJECT_REQUIREMENTS.md`](../../../PROJECT_REQUIREMENTS.md)，团队协作要求

贡献声明仍应把“访谈确认的职责”与“机器可验证的提交、配置和实验结果”分开；Git 作者、提交者或文件所有者本身不能替代设计贡献证明。

## 4. 起点：基础学习器和表示

### 4.1 模型合同

`[VERIFIED RESULT]` Die Hardest 的冻结包声明：

| 项目 | 冻结身份 |
|---|---|
| 算法 | Double DQN |
| 输入 | `continuous-v2`，84 维 |
| 网络 | `Linear(84,128) → ReLU → Linear(128,128) → ReLU → Linear(128,6)` |
| Dueling head | 无 |
| 输出 | 六个动作的 Q 值 |
| Reward | `r7_safe_credit_sparse` |
| Safety | `survival-mask-v9` |
| 来源候选 | Task4 B seed33/c200，即 B33 |
| checkpoint SHA-256 | `ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a8` |

来源：

- `archive/recovery/remote-merge-20260923/die-hardest-validation/official-framework/agent_code/die_hardest/callbacks.py`
- `archive/recovery/remote-merge-20260923/die-hardest-validation/official-framework/agent_code/die_hardest/SUBMISSION_MANIFEST.json`

### 4.2 `continuous-v2` 的设计作用

`[AUTHOR CONFIRMED]` 该 Feature 的设计归属为祝雨嫣主导、团队复用。范思卿在本文路线中的贡献是将其纳入 Continuous DDQN 合同，并围绕课程晋级、安全准入和冻结验证评估该表示，而不是声明发明了新增字段。

`[VERIFIED SUMMARY]` `continuous-v1` 是 70 维局部连续表示，覆盖动作合法性、未来危险、可生存时域/区域、金币、箱子、对手距离与放弹相关事实。`continuous-v2` 在其上加入：

- 上一动作的 7 类 one-hot；
- 金币目标连续性；
- 六个候选动作是否回到上一位置。

总维度为 84。设计目标是让学习器区分“表面状态相似但行为历史不同”的局面，缓解折返和目标丢失。它仍不是完整长期记忆，也不会直接给出最佳动作。

来源：

- [`docs/research/feature-principles-guide.md`](../../../docs/research/feature-principles-guide.md)
- [`thesis/docs/research/DESIGN_EVOLUTION_AND_GAPS.md`](../../docs/research/DESIGN_EVOLUTION_AND_GAPS.md)，第 2 节

`[EVIDENCE GAP]` 现有结果没有提供只改变 `continuous-v1 → continuous-v2`、同时固定 learner、reward、训练预算、seed 和评估世界的完整单变量消融。因此不能把后续全部提升独立归因于新增 14 维。

### 4.3 r7 的设计作用

`[AUTHOR CONFIRMED]` 范思卿主导 r7 的设计。核心动机是将安全信用前移：对不可逃放弹施加强惩罚，同时只给安全且预计有用的放弹有限信用，使延迟数步才显现的爆炸后果更早进入训练信号。它不是为了用 Reward 取代 Survival Mask，也不是一次完成官方得分的完全对齐。

`[VERIFIED RESULT]` Task 2 winner 合同登记了以下关键 Reward 项：金币 `+3.0`、击杀 `+5.0`、炸箱 `+0.2`、被杀 `-10.0`、自杀 `-20.0`、非法动作 `-0.1`、每步 `-0.01`、不可逃放弹 `-20.0`，并包含有限的 useful-bomb crate credit。r7 从 r6 的安全信用设计发展而来，将 unsafe BOMB penalty 从 `-10` 加重到 `-20`。

来源：

- [`experiments/task2_winner.json`](../../../experiments/task2_winner.json)
- [`docs/research/reward-principles-guide.md`](../../../docs/research/reward-principles-guide.md)
- `archive/recovery/remote-merge-20260923/die-hardest-validation/official-framework/agent_code/die_hardest/_vendor/team_agent/rewards.py`

`[EVIDENCE GAP]` r7 与 Feature、Mask 和课程训练共同变化，不能声称每个数值已由独立消融证明最优。尤其不能仅根据平均分反推某个 reward 项的因果作用。

## 5. Task 1–4 时间线

| 阶段 | 主要问题 | 关键改动或检查 | 可支持的结论 | 状态 |
|---|---|---|---|---|
| Task 1 | 能否学习基本导航和收集金币 | 连续状态 + MLP Double DQN；比较多个完整实现组合 | Continuous DDQN 是强导航载体；复杂网络不自动更好 | `[VERIFIED SUMMARY]` |
| Task 2 前期 | 会放弹但会自杀 | 分析探索、贪心、冻结推理和 bootstrap 的 mask 不一致 | 自杀主要来自仍可选择未来必死动作，而不只是看不到危险 | `[DIAGNOSTIC ONLY]` |
| Task 2 winner | 安全炸箱并保留导航 | `continuous-v2 + r7 + survival-mask-v1/all`，三训练 seed + 一次主验证 | 完整合同通过内部 Task 2 晋级门 | `[VERIFIED RESULT]` |
| Task 3 早期 | 弱对手下结果不稳定 | phase 分支、安全版本、历史/计数诊断 | 更复杂 reward/feature 没有稳定跨 seed 晋级 | `[FAILED EXPERIMENT]` |
| Task 3 validated | 训练历史与安全计数合同错误 | 不可变 decision snapshot、修复计数，父子同世界配对 | child 提高 score/coins/crates；无击杀提升证据 | `[VERIFIED RESULT]` |
| Task 4 初期 | 强对手下安全搜索与 CPU 成本 | shared-parent transfer；v5 搜索工程准入 | 首个诊断因搜索超时终止，不能归因为学习失败 | `[FAILED EXPERIMENT]` |
| Safety v6–v9 | 封闭 BOMB 与对手条件安全语义 | 严格 placement、rearming、fixed deadline、proven movement、性能优化 | 修复若干登记反例，但延迟尾部和 27155 仍未闭合 | `[ENGINEERING CHECK]` |
| Task 4 reference B | 在 v9 下形成候选 | 三个 shared-parent seed 的 B 臂训练与候选评估 | B33 在既有候选中有较强历史表现，但 `task4_qualified=false` | `[VERIFIED RESULT]` + 边界 |
| B33 benchmark | 评估冻结候选对三名规则对手的表现 | 1,000 个预登记随机世界 | 4.231 均分、94.7% 存活；不能外推未知强对手 | `[VERIFIED RESULT]` |
| Die Hardest 打包 | 验证重命名和独立提交包 | trace、配对游戏、哈希和资源检查 | 包与 B33 行为等价；不是性能复跑或完整安全认证 | `[PACKAGING VERIFIED]` |

## 6. 关键实验卡

### E1：Task 1 完整方案筛选

- **Problem**：原始状态和简单离散策略能否稳定收集可见金币？
- **Hypothesis**：连续路径、危险和目标特征配合 MLP Double DQN 能提供足够的局部泛化。
- **Change**：实现 Continuous Double DQN，并与 Q-learning、DQN、CNN、Hybrid 等完整组合比较。
- **Controls**：`[EVIDENCE GAP]` 早期筛选同时改变表示、算法、reward 和预算，不是严格单变量对照。
- **Result**：`[VERIFIED SUMMARY]` Continuous Double DQN 的历史 Task 1 结果约为 `49.52/50` coins、`86%` all-coins；Hybrid Dueling 约为 `2.05/50`、`0%` all-coins。
- **Decision**：保留 Continuous Double DQN 作为后续 Task 2 的强导航载体。
- **Limitation**：不能将差距只归因于 Double DQN 或连续表示。
- **Evidence**：[`docs/research/agent-code-timeline-and-experiment-narrative.md`](../../../docs/research/agent-code-timeline-and-experiment-narrative.md)。

### E2：Task 2 自杀诊断

- **Problem**：模型已经学会大量放弹和炸箱，但 Task 2 冻结评估仍出现较高自杀率。
- **Hypothesis**：问题不是缺少基础危险特征，而是 Survival Mask 只约束随机探索，没有一致约束贪心动作、冻结推理和 Double DQN bootstrap。
- **Change**：审计死亡轨迹和动作选择链；提出把同一 survival set 用于训练探索、训练贪心、冻结推理和 next-state target。
- **Result**：`[DIAGNOSTIC ONLY]` 当时最佳候选在 Task 1 为 `49.90` coins、`95%` all-coins；Task 2 为 `5.45` coins、`76.65` crates、约 `30.8` bombs，但 suicide rate 为 `30%`。诊断认为死亡集中在放弹后的动作选择。
- **Decision**：把 Safety 从一个探索技巧提升为独立的 veto-only 动作准入合同。
- **Author-confirmed rationale**：`[AUTHOR CONFIRMED]` 直接触发改动的观察是：模型放弹时通常仍存在逃生路线，但之后的贪心或冻结推理会选择物理合法却不可生存的移动或 WAIT。仅依赖延迟死亡惩罚学习过慢，而且事故一旦执行便不可逆。veto-only 设计只拒绝可证明致命的动作，Q 网络仍在剩余集合中决定资源、路径和攻击取舍。
- **Limitation**：诊断解释了登记的死亡样本，不构成对所有世界的完整因果证明。
- **Evidence**：[`docs/research/task2-double-dqn-suicide-mitigation.md`](../../../docs/research/task2-double-dqn-suicide-mitigation.md)。

### E3：Task 2 winner

- **Problem**：验证模型能否安全炸箱，同时保留 Task 1 导航能力。
- **Hypothesis**：`continuous-v2 + r7_safe_credit_sparse + survival-mask-v1/all` 的完整合同能够消除已观察到的可避免自杀，并维持资源能力。
- **Change**：三个训练 seed 独立完成 Task 1→2；开发门全部通过后，固定 seed22 候选，只执行一次 100-world main validation。
- **Controlled variables**：算法、84 维 Feature、Reward、安全模式、网络与门槛由 winner manifest 固定。
- **Result**：`[VERIFIED RESULT]` 主验证中 child Task 2 为 `7.25/9` coins、`100.26` crates、`0%` suicide、`100%` bomb survival；Task 1 为 `50/50`，retention `1.0`。完整 act P95/max 为 `6.378/20.582 ms`。
- **Decision**：将 seed22 Task 2 winner 作为 Task 3 的干净父模型。
- **Limitation**：all-coins rate 未在该 manifest 中报告；结果证明完整合同通过门槛，不证明 mask、reward 或 v2 单项的净贡献。
- **Evidence**：[`experiments/task2_winner.json`](../../../experiments/task2_winner.json) 与 [`experiments/task2_winner_evaluations.csv`](../../../experiments/task2_winner_evaluations.csv)。

### E4：Task 3 phase 分支失败

- **Problem**：进入弱对手环境后，希望用阶段感知表示和 reward 提高战斗表现，同时保留资源能力。
- **Hypothesis**：117 维 phase representation 与 resource/combat/full phase reward 能在不同局面动态分配学习信用。
- **Change**：比较 phase+r7、phase+r9 resource/combat/full，后续只对排名最高的 phase+r7 试验 mask-v2。
- **Result**：`[FAILED EXPERIMENT]` Round 1 所有 arm 至少有一个 seed 的 suicide 超过 10%；Round 2 mask-v2 的 seed11/22 suicide 仍为 `10%/15%`，Task 2 coin retention 降到 `60%/67%`。
- **Decision**：按预注册停止条件终止，不启动 Round 3、确认、主验证或 Task 4。
- **Limitation**：失败否定的是该完整 phase 配置在该协议下晋级，不证明 phase-aware 方法普遍无效。
- **Evidence**：[`docs/research/task3-phase-iteration-results.md`](../../../docs/research/task3-phase-iteration-results.md)。

### E5：Task 3 生命周期修复与配对验证

- **Problem**：训练回调读取可变历史会破坏 transition 的决策快照；安全 fallback 计数也曾混淆物理 WAIT 与安全替代动作。
- **Hypothesis**：先修复实验合同，再从相同 Task 2 父模型重新训练，才能把结果解释为有效的 Task 3 学习证据。
- **Change**：保存不可变 decision snapshot；按真实 `act → events → next act` 生命周期复用历史；修正计数但不修改权重、Feature、Reward 或门槛。
- **Discovery**：`[AUTHOR CONFIRMED]` 问题首先由实际动作回放与 safety fallback / escape 计数之间的矛盾暴露，继续审计后才定位到读取旧状态时修改了 live history；不是先因分数波动而猜测算法失败。
- **Protocol**：三个训练 seed 做独立 confirmation；固定 seed22/c150 后在 100 个相同世界中比较 Task 2 parent 与 Task 3 child，并用 10,000 次 paired bootstrap。
- **Result**：`[VERIFIED RESULT]` seed22 主验证 score `5.91→7.14`，差 `+1.23`，95% CI `[0.27, 2.14]`；coins `3.66→4.94`，crates `43.78→58.97`；kills `0.45→0.44`，差值区间跨 0。三个 confirmation seed 的 score gain 为 `+1.92/+1.23/+2.56`，suicide 均为 0。
- **Decision**：形成 Task 3 validated lineage，并允许研究 Task 4 transfer。
- **Limitation**：提升主要来自资源获取而不是已证实的击杀提升；100 个世界不是 100 次独立训练。
- **Evidence**：[`docs/research/task3-counter-validation-results.md`](../../../docs/research/task3-counter-validation-results.md) 与 [`experiments/task3_validated_release.json`](../../../experiments/task3_validated_release.json)。

### E6：首次 Task 4 shared-parent 工程准入失败

- **Problem**：将 Task 3 策略迁移到三个 rule-based opponents 时，v5 的对手联合动作和执行顺序枚举产生高计算成本。
- **Hypothesis**：保留网络、Optimizer、旧 Replay 和累计学习状态，以 shared parent 启动三个新学习 seed，可以隔离 Task 4 的迁移与 retention 设置。
- **Author-confirmed rationale**：`[AUTHOR CONFIRMED]` 共享同一个 Task 3 seed22/c150 父模型的首要目的，是固定父策略和已有能力，只测量 Task 4 transfer 之后的学习随机性与复现性；节省完整重跑三条课程链的算力是附带收益，而不是把三条 seed 冒充成独立 curriculum chains。
- **Change**：从固定 Task 3 seed22/c150 父 checkpoint 显式迁移；A/B 仅改变 old-task replay fraction。
- **Result**：`[FAILED EXPERIMENT]` 首个诊断局第 57 次决策触发 `robust_search_timed_out=true`；完整 act `406.534 ms`，57 次决策 P95 `315.344 ms`。正式 A/B、冻结 baseline、confirmation 和 main validation 均未启动。
- **Decision**：按零搜索超时门槛停止；先优化 Safety 计算，不把失败归因于 Replay 或学习退化。
- **Limitation**：中断局不是完整胜负样本，也不能说明所有四人状态都会超时。
- **Evidence**：[`docs/research/task4-frozen-results.md`](../../../docs/research/task4-frozen-results.md) 与 [`docs/adr/0010-task4-shared-parent-transfer.md`](../../../docs/adr/0010-task4-shared-parent-transfer.md)。

### E7：Safety v6–v9 工程线

- **Problem**：v5 仍可能在没有完整证明时放置新炸弹，对手 rearming、责任截止时间和义务外 movement 也存在合同缺口；更强证明同时带来 CPU 成本。
- **Hypothesis**：逐版只加强明确的安全语义，并用保持性、反例和时延门槛决定是否准入，可以避免把规则变化误写成学习收益。
- **Versioning principle**：`[AUTHOR CONFIRMED]` v1–v9 采用反例驱动的小步版本化，是为了让每个版本只处理一个已登记的合同缺口，并在冻结其他组件后分别检查反例、能力保持和 CPU 副作用；目标不是在没有可审计中间状态的情况下直接替换成一个“最强”求解器。
- **Change**：
  - v6：没有可控生存证明时拒绝新 BOMB；
  - v7：加入对手未来恢复放弹资格；
  - v8：固定 placement 时的证明截止点；
  - v9：在存在已完成对手条件证明时优先 proven movement，并保留显式 fallback；
  - reach-grid / compact exact：尝试保持语义等价地降低搜索成本。
- **Result**：`[ENGINEERING CHECK]` v6 的登记保持性检查通过；v9 reach-grid 将历史完整搜索 P95 从 `255.745 ms` 降至 `128.727 ms`，但新世界 24384 出现 `408.911 ms` 搜索超时；compact exact 又在 seed33 第 13 局出现完整 act `262.603 ms`，超过内部 `250 ms` margin gate。
- **Decision**：保留已验证的局部修复和失败证据；不宣称获得全局安全或 Task 4 qualification。
- **Stopping principle**：`[AUTHOR CONFIRMED]` 失败分支服从预注册门槛：结果出现后不放宽阈值、不挑选未登记的中间 checkpoint，也不临时增加新因素来延长实验。
- **Limitation**：安全版本大多冻结 Q 权重，只能回答运行时准入语义变化，不能回答 learner 重新训练后如何适应。
- **Evidence**：[`thesis/docs/research/DESIGN_EVOLUTION_AND_GAPS.md`](../../docs/research/DESIGN_EVOLUTION_AND_GAPS.md)、[`docs/version-comparison-survival-masks.md`](../../../docs/version-comparison-survival-masks.md) 及对应 v6–v9 研究报告。

### E8：Task 4 reference B 与 B33

- **Problem**：在 v9 候选合同下获得可运行的 Task 4 checkpoint，同时处理冻结 v5 reference 已知的无证明放弹失败。
- **Hypothesis**：保持 `continuous-v2`、r7、n-step 5 和 shared-parent transfer，只改变 Safety migration 与 retention arm，可产生工程稳定的候选。
- **Change**：B arm 使用 `survival-mask-v9`、50% parent replay、distillation weight 2、每任务 replay capacity 20,000、current warmup 2,000；三个 seed 均由同一 Task 3 parent 迁移。
- **Result**：`[VERIFIED RESULT]` reference campaign 训练产生 B22/c150、B11/c50、B33/c200。随后复用 25100–25199 的 standalone candidate-only assessment 得到 B22/B11/B33 mean score `4.15/4.17/4.28`，三者工程检查通过；B33 的 first-place rate 为 `47%`，完整 act P95/max 为 `19.93/65.47 ms`。
- **Decision**：后续 submission check 按这组候选观察中的 mean score 选择 B33，B33 随后进入扩大观察和打包阶段。
- **Limitation**：该 100-world assessment 没有完整配对 parent baseline，世界并非全新，manifest 明确记录 `task4_qualified=false`。不能据此声称 B33 获得正式 Task 4 qualification。
- **Evidence**：
  - [`experiments/configs/task4_reference_B.json`](../../../experiments/configs/task4_reference_B.json)
  - [`docs/research/task4-reference-observation.md`](../../../docs/research/task4-reference-observation.md)
  - `archive/worktrees/MLE_final_project_task4_reference/runs/task4_candidate_assessment_25106/report.md`

### E9：B33 1,000-world 历史 benchmark

- **Problem**：100-world 候选观察不足以稳定估计 B33 对规则对手的表现和运行尾部。
- **Hypothesis**：在模型和 ZIP 不变的条件下，将随机世界扩大到 1,000，可获得更精细的描述性估计。
- **Protocol**：`random.Random(2026092103)` 从 `[2,000,000,3,000,000)` 不放回抽取 1,000 个世界；每局三个原版 `rule_based_agent`、classic、`train=False`、learner seed0；每局新进程；均分区间按世界 10,000 次 bootstrap。
- **Result**：`[VERIFIED RESULT]`
  - 1,000/1,000 worlds 完成；
  - mean score `4.231`，95% bootstrap CI `[4.063,4.403]`；
  - first including ties `50.2%`，sole first `37.8%`；
  - survival `94.7%`，53 次死亡、0 次自杀；
  - mean coins `3.111`、mean kills `0.224`、mean crates `37.263`；
  - act P95/P99/max `20.82/31.82/73.51 ms`；
  - framework timeout、skip、robust search timeout、guarantee loss、escape collapse 均为 0；
  - max RSS `320.54 MiB`。
- **Decision**：保留 B33 作为历史性能和本机运行稳定性证据最完整的 Task 4 候选之一。
- **Limitation**：没有相同世界的 parent 或 Rainbow 配对；对手仅为三个 rule-based agents；本机时延不是官方硬件认证；未触发 27155 类问题不等于问题已修复。
- **Evidence**：
  - `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/protocol.md`
  - `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/report.md`
  - `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/summary.json`
  - `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/manifest.json`

### E10：Die Hardest 重命名与独立打包

- **Problem**：将原 B33 以 `die_hardest` 名称放入原版框架时，必须证明引用重命名没有改变 Feature、Mask、Q 值、动作、权重或对局行为。
- **Hypothesis**：仅修改包引用、README 和 submission manifest，保持模型与算法不变，可得到行为等价提交包。
- **Change**：将 B33 源包重命名为 `die_hardest`，保留唯一权重 `final.pt`、vendor 代码和训练模块；以 SHA 锁定源包。
- **Result**：`[PACKAGING VERIFIED]`
  - 47 个 Python 文件只做名称相关变化；
  - 180 条历史 observation 的 Feature、Mask、Q 值和动作完全一致；
  - 3 局 random + 3 局 rule 对手配对游戏完整 replay 一致；
  - Die Hardest 2,130 次决策 P95/max `14.826/44.601 ms`；
  - peak RSS `319.645 MiB`；
  - world 27155 在两个版本中均于 step 77 失败。
- **Decision**：接受重命名、独立打包和运行等价性；不接受为完整 Safety certification。
- **Limitation**：没有在重命名后重新运行 1,000-world benchmark；未测试 Docker 或官方比赛硬件；`uploaded=false`。
- **Evidence**：[`docs/research/die-hardest-submission/REPORT.md`](../../../docs/research/die-hardest-submission/REPORT.md)、[`verification.json`](../../../docs/research/die-hardest-submission/verification.json) 与 [`ARTIFACTS.json`](../../../docs/research/die-hardest-submission/ARTIFACTS.json)。

## 7. Training、迁移与 Replay 合同

### 7.1 基础神经网络训练合同

`[VERIFIED RESULT]` 冻结源码的默认超参数为：

| 参数 | 值 |
|---|---:|
| discount `gamma` | 0.95 |
| learning rate | `3e-4` |
| batch size | 64 |
| replay capacity | 20,000 |
| warmup | 2,000 |
| target sync interval | 1,000 |
| gradient clip | 10.0 |

来源：归档 Die Hardest `callbacks.py`。网络通过 policy network 选择 Double DQN next action，再由 target network 对同一动作估值；Safety Mask 还需要与 next-state bootstrap 的 admissible set 一致。

### 7.2 Task 3 生命周期边界

`[VERIFIED RESULT]` Task 3 修复后，实际动作决策保存不可变快照，训练 callback 不再用之后已变化的 live history 重建旧 transition。新 Task 3 训练从登记的 Task 2 parent 开始；旧 Task 3 checkpoint 不作为干净训练起点。

`[VERIFIED SUMMARY]` n-step 为 5。旧 Task 1/2 Replay 被保留用于能力 retention，但“保留旧 Replay”不等于证明所有历史 Task 2 transition 都不受先前生命周期缺陷影响。

来源：

- [`docs/research/task3-lifecycle-frozen-results.md`](../../../docs/research/task3-lifecycle-frozen-results.md)
- [`docs/research/task3-learning-mechanisms-review.md`](../../../docs/research/task3-learning-mechanisms-review.md)

### 7.3 Task 4 显式 transfer

`[VERIFIED SUMMARY]` Task 4 从注册的 Task 3 seed22/c150 parent 开始，保留 policy、target、Adam、累计 updates/actions 和 Task 1–3 Replay；teacher 替换为冻结 parent policy；stage counters 和学习 RNG streams 在空 episode boundary 重置。A/B 仅以旧任务 replay fraction `75%/50%` 区分。

来源：[`docs/adr/0010-task4-shared-parent-transfer.md`](../../../docs/adr/0010-task4-shared-parent-transfer.md)。

`[EVIDENCE GAP]` 应在写最终论文前从 B33 checkpoint 本体重新提取并冻结一张最终 training metadata 表，确认 B33 累计 action steps、update 数、各任务 Replay 实际占比和 checkpoint lineage；目前本文主要依据注册合同和历史报告。

## 8. Survival Mask 的设计边界

### 8.1 它是什么

`[VERIFIED RESULT]` Survival Mask 是物理合法动作中通过有限时域生存检查的子集。它是 veto boundary，不是规则控制器：Mask 只排除未通过条件的动作，Q 网络仍在剩余集合中排序。

### 8.2 它不是什么

- 不是“规则层选择最佳移动方向”；
- 不是无限时域安全证明；
- 不保证在 mask 为空时的 physical-Q fallback 安全；
- 0% observed suicide 不等于所有状态都安全；
- 更强 Safety 版本不自动意味着更高正式得分。

### 8.3 v1–v9 的问题链

| 版本 | 主要新边界 | 仍保留的限制 |
|---|---|---|
| v1 | H=7，动作后至少存在一条存活路径 | 不建模对手主动封路；后续 learner 可偏离证明路线 |
| v2 | own-bomb 期保留较大 escape area | 面积不等于路径独立性 |
| v3 | BOMB/义务期要求两条内部顶点不相交路径 | 仍是逐步静态证明 |
| v4 | 加入一步对手动作、放弹和执行顺序 | 只精确建模一步，成本上升 |
| v5 | 覆盖 own-bomb danger interval 的 controllable survival | 保守、搜索昂贵，存在 fallback 风险 |
| v6 | 无完整证明时拒绝新 BOMB | 不保证非 BOMB fallback 生存 |
| v7 | 对手 rearming envelope | 暴露 rolling-deadline 问题 |
| v8 | placement 时固定证明终点 | 保持历史反例和计算成本风险 |
| v9 | 有证明移动时优先 proven movements | 仍有尾延迟与 27155 合同缺口 |

来源：[`docs/version-comparison-survival-masks.md`](../../../docs/version-comparison-survival-masks.md) 与 [`thesis/docs/research/DESIGN_EVOLUTION_AND_GAPS.md`](../../docs/research/DESIGN_EVOLUTION_AND_GAPS.md)。

## 9. 负结果与停止分支

下列失败必须保留，因为它们解释最终为何回到 B33，而不是把开发过程写成线性成功故事。

`[AUTHOR CONFIRMED]` 范思卿将预注册门槛视为首要停止原则。即使某个分支仍有优化空间，只要没有按登记条件晋级，就保留失败并停止后续确认；预算、截止时间和已有交付候选是约束，但不用于事后改写实验判定。

| 分支 | 结果 | 可得结论 | 不可得结论 |
|---|---|---|---|
| Task 3 phase+r9 | 跨 seed safety/retention 门槛失败 | 该完整配置未晋级 | phase representation 普遍无效 |
| Task 4 首次 v5 shared-parent | 首个诊断因搜索超时停止 | 当前 v5 搜索不能满足该工程门槛 | Replay 或学习导致失败 |
| v9 reach-grid | 历史 P95 改善，但新世界仍超时 | 平均/历史加速不足以消除尾部失败 | v9 整体一定更慢或更快 |
| compact exact admission | fresh 任务通过，但单次 act 超内部 margin | 工程准入失败 | 官方 0.5 秒一定失败 |
| specialist E1 | 最终 paired comparison `3.725 vs 4.245`，差 `-0.520`，CI `[-1.055,0]` | 未观察到稳定正收益 | 严格证明所有 specialist 训练都负收益 |
| specialist E2/E3 | 开发阶段未晋级，未执行独立最终比较 | 当前变体不应升级为 winner | Reward 或 gamma 的普遍无效性 |
| score L | 20k 开发分 `3.680 vs 4.195`；后续在 27155 安全失败 | 已完成部分没有正收益，正式流程被安全终止 | 40k 模型的最终可比性能 |
| Safety closure recursive prototype | 多数复杂查询在 400 ms 内为 unknown | 当前递归原型不具部署可行性 | 所有闭合 Safety 修复都不可能 |

专项学习证据主要位于：

- [`docs/research/task4-specialist-exploration.md`](../../../docs/research/task4-specialist-exploration.md)
- `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/evidence.md`
- `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/continuous/latest-report.md`
- `archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/safety-closure/report.md`

## 10. B33 的选择边界

### 10.1 当前证据支持的表述

`[VERIFIED SUMMARY]` 新的 Task 4 specialist、Reward、Replay 和 Safety 工程方案没有在各自门槛下形成稳定且可交付的统一胜者。B33 已有冻结身份、100-world 候选观察、单独的 1,000-world 历史 benchmark、本机 CPU 时延记录、归档权重哈希，以及后续完整的重命名/打包等价检查。因此，在截止时间、证据完整度、运行稳定性与交付风险共同约束下，B33 被整理为 Die Hardest。

`[TEAM DECISION]` B33 是团队共同作出的交付选择。范思卿提供 Continuous DDQN、Safety、CPU 与打包证据，其他成员提供各自路线的结果和限制；团队综合历史表现、证据完整性、推理稳定性、可复现打包和截止风险，而不是仅按某一张表中的最高分选模。

### 10.2 当前证据不支持的表述

- “B33 在统一协议下战胜了 Rainbow、Q-learning 和 CNN。”
- “B33 是 Task 4 qualified model。”
- “Die Hardest 重命名后重新跑了 1,000 局。”
- “Die Hardest 已通过完整 Safety certification。”
- “0 次自杀证明它不会自杀。”
- “本机延迟证明它在官方硬件一定满足预算。”

### 10.3 推荐的最终论文措辞

> Based on the strongest verifiable historical evidence available before the delivery deadline, together with CPU inference stability and packaging reproducibility, we selected B33 as the submission candidate and repackaged it as Die Hardest. This was an engineering decision under incomplete cross-model comparability, rather than evidence that B33 was the algorithmic winner under a unified final protocol.

## 11. 已知限制

### 11.1 world 27155

`[VERIFIED RESULT]` 冻结候选仍保留 fixed-deadline 与 rolling-admission 之间的安全合同不一致。重命名前后的模型都在该诊断的 step 77 失败。该检查复用已保存的 step 73–77 observation 与责任 history，是共享 Safety 合同的固定压力测试，不是从 world 27155 初态重新在线运行一整局。1,000 个随机世界未观察到对应告警，只说明这些实际轨迹没有触发已知缺口。

`[FAILED EXPERIMENT]` 隔离的递归 Safety closure 原型能在局部识别风险，但复杂历史查询多次耗尽 400 ms 预算，未满足部署条件。因此该修复没有进入提交包。

`[AUTHOR CONFIRMED]` 未合并该原型的主要现实原因是实时性能：复杂查询无法在允许预算内稳定结束。但最终报告还必须保留另一事实——完整正确性认证也尚未完成。因此不能把该原型描述成“逻辑已经证明正确，只差代码优化”。

### 11.2 外部有效性

`[AUTHOR CONFIRMED]` 本路线最重要的科学缺口是没有完成 B33 与 Rainbow 在共同 Task 4 worlds、相同 CPU 和统一冻结协议下的正式配对横评。它直接限制“为什么选择 B33”从工程决策升级为跨模型性能结论。

- B33 的 1,000-world 数据只覆盖三个原版 rule-based opponents；
- 没有 B33 与 Rainbow 在相同世界、相同 CPU、相同冻结协议下的正式配对比较；
- 没有完成课程 baseline 与全部模型族的统一最终横评；
- Docker 和官方比赛硬件未验证；
- 不同阶段的开发分数不能跨协议直接排名。

### 11.3 因果归因

- `continuous-v2`、r7、Mask 和课程训练通常共同变化；
- v6–v9 多为冻结权重的 Safety 工程实验；
- Task 3 得分提升不等于击杀能力提升；
- 历史 Replay 保留有助于控制遗忘，但不构成旧数据完全正确的证明。

## 12. LaTeX 报告映射

| 本文材料 | 目标 LaTeX 章节 | 建议形式 | 证据状态 | 重复风险 |
|---|---|---|---|---|
| Double DQN 与 veto-only Mask 定义 | Background | 1–2 个小节 + 公式 | 已核验概念 | 与全队共享方法重复，应只写一次 |
| 个人责任、shared assets、停止决策 | Project Planning | 责任矩阵 + 时间线 | summary + 作者确认 | 不得写成三人互不协作 |
| 84-D `continuous-v2` | Methods / State representation | 特征分组表 | summary | 与 Q/Rainbow 复用说明合并 |
| MLP Double DQN、r7 | Methods / Model and reward | 网络图 + reward 表 | result/summary | 不重复完整源码细节 |
| Mask v1→v9 | Methods / Action constraints | 演化流程图 | engineering | 避免写成九个独立模型 |
| Task 1 筛选 | Experiments / Task 1 | 小型历史表 | summary | 不能作为统一排名 |
| Task 2 suicide→winner | Experiments / Task 2 | 关键实验段 + 主验证表 | verified result | 是 Continuous 主线核心证据 |
| Task 3 lifecycle + paired validation | Training + Experiments / Task 3 | 流程图 + paired CI 表 | verified result | 修复与性能结果分开 |
| Task 4 engineering failures | Experiments / Task 4 | failure table | mixed | 不把未运行阶段填 0 |
| B33 historical 1,000 worlds | Experiments / Final candidate | 性能表/置信区间 | verified result | 与包装表分开 |
| Die Hardest packaging equivalence | Experiments / Engineering validation | 独立表 | packaging verified | 不冒充 1,000-world rerun |
| B33 选择措辞 | Conclusion | 工程选择段 | calibrated summary | 不称 algorithm champion |
| 27155、统一横评缺失 | Limitations / Future work | 明确限制列表 | verified/gap | 不弱化或删除 |

## 13. Claim–evidence ledger

| ID | 可用于论文的 Claim | 状态 | 主要证据 | 必须附带的边界 |
|---|---|---|---|---|
| C1 | Continuous DDQN 在早期 Task 1 筛选中达到约 49.52/50 coins | `[VERIFIED SUMMARY]` | agent timeline report | 多因素完整方案筛选 |
| C2 | Task 2 winner 为 84-D v2 + r7 + mask-v1/all | `[VERIFIED RESULT]` | `task2_winner.json` | 完整合同，不作单项归因 |
| C3 | Task 2 main validation 达到 7.25 coins、100.26 crates、0% suicide | `[VERIFIED RESULT]` | winner JSON/CSV | N=100；all-coins 未报告 |
| C4 | Task 3 child 的 paired score 从 5.91 提高到 7.14 | `[VERIFIED RESULT]` | counter-validation report | 同世界配对；不是独立训练次数 |
| C5 | Task 3 没有击杀提升证据 | `[VERIFIED RESULT]` | kill difference CI | “无证据”不等于证明完全无效 |
| C6 | 首次 Task 4 v5 shared-parent 因 Safety 搜索超时而非学习结果终止 | `[VERIFIED RESULT]` | task4 frozen report | 正式训练未启动 |
| C7 | v6–v9 修复了登记安全语义，但仍有尾延迟和未闭合反例 | `[ENGINEERING CHECK]` | safety reports | 不等于新模型性能收益 |
| C8 | B33 是 Task4 B seed33/c200，v2+r7+v9 | `[VERIFIED RESULT]` | submission manifest | `task4_qualified=false` |
| C9 | B33 1,000-world mean score 为 4.231，95% CI [4.063,4.403] | `[VERIFIED RESULT]` | archived protocol/report/summary | 三个 rule-based opponents |
| C10 | B33 first incl. ties 50.2%、sole first 37.8%、survival 94.7% | `[VERIFIED RESULT]` | archived summary | 不是未知对手保证 |
| C11 | 1,000-world run 中无 framework/search timeout 或三类 Safety 告警 | `[VERIFIED RESULT]` | archived summary | 27155 仍未修复 |
| C12 | Die Hardest 与源 B33 在 180 traces 和 6 paired games 中等价 | `[PACKAGING VERIFIED]` | verification JSON | 不是性能重测 |
| C13 | Die Hardest packaging P95/max 为 14.826/44.601 ms | `[PACKAGING VERIFIED]` | verification JSON | 仅本机，非官方硬件 |
| C14 | 27155 在两个包中均于 step 77 失败 | `[PACKAGING VERIFIED]` | verification JSON | full safety 未批准 |
| C15 | B33 的最终选择是证据与交付约束下的工程决定 | `[VERIFIED SUMMARY]` | story blueprint + evidence chain | 不称统一协议冠军 |

## 14. 证据冲突与待修正项

1. [`thesis/report-assets/tables/table5_die_hardest.csv`](../../report-assets/tables/table5_die_hardest.csv) 的 historical benchmark source 现在链接到活动冻结包 [`agent_code/die_hardest/README.md`](../../../agent_code/die_hardest/README.md)；历史数值仍应回到 recovery archive 下的原始 `b33-random1000/{protocol,report,summary}.json/md` 核验。
2. 多份早期文档中的绝对 worktree 路径已经迁移。引用历史路径时应同时给出 [`archive/README.md`](../../../archive/README.md) 的恢复映射。
3. B33 的 100-world candidate-only assessment 与 1,000-world benchmark 使用不同世界与用途，不应合并样本量或直接计算共同置信区间。
4. Task 2 winner 的 `0% suicide` 与 B33 benchmark 的 `0 suicides` 来自不同任务和对手协议，不能写成一条跨阶段连续统计。

## 15. 作者确认的决策记录

本节保存阶段 2 访谈中已经达成共同理解、但不能由实验文件独立证明的上下文。它使用中性第三人称，便于后续迁移到团队论文；这些记录不能覆盖或提升前文的机器证据状态。

| 决策主题 | 已确认结论 | 标签 |
|---|---|---|
| 个人主责 | 范思卿主导 Continuous DDQN、r7、Mask v1–v9，以及实验到交付全链路 | `[AUTHOR CONFIRMED]` |
| `continuous-v2` 归属 | 祝雨嫣主导开发，团队复用；范思卿负责在 Continuous DDQN 中接入和验证 | `[AUTHOR CONFIRMED]` |
| r7 动机 | 将安全信用前移，而不是用 Reward 取代 Mask | `[AUTHOR CONFIRMED]` |
| Mask 边界 | 拒绝可证明致命动作，但不替代 Q 网络的剩余动作排序 | `[AUTHOR CONFIRMED]` |
| Task 2 触发问题 | 放弹后仍选择物理合法但不可生存的动作 | `[AUTHOR CONFIRMED]` |
| Task 3 bug 发现 | 回放行为与安全计数矛盾触发 lifecycle 审计 | `[AUTHOR CONFIRMED]` |
| Task 4 shared parent | 固定父策略以隔离 transfer 后的学习随机性 | `[AUTHOR CONFIRMED]` |
| Safety 迭代方式 | 反例驱动、小步版本化，并检查保持性和 CPU 副作用 | `[AUTHOR CONFIRMED]` |
| 停止规则 | 预注册门槛优先，不在看到结果后改变判定 | `[AUTHOR CONFIRMED]` |
| B33 选模 | 团队依据证据完整度、稳定性、可复现交付与截止风险共同选择 | `[TEAM DECISION]` |
| 27155 原型 | 时延是未部署的主要现实原因，完整正确性认证也未完成 | `[AUTHOR CONFIRMED]` |
| 首要限制 | 缺少 B33–Rainbow 统一 Task 4 配对横评 | `[AUTHOR CONFIRMED]` |
| 协作主线 | Feature、Reward、Mask 在路线间复用，并由其他模型提供反事实验证 | `[TEAM DECISION]` |

## 16. 阶段 1–2 完成标准

- [x] 已区分 active evidence 与 recovery archive。
- [x] 已记录 Task 1–4 的问题、改动、结果、决策和限制。
- [x] 已分开 B33 历史性能与 Die Hardest packaging verification。
- [x] 已保留失败实验、未运行阶段和 27155 反例。
- [x] 已建立 LaTeX 映射与 claim–evidence ledger。
- [x] 未创建外部引用、未编造缺失结果、未修改 LaTeX。
- [x] 已完成作者访谈，并以 `[AUTHOR CONFIRMED]` / `[TEAM DECISION]` 与机器证据分层记录。
- [ ] B33 checkpoint 的最终 training metadata 等待单独审计。
- [ ] 统一跨模型 Task 4 验证仍为未运行，不在本文伪补。
