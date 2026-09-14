# Task 2 Double DQN 自杀问题：证据、参考方案与修复优先级

核查日期：2026-09-14
范围：当前第三轮 `double_dqn_continuous_v2_agent` 的 20-seed Task 2 冻结评估、课程往届参考仓库，以及公开 Pommerman 论文与配套源码。本文只提出后续实验假设，不把外部项目结果直接当作本项目结论。

## 结论

当前 30% 自杀率的首要原因不是“放弹时没有逃生路线”，而是模型在放弹后选择了已被本项目时空搜索判定为必死的移动或等待动作。当前安全 mask 只作用于训练期的随机探索分支；贪心训练、冻结评估和 Double DQN bootstrap 仍只排除物理非法动作。因此模型能够看见安全特征，却仍可执行和高估物理合法但必死的动作。

优先修复顺序应是：

1. 将“只否决可证明必死动作”的生存 mask 一致用于训练探索、训练贪心、冻结推理和 Double DQN 下一动作选择。
2. 对“仍可逃生 → 已无逃生”的首个决策直接分配负信用，并提高这类转移在 replay 中的采样率。
3. 把安全性由单一标量回报中的软偏好提升为约束；若规则 mask 不适合作为最终提交路径，再训练单独的短期死亡风险头。
4. 只在上述措施仍不足时调整终局自杀惩罚。单纯继续放大 `KILLED_SELF` 容易学成“不放弹”，不应是第一步。

## 1. 当前实验的直接证据

第三轮最佳候选在 Task 1 达到平均 49.90 枚金币、全金币完成率 95%；Task 2 达到平均 5.45 枚金币、76.65 个箱子和 30.8 次放弹，但自杀率为 30%。它已经学会高强度炸箱和保留金币能力，失败集中在安全决策，而非 Task 2 能力不足。汇总结果记录在 [`IMPLEMENTATION_GUIDE.md`](../../IMPLEMENTATION_GUIDE.md) 第 8.6 节，原始回放位于 `runs/eval_r3_double_dqn_child_t2_s11_03fbd1b/`。

使用当前环境逐步重放 6 个自杀 seed，并在每个动作前调用与训练相同的 H=7 时空可达性搜索，得到：

| seed | 自杀步 | 对应放弹步 | 放弹时可存活 | 首个必死决策 | 当时可存活替代动作 |
|---:|---:|---:|---|---|---|
| 10002 | 55 | 51 | 是 | 52: `DOWN` | `UP`, `WAIT` |
| 10007 | 336 | 332 | 是 | 333: `UP` | `RIGHT`, `DOWN`, `WAIT` |
| 10009 | 25 | 21 | 是 | 22: `WAIT` | `RIGHT` |
| 10011 | 59 | 55 | 是 | 57: `RIGHT` | `DOWN`, `LEFT` |
| 10018 | 255 | 251 | 是 | 252: `RIGHT` | `UP`, `DOWN`, `LEFT`, `WAIT` |
| 10019 | 378 | 374 | 是 | 375: `RIGHT` | `DOWN`, `LEFT`, `WAIT` |

即 6/6 自杀均属于“安全放弹后走错”，其中 5/6 在放弹后的第一个动作就从可逃生状态转为必死状态；没有一个案例需要先用更严格的放弹条件才能解释。

代码路径与这一现象一致：

- `continuous-v2` 已为五个非炸弹动作提供 `danger_t1..t3`、`safe_horizon` 和 `safe_area`，并为放弹提供 `escape_after_bomb` 与 `safe_area_after_bomb`。缺失的不是基础安全预测。
- [`double_dqn_continuous_v2_agent/callbacks.py`](../../agent_code/double_dqn_continuous_v2_agent/callbacks.py) 仅在 ε 随机分支调用 `survivable_exploration_mask`；贪心路径只使用物理合法 mask。冻结评估不进入 ε 分支，所以完全没有生存 veto。
- [`learning_common/neural.py`](../../agent_code/learning_common/neural.py) 的 Double DQN 下一动作也只在 replay 保存的物理合法动作中 argmax，可能持续 bootstrap 一个实际必死但 Q 值偏高的动作。

## 2. 外部 Bomberman/Pommerman 证据

### 2.1 安全动作过滤比单纯终局惩罚更直接

Pommerman 的 ActionFilter 工作把会导致死亡的动作从策略候选中移除，并让策略继续在剩余动作中决策；论文将它描述为降低长时序避险学习难度的机制，而不是替代高层策略。实验中 action filter 主要把失败转成平局，并改善胜率，说明它能显著降低灾难动作，但也必须监控是否变得过度保守。[Accelerating Training in Pommerman with Imitation and Reinforcement Learning](https://arxiv.org/html/1911.04947v2)

Skynet 同样把过滤器定位为“告诉策略不要做什么”，其检查包含下一步火焰、无逃生位置、已有炸弹覆盖以及放弹后逃生。它还使用带过滤器的随机 Agent 作为更安全的探索策略。[Skynet: A Top Deep RL Agent in the Pommerman Team Competition](https://arxiv.org/html/1905.01360v1)

BorealisAI 配套源码在每步计算安全动作，并在 lookahead 放弹测试中显式模拟新炸弹后所需逃生步数是否小于爆炸期限；SmartRandom 从过滤后的集合采样。[`action_prune.py`](https://github.com/BorealisAI/pommerman-baseline/blob/b909bb09283074747eea1ef694d4275a32d9fc1c/action_prune.py#L236-L326) [`random_agent.py`](https://github.com/BorealisAI/pommerman-baseline/blob/b909bb09283074747eea1ef694d4275a32d9fc1c/random_agent.py#L27-L43)

### 2.2 炸弹的延迟后果使普通 DQN 信用分配困难

另一项 Pommerman 研究指出，放弹的灾难后果会延迟多个动作出现，随机探索在某些放弹局面中的自杀概率很高；只看下一步不足以判断能否存活，浅层模型搜索或安全示范能显著降低样本复杂度。[Safer Deep RL with Shallow MCTS: A Case Study in Pommerman](https://arxiv.org/html/1904.05759v1)

这与本项目的回放吻合：死亡发生在放弹四步后，而决定性错误常发生在中间动作。当前 4-step return 理论上能把部分终局惩罚传播回来，但稀少的死亡样本仍要与大量高炸箱回报竞争，且 bootstrap 本身仍允许选择必死动作。

### 2.3 往届仓库提供的是启发，不是可复制证据

往届 `feature_is_everything` 对每个首步做最多 5 步的时空搜索，并模拟放弹后逃生；奖励中自杀为 `-300`、被杀为 `-100`，同时还有多个强目标奖励。[安全特征源码](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/features.py#L68-L177) [奖励源码](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/a7fe5041b02548ce4438e502ca3adb11576bae75/agent_code/feature_is_everything/train.py#L191-L221)

该仓库没有受控消融或冻结评估，不能证明 `-300` 本身有效；其 `target`/`KILL!` 特征还近似直接推荐动作，不符合本项目“不能用规则直接确定最佳动作”的课程边界。因此可借鉴的是时空安全与延迟信用，不是数值或规划器输出。

## 3. 对当前项目的具体修复建议

### P0：统一生存 mask

定义 `M_physical(s)` 为现有物理合法 mask，`M_survive(s)` 为 H=7 搜索后仍能存活到所有当前爆炸消失的动作集合：

- 行为策略：训练随机、训练贪心和冻结推理都从 `M_survive` 选择；仅当它为空时回退 `M_physical`。
- 放弹动作：在假设当前位置新增炸弹的时空图上判断是否存在完整逃生路径。
- Double DQN target：replay 保存 `next_survival_mask`；policy 网络只在该 mask 内选择下一动作，再由 target 网络评价同一动作。
- 记录 `raw_argmax`、最终动作、shield 是否介入、无安全动作回退，以及被否决动作的 `safe_horizon/area`。

这不是让规则选择“最优”动作：规则只排除根据公开游戏动力学可证明必死的动作，Double DQN 仍在其余动作中决定金币、炸箱和路线取舍。不过课程禁止用规则绕过学习，最终报告必须把这一边界写清并以消融证明模型仍承担策略决策；若课程方给出更严格解释，应改用下面的风险头。

对现有 checkpoint 可先做一次不训练的 `frozen + shield` 反事实实验。由于六个死亡点都有非空安全替代集合，这是最便宜、信息量最高的实验。如果自杀明显下降而金币/箱子不坍塌，再用一致的行为/target mask 从 Task 1 重训。

### P1：把死亡作为约束或单独风险目标

单一奖励最大化允许模型在高收益炸箱与死亡概率之间交易。当前平均 76.65 个箱子的实际塑形奖励约为 `15.33`，平均 5.45 个金币为 `16.35`，另有即时 useful-bomb 奖励；`KILLED_SELF=-20` 在 30% 发生率下的平均代价只有约 `-6`。这些值不能直接还原折扣回报，但足以说明有限终局惩罚可能不会支配高频正回报。

更稳健的方案是新增风险头 `Q_death(s,a)` 或 `P(self_death within K | s,a)`：从自杀前 K=4–8 步回标标签，动作选择先要求风险低于阈值，再在合格动作中最大化收益 Q。训练中平衡抽取死亡、near-miss 和普通样本。它保留学习型决策，但实现和校准成本高于确定性 Task 2 的动力学 shield。

### P1：精确惩罚“从可逃到必死”的动作

若保留纯奖励方案，新增一次性事件 `ENTERED_DOOMED_STATE`：当旧状态存在完整逃生路线，而执行动作后的新状态不再存在任何完整逃生路线时立即惩罚。它把责任放到错误移动，而不是四步后的终局，也比笼统惩罚所有危险格更少误伤必要逃生动作。

同时建议：

- 对该转移使用 prioritized/balanced replay，不能只提高终局样本比例。
- 自杀步同次产生的预测 useful-bomb 正奖励应取消；更彻底的做法是把 predicted bomb credit 延迟到自己的爆炸结束且 Agent 仍存活时再结算。
- 不增加普通存活步、`WAIT` 或 `SURVIVED_ROUND` 奖励，避免刷时长或不放弹。
- 若仍需提高自杀惩罚，先比较 `-20/-50/-100`，同时把“每局放弹数、存活放弹率、零放弹率”作为退化门槛，不直接照搬往届 `-300`。

### P2：加强安全特征的可分性

当前 continuous-v2 已包含每个移动方向的 `safe_horizon` 和归一化 `safe_area`，所以继续简单加入“危险/安全”概念的收益有限。更针对性的改动是：

- 每个动作显式加入 `survives_horizon` 二值位，避免网络从连续 horizon 间接学习阈值。
- 对 `safe_area` 使用桶化或 `log1p(area)`；当前除以整个棋盘面积会把 1、2、3 个逃生端点压到很小的数值区间。
- 加入逃生余量 `deadline - shortest_escape_steps`、下一步安全分支数和是否处于唯一逃生通道。
- 跟踪事实性的 `own_bomb_active`、自己炸弹位置/倒计时与 `escape_obligation`；它描述当前是否正在履行逃生任务，不给出推荐路线。
- 若向 Task 3 扩展，再区分自己与对手炸弹并处理多炸弹时序；Task 2 单 Agent 的六次死亡暂不需要更复杂的空间 CNN 才能修复。

### P2：逃生微课程

可把 Task 2 暂时拆成三个训练子阶段：

1. 地图已有即将爆炸的炸弹，只学习逃生，不允许新放弹。
2. 仅允许在有箱且可逃生时放弹，随后继续由模型在安全动作内选择路线。
3. 恢复完整 Task 2。

这与 Pommerman 文献中“先教放弹后撤离，再恢复完整环境”的课程思想一致，并避免仅靠巨大死亡惩罚收敛到零放弹策略。它需要新场景/状态初始化，优先级低于直接修复现有 mask 使用断层。

## 4. 最小可证伪实验

固定现有开发 seeds `10000–10019`，建议按以下顺序执行，且每步保留同一 checkpoint/seed 的配对结果：

| 实验 | 变化 | 主要回答的问题 |
|---|---|---|
| A | 当前 r7 checkpoint，无 shield | 基线复现 30% 自杀 |
| B | 同一 checkpoint，仅冻结推理加 survival mask | 自杀是否主要来自动作选择断层 |
| C | 从零训练，行为和 Double DQN target 均使用 mask | 一致训练是否保留 coins/crates 并消除高估 |
| D | C + `ENTERED_DOOMED_STATE`/危险样本平衡 | mask 之外的学习信用是否仍有增益 |
| E | C + 显式 survival/slack 特征 | 新特征是否提供超越硬约束的泛化收益 |

沿用联合门槛：Task 2 `coins≥2`、`crates≥5`、Task 1 保留率 `≥90%`、自杀率 `≤5%`、invalid `≤1%`。另新增：

- `unsafe_bomb_rate`；
- `safe_to_doomed_transition_rate`；
- `shield_intervention_rate`；
- `no_safe_action_fallback_rate`；
- `survived_bomb_rate` 与 `crates_per_survived_bomb`；
- `zero_bomb_round_rate`，用于发现过度保守。

若 B 已把 20-seed 自杀降为 0–5% 且金币/箱子满足门槛，应优先推进 C，而不是先改网络或把死亡奖励继续放大。若 B 仍自杀，则说明当前 H=7 搜索与真实环境存在动力学不一致，应先把失败细分为爆炸时序、箱子开放、炸弹占位或多炸弹建模错误，再讨论学习算法。

## 5. 证据边界

- 6/6 分类是对当前 20-seed 回放的确定性重放结果，能解释这批死亡，不能保证更大 seed 集没有“放弹时已无解”的案例。
- 外部 Pommerman 的观测空间、炸弹规则和算法与本课程框架不同，只支持“长时序安全过滤/课程可能有效”的机制假设。
- 往届仓库没有可复现实验表，奖励绝对值不能作为超参数依据，也不得复制其实现或权重。
- 任何方案是否优于当前模型，最终仍以冻结 checkpoint、独立 seeds 和官方 coins/crates/suicide 指标判定。

## 6. 落地状态（2026-09-14）

上述最小实验已冻结为 [`experiments/task2_safety_ablation.json`](../../experiments/task2_safety_ablation.json)：B 为旧 v5 checkpoint 的冻结 `survival-mask-v1/all` 诊断，C–E 合并成 `continuous-v2+r7`、`continuous-v3+r7`、`continuous-v3+r8` 三条从零课程链。实现采用 [`CONTEXT.md`](../../CONTEXT.md) 定义的 veto-only 边界，并以 `training-resume-v6` 保存实际 replay mask 和完整 Safety 合同。实验结果在实际运行结束前保持空白，不把实现通过测试视为性能改善证据。
