# R 版本对比：奖励函数的演进

## 1. 范围与重要说明

本文覆盖 `agent_code/team_agent/rewards.py` 当前注册的全部奖励 ID。`r` 是奖励契约版本，不是网络或特征版本。相同 r 可以配不同 learner、v 和 mask；反过来，改变 v 或 mask 也不自动改变奖励。

奖励线并非严格的编号单链：r1 有多个早期变体，r5/r6/r7 有 sparse/potential 分支，r9 同时有 phase 实验和 anti-loop 主线，r16 是 r15 的冻结副本，r17 是另一条回归主线，r19 有 a/b 消融。文中的“父版本”按源码内容和显式字典继承说明，而不是按数字机械认定。

## 2. 所有版本的共同机制

- 最终 transition 奖励由官方事件奖励、状态势能差、动作条件奖励、历史惩罚和跨步炸弹结果信用中的适用项相加。
- 势能塑形遵循 `gamma * Phi(next) - Phi(old)` 形式；权重为零或字段不存在时，相应项不生效。
- 自杀和被击杀分开计价；实现会避免同一次死亡被重复计数。
- `suicide_dominates_positive_events=1` 的版本在自杀 transition 上抑制正事件，避免“炸到资源但同时死亡”仍获得净正奖励。
- 安全 mask 是动作准入边界，reward 是学习信号。`unsafe_bomb_penalty` 或 `avoidable_fatal_action` 不等于 mask，也不能提供形式化生存保证。
- 同名参数只有在奖励计算函数实际检查该键时才起作用；注册表是 checkpoint 可复现契约，不应随意原地修改旧 ID。

## 3. 演进地图

```text
r1 ─ r2 ─ r3 ─ r4
 │
 ├─ r5_coin_potential
 └─ r5_conditional_loop ─ r6_safe_sparse/potential ─ r7_sparse/potential ─ r8
                                                        ├─ r9 phase 三阶段分支
                                                        └─ r9 anti-loop ─ r10 ─ r11/r12

无 safety 学习支线：r10 → r11 → r13 → r14 → r15 = r16(冻结)
当前资源/战斗主线：r17 → r18 → r19(a/b) → r20 → r21 或 r22
特殊 Task 3 分支：r7_sparse → r9_task3_score_aligned
```

这张图表达设计亲缘，不表示每个实验都实际从对应权重继续训练。

## 4. 逐版本差异

### 4.1 r1 及早期同号变体：稀疏官方目标

`r1`：每步 -0.01，金币 +1，击杀 +5，炸箱 +0.2，死亡 -10，无效动作 -0.1。它只依赖稀疏事件，优点是接近比赛目标，缺点是长路径中的中间动作信用很弱。

- `r1_no_crate`：仅把炸箱奖励从 +0.2 变为 0，用于隔离“炸箱本身”是否造成偏置。
- `r1_coin3`：金币从 +1 提到 +3，其他保持 r1。
- `r1_coin3_no_crate`：同时使用金币 +3 和炸箱 0。

### 4.2 r2_balanced：细分死亡并补充发现/存活反馈

相对 `r1_coin3_no_crate`，增加发现金币 +0.25、炸箱 +0.1、活过回合 +0.25；将单一 death 拆为自杀 -7 和被杀 -5；无效动作加重到 -0.2。它试图平衡导航、资源与生存，但较轻的自杀惩罚后来证明不足。

### 4.3 r3_potential：增加稠密势能差

完整继承 r2，并加入 `gamma=0.95`，金币势能权重 0.5、箱子前沿 0.25、安全 0.25。新增信号奖励“变得更接近目标/更安全”，不是奖励某个固定动作。

### 4.4 r4_anti_oscillation：增加早期反振荡项

在 r3 上增加往返振荡 -0.08，以及随连续 idle 增长的 -0.04/步、最多 3 级。问题是它依赖较粗的历史判断，可能把合理等待或避险回撤也处罚。

### 4.5 r5 的两条分支

`r5_coin_potential` 是简化分支：保留金币 +3、击杀 +5、炸箱 +0.2、死亡 -10、无效 -0.1，只保留金币势能 0.5。它去掉 r2/r3 中发现金币、存活、箱子与安全势能。

`r5_conditional_loop` 则从 r3 思路继续：保留完整多目标势能，用条件循环惩罚 -0.08 和“可避免 WAIT” -0.04 替代 r4 的无条件式 idle 项。其目标是只在存在更好可行选择时惩罚停滞。

### 4.6 r6_safe_sparse / r6_safe_potential：强化死亡与放弹安全

两者将自杀加重到 -20、被杀 -10；加入危险势能 1.0、不安全放弹 -10、每个预期可炸箱子 +0.2（最多 3 个）。`sparse` 没有金币势能，`potential` 额外保留金币势能 0.5。箱子势能 0.25 在两者中都存在。

### 4.7 r7_safe_credit_sparse / r7_safe_credit_potential：进一步加重不安全放弹

相对对应 r6，唯一核心参数变化是 `unsafe_bomb_penalty` 从 -10 加重为 -20。potential 分支仍比 sparse 多金币势能 0.5。r7 后来成为多项 Task 2/3 安全实验的固定奖励基线，以便只改变 mask。

### 4.8 r8_safe_constrained：惩罚可避免的致命选择

从 r7 sparse 出发，移除“根据当前预测箱子数奖励 BOMB”的两项；加入 `avoidable_fatal_action=-20`，并启用自杀压过正事件。重点从“放弹是否看似有用”转为“是否在存在可生存替代动作时选择了必死动作”。

### 4.9 r9 phase 三阶段分支

三者使用由局势进展连续混合的 early/middle/late 事件值：金币 3/1.5/1，箱子 0.2/0.05/0，击杀 5/8/7.5；自杀进一步为 -30。阶段不是硬切回合数，而由箱子消耗、对手减少和回合进度组成。

- `r9_phase_resource`：只启用资源阶段势能。
- `r9_phase_combat`：在 resource 上增加战斗势能。
- `r9_phase_full`：再增加 mobility 势能。

这是一条实验分支，并不是下面 `r9_safe_credit_anti_loop` 的父版本。

### 4.10 r9_safe_credit_anti_loop：r7 的循环控制扩展

回到固定事件奖励：金币 +3、箱子 +0.2、击杀 +5。保留 r7 的安全项，加入无用炸弹 -0.2、条件循环 -0.08、可避免 WAIT -0.04、可避免致命动作 -20及自杀优先。它把 r5 的反循环思想合并到 r7 安全信用。

另有 `r9_task3_score_aligned`：它直接从 r7 sparse 派生，只将击杀从 +5 提高到 +15，用于受限 Task 3 生命周期实验；它与另外两类 r9 不共享 phase 或 anti-loop 扩展。

### 4.11 r10_bounded_history_anti_loop：用有界真实历史替代启发式循环项

在 r9 anti-loop 的目标基础上，删除 `useless_bomb_penalty`、`conditional_loop_penalty`、`avoidable_wait_penalty`，改为：历史循环每级 -0.04、上限 3；WAIT 连续每级 -0.04、上限 3。它与 v4 的显式周期/WAIT 特征配套，使惩罚定义和模型观察对齐。

### 4.12 r11_causal_bomb_credit：把炸弹成败归因到完整炸弹周期

相对 r10：取消即时炸箱事件奖励；自杀加重到 -40；加入无用炸弹 -0.3、由该炸弹导致死亡 -25、炸弹周期结束后存活 +3、实际炸箱每个 +0.1。它不再只在放弹瞬间按预测用途记账，而是等待该炸弹及火焰消失后用可观察结果结算，改善延迟信用归因。

### 4.13 r12_coin_priority_anti_loop：优先金币并加强停滞惩罚

它不继承 r11 的完整因果结算，而是另一种折中：金币势能升到 1.5；炸箱事件恢复 +0.1；有可达金币时抑制预期炸箱奖励；循环惩罚变为 -0.2、最多 5；WAIT 为 -0.08、最多 5；自杀回到 -20。预期炸箱每箱 +0.1、最多 3。

### 4.14 r13_no_safety_survival_credit：为关闭 mask 的实验提供学习信号

基于 r10/r11 思路，在 mask 关闭时加入生存可选动作势能 2.0；自杀 -40、可避免致命动作 -40、无用炸弹 -0.3、因果炸弹致死 -25、周期存活 +3、实际炸箱 +0.1。它的目的是检验 learner 能否仅靠 reward 学到安全，而不是依赖动作 veto。

### 4.15 r14_no_safety_useful_bomb_credit：加强有用/无用炸弹分离

相对 r13：即时预测可炸箱恢复为每箱 +0.2、最多 3；无用炸弹加重到 -1；新增成功基础 +1、每实际炸箱 +1；周期存活奖励降为 +1、实际箱子结算升为 +0.5，并要求有实际 utility 才给 resolved success。直接炸箱事件奖励设为 0，避免重复计分。

### 4.16 r15_no_safety_objective_credit 与 r16 锁定版

r15 在 r14 上进一步偏向可完成目标：金币势能 1.5；有可达金币时抑制预测炸箱奖励；无用炸弹 -2；零实际用途的因果炸弹 -2；循环 -0.2、WAIT -0.08，二者最多 5。

`r16_no_safety_locked` 是 r15 的逐字段拷贝，没有数值变化。其意义是冻结最终 no-safety 契约，保证新实验不再修改旧 ID；因此“r16 加了什么”的答案是：没有新增奖励项，新增的是不可漂移的版本身份。

### 4.17 r17_global_crate_bomb_discipline：回到较稳健的全局炸箱纪律

r17 不是 r16 的数值递增版。它回撤多项过强设置：金币势能回到 0.5，自杀 -20，可避免致命 -20，循环/WAIT 均回到 -0.04×最多3；炸箱事件 +0.2；有用炸弹每箱 +0.2；无用炸弹 -1；成功基础/每箱均 +0.5，零用途 -1。并移除 r16 的生存选项势能、金币可达抑制、因果死亡和 resolved-cycle 奖励。它建立新的资源主线基线。

### 4.18 r18_wait_attractor_escape：只加强 WAIT 吸引子的逃离

直接继承 r17，仅把 WAIT 历史惩罚从 -0.04 提到 -0.2、上限从 3 提到 5，并加入 `useful_bomb_counts_as_wait_progress=1`。后者防止有效放弹打断停滞进度记账，从而避免 Agent 用“放弹”规避 WAIT 惩罚。

### 4.19 r19 对手压力三种强度

三者都继承 r18：

- `r19_opponent_pressure`：对手距离势能 0.5，安全地威胁对手的放弹奖励 +0.5。
- `r19a_opponent_pressure_weak`：二者降为 0.2 和 +0.1。
- `r19b_opponent_potential_only`：保留 0.2 的对手势能，但把安全攻击放弹即时奖励设为 0。

这些是并列消融，不是 r19→r19a→r19b 的训练继承链。

### 4.20 r20_outcome_credit：去掉预测性 BOMB 正奖励

从 r18 出发，保留对手势能 0.5，但把 `useful_bomb_per_crate` 和 `safe_opponent_bomb_reward` 都设为 0。放弹是否成功只由之后实际观察到的 `CRATE_DESTROYED` 和 `KILLED_OPPONENT` 事件奖励。这样减少模型为了预测奖励频繁放弹，而未产生真实比赛收益的问题。

### 4.21 r21_phase_potential：按箱子密度和首杀连续调整势能

在 r20 上加入箱子数量阈值 8/28，以及击杀前后、箱子稠密/稀疏时的金币、箱子、对手和危险势能权重。核心是让早期资源清理、中期追击和首杀后保优势使用不同权重。它依赖 v7 提供的可观察己方击杀状态。

### 4.22 r22_two_phase_potential：简化为首杀前后两阶段

同样从 r20 派生。首杀前保持 r20 的固定平衡势能；首杀后使用金币 0.9、箱子 0.1、对手 0.15、危险 1.5。与 r21 相比，它移除了基于箱子 8/28 阈值的稠密/稀疏细分，只保留“是否首杀”的两阶段切换，减少奖励非平稳性和超参数数量。

## 5. 参数家族的含义

| 家族 | 代表参数 | 作用 |
|---|---|---|
| 官方事件 | `coin_collected`, `crate_destroyed`, `killed_opponent` | 直接对齐比赛收益或资源进展 |
| 终局安全 | `killed_self`, `got_killed`, `survived_round` | 区分自杀、被杀和存活 |
| 势能 | `potential_*_weight`, `potential_gamma` | 给长时延目标提供稠密差分信号 |
| 放弹前信用 | `useful_bomb_per_crate`, `unsafe_bomb_penalty` | 根据当前模型预测评价 BOMB |
| 放弹后信用 | `causal_*`, `resolved_bomb_*` | 等炸弹周期结束后按真实结果归因 |
| 停滞控制 | `history_loop_*`, `history_wait_*` | 用有界历史处罚重复循环或 WAIT |
| 致命选择 | `avoidable_fatal_action`, `suicide_dominates_positive_events` | 针对有安全替代时的必死选择及奖励冲突 |
| 阶段化 | `phase_*` | 根据可观察局势调整不同目标的相对权重 |

## 6. 解释实验结果时的注意事项

- 不能只比较训练 reward：版本的量纲和组成不同，r21 的高低不与 r7 的训练 reward 直接可比。
- 应比较冻结、无探索的官方分数、金币、箱子、击杀、自杀、无效动作和延迟，并保持 v、mask、训练预算和 seeds 不变。
- r13–r16 名称中的 `no_safety` 描述预期实验条件；是否真的关闭 mask 仍由 config 的 `safety.mode` 决定。
- r21/r22 需要模型观察首杀状态；若搭配不含该状态的旧 v，奖励过程对 learner 会更像隐藏状态。
- 旧奖励 ID 为复现 checkpoint 保留，不代表仍推荐用于新训练。

## 7. 主要实现依据

- 注册参数和奖励计算：`agent_code/team_agent/rewards.py`
- 跨步奖励历史：`agent_code/learning_common/temporal_reward.py`
- 阶段奖励决策：`docs/adr/0005-use-observable-phase-rewards.md`
- outcome credit 设计记录：`.scratch/r20-outcome-credit/spec.md`
