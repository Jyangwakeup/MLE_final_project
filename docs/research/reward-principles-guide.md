# Bomberman Reward 方案与原理导读

> 更新日期：2026-09-15。本文覆盖 `agent_code/team_agent/rewards.py` 中当前注册的全部 Reward。
> Reward 的精确数值以源码和 `agent_code/team_agent/README.md` 为准。

## 1. Reward 在学习链路中的作用

Reward 把一次 transition 的结果压成标量，并进入 TD target：

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma Q(s',a')-Q(s,a)\right]$$

Feature 说明局面，模型估计长期动作价值，Reward 定义训练时鼓励什么。训练 reward 高不等于正式游戏得分高；所有 Reward 必须用冻结模型的金币、击杀、自杀率、胜率和完整 `act` 时延验证。

每条完成 transition 都先计 `step`。同一 transition 的死亡只计一次；使用 `death` 的旧版本不区分自杀和被杀，后续版本分别处理 `KILLED_SELF` 与 `GOT_KILLED`。

## 2. 当前全部 Reward ID

Reward 也可以先按对象理解。与 Feature 不同，它不负责完整描述对象，而是对已经发生的事件，
或相邻状态之间可验证的变化赋值。

### 2.1 按对象维度总览

| 维度 | 可奖励/惩罚的信号 | 使用这些信号的方案 | 明确边界 |
|---|---|---|---|
| 自身状态 | 每步成本、非法动作、存活回合、自杀、被击杀、振荡、连续等待、avoidable fatal action | 所有版本至少有步成本和死亡；`r2/r3/r4/r5_conditional/r6-r8` 可区分自杀/被杀；`r4/r5_conditional/r8` 有额外约束 | 不因“站在某个方向”直接给动作推荐分 |
| 金币 | 收集金币、发现金币、到最近可达金币的势能变化 | 所有版本奖励收集；`r2/r3/r4/r5_conditional` 奖励发现；`r3/r4/r5_coin_potential/r5_conditional/r6_safe_potential/r7_safe_credit_potential` 使用金币势能 | 隐藏金币不可用于提前奖励；训练 Reward 不等于正式 +1 比赛得分 |
| 箱子 | 实际炸毁箱子、到箱子 frontier 的势能变化、可逃生放弹时的预计覆盖箱数 | 多数版本奖励实际炸箱；`r3/r4/r5_conditional/r6-r8` 有箱区势能；`r6/r7` 有 useful-bomb credit | `r8` 删除预计 useful-bomb 正奖励，只保留实际结果和约束 |
| 对手 | 实际击杀对手 | 所有版本均为 `+5` | 当前没有接近/远离对手的 Reward，也没有“威胁到对手”预测奖励 |
| 自己放置炸弹 | 放弹后不能活过时域的惩罚；可逃且覆盖箱子的信用 | `r6/r7/r8` 惩罚不可逃放弹；仅 `r6/r7` 奖励预计覆盖箱数 | 放弹本身没有固定正奖励；`r8` 强调结果和生存约束 |
| 场上所有炸弹的时间/范围 | 通过最早危险时间进入安全/危险势能；通过时间展开搜索判断放弹或动作能否存活 | `r3/r4/r5_conditional` 使用安全势；`r6-r8` 使用危险势；`r8` 另罚 avoidable fatal action | Reward 不按炸弹身份分别计分；时间与范围是危险计算输入，不是独立事件奖金 |
| 对手炸弹 | 被对手击杀最终体现为 `GOT_KILLED`；炸弹覆盖也进入公共危险图 | 区分死亡原因的 `r2/r3/r4/r5_conditional/r6-r8`，以及所有使用危险势能的版本 | 因环境炸弹无 owner，不能可靠设置“躲开对手炸弹 +x” |

### 2.2 自身生存与行动效率

所有版本每个完成 transition 都有 `step=-0.01`，用于抑制无目的拖延；非法动作另行扣分。
`r1` 系列和 `r5_coin_potential` 把 `KILLED_SELF` 与 `GOT_KILLED` 合并成一次 `death`；
其余版本分别赋值。
`r4_anti_oscillation` 与 `r5_conditional_loop` 针对循环和可避免等待，`r8_safe_constrained`
则惩罚 **avoidable fatal action**：只有在至少存在一个 horizon-survivable alternative 时，
所选物理合法动作无法活过完整时域才触发。它不是对所有危险动作一概扣分。

### 2.3 金币与箱子

金币有三种层次的信号：`COIN_FOUND` 表示箱子被炸后金币首次出现，`COIN_COLLECTED`
表示实际收集，金币势能表示到最近可达金币的距离变化。箱子同样区分实际的
`CRATE_DESTROYED`、到箱子 frontier 的势能，以及 r6/r7 在放弹当下给出的预计覆盖信用。
势能是相邻状态差分，不是每次向目标移动都固定加分；往返后不会持续累积同方向奖金。

### 2.4 对手与击杀

当前所有 Reward 都只对框架确认的 `KILLED_OPPONENT` 给 `+5`。虽然 Feature 可以描述对手距离、
方向和“假设现在放弹会威胁几名对手”，Reward 没有使用接近对手或 predicted threatened opponents
作为正奖励。这一边界避免模型通过反复制造威胁而不完成击杀来刷分。

### 2.5 双方炸弹的时间与范围

Reward 使用炸弹信息的方式是评价**生存后果**，而不是给倒计时或范围本身定价。公共危险预测把
所有现存炸弹按各自 timer 和固定爆炸范围合并；由于 `game_state["bombs"]` 没有 owner，
无法可靠拆成“己方炸弹 Reward”和“对手炸弹 Reward”。

对于本步主动选择的 `BOMB`，系统知道这是自己的候选动作，所以 r6/r7/r8 能判断放置后是否可逃；
r6/r7 还按预计覆盖箱数给予有限信用。对已经存在于棋盘的炸弹，则无论来源，只通过危险/安全势能、
avoidable fatal action 或最终死亡事件影响 Reward。爆炸半径当前固定，双方没有可分别奖励的范围等级。

### 2.6 按版本总览

下面保留原有的 Reward 版本说明。按对象维度回答“金币、箱子、生存或炸弹分别怎样影响奖励”，
按版本维度则回答“一个 Reward ID 完整启用了哪些事件和塑形机制”。

| Reward ID | 基础事件设计 | 额外塑形 / 约束 | 定位 |
|---|---|---|---|
| `r1` | 金币 +1、击杀 +5、炸箱 +0.2、死亡 -10 | 无 | 历史基线、默认兼容 |
| `r1_no_crate` | `r1` 但炸箱 0 | 无 | 历史消融 |
| `r1_coin3` | `r1` 但金币 +3 | 无 | coin3 单因素消融 |
| `r1_coin3_no_crate` | `r1_coin3` 但炸箱 0 | 无 | 炸箱单因素消融 |
| `r2_balanced` | 金币 +3、发现 +0.25、击杀 +5、炸箱 +0.1、自杀 -7、被杀 -5、存活 +0.25 | 无 | 平衡事件基线 |
| `r3_potential` | 同 `r2_balanced` | 金币/箱区/安全势能 | 一般势能塑形 |
| `r4_anti_oscillation` | 同 `r2_balanced` | `r3` 势能 + 反复反向移动与连续空等惩罚 | 旧循环抑制实验 |
| `r5_coin_potential` | coin3 事件族 | 仅金币路径势能 | Task 1 目标塑形 |
| `r5_conditional_loop` | 同 `r2_balanced` | `r3` 势能 + 条件循环/可避免 WAIT 惩罚 | 条件循环实验 |
| `r6_safe_sparse` | 金币 +3、击杀 +5、炸箱 +0.2、自杀 -20、被杀 -10 | 箱区/危险势，不可逃放弹 -10，有效炸箱放弹信用 | 安全稀疏版 |
| `r6_safe_potential` | 同 `r6_safe_sparse` | 再加金币势能 | 安全势能版 |
| `r7_safe_credit_sparse` | 同 r6 事件族 | 不可逃放弹加重为 -20；无金币势 | 安全信用稀疏主线 |
| `r7_safe_credit_potential` | 同 r6 事件族 | r7 安全信用 + 金币势 | 当前资格迹 Agent 主线 |
| `r8_safe_constrained` | 同 r6 事件族 | 无预测 useful-bomb 正奖励；可避免必死动作 -20；自杀帧抑制正事件 | 当前 Rainbow Lite 主线 |

所有版本 `step=-0.01`。旧事件族的非法动作是 `-0.1`；`r2`、`r3`、`r4`、`r5_conditional_loop` 为 `-0.2`；r5 coin 与 r6–r8 为 `-0.1`。

## 3. 严格消融关系

以下对照每次只改变一个事件系数，适合归因：

```text
r1 -> r1_coin3                 只把金币 +1 改为 +3
r1 -> r1_no_crate              只把炸箱 +0.2 改为 0
r1_coin3 -> r1_coin3_no_crate  只把炸箱 +0.2 改为 0
```

`r1_no_crate` 与 `r1_coin3` 不能互称单因素版本，因为两者同时改变金币和炸箱。更高版本经常同时改变多个机制，比较时应写成“完整 Reward 方案比较”，不能把差异归因到一个系数。

## 4. 势能塑形

使用势能的版本添加：

$$F(s,s')=\gamma\Phi(s')-\Phi(s),\qquad \gamma=0.95$$

终局令下一状态势能为 0。接近度使用 `exp(-d/4)`；有可达金币时使用金币距离，否则使用箱子 frontier。`r3` 系列使用正安全势，r6–r8 使用负危险势。差分势能比“向目标走一步固定加分”更难被往返刷取，但仍必须通过冻结正式指标验证。

## 5. 时间与动作相关塑形

- `r4_anti_oscillation`：重复振荡 `-0.08`；从第二个连续 idle 起按 `-0.04` 递增，最多 3 倍。
- `r5_conditional_loop`：只在满足上下文条件时对循环或可避免 WAIT 各扣分；两者互斥。
- r6/r7：如果 `BOMB` 后不能活过时域，分别扣 `-10/-20`；可逃且覆盖箱子时每箱 `+0.2`，最多 3 箱。
- `r8_safe_constrained`：删除 predicted useful-bomb 奖励；若某动作是 **avoidable fatal action**，即时 `-20`。自杀帧中金币、炸箱、击杀等正事件被抑制，防止“带收益自杀”被净正奖励掩盖。

Reward 中的危险判定与 Safety mask 是两种机制：前者改变学习目标，后者是动作集合的 veto 边界。二者都必须写入 checkpoint/实验 metadata，不能只记录 Reward ID。

## 6. 当前方案对应关系

- Q-learning 历史/基线：`r1`、正式 coin3 对照使用 `r1_coin3`。
- DQN Task 1 矩阵：`r1_coin3` 与 `r5_coin_potential`；Task 2 比较相应安全版本。
- Continuous-v2 Task 2 已选胜者：`r7_safe_credit_sparse` + `survival-mask-v1/all`。
- Double Q(λ) 与 Expected SARSA(λ)：当前配置使用 `r7_safe_credit_potential`。
- Rainbow Lite：主配置使用 `r8_safe_constrained`；fallback 使用 `r7_safe_credit_sparse`。
- CNN/Hybrid/早期 continuous README 中的 `r2/r3` 是已实现研究分支，不代表当前推荐主线。

## 7. 版本纪律

Reward ID 与完整 spec 都是 checkpoint 契约。不能用旧 Reward checkpoint 冒充新 Reward 续训；消融应创建新 ID、独立配置并从零训练，或使用明确登记且兼容的迁移。不同奖励尺度下的训练曲线不可直接比较，选模统一依赖冻结评估。
