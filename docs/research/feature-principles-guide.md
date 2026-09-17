# Bomberman Feature 方案与原理导读

> 更新日期：2026-09-15。本文覆盖当前 Feature registry 中的全部方案。
> 精确字段、shape 和兼容性以
> [`agent_code/team_agent/feature_system/README.md`](../../agent_code/team_agent/feature_system/README.md)
> 与源码为准；课程要求以 `PROJECT_REQUIREMENTS.md` 为准。

## 1. Feature 的职责边界

环境把结构化 `game_state` 交给 Agent，Feature extractor 再把它变成模型可消费的固定结构：

```text
game_state -> Feature -> 学习模型 -> 六个动作的 Q 值 -> 动作约束 -> 动作
```

Feature 描述局面和候选动作的可验证后果，例如物理合法性、未来危险、逃生空间和目标距离变化；它不返回 `best_action`，也不应以规则替模型决定方向。固定动作顺序为：

```python
("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
```

所有表示都返回 `(6,) bool` 的 `legal_mask`。它只表示 **physical legal action**，不声称动作安全或优秀。课程阶段对 `BOMB` 的禁用和 `survival-mask-v1` 是 Feature 之外的独立动作约束。

## 2. 原始状态与公共计算

核心输入包括 `field[x,y]`、`bombs`、`explosion_map`、`coins`、`self`、`others` 和 `step`。公共计算集中在 `feature_system/common.py`：

- 物理合法动作；
- 现有炸弹及“现在放弹”两种情况下的未来危险图；
- 到金币、箱子 frontier 和对手的 BFS 距离；
- 固定第一动作后的时间展开逃生搜索；
- 最后安全 frontier、放弹覆盖箱数和威胁对手数。

静态 BFS 把石墙、箱子、炸弹和其他 Agent 当作障碍；箱子目标用可站立的相邻 frontier 表示。危险张量为：

$$D\in\{0,1\}^{(H+1)\times W\times H_b},\qquad H=7$$

`D[t,x,y]=1` 表示 `(x,y)` 在未来第 `t` 步危险。当前冻结语义是：石墙阻断爆炸、箱子不阻断爆炸、不模拟连锁提前引爆。

逃生搜索在 `(x,y,t)` 上展开，产生 `safe_next`、`safe_horizon`、`survives_horizon` 和 `reachable_area`。因此“下一步没爆炸”和“存在一条能活过完整预测时域的后续路线”是两个不同事实。

## 3. 当前全部 Feature 方案

在按版本阅读之前，可以先按“局面中的对象”理解 Feature。这里区分四类信息：

- **原始观测**：`game_state` 直接提供的位置、倒计时或状态；
- **派生事实**：由 BFS、爆炸模拟或时间展开搜索计算出的距离、危险和可达性；
- **动作条件事实**：分别回答执行六个候选动作后会发生什么；
- **历史事实**：环境当前帧没有提供、由 Agent 跨步维护的信息。

### 3.1 按对象维度总览

| 维度 | 原始状态 | 当前派生信息 | 主要出现在哪些 Feature | 当前没有表达的内容 |
|---|---|---|---|---|
| 自身状态 | 位置、分数、能否放弹、当前步数 | 物理合法动作、当前位置未来危险、各动作的完整时域存活性、逃生余量、最后安全区域 | 所有方案都有部分自身/动作信息；`continuous-v1/v2/v3` 最完整 | 没有把某个安全动作直接标成“最佳动作” |
| 金币 | 所有**可见**金币的位置 | 最近可达金币的 BFS 距离、各动作的距离变化、目标是否延续 | 离散系列、连续系列、`board-v1` | 隐藏在箱子中的金币位置不可观测，不能作为 Feature |
| 箱子 | 箱子位置 | 可站立的箱子 frontier、到 frontier 的距离、现在放弹可覆盖的箱数 | 离散系列、连续系列、`board-v1` | 不预测某个箱子内是否有金币 |
| 对手 | 存活对手的位置及其公开状态 | 最近对手方向/距离、各动作的距离变化、现在放弹可威胁的对手数 | `discrete-v1/q-v2/compact-v1`、连续系列、`board-v1` | 不预测对手策略、下一动作或未来放弹意图 |
| 所有炸弹 | 位置、当前倒计时 | 每颗炸弹按固定爆炸规则产生的未来危险图；各候选动作在 `t1/t2/t3` 及完整时域中的风险 | 连续系列、`board-v1`，离散系列为压缩桶 | 环境不提供炸弹所有者，所以不能从单帧可靠区分己方/对方炸弹 |
| 自己的炸弹 | 当前帧没有所有权字段 | `continuous-v3` 结合 Agent 历史维护最近己方炸弹是否 pending/visible、倒计时、自身是否位于其爆炸轴上 | 仅 `continuous-v3` 的末 4 个 own-bomb 字段（另有全局安全动作比例） | 不支持同时追踪多颗己方炸弹；当前规则下通常一次只能有一颗 |
| 对手的炸弹 | 无独立字段 | 只能作为“所有炸弹”共同进入危险图 | 所有使用危险图的方案 | 不能可靠提取“对手炸弹倒计时”或按对手身份归因 |
| 爆炸范围 | 固定规则，不是 `game_state` 中的可升级属性 | 从炸弹位置沿四个正交方向展开，石墙阻断；形成未来各时刻的危险格 | `board-v1` 显式为空间通道；其他方案压缩为动作危险与逃生事实 | 没有单独的“己方范围/对方范围”数值，因为当前双方范围相同且没有 power-up |

### 3.2 自身状态与候选动作

Feature 的核心不是只记录“我在哪里”，而是把同一局面对六个候选动作分别展开。`legal_mask`
只回答动作在物理上能否执行；`danger_t1/t2/t3` 回答动作落点在准确未来时刻是否危险；
`safe_horizon`、`survives_horizon`、`reachable_area` 和 `escape_slack` 则回答动作之后是否仍存在逃生路线。
因此“当前没踩火”“下一步安全”和“能活过完整预测时域”不能互换。

`continuous-v2` 还加入上一动作、是否返回前一位置和金币目标连续性，供模型识别循环；
`continuous-v3` 再加入每个动作的完整存活标记、逃生余量和第二步可存活分支比例。
这些都是可验证事实，不替模型排序动作。

### 3.3 金币、箱子与对手

三类目标采用不同语义：金币以可见金币格为目标；箱子本身不可进入，所以以可站立的相邻
`frontier` 为目标；对手格可作为距离终点，但不能作为路径中间格。连续 Feature 对每个动作记录
距离变化，正值表示接近，负值表示远离。离散 Feature 则用方向、距离桶或统一 objective 桶压缩。

目标层级也并非所有版本一致：`discrete-objective-v1` 只在没有可达金币时转向箱子 frontier；
`discrete-compact-v1` 在金币、箱子 frontier、对手之间按路径距离和固定平局规则选一个客观目标；
连续系列同时保留三张距离图，不强迫模型只关注其中一种。

### 3.4 双方炸弹的时间与范围

`game_state["bombs"]` 的每项只有 `(position, timer)`，没有 owner。公共计算据此生成
`D[t,x,y]`：它同时汇总场上所有现存炸弹、当前爆炸，以及评估 `BOMB` 动作时假设新增的
自身炸弹。倒计时决定危险出现的时间，固定 `BOMB_POWER` 决定四向覆盖范围；石墙阻断爆炸。

因此当前 Feature 能回答：某格何时会危险、某动作是否进入爆炸覆盖、放弹后能否逃生、
放弹会覆盖多少箱子或对手。它不能仅凭当前帧回答“这颗炸弹属于哪个对手”。
`continuous-v3` 的 own-bomb 历史只补足自己最近一次放置的炸弹；其余炸弹仍统一视为环境危险。
如果未来确实需要区分每名对手的炸弹，必须新增跨步所有权追踪、版本化 Feature ID 和相应测试，
不能改变现有 ID 的语义。

### 3.5 按版本总览

下面保留原有的 Feature 版本说明。按对象维度用于横向查看“某类信息如何表示”，按版本维度用于查看
“一个 Feature ID 完整包含什么”。两种组织方式互为索引，不互相替代。

| Feature ID | 输出 | 主要用途 | 状态 |
|---|---|---|---|
| `discrete-v1` | 14 项离散状态 / 40 维 one-hot | 历史 Q-learning、DQN 对照 | 冻结兼容；旧别名 `v1` |
| `discrete-q-v2` | 16 项 / 50 维 one-hot | 当前 Q-learning、DQN 默认 | 当前基线 |
| `discrete-objective-v1` | 14 项 / 60 维 one-hot | 分层目标与动作历史实验 | 当前可用 |
| `discrete-compact-v1` | 12 项 / 38 维 one-hot | D4 对称压缩、Double Q | 当前可用 |
| `continuous-v1` | 70 维向量 | 早期 MLP Double DQN | 当前可用 |
| `continuous-v2` | 84 维向量 | 历史感知 MLP、线性资格迹 Agent | 当前主线 |
| `continuous-v2-legacy78` | 78 维向量 | 旧 Double DQN checkpoint | 仅冻结加载 |
| `continuous-v3` | 107 维向量 | 显式动作生存事实、安全消融、Rainbow Lite | 当前主线 |
| `board-v1` | `12×W×H` | CNN Double DQN | 当前可用 |
| `hybrid-v1` | `12×W×H + 70` | CNN/MLP 融合 Dueling DDQN | 当前可用、非当前主线 |

### 3.6 各版本详细说明

#### 离散系列

`discrete-v1` 是不再改语义的历史契约，理论最大状态数 1,399,680。`discrete-q-v2` 在基线上增加更适合当前表格/小型 DQN 的事实，成为两者新训练默认。`discrete-objective-v1` 编码四方向相对分层目标进展、当前危险、放弹结果、目标类型/距离、上一动作和连续等待；它描述相对变化，不给出唯一推荐动作。

`discrete-compact-v1` 的理论最大状态数为 622,080。它把金币、箱子 frontier、对手统一为客观目标，并对方形棋盘使用 D4 八种旋转/镜像的规范表示。Q 表动作槽位在规范坐标中，实际动作会双向映射；`WAIT` 和 `BOMB` 不变。

#### 连续系列

`continuous-v1` 为每个动作编码 10 个值：

```text
legal, danger_t1, danger_t2, danger_t3, earliest_danger,
safe_horizon, safe_area, coin_distance_delta,
crate_frontier_distance_delta, opponent_distance_delta
```

六个动作共 60 维，再加目标可达性、放弹能力与效果、存活对手数、回合进度等 10 个全局量。距离变化为正表示接近目标，所有尺度都被归一化以利于梯度学习。

`continuous-v2` 保留前 70 维并追加上一动作 7 类 one-hot、金币目标连续性，以及六个候选动作是否返回前一位置，共 84 维。`continuous-v2-legacy78` 只用于忠实加载旧谱系。

`continuous-v3` 扩展到 107 维：为各动作加入能否活过 `H=7`、危险期限相对最短逃生步的余量、第二步完整生存分支比例，并加入全局安全动作比例及自身最近炸弹事实。它仍是事实表示，不是规则策略。

#### 棋盘与混合系列

`board-v1` 的 12 个通道依次表示石墙、箱子、金币、自己、对手、炸弹位置、炸弹时间、当前爆炸、`danger_t1`、`danger_t2`、`danger_t3` 和 `danger_t2_or_later`。默认棋盘 shape 为 `(12,17,17)`，坐标保持 `board[channel,x,y]`。

`hybrid-v1` 一次公共计算同时产生 `board-v1` 和 `continuous-v1`。典型网络让 CNN 学空间模式、MLP 消费人工摘要，再融合预测 Q 值。

## 4. Feature、Safety 与学习策略

当前项目明确区分三层：

1. `legal_mask` 移除物理上不能执行的动作；
2. 可选的 `survival-mask-v1/all` 只 veto 无法活过完整时域的动作；若没有 horizon-survivable action，则回退到物理合法动作中的学习 Q argmax；
3. 学习模型在剩余动作中排序。

安全掩码同时用于探索、贪心行为、冻结推理和 bootstrap target，避免训练/评估语义漂移。它不提供“最佳安全动作”，因此仍由模型学习选择。

## 5. 版本与 checkpoint 契约

Feature ID 是模型契约的一部分。新 checkpoint 保存 `feature_id`、完整 `feature_schema`、动作顺序、Reward 和 Safety 契约；加载时严格比较字段、shape 与语义。即使维度相同，只要字段意义变化，也必须创建新 ID 并从零训练或按明确迁移协议处理，不能静默复用旧权重。

## 6. 代码阅读顺序

1. `feature_system/types.py`：输出类型与 schema；
2. `feature_system/registry.py`：当前全部 ID；
3. `feature_system/common.py`：危险、BFS 与逃生事实；
4. `continuous_v2.py`、`continuous_v3.py`：当前连续主线；
5. `discrete_q_v2.py`、`discrete_objective_v1.py`：当前离散主线；
6. `board_v1.py`、`hybrid_v1.py`：空间表示；
7. `safety.py`：Feature 之外的生存 veto。
