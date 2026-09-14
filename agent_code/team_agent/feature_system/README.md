# 公共 Feature System

本子包提供确定性、可版本化、可复用的 Bomberman 状态表示。它只描述环境事实和动作后果，不返回 `best_action`、规则分数或动作推荐；最终策略必须由 Q-learning、DQN 等学习算法产生。

## 公共接口与类型

```python
from agent_code.team_agent.feature_system import (
    available_feature_ids, extract_features, get_feature_schema,
)

features = extract_features(game_state, "continuous-v1")
schema = get_feature_schema("continuous-v1", game_state["field"].shape)
```

输出类型为 `DiscreteFeatures`、`VectorFeatures`、`BoardFeatures` 和 `HybridFeatures`。`FeatureSchema` 记录 Feature ID、输出种类、动作顺序、字段/通道名、shape、类别数、理论状态数与归一化规则，可写入 JSON metadata 和 checkpoint。

所有表示遵守：

- 数值数组为 `float32`，`legal_mask` 为 `(6,) bool`；
- 动作顺序固定为 `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`；
- `legal_mask` 只屏蔽物理非法动作，不屏蔽危险但可执行的动作；
- extractor 不修改输入，相同输入产生相同输出；
- `normalize_feature_id()` 只提供旧别名 `v1 -> discrete-v1`。

## 公共客观计算

`common.py` 为需要完整摘要的表示建立 `FeatureContext`，集中复用物理合法动作、爆炸范围、正常及假设放弹危险图、目标 BFS 距离、固定第一动作后的时间展开逃生搜索、最后安全区域，以及放弹覆盖的箱子和对手数量。`discrete-compact-v1` 使用不计算多余目标距离图的安全上下文；`board-v1` 只计算空间通道需要的合法动作与危险图。该分层只减少重复计算，不改变任何字段或通道语义。

五个移动/等待首动作的时间展开搜索使用一次批量棋盘传播；测试会逐项对照原来的单动作搜索结果。静态 BFS 使用整数访问图，返回值仍为原契约规定的 `float32` 距离和 `inf` 不可达语义。

静态 BFS 将石墙、箱子、炸弹和其他 Agent 视为障碍；自身格始终是合法起点。对手格可以作为距离终点，但不能作为通路。箱子 frontier 是与箱子正交相邻且当前可通行的格子。

### 危险时间语义

```text
HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 1 = 7
danger.shape = (HORIZON + 1, width, height)
```

| 切片 | 含义 |
|---|---|
| `danger[0]` | 不用于未来动作特征 |
| `danger[1]` | 下一步该格是否危险 |
| `danger[2]` | 两步后该格是否危险 |
| `danger[3]` | 三步后该格是否危险 |
| `danger[4:]` | 更长期预测 |

危险来自 `explosion_map`、现存炸弹和可选的自身假设炸弹。冻结的底层语义是：石墙阻断爆炸、箱子不阻断爆炸、不模拟连锁提前引爆、爆炸持续 `EXPLOSION_TIMER` 步。

`danger_t1/t2/t3` 描述固定动作后位置在准确未来时刻的危险。移动检查目的格；`WAIT` 检查当前格；`BOMB` 检查当前格并使用加入自身炸弹后的危险图。非法动作的危险位为 0，由 `legal` 区分。

### 时间展开可达性

`safe_horizon` 固定第一动作，然后允许后续继续移动：

- `safe_next`：第一步后是否安全；
- `survives_horizon`：能否活过完整预测时域；
- `safe_horizon`：仍存在安全路径的最远时间；
- `reachable_area`：最后安全时刻可到达的不同格子数。

新接口即使无法活完整时域也保留死亡前的最后安全 frontier。旧 `temporal_safety_features()` 保持旧语义：失败逃生的 `escape_area` 仍为 0。

## Feature ID 总览

| Feature ID | 适用模型 | 输出 | 手工摘要 | 旧 checkpoint |
|---|---|---|---|---|
| `discrete-v1` | 历史 DQN / Q-learning | state 14 / vector 40 | 是 | 兼容旧 `v1` |
| `discrete-q-v2` | Q-learning / DQN | state 16 / vector 50 | 是 | 不兼容 |
| `discrete-compact-v1` | Q-learning / Double Q | state 12 / vector 38 | 是 | 不兼容 |
| `continuous-v1` | MLP Double DQN | vector 70 | 是 | 不兼容 |
| `continuous-v2` | 历史感知 MLP Double DQN | vector 84 | 是 | 不兼容 |
| `board-v1` | CNN | board `12×W×H` | 主要为原始结构化通道 | 不兼容 |
| `hybrid-v1` | CNN + MLP | board `12×W×H` + vector 70 | 两者组合 | 不兼容 |

## `discrete-v1`：冻结基线

类别数 `(3,3,3,3,3,3,3,2,2,2,2,5,4,2)`，理论最大状态数 `1,399,680`，one-hot 为 40 维。

| 索引 | 字段 | 类别语义 |
|---:|---|---|
| 0–3 | 四方向移动安全 | 非法 / 无完整逃生 / 有完整逃生 |
| 4 | 当前格危险 | 无 / 下一步 / 两步及以后 |
| 5 | 放弹安全 | 不能放 / 放后不能逃 / 放后可逃 |
| 6 | 可炸箱桶 | 0 / 1 / 2 个以上 |
| 7–10 | 四方向接近金币 | 否 / 是 |
| 11 | 最近对手方向 | 无 / 上 / 右 / 下 / 左 |
| 12 | 对手 Manhattan 距离 | 无 / 1 / 2–4 / >4 |
| 13 | 对手在爆炸范围 | 否 / 是 |

此版本不增加独立的 `danger_t1/t2/t3`，保证既有 Q-table key 和 DQN 40 维权重不变。

## `continuous-v1`：70 维向量

不做旋转。每个动作占 10 个槽：

```text
legal, danger_t1, danger_t2, danger_t3, earliest_danger,
safe_horizon, safe_area, coin_distance_delta,
crate_frontier_distance_delta, opponent_distance_delta
```

| 动作 | 索引 |
|---|---:|
| UP | 0–9 |
| RIGHT | 10–19 |
| DOWN | 20–29 |
| LEFT | 30–39 |
| WAIT | 40–49 |
| BOMB | 50–59 |

归一化规则：

- `legal`、三个 danger、availability 为 0/1；
- `earliest_danger`：`t=1 -> 0`，`t=k -> (k-1)/HORIZON`，时域内无危险为 1，非法动作是 0；
- `safe_horizon/HORIZON`；
- `safe_area = reachable_area/(width*height)`；
- 距离变化为 `clip((before-after)/(width*height-1),-1,1)`，正数表示接近；
- 无目标或两边均不可达为 0，有限变不可达为 -1，反向为 +1，非法动作为 0。

| 索引 | 全局字段 | 归一化/语义 |
|---:|---|---|
| 60 | `reachable_coin_exists` | 是否有可达可见金币 |
| 61 | `reachable_crate_frontier_exists` | 是否有可达箱子 frontier |
| 62 | `reachable_opponent_exists` | 是否有路径可达对手 |
| 63 | `can_drop_bomb` | 当前能否放弹 |
| 64 | `escape_after_bomb` | 放后能否活完整时域 |
| 65 | `safe_area_after_bomb` | 放后最后安全区域/棋盘面积 |
| 66 | `crates_hit` | `count/(4*BOMB_POWER)` |
| 67 | `opponents_threatened` | `count/(MAX_AGENTS-1)` |
| 68 | `alive_opponents` | `count/(MAX_AGENTS-1)` |
| 69 | `round_progress` | `(clip(step,1,MAX_STEPS)-1)/(MAX_STEPS-1)` |

不能放弹时 64–67 均为 0。

## `continuous-v2`：84 维历史感知向量

前 70 维与 `continuous-v1` 完全一致，随后追加：上一动作的 7 类 one-hot（六动作及 none）、
当前确定性 BFS 金币目标是否延续，以及六个候选动作是否会返回前一位置。历史值由 Agent
回调提供；registry 的无历史直接调用使用 none/false，保持确定性和只读行为。

## `discrete-compact-v1`：12 个离散字段

类别数 `(3,3,3,3,2,2,2,3,4,5,4,4)`，理论最大状态数 `622,080`，one-hot 为 38 维。

| 索引 | 字段 | 类别语义 |
|---:|---|---|
| 0–3 | 规范坐标四方向移动安全 | 非法 / 无完整逃生 / 有完整逃生 |
| 4–6 | 当前格 `danger_t1/t2/t3` | 各自 0 / 1 |
| 7 | `bomb_safety` | 不能放 / 放后不能逃 / 放后可逃 |
| 8 | `objective_kind` | 无 / 金币 / 箱子 frontier / 对手 |
| 9 | `objective_direction` | 无或已到达 / 上 / 右 / 下 / 左 |
| 10 | `objective_distance` | 无 / 0–1 / 2–4 / 5+ |
| 11 | `bomb_utility` | 无 / 只炸箱 / 只威胁对手 / 两者 |

目标以 `(path_distance, kind_priority, target_x, target_y)` 选择，优先级为金币、箱子 frontier、对手；最短方向平局按上、右、下、左。它是状态摘要，不是动作推荐。

方形棋盘在 D4 的 8 个变换中选择客观签名字典序最小者；非方形棋盘只用 identity、180 度旋转、x 镜像、y 镜像。`state_key/vector` 使用规范坐标，`legal_mask` 始终是世界坐标。`ActionTransform` 提供动作槽双向映射，`WAIT/BOMB` 不变。

## `board-v1`：12 个棋盘通道

输出 `float32 (12,width,height)`，不转置坐标：`board[channel,x,y]` 对应 `field[x,y]`。

| 通道 | 名称 | 内容 |
|---:|---|---|
| 0 | `stone_wall` | `field == -1` |
| 1 | `crate` | `field == 1` |
| 2 | `coin` | 可见金币 |
| 3 | `self` | 自身位置 |
| 4 | `opponent` | 存活对手 |
| 5 | `bomb_presence` | 炸弹位置 |
| 6 | `bomb_timer` | `clip(timer/BOMB_TIMER,0,1)` |
| 7 | `current_explosion` | `explosion_map > 0` |
| 8 | `danger_t1` | `danger[1]` |
| 9 | `danger_t2` | `danger[2]` |
| 10 | `danger_t3` | `danger[3]` |
| 11 | `danger_t2_or_later` | `any(danger[2:HORIZON+1])` |

通道允许重叠。本版本不含目标方向、逃生路线或动作评分等策略摘要。

## `hybrid-v1`

输出 `board-v1` 的 `12×W×H` 与 `continuous-v1` 的 70 维向量。它只建立一次 `FeatureContext`，再调用两者的 `build_from_context()`；board、vector 和 legal mask 与分别调用两种 extractor 逐元素一致。

## Checkpoint 契约

新 checkpoint 保存 `feature_id`、完整 `feature_schema`、`reward_id` 和 `actions`，并暂时保留旧字段。加载时严格比较 ID、schema/shape 和动作顺序。旧 checkpoint 只有明确 `feature_version="v1"` 才映射为 `discrete-v1`；没有特征标识、未知 ID 或冲突配置均明确拒绝。当前两个 Agent 仍只接受 `discrete-v1`。
