# team_agent 公共模块说明

本目录存放团队学习型 agent 共用的特征和基础奖励接口。它目前还不是一个完整可运行的 agent，因为目录里还没有 `callbacks.py` 和 `train.py`。这里的代码主要定义统一的状态表示与基础奖励，供不同模型和实验变体复用。

## 文件职责

| 文件 | 作用 |
|---|---|
| `features.py` | 对外公开的特征接口，把安全、金币、对手三类信息组合成统一表示。 |
| `danger.py` | 根据当前炸弹、当前爆炸和可选的假设炸弹，预测未来爆炸危险格。 |
| `temporal_safety_features.py` | 在时间维度上模拟移动安全性和逃生路径。 |
| `rewards.py` | 将框架事件转换为团队统一的基础标量奖励。 |
| `__init__.py` | Python 包标记文件。 |

## 对外接口

主要使用 `features.py` 里的 `extract_features(game_state)`：

```python
from agent_code.team_agent.features import ACTIONS, FEATURE_VERSION, extract_features

features = extract_features(game_state)
state_key = features.state_key
vector = features.vector
legal_mask = features.legal_mask
```

返回值是：

```python
Features(state_key, vector, legal_mask)
```

三个字段的含义：

- `state_key`：14 个离散值组成的 tuple，适合作为 tabular Q-learning 的 Q-table key。
- `vector`：长度为 40 的 `float32` one-hot 向量，适合 DQN 或其他向量模型。
- `legal_mask`：固定动作顺序下的布尔合法动作 mask。

固定动作顺序是：

```python
('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
```

当前特征版本是：

```python
FEATURE_VERSION = 'v1'
```

训练出的模型或 Q-table 应该保存这个版本号。加载 checkpoint 时，如果版本不一致，应该拒绝加载或重置模型，避免把旧特征训练出的参数误用于新特征。

## 特征结构

`FEATURE_CATEGORY_COUNTS` 定义每个离散字段的类别数：

```python
(3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 5, 4, 2)
```

因此：

- `STATE_KEY_SIZE = 14`
- `FEATURE_DIM = 40`

14 个 `state_key` 字段含义如下：

| 索引 | 分组 | 含义 | 类别数 |
|---:|---|---|---:|
| 0-3 | 移动安全 | 对 `UP`、`RIGHT`、`DOWN`、`LEFT` 分别编码：非法 / 合法但没有完整逃生路线 / 合法且有逃生路线 | 每项 3 |
| 4 | 当前危险 | 当前格安全 / 下一步危险 / 之后会危险 | 3 |
| 5 | 放弹安全 | 不能放弹 / 能放但放后逃不掉 / 能放且放后能逃 | 3 |
| 6 | 炸箱覆盖 | 当前放弹可炸 0 / 1 / 2 个及以上箱子 | 3 |
| 7-10 | 金币导航 | 对四个移动方向分别编码：该方向是否更接近最近可达金币 | 每项 2 |
| 11 | 对手方向 | 无对手 / 上 / 右 / 下 / 左，按主导位移方向编码 | 5 |
| 12 | 对手距离 | 无对手 / 相邻 / 较近 / 较远 | 4 |
| 13 | 对手爆炸覆盖 | 当前放弹的爆炸范围是否能覆盖某个对手 | 2 |

`vector` 是这些离散类别拼接后的 one-hot 表示：

- 安全相关：7 个字段 x 3 类 = 21 维
- 金币导航：4 个字段 x 2 类 = 8 维
- 对手相关：5 + 4 + 2 = 11 维
- 总计：40 维

## 安全判断语义

这里的安全特征比简单的“这个格子会不会被炸”更保守，也更适合学习放弹和逃生。

`danger.py` 使用如下时间范围预测未来危险：

```python
HORIZON = settings.BOMB_TIMER + settings.EXPLOSION_TIMER + 1
```

它会标记：

- `explosion_map` 中已经存在的爆炸
- 当前场上炸弹未来爆炸时覆盖的格子
- 可选的“如果我现在放一颗炸弹”产生的未来危险

`temporal_safety_features.py` 在此基础上构建按时间展开的可通行地图。它会考虑：

- 墙和箱子阻挡
- 其他 agent 占位阻挡
- 炸弹在爆炸前阻挡所在格
- 箱子被预测炸毁后变为可通行
- 每个未来时间步的危险格

对每个第一步动作，它会检查：执行这个动作后，agent 能不能在整个预测时间范围内持续找到安全位置。这个结果就是 `features.py` 中“是否存在逃生路线”的来源。

## legal_mask 的含义

`legal_mask` 只表示物理合法性，不会因为动作危险就把它标为非法。

规则是：

- 移动到墙、箱子、炸弹或其他 agent 所在格是非法的。
- `WAIT` 总是物理合法。
- `BOMB` 只有在 `game_state['self'][2]` 表示可以放弹时才合法。

危险动作仍然可能是合法动作。例如站在即将爆炸的格子上选择 `WAIT`，物理上可以等待，但策略上很危险。这个危险信息由安全特征告诉模型，而不是从 `legal_mask` 里删除。这样可以把“动作是否能执行”和“动作是否聪明”分开。

## 和其他 agent 的关系

`q_learning_agent` 现在已经通过兼容包装复用本目录的团队特征接口：

- `state_to_features(game_state)` 返回 `extract_features(game_state).state_key`
- `legal_actions(game_state)` 返回 `extract_features(game_state).legal_mask`

这样 Q-learning 和后续 DQN 可以在同一套特征语义下比较，避免“算法不同”和“特征不同”混在一起，导致实验结论不清楚。

## 公共基础奖励接口

`rewards.py` 提供版本化的基础奖励函数：

```python
from agent_code.team_agent.rewards import REWARD_VERSION, reward_from_events

reward = reward_from_events(events)
```

当前版本为 `REWARD_VERSION = 'base-v1'`，奖励规则如下：

| 条件或事件 | 奖励 |
|---|---:|
| 每一步 | `-0.01` |
| `COIN_COLLECTED` | `+1.0` |
| `KILLED_OPPONENT` | `+5.0` |
| `CRATE_DESTROYED` | `+0.2` |
| `SURVIVED_ROUND` | `+1.0` |
| `INVALID_ACTION` | `-0.2` |
| `KILLED_SELF` 或 `GOT_KILLED` | `-10.0`，同一次死亡只计算一次 |

同一步中的可重复事件会分别累计。公共奖励只描述环境结果，不根据某个方向是否“正确”直接奖励动作。Q-learning 已使用该接口；其他模型如果采用不同奖励，应显式记录奖励版本或变体名称，避免把模型差异和奖励差异混在一起。

## 开发约定

- 不要在不更新 `FEATURE_VERSION` 的情况下改变 `ACTIONS`、`STATE_KEY_SIZE`、`FEATURE_CATEGORY_COUNTS` 或任何已有字段语义。
- 不要在不更新 `REWARD_VERSION` 的情况下改变已有基础奖励的数值或语义。
- `extract_features(game_state)` 必须保持确定性：同一个输入状态应产生同一个输出。
- 不要修改传入的 `game_state`。
- 新特征变体应显式版本化，或者放在清晰命名的变体中，让实验配置能记录 `feature_version`。
- 如果某个实验只想测试一个因素，例如去掉危险特征或去掉炸箱奖励，应尽量只改变这个因素，保持其他编码不变。
