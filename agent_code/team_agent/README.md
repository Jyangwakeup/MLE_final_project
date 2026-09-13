# team_agent 公共特征与奖励

本目录包含可版本化的公共 Feature System 和已有奖励接口。Q-learning 与 DQN 的
`discrete-v1`（旧名 `v1`）特征仍保持冻结；历史 `r1` checkpoint 继续兼容，但新训练只使用
`r2_balanced` 或 `r3_potential`。

公共 registry 还服务于四个新学习型调用方：`double_q_compact_agent` 使用
`discrete-compact-v1`，`double_dqn_continuous_agent` 使用 `continuous-v1`，
`cnn_double_dqn_agent` 使用 `board-v1`，`hybrid_dueling_double_dqn_agent` 使用
`hybrid-v1`。各 Agent 只保留薄适配层；训练代码不会回写 Feature 输出，最终提交由构建器
将本公共包 vendor 到单个 Agent 目录。

详细的字段、通道、归一化、危险时间及 checkpoint 契约见 [`feature_system/README.md`](feature_system/README.md)。初代方案的历史固化记录见 [`docs/research/initial-feature-reward-scheme.md`](../../docs/research/initial-feature-reward-scheme.md)。

## Feature API

```python
from agent_code.team_agent.feature_system import (
    available_feature_ids, extract_features, get_feature_schema,
)

features = extract_features(game_state, "continuous-v1")
schema = get_feature_schema("continuous-v1", game_state["field"].shape)
```

| Feature ID | 适用模型 | 输出 | 理论离散状态数 | 旧 checkpoint |
|---|---|---|---:|---|
| `discrete-v1` | 历史 DQN / Q-learning | 14 state / 40 vector | 1,399,680 | 兼容旧 `v1` |
| `discrete-q-v2` | Q-learning / DQN | 16 state / 50 vector | 34,992,000 | 不兼容 |
| `discrete-compact-v1` | Q-learning / Double Q | 12 state / 38 vector | 622,080 | 不兼容 |
| `continuous-v1` | MLP Double DQN | 70 vector | — | 不兼容 |
| `board-v1` | CNN | `12×W×H` | — | 不兼容 |
| `hybrid-v1` | CNN + MLP | `12×W×H + 70` | — | 不兼容 |

固定动作顺序为 `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`。所有类型都包含 `(6,) bool legal_mask`；它只表示物理合法性，不过滤危险动作。数值数组为 `float32`，提取确定且不修改输入。

旧 import 路径继续可用：

```python
from agent_code.team_agent.features import (
    ACTIONS, FEATURE_ID, FEATURE_VERSION, extract_features,
)
```

这个兼容层固定返回 `discrete-v1`，供 DQN 和旧调用方继续使用。Q-learning v2 通过 registry
明确选择 `discrete-q-v2`。

## 1–3 步危险

新表示明确区分准确未来时刻：

- `danger_t1`：下一步该格危险；
- `danger_t2`：两步后该格危险；
- `danger_t3`：三步后该格危险。

`continuous-v1` 对六个动作分别提供三项；`board-v1` 提供三张全棋盘通道；`discrete-compact-v1` 提供当前位置三个位；`hybrid-v1` 同时包含 board 和 continuous。冻结的 `discrete-v1` 继续使用“下一步 / 两步及以后”的旧聚合类别。

固定格危险只回答“如果仍在动作后位置，该时刻是否危险”。`safe_horizon` 和 `safe_area` 允许第一步以后继续移动，分别表达安全路径能维持多久、最后安全时刻能到达多少格，两者不能混用。

## 表示类型边界

- 原始结构化棋盘通道：`board-v1` 中的墙、箱子、金币、角色、炸弹、爆炸和危险图。
- 手工摘要特征：离散方案和 `continuous-v1` 中的 BFS 距离变化、时间展开生存、放弹覆盖等人工统计。
- 物理合法 mask：只防止执行撞墙、进入箱子/炸弹/Agent 格或不可用炸弹等非法动作。
- 人工决策规则：Feature System 不包含，不输出最佳动作或确定性策略。

所以当前方案是“手工特征 + 学习策略”，不是端到端表示学习，也不是手写规则 Agent。

## 公共奖励

`rewards.py` 当前注册五个 Reward，其中 `r1` 和 `r1_no_crate` 只为历史兼容保留。所有方案每条 transition 都先计一次 step cost；表中的
“—”表示该事件没有额外奖励。

| Reward ID | step | 收金币 | 发现金币 | 击杀 | 炸箱 | 自杀 | 被杀 | 存活 | 非法动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1` | -0.01 | +1.0 | — | +5.0 | +0.2 | -10.0 | -10.0 | — | -0.1 |
| `r1_no_crate` | -0.01 | +3.0 | — | +5.0 | 0.0 | -10.0 | -10.0 | — | -0.1 |
| `r2_balanced` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |
| `r3_potential` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |
| `r4_anti_oscillation` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |

### `r1`：历史冻结基线

奖励尺度最接近官方分数：金币 +1、击杀 +5。它只用于解释或加载已有实验和 checkpoint，
不再用于新训练。其导航信号稀疏；普通移动只有 step cost，表格 Agent 可能难以把远处金币
奖励传播回来。

### `r1_no_crate`：历史金币优先且不奖励炸箱方案

收金币提高到 +3，炸箱为 0。由于金币值也不同于 `r1`，它不是“只关闭炸箱”的严格单因素
消融。当前只为读取历史记录保留，不再用于新训练，也不再提供独立实验配置。

### `r2_balanced`：重新平衡事件结果

收金币为 +3、发现金币为 +0.25、炸箱降至 +0.1。它区分自杀 -7 与被对手击杀 -5，并给
存活 +0.25、非法动作 -0.2。相比 `r1`，它更适合包含箱子和对手的任务，但仍没有逐步接近
目标的密集反馈。

### `r3_potential`：平衡事件奖励加状态势能

事件部分与 `r2_balanced` 相同，并增加：

```text
F(s,s') = 0.95 * Phi(s') - Phi(s)
```

`Phi` 优先使用最近可达金币的 `0.5 * exp(-distance/4)`；没有可达金币时使用箱子 frontier
的 `0.25 * exp(-distance/4)`，再加最多 0.25 的当前位置安全度。终局下一状态势能固定为
0。它只比较前后状态，不按动作名称直接指定策略，适合缓解稀疏奖励下的原地等待或往返循环。
代价是训练时需要额外计算路径和危险上下文。

### `r4_anti_oscillation`：距离势能加连续反转惩罚

继承 `r3_potential`，并在出现第三个动作构成 `LEFT-RIGHT-LEFT`、
`RIGHT-LEFT-RIGHT`、`UP-DOWN-UP` 或 `DOWN-UP-DOWN`，且最近金币距离没有缩短时增加
`-0.08`。连续安全空等从第二次起额外增加 `-0.04`，随后为 `-0.08`、`-0.12` 并封顶；
有未来爆炸危险或没有可达目标时不触发。一次正常回头、首次等待、避弹转向或伴随金币
收集的移动不触发额外惩罚。

实验默认 Reward 是 `r2_balanced`。底层 registry 仍以 `r1` 作为无配置旧调用的兼容回退，
但正式训练必须通过 `experiments.run` 或环境变量明确选择 `r2_balanced`/`r3_potential`。
Feature 与 Reward 是独立契约；metadata/checkpoint 同时记录 `feature_id` 与 `reward_id`。
更换 Reward 必须开启新的训练链，不能接续其他 Reward 的 checkpoint。训练 reward 的数值
尺度不同，不能直接横向比较；最终应比较冻结评估中的正式得分、金币、击杀、存活率和非法
动作数。

## 文件职责

| 路径 | 职责 |
|---|---|
| `features.py` | 冻结 `discrete-v1` 的旧 import 兼容层 |
| `feature_system/` | 类型、registry、共享上下文、对称变换和五种表示 |
| `danger.py` | 公共爆炸范围与未来危险图 |
| `temporal_safety_features.py` | 公共时间展开搜索，并保留旧接口 |
| `rewards.py` | 五个版本化奖励的 registry |

## 版本规则

- Feature 字段、类别、shape、通道顺序或语义变化时创建新 Feature ID。
- Reward 数值或语义变化时创建新 Reward ID。
- 新 checkpoint 严格校验 Feature ID、schema/shape 和动作顺序；只有明确旧 `v1` 可映射到 `discrete-v1`。
- `hybrid-v1` 必须组合 board/continuous 公共实现，不能复制危险与路径逻辑。
- `legal_mask`、手工状态摘要与最终 policy 始终分离。
