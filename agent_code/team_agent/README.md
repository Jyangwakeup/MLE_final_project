# team_agent 公共特征与奖励

本目录提供所有学习型 Agent 共用、可版本化的特征与奖励契约。各 Agent 只保留薄适配层；打包工具会把所需公共代码复制到最终单 Agent 包中。

## Feature System

```python
from agent_code.team_agent.feature_system import extract_features, get_feature_schema

features = extract_features(game_state, "discrete-q-v2")
schema = get_feature_schema("discrete-q-v2", game_state["field"].shape)
```

| Feature ID | 主要模型 | 输出 | 兼容性 |
|---|---|---|---|
| `discrete-v1` | Q-learning / DQN 冻结对照 | 14 项状态、40 维向量 | 兼容旧 `v1` checkpoint |
| `discrete-q-v2` | Q-learning / DQN 默认 | 16 项状态、50 维向量 | 不兼容 v1 |
| `discrete-compact-v1` | Double Q | 12 项状态、38 维向量 | 独立契约 |
| `continuous-v1` | MLP Double DQN | 70 维向量 | 独立契约 |
| `continuous-v2` | Double DQN 安全消融基线 | 目标连续性与位置历史，84 维 | v6 新谱系 |
| `continuous-v2-legacy78` | 旧 Double DQN | 距离增量与动作历史，78 维 | 仅冻结加载 |
| `continuous-v3` | Double DQN 安全候选 | v2 加逐动作生存、逃生余量/分支和自身炸弹事实，107 维 | v6 新谱系 |
| `board-v1` | CNN Double DQN | `12×W×H` | 独立契约 |
| `hybrid-v1` | Hybrid Dueling Double DQN | `12×W×H + 70` | 独立契约 |

固定动作顺序为 `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`。所有表示都返回 `(6,) bool legal_mask`；它只表达物理合法性，不屏蔽危险但可执行的动作。数值数组为 `float32`，提取过程确定且不修改输入。

旧路径 `agent_code.team_agent.features` 固定提供 `discrete-v1`，用于历史 checkpoint。Q-learning 和 DQN 在没有显式配置或 checkpoint 时默认 `discrete-q-v2`；实验配置可显式选择 `discrete-v1`，checkpoint 加载则采用并校验其完整 Feature 契约。

危险表示精确区分未来 1–3 步，并共享墙、箱子、炸弹、爆炸、其他 Agent 占位及时间展开逃生搜索。具体字段、shape、归一化和对称变换见 [`feature_system/README.md`](feature_system/README.md)。

## Reward registry

所有奖励每条完成 transition 先计 step cost；死亡事件在同一步只计一次。

| Reward ID | step | 金币 | 发现金币 | 击杀 | 炸箱 | 自杀 | 被杀 | 存活 | 非法动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1` | -0.01 | +1.0 | — | +5.0 | +0.2 | -10.0 | -10.0 | — | -0.1 |
| `r1_no_crate` | -0.01 | +1.0 | — | +5.0 | 0.0 | -10.0 | -10.0 | — | -0.1 |
| `r1_coin3` | -0.01 | +3.0 | — | +5.0 | +0.2 | -10.0 | -10.0 | — | -0.1 |
| `r1_coin3_no_crate` | -0.01 | +3.0 | — | +5.0 | 0.0 | -10.0 | -10.0 | — | -0.1 |
| `r2_balanced` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |
| `r3_potential` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |
| `r4_anti_oscillation` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |
| `r5_conditional_loop` | -0.01 | +3.0 | +0.25 | +5.0 | +0.1 | -7.0 | -5.0 | +0.25 | -0.2 |

`r1_coin3` 相对 `r1` 只改变金币奖励；`r1_coin3_no_crate` 相对它只关闭炸箱奖励，因此可作为严格单因素消融。`r3_potential` 在 `r2_balanced` 事件奖励上增加状态势能；`r4_anti_oscillation` 再增加旧的连续反向移动和安全空等惩罚。`r5_conditional_loop` 从 `r3_potential` 分叉，改用条件式循环和可避免 WAIT 惩罚，不叠加 `r4` 的旧惩罚。

| `r5_coin_potential` | -0.01 | +3.0 | — | +5.0 | +0.2 | -10.0 | -10.0 | — | -0.1 |
| `r6_safe_sparse` | -0.01 | +3.0 | — | +5.0 | +0.2 | -20.0 | -10.0 | — | -0.1 |
| `r6_safe_potential` | -0.01 | +3.0 | — | +5.0 | +0.2 | -20.0 | -10.0 | — | -0.1 |
| `r7_safe_credit_sparse` | -0.01 | +3.0 | — | +5.0 | +0.2 | -20.0 | -10.0 | — | -0.1 |
| `r7_safe_credit_potential` | -0.01 | +3.0 | — | +5.0 | +0.2 | -20.0 | -10.0 | — | -0.1 |
| `r8_safe_constrained` | -0.01 | +3.0 | — | +5.0 | +0.2 | -20.0 | -10.0 | — | -0.1 |

`r5` 使用金币路径势；r6 增加箱区/危险势、可逃有收益放弹信用和不可逃放弹 −10；r7 将不可逃放弹改为 −20。r8 删除预测 useful-bomb 正奖励，对有安全替代的必死动作即时扣 20，并在自杀同帧抑制金币、炸箱和击杀正奖励。势函数统一使用 `γΦ(s')−Φ(s)`，不直接奖励移动、等待、普通放弹、发现或存活事件。

Feature、Reward 与 Safety 是独立契约。`survival-mask-v1/all` 对探索、贪心、冻结推理和 Double DQN target 使用同一 H=7 生存集合；无安全动作时回退物理合法 Q argmax。`survival-mask-v5` 在准备放弹及自身炸弹危险期进一步计算对手鲁棒的可控生存集合：第一步精确执行官方转移，后续用保守占位/爆炸集合反向求可生存区域。它只否决动作，不修改剩余动作的 Q 排序。新实验的metadata、final checkpoint和`training-resume-v11`快照记录完整schema/spec；v10及更早schema默认只允许冻结评估，例外迁移必须使用专用CLI。

## 开发约定

- 不直接改变已有 Feature/Reward ID 的字段、shape、顺序、数值或语义。
- 特征提取必须确定且不能修改传入的 `game_state`。
- 合法动作、人工状态摘要和学习策略保持分离。
- 消融必须创建独立版本并从零训练。
- 奖励尺度不同的训练曲线不可直接比较；统一以冻结评估的正式游戏指标选模。
