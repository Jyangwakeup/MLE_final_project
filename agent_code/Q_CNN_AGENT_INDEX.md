# Q-learning 与 CNN Agent 索引

主要作者：Ji（提交前确认）

## 当前主运行目录

| Agent | 路线 | 当前最强已验证能力 | 实验记录 |
|---|---|---|---|
| `optimized_double_q_lambda_agent` | Watkins Double Q(lambda)，per-action tile coding，`continuous-v2` | Task 1 开发集与 100 局独立确认均通过（96% 捡满、48.95 平均金币）；Task 2 r20 最佳为 3.95 coins、53.45 crates，未达标。 | [`EXPERIMENT_LOG.md`](optimized_double_q_lambda_agent/EXPERIMENT_LOG.md) |
| `cnn_distilled_double_dqn_agent` | 17通道残差 CNN、Double DQN、动作 mask、团队 teacher 蒸馏 | Task 1 reserved 集 96% 捡满、49.84 平均金币；Task 2 主验证 2.60 coins、44.15 crates，安全/炸箱迁移通过但未捡满全部金币。 | [`EXPERIMENT_LOG.md`](cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md) |

这两个目录是后续打包候选；它们不是团队最终比赛 ZIP 的自动选择。

## 历史变体

早期 Q-learning、Q(lambda) 消融及 CNN path 变体已移动至
[`experiments/agent_variants`](../experiments/agent_variants/MANIFEST.md)。它们仍保留代码、
配置、日志和恢复入口，但不参与默认运行或打包。

其中 `q_learning_agent` 留有仅为历史模块名兼容的轻量入口；其源码仍以
`experiments/agent_variants/q_learning_agent/` 为唯一归档副本，历史 `final.pkl` 留在原相对
路径，未随归档移动。
