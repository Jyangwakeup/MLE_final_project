# Task 1–4 统一实验与评估协议

## 目的与适用范围

本协议统一报告中的术语和呈现规则，不追溯性地改变历史实验合同。课程最终评估仍以官方原始环境与 MaMPF 为准；本仓库证据用于复现项目内部的模型比较和选择说明。

## 任务与冻结评估

| Task | 场景 / 对手 | 主能力 | 主要正式指标 | 说明 |
|---|---|---|---|---|
| 1 | `coin-heaven`，无对手、禁用 BOMB | 可见金币导航 | mean coins/score、all-coins rate、steps、invalid action、latency | 50 枚金币；不报告 win rate。 |
| 2 | `classic`，无对手 | 炸箱、隐藏金币与逃生 | coins/9、crates、bomb survival、suicide、WAIT/loop、latency | 训练 reward 不是验收指标。 |
| 3 | `peaceful_agent`、`coin_collector_agent` | 保留能力并改善竞争得分 | score、coins、crates、kills、first place、survival、latency | 父/子须在同一世界集合配对。 |
| 4 | 三个 `rule_based_agent` | 强对手下的实际得分 | score、first place、kills、survival、safety、latency | 历史专项与最终候选必须分开说明。 |

评估必须冻结 checkpoint、关闭探索和参数更新、以单线程 CPU 运行。每局最多 400 steps；`act` 的课程限制为 500 ms，项目内部工程门槛使用 P95 < 50 ms、max < 500 ms（历史 Task 3 的证书阈值另有说明）。

## 种子、划分与统计单位

| 名称 | 环境 seeds / rounds | 用途 | 可否用于选模 |
|---|---|---|---|
| training seeds | 11、22、33 | 独立初始化和训练链 | 是；每条链独立。 |
| Task 1 收敛监控 | 9000–9019，各 1 局 | Task 1 三次连续通过监控 | 仅用于停止。 |
| development stage gate | 10000–10004，各 20 局 | 100 局冻结开发评估 | 是。 |
| historical CNN/Q confirmation | 实验专用 reserved seeds，通常 100 局 | 固定候选后的独立确认 | 不用于同一轮选模。 |
| Task 3 confirmation | 21000–21099，各 1 局 | 三个 seed 的确认 | 不用于主验证。 |
| Task 3 main validation | 21100–21199，各 1 局 | seed22/c150 的一次主验证 | 否。 |
| planned final test | 20000–20099，各 1 局 | 唯一胜者最终报告 | 本目录没有将其误标为已运行。 |

`training seed` 是独立训练重复；同一 checkpoint 的 environment-seed 局是环境重复，不能写成多次独立训练。三个 training seeds 的结果应先单列，再报告均值和样本标准差。父—子差使用相同世界的 paired bootstrap（Task 3 为 10,000 次）。

## 选模与可比性

1. checkpoint 在开发集按任务预注册排序选取；训练 reward、单局高分或 replay 不得替代冻结指标。
2. 后续课程只可从同 seed 的前一 Task checkpoint warm start；源码/特征/奖励/安全合同改变时须重新从 Task 1 cold start。
3. Q-learning 与 CNN 可作为模型家族比较，但只有场景、数据集合、checkpoint 阶段和指标相同的行才可作数值比较。
4. Task 4 的专项训练、工程世界、历史对手、最终候选 1,000-world benchmark 和六局 rename-equivalence 分属不同协议，不能合并成一条“最终性能”。

## 主比较、效率、消融与局限

- 主比较：Task 1 的 Double Q(lambda) 与 distilled CNN；Task 2 分别报告它们的迁移失败模式；Task 3 使用 parent/child 配对；Task 4 使用候选/尝试清单而非选择后再比较。
- 效率：完整 `act` 的 P95/max、timeout、skipped action；只报告实际运行过的 CPU 数据。
- 消融：Q-learning 的历史、crate quantization、grouped tiles、team demo；CNN 的 r3/r5、蒸馏和 Task 2 TD fine-tuning。负结果同样保留。
- 泛化/稳健性：Task 3 使用确认集与一次主验证；Task 4 未完成官方硬件、Docker 与封存 final-test，不得做外推结论。
- 本项目不提出 XAI 贡献；不设置 explainability 评估。
