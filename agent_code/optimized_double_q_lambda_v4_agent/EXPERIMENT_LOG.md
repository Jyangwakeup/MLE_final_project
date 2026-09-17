# 历史感知 Double Q(lambda) 实验日志

本日志记录 `optimized_double_q_lambda_v4_agent` 的 Task 1 → Task 2 正式训练链。
原 `optimized_double_q_lambda_agent` 及其 checkpoint 保持冻结。

## 固定模型合同

| 项目 | 配置 |
|---|---|
| 算法 | Watkins Double Q(lambda)，per-action tile coding |
| 特征 | `continuous-v4`，126 维；包含 WAIT streak、最近位置重访、循环周期和重复次数 |
| 超参数 | gamma 0.95、lambda 0.8、8 tilings、8 bins、memory 32768、学习率 0.08→0.02/200k actions |
| safety | `survival-mask-v1/all` |
| Task 1 晋级 | seeds 9000–9019 连续三次 mean score ≥48，再通过 seeds 10000–10019 stage gate |
| Task 2 筛选 | seed 11、100k actions、每25k快照、Task 1/2各20局冻结评估 |

## 计划中的实验

| ID | 特征 | 奖励 | 状态 | 目的 |
|---|---|---|---|---|
| V4-T1-R7 | continuous-v4 | r7_safe_credit_potential | planned | 建立正式 Task 1 父链 |
| V4-T1-R10 | continuous-v4 | r10_bounded_history_anti_loop | planned | 建立正式 Task 1 父链 |
| V4-T2-R7 | continuous-v4 | r7_safe_credit_potential | blocked on parent | 测量仅增加历史可观测性的收益 |
| V4-T2-R10 | continuous-v4 | r10_bounded_history_anti_loop | blocked on parent | 测量有界反循环奖励的增量收益 |

100k 胜者必须先满足自杀率 ≤5%、放弹存活率 ≥95%、Task 1 保留率 ≥90%；
再按通过门槛数、`min(mean_coins/6, mean_crates/60)`、循环率和更早 checkpoint 排序。
唯一胜者续训至200k。若仍未通过全部19项门槛，才启用不含硬动作屏蔽的
`r12_coin_priority_anti_loop_reward_only` 后备链。
