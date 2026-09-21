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
| Task 2 成功式早停 | 仅限筛选胜者的 100k→200k 延长；从100k起每25k做配对冻结评估，连续两次通过全部19项门槛才以 `task2_quality_converged` 停止 |

## 已完成实验

| ID | 特征 | 奖励 | 状态 | 目的 |
|---|---|---|---|---|
| V4-T1-R7 | continuous-v4 | r7_safe_credit_potential | completed：`472903` | 父链完成；后续 r7 Task 2 筛选未胜出。 |
| V4-T1-R10 | continuous-v4 | r10_bounded_history_anti_loop | completed：`472904` | 父链完成；r10 筛选后续延长。 |
| V4-T2-R7 | continuous-v4 | r7_safe_credit_potential | completed：`472949` | 100k 筛选完成，未满足完整门槛。 |
| V4-T2-R10 | continuous-v4 | r10_bounded_history_anti_loop | completed：`472950` / `472952` | 100k 筛选与200k延长完成，未达标。 |
| V4-T1-R12 / V4-T2-R12 | continuous-v4 | r12 reward-only | completed：`472954` / `472956` / `472958` | 后备链完成，未达标，归档。 |

## Task 2 启动失败与修复

| Jobs | 阶段 | 结果 | 是否产生 Task 2 训练证据 | 诊断 |
|---|---|---|---|---|
| 472945 / 472946 | r7/r10 100k首次提交 | failed，约2分钟 | 否 | 父模型基线评估后，runner源码哈希与Task1父链不一致 |
| 472947 / 472948 | r7/r10 100k重试 | failed，约2分钟 | 否 | batch先执行约2分钟基线评估，真正resume启动前工作区已恢复新版runner |

四个 job 均未执行 Task 2 learner update，不进入模型横向比较。修复采用显式
`runtime-migration-v1`：普通 resume 继续严格比较源码哈希；只有通过 Task 1
promotion audit、学习合同完全一致且 Git 变更严格限于实验编排/配置/测试/文档时，
才允许 runner-only 迁移。迁移审计随新 run 保存。

100k 胜者必须先满足自杀率 ≤5%、放弹存活率 ≥95%、Task 1 保留率 ≥90%；
再按通过门槛数、`min(mean_coins/6, mean_crates/60)`、循环率和更早 checkpoint 排序。
唯一胜者续训至200k。若仍未通过全部19项门槛，才启用不含硬动作屏蔽的
`r12_coin_priority_anti_loop_reward_only` 后备链。

## 训练控制变更（2026-09-17）

首轮 r7/r10 的 100k action 筛选维持固定预算，因此不会受早停影响，保证消融可比。
后续唯一胜者的 200k extension 启用 `task2_success_stopping`：它不读取训练 reward，也不会因长期无提升而淘汰模型；每次评估均保存不可变 checkpoint、19项 gates 与配对冻结结果。任一门槛失败会把连续通过次数清零。两次连续全通过才提前停止，否则仍跑满200k。全部相关 job 已结束。

## 最终归档结果

`r12` 的 200k 链在 125,200 Task-2 actions 的最佳冻结快照仍未通过完整质量门槛：Task 1 保留率 100%，Task 2 平均金币 0.95/9、平均炸箱 18.40、长 WAIT 0%、长往返 35%、跑满400步率 100%。最佳 checkpoint 为 `runs/qlambda_v4_r12_t2_200k_s11_j472958/checkpoints/snapshots/step_0125200.pkl`，SHA-256 为 `0f8db54eb0524187649efc5278c3cf940f2cfa7c36e7863be7bd2752a08de548`。该变体未成为主线。
