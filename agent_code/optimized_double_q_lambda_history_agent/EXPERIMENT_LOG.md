# Double Q(λ) 历史输入修复实验日志

## 实验目标

旧 `continuous-v2` 运行时只传入 `previous_action`，使索引 77–83 在真实训练和
冻结推理中恒为 0。H-E01 只修复状态对应的 `previous_coin_target` 与
`previous_position` 输入，用冻结评估判断该缺失是否造成 Task 2 往返循环。

旧 r20 200k 基线：平均金币 3.95、平均炸箱 53.45、长 WAIT 局率 30%、长往返
局率 75%。这些数字是比较基线，不代表历史输入缺失已被证明是失败根因。

## 固定合同

| 项目 | H-E01 |
|---|---|
| 算法 | Watkins Double Q(λ) |
| 特征 | `continuous-v2`，84 维，`history_input_version=1` |
| 历史位 | 77 目标连续；78–81 四向返回上一位置；82–83 恒 0 |
| 奖励 | `r20_safe_credit_targeted_wait` |
| Safety | `survival-mask-v1/all` |
| γ / λ | 0.95 / 0.8 |
| 学习率 | 0.08→0.02（200k actions） |
| Tile coding | 8 tilings、8 bins、memory 32768、per-action projection |
| 训练设备 | CPU，单线程 |

## H-E01：seed 11 正式链

状态：completed，未达到延长条件

- 稳定训练源码：`8b9a5b8`。
- 训练前验证：专用及运行器回归 34 项通过；3 局 smoke、checkpoint 冻结重载和
  官方式隔离包运行通过。CPU act p95 14.64 ms、max 19.23 ms。
- Task 1 Slurm job：`473164`；run：
  `runs/qlambda_history_r20_t1_s11_j473164`；提交于 2026-09-18 19:42 CEST，
  预计 50–80 分钟。
- Task 2 Slurm job：`473165`，依赖 `afterok:473164`；预计在 Task 1 完成后再运行
  45–70 分钟。若父链训练或晋级审计失败，该 job 不会训练 Task 2。
- 实际耗时：Task 1 1:01:32；Task 2 0:46:13。Task 1 在400局通过三次连续
  50分评估和独立20-seed晋级审计。
- 最佳为100k快照，SHA-256
  `08cd3deca4e5ee707d6bab67fbc3446cf50261188c450196464d3fa71405a584`；
  平均金币1.65、平均炸箱29.70、长往返率85%，三项延长条件均未通过。
- `training_summary.json` 中 `avoidable_wait_count=0` 来自线性训练CSV未记录该列后
  的默认补零，不能解释为r20惩罚从未触发。

- Task 1：从零训练；200 局起每 50 局在 seeds 9000–9019 冻结检查，连续三次
  `mean_score ≥ 48`；随后在 seeds 10000–10019 做晋级审计。
- Task 2：从通过审计的父链 resume，固定 100k actions、至少 250 局，每 25k
  保存快照并在相同 20 seeds 上评估 Task 1/2。
- 100k 延长条件：平均金币 >3.70、长往返局率 <70%、平均炸箱 ≥50.25，并通过
  安全及 Task 1 保留底线。

| 指标 | 25k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|
| 平均金币 | 0.95 | 0.60 | 1.50 | 1.65 |
| 平均炸箱 | 14.10 | 16.20 | 28.65 | 29.70 |
| 长 WAIT 局率 | 90% | 70% | 75% | 50% |
| 长往返局率 | 100% | 100% | 100% | 85% |
| Task 1 保留率 | 100% | 100% | 100% | 100% |
| 通过门槛数 | 11/19 | 11/19 | 11/19 | 11/19 |

历史输入缺失是已修复的正确性问题，但本轮冻结结果没有证明它是Task 2失败主因。
