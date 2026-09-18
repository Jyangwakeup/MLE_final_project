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

状态：planned

- Task 1：从零训练；200 局起每 50 局在 seeds 9000–9019 冻结检查，连续三次
  `mean_score ≥ 48`；随后在 seeds 10000–10019 做晋级审计。
- Task 2：从通过审计的父链 resume，固定 100k actions、至少 250 局，每 25k
  保存快照并在相同 20 seeds 上评估 Task 1/2。
- 100k 延长条件：平均金币 >3.70、长往返局率 <70%、平均炸箱 ≥50.25，并通过
  安全及 Task 1 保留底线。

| 指标 | 25k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|
| 平均金币 | 待运行 | 待运行 | 待运行 | 待运行 |
| 平均炸箱 | 待运行 | 待运行 | 待运行 | 待运行 |
| 长 WAIT 局率 | 待运行 | 待运行 | 待运行 | 待运行 |
| 长往返局率 | 待运行 | 待运行 | 待运行 | 待运行 |
| Task 1 保留率 | 待运行 | 待运行 | 待运行 | 待运行 |
| 通过门槛数 | 待运行 | 待运行 | 待运行 | 待运行 |

完成后在此补充 job ID、实际耗时、源码 commit、父链审计、checkpoint SHA-256、
逐快照冻结结果与是否延长的结论。
