# Grouped Tile Double Q(λ) 实验日志

## G-E01：Task 1→Task 2 seed 11

状态：completed（未达标，归档）

- 模型、历史缓存、Double Q bootstrap、Watkins trace、终局清理和checkpoint恢复
  专用测试通过；实验运行器与提交包回归通过。
- 3局smoke、冻结重载、same-task resume和官方式隔离包运行通过。
- CPU act p95 15.00 ms、max 20.51 ms。
- 稳定训练源码：`9799869`。
- Task 1 job `473166` 已完成并通过其晋级审计。
- Task 2 job `473167` 已完成100k筛选；不满足延长条件，未提交200k或seed 22/33复制链。

旧 r20 100k 比较门槛为平均金币 3.70、平均炸箱 50.25、长往返率 70%。本轮只改变
tile 编码：基础组与历史组分别使用4个tilings和16,384容量，Q值为两组贡献之和。
r20日志中过去显示的 `avoidable_wait_count=0` 是未记录字段的默认值，不能解释为
惩罚从未触发。

| 指标 | 25k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|
| 平均金币 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 0.00 |
| 平均炸箱 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 0.00 |
| 长 WAIT 局率 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 5% |
| 长往返局率 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 65% |
| Task 1 保留率 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 100% |
| 通过门槛数 | 见原始快照 JSON | 见原始快照 JSON | 见原始快照 JSON | 5/19 |

最终候选为100k checkpoint `runs/qlambda_grouped_r20_t2_100k_s11_j473167/checkpoints/snapshots/step_0100000.pkl`（SHA-256 `2af1f09ac4f025b97c51613e7e13deea5be5bde6fd09971f607a442184550452`）。Task 2 完全没有炸箱或金币，且跑满400步率为100%，不如 r20 基线，故归档。
