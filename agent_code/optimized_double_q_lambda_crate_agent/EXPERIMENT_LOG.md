# Double Q(λ) 箱子距离量化对照

## CQ-E01：计划中

仅将 continuous-v2 的 crate_frontier_distance_delta 在 tile coding 前映射为 -1/0/+1；金币距离原来已有同类处理。84维输入、每动作29维投影、奖励 r7_safe_credit_potential、safety-all、γ=0.95、λ=0.8、学习率0.08→0.02（累计200k更新）、8 tilings、8 bins、memory32768 全部不变。不引入 DQN 或规则动作覆盖。

旧 v2 Task2 最佳平均金币3.60、炸箱54.2；这是历史参照，不是同批重新训练的对照。v4 的额外特征不带入本实验。

箱子距离归一化幅度约1/288，原量化可能无法区别接近与不变；这是可验证的表示问题，尚不能证明修正后任务表现更好。

## 训练协议

seed11 从零建立独立 Task1 正式父链，连续3次冻结平均分≥48并通过晋级审计后，正常 resume 到 Task2。禁止复用旧量化 checkpoint。Task2 固定100k actions、至少250局，每25k评估 Task1/Task2 seeds10000–10019。使用现有19项门槛，不改门槛，不根据训练奖励宣称成功。

配置见 experiments/configs/optimized_double_q_lambda_crate_task1.json 与 optimized_double_q_lambda_crate_task2_100k.json。运行脚本位于 scripts/run_optimized_double_q_lambda_crate_*_cpu.sh。

| 指标 | 历史 v2 | CQ-E01 |
|---|---:|---:|
| Task2平均金币 | 3.60 | 待评估 |
| 平均炸箱 | 54.2 | 待评估 |
| 长WAIT局率 | 70% | 待评估 |
| 长往返局率 | 60% | 待评估 |
| 自杀率 | 0% | 待评估 |

预计 Task1 60–100分钟，Task2训练及冻结60–90分钟，排队另计。保留旧模型；未通过父链审计不得启动Task2训练。checkpoint哈希和逐快照指标在产物完成后回填。
