# Q-learning 与 CNN 冻结结果及证据

本表只记录已运行的固定 checkpoint 评估。`训练 seed` 是模型训练链，`评估局数` 是同一组权重经历的环境局数，不能当作独立训练次数。Task 1 是 50 枚可见金币；Task 2 是 9 枚隐藏金币。下表中的 `—` 表示该任务不适用，`未报告` 表示现有来源未给出；两者都不是零。指标定义见[团队字典](../../report-assets/tables/task-metric-dictionary.md)。指向 `runs/` 的原始结果与部分 checkpoint 链接仅在本地工作区有效；该目录不随 Git 提交。公开仓库中的可读证据以已跟踪的实验日志和汇总表为准，原始产物需由团队另行保存或交接。

## 代表结果

| 路线与 checkpoint | 训练 seed；冻结协议 | 局数 | 平均金币 | 全收集率 | 平均炸箱 | 长 WAIT／往返局率 | 自杀／已结算炸弹存活 | CPU act P95／max | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q-learning Task 1，100,008 步 | 11；独立确认 seeds 11000–11099，safety off | 100 | **49.58/50** | **96%** | — | 0%／0% | 0%／— | 6.62／11.63 ms | [原始汇总](../../../runs/qlambda_t1_confirm_s11000_j472894/qlambda_t1_confirm_s11000_j472894_summary/summary.csv)；`verified_raw` |
| CNN Task 1，预训练全局输出头 | teacher 数据；reserved seeds 21000–21099 | 100 | **49.84/50** | **96%** | — | 0%／0% | 0%／— | 8.39／15.55 ms | [原始汇总](../../../runs/cnn_distilled_t1_j471456_reserved/cnn_distilled_t1_j471456_reserved_summary/summary.csv)；`verified_raw` |
| Q-learning Task 2，r20 **200k** | 11；classic，开发 seeds 10000–10019，safety all | 20 | **3.95/9** | 0% | **53.45** | 30%／75% | 0%／100% | 11.32／81.80 ms | [原始选模文件](../../../runs/qlambda_r20_t2_200k_s11_j473159/best_task2_selection.json)；`verified_raw` |
| CNN Task 2，D02 **200k** | 22；classic，选定后主验证 100 seeds，safety all | 100 | **2.60/9** | 0% | **44.15** | 43%／79% | 0%／100% | 8.88／19.11 ms | [原始选择与主验证](../../../runs/cnn_distilled_t2_wait_j472342/selection.json)；`verified_raw` |

前两行各用自己的独立确认环境集合。Task 2 的 Q-learning 行是开发结果，CNN 行是模型固定后的主验证，不能用均值直接给两种算法排名。CPU 数据来自本地冻结评估，并非官方硬件认证。两条路线在上述 Task 1 集合均通过预设的 90% 全收集门槛；上述 Task 2 结果都没有达到 6/9 平均金币和 60 平均炸箱门槛。

## 代表 checkpoint 身份

| 用途 | 文件 | SHA-256 |
|---|---|---|
| Q-learning Task 1 独立确认 | [step_0100008.pkl](../../../runs/qlambda_opt_t1_s11_j472872/checkpoints/snapshots/step_0100008.pkl) | `5172c40830ea74c5ffa4344f75d2d6a5ee11141a7e2a816f26d9f544a5552827` |
| CNN Task 1 保留模型 | [final.pt](../../../agent_code/cnn_distilled_double_dqn_agent/final.pt) | `8c148fee3fcfe8dd19622cdc26d1f0e191d5f7512fb3cbf1d6ad38ee5f77425e` |
| Q-learning Task 2，3.95／53.45 对应快照 | [step_0200000.pkl](../../../runs/qlambda_r20_t2_200k_s11_j473159/checkpoints/snapshots/step_0200000.pkl) | `0d2410976e4b7a632fefc6591919faff36d60589646bc3f57b488275a8c739c1` |
| CNN Task 2 主验证候选 | [best_task2.pt](../../../runs/cnn_distilled_t2_wait_j472342/best_task2.pt) | `912d2f7ae7e53c075e4af238b1e648b69c836beba0467263e06f51947e359569` |

这些哈希已与本地文件重新计算比对。Q-learning Task 1 的表中 P95 为 100 局汇总的 6.62 ms；原实验日志的 7.83 ms 是五个确认 seed 分组中最大的组内 P95，两种聚合不能混写。Q-learning Task 1 的独立确认使用 safety off；进入 Task 2 的正式父链改用 safety all，并从零重建。它们是不同训练合同，不能把 Task 1 确认权重直接写成 Task 2 父模型。

## Task 1 开发对照与蒸馏筛选

| 候选 | 冻结协议／局数 | 平均金币 | 全收集率 | 来源与结论 |
|---|---|---:|---:|---|
| 单表 Q-learning 最佳快照 | Task 1 开发，100 局 | 47.98/50 | 71% | [Q 实验日志](../../../agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)；低于 90% 门槛 |
| Double Q($\lambda$) 100,008 步 | 同轮 Task 1 开发，100 局 | 49.51/50 | 96% | [Q 实验日志](../../../agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)；随后才运行独立确认 |
| 修正输入后的路径 CNN，基础奖励 | Task 1 开发，100 局 | 13.35/50 | 0% | [路径 CNN 日志](../../../experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)；未通过 |
| 路径 CNN，条件循环惩罚 | Task 1 开发，100 局 | 41.78/50 | 17% | [路径 CNN 日志](../../../experiments/agent_variants/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)；未通过 |
| CNN，全局输出头 | 结构筛选 20 局；开发 100 局 | 49.75；49.96/50 | 95%；97% | [CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)；全局头胜出 |
| CNN，动作对齐头 | 结构筛选 20 局 | 48.60/50 | 85% | [CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)；100 局开发和 reserved 均未运行 |
| CNN，动作对齐头加 D4 | 结构筛选 20 局 | 49.70/50 | 90% | [CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)；100 局开发和 reserved 均未运行 |
| 全局头继续 50k 在线更新 | Task 1 开发，100 局 | 37.21/50 | 55% | [CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)；性能退化 |

路径 CNN 早期的 feature-v1 结果受玩家位置通道错误影响。表中 13.35 与 41.78 来自修正输入后的实验，但奖励和训练设置不完全相同，不能作为单因素奖励消融。

## Task 2 开发与失败变体

以下 Q-learning 变体均为 Task 2 冻结开发评估，每行 20 局；训练预算、输入和奖励合同按各自来源读取。列出的均值不代表相同预算的单因素比较。

| 候选与主要变化 | 最佳已评估快照 | 平均金币/9 | 平均炸箱 | 长 WAIT／往返局率 | 证据 |
|---|---:|---:|---:|---:|---|
| 旧 v2 正式 pilot | 175.2k | 3.60 | 54.20 | 70%／60% | [Q 主线日志](../../../agent_code/optimized_double_q_lambda_agent/EXPERIMENT_LOG.md)；`verified_summary` |
| 126 维历史输入，基础奖励 | 100k | 0.55 | 10.40 | 10%／50% | [原始选模](../../../runs/qlambda_v4_r7_t2_100k_s11_j472949/best_task2_selection.json)；`verified_raw` |
| 126 维历史输入，加反循环奖励 | 75.2k／100k 筛选 | 0.60 | 14.35 | 0%／65% | [原始选模](../../../runs/qlambda_v4_r10_t2_100k_s11_j472950/best_task2_selection.json)；`verified_raw` |
| 同配置延长训练 | 200k | 1.35 | 21.20 | 0%／25% | [原始选模](../../../runs/qlambda_v4_r10_t2_200k_s11_j472952/best_task2_selection.json)；`verified_raw` |
| 126 维历史输入，加金币优先反循环奖励 | 125.2k／200k 链 | 0.95 | 18.40 | 0%／35% | [原始选模](../../../runs/qlambda_v4_r12_t2_200k_s11_j472958/best_task2_selection.json)；`verified_raw` |
| 箱子距离三值量化 | 100k | 1.35 | 23.70 | 80%／90% | [箱子量化日志](../../../experiments/agent_variants/optimized_double_q_lambda_crate_agent/EXPERIMENT_LOG.md)；`verified_summary` |
| 修复历史输入 | 100k | 1.65 | 29.70 | 50%／85% | [原始选模](../../../runs/qlambda_history_r20_t2_100k_s11_j473165/best_task2_selection.json)；`verified_raw` |
| 分组 tile 编码 | 100k | 0.00 | 0.00 | 5%／65% | [分组日志](../../../experiments/agent_variants/optimized_double_q_lambda_grouped_agent/EXPERIMENT_LOG.md)；`verified_summary` |
| 条件折返奖励 | 75.2k／100k 筛选 | 3.20 | 46.80 | 50%／70% | [原始选模](../../../runs/qlambda_r21_t2_100k_s11_j473169/best_task2_selection.json)；`verified_raw` |
| 团队示范离线更新加 50k 在线训练 | 50k 在线；另有 40,100 条示范转移、5 遍离线更新 | 3.75 | 56.90 | 65%／75% | [示范日志](../../../experiments/agent_variants/optimized_double_q_lambda_demo_agent/EXPERIMENT_LOG.md)；`verified_summary` |

CNN 的 Task 2 稀疏奖励迁移基线在 seed 11 的 200k 开发快照为 **2.00 金币、34.35 炸箱、55.20% WAIT 动作占比**。同 seed 的条件等待惩罚版本为 **2.80 金币、54.20 炸箱、17.26% WAIT 动作占比**。seed 22 的选中开发快照为 **4.20 金币、67.25 炸箱**；固定该 checkpoint 后，100 局主验证降为首表所列 **2.60 金币、44.15 炸箱**。这些值及三条训练链见[原始选择文件](../../../runs/cnn_distilled_t2_wait_j472342/selection.json)和[CNN 日志](../../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)。计划中的势能奖励对照与五步回报没有运行，状态为 `unrun`，没有可报告成绩。

## 来源冲突与修订事项

1. Q-learning Task 1 [独立确认原始 CSV](../../../runs/qlambda_t1_confirm_s11000_j472894/qlambda_t1_confirm_s11000_j472894_summary/summary.csv)包含 100 个不同评估 run，金币总数 4,958，`AVERAGE` 行为 **49.58/50、96%**。其逐局元数据指向 100,008 步 checkpoint；该文件 SHA-256 与上表一致。原共享表的 **48.95/50** 只回溯到旧汇总，未找到本次确认集合的原始记录支持它。当前 [Task 1–2 表](../../report-assets/tables/table2_task1_task2.csv)、团队中文稿和图数据源均已改为 **49.58/50**。
2. Q-learning r20 [选模 JSON](../../../runs/qlambda_r20_t2_200k_s11_j473159/best_task2_selection.json)把 **3.95 金币、53.45 炸箱**绑定到 `step_0200000.pkl`（哈希 `0d2410…739c1`）。同一文件中的 `step_0175200.pkl`（哈希 `01f12f…47d`）为 **3.45 金币、48.95 炸箱**。共享清单、团队正文和表格已按 200k 身份修正；两快照不能互换。
3. Q-learning 主实验日志的早期 pilot 与后续 r20 属于不同训练链，部分日志状态尚未回填；两者不能按快照名直接合并。CNN 的 D02 主验证也不能代替独立的新训练种子复制实验。
