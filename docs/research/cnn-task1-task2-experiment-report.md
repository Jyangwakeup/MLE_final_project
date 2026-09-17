# CNN Double DQN：Task 1 到 Task 2 实验总结

> 本文是 CNN 子项目的中文实验总结，可作为课程最终 PDF 的材料来源；它不是最终报告 PDF。
> 动态原始记录以
> [`EXPERIMENT_LOG.md`](../../agent_code/cnn_distilled_double_dqn_agent/EXPERIMENT_LOG.md)
> 和对应的 `runs/` JSON/CSV 产物为准。

## 1. 目标、模型与评估口径

本子项目研究一个以空间棋盘为输入的 CNN Double DQN agent。目标不是用规则直接输出
移动方向，而是让网络从棋盘状态学习六个动作（四向移动、WAIT、BOMB）的 Q 值。模型输入
为 `board-path-history-v2`：石墙、箱子、金币、自身、对手、炸弹/爆炸危险、金币/箱子/对手
BFS 距离场、上一位置和最近 16 步访问热图，共 17 个通道。

网络采用无降采样残差 CNN（64 channels，dilation 1/2/4/8）和 Double DQN target。
Task 2 使用 `survival-mask-v1/all`，在行为选择、冻结推理和 TD target 使用同一安全动作集。
Task 1 的核心指标是 100 局捡满率、平均金币和 CPU act 时间；Task 2 额外报告平均金币、
炸箱、炸弹、自杀、零放弹、WAIT/循环及炸弹存活率。所有冻结评估均关闭探索。

## 2. Task 1：从错误输入到蒸馏 CNN

早期 path CNN 的 self channel 曾因金币循环复用了坐标变量而标记错误位置，因此旧
feature-v1 结果不作为奖励或架构的可信比较。修复后使用 17 通道 `feature-v2` 重新训练：
r3 基线在 100k steps 的最佳平均金币为 13.35；r5 conditional-loop 在 75,130 steps 达到
17% 捡满率、41.78 平均金币，但后期发生策略退化。4-step r5 也未改善该问题。

随后使用团队内部 `double_dqn_continuous_v2_agent` 的已验证 Task 1 Q 值进行离线 masked
KL 蒸馏。teacher 数据来自 seeds 6000--6099，student 只保存 CNN 权重；最终推理不加载
teacher 或数据集。下表的 global head 是三种候选中最好的结构。

| Task 1 候选 | 开发 100 局 | reserved 100 局 | 结论 |
|---|---:|---:|---|
| global CNN + teacher pretrained | 97% 捡满，49.96 金币 | 96% 捡满，49.84 金币 | 选定 |
| action-aligned + teacher | 20-seed 85% 捡满 | 未继续 | 未胜出 |
| action-aligned + D4 + teacher | 20-seed 90% 捡满 | 未继续 | 未胜出 |
| pretrained 后 50k TD 微调 | 55% 捡满，37.21 金币 | 未继续 | 策略退化 |

最佳 Task 1 checkpoint 的 CPU act p95/max 为 8.39/15.55 ms，低于 0.5 s 限制。实验说明：
对本环境而言，较低离线 KL loss 不能保证更好的闭环导航；任务特定 TD 微调还会增加 WAIT。

## 3. Task 2：迁移、奖励与 safety 诊断

Task 2 从 Task 1 global pretrained checkpoint 开始，保留 Task 1 teacher KL=2 以避免丢失
导航能力，采用 r7 safe credit sparse、4-step return、BOMB enabled 和 safety all。D01 的
200k 迁移模型保持 Task 1，但冻结 Task 2 仅 34.35 平均炸箱、20% 零放弹、55.2% WAIT，表现为
部分回合进入 WAIT 吸引子。

D02 唯一改变为：当 WAIT 安全、且存在安全推进移动或能炸箱的安全 BOMB 时，给予
`task2_avoidable_wait_penalty=-0.04`。其他网络、特征、KL、探索日程、seed 预算和安全 mask
不变。三条训练 seed 和主验证如下。

| 指标 | D01 200k | D02 seed 11 | D02 seed 22（选定） | D02 seed 33 | D02 主验证 100 局 |
|---|---:|---:|---:|---:|---:|
| Task 1 平均分 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| Task 2 平均金币/分 | 2.00 | 2.80 | 4.20 | 3.65 | 2.60 |
| 平均炸箱 | 34.35 | 54.20 | 67.25 | 57.35 | 44.15 |
| 平均炸弹数 | 18.45 | 22.40 | 27.35 | 23.00 | 18.89 |
| 零放弹率 | 20% | 0% | 0% | 0% | 0% |
| WAIT 率 | 55.20% | 17.26% | 14.90% | 25.69% | 22.56% |
| 自杀率 / 炸弹存活率 | 0% / 100% | 0% / 100% | 0% / 100% | 0% / 100% | 0% / 100% |

100-seed 主验证满足本迁移阶段的安全与炸箱门槛：Task 1 保留、Task 2 平均炸箱 44.15、
零放弹 0%、自杀 0%、非法动作 0%。但它**没有完成 Task 2 全部金币目标**：平均仅 2.60 分，
all-coins rate 为 0%。报告和最终模型选择不应把这一结果描述为通关。

为判断 safety mask 是否阻碍放弹，对 D01 的相同 20 个 seeds 进行了只读反事实诊断。安全
mask 否决 433/5,795 次可物理执行的 BOMB（7.47%），并否决 421/790 次 raw-Q 首选 BOMB
（53.29%），因此运行 paired safety-off 评估。关闭 mask 后平均炸箱从 34.35 降到 3.55，
自杀率升至 90%，WAIT 升至 91.82%。因此保留 safety all：它会改变 Q 排序，但对当前模型
是必要保护，而非 D01 少炸箱的主要原因。

由于 D02 已在三 seed 和主验证通过预注册门槛，potential reward 对照（D03）和 5-step
return（D04）按条件流水线正常跳过，避免在截止前扩大搜索。

## 4. 可复现性、发布物与限制

选定 Task 2 checkpoint 的 SHA-256 为
`912d2f7ae7e53c075e4af238b1e648b69c836beba0467263e06f51947e359569`，本地生成路径为
`runs/cnn_distilled_t2_wait_j472342/best_task2.pt`。该文件约 115 MB，超过 GitHub 普通
文件限制，因此以 Git LFS 作为 agent 内的 `task2_best.pt` 发布；`final.pt`（Task 1，2.6 MB）
也随 agent 发布。训练入口包括
`scripts/run_cnn_distillation_task1_gpu.sh`、
`scripts/run_cnn_distillation_task2_gpu.sh` 和
`scripts/run_cnn_task2_wait_gpu.sh`；详见 agent README 与实验日志。

本项目的主要限制是 Task 2 金币收集仍弱、长 WAIT/往返仍存在，且 Task 3/4 未纳入本 CNN
子项目。若有更多时间，优先应在不降低安全性的前提下改进 BOMB 后的金币信用分配和
Task 2 的任务级选模，而不是直接关闭 safety mask。
