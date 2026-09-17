# CNN Path Double DQN

Task 1 专用的空间 CNN Double DQN 实验 agent。它独立于旧的
`cnn_double_dqn_agent`，不会修改共享特征编码器或其 checkpoint。

## 模型与输入

- `path-spatial-residual-v1`：17→64 卷积、dilation 1/2/4/8 残差块，拼接自身位置、全局平均和全局最大池化，输出六个 Q 值。
- `board-path-history-v2`：12 个客观棋盘通道，加金币/箱子前沿/对手 BFS 距离、上一位置和最近 16 步访问频率。
- Task 1 中 BOMB 被课程 mask 禁用；物理动作 mask 始终保留。
- 当前实验使用 r5 条件抗循环奖励；4-step 版本保存实际 return horizon，并对终局尾部截断。

完整通道定义、奖励权重、超参数和 checkpoint 契约见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。该日志是所有性能数字与选模结论的唯一权威来源。

## Task 1 命令

从仓库根目录运行。正式 4-step 筛选使用独立配置，旧 1-step checkpoint 仍应使用原 r5 配置进行冻结加载。

```bash
CONFIG=experiments/configs/cnn_path_task1_v2_r5_n4.json \
LABEL=v2_r5_n4 TRAIN_SEED=11 TARGET_STEPS=100000 \
sbatch scripts/train_path_task1_experiment.sh
```

训练脚本会先进行三局 smoke、CPU reload，再使用一张学生 GPU 训练，并对每个快照做固定 100 局 CPU 冻结评估。结果路径由 Slurm job ID 决定，例如 `runs/cnn_path_v2_r5_n4_s11_j<job-id>/`。

## 边界

本 agent 只是在 Task 1 上筛选 CNN 导航能力；开发集达到既定门槛、完成独立确认和多 seed 稳定性检查前，不进入 Task 2，也不把单个 checkpoint 描述为最终模型。
