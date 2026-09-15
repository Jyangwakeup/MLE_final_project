# CNN Bomberman Agent：设计与实验导读

本文解释 CNN Double DQN 路线的设计边界和可证伪的优化顺序；它不保存动态实验数字。当前的配置、checkpoint、训练/冻结结果及选模结论统一记录在 [CNN Path 实验日志](../../agent_code/cnn_path_double_dqn_agent/EXPERIMENT_LOG.md)。

## 当前路线

旧 `cnn_double_dqn_agent` 保持冻结，新的 `cnn_path_double_dqn_agent` 用于 Task 1 的独立实验。它使用 17 个空间通道：12 个棋盘/危险通道、三个客观 BFS 距离场、上一位置和 16 步访问热图。距离场描述客观可达性，不给出“正确动作”；六动作决策仍由 Q 网络学习。

网络是无降采样的残差 CNN：17→64 卷积后依次使用 dilation 1/2/4/8，并将自身位置局部向量、全局平均和最大池化拼接后输出六个 Q 值。Double DQN 在合法动作集合中用 online 网络选择、target 网络估值；正式推理使用 CPU，完整 `act` 必须低于课程的 0.5 秒限制。

## 为什么实验先于扩容

空间 CNN 最容易被输入契约错误、回放终局处理、动作 mask 或奖励局部最优误导。路径 CNN 曾发现共享 `board-v1` 的 self 坐标会被金币循环覆盖，因此 v2 在 agent 内重新写入唯一真实 self 通道；此前结果只作为历史证据，不能参与奖励比较。

Task 1 选模使用固定 100 局冻结评估，而非训练 reward 或 loss。排序为：捡满率、平均金币、每百步金币、成功完成步数、循环率。开发集通过后仍需未参与选模的确认集；一个训练 seed 的改善不能代表稳定性。

## 优化顺序

1. 验证特征、回放编解码、终局 transition、checkpoint reload 和 CPU act。
2. 固定网络/特征/seed，对比奖励。r5 只惩罚安全且目标未切换时的立即折返，以及有安全前进动作时的 WAIT。
3. 当冻结曲线表现出延迟信用分配或后期退化时，单独比较 1-step 与 4-step return；每个 transition 保存实际 horizon，并以 `gamma^h` bootstrap，终局尾部不 bootstrap。
4. 只有 n-step 的冻结指标改善后，才依次考虑 D4 等变增强、dueling head、PER 或网络扩容；每轮只变动一项。
5. 领先组合必须在 seeds 11/22/33 上重新训练，再做独立确认，才可迁移到 Task 2。

## 课程与研究边界

CNN 训练可以用 GPU，但提交的 agent 必须在官方 CPU 环境独立运行。不得将基于 BFS 的“最佳动作”作为特征或规则绕过学习，也不得复制外部 Bomberman 实现或权重。外部项目和论文只能提供设计动机：例如 [Double DQN](https://arxiv.org/abs/1509.06461)、[Dueling](https://arxiv.org/abs/1511.06581) 和 [Rainbow 的多步回报](https://arxiv.org/abs/1710.02298)，不保证本项目收益。

更一般的特征、危险预测与其他 agent 的说明见 [Feature 与 Agent 原理导读](feature-agent-principles-guide.md)；旧仓库的调研证据见 [训练流程核查](past-repository-agent-training-flow.md)。
