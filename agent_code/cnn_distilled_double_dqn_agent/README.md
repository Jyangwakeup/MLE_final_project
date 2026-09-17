# CNN Distilled Double DQN

Task 1 的独立实验 agent。它保留 `board-path-history-v2` 的 17 通道输入和
1-step Double DQN，以团队 `continuous-v2` 成功模型的冻结 Q 值进行策略蒸馏。
最终推理只加载本目录的 student CNN checkpoint，不加载 teacher、数据集或规则策略。

## 模型

- 无降采样 64-channel residual CNN，dilation 为 1/2/4/8。
- `global` 是旧六输出 head 的蒸馏对照。
- `action_aligned` 用共享 scorer 读取 UP/RIGHT/DOWN/LEFT 的相邻格 embedding，
  WAIT/BOMB 使用独立 head，再作 dueling 聚合。
- `action_aligned_d4` 对八个 D4 视图作 orbit average，保证输出按动作同步等变。
- 物理 legal mask 只屏蔽不可执行动作；Task 1 禁用 BOMB、safety off。

17 个通道依次为石墙、箱子、金币、自身、对手、炸弹存在、炸弹计时、当前爆炸、
未来 t1/t2/t3 危险、t2+ 危险、金币/箱子前沿/对手客观 BFS 距离场、上一位置、
最近 16 步访问频率。输入不含“最佳动作”或金币方向标签。

## 可复现实验

完整流水线遵守学校 Slurm 手册，只申请一张 `students` GPU：

```bash
sbatch scripts/run_cnn_distillation_task1_gpu.sh
```

流水线执行：seeds 6000–6099 teacher 数据采集；6000–6079/6080–6099
训练/验证；三种 head 在 12000–12019 筛选；胜者以 r5、KL=1、lr=5e-5、
epsilon 0.20→0.05 做 50k 1-step DDQN 微调；pretrained/10k/25k/50k
统一在 100 局开发集选模。只有开发集达到 90% 才运行 21000–21099 reserved。

关键入口：

- `experiments/collect_cnn_teacher_data.py`：生成对齐 NPZ；
- `experiments/pretrain_cnn_distilled.py`：单候选离线蒸馏；
- `experiments/run_cnn_distillation_pipeline.py`：完整自动流水线；
- `EXPERIMENT_LOG.md`：配置、哈希、结果和决策的唯一动态记录。

当前结果：global head 的蒸馏 pretrained 在开发集 97/100、reserved 96/100
局捡满，通过本轮 Task 1 数值验收。动作对齐/D4 未胜过 global，DDQN 微调发生退化，
因此最佳模型是纯蒸馏 pretrained，而非微调 final。详见 `EXPERIMENT_LOG.md`。

权重文件：`final.pt` 是默认加载的 Task 1 导航 checkpoint。D02 seed 22、200k steps 的
Task 2 炸箱 checkpoint 因为超过 GitHub 普通文件 100 MB 限制，不随此源码副本发布；其
SHA-256、生成路径和主验证结果记录在 `EXPERIMENT_LOG.md`。该模型在 Task 2 主验证达到
平均 44.15 炸箱、0% 自杀，但不应被描述为全部金币通关模型。

## Task 2 迁移协议

Task 2 从冻结的 Task 1 `global` checkpoint 建立独立子链，使用
`r7_safe_credit_sparse`、`survival-mask-v1/all`、4-step Double DQN 和 200k
阶段动作。原 Task 1 teacher 数据只通过 KL 权重 2.0 保留导航能力，不包含 Task 2
状态或 BOMB 监督。随机初始化对照使用相同网络、seed、训练预算和评估集，KL 权重为 0。

正式入口为 `scripts/run_cnn_distillation_task2_gpu.sh`。每 25k 保存快照，每 50k
同时回测 Task 1/2；seed 11 的迁移链通过联合门槛后才训练 seeds 22/33。

D01 暴露出冻结策略的 WAIT 吸引子后，后续采用单变量条件链：D02 加入窄条件
avoidable-WAIT `-0.04`；失败后 D03 仅将 sparse reward 换为 potential reward；两者
均失败后 D04 才把 n-step 从 4 改为 5。Safety mask 先以同 checkpoint 的只读 Q/mask
日志诊断，不直接关闭。运行任务、checkpoint 和表格结果见 `EXPERIMENT_LOG.md`，机制与
门槛见
[`docs/research/cnn-task2-reward-safety.md`](../../docs/research/cnn-task2-reward-safety.md)。
