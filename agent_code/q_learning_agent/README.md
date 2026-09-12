# Q-learning agent

这是一个不依赖深度学习框架的表格型 Q-learning agent。状态使用局部障碍、爆炸危险、
最近的金币/木箱/对手方向以及炸弹可用性编码；行动时会屏蔽撞墙等非法动作。

正式训练由 Runner 传入 Agent seed，并与 DQN 共用 `linear-v1` 探索：ε 从 1.0 在
1,920,000 次自身动作内线性降到 0.05，跨 Task 保留累计动作步。直接使用官方框架时
Agent seed 默认为 0。

训练：

```bash
python3 main.py play --agents q_learning_agent --train 1 --scenario classic --n-rounds 10000 --no-gui
```

Task 1 (`coin-heaven`) 可通过环境变量禁用炸弹动作：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 main.py play --agents q_learning_agent --train 1 --scenario coin-heaven --n-rounds 10000 --no-gui
```

`Q_LEARNING_ALLOW_BOMB` 接受 `true/false`、`1/0`、`yes/no` 和 `on/off`；未指定时默认允许炸弹。

直接使用官方入口时，模型会保存为本目录下的 `final.pkl`。测试训练结果时不要传 `--train`：

```bash
python3 main.py play --agents q_learning_agent --scenario classic --n-rounds 10
```

# 训练

## Task 1
Q_LEARNING_ALLOW_BOMB=false python3 main.py play \
  --agents q_learning_agent \
  --train 1 \
  --scenario coin-heaven \
  --n-rounds 10000 \
  --no-gui

## Task 2
Q_LEARNING_ALLOW_BOMB=true python3 main.py play \
  --agents q_learning_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 10000 \
  --no-gui

跨 Task 训练请使用仓库级实验入口的 `--resume-from`。它保留 Q-table、动作步数和
Agent RNG，因此 epsilon 不会在新 Task 重新开始；同时会为下一 Task 重建环境和对手随机流。
不要通过复制 `final.pkl` 模拟精确恢复，因为冻结推理 checkpoint 不包含 Runner 状态。
当前精确恢复协议为 `training-resume-v3`，并校验 Agent seed 与完整探索配置；旧 v1/v2
resume 和旧 final checkpoint 只能冻结评估。

正式六链命令、每阶段预算和门槛见仓库根目录 `IMPLEMENTATION_GUIDE.md`；正式配置固定 CPU，
不使用上面的 10,000 局官方入口示例作为课程链。

## Task 2 测试
python3 main.py play \
  --agents q_learning_agent \
  --scenario classic \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/q_learning_test.json
