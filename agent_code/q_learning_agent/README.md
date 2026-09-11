# Q-learning agent

这是一个不依赖深度学习框架的表格型 Q-learning agent。状态使用局部障碍、爆炸危险、
最近的金币/木箱/对手方向以及炸弹可用性编码；行动时会屏蔽撞墙等非法动作。

训练：

```bash
python3 main.py play --agents q_learning_agent --train 1 --scenario classic --n-rounds 10000 --no-gui
```

Task 1 (`coin-heaven`) 可通过环境变量禁用炸弹动作：

```bash
Q_LEARNING_ALLOW_BOMB=false python3 main.py play --agents q_learning_agent --train 1 --scenario coin-heaven --n-rounds 10000 --no-gui
```

`Q_LEARNING_ALLOW_BOMB` 接受 `true/false`、`1/0`、`yes/no` 和 `on/off`；未指定时默认允许炸弹。

模型会保存为本目录下的 `q-table.pkl`。测试训练结果时不要传 `--train`：

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

## Task 2：保留 Task 1 的 Q-table，重新开始 epsilon 衰减
BOMBERMAN_TRAINING_TASK=task2 \
Q_LEARNING_ALLOW_BOMB=true \
python3 main.py play \
  --agents q_learning_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 10000 \
  --no-gui

## Task 2 测试
python3 main.py play \
  --agents q_learning_agent \
  --scenario classic \
  --n-rounds 100 \
  --no-gui \
  --save-stats results/q_learning_test.json