# Q-learning agent

这是一个不依赖深度学习框架的表格型 Q-learning agent。状态使用局部障碍、爆炸危险、
最近的金币/木箱/对手方向以及炸弹可用性编码；行动时会屏蔽撞墙等非法动作。

训练：

```bash
python main.py play --agents q_learning_agent --train 1 --scenario classic --n-rounds 10000 --no-gui
```

模型会保存为本目录下的 `q-table.pkl`。测试训练结果时不要传 `--train`：

```bash
python main.py play --agents q_learning_agent --scenario classic --n-rounds 10
```
