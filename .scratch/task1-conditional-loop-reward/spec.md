# Task 1 conditional loop reward

只比较 `continuous-v2 + Double DQN` 的两个从零训练分支：`r3_potential` 与
`r5_conditional_loop`。训练 seed、初始化、轮数、探索计划及冻结评估 seeds 必须相同。

`r5_conditional_loop` 完整保留 `r3_potential` 的事件奖励和状态势能，只额外包含：

- 条件式循环惩罚 `-0.08`：当前格在动作后死亡判定时刻无危险、存在确定性最近可达金币、
  该目标与上一决策目标相同，且动作后回到上一决策所在位置；
- 可避免 WAIT 惩罚 `-0.04`：选择 WAIT，原地在未来 `t=1` 和 `t=2` 均安全，并存在一个
  物理合法、`t=1` 安全且严格缩短同一目标金币静态 BFS 距离的移动。

目标金币按 `(BFS 距离, x, y)` 确定性选择。无可达金币、目标变化、迫近危险、必要等待和
不缩短路径的移动均不触发额外惩罚。`r4_anti_oscillation` 保留旧命名和行为，新方案不
叠加它的粗粒度惩罚。
