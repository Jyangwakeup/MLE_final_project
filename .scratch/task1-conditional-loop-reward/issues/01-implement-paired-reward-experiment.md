# Implement paired conditional-loop reward experiment

Type: task
Status: resolved

实现独立的 `r5_conditional_loop` Reward，接入所有共享训练回调，添加边界测试，并提供只改变
Reward ID 的 Task 1 Double DQN / continuous-v2 配对配置。

## Answer

实现位于 `agent_code/learning_common/temporal_reward.py` 与
`agent_code/team_agent/rewards.py`。`continuous-v2` 增加历史可观测性；训练 CSV 和冻结评估
summary 记录条件循环、WAIT 触发/豁免原因及惩罚贡献。配对配置位于
`experiments/configs/`；相关单元测试和两局端到端训练 smoke 通过。

## Comments

未启动长时间训练；该 ticket 的完成范围是实现、实验冻结和自动测试。
