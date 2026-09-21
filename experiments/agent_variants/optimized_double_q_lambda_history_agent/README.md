# History-input Double Q(λ)

该 agent 是 `optimized_double_q_lambda_agent` 的隔离修复实验。算法、84 维
`continuous-v2`、tile coding、Watkins trace cut 与超参数保持不变；唯一行为变化是
把真实的 `previous_coin_target` 和 `previous_position` 输入索引 77–81。

checkpoint 合同包含 `history_input_version: 1`，不能加载旧 agent 权重。实验配置、
结果和复现入口见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。
