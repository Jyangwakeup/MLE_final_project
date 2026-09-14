# Migrate R5 Task 1 exploration contract

Type: task
Status: resolved

创建原 Task 1 run 的不可覆盖迁移副本，将错误记录的 1,920,000-step exploration contract
更正为 checkpoint 静态超参数证明的实际 80,000-step 行为。保留原 run，并重新签署两代 resume
manifest，记录原 generation hash、源码身份和迁移证据。

## Comments

- 原 learner 的 `hyperparameters.epsilon_decay_action_steps` 为 80,000，训练 CSV 在 80,000
  action steps 后记录 epsilon 0.05；`exploration_spec` 错误记录为 1,920,000。

## Answer

- 原 run 已改名为 `runs/ddqn_continuous_v2_r5_s11_t1_train_legacy_recorded_1920000`。
- 迁移后的 run 使用标准路径 `runs/ddqn_continuous_v2_r5_s11_t1_train`。
- 最终 checkpoint 与两代 resume generation 的 exploration contract 已更正为 80,000，并
  重新计算文件 hash、manifest hash 和 latest 指针。
- metadata 记录原 run、原 generation hash、原源码 hash、错误值、实际值和证据字段。
- 递归比较确认 learner 状态除 `exploration_spec` 外完全相同；generation 1000 完整性加载通过，
  当前 Task 2 resume contract 预检通过。
