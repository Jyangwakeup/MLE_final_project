# Q-learning 与 CNN 历史 Agent 归档

这些目录保存已完成或未胜出的实验变体。它们不属于默认的正式运行
`agent_code` 集合；如需复现实验，先运行：

```bash
scripts/restore_archived_q_cnn_agent.sh <agent-name>
```

恢复操作拒绝覆盖已有同名目录。归档不会移动 `runs/` 中的 checkpoint；下表的路径和
SHA-256 是提交前重新定位或复制权重的依据。

`q_learning_agent` 是例外：其源码已归档，但 `agent_code/q_learning_agent/` 保留一个只含
历史 `final.pkl` 与指向归档源码的兼容入口，因为旧基线与队友回归测试仍以该模块名导入它。
如需在隔离位置复现，可对恢复脚本使用 `--destination <temporary-root>`。

仅属于归档变体的历史训练/评估脚本位于
`experiments/agent_variants/scripts/`；恢复 agent 后从该目录执行相应脚本。主运行器和最终
打包器默认不会把归档变体当作候选，只有显式恢复后才允许对可打包变体执行打包。

| 原 agent | 算法/合同 | 最佳已记录结果 | checkpoint / SHA-256 | 归档原因 |
|---|---|---|---|---|
| `q_learning_agent` | 表格 Q-learning，`discrete-q-v2` | 早期 Task 1 基线 | agent-local `final.pkl` | 被 Double Q(lambda) 主线替代；保留兼容链接供历史基线测试。 |
| `optimized_double_q_lambda_v4_agent` | Double Q(lambda)，`continuous-v4` 126维历史 | Task 2 最佳 0.95 coins、18.40 crates；未达标 | `runs/qlambda_v4_r12_t2_200k_s11_j472958/checkpoints/snapshots/step_0125200.pkl` / `0f8db54e…8de548` | 历史/反循环扩展未成为主线。 |
| `optimized_double_q_lambda_crate_agent` | Double Q(lambda)，crate 三值量化 | Task 2 1.35 coins、11/19 gates | `runs/qlambda_crate_r7_t2_100k_s11_j473117/checkpoints/snapshots/step_0100000.pkl` / `084837aa…4d6997` | 不如 r20 基线。 |
| `optimized_double_q_lambda_history_agent` | Double Q(lambda)，修复历史输入 | Task 2 1.65 coins、29.70 crates、11/19 gates | `runs/qlambda_history_r20_t2_100k_s11_j473165/checkpoints/snapshots/step_0100000.pkl` / `08cd3dea…05a584` | 正确性修复未改善瓶颈。 |
| `optimized_double_q_lambda_grouped_agent` | Double Q(lambda)，grouped tile coding | 结果见 `runs/qlambda_grouped_*` | `runs/qlambda_grouped_r20_t2_100k_s11_j473167/checkpoints/snapshots/step_0100000.pkl` / `2af1f09a…550452` | 实验性编码，未作为主线交付。 |
| `optimized_double_q_lambda_demo_agent` | r20 warm-start + team demonstration 1-step Double Q | 3.75 coins、56.90 crates、12/19 gates | `runs/qdemo_demo_s11_j473177/checkpoints/snapshots/step_0050000.pkl` / `b41abc50…f1caad` | 示范未解决循环与金币收集。 |
| `cnn_double_dqn_agent` | 早期 board CNN Double DQN | 历史 Task 1 基线 | 无发布候选 | 由 path/distilled CNN 迭代替代。 |
| `cnn_path_double_dqn_agent` | 17通道 path CNN Double DQN | r5 最佳 Task 1：17% 捡满、41.78 coins | `runs/cnn_path_v2_r5_s11_j471439/checkpoints/best_task1.pt` / `a2fa243b…f4b584` | 被蒸馏 CNN 的 Task 1 结果替代。 |

`optimized_double_q_lambda_agent` 已归档到 `all_other_agent_code/`；`cnn_distilled_double_dqn_agent`
仍保留在 `agent_code/`。详细方法和结果见
[`docs/research/q-cnn-experiment-report.md`](../../docs/research/q-cnn-experiment-report.md)。
