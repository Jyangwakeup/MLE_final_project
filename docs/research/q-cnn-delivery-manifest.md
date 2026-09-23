# Q-learning / CNN 交付清单

主要作者：Ji（提交前确认）
公开仓库 URL：`<REPOSITORY_URL_TO_CONFIRM>`

本清单不复制权重。所有 `runs/` 路径仅是本地实验产物；正式比赛 ZIP 选定后必须将对应
权重按 agent 自身相对路径放入该 agent 目录，并重新执行独立打包验证。

| 路线 | 角色 | 当前路径 | SHA-256 | 状态 |
|---|---|---|---|---|
| Q(lambda) | Task 1 已确认主线 | `runs/qlambda_r20_t2_200k_s11_j473159/checkpoints/snapshots/step_0175200.pkl` | `01f12ffc8247ff28488a17a4507828bd40b5313eb267178084c15c40e47a647d` | 位于 `runs/`；本次不复制。 |
| Q demo | Task 2 示范消融最佳 | `runs/qdemo_demo_s11_j473177/checkpoints/snapshots/step_0050000.pkl` | `b41abc507afc169c0cb39ae9a151fb939e4347f4651c485e871f950a69f1caad` | 归档实验；未达标。 |
| CNN distilled | Task 1 final | `agent_code/cnn_distilled_double_dqn_agent/final.pt` | `8c148fee3fcfe8dd19622cdc26d1f0e191d5f7512fb3cbf1d6ad38ee5f77425e` | 已在 agent 目录。 |
| CNN distilled | Task 2 best | `runs/cnn_distilled_t2_wait_j472342/best_task2.pt` | `912d2f7ae7e53c075e4af238b1e648b69c836beba0467263e06f51947e359569` | 本次不复制；最终 ZIP 前需重新选择。 |
| CNN path | 历史 Task 1 最佳 | `runs/cnn_path_v2_r5_s11_j471439/checkpoints/best_task1.pt` | `a2fa243b8ba6369ce779046c8e5570a3e3f5a80ca35f9c5acc9291c087f4b584` | 归档实验。 |

所有 checkpoint 的任务指标、训练 seed 与冻结评估来源见总报告和 agent-local 日志。
