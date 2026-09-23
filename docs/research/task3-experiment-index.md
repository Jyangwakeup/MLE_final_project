# Task3 三条历史实验线索引

本索引将 v3 → v4 → v5 的实验历史集中到 main。三条旧分支的提交在整理前已全部包含于本地及远端 main `c34b02bb464b0c679f750d6ccc15073f33fb2d5b`，各自未合入提交数为 0；不需要重新合并算法，也不重新运行实验。

## 分支与源码身份

| 旧分支 | 删除前完整 tip | 对应实现 |
|---|---|---|
| `task3-escape-obligation` | `e68b9e9793897bdc027d5ae82de178c1882f29c3` | 同一提交引入 v3 自身炸弹逃生责任 |
| `task3-opponent-robust` | `591cd8e0f30e553e1a39eef9e097232a55ce63a7` | 同一提交引入 v4 对手下一步情景枚举 |
| `task3-controllable-survival` | `79d1e87dc538241b7032d4cbd0714b6a35b8517a` | v5 初始实现为 `5e805c97`；此 tip 包含后续保留能力与平台早停逻辑 |

分支 tip 是历史源码定位，不等于最终最佳模型。部分结果报告产生于实现提交之后，按下列结果入口读取，不应只查看旧 tip 的文档。

## 实验目标、实际执行和结论

三条路线围绕 84 维 `continuous-v2` Double DQN 与 `r7_safe_credit_sparse` 展开；各次模型、训练设置与安全版本以对应协议和 checkpoint 为准，不能把三条线混称为同一训练实验。

| 实验线 | 主要测试 | 实际结果与停止位置 |
|---|---|---|
| v3：逃生责任 | 固定旧 seed22/c1500 checkpoint，只改变推理安全掩码。放弹前要求两条独立逃生路线，放弹后持续保留路线。 | 7 个已知自炸世界中，自炸 7/7 → 1/7，未达到 0/7 门槛；停止于首个准入检查。完整 20 世界对比、训练、确认和主验证均未执行。定向反例集不能估计一般收益。 |
| v4：对手下一步鲁棒性 | 枚举对手下一步联合动作，检查这些后继局面中的逃生可能性；冻结比较 Task1/2/3 能力与 CPU 时延。 | 已知 7 世界反例全部通过；三个训练种子的 20 世界 Task3 准入仍失败，自杀率为 5%、5%、20%，P95 超过本轮 50ms 门槛。未执行后续训练、100 世界冻结确认和主验证。只覆盖对手下一步，不能保证整个炸弹周期。 |
| v5：可控生存及平台早停 | 首步精确模拟，后续使用保守的对手可达占据/炸弹危险集合和反向可达性；在 Task3 学习中兼顾 Task1/2 保留，按冻结能力平台选择 checkpoint。 | tip `79d1e87d` 对应流水线状态为 `stopped_confirmation_failure`，未进入主验证、未取得当轮 Task4 资格。seed22 的 Task2 金币保留率为 88.97%；seed22/33 未达到战斗提升要求。随后发现训练回调清除自身炸弹责任的生命周期缺陷，不能据此断言 v5 或模型已达到性能上限。 |

### v3：协议与结果（已随 Git 推送）

- [实验协议与配置入口](../../experiments/task3_escape_obligation.json)、[训练配置（计划但未执行）](../../experiments/configs/task3_escape_train.json)。
- [结果报告](task3-escape-obligation-results.md)、[指标、权重和反例哈希](../../experiments/task3_escape_obligation_results.json)。
- [设计依据](../adr/0003-own-bomb-escape-obligation.md)。

### v4：协议与结果（已随 Git 推送）

- [实验协议与配置入口](../../experiments/task3_opponent_robust.json)、[冻结开发对比配置](../../experiments/configs/task3_opponent_robust_dev_v4.json)。
- [结果与多步失效诊断](task3-opponent-robust-results.md)、[机器结果](../../experiments/task3_opponent_robust_results.json)、[逐局指标 CSV](../../experiments/task3_opponent_robust_evaluations.csv)。

### v5：协议与结果（已随 Git 推送）

- [可控生存协议与配置入口](../../experiments/task3_controllable_survival.json)、[训练配置](../../experiments/configs/task3_controllable_safety_train_v5.json)、[可控生存设计](../adr/0004-use-controllable-survival-for-bomb-safety.md)。
- [保留能力选择协议](../../experiments/task3_retention_prefix.json)、[平台早停协议](../../experiments/task3_plateau_stopping.json)、[确认失败与生命周期缺陷分析](task3-plateau-next-steps.md)。
- 后续独立工作另见[生命周期修复实验](task3-lifecycle-frozen-results.md)及[计数修正后的验证](task3-counter-validation-results.md)。这些后续结果不改变本表三个历史实验的失败或停止结论。

## 本机原始产物与旧路径映射

以下链接用于本机检索。原始日志、回放、权重和归档本体不会因删除分支而消失，也不因本次索引提交而自动上传 GitHub；仅克隆仓库不能获得所有原始产物。历史 manifest 和报告保留原文，读取时按映射定位。

| 资料 | 当前主项目内位置 |
|---|---|
| v3 seed22 原 checkpoint | [runs/task3_pilot_ddqn_cv2_r7_s22_t3_c1500_814173b/checkpoints/final.pt](../../runs/task3_pilot_ddqn_cv2_r7_s22_t3_c1500_814173b/checkpoints/final.pt) |
| v3 已知反例对比 | [v1 运行组](../../runs/task3escape_known_s22_v1_e68b9e9/)、[v3 运行组](../../runs/task3escape_known_s22_v3_e68b9e9/)；12017 逐步反例的精确路径及 SHA 在 v3 结果 JSON 中 |
| v4 多步失效现场 | [seed16013 重放目录](../../runs/task3_opprobust_collapse_repro_s11_e16013_591cd8e/)；其他运行组为主项目 `runs/task3_opprobust_*` |
| v5 原工作区整体资料 | [archive/worktrees/MLE_final_project_task3_v5/](../../archive/worktrees/MLE_final_project_task3_v5/) |
| v5 旧 `runs/task3_plateau_pipeline_79d1e87/result.json` | [归档后的流水线结果](../../archive/worktrees/MLE_final_project_task3_v5/runs/task3_plateau_pipeline_79d1e87/result.json) |
| v5 旧 `runs/task3_plateau_confirmation_79d1e87/result.json` | [归档后的确认结果](../../archive/worktrees/MLE_final_project_task3_v5/runs/task3_plateau_confirmation_79d1e87/result.json) |
| v5 seed22 开发选择与 checkpoint | [选择证据](../../archive/worktrees/MLE_final_project_task3_v5/runs/task3_plateau_evidence_s22_79d1e87/result.json)、[c150 权重](../../archive/worktrees/MLE_final_project_task3_v5/runs/task3_plateau_ddqn_cv2_r7_s22_c0150_79d1e87/checkpoints/final.pt) |

v3/v4 使用的旧 seed22 checkpoint SHA-256 为 `822538947be5badb248876404357aed1dffdb3a2bf6cebc9ff88284a5685a199`；v5 上述开发所选 c150 为 `6c16234f1c74506fb0b8eb35cefad5c9dca5ded8594a6299678aab024f9adabc`，后者不是通过独立确认的获胜模型。整理时还按 v4 协议核验了 seed11/33 模型，按原记录核验 v3 反例回放、v4 CSV，以及 v5 归档清单中的结果文件。

完整旧根目录映射、未提交修改及恢复规则见 [归档说明](../../archive/README.md)和本机 `archive/recovery/registry.json`。本次分支身份、产物核验、独立 bundle 恢复检查与删除结果集中保存在本机 `archive/recovery/task3-branch-retirement/`。

## 删除分支后的源码恢复

旧 tip 仍是 main 的祖先，不依赖同名分支存在。可用完整提交 SHA 查看历史，或导出到临时目录，例如：

```bash
git show e68b9e9793897bdc027d5ae82de178c1882f29c3:agent_code/double_dqn_continuous_v2_agent/callbacks.py
git archive --format=tar --output=/tmp/task3-v3-source.tar e68b9e9793897bdc027d5ae82de178c1882f29c3
```

另外两个 tip 同理。需要原始产物时，根据上表和 registry 配置临时恢复目录；源码历史不会自动包含被忽略的数据。已有 `archive/recovery/remote-merge-20260923/before.bundle` 已通过独立临时 bare clone 检查，能够恢复三条 tip 及对应源码。恢复不意味着获准启动旧训练或已终止 campaign。

本次只新增索引、清理分支引用，不修改算法、旧成绩或原始产物。Die Hardest 的 53 个冻结文件、权重和提交 ZIP 保持不变；27155 安全缺口仍未修复。备份分支与队友的 Rainbow 分支不在此次删除范围。
