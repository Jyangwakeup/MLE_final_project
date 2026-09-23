# 实验资料归档

本目录集中保存原工作区的本地资料。`worktrees/` 中的目录是数据归档，不是 Git worktree；唯一活动仓库是主项目 main。

- `worktrees/<原目录名>/`：保持原相对路径的权重、配置、日志、回放、诊断，以及未提交源码。
- `recovery/registry.json`：旧根目录、新资料位置和原提交的映射。
- `recovery/worktree-manifests/`：逐文件 SHA-256、链接去向、未提交修改 patch、缓存排除与环境依赖记录。
- `recovery/main.bundle`：归档开始时的 main 及可达源码历史；迁移工具和说明由后续 main 提交保存。
- `recovery/previous-backup/`：此前备份的清单、原分支 bundle 和执行记录。
- `recovery/backup-destinations.json`：此前备份每个数据文件在本项目内的最终去向。
- `recovery/external-backup-audit.json`：删除前实际备份的完整文件、链接与保存位置核对。
- `recovery/deletion-journal.json`：逐目录删除与中断恢复记录。
- `recovery/validation/result.json`：归档路径下的行为、运行和包重建验收。
- `backup-only/`：与当前主项目/工作区资料不同的备份独有版本，绝不覆盖现版本。

原 manifest、报告、脚本保留原文；其中的旧绝对路径是历史出处，不代表目录仍存在。根据 registry 做路径映射，不对历史文档批量替换。

## 常用资料

- [Task3 v3/v4/v5 实验统一索引](../docs/research/task3-experiment-index.md)：旧分支身份、实际结果、配置与原始产物路径映射。

- 原 B33 包：`worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-submission-check/candidate-not-approved/final-project-agent-code.zip`
- 1,000 局结果：`worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-random1000/`
- 诊断地图与反例：`worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/`
- B33 原 checkpoint：`worktrees/MLE_final_project_task4_reference/runs/task4_reference_20260918_B_badcb498_train_s33_c0200_f3f17d7/checkpoints/final.pt`
- 27155 原现场：`worktrees/MLE_final_project_task4_score/runs/task4_score_20260920_L_s22_a40309_p1/task4_score_20260920_L_s22_a40309_p1_s27155/`

## 当前入口

从主项目根目录打包，使用归档来源，不依赖兄弟工作区：

```bash
python experiments/package_die_hardest.py --source archive/worktrees/MLE_final_project_task4_score/.scratch/task4-diagnosis/b33-submission-check/candidate-not-approved/final-project-agent-code.zip
```

运行参赛 Agent：

```bash
python main.py play --agents die_hardest rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --no-gui
```

迁移验收入口（输出目录必须不存在）：

```bash
python experiments/verify_consolidated_agent.py --output archive/recovery/validation-new
```

此验收只做离线观察检查与两局冻结推理，不训练。已知 27155 第77步反例仍记录为失败；归档成功不代表安全缺口修复。

## 恢复旧实验

1. 在 `recovery/registry.json` 选择原目录，取得 `HEAD` 和 `archive` 字段。
2. 使用 `git archive <HEAD>` 导出源码到临时目录；若恢复仓库不可用，可从 `main.bundle` 或原分支 bundle 克隆到临时目录。无需长期新增 worktree。
3. 先对导出的原源码应用同名 `.patch`，再把该来源归档资料复制到相同相对位置。未跟踪源码和修改后的源码都已在资料中；patch 还记录了文件删除。
4. 从清单的环境记录重建环境。对复制出的临时脚本显式适配 registry 中的路径映射；绝不修改归档里的原 manifest，也不假装适配后的脚本仍有原哈希。
5. 恢复时若需要被链接引用的已提交源码，按记录的原提交导出相应路径。目录内相对链接可解析到其他归档或主项目；在外部临时目录使用时需按记录重新建立链接。
6. 先核验源码、模型和输入身份，再决定运行哪一个只读分析入口。历史训练、一次性注册器和终止 campaign 不因恢复而自动获准执行。

大文件只在本机保存，不批量加入 Git；本目录不代表所有模型已发布到远端。虚拟环境和缓存不保存本体，依赖记录可用于重建。
