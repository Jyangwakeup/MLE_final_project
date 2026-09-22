# 本地分支整合验收

2026-09-22。整合分支已通过验收；切换与清理的最终结果记录在主工作区 `.scratch/branch-integration/REPORT.md` 和 `final-state.json`，不能以本验收文件代替最终状态。

## 冻结候选

保留 B33 seed33/c200 → Die Hardest，53 个文件逐字节不变。原 ZIP SHA-256 为 `0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928`，checkpoint 为 `ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a`。

这表示保留已选定参赛候选，不是全项目统一比较后证明唯一最强。27155 第77步的已知安全缺口未修复，本次通过不构成完整安全认证。

## 备份与历史

仓库外备份 `/export/data/sfan/branch-integration-backups/20260922T145758Z`：280,314 文件、97,365,833,006 字节、7,361 符号链接，无外部未覆盖或已损坏目标。`inventory.json` 保存逐文件哈希；Git bundle 经过独立 bare 恢复、fsck 和全部23个分支 tip 校验。顺序备份为提高吞吐主动中断并续作4线程备份；保留两段日志，完成清单已覆盖19个原worktree。

原 main 为 0974eafc，先接入 score，再正常合并 backup、opponent-robust、certified、pending、pooled、v9-reference。所有原 tip 都可从整合提交到达；历史原分支代码可按 original tip 检出。保留实验 worktree 原提交以维持冻结源码身份，不能把整合代码说成历史实验代码。

## 冲突取舍

- attribution备份文件树等于原main：逐文件保留已演进的score实现，保留原提交历史。
- 旧v4功能与测试已包含在后续实现：保留v11/v5–v9合同，补回失败结果、原始数据和历史回退说明，不重新启用失败方案。
- pending诊断与pooled统计修正已包含：合入独有审计证据，并保留后续图表尺寸测试。
- v9-reference新增协议由显式schema分派，领域词汇并集保留。未启动campaign。
- 8个重号ADR重编号并更新活动引用；原始实验产物、冻结Agent和旧分支不改写。映射见 `adr-renames.json`。

逐文件冲突记录见 `resolution-*.json`；所有旧score结果文件均未改写，新增历史结果文件另计于 `historical-integrity.json`。共享Agent源码相对score无净变更，Die Hardest隔离保存。

## 验证

7组共55项单元测试通过。180条历史观察的特征、掩码、Q值、动作完全一致。隔离原版框架随机/规则对手各1局，均400次决策，轨迹和成绩与原验收一致，无超时、跳过、搜索超时或安全告警。P95分别9.44/16.34ms，最大23.50/39.24ms，最大RSS约314.8MiB。CPU单线程、每子进程60秒；没有训练或调参，没有重跑1000局。参赛回归28.38秒，七组测试合计31.12秒（两组工作并行）。

Standards与Spec独立审查均无新增阻断发现，详见 `review.md`。本次不重新验证Docker、比赛硬件或官方平台，不上传、不推送。备份与完整诊断产物继续保留在本机，不将全部缓存/原始产物加入Git。

执行脚本保留在主工作区 `.scratch/branch-integration/verify_candidate.py`、`run_tests.py`、`backup.py`、`backup_resume.py`；诊断结果和原始日志也位于该目录。
