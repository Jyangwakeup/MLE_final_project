# 安全搜索优化准入结果

**结论：优化准入通过；尚未宣告Task4合格。** 执行顺序等价归并单独使用时仍失败，后续在保留全部对手和原安全合同的前提下加入等价生存性向量化与同环境整图缓存。本报告针对源码5445dfaae85bc5889f6298ef76882077c0178184，不将先前失败改写为通过。

完整测试419项通过（跳过1项）；场景枚举256个固定合成状态的有序结果等价，完整安全证明60个历史现场及128个随机状态与旧标量参考一致。400ms预算下60现场各10次配对测量，新版600次零超时。最新失败现场平均搜索耗时从331.64ms降至91.66ms，新版最大95.86ms；旧版均值含一次超时截断，不能视为严格同工作量加速比。详见[冻结前报告](task4-viability-cache-preflight.md)。

三个独立20局工程诊断均从登记Task3父模型重新迁移，使用A配置与22422、22411、22433诊断RNG，属于可重复工程回归数据。完整act包含特征、模型和安全搜索。

| 逻辑种子 | 完成局数 | 决策数 | P95 ms | 最大 ms | 自身炸弹死亡 | 全部死亡 | 物理动作掩码回退次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22 | 20 | 7275 | 44.03 | 328.10 | 0 | 2 | 8 |
| 11 | 20 | 6798 | 50.26 | 245.20 | 0 | 4 | 14 |
| 33 | 20 | 7151 | 40.25 | 320.35 | 0 | 3 | 7 |

总计60局、21224次决策。三组决策超时、框架跳过、搜索超时、可避免逃生塌缩和鲁棒保证丢失均为零；逐步审计未发现责任丢失或掩码不一致。9次死亡均由对手炸弹造成；29次物理动作掩码回退保留记录，不能描述为“全程无回退”。原第15局第131步的实测完整act从失败轮407.672ms降至115.500ms，并完整证明五个动作；超时前5061次动作及安全诊断与上一版完全一致。

父权重：agent_code/double_dqn_continuous_v2_agent/task3_validated.pt；SHA-256：b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d。60个回放、9组死亡状态、全部计时和诊断检查点保留在源worktree runs下；精确绝对路径、SHA-256及大小见experiments/results/viability_cache_20260917/admission/summary.json。诊断权重不会接续正式训练，也不是Task4合格权重。父Replay保留不等于证明旧Task2历史数据无生命周期缺陷。

运行源码保留在/export/data/sfan/MLE_final_project_task4_fast，报告通过独立worktree提交，避免改变正在运行实验的源码身份。新manifest为experiments/task4_viability_cache_campaign.json，哈希674c1df171c9cf3a6d40c21fde109063ab2b9ec910b2aac77bf89a3d34e93a8c，冻结诊断世界22900–22919、开发23000–23059、确认23100–23199、主验证23200–23299；20000–20099继续封存。

后续Task4控制器先检查混合对手和Task1–4父基线的工程门槛，然后才允许A训练。原24小时总截止为2026-09-18 16:36:29 UTC；确认或主验证失败仍立即终止。本报告只确认优化准入，后续进度以runs/task4_viability_cache_20260917_5445dfa/result.json为准。

复现命令见冻结前报告。完整实验命令须在源码5445dfa的干净worktree运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m experiments.task4_campaign --manifest experiments/task4_viability_cache_campaign.json
```

已有终止实验不得重启；仅同身份基础设施恢复可使用--resume。没有推送、打包或发布权重。
