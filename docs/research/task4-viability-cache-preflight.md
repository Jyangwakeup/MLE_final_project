# 整张生存性图复用：冻结前证据

在8c13a29保留首轮向量化失败后，仅调整同一次安全搜索内部的缓存单位：以地形、危险、全部对手与剩余时域为键，缓存整个生存性图，再按每个本方候选位置分别查询。保留所有对手、场景顺序、证明计数、失败证据与400ms运行预算；奖励、特征、网络和父模型不变。

原失败现场（诊断第15局第131步）的700次场景查询包含140个不同环境。缓存将环境图计算从700次减少至140次；完整证明仍包含700个场景、460405个状态检查计数。60个历史状态和128个固定随机状态均与独立标量参考的完整搜索结果一致，另有缓存命中与不同位置查询反例。开发阶段差分测试曾发现位置变量复用错误，在冻结前修复；失败日志保留于runs/viability_cache。

CPU0、单线程、每个现场预热一次后交替测量10次。以下为三次失败所在状态的完整搜索平均值；旧版为1eaee8d向量化实现，新版额外启用环境图缓存。

| 失败现场 | 旧版均值 ms | 新版均值 ms | 新版最大 ms |
|---|---:|---:|---:|
| 第1局第57步 | 92.74 | 43.47 | 45.71 |
| 第4局第95步 | 114.12 | 48.10 | 50.03 |
| 第15局第131步 | 331.64 | 91.66 | 95.86 |

旧版第三组发生一次400ms超时，因此其均值含截断搜索，不能解释为严格同工作量加速比；完整正确性比较使用单独60000ms离线预算。新版全部60个现场共600次测量及预热均无超时，全体测量最大171.00ms。

父权重为agent_code/double_dqn_continuous_v2_agent/task3_validated.pt，SHA-256 b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d。父Replay保留并不证明历史Task2数据完全不受旧生命周期实现影响。

完整测试记录和上述配对原始数据由新manifest绑定。新实验使用cached-viability-v2；诊断22422、22411、22433属于重复工程回归，旧实验保留终态。原24小时截止仍为2026-09-18 16:36:29 UTC。真实诊断、基线与正式阶段是否通过以新运行result.json为准，本文件不宣告Task4合格。

复现：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 python experiments/benchmark_viability.py --output runs/viability_cache_retry/benchmark.json --deadline 2026-09-18T16:36:29Z
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m experiments.task4_campaign --manifest experiments/task4_viability_cache_campaign.json
```

以上命令要求本地提交干净、父权重可用且未超出预算；不得覆盖已冻结结果或通过修改旧manifest续跑已终止实验。
