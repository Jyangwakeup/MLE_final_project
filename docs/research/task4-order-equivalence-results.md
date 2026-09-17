# 执行顺序等价归并：优化与测试结果

**结论：优化未达标。** 等价性验证通过，枚举耗时显著下降，旧故障现场不再超时；
但首个真实诊断在第4局第95步再次触发安全搜索超时，未达到工程准入。
本轮已停止，未继续优化、未运行另外两个种子，也未启动正式Task 4训练。

## 身份与范围

工作树：`/export/data/sfan/MLE_final_project_task4_orders`，分支`task4-order-equivalence`。
源自`ae86fac8d1a89622e6e0758947398f691d381d02`，冻结源码为
`859ee69dc08c86acd53e3418b2d96128377932e1`。
实施开始2026-09-17 17:17:31 UTC，原定截止21:17:31 UTC，
诊断于17:31:37 UTC因工程失败提前停止；文档归档在同一预算内完成。

manifest规范化SHA-256：`aa2c8ecc2b984c9db34ee9914b62ac75f10f00cfce261b766df98fc3f772f0ff`。
配置SHA-256：`846bb0aa1822fd178a21f19c2a45ed935388fe39825e8ac5ce4dfaa77024bb0e`。
测量与诊断固定CPU 0、单线程，机器为AMD EPYC 7452；完整测试使用CPU 1。
这不是课程官方Ryzen机器的兼容认证。

## 实现与正确性

只对固定联合动作的执行顺序做等价归并：移动冲突、移动与放弹冲突、
两次放弹保留先后关系；每种依赖关系方向保留旧遍历中的首个排列，缓存代表排列。
全部对手、动作组合顺序、世界推进、状态去重和公开枚举接口保持不变。
未删除或近似对手，未修改84维输入、奖励、网络、父权重或400ms运行预算。

旧枚举器保存在独立测试参考中。256个固定RNG合成状态覆盖0–3名对手，
并对推进世界前后及全部本方动作比较完整有序场景列表；额外覆盖移动链与循环、
非法本方移动、争抢位置、多方放弹、容量不足、炸弹列表顺序、清箱、火焰和死亡。
20个历史状态的完整安全搜索结果、首个失败证据、场景及状态计数也全部一致；
60秒上限只用于离线参考，未用于运行或性能准入。

最终完整unittest **411项，OK，1项跳过，156.740秒**。
新增检查还覆盖诊断失败停止、终态不重启、四小时截止、残缺基准拒绝和回合结束自杀审计。
冻结前补齐证据样本数量、运行源码/辅助代码/历史数据哈希、最终截止检查；
未解释自杀在该局结束保存证据后即停止后续局。

## 配对性能测量

每个历史状态预热一次，新旧实现交替各测10次；20个状态均已完成。
下表“枚举”覆盖同一组候选动作的全部第一步场景，时间单位ms。

| 历史步 | 动作模拟次数 | 枚举中位耗时 | 400ms预算搜索中位耗时 | 搜索超时次数 |
|---|---:|---:|---:|---:|
| 51 | 2880 → 190 | 73.23 → 6.09 | 268.05 → 199.42 | 0/10 → 0/10 |
| 55 | 2592 → 156 | 66.20 → 5.26 | 249.90 → 190.95 | 0/10 → 0/10 |
| 57 | 7776 → 342 | 310.57 → 17.29 | 491.91 → 294.25 | 10/10 → 0/10 |

第57步模拟次数减少约95.6%，枚举中位耗时约降至原来的1/18。
新实现200次正式历史测量及20次预热均未超时。旧实现第57步的搜索是
**未完成的超时结果**，新实现是完整结果，不能将两者当成等工作量的完整搜索加速比。
旧搜索的预算检查位于枚举返回后，因此其实际耗时可能超过400ms。

第57步新实现完整证明RIGHT/LEFT/WAIT，累计210个场景、113,889个状态检查，
与离线完整旧参考一致；状态检查计数包含memo命中，不是唯一状态数量。
枚举约17ms而搜索约294ms，提示剩余证明计算已经占据主要耗时；
这只是计时差异支持的判断，本轮未追加函数级性能分析或第二项优化。

补齐证据合同之前的一次预备测量也通过，保留于
`runs/order_admission/benchmark_preliminary.json`；本表采用带完整身份记录的最终测量，
没有选择性删掉失败测量。

## 真实诊断与停止原因

| 逻辑种子 | 诊断RNG | 完成范围 | 结果 |
|---|---:|---|---|
| 22 | 22422 | 3/20局完成，第4局第95步中断 | 安全搜索超时 |
| 11 | 22411 | 未启动 | 前一诊断触发停止 |
| 33 | 22433 | 未启动 | 前一诊断触发停止 |

已观察 **940次完整act**，P95为**115.925ms**，最大为
**406.740ms**。这是中断短跑的部分样本，不能冒充完成20局的准入指标；
也不能直接与旧实验仅57次决策的P95做配对比较。

第94步在`(9,13)`放弹，第95步责任仍在、炸弹可见、timer=3。
三个对手均可放弹，分别有6、4、4种物理动作，共96种联合动作。
搜索已完成RIGHT和DOWN的证明，选择DOWN；WAIT证明未完成即触发400ms预算。
现场计数为254个场景、154,022个状态检查（同样含memo累计），
完整act为406.740ms。已记录决策中鲁棒保证丢失和可避免逃生塌缩均为零，
物理回退共4次；本次超时动作没有物理回退。
前三个完整局自杀数为零；但安全搜索超时一次已足以阻止准入。

首次死亡发生在第1局第123步，记录为对手炸弹；完整回放与死亡前状态已保存。
第4局未正常结束，不计入胜负、自杀率或炸弹存活率。
优化后的实际动作可能因原来未完成的证明现在完成而发生变化；
本轮保证的是完整搜索语义等价，不声称所有受超时影响的旧轨迹完全相同。

诊断保存的第3局检查点阶段动作845、累计更新356,707，
policy仍与父模型逐项相同，未达到当前任务2,000样本warmup。
因此本次超时不是新学习更新造成的。

## 权重、证据与复现

父权重仍为`agent_code/double_dqn_continuous_v2_agent/task3_validated.pt`，
SHA-256 `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`，未修改。
诊断检查点仅用于复现，不能作为Task 4合格模型：

```text
/export/data/sfan/MLE_final_project_task4_orders/runs/order_equivalence_20260917_A_aa2c8ecc_diag_s22422_c0020_859ee69/checkpoints/final.pt
SHA-256 8affdb8df88afb987a8758955e198261f6d9c6964a2f8853098fd4da10df4ef8
```

- [汇总与原始产物哈希](../../experiments/results/order_equivalence_20260917/summary.json)
- [全部配对测量](../../experiments/results/order_equivalence_20260917/preflight/benchmark.json)
- [完整测试日志](../../experiments/results/order_equivalence_20260917/preflight/unittest.log)
- [控制器终止结果](../../experiments/results/order_equivalence_20260917/admission_result.json)
- [新超时诊断](../../experiments/results/order_equivalence_20260917/failure.json)
- [失败前20个状态](../../experiments/results/order_equivalence_20260917/failure_states.json)
- [完整已记录timing，gzip](../../experiments/results/order_equivalence_20260917/timing.jsonl.gz)

原始运行目录为`/export/data/sfan/MLE_final_project_task4_orders/runs/order_equivalence_20260917_A_aa2c8ecc_diag_s22422_c0020_859ee69`，
其中保存逐局回放、死亡状态、恢复世代、checkpoint及原始训练CSV。
提交中的CSV只规范了换行符，原始字节文件与哈希保留在运行目录和汇总清单中。
原Task 4失败证据保持不变；开发、确认、主验证与封存世界未使用。
父Replay的历史生命周期局限仍成立，本轮没有重新证明历史训练数据质量。

在独立检出冻结源码859ee69的worktree中，以mle环境执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m unittest discover -s tests

taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python experiments/benchmark_opponent_orders.py --output runs/order_reproduce/benchmark.json \
  --deadline 2026-09-17T21:17:31Z

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python experiments/order_equivalence_admission.py
```

上述截止时间属于原协议；截止后不会继续实验。以后重测性能须显式提供新的
测量截止时间，重开诊断须另登记新身份，不能把已停止本轮改成继续运行。
时间受机器负载影响，复现不保证每次都在完全相同的一步超时。

本轮保留正确且有效的等价归并，但不宣称它已经解决所有四人局面的耗时问题。
后续若继续，应先定位剩余时域生存性计算的开销，再制定独立方案；本轮没有追加此优化。
