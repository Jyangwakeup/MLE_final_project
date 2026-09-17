# Task 3 唯一主验证失败项：计数语义诊断

结论：世界19489的 `avoidable_escape_collapses=1` 是一次已复现的计数误报。
实际死亡确实存在，但发生在对手已封住出口的情况下，不符合项目所定义的
“本次自身炸弹周期中先存在安全替代动作、随后逃生塌缩”。最小计数修正可消除
这项误报，不改变动作或权重，也不会防止本局死亡。

本文件是对[冻结报告](task3-lifecycle-frozen-results.md)的后续诊断。
上一轮 `stopped_main_validation_failure` 保留；没有追溯改判，也没有启动新一轮选模。
被测源码为15d4721（诊断时HEAD为仅增补报告的1e4a5a0）。生产源码未修改；
下述修正仅在独立诊断进程中通过内存替换验证。

## 复现与反证

诊断产物在 `runs/task3_failure_diagnosis_1e4a5a0/`，包括完整状态、决策对象、
第209步之前的Agent快照、脚本、日志及summary.json。对应独立运行目录为
`runs/task3_diagnose_19489_a`、`_b`、`_corrected`及
`runs/task3_diagnose_19517_positive`。这些已揭盲世界仅用于诊断。

| 实验 | 结果 |
|---|---|
| 原权重、原实现完整世界19489，两次独立重放 | 均在210步记录1次塌缩 |
| 从实际209步之前Agent快照运行209→210两次真实act | 稳定复现，核心调用约10ms |
| 仅补计数条件，重跑两步 | 塌缩0次；仍选择BOMB、WAIT |
| 仅补计数条件，重跑完整世界19489 | 塌缩0次；全体Agent的1012条动作记录与最终agent指标完全一致 |
| 阳性对照：开发seed22/c50、世界19517 | 修正后仍在150步记录真实塌缩，151步自杀 |

最小复现（预期退出码1，断言捕捉原误报）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /export/data/sfan/miniforge3/envs/mle/bin/python \
  runs/task3_failure_diagnosis_1e4a5a0/minimal.py
```

诊断修正（预期退出码0）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /export/data/sfan/miniforge3/envs/mle/bin/python \
  runs/task3_failure_diagnosis_1e4a5a0/probe_fix.py
```

## 根因：回退动作集被解释为安全动作集

`agent_code/team_agent/safety.py:146` 的 `survival_mask` 合同是：有可存活动作时
返回安全集合；没有时返回物理合法集合，并设置 `physical_fallback=True`。
`safety_decision` 在这个分支把回退集合也放入 `v1_mask`。

`agent_code/double_dqn_continuous_v2_agent/callbacks.py:343` 计算
`placement_safe_alternative` 时只检查 `v1_mask` 是否含非BOMB动作，
没有检查是否已经发生物理回退。因此物理合法但不能存活的WAIT被误当作安全替代。

| 步 | 真实安全动作 | 物理合法动作 | 行为和计数状态 |
|---|---|---|---|
| 208 | WAIT | WAIT、BOMB | 自身炸弹pending为false，周期标志为false |
| 209 | 空集 | WAIT、BOMB | 选BOMB；错误地将WAIT记为安全替代，周期标志变true |
| 210 | 空集 | WAIT | pending为true，错误周期标志触发塌缩计数 |
| 212 | 空集 | WAIT | 被对手炸弹击杀；非自杀 |

209步之前周期标志明确为false，排除了旧周期状态未清理的解释。
208步的WAIT在放弹周期开始之前，不符合该指标的“同一自身炸弹周期”前提。
210–212步自身炸弹责任持续保留，未复发先前训练生命周期错误。

建议的最小修正是让记录安全替代的条件与实际安全语义一致：

```python
placement_safe_alternative = bool(
    enabled
    and not decision.physical_fallback
    and action == "BOMB"
    and decision.v1_mask[:ACTIONS.index("BOMB")].any()
)
```

这是纠正指标实现，使其符合现有CONTEXT定义，不是把零阈值放宽为1。
该标志及放弹证书的当前用途是诊断和保存，不参与Q值选择；完整对局动作一致
也直接验证了本例中没有改变策略。真实阳性对照仍计数，说明不是简单关闭指标。

## 死亡机制与解决范围

205步模型从(11,3)进入(12,3)的死胡同。该位置上、下为石墙，右侧(13,3)为箱子，
唯一出口在左侧(11,3)。207步尝试LEFT但位置未改变；208步该出口被对手占据。
209步观测到对手已在(11,3)放置倒计时3的炸弹，出口被炸弹阻挡，爆炸覆盖自身位置。
从该状态开始，WAIT或BOMB都不能在爆炸前打开出口。212步由对手炸弹击杀。

当前v5对“自己准备放弹”和“自己已有炸弹责任”执行对手鲁棒检查；没有自身炸弹
责任时的普通移动主要使用v1对当前已知危险的预测，因此不能保证不会走进后来
被对手封锁的区域。这是安全保证的范围限制，与209步的计数误报不同。

仅为解决本次唯一失败项，优先修正计数、加入真实两步回归与阳性对照，无需重训。
若要减少这类真实死亡，需要另行评估无自身炸弹时的对手威胁约束、移动掩码范围
及CPU代价，或改善学到的避困行为；不能只禁止209步放弹，因为此时出口已经封死。
这一策略改动不属于已经停止的原实验臂范围。

正式落地应有新源码/诊断版本与回归验证，并用冻结权重核对各项指标。已使用的
开发、确认、主验证世界可以作为审计和回归数据，不再充当新的盲测。
若要重新给出Task3通过结论，应另行登记独立验证协议和未使用世界，保持原阈值。
