# Task 4 共享父模型实验：工程准入失败

**结论：验证失败，Task 4 未通过。** 本轮在首个诊断短跑遇到安全搜索超时，
按预注册停止规则结束；正式 A/B、冻结基线、确认和主验证均未开始。
这不是预算耗尽，也没有证据表明 Task 4 已收敛或学习收益不够。

## 实施与身份

- 工作目录：`/export/data/sfan/MLE_final_project_task4`；分支 `task4-frozen-transfer`。
- 冻结源码：`013fef5b6456606a9dc5a8665b4523c5f63374d9`；实验以干净工作树启动。
- manifest 规范化 SHA-256：`2aac777396b30314d98f5a74ea6039ad3c19de71644b8d9cf06bcee4cde6430b`。
- A 配置 SHA-256：`6a1c9f9d0714afdf0ae4881f2d7a08b9208cd86dbab6e1962ade8e6c1e48118d`。
- B 配置 SHA-256：`8c5deb0ded4371942fc65bdbae577ec512c04c8b6407fb4329e05d925e187617`。
- 实施开始：2026-09-17 16:36:29 UTC；实验停止：2026-09-17 16:58:46 UTC。
- 原定硬截止：2026-09-18 16:36:29 UTC；实际因工程失败提前终止。

实现了显式 Task 4 迁移、严格同实验恢复、共享父模型三种子控制器、
A/B 固定采样合同、分阶段世界隔离、不可变检查点、缓存身份复核、
首个合格点冻结、确认/主验证失败停止、18/23/24 小时预算控制，
以及逐动作审计、失败状态和死亡轨迹保存。安全算法、84 维、网络、r7、n-step 5
及 v5 的 400ms 搜索预算保持不变。术语和 ADR 0010/0011 已同步。

父权重仍为 `agent_code/double_dqn_continuous_v2_agent/task3_validated.pt`，
SHA-256 `b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d`。
其旧来源提交、归属修正后的映射及备份引用见 manifest，旧 metadata 未修改。
Task 1–3 Replay 按计划保留；这不等于已经证明历史 Task 2 数据不受旧生命周期问题影响。

## 准入测试

完整 unittest：**402 项，OK，跳过 1 项，98.127 秒**。
新增 12 项覆盖迁移状态与 RNG、父 teacher、A/B 实际采样、warmup、
实际学习更新的保存恢复，以及真实 runner 连续/分段训练的全部 Agent 动作和模型参数一致性；
另覆盖错误身份拒绝、门槛、B 资格、失败停止与预算。现有安全、快照生命周期、
n-step 和恢复测试包含在完整测试中。

首次完整测试有一项旧测试因新 worktree 尚无 `runs/` 父目录失败；
建立目录后完整重跑通过，未因此修改模型或安全实现。
[完整测试日志](../../experiments/results/task4_shared_parent_20260917/unittest.log)。

## 完成范围

| 阶段 | 实际结果 |
|---|---|
| 逻辑 seed22 独立诊断，RNG 22422 | 第 1 局、第 57 次决策中断；0/20 局完成 |
| 逻辑 seed11/33 独立诊断，RNG 22411/22433 | 未启动 |
| 父模型混合对手 20 世界 | 未启动 |
| Task 1–4 开发基线各60世界 | 未启动 |
| A：seed22/11/33 正式训练 | 全部未启动，无选中检查点 |
| B：seed22/11/33 正式训练 | 全部未启动 |
| 三种子100世界确认 | 未启动 |
| 唯一候选100世界主验证 | 未启动 |

登记世界为诊断22000–22019、开发22100–22159、确认22200–22299、
主验证22300–22399；本轮中均未打开。20000–20099仍封存。
没有完整冻结指标、父子差值或 bootstrap 区间，报告为不适用，不能填零或推断通过。

## 失败现场与原因

正式三名 `rule_based_agent` 对手全部存活。第56步 Agent 在 `(6,15)` 放弹，
第57步炸弹 timer=3、`own_bomb_pending=true`、`own_bomb_visible=true`。
物理候选为 RIGHT/LEFT/WAIT，决策掩码只保留已证明的 RIGHT，实际选择也是 RIGHT。
未发现责任丢失或掩码不一致，鲁棒保证丢失、可避免逃生塌缩和物理回退计数均为零。

**终止门槛：`robust_search_timed_out=true`（允许值为零）。**
完整 act 为 **406.534ms**，低于480ms单次上限；内部搜索耗尽400ms预算。
已观察的57次决策 P95 为 **315.344ms**，也高于250ms，
但这是中断局的部分样本，不冒充完整冻结阶段指标。第51/55步分别约319.36/297.75ms，
说明开销并非只出现在终止动作。

源码与保存状态支持以下诊断：三名对手的物理动作数为6、3、6，
共108种联合动作；四人执行顺序有24种，因此每个本方候选动作在去重前
最多要模拟2592种组合，随后还要进行剩余危险期的可生存性计算。
超时现场计数为104个场景、55,839个状态检查；后者包含 memo 命中累计，
不能解释成55,839个唯一状态。现有枚举先构造完整结果再返回，
也使总成本对对手动作数和执行顺序敏感。

这说明当前 v5 搜索在本次四人状态上无法满足既定预算；
并不证明所有四人状态都会失败，也不是“已经选了必死动作”。
RIGHT 的证明已完成，其他候选检查未全部完成仍足以触发零超时规则。
尚未达到当前任务2,000样本 warmup；保存的初始化文件 policy 与父模型逐项一致，
累计更新仍为356,707。没有证据把问题归因于 Replay 比例或新学习导致的遗忘。

控制器在异常决策审计处中断，整局未正常结束，不能把这局记作胜负、
自杀率或炸弹存活率样本。失败前20个真实状态、全部已写 timing、metadata 与错误诊断均保留。

## 权重与证据

父权重路径和 SHA 见上文。没有 Task 4 合格权重。
诊断目录中的 `checkpoints/final.pt` 是**迁移初始化文件**，不是已训练/合格模型：

```text
/export/data/sfan/MLE_final_project_task4/runs/task4_shared_parent_20260917_A_2aac7773_diag_s22422_c0020_013fef5/checkpoints/final.pt
SHA-256 81294e79fb6a2b1c76c68de500c8ab0e3beb64f303ed823c3fa300564dab5cf0
```

- [机器结果与产物哈希](../../experiments/results/task4_shared_parent_20260917/summary.json)
- [控制器终止状态](../../experiments/results/task4_shared_parent_20260917/campaign_result.json)
- [超时动作诊断](../../experiments/results/task4_shared_parent_20260917/failure.json)
- [失败前20个真实状态](../../experiments/results/task4_shared_parent_20260917/failure_states.json)
- [完整已记录 timing](../../experiments/results/task4_shared_parent_20260917/timing.jsonl)
- [运行 metadata](../../experiments/results/task4_shared_parent_20260917/diagnostic_metadata.json)

## 复现

实际实验入口（源码013fef5、原登记预算内执行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/export/data/sfan/miniforge3/envs/mle/bin/python experiments/task4_campaign.py
```

在独立检出上述源码的 worktree 中，可使用以下命令复现首个诊断；
新 run ID 必须未使用。这是故障复现，不是恢复已停止的本轮正式实验。
计时受机器负载影响，复跑不保证在完全同一步超时；原始现场为本次结论依据。

```bash
taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/export/data/sfan/miniforge3/envs/mle/bin/python experiments/task4_worker.py \
  --config experiments/configs/task4_A.json --mode train --device cpu --task 4 \
  --agent double_dqn_continuous_v2_agent --seed 22422 --n-rounds 20 \
  --transfer-task4-from-checkpoint agent_code/double_dqn_continuous_v2_agent/task3_validated.pt \
  --run-id task4_diag_reproduce_22422 --replay-policy all
```

下一轮应先研究如何在不改变安全语义的前提下降低三对手搜索成本，
验证失败现场与完整安全测试，再重新登记实验。当前证据不足以证明某种优化必然解决；
本轮遵守停止承诺，没有改安全机制、提高预算、启动 B 或放宽门槛。
