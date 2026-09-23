# 远端 main 合并与 Die Hardest 冻结验收

## 来源与范围

本地父提交：`9a5cf77daa13f8df930a7218305b198d6304726b`；远端父提交：`1d182d9e4b9e2facada944cdcda6041452d6d0d2`。合并前分别有 97、28 个独有提交。采用普通 merge，保留双方历史；不重写、不强推。

Git bundle、原始状态、完整测试日志、临时路径适配器和机器结果保存在主项目 `archive/recovery/remote-merge-20260923/`。原始实验归档、提交 ZIP 和未跟踪 .scratch 不纳入此次提交。

## 冲突取舍

- continuous-v2 callbacks/train、DQN model、training_spec、共享 safety 的冲突块保留本地 Task4 演进。远端对应文件与共同祖先 `22eb1c2a` 一致，没有新的功能要求删除这些扩展。保留 observed_kill、sampling_version、迁移合同及安全版本分派。
- Q-learning README/callbacks 接受远端相对符号链接。链接目标源码与本地原文件逐字节一致，权重继续从原 agent 目录定位；其他旧 Q/CNN 变体采用远端归档与显式恢复入口。
- run/training 同时保留 Task4 迁移及远端 runtime migration，合并互斥参数检查，保留 transfer_contract 和 runtime_migration 元数据。
- resume 使用源码 hash、覆盖范围及训练合同核验身份；Git commit 保留来源记录，不因仅文档提交变化而拒绝恢复。
- 自动合并的奖励新增独立 ID、条件奖励开关、Task2 停止条件和归档打包入口均经过相关回归。没有修改 Die Hardest 的 vendored 依赖。

## 验证结果及限制

共覆盖 186 个测试：182 通过，4 按原条件跳过（缺少指定的历史 smoke checkpoint）。覆盖迁移、Replay、恢复、奖励、安全版本、Task2 停止、Q-learning 示范、归档合同和通用打包。另验证四个历史变体可恢复到临时目录，重复恢复拒绝覆盖、文件内容不变；未启动训练 campaign 或历史调度任务。

新增 3 个合并回归：runtime migration 与各 Task4 来源互斥、评估模式拒绝迁移参数、Q-learning 兼容链接保留原模型路径。实际启动训练的 runner 集成测试未执行；部分既有 learner 单元测试只对临时内存样本执行有界更新，不产生候选或修改原权重。

原始回归有 5 个测试因旧工作区已删除而报文件不存在，涉及 score、frozen opponents 和 frozen controller。根据 registry 将读取导向归档；已提交的父权重还核对历史 Git blob 与注册 SHA。原 manifest、测试断言及生产实现均未改动。经临时只读适配后全部通过；原失败日志与适配代码均保留，不能视为原绝对路径已恢复可用。

远端 campaign 单元测试的固定截止日已过，出现 1 项时钟依赖失败。仅在测试中固定时钟到实验期内，补充截止时刻包含性测试；原断言和生产截止规则不变。修正后 4 项测试通过。Standards 与 Spec 审查均无剩余阻断项。

Die Hardest：53 文件冻结哈希全部一致；180 条历史观察的特征、掩码、Q 值和动作相同；原版隔离框架中对随机/规则对手各一局，动作与成绩对齐历史基准，无超时或跳过，CPU 单线程。提交包从归档 B33 来源重建后逐字节一致。

权重 SHA-256：`ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a`。

提交 ZIP SHA-256：`0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928`。

27155 第77步安全缺口继续复现，仍未修复；此次合并不构成安全修复。唯一活动工作区继续是主项目 main。实际推送身份与本地/远端一致性记录见本地恢复目录的 `push-result.json`。
