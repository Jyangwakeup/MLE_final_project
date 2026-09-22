# 本地分支整合

Status: claimed

用户批准：冻结 B33 seed33/c200 Die Hardest；整合全部本地分支到 main，验证后仅用 branch -d 删除已合并分支，所有实验 worktree 保留原提交（detached）；不推送、不训练、不调参、不改参赛包。

执行顺序：备份校验 → 独立 worktree 从 main 接入 score → backup、opponent-robust、certified、pending、pooled、v9-reference → 独立提交 Die Hardest 和验收依据 → 双轴审查和针对性回归 → main 快进与清理。

共享研发代码保留已演进实现；等价/被替代改动记录出处，不恢复旧算法；每个原分支 tip 必须可从 main 到达。冲突按意图解决，不能确定即停止，不移动 main。历史实验产物不得改写。ADR 重号需重编号及修正非冻结资料引用。

验收：53 文件及 checkpoint/ZIP 哈希保持；180 观察特征/mask/Q/动作等价；隔离原框架随机/规则各一局 CPU 单线程限时60秒；相关单元测试；27155 已知失败保留；工作区和备份完整。原有诊断预算不新增训练，不启动独立确认。

原 main: 0974eafc（完整 SHA 见 initial.json）；code-review 固定点为该提交，具体新增整合差异另与 bbf2ec08 比较以区分继承历史。
