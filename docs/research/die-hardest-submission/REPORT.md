# Die Hardest 整理验收报告

结论：**重命名、独立打包与运行等价性验收通过。** 原B33权重、算法与安全版本不变。历史27155安全合同问题仍未解决，整理验收不代表完整安全认证。未上传。

## 交付

- Agent目录：`agent_code/die_hardest/`
- 参赛名称：**Die Hardest**；原版框架的加载名与默认显示名：`die_hardest`
- 本目录 `final-project-agent-code.zip` 为提交文件；ZIP顶层仅一个 `die_hardest/`，共53文件，唯一权重 `final.pt`
- 机器结果：`verification.json`；包校验值：`SHA256SUMS`
- 专用整理工具：`experiments/package_die_hardest.py`；原源包SHA锁定，不从当前工作区重新抓取Agent源码

## 身份与内容

来源为已测试B33 seed33/c200 ZIP。所有47个Python文件与源包相比仅作包引用重命名（没有对应引用的文件保持原样），包含延迟导入；其余仅README及提交清单更新。保留完整vendor和原train.py，未加入隔离安全修复。依赖为NumPy2.2.6、Torch2.5.1。公开回调签名和所有算法/特征/奖励/checkpoint合同字段不变。

原checkpoint SHA256：`ae5cf37c7efaa9e7c6d2f41bbe0d1a3c8aa2a58c164ade9cf7cf1ee145686f7a`

新ZIP SHA256：`0f64069f1ebdf7e7097780686514d123b245e6f71c1920e76ab60372303ac928`

原ZIP SHA256：`3fbda6106e810ef49aa2e8fcb8097f66cb8738628faed229177a8de1989e71d5`

新ZIP、当前IDE目录及隔离解压目录逐文件一致；原ZIP、原权重保留。整理脚本重复执行得到相同ZIP，已有相同目标通过身份核对，来源不同则拒绝覆盖。

## 验证结果

| 检查 | 结果 |
|---|---|
| 180条历史观察，包含self名称修改 | 特征、掩码、Q值及动作完全一致 |
| 隔离原版框架 | 两份各45个原版文件未变；仅接入各自版本Agent，无旧名称Agent依赖 |
| 配对对局 | 3局random对手＋3局rule对手；每版本6局，共12局 |
| 完整配对回放 | 世界初态、所有Agent动作、执行排列、局长、成绩与统计均一致（只归一化名称，计时另报） |
| die_hardest决策数 | 2130 |
| die_hardest P95 / 最大 | 14.83 / 44.60ms |
| 原B33 P95 / 最大 | 14.63 / 48.97ms |
| 峰值进程RSS | 319.64MiB |
| CPU与推理模式 | CPU单线程、train=False，模型定位为本Agent目录的final.pt |
| 正常对局 | 无超时、跳过、搜索超时、保证丢失或逃生塌缩告警 |
| 27155责任周期 | 两份均在第77步失败；诊断输出除计时/名称外一致，原失败未更改 |

配对世界和各对手随机种子复用前1000局前六条登记配置，各进程重新初始化且对手随机流隔离。随机对手三局各400次决策；规则对手三局400/130/400次决策，其中一局130步被击杀，双方一致。这些局只验证重命名等价性，不重新估计整体胜率或重新选择候选。

本轮运行检查约124.3秒，在10分钟预算内，每子命令不超过60秒，计入原累计计算预算。新旧模型都禁止梯度入口。没有执行训练。

## 使用

从项目或原版框架根目录运行，使用满足requirements的Python环境：

```bash
python main.py play --agents die_hardest rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --no-gui
```

使用默认推理配置，不加`--train`，不要设置覆盖模型/安全配置的`BOMBERMAN_*`变量。只上传 `final-project-agent-code.zip`，不要把本验收目录整体打包上传。

## 保留的限制

27155为未解决安全合同缺口，1000局未观察到告警不等于修复。新名字不改变这项事实；源B33的1000局均分4.231、含并列第一50.2%是原候选评估，不能说本次重新跑了1000局。

沿用已验证干净Python环境，没有重新安装依赖；未做Docker、官方比赛硬件、MaMPF团队登记核验或上传。目录名不直接更改平台队伍登记名称。

诊断脚本、日志与fixture位于 `.scratch/die-hardest-submission/`，未放入参赛ZIP。原工作区Agent及旧实验结果均保留。
