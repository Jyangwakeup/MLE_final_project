# 预测试提交包

候选为已通过冻结验证的Task3 seed22/c150，使用等价安全搜索加速实现；并非Task4合格模型。ZIP只含一个Agent目录，默认final.pt为原Task3权重，字节SHA不变。共享运行依赖、requirements.txt和SUBMISSION_MANIFEST.json均已放入该目录。

从原始框架导入提交c4ecff8恢复隔离目录，框架源码未修改；清除全部BOMBERMAN_*环境变量，CPU单线程，self.train=False，针对三个random_agent连续运行3局。1200次act，P95 7.30ms、最大27.99ms，未出现搜索超时。测试通过后未修改ZIP。详细计时和权重加载断言在official-framework/verification.json，运行命令在smoke-command.json，文件校验在submission-verification.json和SHA256SUMS。

Docker不可用，未运行Docker镜像；官方预测试机器的结果仍需上传后获取。此处只证明本地原始框架的加载与运行兼容性，不代表Task4能力通过。

上传文件仅为同目录final-project-agent-code.zip，无需上传测试框架、日志、PDF或报告。

构建工具：experiments/package_agent.py。打包修正会排除源Agent目录中其他.pt/.pkl文件，选中checkpoint复制为唯一final.pt；增加权重SHA清单与实测版本requirements声明。针对双Q、CNN和实际Task3 checkpoint的3项打包测试通过。没有修改策略权重、奖励或安全门槛。

复现构建（输出文件必须不存在）：

```bash
python -m experiments.package_agent --agent double_dqn_continuous_v2_agent --checkpoint agent_code/double_dqn_continuous_v2_agent/task3_validated.pt --output output/task3-pretest-rebuild/final-project-agent-code.zip
```

当前可上传ZIP绝对路径：/export/data/sfan/MLE_final_project/output/submission-pretest-20260917/final-project-agent-code.zip。校验SHA-256：b4f96841e573eed9d083b0f21dea9eb4b9941a0722dc850ac015327afa4cb6a4。证据归档在experiments/results/task3_pretest_package_20260917。ZIP不含本报告、测试日志或其他Agent。
