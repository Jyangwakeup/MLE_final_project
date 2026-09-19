# Demonstration Double Q(lambda)

该实验保持 r20 的 `continuous-v2`、联合 tile coding、Watkins Double Q(lambda)、
奖励及 safety-all 合同，只比较冻结团队 teacher 示范经验是否改善 Task 2。
最终推理仅加载本 agent 的 `final.pkl`，不依赖 teacher 或示范数据。
