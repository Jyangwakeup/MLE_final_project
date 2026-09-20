# Task4 三步得分优化报告

结论：验证失败
源码：35382ac132f85df4a2ec8c86b19cc4e629226b02
父权重 SHA-256：b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d
三步分别为学习率、金币势函数、击杀片段保留与重采样；不承诺单调进化，不代表旧课程 Task4 合格。
旧 C 快照仅作为新协议探索参照，旧固定终点失败结论不变。所有新训练不继承父 Replay。
击杀标签只来自真实回调；死亡后的炸弹得分可能没有对应学习回调，不能声称训练回报等于最终得分。

| 配置 | 种子 | 实际动作 | 得分 | 金币 | 击杀 | 独占/并列第一 | 自杀率 | 炸弹存活 | P95/最大ms | 失败项 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|

测试证据：{'full_suite': {'skipped': 1, 'tests': 520}, 'logs': {'/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_20260920/unit_score_release.log': '0b38e8624ebbdd244009cb0cdcdfa58c84302b1d036b45a72cbbc05ca0c703be', '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_20260920/unittest_full.log': '3169745f5398d020efe2ccd2804087b3c585c1259796af79f45a966b2fc1a58d'}, 'passed': True, 'runtime_source_hash': 'aac3991ba238bb4b34139528736b666c36f8c813df6db4da793be4b452c2e00b', 'targeted_tests': 9}；真实恢复对照：{'actions': 2610, 'checkpoint_hashes': {'/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_LPK4restore/resume/generation-00000008/learner.pt': '59a76c138f9e530fde482f9e8ae1334ffb2bb7271b7196a463e147172a63684a', '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_LPK8/resume/generation-00000008/learner.pt': 'a2235d6b3c6af68b31f1826473f8e8c056b78b7a0551f3835507af4718608c64'}, 'control': '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_C8', 'control_matched_decisions': 8764, 'manifest_sha256': 'b7e17ddc4df5e344cd5ac7643bb5a0512132e461b0fe403d03dd21d95048196e', 'matched_decisions': 8838, 'passed': True, 'prefix': '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_LPK4', 'replay': {'kills': 15, 'ordinary': 2595}, 'restored': '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_LPK4restore', 'source_commit': 'e808358b0bfd90ed38d4304d7287b2ce54b0ad0a', 'updates': 611, 'whole': '/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_preflight_1789857367_LPK8'}
| C | 22 | 20168 | 3.7250 | 2.6250 | 0.2200 | 0.2750/0.1150 | 0.0000 | 1.0000 | 19.05/187.48 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s22_60000/checkpoints/snapshots/step_0020168.pt；SHA-256：e67952dedfcd892d7f001a8c9339ce2e286a25daef5eaadde212cb873dde9da0。
| C | 22 | 40289 | 3.8650 | 2.7150 | 0.2300 | 0.2850/0.1300 | 0.0000 | 1.0000 | 19.91/173.95 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s22_60000/checkpoints/snapshots/step_0040289.pt；SHA-256：45996184c754df2d8780e7b268a76ff3114cec825aa7637af28613c67d2d8bf1。
| C | 22 | 60055 | 4.0950 | 2.8950 | 0.2400 | 0.3350/0.0700 | 0.0000 | 1.0000 | 20.29/158.68 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s22_60000/checkpoints/snapshots/step_0060055.pt；SHA-256：15c16e161b27d8975c736946ad70be3da3387bc6b2df19bca7fcd6c5297980ab。
| C | 11 | 20228 | 3.7900 | 2.4150 | 0.2750 | 0.3450/0.1000 | 0.0000 | 1.0000 | 18.84/168.49 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s11_60000/checkpoints/snapshots/step_0020228.pt；SHA-256：e6bf0a051944388952f55e9a8044b8517baba31d37d205ef53480611cd699ab2。
| C | 11 | 40001 | 3.6400 | 2.6900 | 0.1900 | 0.2900/0.0800 | 0.0000 | 1.0000 | 19.97/164.68 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s11_60000/checkpoints/snapshots/step_0040001.pt；SHA-256：6821562805f51bb666e8a8f29b832b91cd1016363897733ebbabfbaf3762f7d6。
| C | 11 | 60083 | 3.9450 | 2.6700 | 0.2550 | 0.3050/0.1350 | 0.0000 | 1.0000 | 21.72/163.79 | [] |

权重：/export/data/sfan/MLE_final_project_task4_frozen/runs/task4_frozen_20260919_C_s11_60000/checkpoints/snapshots/step_0060083.pt；SHA-256：92bb9ce9459b4d08158519bd869216bfdc02fee441a042b84d68f1d908ba8b8d。
| L | 22 | 20109 | 3.6800 | 2.7550 | 0.1850 | 0.3450/0.1200 | 0.0000 | 1.0000 | 19.48/178.01 | [] |

权重：/export/data/sfan/MLE_final_project_task4_score/runs/task4_score_20260920_L_s22_20000/resume/generation-00000053/learner.pt；SHA-256：fe5761804d2dbcfad9124e4fb8b1a3421efcd2bdf5c5b591b75268a2f4054bf6。

Worker failed; inspect /export/data/sfan/MLE_final_project_task4_score/runs/task4_score_20260920/logs/task4_score_20260920_L_s22_a40309_p1.log

状态文件保存诊断、三种子实际范围、导航统计、Replay占用、最终候选和完整身份；日志目录保存逐批复现命令。
复现：`/export/data/sfan/miniforge3/envs/mle/bin/python experiments/task4_score_campaign.py --manifest /export/data/sfan/MLE_final_project_task4_score/experiments/task4_score_manifest.json`。
