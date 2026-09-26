# Table5 Die Hardest

由同名 CSV 自动生成；修改 CSV 后运行 python3 render_tables.py。

| record_type | candidate | protocol | episodes_or_worlds | mean_score | first_place_rate | exclusive_first_rate | survival_rate | act_p95_ms | act_max_ms | peak_rss_mib | known_limitations | evidence_status | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_benchmark | Task4 B seed33/c200 | 1000 worlds versus three rule_based agents | 1000 | 4.231 | 0.502 | 0.378 | 0.947 | not_reported | not_reported | not_reported | not a rerun after rename; historical safety counterexample remains | verified_summary | all_other_agent_code/die_hardest/README.md |
| package_equivalence | die_hardest | 6 paired games plus trace equivalence | 6 | not_reported | not_reported | not_reported | not_reported | 14.826 | 44.601 | 319.645 | packaging only; Docker and official hardware untested; full safety not approved | packaging_verified | docs/research/die-hardest-submission/verification.json |
| trace_equivalence | die_hardest | renamed package against source candidate | 180 trace rows | not_reported | not_reported | not_reported | not_reported | not_reported | not_reported | not_reported | 27155 safety counterexample reproduces at step 77 | packaging_verified | docs/research/die-hardest-submission/verification.json |
