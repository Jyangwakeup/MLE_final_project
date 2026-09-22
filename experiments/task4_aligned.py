"""v4: certify the unchanged Task3 policy under v9, then train only arm B."""
import json
import re
import time
from pathlib import Path
from experiments.task4_protocol import ALIGNED_VERSION, CANDIDATE
from experiments.task4_transfer import PARENT_SHA256, PARENT_SAFETY, TARGET_SAFETY, sha256


def validate_protocol(m):
    if m['schema_version'] != ALIGNED_VERSION or set(m['arm_configs']) != {'B'}:
        raise ValueError('Aligned protocol requires only B')
    if m['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Wrong Task3 parent')
    if m['limits'] != {r:CANDIDATE.__dict__ for r in ('candidate','reference')}:
        raise ValueError('Both roles require 100/300ms')
    if (m['budget_hours'],m['new_arm_cutoff_hours'],m['evaluation_cutoff_hours']) != (24,18,23):
        raise ValueError('Wrong budget')
    if m.get('reference_observation_policy') or m.get('diagnostic_evidence'):
        raise ValueError('No reference exception or reused training diagnostics')
    groups=[m[k+'_seeds'] for k in ('engineering','development','confirmation','main_validation')]
    if list(map(len,groups)) != [100,60,100,100] or len(set(sum(groups,[]))) != 360:
        raise ValueError('Need disjoint 100/60/100/100 worlds')
    forbidden=set(m['excluded_seeds'])|set(range(20000,20100))|set(m['training_seeds'])|set(m['diagnostic_training_seeds'].values())
    forbidden.update(v[k] for v in m['rng_plan'].values() for k in ('environment_seed','official_opponent_seed'))
    if set(sum(groups,[])) & forbidden:
        raise ValueError('Used or reserved world range')
    if not 1<=len(m['cpus'])<=6 or len(set(m['cpus']))!=len(m['cpus']):
        raise ValueError('Need 1..6 exclusive physical cores')
    return m


def validate_evidence(root,m):
    from experiments.run import _source_hash
    root=Path(root)
    if _source_hash(m['agent']) != m['runtime_source_sha256']:
        raise ValueError('Protected runtime changed')
    for file,expected in m['frozen_files'].items():
        if sha256(root/file)!=expected:raise ValueError('Changed frozen evidence: '+file)
    b=json.loads((root/m['arm_configs']['B']).read_text())
    reference=json.loads((root/m['reference_config']).read_text())
    if b['safety']!=TARGET_SAFETY or reference['safety']!=TARGET_SAFETY:
        raise ValueError('Both active roles must use v9')
    if reference.get('task4_contract'):
        raise ValueError('Reference config cannot train or migrate')
    if b['training']['retention']['parent_fraction']!=.5 or b['task4_contract']['arm']!='B' or b['task4_contract']['safety_migration']!='task4-safety-v5-v9-v1':
        raise ValueError('Wrong B migration or replay mixture')
    if json.loads((root/m['historical_reference_config']).read_text())['safety']!=PARENT_SAFETY:
        raise ValueError('Historical retention reference changed')
    if not re.search(r'Ran \d+ tests in [^\n]+\n\nOK(?: \(skipped=\d+\))?\s*$',(root/m['unittest_log']).read_text()):
        raise ValueError('Full tests not completed')
    if not json.loads((root/m['regression_result']).read_text())['passed']:
        raise ValueError('World25106 regression incomplete')


def set_stage(c,stage):
    c.state['status']=stage;c.write(c.path,c.state)


def await_previous(c):
    path=Path(c.m['previous_assessment_status'])
    while True:
        c.check_time();value=json.loads(path.read_text())
        if value['status']!='running':
            c.state['previous_assessment']={'path':str(path),'status':value['status'],'scope':'historical only'}
            c.write(c.path,c.state);return
        time.sleep(5)


def certify_retention(c):
    from experiments.task4_campaign import EngineeringFailure
    from experiments.task3_retention_prefix import _evaluation_directories
    from experiments.compare_evaluations import compare_evaluations
    audit=json.loads((c.root/c.m['retention_evidence']).read_text())
    # This artifact was independently verified before freezing; missing or changed
    # historical files trigger a fresh run instead of silently weakening identity.
    reusable=audit.get('verified') and all(Path(f).exists() and sha256(f)==h for f,h in audit.get('files',{}).items())
    if reusable:
        c.state['retention']={'mode':'reused_engineering_evidence','checks':audit['checks'],'evidence':c.m['retention_evidence']}
        c.write(c.path,c.state);return
    results={};intervals={};checks={};worlds=list(range(24000,24060))
    for task,metrics in ((1,('mean_score',)),(2,('mean_coins','mean_crates')),(3,('mean_score','mean_coins','mean_crates'))):
        for role in ('historical_reference','reference'):
            results[role]=c.evaluate('B',c.root/c.m['parent']['checkpoint'],f'retention_{task}_{role}','retention',worlds,tasks=(task,),role=role)
        parent=results['historical_reference']['summaries'][f'task{task}'];child=results['reference']['summaries'][f'task{task}']
        checks[str(task)]={k:child[k]/parent[k] if parent[k] else 1. for k in metrics}
        intervals[str(task)]=compare_evaluations(_evaluation_directories(c.root/'runs',results['reference']['runs'][str(task)],worlds),_evaluation_directories(c.root/'runs',results['historical_reference']['runs'][str(task)],worlds),c.directory/'comparisons'/f'retention_{task}',bootstrap_samples=10000)
        c.state['retention']={'mode':'rerun','checks':checks,'bootstrap':intervals};c.write(c.path,c.state)
        if any(x<.9 for x in checks[str(task)].values()):raise EngineeringFailure('Baseline retention failed')


def run_aligned(c):
    set_stage(c,'waiting_previous_assessment');await_previous(c)
    set_stage(c,'certification_retention');certify_retention(c)
    set_stage(c,'certification_safety')
    c.state['engineering_baseline']=c.evaluate('B',c.root/c.m['parent']['checkpoint'],'engineering_parent','engineering',c.m['engineering_seeds'],role='reference')
    set_stage(c,'certification_diagnostics')
    for seed in (22,11,33):
        if str(seed) not in c.state['diagnostics']:
            c.state['diagnostics'][str(seed)]=c.train('B',c.m['diagnostic_training_seeds'][str(seed)],20,diagnostic=True)
            c.write(c.path,c.state)
    c.state['baseline_certified']=True;set_stage(c,'development_baseline')
    parent=c.evaluate('B',c.root/c.m['parent']['checkpoint'],'development_parent','development',c.m['development_seeds'],role='reference')
    c.state['parent_baseline']=parent
    if 'B' not in c.state['arms']:
        c.check_time(new_arm=True);c.state['arms']['B']={}
    set_stage(c,'development_B');results=c.state['arms']['B']
    for seed in (22,11,33):
        results[str(seed)]=c.seed('B',seed,parent);c.write(c.path,c.state)
        if results[str(seed)]['selected'] is None:
            set_stage(c,'stopped_development_failure');return
    set_stage(c,'confirmation');c.validation('B',results)
