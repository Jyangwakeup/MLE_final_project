"""Identity and completeness checks before the new Task 4 campaign opens worlds."""
from pathlib import Path
import re
from experiments.task4_transfer import sha256
from experiments.run import _source_hash
import json

def validate(root,manifest):
    root=Path(root);prerequisites=manifest['admission_prerequisites']
    for spec in prerequisites.values():
        if sha256(root/spec['path'])!=spec['sha256']:raise ValueError('Changed viability prerequisite')
    b=json.loads((root/prerequisites['benchmark']['path']).read_text())
    if (b['status']!='historical_passed' or b['failed_states'] or b['search_budget_ms']!=400
            or b['repetitions']!=10 or b['warmups']!=1 or len(b['cpu'])!=1):
        raise ValueError('Incomplete viability benchmark')
    expected=[('task4_shared_parent_20260917',i) for i in range(38,58)]+[('order_equivalence_20260917',i) for i in range(76,96)]
    if [(s['corpus'],s['step']) for s in b['states']]!=expected:raise ValueError('Incomplete failure corpus')
    for state in b['states']:
        if set(state['warmup'])!={'old','new'} or state['warmup']['new']['search_result']['timed_out']:raise ValueError('Failed warmup')
        samples=state['samples']
        if len(samples)!=20 or {(s['implementation'],s['repeat']) for s in samples}!={(v,i) for v in ('old','new') for i in range(10)}:raise ValueError('Incomplete paired samples')
        if any(s['search_result']['timed_out'] for s in samples if s['implementation']=='new'):raise ValueError('Viability timeout persists')
    if b['runtime_source_hash']!=_source_hash('double_dqn_continuous_v2_agent'):raise ValueError('Runtime changed after benchmark')
    for path,sha in b['source_hashes'].items():
        if sha256(root/path)!=sha:raise ValueError('Benchmark helper changed')
    for name,sha in b['corpus_hashes'].items():
        if sha256(root/'experiments/results'/name/'failure_states.json')!=sha:raise ValueError('Failure corpus changed')
    log=(root/prerequisites['unittest']['path']).read_text()
    match=re.search(r'Ran (\d+) tests in [^\n]+\n\nOK(?: \(skipped=\d+\))?\s*$',log)
    if not match or int(match[1])!=prerequisites['unittest']['test_count'] or int(match[1])<415:raise ValueError('Full suite not verified')
