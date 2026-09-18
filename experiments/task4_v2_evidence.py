"""Evidence reuse is explicit provenance, never a retroactive passing verdict."""
import json
import pickle
import re
from pathlib import Path

from experiments.task4_protocol import VERSION, OBSERVED_VERSION
from experiments.task4_transfer import sha256, PARENT_SAFETY, TARGET_SAFETY


def validate_protocol(m):
    if m['schema_version']==OBSERVED_VERSION:
        from experiments.task4_reference_observation import POLICY
        if m.get('reference_observation_policy')!=POLICY or not m.get('diagnostic_evidence'):
            raise ValueError('Missing observed-reference contract')
    if m['limits'] != {'candidate': {'p95': .1, 'maximum': .3},
                       'reference': {'p95': .25, 'maximum': .48}}:
        raise ValueError('Unregistered timing limits')
    if (m['budget_hours'], m['new_arm_cutoff_hours'], m['evaluation_cutoff_hours']) != (24, 18, 23):
        raise ValueError('Changed budget contract')
    if m['diagnostic_training_seeds'] != {'22': 22422, '11': 22411, '33': 22433}:
        raise ValueError('Changed diagnostic seeds')
    if not m.get('reference_config') or not m.get('reused_evidence'):
        raise ValueError('Missing reference or evidence contract')


def validate_evidence(root, m):
    root = Path(root)
    from experiments.run import _source_hash
    if _source_hash(m['agent']) != m['runtime_source_sha256']:
        raise ValueError('Agent/game semantics changed; evidence cannot be reused')
    for path, expected in m['reused_evidence'].items():
        if sha256(root / path) != expected:
            raise ValueError('Changed reused evidence: ' + path)
    reference = json.loads((root / m['reference_config']).read_text())
    if reference['safety'] != PARENT_SAFETY:
        raise ValueError('Reference must retain original v5 safety')
    configs = [json.loads((root / m['arm_configs'][arm]).read_text()) for arm in ('A', 'B')]
    for arm, config in zip(('A', 'B'), configs):
        if (config['safety'] != TARGET_SAFETY or config['task4_contract']['arm'] != arm
                or config['task4_contract']['safety_migration'] != 'task4-safety-v5-v9-v1'):
            raise ValueError('Candidate migration/config mismatch')
    # The manifest/arm identifiers and replay fraction are the only arm differences.
    a, b = configs
    for c in (a, b):
        c['task4_contract'].pop('arm')
    if a['training']['retention']['parent_fraction'] != .75 or b['training']['retention']['parent_fraction'] != .5:
        raise ValueError('Wrong replay fractions')
    b['training']['retention']['parent_fraction'] = .75
    if a != b:
        raise ValueError('A/B differ beyond the registered replay fraction')
    old = json.loads((root / m['old_result']).read_text())
    if old['status'] != 'admission_failed':
        raise ValueError('Original failed outcome must remain intact')
    for task in range(1, 5):
        audit = json.loads((root / m['old_result']).parent.joinpath(f'fresh_task{task}_audit.json').read_text())
        if audit['failures'] or audit['summary']['episodes'] != 100:
            raise ValueError('Incomplete reused engineering evaluation')
    if any(value < .9 for task in old['retention'].values() for value in task.values()):
        raise ValueError('Reused retention failed')
    regression = json.loads((root / m['regression_result']).read_text())
    if regression['status'] != 'passed' or regression['states'] != 20 or regression['repetitions'] != 10:
        raise ValueError('Incomplete new failure regression')
    if m['schema_version']==OBSERVED_VERSION:
        probe=json.loads((root/m['reference_regression_result']).read_text())
        if probe['status']!='passed' or probe['candidate']['failures']:
            raise ValueError('Observed-reference regression incomplete')
        events=[event for record in probe['reference_observations'] for event in record['events']]
        if events!=['robust_guarantee_loss','avoidable_escape_collapse']:
            raise ValueError('Known reference chain not verified')
    if not re.search(r'Ran \d+ tests in [^\n]+\n\nOK(?: \(skipped=\d+\))?\s*$',
                     (root / m['unittest_log']).read_text()):
        raise ValueError('Full unittest not completed')


def fingerprint(state):
    import hashlib
    return hashlib.sha256(json.dumps(state, sort_keys=True,
        default=lambda value: value.tolist(), separators=(',', ':')).encode()).hexdigest()


def check_failure_prefix(root, m, target):
    result = json.loads((target / 'failure_regression.json').read_text())
    original = pickle.loads((Path(root) / m['failure_corpus']).read_bytes())
    if result['matched_steps'] != [row['state']['step'] for row in original]:
        raise ValueError('Diagnostic did not cover the complete original failure prefix')
    if result['round'] != 13 or not result['matched']:
        raise ValueError('Diagnostic failure trajectory changed')
