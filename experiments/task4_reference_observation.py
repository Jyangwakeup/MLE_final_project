"""Recognize only the known v5 unproved-placement reference defect."""
import copy
import json
from pathlib import Path

POLICY = 'record_uncertified_placement_consequences_v1'
KNOWN = ('robust_guarantee_loss', 'avoidable_escape_collapse')


class ReferenceObservation:
    def __init__(self):
        self.round = None
        self.unproved_placement = None

    def inspect(self, state, action, safety, reason, original, **kwargs):
        if self.round != state['round'] or not safety.get('own_bomb_pending'):
            self.unproved_placement = None
        self.round = state['round']
        if action == 'BOMB' and safety.get('v1_to_physical_fallback') and not safety['opponent_passing_counts'][5]:
            self.unproved_placement = state['step']
        if (reason not in KNOWN or self.unproved_placement is None
                or safety.get('own_bomb_placed_step') != self.unproved_placement):
            return reason, None
        # Do not hide any second failure behind the first reported flag.
        remaining = original(state, action, {**safety, **{key:False for key in KNOWN}}, **kwargs)
        if remaining:
            return remaining, None
        record = dict(policy=POLICY, round=state['round'], step=state['step'],
                      placed_step=self.unproved_placement, reason=reason,
                      events=[key for key in KNOWN if safety.get(key)],
                      action=action, think_time=kwargs['elapsed'])
        return None, record


def engineering_view(directory, summary):
    """Keep reported counters intact; account separately for recognized cases."""
    records=[]
    for path in Path(directory).glob('*/reference_observations.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    if any(r['policy'] != POLICY or r['reason'] not in KNOWN or not set(r['events']) <= set(KNOWN) for r in records):
        raise ValueError('Unknown reference observation')
    checked=copy.deepcopy(summary)
    for event,counter in [('robust_guarantee_loss','robust_guarantee_losses'),('avoidable_escape_collapse','avoidable_escape_collapses')]:
        count=sum(event in r['events'] for r in records)
        if count>summary[counter]:raise ValueError('Reference observations exceed measured events')
        checked[counter]-=count
    return checked, records


def reused_diagnostics(root, manifest):
    """Reuse completed diagnostic evidence, never the diagnostic learner state."""
    from experiments.task4_transfer import sha256
    spec=manifest['diagnostic_evidence']
    source=Path(root)/spec['result']
    if sha256(source)!=spec['sha256']:
        raise ValueError('Diagnostic evidence changed')
    result=json.loads(source.read_text())
    if result['parent_sha256']!=manifest['parent']['checkpoint_sha256']:
        raise ValueError('Diagnostic parent mismatch')
    for path,expected in spec['metadata_hashes'].items():
        if sha256(path)!=expected:raise ValueError('Diagnostic metadata changed')
    old_config=json.loads((Path(root)/spec['config']).read_text())
    new_config=json.loads((Path(root)/manifest['arm_configs']['A']).read_text())
    for config in (old_config,new_config):
        config.pop('evaluation',None)
        config['task4_contract'].pop('manifest')
    if old_config!=new_config:
        raise ValueError('Learning configuration changed; diagnostics cannot be reused')
    diagnostics=copy.deepcopy(result['diagnostics'])
    if set(diagnostics)!={'11','22','33'}:
        raise ValueError('Incomplete three-seed diagnostics')
    for seed, evidence in diagnostics.items():
        audit=evidence['audit']; raw=audit['raw']
        if (audit['rounds']!=20 or raw['summary']['episodes']!=20 or raw['failures']
                or raw['summary']['act_p95_seconds']>.1 or raw['summary']['act_max_seconds']>.3):
            raise ValueError('Diagnostic gates did not pass')
        metadata=json.loads(Path(evidence['run'],'metadata.json').read_text())
        if metadata['source_hash']!=manifest['runtime_source_sha256']:
            raise ValueError('Diagnostic runtime mismatch')
        if metadata['agent_seed']!=manifest['diagnostic_training_seeds'][seed]:
            raise ValueError('Diagnostic seed mismatch')
        evidence['evidence_reused']=True
        evidence['evidence_source_commit']=result['source_commit']
    return diagnostics
