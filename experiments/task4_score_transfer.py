"""Explicit score experiment transfer; archived controls are evaluation-only."""
import json
from pathlib import Path
import subprocess
from experiments.task4_transfer import PARENT_SHA256, TARGET_SAFETY, digest, sha256
from experiments.task4_exploration_transfer import initialize as base_initialize, RETENTION, EXPLORATION
from agent_code.learning_common.effective_learning import resolve
from agent_code.learning_common.kill_replay import CONTRACT as KILL_CONTRACT

VERSION = 'task4-score-improvement-v1'

def retention(arm):
    return dict(RETENTION, sampling_version='task4-kill-replay-v1' if arm=='LPK' else 'task4-only-v1')

def initialize(payload, config, seed, training_budget, contract):
    return base_initialize(payload, config, seed, training_budget, contract,
                           retention_spec=retention(contract['arm']))

def contract_for(root, config_path, seed):
    root, config_path = Path(root), Path(config_path)
    config=json.loads(config_path.read_text()); spec=config['score_contract']
    m=json.loads((root/spec['manifest']).read_text()); arm=spec['arm']
    if (m['schema_version']!=VERSION or spec['version']!=VERSION or spec['role']!='learner'
            or arm not in ('C','L','LP','LPK') or (root/m['arm_configs'][arm]).resolve()!=config_path.resolve()
            or seed not in m['training_seeds']+list(m['diagnostic_training_seeds'].values())):
        raise ValueError('Score protocol/role/config/seed mismatch')
    from agent_code.double_dqn_continuous_v2_agent.callbacks import HYPERPARAMETERS
    effective=resolve(HYPERPARAMETERS,config)
    if (config['safety']!=TARGET_SAFETY or config['feature_id']!='continuous-v2'
            or config['reward_id']!=('r7_safe_credit_potential' if arm in ('LP','LPK') else 'r7_safe_credit_sparse')
            or config['training']['n_step']!=5 or config['training']['retention']!=retention(arm)
            or config['training']['exploration']!=EXPLORATION or config['training']['safety_replay']['enabled']
            or config['frozen_opponents']['arm']!='C'):
        raise ValueError('Score learning/safety/opponent contract mismatch')
    if m['parent']['checkpoint_sha256']!=PARENT_SHA256:raise ValueError('Score parent mismatch')
    for path,h in m['opponent_hashes'].items():
        if sha256(root/path)!=h:raise ValueError('Opponent code changed')
    from experiments.run import _source_hash
    return dict(version=VERSION,role='learner',arm=arm,seed=seed,parent_sha256=PARENT_SHA256,
        target_safety=TARGET_SAFETY,hyperparameters=effective,reward_id=config['reward_id'],
        retention=retention(arm),kill_replay=KILL_CONTRACT if arm=='LPK' else None,
        config_sha256=sha256(config_path),manifest_sha256=digest(m),
        opponent_hashes=m['opponent_hashes'],schedule_seed=m['schedule_seed'],opponent_rng_root=m['opponent_rng_root'],
        source_hash=_source_hash('double_dqn_continuous_v2_agent'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())

def materialize(parent,destination,*,root,config_path,seed,training_budget):
    import torch
    if sha256(parent)!=PARENT_SHA256:raise ValueError('Requires registered original Task3 parent')
    config=json.loads(Path(config_path).read_text());contract=contract_for(root,config_path,seed)
    payload=torch.load(parent,map_location='cpu',weights_only=True)
    child=initialize(payload,config,seed,training_budget,contract)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');torch.save(child,temporary);temporary.replace(destination)
    return contract

def validate_resume(contract,*,root,config_path,seed):
    if contract!=contract_for(root,config_path,seed):raise ValueError('Score strict resume identity mismatch')

def validate_archive(payload,path,manifest):
    entries=[e for group in manifest['archived_controls'].values() for e in group]
    entry=next((e for e in entries if Path(e['checkpoint']).resolve()==Path(path).resolve()),None)
    if entry is None or sha256(path)!=entry['checkpoint_sha256']:raise ValueError('Unregistered archived control')
    contract=payload.get('transfer_contract',{})
    if contract!=entry['transfer_contract'] or contract.get('arm')!='C' or contract.get('version')!='task4-frozen-opponents-v1':
        raise ValueError('Archived control contract mismatch')
    if (payload['stage_action_steps']!=entry['actual_actions'] or payload['agent_seed']!=entry['seed']
            or payload['hyperparameters']!=entry['hyperparameters'] or payload['retention_spec']!=RETENTION
            or payload['reward_id']!='r7_safe_credit_sparse' or payload['safety_spec']!=TARGET_SAFETY):
        raise ValueError('Archived control state mismatch')
