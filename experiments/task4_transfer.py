"""Audited frozen Task 3 -> Task 4 transfer; ordinary resume remains exact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

VERSION = 'task4-transfer-v1'
PARENT_SHA256 = 'b0b9e7ae9cbefa6523ed01e1d7a6d474b90b6272f9de1706ee80f1f109cc803d'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def contract_for(root, config_path, seed):
    root, config_path = Path(root), Path(config_path)
    config = json.loads(config_path.read_text())
    spec = config['task4_contract']
    manifest_path = root / spec['manifest']
    manifest = json.loads(manifest_path.read_text())
    arm = spec['arm']
    if arm not in ('A', 'B') or Path(manifest['arm_configs'][arm]).name != config_path.name:
        raise ValueError('Task 4 arm/config mismatch')
    if seed not in manifest['training_seeds'] + list(manifest['diagnostic_training_seeds'].values()):
        raise ValueError('Unregistered Task 4 training seed')
    if manifest['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Task 4 requires the published Task 3 parent')
    for name, expected in manifest['opponent_hashes'].items():
        if sha256(root/name) != expected:
            raise ValueError('Task 4 opponent source changed')
    return dict(version=VERSION, arm=arm, seed=seed,
                parent_sha256=PARENT_SHA256,
                parent_source_commit=manifest['parent']['source_commit'],
                manifest_sha256=digest(manifest), config_sha256=sha256(config_path),
                opponent_hashes=manifest['opponent_hashes'],
                opponents=['rule_based_agent']*3)


def validate_payload(payload, config):
    from agent_code.double_dqn_continuous_v2_agent.callbacks import (
        FEATURE_SCHEMA, NETWORK_SPEC, HYPERPARAMETERS, ACTIONS,
    )
    from agent_code.team_agent.rewards import resolve_reward_spec
    expected = dict(checkpoint_schema='training-resume-v11',
                    lifecycle_version='decision-snapshot-v1', training_task='weak_opponents',
                    algorithm='double_dqn', feature_id='continuous-v2',
                    feature_schema=FEATURE_SCHEMA, network_spec=NETWORK_SPEC,
                    hyperparameters=HYPERPARAMETERS, actions=list(ACTIONS), n_step=5,
                    reward_id='r7_safe_credit_sparse', training_device_type='cpu',
                    training_device_name=None, safety_spec=config['safety'],
                    reward_spec=resolve_reward_spec('r7_safe_credit_sparse'))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'Task 4 parent {key} mismatch')
    if config['feature_id'] != 'continuous-v2' or config['reward_id'] != expected['reward_id']:
        raise ValueError('Task 4 feature/reward must remain frozen')
    training = config['training']
    retention = dict(payload['retention_spec'])
    retention['parent_fraction'] = .75 if config['task4_contract']['arm']=='A' else .5
    if (training['retention'] != retention or training['n_step'] != 5
            or training['exploration'] != payload['exploration_spec']
            or training['safety_replay'] != payload['safety_replay_spec']):
        raise ValueError('Task 4 training contract mismatch')
    if payload['n_step_state']['pending']:
        raise ValueError('Task 4 requires an empty n-step boundary')
    history = payload['action_history_state']
    if any(value not in (None, False, 0, (), []) for value in history.values()):
        raise ValueError('Task 4 requires empty decision history')
    if any(value not in (None, False) for value in payload['own_bomb_escape_state'].values()):
        raise ValueError('Task 4 requires an empty bomb responsibility certificate')
    if set(payload['replay']['task_ids']) != {'coin_navigation', 'crate_navigation', 'weak_opponents'}:
        raise ValueError('Task 4 parent replay partitions mismatch')
    if payload.get('transfer_contract') or payload.get('distillation_dataset') is not None:
        raise ValueError('Unsupported prior transfer/distillation dataset')


def materialize(parent, destination, *, root, config_path, seed, training_budget):
    import torch
    contract = contract_for(root, config_path, seed)
    if sha256(parent) != PARENT_SHA256:
        raise ValueError('Task 4 parent checkpoint SHA-256 mismatch')
    config = json.loads(Path(config_path).read_text())
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    validate_payload(payload, config)
    contract['parent_agent_seed'] = payload['agent_seed']
    contract['parent_stage_action_steps'] = payload['stage_action_steps']
    payload['teacher'] = {k: value.clone() for k,value in payload['policy'].items()}
    payload['training_task'] = 'full_match'
    payload['retention_spec'] = config['training']['retention']
    payload['stage_action_steps'] = 0
    payload['agent_seed'] = seed
    payload['agent_rng_state'] = random.Random(seed).getstate()
    payload['replay']['rng_state'] = random.Random(seed).getstate()
    payload['replay']['current_task'] = 'full_match'
    payload['torch_rng_state'] = torch.Generator(device='cpu').manual_seed(seed).get_state()
    payload['distillation_rng_state'] = random.Random(seed+104729).getstate()
    payload['training_budget'] = training_budget
    counters = (
        'safe_exploration_decisions','safe_exploration_fallbacks','safety_decisions',
        'safety_interventions','safety_fallbacks','robust_safety_interventions',
        'robust_to_v1_fallbacks','opponent_robust_interventions','opponent_to_v3_fallbacks',
        'opponent_scenarios_evaluated','robust_guarantee_losses','robust_search_timeouts',
        'robust_states_evaluated','v1_to_physical_fallbacks','avoidable_escape_collapses',
        'own_bomb_cycles',
    )
    contract['parent_safety_counts'] = {k:payload.get(k,0) for k in counters}
    for key in counters:
        payload[key] = 0
    payload['transfer_contract'] = contract
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)
    return contract


def validate_resume(contract, *, root, config_path, seed):
    expected = contract_for(root, config_path, seed)
    if not contract or any(contract.get(k) != value for k,value in expected.items()):
        raise ValueError('Task 4 resume provenance/arm/manifest/config mismatch')
