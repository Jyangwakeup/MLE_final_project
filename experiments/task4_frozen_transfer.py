"""Strict policy-only transfer for the frozen historical-opponent experiment."""
import json
from pathlib import Path
import subprocess
from experiments.task4_transfer import PARENT_SHA256, TARGET_SAFETY, digest, sha256
from experiments.task4_exploration_transfer import initialize, RETENTION, EXPLORATION
from agent_code.learning_common.effective_learning import resolve

VERSION = 'task4-frozen-opponents-v1'


def contract_for(root, config_path, seed):
    root, config_path = Path(root), Path(config_path)
    config = json.loads(config_path.read_text())
    spec = config['frozen_opponents']
    manifest = json.loads((root/spec['manifest']).read_text())
    if spec.get('role') != 'learner':raise ValueError('Only learner configurations may transfer')
    arm = spec['arm']
    if (manifest['schema_version'] != VERSION or spec['version'] != VERSION
            or spec['role'] != 'learner' or arm not in ('C', 'S')
            or (root/manifest['arm_configs'][arm]).resolve() != config_path.resolve()):
        raise ValueError('Frozen opponent protocol/role/arm/config mismatch')
    if seed not in manifest['training_seeds'] + list(manifest['diagnostic_training_seeds'].values()):
        raise ValueError('Unregistered frozen-opponent learning seed')
    from agent_code.double_dqn_continuous_v2_agent.callbacks import HYPERPARAMETERS
    effective = resolve(HYPERPARAMETERS, config)
    if (config['safety'] != TARGET_SAFETY or config['feature_id'] != 'continuous-v2'
            or config['reward_id'] != 'r7_safe_credit_sparse' or config['training']['n_step'] != 5
            or config['training']['retention'] != RETENTION or config['training']['exploration'] != EXPLORATION
            or config['training']['safety_replay']['enabled']):
        raise ValueError('Frozen opponent training configuration mismatch')
    if manifest['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Frozen opponent parent mismatch')
    from experiments.frozen_opponents import VERSION as SCHEDULE
    if manifest['schedule_version'] != SCHEDULE or manifest['pool_sha256'] != digest(manifest['models']):
        raise ValueError('Frozen pool/scheduler mismatch')
    for path, expected in manifest['opponent_hashes'].items():
        if sha256(root/path) != expected:raise ValueError('Opponent source changed')
    for entry in manifest['models'].values():
        if sha256(entry['checkpoint']) != entry['sha256']:raise ValueError('Registered frozen weight changed')
    from experiments.run import _source_hash
    return dict(version=VERSION, role="learner", arm=arm, seed=seed, parent_sha256=PARENT_SHA256,
                target_safety=TARGET_SAFETY, hyperparameters=effective, reward_id=config['reward_id'],
                config_sha256=sha256(config_path), manifest_sha256=digest(manifest),
                opponent_hashes=manifest['opponent_hashes'], pool_sha256=manifest['pool_sha256'],
                schedule_version=SCHEDULE, schedule_seed=manifest['schedule_seed'],
                opponent_rng_root=manifest['opponent_rng_root'],
                source_hash=_source_hash('double_dqn_continuous_v2_agent'),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())


def materialize(parent, destination, *, root, config_path, seed, training_budget):
    import torch
    if sha256(parent) != PARENT_SHA256:raise ValueError('Requires registered original Task3 parent SHA')
    config = json.loads(Path(config_path).read_text())
    contract = contract_for(root, config_path, seed)
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    child = initialize(payload, config, seed, training_budget, contract)
    destination = Path(destination);destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp');torch.save(child, temporary);temporary.replace(destination)
    return contract


def validate_resume(contract, *, root, config_path, seed):
    if contract != contract_for(root, config_path, seed):
        raise ValueError('Frozen-opponent resume source/arm/pool/scheduler/manifest/config mismatch')
