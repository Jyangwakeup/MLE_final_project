"""Policy-only specialist initialization, never a relaxation of legacy resume."""
from pathlib import Path
import copy
import json
import random
import subprocess

from experiments.task4_transfer import PARENT_SHA256, TARGET_SAFETY, SAFETY_MIGRATION, sha256, digest, validate_payload
from agent_code.learning_common.effective_learning import VERSION, resolve


def contract_for(root, config_path, seed):
    root, config_path = Path(root), Path(config_path)
    config = json.loads(config_path.read_text())
    spec = config['exploration_contract']
    manifest = json.loads((root / spec['manifest']).read_text())
    arm = spec['arm']
    if manifest['schema_version'] != VERSION or arm not in ('E1','E2','E3'):
        raise ValueError('Exploration protocol/arm mismatch')
    if (root / manifest['arm_configs'][arm]).resolve() != config_path.resolve():
        raise ValueError('Exploration config path mismatch')
    if seed not in manifest['training_seeds'] + list(manifest['diagnostic_training_seeds'].values()):
        raise ValueError('Unregistered exploration learning seed')
    from agent_code.double_dqn_continuous_v2_agent.callbacks import HYPERPARAMETERS
    effective = resolve(HYPERPARAMETERS, config)
    if (config['safety'] != TARGET_SAFETY or config['feature_id'] != 'continuous-v2'
            or config['reward_id'] != ('r7_safe_credit_sparse' if arm == 'E1' else 'task4-score-v1')
            or config['training']['n_step'] != 5
            or config['training']['retention'] != RETENTION
            or config['training']['exploration'] != EXPLORATION
            or config['training']['safety_replay']['enabled']):
        raise ValueError('Exploration configuration mismatch')
    if manifest['parent']['checkpoint_sha256'] != PARENT_SHA256:
        raise ValueError('Exploration parent mismatch')
    for path, expected in manifest['opponent_hashes'].items():
        if sha256(root/path) != expected:
            raise ValueError('Opponent source changed')
    from experiments.run import _source_hash
    return dict(version=VERSION, arm=arm, seed=seed, parent_sha256=PARENT_SHA256,
                target_safety=TARGET_SAFETY, hyperparameters=effective,
                reward_id=config['reward_id'], config_sha256=sha256(config_path),
                manifest_sha256=digest(manifest), opponent_hashes=manifest['opponent_hashes'],
                source_hash=_source_hash('double_dqn_continuous_v2_agent'),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())

RETENTION = dict(parent_fraction=0., distillation_weight=0., temperature=1.,
                 per_task_capacity=20000, current_warmup=2000, sampling_version='task4-only-v1')
EXPLORATION = dict(version='linear-v1', start=.20, end=.05, decay_action_steps=100000)


def initialize(payload, config, seed, training_budget, contract, *, retention_spec=None):
    """Pure construction from a validated parent; no inherited learning samples."""
    import torch
    from agent_code.dqn_agent.model import DQN
    from agent_code.team_agent.rewards import resolve_reward_spec
    # Reuse the full original parent contract validator without weakening it.
    original = copy.deepcopy(config)
    original['reward_id'] = 'r7_safe_credit_sparse'
    original['task4_contract'] = dict(arm='B', safety_migration=SAFETY_MIGRATION)
    original['training']['retention'] = dict(payload['retention_spec'], parent_fraction=.5)
    original['training']['exploration'] = payload['exploration_spec']
    validate_payload(payload, original)
    child = dict(payload)
    h = config['learning']
    model = DQN(84, 6, seed=seed, gamma=h['gamma'], learning_rate=h['learning_rate'],
                batch_size=64, replay_capacity=20000, warmup=2000, target_sync_interval=1000,
                device='cpu', training_task='full_match', retention_spec=RETENTION if retention_spec is None else retention_spec,
                safety_replay_spec=config['training']['safety_replay'], double_dqn=True, hidden_size=128)
    model.load_policy_weights(payload)
    child.update(model.checkpoint())
    child.update(training_task='full_match', agent_seed=seed,
                 action_steps=0, total_action_steps=0, stage_action_steps=0,
                 agent_rng_state=random.Random(seed).getstate(),
                 torch_rng_state=torch.Generator().manual_seed(seed).get_state(),
                 distillation_rng_state=random.Random(seed+104729).getstate(),
                 hyperparameters=h, reward_id=config['reward_id'], reward_version=config['reward_id'],
                 reward_spec=resolve_reward_spec(config['reward_id']),
                 exploration_spec=EXPLORATION, safety_spec=TARGET_SAFETY,
                 training_budget=training_budget, transfer_contract=contract)
    # All cumulative safety counters reset; episode state already validated empty.
    for key in ('safe_exploration_decisions','safe_exploration_fallbacks','safety_decisions',
                'safety_interventions','safety_fallbacks','robust_safety_interventions',
                'robust_to_v1_fallbacks','opponent_robust_interventions','opponent_to_v3_fallbacks',
                'opponent_scenarios_evaluated','robust_guarantee_losses','robust_search_timeouts',
                'robust_states_evaluated','v1_to_physical_fallbacks','avoidable_escape_collapses','own_bomb_cycles'):
        child[key] = 0
    child['n_step_state'] = dict(payload['n_step_state'], gamma=h['gamma'])
    return child


def materialize(parent, destination, *, root, config_path, seed, training_budget):
    import torch
    if sha256(parent) != PARENT_SHA256:
        raise ValueError('Exploration requires registered Task3 parent SHA')
    config = json.loads(Path(config_path).read_text())
    contract = contract_for(root, config_path, seed)
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    child = initialize(payload, config, seed, training_budget, contract)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(child, temporary)
    temporary.replace(destination)
    return contract


def validate_resume(contract, *, root, config_path, seed):
    if contract != contract_for(root, config_path, seed):
        raise ValueError('Exploration resume source/arm/manifest/config mismatch')
