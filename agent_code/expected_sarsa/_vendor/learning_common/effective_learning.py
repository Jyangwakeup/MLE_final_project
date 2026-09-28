"""Explicit specialist parameters, leaving historical defaults unchanged."""
import json
import os
from pathlib import Path

VERSION = 'task4-exploration-v1'

def resolve(defaults, config=None, checkpoint=None):
    score = (config or {}).get('score_contract')
    saved_score = (checkpoint or {}).get('transfer_contract') or {}
    if (score and score['role'] != 'reference') or saved_score.get('version') == 'task4-score-improvement-v1':
        arm = score['arm'] if score else saved_score['arm']
        if arm not in ('C', 'L', 'LP', 'LPK'): raise ValueError('Unknown score improvement arm')
        expected = dict(defaults, gamma=.95, learning_rate=1e-4 if arm == 'C' else 3e-5)
        if score and (config or {}).get('learning') != expected:
            raise ValueError('Score effective learning mismatch')
        if checkpoint is not None:
            archive = score and score['role'] == 'archive'
            version = 'task4-frozen-opponents-v1' if archive else 'task4-score-improvement-v1'
            if saved_score.get('version') != version or saved_score.get('arm') != arm or checkpoint['hyperparameters'] != expected:
                raise ValueError('Score checkpoint learning/arm mismatch')
        return expected
    frozen = (config or {}).get('frozen_opponents')
    saved_contract = (checkpoint or {}).get('transfer_contract') or {}
    if (frozen and frozen.get('role') == 'learner') or saved_contract.get('version') == 'task4-frozen-opponents-v1':
        expected = dict(defaults, gamma=.95, learning_rate=1e-4)
        if frozen and ((config or {}).get('learning') != expected or frozen.get('arm') not in ('C','S')):
            raise ValueError('Frozen-opponent learner parameter contract mismatch')
        if checkpoint is not None and (saved_contract.get('version') != 'task4-frozen-opponents-v1'
                or checkpoint['hyperparameters'] != expected
                or (frozen and saved_contract['arm'] != frozen['arm'])):
            raise ValueError('Frozen-opponent checkpoint parameter/arm mismatch')
        return expected
    spec = (config or {}).get('exploration_contract')
    saved = (checkpoint or {}).get('transfer_contract') or {}
    specialist = saved.get('version') == VERSION
    if spec is None and not specialist:
        return dict(defaults)
    arm = spec['arm'] if spec else saved['arm']
    if arm not in ('E1', 'E2', 'E3'):
        raise ValueError('Unknown exploration arm')
    expected = dict(defaults, gamma=.99 if arm == 'E3' else .95, learning_rate=1e-4)
    if spec and (spec.get('version') != VERSION or (config or {}).get('learning') != expected):
        raise ValueError('Exploration effective parameters mismatch')
    if checkpoint is not None and (not specialist or saved['arm'] != arm or checkpoint['hyperparameters'] != expected):
        raise ValueError('Exploration checkpoint parameters/arm mismatch')
    return expected

def from_environment(defaults, checkpoint=None):
    path = os.getenv('BOMBERMAN_CONFIG')
    config = json.loads(Path(path).read_text()) if path else None
    return resolve(defaults, config, checkpoint)
