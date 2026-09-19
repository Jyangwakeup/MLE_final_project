"""Explicit specialist parameters, leaving historical defaults unchanged."""
import json
import os
from pathlib import Path

VERSION = 'task4-exploration-v1'

def resolve(defaults, config=None, checkpoint=None):
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
