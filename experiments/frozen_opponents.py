"""Round-local frozen seats with isolated random streams and balanced scheduling."""
from contextlib import contextmanager
import copy
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
from experiments.task4_transfer import sha256, digest, TARGET_SAFETY

VERSION = 'frozen-opponent-schedule-v1'
WRAPPER = 'frozen_history_agent'
_TEMPLATES = {}


def stream_seed(*parts):
    # A separate high seed domain cannot collide with registered evaluation blocks.
    value=int.from_bytes(hashlib.sha256(json.dumps([VERSION,*parts]).encode()).digest()[:8],'big')
    return 100_000_000 + value % 1_900_000_000


def assignment(arm, round_index, schedule_seed):
    if arm not in ('C','S') or round_index<1:raise ValueError('Invalid schedule request')
    if arm=='C':return ('rule',)*3
    index=round_index-1;block,offset=divmod(index,10)
    modes=[False]*5+[True]*5
    random.Random(stream_seed(schedule_seed,'block',block)).shuffle(modes)
    if not modes[offset]:return ('rule',)*3
    historical=block*5+sum(modes[:offset]);cycle,position=divmod(historical,24)
    triples=list(itertools.permutations(range(4),3))
    random.Random(stream_seed(schedule_seed,'historical',cycle)).shuffle(triples)
    return tuple(('parent','e1_22_40k','e1_11_40k','e1_33_20k')[i] for i in triples[position])


def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state()


def set_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])


@contextmanager
def isolated(state=None):
    outside=rng_state()
    try:
        if state is not None:set_rng(state)
        yield
    finally:set_rng(outside)


@contextmanager
def environment(values):
    previous={k:v for k,v in os.environ.items() if k.startswith('BOMBERMAN_')}
    try:
        for k in list(previous):os.environ.pop(k,None)
        os.environ.update(values)
        yield
    finally:
        for k in list(os.environ):
            if k.startswith('BOMBERMAN_'):os.environ.pop(k,None)
        os.environ.update(previous)


def policy_digest(model, target=False):
    h=hashlib.sha256()
    for key,tensor in sorted((model.target if target else model.policy).state_dict().items()):
        h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def template(entry):
    path=Path(entry['checkpoint']);signature=(path.stat().st_size,path.stat().st_mtime_ns)
    key=(str(path.resolve()),entry['sha256'],digest(TARGET_SAFETY))
    cached=_TEMPLATES.get(key)
    if cached and cached[0]==signature:return cached[1]
    if sha256(path)!=entry['sha256']:raise ValueError('Frozen opponent weight SHA mismatch')
    from agent_code.double_dqn_continuous_v2_agent import callbacks
    env={'BOMBERMAN_CHECKPOINT':str(path),'BOMBERMAN_TORCH_DEVICE':'cpu',
         'BOMBERMAN_AGENT_SEED':'0','BOMBERMAN_ALLOW_BOMB':'true',
         'BOMBERMAN_SAFETY_SPEC':json.dumps(TARGET_SAFETY)}
    with isolated(),environment(env):
        owner=SimpleNamespace(train=False,logger=logging.getLogger('frozen-policy'))
        callbacks.setup(owner)
    if (owner.effective_hyperparameters != entry['hyperparameters'] or owner.reward_id!=entry['reward_id']
            or owner.safety_spec!=TARGET_SAFETY or owner.feature_id!='continuous-v2'):
        raise ValueError('Frozen opponent runtime contract mismatch')
    owner.model.policy.eval();owner.model.target.eval()
    for net in (owner.model.policy,owner.model.target):
        for parameter in net.parameters():parameter.requires_grad_(False)
    owner._frozen_policy_digest=policy_digest(owner.model)
    owner._frozen_target_digest=policy_digest(owner.model, True)
    owner._frozen_updates=owner.model.updates
    _TEMPLATES[key]=(signature,owner)
    return owner


class Delegate:
    def __init__(self, kind, entry, seed, logger):
        self.kind=kind
        with isolated():
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if kind=='rule':
                from agent_code.rule_based_agent import callbacks
                self.owner=SimpleNamespace(train=False,logger=logger)
                callbacks.setup(self.owner)
                # Official setup calls np.random.seed() without an argument.
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            else:
                from agent_code.double_dqn_continuous_v2_agent import callbacks
                from agent_code.learning_common.action_history import init_action_history
                self.owner=copy.copy(template(entry))
                self.owner.logger=logger;self.owner.rng=random.Random(seed)
                init_action_history(self.owner)
                self.owner._decision_snapshot=None
                self.owner._feature_cache_key=None;self.owner._feature_cache_value=None
                self.owner._own_bomb_cycle_had_safe_alternative=False
                self.owner._own_bomb_cycle_collapse_recorded=False
                self.owner._own_bomb_placement_certificate=None
                self.owner.last_safety_diagnostic=None
            self.callbacks=callbacks;self.random_state=rng_state()

    def act(self,state):
        with isolated(self.random_state):
            action=self.callbacks.act(self.owner,state)
            self.random_state=rng_state()
        return action


class Manager:
    def __init__(self, root, config, kind, environment_seed, learning_seed):
        self.root=Path(root);self.config=config;self.kind=kind
        self.manifest=json.loads((self.root/config['frozen_opponents']['manifest']).read_text())
        self.arm=config['frozen_opponents'].get('arm','C')
        self.environment_seed=environment_seed;self.learning_seed=learning_seed
        self.schedule_seed=stream_seed(self.manifest['schedule_seed'],learning_seed)
        if kind not in ('training','rules','heldout','engineering'):raise ValueError('Unknown opponent role')
        ids=(list(self.manifest['pool']) if kind in ('training','engineering') and (self.arm=='S' or kind=='engineering')
             else self.manifest['heldout_ids'] if kind=='heldout' else [])
        for key in ids:template(self.manifest['models'][key])

    def bind(self,world):
        number=int(world.round)+1
        if self.kind=='training':roster=assignment(self.arm,number,self.schedule_seed)
        elif self.kind=='rules':roster=('rule',)*3
        elif self.kind=='heldout':
            roster=list(itertools.permutations(self.manifest['heldout_ids']))[self.environment_seed%6]
        else:
            triples=list(itertools.permutations(self.manifest['pool'],3))
            roster=triples[self.manifest['diagnostic_seeds'].index(self.environment_seed)%24]
        record=dict(round=number,kind=self.kind,roster=roster,version=VERSION,seats=[])
        if len(world.agents)!=4:raise ValueError('Frozen opponents require exactly four players')
        for seat,(agent,key) in enumerate(zip(world.agents[1:],roster)):
            if agent.code_name!=WRAPPER or agent.train:raise ValueError('Invalid frozen framework seat')
            seed=stream_seed(self.manifest['opponent_rng_root'],self.environment_seed,number,seat)
            owner=agent.backend.runner.fake_self
            entry=None if key=='rule' else self.manifest['models'][key]
            owner.frozen_delegate=Delegate('rule' if key=='rule' else 'neural',entry,seed,owner.logger)
            owner.last_safety_diagnostic=None
            owner.frozen_model_id=key
            record['seats'].append(dict(agent=agent.name,model_id=key,seed=seed,sha256=None if entry is None else entry['sha256']))
        with (world._output/'opponent_schedule.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')

    def verify(self,world):
        for agent in world.agents[1:]:
            delegate=agent.backend.runner.fake_self.frozen_delegate
            if delegate.kind!='neural':continue
            owner=delegate.owner
            if (policy_digest(owner.model)!=owner._frozen_policy_digest or len(owner.model.replay)
                    or owner.model.teacher is not None or owner.model.optimizer.state
                    or policy_digest(owner.model, True)!=owner._frozen_target_digest
                    or owner.model.updates!=owner._frozen_updates):
                raise ValueError('Frozen opponent learning state mutated')
