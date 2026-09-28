"""Runtime callbacks shared by continuous tile-coded control agents."""
from __future__ import annotations
import csv, os, pickle
from pathlib import Path
from typing import NamedTuple
import numpy as np
from .action_history import action_history_for_state, action_history_state, init_action_history, load_action_history_state, record_selected_action
from .runtime import CHECKPOINT_SCHEMA, INIT_CHECKPOINT_ENV, adopt_checkpoint_reward, adopt_checkpoint_safety, effective_legal_mask, load_common_configuration, save_checkpoint_atomic, validate_checkpoint
from .temporal_reward import observed_terminal_state, reset_temporal_reward_state, temporal_reward_context
from ..team_agent.exploration import epsilon_at
from ..team_agent.rewards import reward_from_events
from ..team_agent.safety import avoidable_fatal_action, mask_for_decision

class LinearTransition(NamedTuple):
    state: np.ndarray; action: int; reward: float; next_state: np.ndarray|None; done: bool; next_legal: np.ndarray|None; steps: int=1

def repeated_cycle_mask(item, legal, reward_id):
    """For r12, veto moves that would close an already repeated cycle."""
    result=np.asarray(legal,dtype=bool).copy()
    if reward_id != "r12_coin_priority_anti_loop": return result
    vector=np.asarray(item.vector)
    if vector.ndim != 1 or vector.size < 126: return result
    constrained=result.copy()
    for action in range(4):
        cycle_repeats=float(vector[110 + action * 3])
        if constrained[action] and cycle_repeats >= 0.5:
            constrained[action]=False
    return constrained if np.any(constrained) else result

def setup(self, config):
    load_common_configuration(self, feature_id=config.FEATURE_ID, default_model=config.MODEL_FILE)
    self.model = config.make_model(self.agent_seed); self.linear_config = config
    if self.model_file.exists():
        with self.model_file.open("rb") as file: checkpoint=pickle.load(file)
        adopt_checkpoint_reward(self, checkpoint); adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(checkpoint, algorithm=config.ALGORITHM, feature_id=config.FEATURE_ID, feature_schema=config.FEATURE_SCHEMA, actions=config.ACTIONS, reward_id=self.reward_id, hyperparameters=config.HYPERPARAMETERS, network_spec=None, training=self.train, training_task=self.training_task, safety_spec=self.safety_spec)
        self.model.load_checkpoint(checkpoint, training=self.train)
        self.total_action_steps=int(checkpoint.get("total_action_steps",0)); same=checkpoint.get("training_task")==self.training_task
        self.stage_action_steps=int(checkpoint.get("stage_action_steps",self.total_action_steps)) if same else 0; self.action_steps=self.total_action_steps
        for name in ("safe_exploration_decisions","safe_exploration_fallbacks","safety_decisions","safety_interventions","safety_fallbacks"): setattr(self,name,int(checkpoint.get(name,0)))
        if self.train:
            self.rng.setstate(checkpoint["agent_rng_state"])
            if same:
                loader=getattr(config,"load_action_history_state",load_action_history_state)
                loader(self,checkpoint.get("action_history_state"))
    elif self.train and os.getenv(INIT_CHECKPOINT_ENV):
        source=Path(os.environ[INIT_CHECKPOINT_ENV]).expanduser().resolve()
        with source.open("rb") as file: checkpoint=pickle.load(file)
        if checkpoint.get("algorithm") != config.ALGORITHM: raise ValueError("warm-start checkpoint uses an incompatible algorithm")
        if checkpoint.get("feature_id") != config.FEATURE_ID or checkpoint.get("feature_schema") != config.FEATURE_SCHEMA: raise ValueError("warm-start checkpoint uses incompatible features")
        if checkpoint.get("hyperparameters") != config.HYPERPARAMETERS: raise ValueError("warm-start checkpoint uses incompatible hyperparameters")
        weights=np.asarray(checkpoint["weights"],dtype=np.float32)
        if weights.shape != self.model.weights.shape: raise ValueError("warm-start checkpoint has incompatible weights")
        self.model.weights=weights.copy(); self.model.coder.load_state_dict(checkpoint["tile_coder"])
        self.model.traces.fill(0.0); self.model.updates=0
    elif not self.train: raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")

def features(self, state):
    key=(state.get("round"),state.get("step"))
    if key != self._feature_cache_key:
        previous,wait=action_history_for_state(self,state); self._feature_cache_key=key
        self._feature_cache_value=self.linear_config.features_for_state(self,state,previous,wait)
    return self._feature_cache_value

def act(self, state):
    from agent_code.die_hardest._vendor.learning_common.action_history import advance_observation
    advance_observation(self, state)
    item=features(self,state); physical=effective_legal_mask(item.legal_mask,self.linear_config.ACTIONS,self.curriculum_allows_bomb)
    values=self.model.q_values(item.vector); raw=int(np.flatnonzero(physical)[np.argmax(values[physical])]); exploring=False
    if self.train:
        epsilon=epsilon_at(self.stage_action_steps,self.exploration_spec); self.model.set_epsilon(epsilon)
        self.stage_action_steps+=1; self.total_action_steps+=1; self.action_steps=self.total_action_steps; exploring=self.rng.random()<epsilon
    legal,fallback=mask_for_decision(state,physical,self.safety_spec,allow_bomb=self.curriculum_allows_bomb,exploring=exploring)
    legal=repeated_cycle_mask(item,legal,self.reward_id)
    enabled=self.safety_spec["mode"]=="all" or (self.safety_spec["mode"]=="exploration" and exploring)
    if enabled: self.safety_decisions+=1; self.safety_fallbacks+=int(fallback)
    indices=np.flatnonzero(legal).tolist()
    best=max(float(values[i]) for i in indices); tied=[i for i in indices if float(values[i])==best]
    if exploring: selected=self.rng.choice(indices)
    else: selected=self.rng.choice(tied) if self.train else tied[0]
    marker=getattr(self.model,"set_behavior_greedy",None)
    if marker is not None: marker(selected in tied)
    self.safety_interventions+=int(enabled and not fallback and not legal[raw]); action=self.linear_config.ACTIONS[selected]
    recorder=getattr(self.linear_config,"record_selected_action",record_selected_action)
    recorder(self,state,action); return action

def setup_training(self):
    self.round_reward=0.; self.pending=None; self.causal_bomb_transition=None
    reset_temporal_reward_state(self)

def transition(self, old_state, action, reward, new_state, done):
    old=features(self,old_state)
    if done: return LinearTransition(old.vector.copy(),self.linear_config.ACTIONS.index(action),reward,None,True,None)
    new=features(self,new_state); legal=effective_legal_mask(new.legal_mask,self.linear_config.ACTIONS,self.curriculum_allows_bomb)
    legal,_=mask_for_decision(new_state,legal,self.safety_spec,allow_bomb=self.curriculum_allows_bomb,exploring=False)
    legal=repeated_cycle_mask(new,legal,self.reward_id)
    return LinearTransition(old.vector.copy(),self.linear_config.ACTIONS.index(action),reward,new.vector.copy(),False,legal)

def game_events(self, old_state, action, new_state, events):
    if old_state is None or action not in self.linear_config.ACTIONS:return
    old=features(self,old_state); context=temporal_reward_context(self,action,old_state,new_state,events,reward_id=self.reward_id)
    physical=effective_legal_mask(old.legal_mask,self.linear_config.ACTIONS,self.curriculum_allows_bomb)
    context["avoidable_fatal"]=avoidable_fatal_action(old_state,action,physical,allow_bomb=self.curriculum_allows_bomb)
    reward=reward_from_events(events,self.reward_id,old_game_state=old_state,new_game_state=new_state,action=action,**context)
    current=transition(self,old_state,action,reward,new_state,False)
    self.model.observe(current); self.round_reward+=reward
    if action == "BOMB" and "BOMB_DROPPED" in events:
        self.causal_bomb_transition=(current.state.copy(),current.action)

def end_round(self, state, action, events):
    if state is not None and action in self.linear_config.ACTIONS:
        old=features(self,state); physical=effective_legal_mask(old.legal_mask,self.linear_config.ACTIONS,self.curriculum_allows_bomb)
        context=temporal_reward_context(self,action,state,observed_terminal_state(state,action,events),events,reward_id=self.reward_id)
        reward=reward_from_events(events,self.reward_id,old_game_state=state,terminal=True,action=action,avoidable_fatal=avoidable_fatal_action(state,action,physical,allow_bomb=self.curriculum_allows_bomb),**context)
        if "KILLED_SELF" in events and self.causal_bomb_transition is not None:
            penalty=float(self.reward_spec.get("causal_bomb_death_penalty",0.0))
            if penalty:
                bomb_state,bomb_action=self.causal_bomb_transition
                self.model.retrospective_penalty(bomb_state,bomb_action,penalty)
                self.round_reward+=penalty
        self.model.observe(transition(self,state,action,reward,None,True)); self.round_reward+=reward
    self.causal_bomb_transition=None
    reset_temporal_reward_state(self); config=self.linear_config
    initializer=getattr(config,"init_action_history",init_action_history); initializer(self)
    history_state=getattr(config,"action_history_state",action_history_state)
    payload=self.model.checkpoint(); payload.update({"checkpoint_schema":CHECKPOINT_SCHEMA,"algorithm":config.ALGORITHM,"actions":list(config.ACTIONS),"feature_id":config.FEATURE_ID,"feature_version":None,"feature_schema":config.FEATURE_SCHEMA,"reward_id":self.reward_id,"reward_version":self.reward_id,"reward_spec":self.reward_spec,"hyperparameters":config.HYPERPARAMETERS,"network_spec":None,"action_steps":self.total_action_steps,"total_action_steps":self.total_action_steps,"stage_action_steps":self.stage_action_steps,"agent_seed":self.agent_seed,"agent_rng_state":self.rng.getstate(),"exploration_spec":self.exploration_spec,"safe_exploration":self.safe_exploration,"safe_exploration_decisions":self.safe_exploration_decisions,"safe_exploration_fallbacks":self.safe_exploration_fallbacks,"safety_spec":self.safety_spec,"safety_decisions":self.safety_decisions,"safety_interventions":self.safety_interventions,"safety_fallbacks":self.safety_fallbacks,"action_history_state":history_state(self),"n_step":1,"n_step_state":None,"retention_spec":self.retention_spec,"training_budget":self.training_budget,"training_task":self.training_task,"training_device_name":None,"training_device_type":"cpu"})
    save_checkpoint_atomic(self.model_file,payload,_save_pickle); _append_metrics(self,state); self.round_reward=0.

def _save_pickle(data, path):
    with path.open("wb") as file: pickle.dump(data,file,protocol=pickle.HIGHEST_PROTOCOL)

def _append_metrics(self,state):
    run_dir=os.getenv("BOMBERMAN_RUN_DIR")
    if not run_dir:return
    path=os.path.join(run_dir,"training.csv"); fields=("schema_version","algorithm","round","reward","action_steps","stage_action_steps","epsilon","q_states","loss","updates","replay_size","safe_exploration_decisions","safe_exploration_fallbacks","safety_decisions","safety_interventions","safety_fallbacks","checkpoint")
    row={"schema_version":"training-v2","algorithm":self.linear_config.ALGORITHM,"round":"" if state is None else state.get("round",""),"reward":self.round_reward,"action_steps":self.total_action_steps,"stage_action_steps":self.stage_action_steps,"epsilon":epsilon_at(self.stage_action_steps,self.exploration_spec),"q_states":self.linear_config.HYPERPARAMETERS["memory_size"],"loss":"","updates":self.model.updates,"replay_size":"","safe_exploration_decisions":self.safe_exploration_decisions,"safe_exploration_fallbacks":self.safe_exploration_fallbacks,"safety_decisions":self.safety_decisions,"safety_interventions":self.safety_interventions,"safety_fallbacks":self.safety_fallbacks,"checkpoint":str(self.model_file)}
    exists=os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        if not exists:writer.writeheader()
        writer.writerow(row)
