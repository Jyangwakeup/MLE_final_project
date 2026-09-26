import os
import pickle
from pathlib import Path
import numpy as np

from all_other_agent_code.expected_sarsa_lambda_agent.callbacks import (
    ACTIONS, ACTION_FEATURE_INDICES, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA,
    HYPERPARAMETERS, make_model,
)
from agent_code.learning_common import linear_agent
from agent_code.learning_common.runtime import (
    INIT_CHECKPOINT_ENV, adopt_checkpoint_reward, adopt_checkpoint_safety,
    load_common_configuration, validate_checkpoint,
)
from .features import features_for_state

MODEL_FILE = Path(__file__).with_name("final.pkl")
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": None,
    "hyperparameters": HYPERPARAMETERS,
}

def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    self.model = make_model(self.agent_seed)
    self.linear_config = __import__(__name__, fromlist=["x"])
    if self.model_file.exists():
        with self.model_file.open("rb") as file:
            checkpoint = pickle.load(file)
        adopt_checkpoint_reward(self, checkpoint)
        adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
            reward_id=self.reward_id, hyperparameters=HYPERPARAMETERS,
            network_spec=None, training=self.train,
            training_task=self.training_task, safety_spec=self.safety_spec)
        self.model.load_checkpoint(checkpoint, training=self.train)
    elif self.train and os.getenv(INIT_CHECKPOINT_ENV):
        source = Path(os.environ[INIT_CHECKPOINT_ENV]).expanduser().resolve()
        with source.open("rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("algorithm") != ALGORITHM:
            raise ValueError("warm-start checkpoint uses an incompatible algorithm")
        if checkpoint.get("feature_id") != FEATURE_ID or checkpoint.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("warm-start checkpoint uses incompatible features")
        if checkpoint.get("hyperparameters") != HYPERPARAMETERS:
            raise ValueError("warm-start checkpoint uses incompatible hyperparameters")
        weights = np.asarray(checkpoint["weights"], dtype=np.float32)
        if weights.shape != self.model.weights.shape:
            raise ValueError("warm-start checkpoint has incompatible weights")
        self.model.weights = weights.copy()
        self.model.coder.load_state_dict(checkpoint["tile_coder"])
        self.model.traces.fill(0.0)
        self.model.updates = 0
    elif not self.train:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")

def act(self, game_state): return linear_agent.act(self, game_state)
def state_to_features(game_state):
    value=features_for_state(type("FeatureOwner",(),{})(),game_state)
    return None if value is None else value.vector
def legal_actions(game_state):
    value=features_for_state(type("FeatureOwner",(),{})(),game_state)
    return np.zeros(len(ACTIONS),dtype=bool) if value is None else value.legal_mask.copy()
