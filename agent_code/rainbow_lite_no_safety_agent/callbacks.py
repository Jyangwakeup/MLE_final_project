import os
from pathlib import Path
import torch

from agent_code.rainbow_lite_agent.callbacks import (
    ACTIONS, ALGORITHM, FEATURE_ID, FEATURE_SCHEMA, HYPERPARAMETERS, NETWORK_SPEC,
    _features_for, act, legal_actions, state_to_features,
)
from agent_code.rainbow_lite_agent.model import RainbowLite
from agent_code.learning_common.runtime import (
    INIT_CHECKPOINT_ENV, adopt_checkpoint_reward, adopt_checkpoint_safety,
    load_common_configuration, validate_checkpoint,
    validate_warm_start_checkpoint,
)

MODEL_FILE = Path(__file__).with_name("final.pt")
AGENT_METADATA = {
    "algorithm": ALGORITHM, "feature_id": FEATURE_ID,
    "checkpoint_name": MODEL_FILE.name, "network_spec": NETWORK_SPEC,
    "hyperparameters": HYPERPARAMETERS,
}

def setup(self):
    load_common_configuration(self, feature_id=FEATURE_ID, default_model=MODEL_FILE)
    if self.n_step not in (1, HYPERPARAMETERS["n_step"]):
        raise ValueError("Rainbow-lite fixes n-step at 4")
    self.n_step = HYPERPARAMETERS["n_step"]
    self.model = RainbowLite(
        126, len(ACTIONS), seed=self.agent_seed,
        hyperparameters=HYPERPARAMETERS,
        device=os.getenv("BOMBERMAN_TORCH_DEVICE", "cpu"),
        training_task=self.training_task, retention_spec=self.retention_spec)
    if self.model_file.exists():
        checkpoint = torch.load(self.model_file, map_location="cpu", weights_only=True)
        adopt_checkpoint_reward(self, checkpoint); adopt_checkpoint_safety(self, checkpoint)
        validate_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
            reward_id=self.reward_id, hyperparameters=HYPERPARAMETERS,
            network_spec=NETWORK_SPEC, training=self.train,
            training_task=self.training_task, safety_spec=self.safety_spec)
        self.model.load_checkpoint(checkpoint, training=self.train,
                                   training_task=self.training_task)
    elif self.train and os.getenv(INIT_CHECKPOINT_ENV):
        source = Path(os.environ[INIT_CHECKPOINT_ENV]).expanduser().resolve()
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        validate_warm_start_checkpoint(
            checkpoint, algorithm=ALGORITHM, feature_id=FEATURE_ID,
            feature_schema=FEATURE_SCHEMA, actions=ACTIONS,
            network_spec=NETWORK_SPEC)
        self.model.policy.load_state_dict(checkpoint["policy"])
        self.model.target.load_state_dict(checkpoint["policy"])
        self.model.updates = 0
    elif not self.train:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {self.model_file}")
