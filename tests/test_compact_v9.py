from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from agent_code.rainbow_lite_agent.model import RainbowLite
from agent_code.rainbow_lite_v8_agent.features import (
    features_for_state as v8_features_for_state,
    init_action_history as init_v8_history,
)
from agent_code.rainbow_lite_v9_agent.callbacks import HYPERPARAMETERS
from agent_code.rainbow_lite_v9_agent.features import (
    features_for_state as v9_features_for_state,
    init_action_history as init_v9_history,
)
from agent_code.team_agent.feature_system import continuous_v8, continuous_v9
from experiments.migrate_rainbow_v8_to_v9 import migrate
from tests.test_danger import make_game_state


class CompactV9Tests(unittest.TestCase):
    def test_contract_removes_only_declared_fields(self):
        self.assertEqual(continuous_v9.FEATURE_DIM, 139)
        self.assertEqual(len(continuous_v9.DROPPED_FIELDS), 24)
        self.assertIn("own_bomb_pending", continuous_v9.VECTOR_FIELDS)
        self.assertIn("survivable_crate_bomb_utility", continuous_v9.VECTOR_FIELDS)
        self.assertTrue(
            continuous_v9.DROPPED_FIELDS.isdisjoint(continuous_v9.VECTOR_FIELDS))

    def test_projection_matches_v8_values_by_name(self):
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1
        v8_owner, v9_owner = SimpleNamespace(), SimpleNamespace()
        init_v8_history(v8_owner)
        init_v9_history(v9_owner)
        old = v8_features_for_state(v8_owner, state).vector
        new = v9_features_for_state(v9_owner, state).vector
        expected = old[np.asarray(continuous_v9.KEEP_INDICES)]
        np.testing.assert_array_equal(new, expected)

    def test_migration_preserves_initial_q_values(self):
        source_model = RainbowLite(
            163, 6, seed=7, hyperparameters=HYPERPARAMETERS,
            device="cpu", training_task="full_match")
        payload = {
            "checkpoint_schema": "training-resume-v7",
            "algorithm": "rainbow_lite",
            "feature_id": continuous_v8.FEATURE_ID,
            "policy": source_model.policy.state_dict(),
        }
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1
        v8_owner, v9_owner = SimpleNamespace(), SimpleNamespace()
        init_v8_history(v8_owner)
        init_v9_history(v9_owner)
        old_features = v8_features_for_state(v8_owner, state).vector
        new_features = v9_features_for_state(v9_owner, state).vector

        with TemporaryDirectory() as directory:
            source = Path(directory) / "v8.pt"
            output = Path(directory) / "v9.pt"
            torch.save(payload, source)
            migrate(source, output)
            migrated = torch.load(output, map_location="cpu", weights_only=True)

        target_model = RainbowLite(
            139, 6, seed=9, hyperparameters=HYPERPARAMETERS,
            device="cpu", training_task="full_match")
        target_model.policy.load_state_dict(migrated["policy"])
        np.testing.assert_allclose(
            target_model.q_values(new_features),
            source_model.q_values(old_features), rtol=0.0, atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
