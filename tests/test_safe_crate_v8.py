from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from agent_code.rainbow_lite_agent.model import RainbowLite
from all_other_agent_code.rainbow_lite_v7_agent.features import (
    features_for_state as v7_features_for_state,
    init_action_history as init_v7_history,
)
from all_other_agent_code.rainbow_lite_v8_agent.callbacks import HYPERPARAMETERS
from all_other_agent_code.rainbow_lite_v8_agent.features import (
    features_for_state as v8_features_for_state,
    init_action_history as init_v8_history,
)
from agent_code.team_agent.feature_system import get_feature_schema
from experiments.migrate_rainbow_v7_to_v8 import migrate
from tests.test_danger import make_game_state


class SafeCrateV8Tests(unittest.TestCase):
    def test_feature_requires_legal_survivable_crate_hitting_bomb(self):
        owner = SimpleNamespace()
        init_v8_history(owner)
        useful = make_game_state(position=(3, 3))
        useful["field"][5, 3] = 1
        value = v8_features_for_state(owner, useful).vector

        self.assertEqual(
            get_feature_schema(
                "continuous-v8-safe-crate-opportunity").vector_shape,
            (163,))
        self.assertEqual(value.shape, (163,))
        self.assertGreater(value[-1], 0.0)

        no_crate = make_game_state(position=(3, 3))
        self.assertEqual(v8_features_for_state(owner, no_crate).vector[-1], 0.0)

        no_bomb = make_game_state(position=(3, 3), bombs_left=False)
        no_bomb["field"][5, 3] = 1
        self.assertEqual(v8_features_for_state(owner, no_bomb).vector[-1], 0.0)

    def test_zero_extended_migration_preserves_initial_q_values(self):
        source_model = RainbowLite(
            162, 6, seed=7, hyperparameters=HYPERPARAMETERS,
            device="cpu", training_task="full_match")
        payload = {
            "checkpoint_schema": "training-resume-v7",
            "algorithm": "rainbow_lite",
            "feature_id": "continuous-v7-phase-aware",
            "policy": source_model.policy.state_dict(),
        }
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1
        v7_owner = SimpleNamespace()
        v8_owner = SimpleNamespace()
        init_v7_history(v7_owner)
        init_v8_history(v8_owner)
        old_features = v7_features_for_state(v7_owner, state).vector
        new_features = v8_features_for_state(v8_owner, state).vector

        with TemporaryDirectory() as directory:
            source = Path(directory) / "v7.pt"
            output = Path(directory) / "v8.pt"
            torch.save(payload, source)
            migrate(source, output)
            migrated = torch.load(output, map_location="cpu", weights_only=True)

        target_model = RainbowLite(
            163, 6, seed=9, hyperparameters=HYPERPARAMETERS,
            device="cpu", training_task="full_match")
        target_model.policy.load_state_dict(migrated["policy"])
        old_q = source_model.q_values(old_features)
        new_q = target_model.q_values(new_features)

        np.testing.assert_allclose(new_q, old_q, rtol=0.0, atol=1e-7)
        self.assertTrue(torch.count_nonzero(
            migrated["policy"]["trunk.0.weight"][:, -1]) == 0)


if __name__ == "__main__":
    unittest.main()
