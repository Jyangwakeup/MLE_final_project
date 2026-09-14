import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_code.legal_random_agent.callbacks import act, setup
from agent_code.team_agent.features import ACTIONS, extract_features
from agent_code.team_agent.rewards import resolve_reward_spec
from tests.test_danger import make_game_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "configs"


class FormalConfigurationTestCase(unittest.TestCase):
    def test_existing_formal_configs_preserve_r1(self):
        for name in (
            "base.json",
            "formal_training.json",
            "stage_gate.json",
            "main_validation.json",
            "final_test.json",
        ):
            with self.subTest(name=name):
                config = json.loads((CONFIG_ROOT / name).read_text())
                self.assertEqual(config["reward_version"], "r1")
                self.assertEqual(
                    resolve_reward_spec(config["reward_version"])["coin_collected"], 1.0)

    def test_coin3_training_is_an_explicit_frozen_feature_variant(self):
        config = json.loads(
            (CONFIG_ROOT / "formal_training_coin3.json").read_text())
        self.assertEqual(config["feature_id"], "discrete-v1")
        self.assertEqual(config["feature_version"], "v1")
        self.assertEqual(config["reward_id"], "r1_coin3")
        self.assertEqual(
            resolve_reward_spec(config["reward_id"])["coin_collected"], 3.0)

    def test_coin3_stage_gate_has_frozen_contract_and_diagnostics(self):
        config = json.loads((CONFIG_ROOT / "stage_gate_coin3.json").read_text())
        self.assertEqual(config["feature_id"], "discrete-v1")
        self.assertEqual(config["feature_version"], "v1")
        self.assertEqual(config["reward_id"], "r1_coin3")
        self.assertEqual(config["reward_version"], "r1_coin3")
        self.assertTrue(config["evaluation"]["navigation_diagnostics"])
        self.assertEqual(config["evaluation"]["seeds"], list(range(10_000, 10_005)))
        self.assertEqual(config["evaluation"]["n_rounds"], 20)
        self.assertEqual(config["evaluation"]["device"], "cpu")

    def test_formal_training_budget_and_exploration_are_pre_registered(self):
        config = json.loads((CONFIG_ROOT / "formal_training.json").read_text())

        self.assertEqual(config["feature_version"], "v1")
        self.assertEqual(config["reward_version"], "r1")
        self.assertEqual(config["training"]["device"], "cpu")
        self.assertFalse(config["training"]["early_stopping"]["enabled"])
        self.assertEqual(config["training"]["exploration"], {
            "version": "linear-v1",
            "start": 1.0,
            "end": 0.05,
            "decay_action_steps": 80_000,
        })
        self.assertEqual(config["curriculum"], {
            "training_seeds": [11, 22, 33],
            "rounds": {"1": 500, "2": 1000, "3": 1500, "4": 3000},
            "retry_rounds": {"1": 125, "2": 250, "3": 375, "4": 750},
        })

    def test_evaluation_configs_use_registered_world_seeds_and_rounds(self):
        cases = (
            ("stage_gate.json", list(range(10_000, 10_005)), 20),
            ("main_validation.json", list(range(10_000, 10_100)), 1),
            ("final_test.json", list(range(20_000, 20_100)), 1),
        )
        for name, expected_seeds, expected_rounds in cases:
            with self.subTest(name=name):
                config = json.loads((CONFIG_ROOT / name).read_text())
                self.assertEqual(config["evaluation"]["device"], "cpu")
                self.assertEqual(config["evaluation"]["n_rounds"], expected_rounds)
                self.assertEqual(config["evaluation"]["seeds"], expected_seeds)


class LegalRandomAgentTestCase(unittest.TestCase):
    def test_seeded_agent_samples_only_the_shared_legal_task1_actions(self):
        state = make_game_state(bombs_left=True)
        legal = extract_features(state).legal_mask.copy()
        legal[ACTIONS.index("BOMB")] = False
        first = SimpleNamespace(logger=Mock())
        second = SimpleNamespace(logger=Mock())

        with patch.dict(os.environ, {
            "BOMBERMAN_AGENT_SEED": "10001",
            "BOMBERMAN_ALLOW_BOMB": "false",
        }, clear=True):
            setup(first)
            setup(second)
            first_actions = [act(first, state) for _ in range(50)]
            second_actions = [act(second, state) for _ in range(50)]

        self.assertEqual(first_actions, second_actions)
        self.assertNotIn("BOMB", first_actions)
        self.assertTrue(all(legal[ACTIONS.index(action)] for action in first_actions))


if __name__ == "__main__":
    unittest.main()
