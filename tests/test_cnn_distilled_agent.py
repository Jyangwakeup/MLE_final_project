import unittest
import json
import logging
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def game_state(round_index=1, step=1, position=(1, 1)):
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    field[2::2, 2::2] = -1
    return {
        "round": round_index, "step": step, "field": field,
        "coins": [(3, 1), (5, 1)], "bombs": [],
        "explosion_map": np.zeros_like(field),
        "self": ("me", 0, True, position), "others": [],
    }


class DistilledCnnSymmetryTests(unittest.TestCase):
    def test_rotation_moves_board_action_mask_and_teacher_q_together(self):
        from agent_code.cnn_distilled_double_dqn_agent.symmetry import (
            transform_transition,
        )

        board = np.zeros((17, 3, 3), dtype=np.float32)
        board[2, 2, 1] = 1.0
        legal = np.array([True, False, False, False, True, False])
        teacher_q = np.arange(6, dtype=np.float32)

        moved, action, moved_legal, moved_q = transform_transition(
            board, 0, legal, teacher_q, "rot90"
        )

        self.assertEqual(action, 1)  # With field[x,y], np.rot90 maps UP to RIGHT.
        self.assertEqual(tuple(np.argwhere(moved[2] == 1)[0]), (1, 2))
        np.testing.assert_array_equal(
            moved_legal, [False, True, False, False, True, False]
        )
        np.testing.assert_array_equal(moved_q, [3, 0, 1, 2, 4, 5])

    def test_d4_averaged_network_is_action_equivariant(self):
        from agent_code.cnn_distilled_double_dqn_agent.model import (
            ActionAlignedQNetwork, D4AveragedQNetwork,
        )
        from agent_code.cnn_distilled_double_dqn_agent.symmetry import (
            TRANSFORMS, transform_actions, transform_board,
        )

        torch.manual_seed(7)
        board = np.random.default_rng(7).normal(size=(17, 9, 9)).astype(np.float32)
        board[3] = 0
        board[3, 4, 4] = 1
        network = D4AveragedQNetwork(ActionAlignedQNetwork(channels=8)).eval()
        with torch.no_grad():
            original = network(torch.from_numpy(board[None])).numpy()[0]
            for name in TRANSFORMS:
                transformed = network(torch.from_numpy(
                    transform_board(board, name)[None])).numpy()[0]
                np.testing.assert_allclose(
                    transformed, transform_actions(original, name),
                    rtol=1e-5, atol=1e-5,
                )


class DistillationLossTests(unittest.TestCase):
    def test_masked_kl_is_finite_with_one_legal_action_and_extreme_q(self):
        from agent_code.cnn_distilled_double_dqn_agent.distillation import masked_kl

        student = torch.tensor([[1e6, -1e6, 2.0, 3.0, 4.0, 5.0]])
        teacher = torch.tensor([[-1e6, 1e6, 8.0, 7.0, 6.0, 5.0]])
        legal = torch.tensor([[False, True, False, False, False, False]])
        loss = masked_kl(student, teacher, legal, temperature=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

        tied_teacher = torch.zeros((1, 6), requires_grad=False)
        all_legal = torch.ones((1, 6), dtype=torch.bool)
        student = torch.zeros((1, 6), requires_grad=True)
        tied_loss = masked_kl(student, tied_teacher, all_legal, temperature=1.0)
        tied_loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())


class DistilledAgentContractTests(unittest.TestCase):
    def test_five_step_return_is_reserved_for_the_distilled_cnn(self):
        from experiments.training import allowed_n_steps

        self.assertEqual(allowed_n_steps("cnn_distilled_double_dqn"), {1, 4, 5})
        self.assertEqual(allowed_n_steps("double_dqn"), {1, 4})

    def test_task2_reward_and_n5_configs_are_single_variable_ablations(self):
        root = Path(__file__).resolve().parents[1] / "experiments" / "configs"
        sparse = json.loads((
            root / "task2_cnn_wait_shaping.json").read_text(encoding="utf-8"))
        potential = json.loads((
            root / "task2_cnn_reward_potential.json").read_text(encoding="utf-8"))
        n5 = json.loads((
            root / "task2_cnn_reward_potential_n5.json").read_text(encoding="utf-8"))
        sparse_n5 = json.loads((
            root / "task2_cnn_wait_shaping_n5.json").read_text(encoding="utf-8"))

        self.assertEqual(potential["reward_id"], "r7_safe_credit_potential")
        potential_without_reward = dict(potential)
        potential_without_reward["reward_id"] = sparse["reward_id"]
        self.assertEqual(potential_without_reward, sparse)
        self.assertEqual(n5["training"]["n_step"], 5)
        n5_without_horizon = json.loads(json.dumps(n5))
        n5_without_horizon["training"]["n_step"] = 4
        self.assertEqual(n5_without_horizon, potential)
        sparse_n5["training"]["n_step"] = 4
        self.assertEqual(sparse_n5, sparse)

    def test_safety_diagnostics_report_bomb_veto_and_raw_q_interventions(self):
        from experiments.summarize_cnn_decisions import summarize

        records = [
            {
                "round": 1, "step": 1, "action": "RIGHT",
                "q_values": [0, 1, 0, 0, 2, 5],
                "physical_mask": [True] * 6,
                "legal_mask": [True, True, True, True, True, False],
                "safety_fallback": False,
            },
            {
                "round": 1, "step": 2, "action": "BOMB",
                "q_values": [0, 1, 0, 0, 5, 2],
                "physical_mask": [True] * 6,
                "legal_mask": [True, True, True, True, False, True],
                "safety_fallback": True,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            result = summarize([path])

        self.assertEqual(result["physical_bomb_available_count"], 2)
        self.assertEqual(result["safe_bomb_available_count"], 1)
        self.assertEqual(result["bomb_veto_count"], 1)
        self.assertEqual(result["bomb_veto_rate"], 0.5)
        self.assertEqual(result["raw_argmax_veto_count"], 2)
        self.assertEqual(result["raw_argmax_veto_by_action"], {
            "UP": 0, "RIGHT": 0, "DOWN": 0, "LEFT": 0,
            "WAIT": 1, "BOMB": 1,
        })
        self.assertEqual(result["raw_bomb_argmax_veto_rate"], 1.0)
        self.assertEqual(result["safety_fallback_count"], 1)

    def test_safety_override_is_frozen_diagnostic_only(self):
        from agent_code.cnn_distilled_double_dqn_agent.callbacks import (
            _diagnostic_safety_override,
        )

        original = {
            "version": "survival-mask-v1", "mode": "all",
            "horizon": 7, "fallback": "physical_q",
        }
        options = {
            "decision_diagnostics": True,
            "diagnostic_safety_override": "off",
        }
        overridden = _diagnostic_safety_override(original, options, train=False)
        self.assertEqual(overridden["mode"], "off")
        self.assertEqual(original["mode"], "all")
        with self.assertRaises(ValueError):
            _diagnostic_safety_override(original, options, train=True)

    def test_task2_wait_is_penalized_only_with_safe_objective_progress(self):
        from agent_code.cnn_distilled_double_dqn_agent.task2_shaping import (
            avoidable_task2_wait,
        )

        state = game_state(position=(3, 3))
        state["coins"] = []
        state["field"][5, 3] = 1
        safe = np.ones(6, dtype=bool)
        facts = avoidable_task2_wait(state, "WAIT", safe)
        self.assertTrue(facts["avoidable"])
        self.assertTrue(facts["safe_progress_move"])

        no_objective = game_state(position=(3, 3))
        no_objective["coins"] = []
        self.assertFalse(avoidable_task2_wait(
            no_objective, "WAIT", safe)["avoidable"])
        self.assertFalse(avoidable_task2_wait(
            state, "RIGHT", safe)["avoidable"])

    def test_task2_wait_detects_safe_useful_bomb(self):
        from agent_code.cnn_distilled_double_dqn_agent.task2_shaping import (
            avoidable_task2_wait,
        )

        state = game_state(position=(3, 3))
        state["coins"] = []
        state["field"][3, 4] = 1
        safe = np.zeros(6, dtype=bool)
        safe[4:] = True
        facts = avoidable_task2_wait(state, "WAIT", safe)
        self.assertTrue(facts["avoidable"])
        self.assertTrue(facts["safe_useful_bomb"])

    def test_four_step_return_flushes_terminal_prefixes(self):
        from agent_code.cnn_distilled_double_dqn_agent.learner import Transition
        from agent_code.learning_common.n_step import NStepAccumulator

        accumulator = NStepAccumulator(4, 0.95)
        state = np.zeros((17, 17, 17), dtype=np.float32)
        self.assertEqual(accumulator.append(Transition(
            state, 0, 1.0, state, False, np.ones(6, dtype=bool))), [])
        self.assertEqual(accumulator.append(Transition(
            state, 1, 2.0, state, False, np.ones(6, dtype=bool))), [])
        emitted = accumulator.append(Transition(
            state, 2, 3.0, None, True, None))

        self.assertEqual([item.steps for item in emitted], [3, 2, 1])
        self.assertTrue(all(item.done for item in emitted))
        self.assertAlmostEqual(emitted[0].reward, 1.0 + 0.95 * 2.0 + 0.95**2 * 3.0)
        self.assertAlmostEqual(emitted[1].reward, 2.0 + 0.95 * 3.0)
        self.assertEqual(emitted[2].reward, 3.0)

    def test_five_step_return_directly_includes_delayed_crate_reward(self):
        from agent_code.cnn_distilled_double_dqn_agent.learner import Transition
        from agent_code.learning_common.n_step import NStepAccumulator

        state = np.zeros((17, 17, 17), dtype=np.float32)
        legal = np.ones(6, dtype=bool)
        rewards = (0.6, -0.01, -0.01, -0.01, 0.2)
        four = NStepAccumulator(4, 0.95)
        five = NStepAccumulator(5, 0.95)
        emitted_four, emitted_five = [], []
        for reward in rewards:
            transition = Transition(state, 5, reward, state, False, legal)
            emitted_four.extend(four.append(transition))
            emitted_five.extend(five.append(transition))

        self.assertAlmostEqual(
            emitted_four[0].reward,
            0.6 - 0.01 * 0.95 - 0.01 * 0.95**2 - 0.01 * 0.95**3)
        self.assertAlmostEqual(
            emitted_five[0].reward,
            emitted_four[0].reward + 0.2 * 0.95**4)
        self.assertEqual(emitted_four[0].steps, 4)
        self.assertEqual(emitted_five[0].steps, 5)

    def test_task2_gate_requires_safety_retention_and_d01_crate_floor(self):
        from experiments.run_cnn_task2_pipeline import _gate

        task1 = {"mean_score": "49.0"}
        task2 = {
            "mean_coins": "6.0", "mean_crates": "80.0",
            "suicide_rate": "0.05", "zero_bomb_round_rate": "0.10",
            "survived_bomb_rate": "0.95", "invalid_action_rate": "0.01",
            "act_p95_time": "0.01", "act_max_time": "0.02",
        }
        self.assertTrue(_gate(task1, task2, 50.0, 79.6)["passed"])
        task2["suicide_rate"] = "0.051"
        self.assertFalse(_gate(task1, task2, 50.0, 79.5)["passed"])

    def test_task2_rank_compares_capability_after_the_safety_gate(self):
        from experiments.run_cnn_task2_pipeline import _rank

        common = {
            "mean_crates": "40.0", "crates_per_survived_bomb": "2.0",
            "wait_rate": "0.20", "long_wait_issue_rate": "0.0",
        }
        lower_coins = {
            "step": 100, "gate": {"passed": True, "task1_retention": 1.0},
            "task1": {}, "task2": {
                **common, "mean_coins": "3.0", "all_coins_rate": "0.2"},
        }
        higher_coins = {
            "step": 200, "gate": {"passed": True, "task1_retention": 1.0},
            "task1": {}, "task2": {
                **common, "mean_coins": "4.0", "all_coins_rate": "0.1"},
        }
        self.assertGreater(_rank(higher_coins), _rank(lower_coins))

    def test_task2_gate_does_not_regress_below_d01_crate_baseline(self):
        from experiments.run_cnn_task2_pipeline import _gate

        task1 = {"mean_score": "50.0"}
        task2 = {
            "mean_coins": "3.0", "mean_crates": "34.34",
            "suicide_rate": "0.0", "zero_bomb_round_rate": "0.0",
            "survived_bomb_rate": "1.0", "invalid_action_rate": "0.0",
            "act_p95_time": "0.01", "act_max_time": "0.02",
        }
        gate = _gate(task1, task2, 50.0, 0.0)
        self.assertFalse(gate["checks"]["task2_d01_crate_floor"])
        self.assertFalse(gate["passed"])

    def test_followup_skips_only_after_all_seeds_and_main_pass_strict_gate(self):
        from experiments.run_cnn_task2_followup import selection_qualified

        task1 = {"mean_score": "50.0"}
        task2 = {
            "mean_coins": "4.0", "mean_crates": "40.0",
            "suicide_rate": "0.0", "zero_bomb_round_rate": "0.0",
            "survived_bomb_rate": "1.0", "invalid_action_rate": "0.0",
            "act_p95_time": "0.01", "act_max_time": "0.02",
        }
        document = {
            "parent": {
                "task1": {"mean_score": "50.0"},
                "task2": {"mean_crates": "0.0"},
            },
            "transfer": {
                str(seed): {"best": {"task1": task1, "task2": dict(task2)}}
                for seed in (11, 22, 33)
            },
            "main_validation": {"task1": task1, "task2": dict(task2)},
        }
        self.assertTrue(selection_qualified(document))
        del document["transfer"]["33"]
        self.assertFalse(selection_qualified(document))

    def test_followup_rechecks_all_snapshots_with_the_current_gate(self):
        from experiments.run_cnn_task2_followup import selection_qualified

        task1 = {"mean_score": "50.0"}
        good_task2 = {
            "mean_coins": "4.0", "mean_crates": "40.0",
            "suicide_rate": "0.0", "zero_bomb_round_rate": "0.0",
            "survived_bomb_rate": "1.0", "invalid_action_rate": "0.0",
            "act_p95_time": "0.01", "act_max_time": "0.02",
        }
        bad_task2 = dict(good_task2, mean_crates="20.0")
        document = {
            "parent": {
                "task1": {"mean_score": "50.0"},
                "task2": {"mean_crates": "34.35"},
            },
            "transfer": {
                str(seed): {
                    "best": {"step": 200, "task1": task1, "task2": bad_task2},
                    "candidates": [
                        {"step": 100, "task1": task1, "task2": good_task2},
                        {"step": 200, "task1": task1, "task2": bad_task2},
                    ],
                }
                for seed in (11, 22, 33)
            },
            "main_validation": {"task1": task1, "task2": good_task2},
        }
        self.assertTrue(selection_qualified(document))

    def test_history_resets_before_first_observation_of_new_round(self):
        from agent_code.cnn_distilled_double_dqn_agent.features import (
            features_for_state, initialize_history, record_position,
        )
        owner = SimpleNamespace()
        initialize_history(owner)
        first = game_state(position=(1, 1))
        record_position(owner, first)
        same_round = features_for_state(owner, game_state(step=2, position=(1, 2)))
        self.assertEqual(same_round.board[15, 1, 1], 1.0)
        new_round = features_for_state(owner, game_state(round_index=2, position=(1, 1)))
        self.assertEqual(float(new_round.board[15].sum()), 0.0)
        self.assertEqual(tuple(np.argwhere(new_round.board[3] == 1)[0]), (1, 1))

    def test_frozen_checkpoint_loads_and_acts_without_teacher_dataset(self):
        from agent_code.cnn_distilled_double_dqn_agent import callbacks
        from agent_code.cnn_distilled_double_dqn_agent.learner import (
            DistilledDoubleDQNLearner,
        )
        from agent_code.cnn_distilled_double_dqn_agent.model import build_network
        from agent_code.learning_common.runtime import CHECKPOINT_SCHEMA
        from agent_code.team_agent.rewards import resolve_reward_spec
        from agent_code.team_agent.safety import resolve_safety_spec

        learner = DistilledDoubleDQNLearner(
            build_network("action_aligned"), build_network("action_aligned"),
            hyperparameters=callbacks.HYPERPARAMETERS, seed=3,
        )
        checkpoint = learner.checkpoint()
        checkpoint.update({
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "algorithm": callbacks.ALGORITHM, "actions": list(callbacks.ACTIONS),
            "feature_id": callbacks.FEATURE_ID,
            "feature_schema": callbacks.FEATURE_SCHEMA,
            "reward_id": "r5_conditional_loop",
            "reward_spec": resolve_reward_spec("r5_conditional_loop"),
            "hyperparameters": callbacks.HYPERPARAMETERS,
            "network_spec": callbacks.NETWORK_SPEC,
            "model_architecture": "action_aligned", "total_action_steps": 0,
            "stage_action_steps": 0, "agent_rng_state": random.Random(3).getstate(),
            "safety_spec": resolve_safety_spec({
                "version": "survival-mask-v1", "mode": "off", "horizon": 7,
                "fallback": "physical_q",
            }),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            torch.save(checkpoint, path)
            environment = {
                "BOMBERMAN_CHECKPOINT": str(path),
                "BOMBERMAN_FEATURE_ID": callbacks.FEATURE_ID,
                "BOMBERMAN_REWARD_ID": "r5_conditional_loop",
                "BOMBERMAN_ALLOW_BOMB": "false",
                "BOMBERMAN_AGENT_SEED": "3", "BOMBERMAN_TORCH_DEVICE": "cpu",
                "BOMBERMAN_SAFETY_SPEC": json.dumps({
                    "version": "survival-mask-v1", "mode": "off", "horizon": 7,
                    "fallback": "physical_q",
                }),
            }
            with patch.dict(os.environ, environment, clear=False):
                owner = SimpleNamespace(train=False, logger=logging.getLogger("distilled-test"))
                callbacks.setup(owner)
                action = callbacks.act(owner, game_state())
        self.assertIn(action, callbacks.ACTIONS[:5])
        self.assertIsNone(owner.model.teacher_dataset)


if __name__ == "__main__":
    unittest.main()
