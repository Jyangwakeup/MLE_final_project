import copy
import unittest

import numpy as np

import settings as s
from agent_code.team_agent.feature_system import (
    ACTIONS,
    available_feature_ids,
    extract_features,
    get_feature_schema,
    normalize_feature_id,
    feature_schema_contract,
    validate_checkpoint_feature_contract,
)
from agent_code.team_agent.feature_system.discrete_compact_v1 import (
    _transform_game_state,
)
from agent_code.team_agent.feature_system.symmetry import symmetries
from agent_code.team_agent.temporal_safety_features import (
    ACTIONS as TEMPORAL_ACTIONS,
    detailed_reachability_after_first_step,
    detailed_reachability_all_first_steps,
    temporal_maps,
)
from tests.test_danger import CRATE, WALL, make_game_state


def assert_state_unchanged(test_case, actual, expected):
    test_case.assertEqual(actual.keys(), expected.keys())
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            test_case.assertEqual(actual[key], expected[key])


class RegistryAndBaselineTestCase(unittest.TestCase):
    def test_registry_has_stable_semantic_ids_and_legacy_alias(self):
        self.assertEqual(available_feature_ids(), (
            "discrete-v1", "discrete-q-v2", "discrete-objective-v1",
            "discrete-compact-v1", "continuous-v1", "continuous-v2",
            "continuous-v2-legacy78", "continuous-v3", "continuous-phase-v1",
            "continuous-v4", "continuous-v5",
            "board-v1", "hybrid-v1",
        ))
        self.assertEqual(normalize_feature_id(legacy_version="v1"), "discrete-v1")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            normalize_feature_id("continuous-v1", "v1")
        with self.assertRaisesRegex(ValueError, "unknown"):
            normalize_feature_id("unknown")

    def test_discrete_v1_is_frozen(self):
        result = extract_features(make_game_state(), "discrete-v1")
        schema = get_feature_schema("discrete-v1")
        self.assertEqual(len(result.state_key), 14)
        self.assertEqual(result.vector.shape, (40,))
        self.assertEqual(result.vector.dtype, np.float32)
        self.assertEqual(schema.category_counts, (
            3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 5, 4, 2,
        ))
        self.assertEqual(schema.theoretical_state_count, 1_399_680)

    def test_discrete_q_v2_adds_distance_and_previous_move(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(8, 3)]
        from agent_code.team_agent.feature_system.discrete_q_v2 import extract
        result = extract(state, previous_action="LEFT")
        self.assertEqual(len(result.state_key), 16)
        self.assertEqual(result.state_key[-2:], (2, 4))
        self.assertEqual(result.vector.shape, (50,))
        self.assertEqual(result.vector.dtype, np.float32)

    def test_checkpoint_contract_accepts_explicit_legacy_and_rejects_mismatch(self):
        validate_checkpoint_feature_contract(
            {"feature_version": "v1"}, "discrete-v1", ACTIONS)
        current = {
            "feature_id": "discrete-v1",
            "feature_version": "v1",
            "feature_schema": feature_schema_contract("discrete-v1"),
            "actions": list(ACTIONS),
        }
        validate_checkpoint_feature_contract(current, "discrete-v1", ACTIONS)
        with self.assertRaisesRegex(ValueError, "feature schema"):
            validate_checkpoint_feature_contract(
                {**current, "feature_schema": {"vector_shape": [41]}},
                "discrete-v1", ACTIONS)
        with self.assertRaisesRegex(ValueError, "action order"):
            validate_checkpoint_feature_contract(
                {**current, "actions": list(reversed(ACTIONS))},
                "discrete-v1", ACTIONS)
        with self.assertRaisesRegex(ValueError, "no feature"):
            validate_checkpoint_feature_contract({}, "discrete-v1", ACTIONS)

    def test_every_extractor_is_deterministic_and_non_mutating(self):
        state = make_game_state(bombs=[((3, 5), 1)])
        state["coins"] = [(5, 3)]
        state["others"] = [("other", 0, True, (7, 3))]
        original = copy.deepcopy(state)
        for feature_id in available_feature_ids():
            with self.subTest(feature_id=feature_id):
                first = extract_features(state, feature_id)
                second = extract_features(state, feature_id)
                if hasattr(first, "state_key"):
                    self.assertEqual(first.state_key, second.state_key)
                if hasattr(first, "vector"):
                    np.testing.assert_array_equal(first.vector, second.vector)
                if hasattr(first, "board"):
                    np.testing.assert_array_equal(first.board, second.board)
                np.testing.assert_array_equal(first.legal_mask, second.legal_mask)
                assert_state_unchanged(self, state, original)


class ExactDangerFeatureTestCase(unittest.TestCase):
    def _continuous_wait(self, timer):
        state = make_game_state(bombs=[((3, 5), timer)])
        return extract_features(state, "continuous-v1").vector[40:50]

    def test_timer_zero_one_two_map_to_exact_future_steps(self):
        timer_zero = self._continuous_wait(0)
        timer_one = self._continuous_wait(1)
        timer_two = self._continuous_wait(2)
        np.testing.assert_array_equal(timer_zero[1:4], [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(timer_one[1:4], [0.0, 1.0, 1.0])
        np.testing.assert_array_equal(timer_two[1:4], [0.0, 0.0, 1.0])

    def test_action_destination_can_be_safe_then_dangerous(self):
        state = make_game_state(bombs=[((4, 5), 1)])
        right = extract_features(state, "continuous-v1").vector[10:20]
        self.assertEqual(tuple(right[1:4]), (0.0, 1.0, 1.0))
        self.assertEqual(right[0], 1.0)

    def test_current_explosion_and_multiple_bombs_are_unioned(self):
        explosion = np.zeros((s.COLS, s.ROWS), dtype=int)
        explosion[3, 3] = 1
        state = make_game_state(
            bombs=[((3, 5), 1), ((5, 3), 2)], explosion_map=explosion)
        board = extract_features(state, "board-v1").board
        self.assertEqual(board[7, 3, 3], 1.0)
        self.assertEqual(board[8, 3, 3], 1.0)
        self.assertEqual(board[9, 3, 3], 1.0)
        self.assertEqual(board[10, 3, 3], 1.0)
        self.assertEqual(board[11, 3, 3], 1.0)

    def test_bomb_action_uses_hypothetical_bomb_danger(self):
        vector = extract_features(make_game_state(), "continuous-v1").vector
        wait = vector[40:50]
        bomb = vector[50:60]
        self.assertEqual(wait[4], 1.0)
        self.assertAlmostEqual(bomb[4], 4.0 / 7.0)
        self.assertEqual(tuple(bomb[1:4]), (0.0, 0.0, 0.0))

    def test_batched_reachability_matches_scalar_search(self):
        state = make_game_state(bombs=[((3, 5), 1), ((5, 3), 2)])
        danger, blocked = temporal_maps(state, 7, hypothetical_bomb=False)
        position = state["self"][3]
        batched = detailed_reachability_all_first_steps(
            position, danger, blocked)
        for action in TEMPORAL_ACTIONS:
            self.assertEqual(
                batched[action],
                detailed_reachability_after_first_step(
                    position, action, danger, blocked),
            )


class ContinuousFeatureTestCase(unittest.TestCase):
    def test_shape_dtype_ranges_and_global_availability(self):
        state = make_game_state(bombs_left=False)
        result = extract_features(state, "continuous-v1")
        self.assertEqual(result.vector.shape, (70,))
        self.assertEqual(result.vector.dtype, np.float32)
        self.assertTrue(np.all(result.vector >= -1.0))
        self.assertTrue(np.all(result.vector <= 1.0))
        self.assertEqual(tuple(result.vector[60:68]), (0.0,) * 8)
        self.assertEqual(result.vector[68], 0.0)
        self.assertEqual(result.vector[69], 0.0)

    def test_unreachable_coin_has_zero_delta_and_false_flag(self):
        state = make_game_state()
        state["coins"] = [(6, 6)]
        for position in ((5, 6), (7, 6), (6, 5), (6, 7)):
            state["field"][position] = WALL
        vector = extract_features(state, "continuous-v1").vector
        self.assertEqual(vector[60], 0.0)
        for offset in range(0, 60, 10):
            self.assertEqual(vector[offset + 7], 0.0)

    def test_dead_end_bomb_cannot_survive_but_retains_last_safe_area(self):
        state = make_game_state()
        for position in ((2, 3), (4, 3), (3, 2), (3, 4)):
            state["field"][position] = WALL
        vector = extract_features(state, "continuous-v1").vector
        self.assertEqual(vector[63], 1.0)
        self.assertEqual(vector[64], 0.0)
        self.assertGreater(vector[65], 0.0)

    def test_legal_mask_does_not_filter_danger(self):
        state = make_game_state(bombs=[((3, 5), 0)])
        result = extract_features(state, "continuous-v1")
        self.assertTrue(result.legal_mask[ACTIONS.index("WAIT")])
        self.assertEqual(result.vector[41], 1.0)

    def test_round_progress_clips_at_endpoints(self):
        start = make_game_state()
        end = make_game_state()
        end["step"] = s.MAX_STEPS + 10
        self.assertEqual(extract_features(start, "continuous-v1").vector[69], 0.0)
        self.assertEqual(extract_features(end, "continuous-v1").vector[69], 1.0)


class CompactFeatureTestCase(unittest.TestCase):
    def test_vectorized_symmetry_matches_coordinate_definition(self):
        for shape in ((7, 9), (9, 9)):
            source = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
            for symmetry in symmetries(*shape):
                expected = np.empty_like(source)
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        xx, yy = symmetry.coordinate((x, y))
                        expected[xx, yy] = source[x, y]
                actual = symmetry.transform_array(source)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(actual.flags.c_contiguous)
                self.assertFalse(np.shares_memory(actual, source))

    def test_schema_key_and_one_hot_contract(self):
        result = extract_features(make_game_state(), "discrete-compact-v1")
        schema = get_feature_schema("discrete-compact-v1")
        self.assertEqual(schema.category_counts, (3, 3, 3, 3, 2, 2, 2, 3, 4, 5, 4, 4))
        self.assertEqual(schema.theoretical_state_count, 622_080)
        self.assertEqual(len(result.state_key), 12)
        self.assertEqual(result.vector.shape, (38,))
        self.assertEqual(result.vector.dtype, np.float32)
        self.assertEqual(float(result.vector.sum()), 12.0)
        for value, count in zip(result.state_key, schema.category_counts):
            self.assertIn(value, range(count))

    def test_current_tile_has_exact_t1_t2_t3_bits(self):
        state = make_game_state(bombs=[((3, 5), 1)])
        result = extract_features(state, "discrete-compact-v1")
        self.assertEqual(result.state_key[4:7], (0, 1, 1))

    def test_square_rotation_preserves_canonical_key(self):
        state = make_game_state(bombs=[((3, 5), 1)], position=(3, 3))
        state["coins"] = [(6, 4)]
        state["field"][4, 6] = CRATE
        rotation = next(item for item in symmetries(*state["field"].shape)
                        if item.name == "rotate_90")
        rotated = _transform_game_state(state, rotation)
        first = extract_features(state, "discrete-compact-v1")
        second = extract_features(rotated, "discrete-compact-v1")
        self.assertEqual(first.state_key, second.state_key)
        for world, canonical in enumerate(first.action_transform.world_to_canonical):
            self.assertEqual(first.action_transform.canonical_to_world[canonical], world)

    def test_non_square_board_uses_shape_preserving_symmetries(self):
        field = np.zeros((7, 9), dtype=int)
        field[0, :] = field[-1, :] = WALL
        field[:, 0] = field[:, -1] = WALL
        state = make_game_state(field=field, position=(3, 4))
        result = extract_features(state, "discrete-compact-v1")
        self.assertEqual(result.vector.shape, (38,))
        self.assertEqual(
            tuple(item.name for item in symmetries(7, 9)),
            ("identity", "rotate_180", "mirror_x", "mirror_y"),
        )


class BoardAndHybridFeatureTestCase(unittest.TestCase):
    def test_board_channels_keep_field_x_y_orientation(self):
        field = np.zeros((7, 9), dtype=int)
        field[0, :] = field[-1, :] = WALL
        field[:, 0] = field[:, -1] = WALL
        field[2, 5] = CRATE
        state = make_game_state(field=field, position=(3, 4), bombs=[((4, 6), 2)])
        state["coins"] = [(5, 2)]
        state["others"] = [("other", 0, True, (1, 6))]
        board = extract_features(state, "board-v1").board
        self.assertEqual(board.shape, (12, 7, 9))
        self.assertEqual(board.dtype, np.float32)
        self.assertEqual(board[1, 2, 5], 1.0)
        self.assertEqual(board[1, 5, 2], 0.0)
        self.assertEqual(board[2, 5, 2], 1.0)
        self.assertEqual(board[4, 1, 6], 1.0)
        self.assertEqual(board[5, 4, 6], 1.0)
        self.assertAlmostEqual(board[6, 4, 6], 2.0 / s.BOMB_TIMER)

    def test_hybrid_is_identical_to_individual_extractors(self):
        state = make_game_state(bombs=[((3, 5), 1)])
        state["coins"] = [(6, 3)]
        board = extract_features(state, "board-v1")
        vector = extract_features(state, "continuous-v1")
        hybrid = extract_features(state, "hybrid-v1")
        np.testing.assert_array_equal(hybrid.board, board.board)
        np.testing.assert_array_equal(hybrid.vector, vector.vector)
        np.testing.assert_array_equal(hybrid.legal_mask, board.legal_mask)
        np.testing.assert_array_equal(hybrid.legal_mask, vector.legal_mask)


if __name__ == "__main__":
    unittest.main()
