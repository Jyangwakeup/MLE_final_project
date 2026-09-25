import unittest

from experiments.navigation_diagnostics import navigation_diagnostic
from tests.test_danger import make_game_state


class NavigationDiagnosticTestCase(unittest.TestCase):
    def test_reports_progress_without_mutating_the_state(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3), (3, 5)]
        before = list(state["coins"])

        record, target = navigation_diagnostic(state, "RIGHT")

        self.assertEqual(target, (3, 5))
        self.assertEqual(record["nearest_coin_distance_before"], 2.0)
        self.assertEqual(record["predicted_nearest_coin_distance_after"], 1.0)
        self.assertTrue(record["distance_reduced"])
        self.assertTrue(record["multiple_nearest_coins"])
        self.assertFalse(record["previous_target_still_available"])
        self.assertEqual(state["coins"], before)

    def test_flags_immediate_reversal_and_only_live_target_switches(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3)]

        record, _ = navigation_diagnostic(
            state, "LEFT", previous_target=(4, 3), previous_action="RIGHT")

        self.assertTrue(record["immediate_reverse"])
        self.assertFalse(record["previous_target_still_available"])
        self.assertFalse(record["target_switched_while_previous_available"])

    def test_target_switch_denominator_requires_a_still_available_target(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3), (3, 4)]

        record, target = navigation_diagnostic(
            state, "DOWN", previous_target=(5, 3), previous_action="WAIT")

        self.assertEqual(target, (3, 4))
        self.assertTrue(record["previous_target_still_available"])
        self.assertTrue(record["target_switched_while_previous_available"])

    def test_reports_true_conditional_loop_and_wait_disposition(self):
        state = make_game_state(position=(3, 3))
        state["coins"] = [(5, 3)]
        loop, _ = navigation_diagnostic(
            state, "LEFT", previous_target=(5, 3), previous_action="RIGHT",
            previous_position=(2, 3), actual_position=(2, 3))
        self.assertTrue(loop["conditional_loop"])
        self.assertEqual(loop["wait_disposition"], "not_wait")

        waited, _ = navigation_diagnostic(
            state, "WAIT", previous_target=(5, 3),
            previous_position=(3, 3))
        self.assertFalse(waited["conditional_loop"])
        self.assertTrue(waited["avoidable_wait"])
        self.assertEqual(waited["wait_disposition"], "penalized")

    def test_reports_survivable_useful_crate_bomb_opportunity(self):
        state = make_game_state(position=(3, 3))
        state["field"][5, 3] = 1

        selected, _ = navigation_diagnostic(state, "BOMB")
        declined, _ = navigation_diagnostic(state, "WAIT")

        self.assertEqual(selected["crates_in_blast_if_bomb"], 1)
        self.assertTrue(selected["useful_crate_bomb_available"])
        self.assertTrue(selected["survivable_useful_crate_bomb_available"])
        self.assertTrue(selected["selected_useful_crate_bomb"])
        self.assertTrue(selected["selected_survivable_useful_crate_bomb"])
        self.assertFalse(declined["selected_useful_crate_bomb"])
        self.assertFalse(declined["selected_survivable_useful_crate_bomb"])


if __name__ == "__main__":
    unittest.main()
