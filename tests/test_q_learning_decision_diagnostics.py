import json
from pathlib import Path
import tempfile
import unittest

from experiments.summarize_q_learning_decisions import summarize


class QLearningDecisionSummaryTests(unittest.TestCase):
    def test_reports_wait_ties_and_mask_effect_without_mutating_inputs(self):
        rows = [
            {"action": "WAIT", "in_current_danger": False,
             "visible_coin_count": 1, "reachable_coin_exists": True,
             "reachable_crate_frontier_exists": True, "raw_argmax_vetoed": False,
             "raw_argmax_ties": [4], "legal_argmax_ties": [4, 5],
             "physical_mask": [True, True, False, False, True, True],
             "legal_mask": [True, True, False, False, True, False],
             "physical_useful_bomb": True},
            {"action": "BOMB", "in_current_danger": True,
             "visible_coin_count": 0, "reachable_coin_exists": False,
             "reachable_crate_frontier_exists": True, "raw_argmax_vetoed": True,
             "raw_argmax_ties": [5], "legal_argmax_ties": [4],
             "physical_mask": [True, False, False, False, True, True],
             "legal_mask": [True, False, False, False, True, False],
             "physical_useful_bomb": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = summarize([path])
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["action_counts"]["WAIT"], 1)
        self.assertEqual(result["raw_bomb_veto_count"], 1)
        self.assertEqual(result["safe_bomb_count"], 0)
        self.assertEqual(result["legal_argmax_tie_rate"], 0.5)
