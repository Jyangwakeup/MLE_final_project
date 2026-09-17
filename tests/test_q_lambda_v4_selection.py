"""Tests for the preregistered v4 candidate ordering."""

import unittest

from experiments.select_q_lambda_v4_ablation import _eligible, _rank


def candidate(*, coins=3.0, crates=40.0, wait=0.2, ping=0.1,
              suicide=0.0, survival=1.0, retention=1.0, gates=10):
    checks = {f"gate-{index}": index < gates for index in range(19)}
    return {
        "checkpoint": "candidate.pkl", "gates": checks,
        "task1_retention": retention,
        "task2": {
            "mean_coins": coins, "mean_crates": crates,
            "long_wait_loop_rate": wait,
            "long_ping_pong_loop_rate": ping,
            "suicide_rate": suicide, "survived_bomb_rate": survival,
        },
    }


class QLambdaV4SelectionTests(unittest.TestCase):
    def test_safety_floor_is_mandatory(self):
        self.assertTrue(_eligible(candidate()))
        self.assertFalse(_eligible(candidate(suicide=0.1)))
        self.assertFalse(_eligible(candidate(survival=0.9)))
        self.assertFalse(_eligible(candidate(retention=0.89)))

    def test_gate_count_then_balance_then_loops(self):
        stronger = candidate(gates=11, coins=2.0)
        self.assertLess(_rank(stronger), _rank(candidate(gates=10, coins=6.0)))
        balanced = candidate(gates=11, coins=4.0, crates=50.0)
        self.assertLess(_rank(balanced), _rank(stronger))
        low_loop = candidate(gates=11, coins=4.0, crates=50.0, wait=0.0, ping=0.0)
        self.assertLess(_rank(low_loop), _rank(balanced))


if __name__ == "__main__":
    unittest.main()
