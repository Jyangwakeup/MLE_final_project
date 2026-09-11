import unittest

from environment import format_results_table


class ResultsTableTestCase(unittest.TestCase):
    def test_formats_agent_totals_and_average_score(self):
        results = {
            "by_agent": {
                "q_learning_agent": {
                    "rounds": 10,
                    "score": 17,
                    "coins": 12,
                    "crates": 8,
                    "kills": 1,
                    "suicides": 2,
                    "invalid": 3,
                },
            },
        }

        table = format_results_table(results)

        self.assertIn("q_learning_agent", table)
        self.assertIn("Avg score", table)
        self.assertIn("1.70", table)
        self.assertIn("Suicides", table)

    def test_zero_rounds_do_not_divide_by_zero(self):
        table = format_results_table({"by_agent": {"agent": {"score": 0}}})

        self.assertIn("0.00", table)


if __name__ == "__main__":
    unittest.main()
