import json
import csv
import tempfile
import unittest
from pathlib import Path


from experiments.analyze import analyze_runs, summarize_runs


def _episode(run_id, round_index, agents, round_steps=10):
    return {
        "schema_version": "episode-v1",
        "run_id": run_id,
        "round_index": round_index,
        "seed": 10001,
        "environment_seed": 10001,
        "scenario": "classic",
        "stage": "official_baseline",
        "round_steps": round_steps,
        "agents": agents,
    }


def _agent(name, score, survived=True, **metrics):
    return {
        "name": name,
        "score": score,
        "coins": metrics.get("coins", 0),
        "kills": metrics.get("kills", 0),
        "suicides": metrics.get("suicides", 0),
        "crates": metrics.get("crates", 0),
        "bombs": metrics.get("bombs", 0),
        "invalid": metrics.get("invalid", 0),
        "survived": survived,
        "dead": not survived,
    }


class ExperimentAnalysisTest(unittest.TestCase):
    def _write_run(self, root, run_id, episodes):
        directory = Path(root) / run_id
        directory.mkdir()
        (directory / "episodes.jsonl").write_text(
            "".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8"
        )
        return directory

    def test_metrics_and_ranking_categories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = self._write_run(
                temporary_directory,
                "ranking-run",
                [
                    _episode(
                        "ranking-run",
                        1,
                        [
                            _agent("a", 5, coins=2, kills=1, crates=3, invalid=4),
                            _agent("b", 3, survived=False),
                            _agent("c", 2, survived=False),
                            _agent("d", 0, survived=False),
                        ],
                    ),
                    _episode(
                        "ranking-run",
                        2,
                        [_agent("a", 5), _agent("b", 5), _agent("c", 0), _agent("d", 0)],
                    ),
                    _episode(
                        "ranking-run",
                        3,
                        [_agent("a", 0), _agent("b", 0), _agent("c", 0), _agent("d", 0)],
                    ),
                ],
            )

            rows = {row["agent_name"]: row for row in summarize_runs([run])}

            self.assertAlmostEqual(rows["a"]["mean_score"], 10 / 3)
            self.assertEqual(rows["a"]["total_score"], 10.0)
            self.assertEqual(rows["a"]["rank_by_total_score"], 1)
            self.assertAlmostEqual(rows["a"]["score_std"], 2.3570226039)
            self.assertEqual(rows["a"]["coins"], 2)
            self.assertAlmostEqual(rows["a"]["mean_coins"], 2 / 3)
            self.assertEqual(rows["a"]["kills"], 1)
            self.assertEqual(rows["a"]["crates"], 3)
            self.assertEqual(rows["a"]["invalid_actions"], 4)
            self.assertAlmostEqual(rows["a"]["invalid_action_rate"], 4 / 30)
            self.assertAlmostEqual(rows["a"]["survival_rate"], 1.0)
            self.assertAlmostEqual(rows["a"]["mean_survival_steps"], 10.0)
            self.assertAlmostEqual(rows["b"]["survival_rate"], 2 / 3)
            self.assertEqual(rows["a"]["exclusive_wins"], 1)
            self.assertAlmostEqual(rows["a"]["exclusive_win_rate"], 1 / 3)
            self.assertEqual(rows["a"]["tied_first"], 1)
            self.assertAlmostEqual(rows["a"]["tied_first_rate"], 1 / 3)
            self.assertEqual(rows["a"]["zero_score_ties"], 1)
            self.assertAlmostEqual(rows["a"]["zero_score_tie_rate"], 1 / 3)
            self.assertEqual(rows["b"]["exclusive_wins"], 0)
            self.assertEqual(rows["b"]["tied_first"], 1)
            self.assertEqual(rows["b"]["zero_score_ties"], 1)

    def test_multiple_runs_create_distinct_provenance_rows_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = self._write_run(
                temporary_directory, "first", [_episode("first", 1, [_agent("a", 1)])]
            )
            second = self._write_run(
                temporary_directory, "second", [_episode("second", 1, [_agent("a", 3)])]
            )
            output = Path(temporary_directory) / "analysis"

            rows = analyze_runs([first, second], output)

            self.assertEqual([(row["run_id"], row["mean_score"]) for row in rows], [("first", 1.0), ("second", 3.0)])
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "mean_score.png").is_file())
            with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(summary_rows[-1]["run_id"], "AVERAGE")
            self.assertEqual(summary_rows[-1]["agent_name"], "a")
            self.assertEqual(float(summary_rows[-1]["mean_score"]), 2.0)

    def test_task1_completion_and_loop_rates_are_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_agent = _agent("a", 50, coins=50)
            first_agent.update({
                "all_coins": True, "max_steps": False,
                "long_wait_loop": False, "long_ping_pong_loop": False,
            })
            second_agent = _agent("a", 12, coins=12)
            second_agent.update({
                "all_coins": False, "max_steps": True,
                "long_wait_loop": True, "long_ping_pong_loop": False,
            })
            episodes = [
                {**_episode("task1", 1, [first_agent], round_steps=120),
                 "exploration_disabled": True},
                {**_episode("task1", 2, [second_agent], round_steps=400),
                 "exploration_disabled": True},
            ]
            run = self._write_run(temporary_directory, "task1", episodes)
            row = summarize_runs([run])[0]
            self.assertEqual(row["mean_coins"], 31.0)
            self.assertEqual(row["all_coins_rate"], 0.5)
            self.assertEqual(row["max_steps_rate"], 0.5)
            self.assertEqual(row["long_wait_loop_rate"], 0.5)
            self.assertEqual(row["long_ping_pong_loop_rate"], 0.0)
            self.assertTrue(row["exploration_disabled"])

    def test_missing_or_malformed_episodes_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            missing.mkdir()
            with self.assertRaisesRegex(ValueError, "episodes.jsonl"):
                summarize_runs([missing])

            malformed = Path(temporary_directory) / "malformed"
            malformed.mkdir()
            (malformed / "episodes.jsonl").write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                summarize_runs([malformed])


if __name__ == "__main__":
    unittest.main()
