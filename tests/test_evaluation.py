import json
import csv
import tempfile
import unittest
from pathlib import Path


from experiments.analyze import analyze_runs, summarize_runs
from experiments.compare_evaluations import compare_evaluations


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
        "bombs_resolved": metrics.get("bombs_resolved", 0),
        "bombs_survived": metrics.get("bombs_survived", 0),
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
            self.assertEqual(rows["a"]["min_score"], 0.0)
            self.assertEqual(rows["a"]["max_score"], 5.0)
            self.assertEqual(rows["a"]["zero_score_round_count"], 1)
            self.assertAlmostEqual(rows["a"]["zero_score_round_rate"], 1 / 3)
            self.assertEqual(rows["a"]["rank_by_total_score"], 1)
            self.assertAlmostEqual(rows["a"]["score_std"], 2.3570226039)
            self.assertEqual(rows["a"]["coins"], 2)
            self.assertAlmostEqual(rows["a"]["mean_coins"], 2 / 3)
            self.assertEqual(rows["a"]["kills"], 1)
            self.assertEqual(rows["a"]["kill_round_count"], 1)
            self.assertAlmostEqual(rows["a"]["kill_round_rate"], 1 / 3)
            self.assertEqual(rows["a"]["coin_score"], 2)
            self.assertAlmostEqual(rows["a"]["mean_coin_score"], 2 / 3)
            self.assertEqual(rows["a"]["kill_score"], 5)
            self.assertAlmostEqual(rows["a"]["mean_kill_score"], 5 / 3)
            self.assertEqual(rows["a"]["score_residual"], 3)
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
            self.assertAlmostEqual(rows["a"]["first_place_rate"], 2 / 3)
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

    def test_aggregate_p95_uses_all_decisions_not_mean_of_run_percentiles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = []
            for name, count, duration in (("fast", 90, .01), ("slow", 10, .45)):
                run = self._write_run(temporary_directory, name,
                    [_episode(name, 1, [_agent("a", 1), _agent("b", 0)])])
                records = [dict(schema_version="timing-v1", run_id=name, agent_name="a", round_index=1,
                    action="WAIT", think_time=duration, timed_out=False, skipped=False)
                    for _ in range(count)]
                records.append(dict(schema_version="timing-v1", run_id=name, agent_name="b", round_index=1,
                    action="WAIT", think_time=9., timed_out=True, skipped=False))
                (run / "timing.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
                runs.append(run)
            output = Path(temporary_directory) / "summary"
            analyze_runs(iter(runs), output)
            with (output / "summary.csv").open() as handle:
                row = next(r for r in csv.DictReader(handle)
                    if r["run_id"]=="AVERAGE" and r["agent_name"]=="a")
            self.assertAlmostEqual(float(row["act_p95_time"]), .45)
            self.assertGreater(float(row["act_p95_time"]), .25)
            self.assertAlmostEqual(float(row["act_mean_time"]), .054)
            self.assertAlmostEqual(float(row["act_max_time"]), .45)
            self.assertEqual(int(row["act_count"]), 100)

    def test_hundred_world_four_agent_chart_has_bounded_pixel_dimensions(self):
        from experiments.analyze import _write_mean_score_chart
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = [dict(run_id=f"world_{world}", agent_name=f"agent_{agent}",
                         mean_score=float(world % 10 + agent))
                    for world in range(100) for agent in range(4)]
            chart = _write_mean_score_chart(rows, Path(temporary_directory))
            with Image.open(chart) as image:
                self.assertLessEqual(image.width, 3600)
                self.assertLessEqual(image.height, 1000)

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
            self.assertEqual(row["min_coins_per_round"], 12.0)
            self.assertEqual(row["max_coins_per_round"], 50.0)
            self.assertEqual(row["zero_coin_round_count"], 0)
            self.assertEqual(row["zero_coin_round_rate"], 0.0)
            self.assertEqual(row["all_coins_rate"], 0.5)
            self.assertEqual(row["max_steps_rate"], 0.5)
            self.assertEqual(row["long_wait_loop_rate"], 0.5)
            self.assertEqual(row["long_ping_pong_loop_rate"], 0.0)
            self.assertTrue(row["exploration_disabled"])

    def test_bomb_efficiency_metrics_use_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = _agent("a", 0, crates=3, bombs=2)
            first["zero_utility_bombs"] = 1
            second = _agent("a", 0, crates=6, bombs=4)
            second["zero_utility_bombs"] = 0
            run = self._write_run(
                temporary_directory, "bomb-efficiency", [
                    _episode("bomb-efficiency", 1, [first]),
                    _episode("bomb-efficiency", 2, [second]),
                ])
            row = summarize_runs([run])[0]
            self.assertEqual(row["zero_utility_bombs"], 1)
            self.assertAlmostEqual(row["zero_utility_bomb_rate"], 1 / 6)
            self.assertAlmostEqual(row["crates_per_bomb"], 1.5)

    def test_zero_bomb_efficiency_metrics_are_null(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            agent = _agent("a", 0, crates=0, bombs=0)
            agent["zero_utility_bombs"] = 0
            run = self._write_run(
                temporary_directory, "zero-bombs",
                [_episode("zero-bombs", 1, [agent])])
            row = summarize_runs([run])[0]
            self.assertIsNone(row["zero_utility_bomb_rate"])
            self.assertIsNone(row["crates_per_bomb"])

    def test_coin_navigation_efficiency_and_action_diagnostics_are_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            agent = _agent("a", 2, coins=2)
            agent.update({
                "all_coins": False, "max_steps": False,
                "long_wait_loop": False, "long_ping_pong_loop": False,
            })
            run = self._write_run(
                temporary_directory,
                "navigation",
                [{**_episode("navigation", 1, [agent], round_steps=20),
                  "exploration_disabled": True}],
            )
            records = []
            for step, values in enumerate((
                (False, True, False, False),
                (True, False, True, True),
            ), 1):
                waited, reduced, reversed_action, switched = values
                records.append({
                    "schema_version": "timing-v1", "run_id": "navigation",
                    "round_index": 1, "step": step, "agent_name": "a",
                    "action": "WAIT" if waited else "RIGHT",
                    "requested_action": "WAIT" if waited else "RIGHT",
                    "think_time": 0.001, "skipped": False, "timed_out": False,
                    "available_before": 0.5, "available_after": 0.5,
                    "navigation": {
                        "coin_count_before": 2,
                        "nearest_coin_distance_before": 2.0,
                        "predicted_nearest_coin_distance_after": 1.0,
                        "distance_comparable": True,
                        "distance_reduced": reduced,
                        "multiple_nearest_coins": step == 1,
                        "selected_target": [5, 3],
                        "previous_target_still_available": True,
                        "target_switched_while_previous_available": switched,
                        "waited": waited,
                        "immediate_reverse": reversed_action,
                        "movement_legal_in_observed_state": True,
                    },
                })
            (run / "timing.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            row = summarize_runs([run])[0]

            self.assertEqual(row["coins_per_100_steps"], 10.0)
            self.assertEqual(row["steps_per_coin"], 10.0)
            self.assertEqual(row["total_round_steps"], 20.0)
            self.assertEqual(row["navigation_decisions"], 2)
            self.assertEqual(row["coin_distance_comparable_count"], 2)
            self.assertEqual(row["coin_target_observation_count"], 2)
            self.assertEqual(
                row["coin_target_continuity_opportunity_count"], 2)
            self.assertEqual(row["wait_action_rate"], 0.5)
            self.assertEqual(row["immediate_reverse_rate"], 0.5)
            self.assertEqual(row["coin_distance_reducing_rate"], 0.5)
            self.assertEqual(row["coin_target_switch_rate"], 0.5)
            self.assertEqual(row["multiple_nearest_coin_rate"], 0.5)
            self.assertEqual(row["right_action_count"], 1)
            self.assertEqual(row["wait_action_count"], 1)
            self.assertEqual(row["wait_action_rate_all_actions"], 0.5)
            self.assertEqual(row["median_longest_wait_streak"], 1)
            self.assertEqual(row["min_longest_wait_streak"], 1)
            self.assertEqual(row["max_longest_wait_streak"], 1)

    def test_distribution_metrics_are_written_for_multi_seed_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = []
            for index, score in enumerate((1, 3, 8), 1):
                run_id = f"seed-{index}"
                runs.append(self._write_run(
                    temporary_directory, run_id,
                    [_episode(run_id, 1, [_agent("a", score, coins=score)])],
                ))
            output = Path(temporary_directory) / "summary"
            analyze_runs(runs, output)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
                average = list(csv.DictReader(file))[-1]
            self.assertEqual(float(average["median_score"]), 3.0)
            self.assertLess(float(average["mean_score_ci95_low"]), 4.0)
            self.assertGreater(float(average["mean_score_ci95_high"]), 4.0)

    def test_bomb_metrics_are_aggregated_across_seed_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = []
            for index, (bombs, crates) in enumerate(((0, 0), (4, 6)), 1):
                run_id = f"bomb-seed-{index}"
                runs.append(self._write_run(
                    temporary_directory, run_id,
                    [_episode(run_id, 1, [_agent(
                        "a", 0, bombs=bombs, bombs_resolved=bombs,
                        bombs_survived=bombs, crates=crates)])],
                ))
            output = Path(temporary_directory) / "summary"
            analyze_runs(runs, output)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as file:
                average = list(csv.DictReader(file))[-1]
            self.assertEqual(float(average["mean_bombs"]), 2.0)
            self.assertEqual(float(average["zero_bomb_round_rate"]), 0.5)
            self.assertEqual(float(average["bombs_resolved"]), 4.0)
            self.assertEqual(float(average["bombs_survived"]), 4.0)
            self.assertEqual(float(average["survived_bomb_rate"]), 1.0)
            self.assertEqual(float(average["crates_per_survived_bomb"]), 1.5)

    def test_paired_evaluation_comparison_matches_environment_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate = self._write_run(
                root, "candidate", [_episode("candidate", 1, [_agent("a", 5, coins=5)])])
            reference = self._write_run(
                root, "reference", [_episode("reference", 1, [_agent("b", 3, coins=3)])])
            for directory, agent in ((candidate, "a"), (reference, "b")):
                (directory / "metadata.json").write_text(json.dumps({
                    "task": "coin_navigation", "agent": agent,
                }), encoding="utf-8")
            report = compare_evaluations(
                [candidate], [reference], root / "comparison", bootstrap_samples=100)
            score = next(row for row in report["metrics"] if row["metric"] == "score")
            self.assertEqual(score["mean_difference"], 2.0)
            self.assertEqual(score["candidate_wins"], 1)
            self.assertEqual(report["replication_unit"], "environment_seed")
            self.assertTrue((root / "comparison" / "paired_comparison.csv").is_file())

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
