"""State-machine tests for the preregistered Task 2 campaign."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.run_q_lambda_v4_task2_campaign import advance, start


class QLambdaV4CampaignTests(unittest.TestCase):
    def setUp(self):
        # Exercise the registered campaign before its deadline, independent of
        # the calendar date when the regression suite is run.
        clock_patch = patch(
            "experiments.run_q_lambda_v4_task2_campaign.datetime", wraps=datetime)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.clock.now.return_value = datetime(2026, 9, 19, 10, tzinfo=timezone.utc)

    def test_deadline_boundary_is_inclusive(self):
        from experiments.run_q_lambda_v4_task2_campaign import _deadline_reached
        state = {"deadline": "2026-09-20T12:00:00+02:00"}
        self.assertFalse(_deadline_reached(state))
        self.clock.now.return_value = datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
        self.assertTrue(_deadline_reached(state))

    def test_start_submits_parallel_screens_once(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state_path = root / "campaign.json"
            jobs = iter(("101", "102"))
            with patch(
                "experiments.run_q_lambda_v4_task2_campaign._submit_screen",
                side_effect=lambda *_args, **_kwargs: next(jobs),
            ), patch(
                "experiments.run_q_lambda_v4_task2_campaign._schedule_controller",
                return_value="103",
            ):
                state = start(state_path, root / "r7", root / "r10")
                again = start(state_path, root / "other7", root / "other10")
            self.assertEqual(state["screen_jobs"], {"r7": "101", "r10": "102"})
            self.assertEqual(state["controller_job"], "103")
            self.assertEqual(again["screen_jobs"], state["screen_jobs"])

    def test_no_safe_screen_candidate_submits_r12_parent(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "campaign.json"
            state = {
                "schema_version": "qlambda-v4-task2-campaign-v1",
                "status": "running", "phase": "screens",
                "deadline": "2026-09-20T12:00:00+02:00",
                "screen_jobs": {"r7": "101", "r10": "102"},
                "parents": {"r7": "/r7", "r10": "/r10"},
                "jobs": [], "max_infrastructure_retries": 1,
            }
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with patch(
                "experiments.run_q_lambda_v4_task2_campaign._screen_run",
                side_effect=lambda label, *_: Path(raw) / label,
            ), patch("pathlib.Path.is_file", return_value=True), patch(
                "experiments.run_q_lambda_v4_task2_campaign._run_ablation",
                return_value={"winner": None},
            ), patch(
                "experiments.run_q_lambda_v4_task2_campaign._submit",
                return_value="200",
            ), patch(
                "experiments.run_q_lambda_v4_task2_campaign._schedule_controller",
                return_value="201",
            ):
                result = advance(state_path)
            self.assertEqual(result["phase"], "r12_parent")
            self.assertEqual(result["r12_parent_job"], "200")

    def test_deadline_stops_new_submissions(self):
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "campaign.json"
            state_path.write_text(json.dumps({
                "status": "running", "phase": "screens",
                "deadline": "2020-01-01T00:00:00+00:00",
            }), encoding="utf-8")
            result = advance(state_path)
            self.assertEqual(result["status"], "deadline_reached")


if __name__ == "__main__":
    unittest.main()
