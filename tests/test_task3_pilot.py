import copy
import json
from pathlib import Path
import unittest

from experiments.task3_pilot import assess, load_preregistration, select_candidate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "task3_pilot.json"


def rows(role, *, score, coins, crates=0, kills=0, first=0, bombs=10):
    task = {
        "task1": "coin_navigation",
        "task2": "crate_navigation",
        "task3": "weak_opponents",
    }[role.split("_")[-1]]
    result = []
    for index, environment_seed in enumerate(range(12000, 12020)):
        result.append({
            "training_seed": 11,
            "cumulative_task3_rounds": 500,
            "role": role,
            "task": task,
            "environment_seed": environment_seed,
            "run_id": f"{role}_{environment_seed}",
            "checkpoint": "checkpoint.pt",
            "source_commit": "814173b618fb38990db7a5b7ee200369c8b25289",
            "source_hash": "b3dafd35f05214bac1058d7e16407e8cb113322a10c590efb748a51eb6e9364e",
            "status": "completed",
            "score": score,
            "coins": coins,
            "crates": crates,
            "kills": kills,
            "suicides": 0,
            "bombs": bombs,
            "bombs_resolved": bombs,
            "bombs_survived": bombs,
            "invalid_actions": 0,
            "survived": 1,
            "exclusive_first": float(index < int(first * 20)),
            "tied_first": 0,
            "act_count": 100,
            "act_p95_seconds": 0.01,
            "act_max_seconds": 0.02,
            "timeouts": 0,
            "skipped_actions": 0,
        })
    return result


def passing_evidence():
    return {
        "parent_task1": rows("parent_task1", score=50, coins=50, bombs=0),
        "parent_task2": rows("parent_task2", score=6, coins=6, crates=90),
        "parent_task3": rows(
            "parent_task3", score=2, coins=2, crates=80, kills=0, first=.2),
        "child_task1": rows("child_task1", score=50, coins=50, bombs=0),
        "child_task2": rows("child_task2", score=6, coins=6, crates=90),
        "child_task3": rows(
            "child_task3", score=3, coins=2, crates=80, kills=.2, first=.3),
    }


class Task3PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.config = load_preregistration(MANIFEST)

    def training_run(self, seed=11, score=3):
        return {
            "run": f"runs/task3_seed_{seed}_{score}",
            "checkpoint": f"runs/task3_seed_{seed}_{score}/checkpoints/final.pt",
            "checkpoint_sha256": "a" * 64,
            "latest_generation": "generation-00000500",
            "latest_generation_hash": "b" * 64,
            "updates": 10,
            "loss": 1.0,
            "local_rounds": 500,
        }

    def test_preregistration_keeps_the_winner_contract_and_no_action_stop(self):
        self.assertEqual(self.config["algorithm"], "double_dqn")
        self.assertEqual(self.config["feature_id"], "continuous-v2")
        self.assertEqual(self.config["reward_id"], "r7_safe_credit_sparse")
        self.assertEqual(self.config["safety"]["mode"], "all")
        self.assertIsNone(
            self.config["training"]["target_stage_action_steps"])
        self.assertNotIn("performance_stopping", self.config["training"])
        self.assertEqual(
            self.config["evaluation"]["seeds"], list(range(12000, 12020)))
        self.assertFalse(self.manifest["outcome"]["qualified_for_task4"])

    def test_one_assessment_passes_all_resource_combat_and_safety_gates(self):
        result = assess(
            self.manifest, passing_evidence(), self.training_run(),
            training_seed=11, cumulative_rounds=500)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertAlmostEqual(
            result["paired"]["task3_score"]["mean_difference"], 1.0)
        self.assertEqual(
            result["paired"]["task3_score"]["bootstrap_ci95"], [1.0, 1.0])

    def test_score_gain_does_not_hide_missing_combat_gain(self):
        evidence = passing_evidence()
        for row in evidence["child_task3"]:
            row["kills"] = 0
            row["exclusive_first"] = evidence["parent_task3"][
                row["environment_seed"] - 12000]["exclusive_first"]
        result = assess(
            self.manifest, evidence, self.training_run(),
            training_seed=11, cumulative_rounds=500)
        self.assertFalse(result["checks"]["task3_combat_gain"])
        self.assertFalse(result["passed"])

    def test_score_gain_does_not_hide_crate_regression(self):
        evidence = passing_evidence()
        for row in evidence["child_task3"]:
            row["crates"] = 70
        result = assess(
            self.manifest, evidence, self.training_run(),
            training_seed=11, cumulative_rounds=500)
        self.assertFalse(result["checks"]["task3_crate_retention"])
        self.assertFalse(result["passed"])

    def test_parent_and_child_must_use_the_same_environment_seeds(self):
        evidence = passing_evidence()
        evidence["child_task3"][0]["environment_seed"] = 13000
        with self.assertRaisesRegex(ValueError, "seeds do not match"):
            assess(
                self.manifest, evidence, self.training_run(),
                training_seed=11, cumulative_rounds=500)

    def test_candidate_requires_all_three_seeds_and_ranks_score_first(self):
        assessments = []
        for seed, score in ((11, 3), (22, 4), (33, 3)):
            evidence = copy.deepcopy(passing_evidence())
            for role_rows in evidence.values():
                for row in role_rows:
                    row["training_seed"] = seed
            for row in evidence["child_task3"]:
                row["score"] = score
            item = assess(
                self.manifest, evidence, self.training_run(seed, score),
                training_seed=seed, cumulative_rounds=500)
            assessments.append(item)
        result = select_candidate(self.manifest, assessments)
        self.assertTrue(result["all_training_seeds_passed"])
        self.assertEqual(result["checkpoint_rank"], [22, 11, 33])
        self.assertEqual(result["selected"]["training_seed"], 22)
        self.assertFalse(result["qualified_for_task4"])

        failed = copy.deepcopy(assessments)
        failed[-1]["passed"] = False
        rejected = select_candidate(self.manifest, failed)
        self.assertFalse(rejected["all_training_seeds_passed"])
        self.assertIsNone(rejected["selected"])


if __name__ == "__main__":
    unittest.main()
