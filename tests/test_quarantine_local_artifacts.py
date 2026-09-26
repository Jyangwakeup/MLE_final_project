import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import quarantine_local_artifacts as q


class QuarantineLocalArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.root), "config", "user.name", "Test"],
            check=True,
        )
        (self.root / "README.md").write_text("test\n")
        subprocess.run(["git", "-C", str(self.root), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "commit", "-qm", "initial"], check=True
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_output(self, content=b"artifact"):
        output = self.root / "output"
        output.mkdir()
        (output / "result.bin").write_bytes(content)
        return output

    def quarantine(self, targets=("output",), **kwargs):
        return q.quarantine(
            self.root,
            targets,
            operation=kwargs.pop("operation", "test-operation"),
            activity_checker=kwargs.pop("activity_checker", lambda paths: []),
            **kwargs,
        )

    def test_dry_run_does_not_create_or_move_anything(self):
        output = self.make_output()
        result = self.quarantine(execute=False)
        self.assertTrue(output.exists())
        self.assertFalse(q.local_root(self.root).exists())
        self.assertEqual(result["state"], "planned")

    def test_only_exact_whitelisted_targets_are_accepted(self):
        for target in ("../output", "runs", ".scratch/branch-integration/extra"):
            with self.assertRaises(ValueError):
                q.validate_target(self.root, target)
        self.assertEqual(q.validate_target(self.root, "output"), self.root / "output")

    def test_symlinked_store_and_parent_escape_are_refused(self):
        outside = self.root.parent / f"{self.root.name}-outside"
        outside.mkdir()
        try:
            (self.root / ".local-artifacts").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "may not be a symlink"):
                self.quarantine(execute=False)
            (self.root / ".local-artifacts").unlink()
            (self.root / ".scratch").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                q.validate_target(self.root, ".scratch/branch-integration")
        finally:
            if (self.root / ".local-artifacts").is_symlink():
                (self.root / ".local-artifacts").unlink()
            if (self.root / ".scratch").is_symlink():
                (self.root / ".scratch").unlink()
            outside.rmdir()

    def test_tracked_content_is_refused(self):
        self.make_output()
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "output"], check=True)
        with self.assertRaisesRegex(ValueError, "tracked"):
            self.quarantine(execute=False)

    def test_existing_destination_is_never_overwritten(self):
        self.make_output()
        destination = q.quarantine_root(self.root) / "test-operation" / "output"
        destination.mkdir(parents=True)
        (destination / "existing").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.quarantine(execute=True)
        self.assertEqual((destination / "existing").read_text(), "keep")

    def test_source_change_is_detected_before_move(self):
        output = self.make_output()
        plan = q.build_quarantine_plan(
            self.root, ["output"], "test-operation"
        )
        (output / "result.bin").write_bytes(b"changed")
        current = q.scan_tree(output)
        self.assertNotEqual(
            current["content_sha256"], plan["entries"][0]["snapshot"]["content_sha256"]
        )

    def test_activity_blocks_execution_without_writing_manifest(self):
        self.make_output()
        with self.assertRaisesRegex(RuntimeError, "in use"):
            self.quarantine(
                execute=True,
                activity_checker=lambda paths: [{"pid": 123, "paths": ["output"]}],
            )
        self.assertFalse(q.local_root(self.root).exists())
        self.assertTrue((self.root / "output").exists())

    def test_symlink_is_recorded_lexically_and_restored(self):
        output = self.make_output()
        (output / "relative-link").symlink_to("result.bin")
        result = self.quarantine(execute=True)
        manifest = self.root / result["manifest"]
        records = result["entries"][0]["snapshot"]["records"]
        link = next(row for row in records if row["kind"] == "symlink")
        self.assertEqual(link["link_text"], "result.bin")
        preview = q.restore(self.root, manifest, execute=False)
        self.assertEqual(preview["targets"], ["output"])
        q.restore(self.root, manifest, execute=True, activity_checker=lambda paths: [])
        self.assertEqual(os.readlink(output / "relative-link"), "result.bin")

    def test_tampering_blocks_restore(self):
        self.make_output()
        result = self.quarantine(execute=True)
        manifest = self.root / result["manifest"]
        quarantined = self.root / result["entries"][0]["destination"] / "result.bin"
        quarantined.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            q.restore(self.root, manifest, execute=False)

    def test_partial_move_is_recorded_and_can_be_restored(self):
        self.make_output()
        scratch = self.root / ".scratch" / "archive-consolidation"
        scratch.mkdir(parents=True)
        (scratch / "report.log").write_text("complete")
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interruption")
            source.rename(destination)

        with self.assertRaisesRegex(OSError, "simulated"):
            self.quarantine(
                targets=("output", ".scratch/archive-consolidation"),
                execute=True,
                rename=fail_second,
            )
        manifest = q.manifests_root(self.root) / "test-operation.json"
        value = json.loads(manifest.read_text())
        self.assertEqual(value["state"], "partial")
        self.assertEqual(value["entries"][0]["status"], "moved")
        self.assertEqual(value["entries"][1]["status"], "planned")
        # A partial manifest safely restores the entries that actually moved.
        q.restore(self.root, manifest, execute=True, activity_checker=lambda paths: [])
        self.assertTrue((self.root / "output" / "result.bin").exists())
        self.assertTrue((scratch / "report.log").exists())
        again = q.restore(self.root, manifest, execute=True, activity_checker=lambda paths: [])
        self.assertEqual(again["state"], "restored")

    def test_destination_created_after_planning_is_refused(self):
        self.make_output()

        def create_conflict(paths):
            destination = q.quarantine_root(self.root) / "test-operation" / "output"
            destination.mkdir(parents=True)
            return []

        with self.assertRaisesRegex(ValueError, "appeared after planning"):
            self.quarantine(execute=True, activity_checker=create_conflict)
        self.assertTrue((self.root / "output" / "result.bin").exists())

    def test_restore_is_idempotent_and_parser_has_no_delete_command(self):
        self.make_output()
        result = self.quarantine(execute=True)
        manifest = self.root / result["manifest"]
        repeated_quarantine = self.quarantine(execute=True)
        self.assertEqual(repeated_quarantine["state"], "no-op")
        q.restore(self.root, manifest, execute=True, activity_checker=lambda paths: [])
        again = q.restore(self.root, manifest, execute=True, activity_checker=lambda paths: [])
        self.assertEqual(again["state"], "restored")
        parser = q.build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        self.assertEqual(set(subparsers.choices), {"audit", "quarantine", "restore"})


if __name__ == "__main__":
    unittest.main()
