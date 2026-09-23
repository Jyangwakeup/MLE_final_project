"""Regression boundaries where Task4 transfer and remote runner migration meet."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import run

ROOT = Path(__file__).resolve().parents[1]
TASK4_FLAGS = (
    '--transfer-task4-from-checkpoint',
    '--transfer-task4-exploration-from-checkpoint',
    '--transfer-task4-frozen-opponents-from-checkpoint',
    '--transfer-task4-score-from-checkpoint',
)


class RemoteMergeContractTests(unittest.TestCase):
    def args(self, mode='evaluate'):
        return ['--agent', 'q_learning_agent', '--task', '1', '--mode', mode,
                '--config', str(ROOT/'experiments/configs/reward_r2_balanced.json'),
                '--run-id', 'must-not-be-created']

    def test_runtime_migration_is_exclusive_with_every_task4_transfer(self):
        for flag in TASK4_FLAGS:
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run._parser().parse_args(self.args('train') + [
                        '--runtime-migrate-from', 'parent', flag, 'checkpoint'])
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_rejects_all_migration_entries_before_running(self):
        for flag in ('--runtime-migrate-from', *TASK4_FLAGS):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()) as stderr:
                with patch.object(run, 'run_evaluation_mode') as evaluate:
                    self.assertEqual(run.main(self.args() + [flag, 'parent']), 2)
                    evaluate.assert_not_called()
                self.assertIn('only valid with --mode train', stderr.getvalue())

    def test_q_compatibility_entry_keeps_agent_local_checkpoint_path(self):
        from agent_code.q_learning_agent import callbacks
        entry = ROOT/'agent_code/q_learning_agent/callbacks.py'
        self.assertTrue(entry.is_symlink())
        self.assertEqual(entry.resolve(), ROOT/'experiments/agent_variants/q_learning_agent/callbacks.py')
        self.assertEqual(Path(callbacks.__file__).with_name('final.pkl'),
                         ROOT/'agent_code/q_learning_agent/final.pkl')


if __name__ == '__main__':
    unittest.main()
