from dataclasses import FrozenInstanceError
from unittest import TestCase
from unittest.mock import patch

from experiments.progress_plugin import INLINE_PROGRESS


class InlineProgressPluginTests(TestCase):
    def test_policy_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            INLINE_PROGRESS.ncols = 120

    def test_create_always_uses_inline_fixed_width_policy(self):
        with patch("experiments.progress_plugin.tqdm") as mocked_tqdm:
            INLINE_PROGRESS.create(range(3), desc="Training")

        _, kwargs = mocked_tqdm.call_args
        self.assertEqual(kwargs["ncols"], 80)
        self.assertFalse(kwargs["dynamic_ncols"])
