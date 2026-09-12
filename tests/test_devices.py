import unittest
from unittest.mock import patch

from experiments.devices import resolve_device


class DeviceResolutionTestCase(unittest.TestCase):
    def test_q_learning_rejects_an_explicit_cuda_device(self):
        with self.assertRaisesRegex(ValueError, "Q-learning.*CPU"):
            resolve_device("q_learning", "train", "cuda")

    def test_evaluation_rejects_cuda_for_official_cpu_compatibility(self):
        with self.assertRaisesRegex(ValueError, "evaluation.*CPU"):
            resolve_device("dqn", "evaluate", "cuda")

    @patch("experiments.devices.torch.cuda.is_available", return_value=False)
    def test_auto_training_falls_back_to_cpu_without_cuda(self, _is_available):
        device = resolve_device("dqn", "train", "auto")
        self.assertEqual(device["requested"], "auto")
        self.assertEqual(device["actual"], "cpu")
        self.assertEqual(device["type"], "cpu")

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_records_the_visible_gpu(self):
        device = resolve_device("dqn", "train", "cuda")
        self.assertEqual(device["actual"], "cuda:0")
        self.assertEqual(device["type"], "cuda")
        self.assertTrue(device["name"])
        self.assertTrue(device["torch_version"])
        self.assertTrue(device["cuda_version"])


if __name__ == "__main__":
    unittest.main()
