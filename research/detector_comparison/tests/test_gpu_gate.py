import unittest

from research.detector_comparison.gpu_gate import assess_gpu


class GpuGateTests(unittest.TestCase):
    def test_allows_an_idle_gpu_with_low_memory_use(self) -> None:
        decision = assess_gpu(memory_used_mib=300, memory_total_mib=12288, utilization_percent=2)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reasons, [])

    def test_blocks_a_busy_gpu(self) -> None:
        decision = assess_gpu(memory_used_mib=2469, memory_total_mib=12288, utilization_percent=79)

        self.assertFalse(decision.allowed)
        self.assertIn("GPU utilization 79% exceeds 10% admission limit", decision.reasons)
        self.assertIn("GPU memory use 2469 MiB exceeds 1024 MiB admission limit", decision.reasons)

    def test_rejects_invalid_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            assess_gpu(memory_used_mib=-1, memory_total_mib=12288, utilization_percent=0)
