import unittest

from research.detector_comparison.configurations import get_configuration, validate_vram


class ConfigurationTests(unittest.TestCase):
    def test_baseline_is_eligible_on_a_12_gib_gpu(self) -> None:
        baseline = get_configuration("neo_2p7b__neo_2p7b__analytic")

        self.assertEqual(validate_vram(baseline, available_gib=12), [])
        self.assertEqual(baseline["sampling_model"], "gpt-neo-2.7B")
        self.assertEqual(baseline["scoring_model"], "gpt-neo-2.7B")
        self.assertEqual(baseline["criterion_mode"], "analytic")

    def test_stronger_gpt_j_pair_is_rejected_on_a_12_gib_gpu(self) -> None:
        configuration = get_configuration("gptj_6b__neo_2p7b__analytic")

        errors = validate_vram(configuration, available_gib=12)

        self.assertIn("requires at least 20 GiB VRAM; only 12 GiB available", errors)

    def test_unknown_configuration_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown configuration"):
            get_configuration("made-up")
