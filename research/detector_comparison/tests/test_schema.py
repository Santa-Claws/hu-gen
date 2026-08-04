import unittest

from research.detector_comparison.schema import validate_sample


class SampleSchemaTests(unittest.TestCase):
    def test_accepts_a_complete_ai_generated_record(self) -> None:
        text = "A labelled synthetic benchmark sample."
        sample = {
            "id": "ai-001",
            "label": "ai_generated",
            "text": text,
            "word_count": 5,
            "text_sha256": "f0c551093296777bc8ed1004b71b23388d780cc9ca58c4212d4b82def90fb556",
            "source_type": "assistant_generated",
            "license_or_consent": "synthetic-test-data",
            "generator_model": "test-model",
            "generator_prompt": "test prompt",
            "source_group": "prompt-family-a",
        }

        self.assertEqual(validate_sample(sample), [])

    def test_rejects_ai_record_without_generator_provenance(self) -> None:
        sample = {
            "id": "ai-001",
            "label": "ai_generated",
            "text": "A labelled synthetic benchmark sample.",
            "word_count": 5,
            "text_sha256": "f0c551093296777bc8ed1004b71b23388d780cc9ca58c4212d4b82def90fb556",
            "source_type": "assistant_generated",
            "license_or_consent": "synthetic-test-data",
            "source_group": "prompt-family-a",
        }

        self.assertIn("generator_model is required for ai_generated samples", validate_sample(sample))

    def test_rejects_tampered_text_hash(self) -> None:
        sample = {
            "id": "human-001",
            "label": "human_authored",
            "text": "A known human-authored source passage.",
            "word_count": 5,
            "text_sha256": "0" * 64,
            "source_type": "public_domain_online",
            "license_or_consent": "public domain",
            "source_url": "https://example.test/source",
            "source_group": "document-1",
        }

        self.assertIn("text_sha256 does not match text", validate_sample(sample))

    def test_rejects_mixed_sample_without_both_source_ids(self) -> None:
        sample = {
            "id": "mixed-001",
            "label": "mixed",
            "text": "Human segment. AI segment.",
            "word_count": 4,
            "text_sha256": "a" * 64,
            "source_type": "mixed",
            "license_or_consent": "derived from labelled benchmark inputs",
            "source_group": "mixed-001",
            "source_ids": ["human-001"],
        }

        self.assertIn("mixed samples require at least two source_ids", validate_sample(sample))
