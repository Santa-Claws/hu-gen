import unittest

from research.detector_comparison.corpus import build_mixed_sample, sha256_text


class CorpusTests(unittest.TestCase):
    def test_build_mixed_sample_preserves_segments_and_provenance(self):
        human = "Human passage with a specific observation. " * 10
        ai = "AI passage with a different observation. " * 10

        sample = build_mixed_sample(
            sample_id="mixed-example-001",
            human_text=human,
            ai_text=ai,
            human_source_id="human-alice-001",
            ai_source_id="ai-story-001",
        )

        self.assertEqual(sample["id"], "mixed-example-001")
        self.assertEqual(sample["label"], "mixed")
        self.assertEqual(sample["source_ids"], ["human-alice-001", "ai-story-001"])
        self.assertIn(human.strip(), sample["text"])
        self.assertIn(ai.strip(), sample["text"])
        self.assertEqual(sample["text_sha256"], sha256_text(sample["text"]))
        self.assertGreater(sample["word_count"], 100)


if __name__ == "__main__":
    unittest.main()
