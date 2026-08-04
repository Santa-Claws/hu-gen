import unittest

from research.detector_comparison.corpus import limit_words


class HumanSourceTests(unittest.TestCase):
    def test_limit_words_normalizes_whitespace_without_cutting_before_limit(self):
        text = "  one\n two   three\tfour five six  "

        self.assertEqual(limit_words(text, 4), "one two three four")
        self.assertEqual(limit_words(text, 10), "one two three four five six")


if __name__ == "__main__":
    unittest.main()
