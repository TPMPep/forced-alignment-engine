import unittest

from app import InputWord, chunk_words, normalize_aligned_timeline


class TimelinePolicyTests(unittest.TestCase):
    def test_overlapping_provider_words_stay_in_one_chunk(self):
        words = [
            InputWord(key="p:0", text="uno", provider_start_ms=1000, provider_end_ms=1500),
            InputWord(key="p:1", text="dos", provider_start_ms=1400, provider_end_ms=1800),
            InputWord(key="p:2", text="tres", provider_start_ms=1800, provider_end_ms=2200),
        ]
        self.assertEqual(len(chunk_words(words)), 1)

    def test_bounded_regression_is_preserved_and_neutralized(self):
        words = [
            {"key": "p:0", "text": "uno", "start_ms": 1000, "end_ms": 1200, "confidence": 0.9},
            {"key": "p:1", "text": "dos", "start_ms": 900, "end_ms": 1100, "confidence": 0.9},
            {"key": "p:2", "text": "tres", "start_ms": 1300, "end_ms": 1500, "confidence": 0.9},
        ]
        normalized, repairs, max_regression = normalize_aligned_timeline(words)
        self.assertEqual(repairs, 1)
        self.assertEqual(max_regression, 100)
        self.assertEqual(normalized[1]["raw_start_ms"], 900)
        self.assertEqual(normalized[1]["start_ms"], 1000)
        self.assertEqual(normalized[1]["confidence"], 0.0)
        self.assertTrue(normalized[1]["timing_repaired"])


if __name__ == "__main__":
    unittest.main()
