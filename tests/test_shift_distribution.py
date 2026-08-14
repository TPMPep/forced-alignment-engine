import unittest

from app import OUTLIER_TOLERANCE_MS, shift_distribution


def word(key: str, aligned_start: int, provider_start: int) -> dict:
    return {
        "key": key,
        "start_ms": aligned_start,
        "end_ms": aligned_start + 200,
        "provider_start_ms": provider_start,
        "provider_end_ms": provider_start + 200,
    }


class ShiftDistributionTests(unittest.TestCase):
    def test_single_outlier_does_not_move_p99_on_a_long_program(self):
        words = [word(f"p:{index}", index * 1000, index * 1000 + 50) for index in range(1000)]
        words.append(word("p:1000", 1_000_000, 1_000_000 + OUTLIER_TOLERANCE_MS + 124_679))
        distribution = shift_distribution(words)
        self.assertGreater(distribution["max_provider_shift_ms"], OUTLIER_TOLERANCE_MS)
        self.assertLessEqual(distribution["p99_provider_shift_ms"], OUTLIER_TOLERANCE_MS)
        self.assertEqual(distribution["outlier_word_count"], 1)
        self.assertLess(distribution["outlier_ratio"], 0.005)
        self.assertEqual(distribution["outlier_sample"][0]["key"], "p:1000")

    def test_systemic_drift_moves_p99_and_the_outlier_ratio(self):
        words = [word(f"p:{index}", index * 1000, index * 1000 + OUTLIER_TOLERANCE_MS + 5_000) for index in range(100)]
        distribution = shift_distribution(words)
        self.assertGreater(distribution["p99_provider_shift_ms"], OUTLIER_TOLERANCE_MS)
        self.assertEqual(distribution["outlier_word_count"], 100)
        self.assertEqual(distribution["outlier_ratio"], 1.0)

    def test_clean_alignment_reports_zero_outliers(self):
        words = [word(f"p:{index}", index * 1000, index * 1000 + 40) for index in range(50)]
        distribution = shift_distribution(words)
        self.assertEqual(distribution["outlier_word_count"], 0)
        self.assertEqual(distribution["outlier_ratio"], 0.0)
        self.assertEqual(distribution["outlier_sample"], [])
        self.assertEqual(distribution["max_provider_shift_ms"], 40)


if __name__ == "__main__":
    unittest.main()
