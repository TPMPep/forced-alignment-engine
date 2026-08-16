"""Edge-padding absorption bound.

WHY THIS TEST EXISTS: a fixed chunk pad is a fixed absorption budget. Padding is
audio with no transcript, and forced alignment must account for every millisecond
it is given, so the pad lands on the edge word. Ground truth from project
6a7d874aa2ddd372f426a4df row 17: provider speech ended at 93,664ms, the chunk
therefore ended at 93,664 + 2,000 = 95,664, and the final word was aligned ending
at 95,639 — the entire pad, absorbed, to within 25ms. Every divergence flagged on
that program equalled the pad almost exactly.

The bound these tests lock is the ONLY reason the defect cannot recur: an edge
word's maximum absorption is EDGE_PADDING_MS, which must stay below the 650ms
breath boundary used by segment shaping and the 1,500ms divergence trust
threshold used by the worker's timeline-integrity audit. If either constant drifts
up past those, edge absorption can once again split a line and misattribute a
speaker, and these tests must fail loudly rather than let it ship.
"""

import unittest

from app import EDGE_PADDING_FLOOR_MS, EDGE_PADDING_MS, InputWord, chunk_windows

# Mirrored from lib/segment-shaping.js and the worker timeline-integrity policy.
# Duplicated deliberately: if either side moves, this test is the alarm.
BREATH_BOUNDARY_MS = 650
DIVERGENCE_TRUST_THRESHOLD_MS = 1_500


def word(key: str, start_ms: float, end_ms: float) -> InputWord:
    return InputWord(key=key, text="word", provider_start_ms=start_ms, provider_end_ms=end_ms)


class EdgePaddingTests(unittest.TestCase):
    def test_absorption_ceiling_stays_below_every_downstream_threshold(self):
        # The whole control rests on this inequality.
        self.assertLess(EDGE_PADDING_MS, BREATH_BOUNDARY_MS)
        self.assertLess(EDGE_PADDING_MS, DIVERGENCE_TRUST_THRESHOLD_MS)
        self.assertLessEqual(EDGE_PADDING_FLOOR_MS, EDGE_PADDING_MS)

    def test_pad_never_exceeds_half_the_measured_silence(self):
        # A 900ms measured gap may spend AT MOST half (450ms) per side, so two
        # adjacent chunks can never pad into the same silence or into each other's
        # speech. BOTH bounds apply and the tighter one wins: the pad is
        # min(EDGE_PADDING_MS, half the gap), floored. At a 900ms gap the 350ms cap
        # is tighter than the 450ms half-gap, so 350 is the correct answer — an
        # earlier revision of this test asserted 450 and was simply wrong about the
        # rule. The invariant that actually matters (never more than half) is
        # asserted directly rather than inferred from one arithmetic result, so this
        # keeps failing loudly if EITHER bound is ever loosened.
        groups = [[word("a:0", 0, 1_000)], [word("b:0", 1_900, 2_500)]]
        first, second = chunk_windows(groups)
        half_gap = 450
        expected = min(EDGE_PADDING_MS, half_gap)
        self.assertEqual(first["trail_gap_ms"], 900)
        self.assertEqual(first["trail_pad_ms"], expected)
        self.assertEqual(second["lead_pad_ms"], expected)
        self.assertLessEqual(first["trail_pad_ms"], half_gap)
        self.assertLessEqual(second["lead_pad_ms"], half_gap)
        # The two windows still cannot meet inside the measured silence.
        self.assertLessEqual(first["end_ms"], second["start_ms"])

    def test_wide_silence_is_capped_not_consumed(self):
        # The row-17 shape: speech, then a 2.57s untranscribed gap. The old fixed
        # 2,000ms pad gave the edge word the whole budget; the cap now bounds it.
        groups = [[word("a:0", 91_000, 93_664)], [word("b:0", 96_232, 97_000)]]
        first, second = chunk_windows(groups)
        self.assertEqual(first["trail_gap_ms"], 2_568)
        self.assertEqual(first["trail_pad_ms"], EDGE_PADDING_MS)
        self.assertEqual(first["end_ms"], 93_664 + EDGE_PADDING_MS)
        self.assertLess(first["end_ms"], 95_664)
        self.assertEqual(second["lead_pad_ms"], EDGE_PADDING_MS)

    def test_program_edges_use_the_cap_and_never_go_negative(self):
        groups = [[word("a:0", 120, 900)]]
        (only,) = chunk_windows(groups)
        self.assertIsNone(only["lead_gap_ms"])
        self.assertIsNone(only["trail_gap_ms"])
        self.assertEqual(only["lead_pad_ms"], EDGE_PADDING_MS)
        self.assertGreaterEqual(only["start_ms"], 0.0)

    def test_tight_or_overlapping_streams_fall_back_to_the_floor(self):
        # Overlapping provider streams yield a negative measured gap. Clipping the
        # word is the worse failure there, so the floor applies — and it is still a
        # sub-breath amount of absorbable audio.
        groups = [[word("a:0", 5_000, 6_000)], [word("b:0", 5_900, 6_500)]]
        first, second = chunk_windows(groups)
        self.assertEqual(first["trail_pad_ms"], EDGE_PADDING_FLOOR_MS)
        self.assertEqual(second["lead_pad_ms"], EDGE_PADDING_FLOOR_MS)
        self.assertLess(EDGE_PADDING_FLOOR_MS, BREATH_BOUNDARY_MS)

    def test_window_always_contains_the_full_speech_span(self):
        groups = [
            [word("a:0", 10_000, 10_500), word("a:1", 10_600, 11_400)],
            [word("b:0", 20_000, 20_800)],
        ]
        for window, (speech_start, speech_end) in zip(chunk_windows(groups), [(10_000, 11_400), (20_000, 20_800)]):
            self.assertLessEqual(window["start_ms"], speech_start)
            self.assertGreaterEqual(window["end_ms"], speech_end)
            self.assertEqual(window["speech_start_ms"], speech_start)
            self.assertEqual(window["speech_end_ms"], speech_end)


if __name__ == "__main__":
    unittest.main()
