"""Adaptive search-window expansion policy.

GROUND TRUTH FIXTURE — project 6a7d874aa2ddd372f426a4df, transcript line 18
("...People's Army marshal who played a key role in shaping his leadership.").
Values below are the real archived alignment evidence, not invented numbers:

    provider last word end   93,664ms
    trail pad granted        350ms      → audio slice ended 94,014ms
    aligned result           role / in / shaping / his / leadership.
                             ALL at 93,986 → 93,987 (1ms each, one stack)
    displacement ramp        marshal +316, who +661, played +808, a +890, key +944

The words are really spoken out to ~96,200ms, so no downstream repair could ever
recover them: that audio was never analysed. These tests lock the behaviour that
the region is EXPANDED on this evidence, that the expansion is bounded by the
neighbour's observed speech rather than by the ASR provider's segment boundary,
and — critically — that ordinary alignments are left completely alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.expansion import (  # noqa: E402
    NEIGHBOUR_GUARD_MS,
    detect_saturation,
    expansion_plan,
    unresolved_words,
)

WINDOW = {
    "start_ms": 87_000.0,
    "end_ms": 94_014.0,
    "previous_speech_end_ms": 86_500,
    "next_speech_start_ms": 96_232,
}


def word(key, text, start, end, provider_start=None, provider_end=None):
    return {
        "key": key,
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "provider_start_ms": provider_start if provider_start is not None else start,
        "provider_end_ms": provider_end if provider_end is not None else end,
    }


def line_18_aligned():
    """The archived result: a displacement ramp terminating in a five-word stack."""
    return [
        word("s18:0", "People's", 92_306, 92_706, 92_249, 92_594),
        word("s18:1", "Army", 92_727, 92_966, 92_594, 92_710),
        word("s18:2", "marshal", 93_026, 93_447, 92_710, 92_825),
        word("s18:3", "who", 93_486, 93_626, 92_825, 92_858),
        word("s18:4", "played", 93_666, 93_826, 92_858, 92_956),
        word("s18:5", "a", 93_846, 93_906, 92_956, 93_022),
        word("s18:6", "key", 93_966, 93_986, 93_022, 93_072),
        word("s18:7", "role", 93_986, 93_987, 93_072, 93_170),
        word("s18:8", "in", 93_986, 93_987, 93_170, 93_220),
        word("s18:9", "shaping", 93_986, 93_987, 93_220, 93_368),
        word("s18:10", "his", 93_986, 93_987, 93_483, 93_533),
        word("s18:11", "leadership.", 93_986, 93_987, 93_549, 93_664),
    ]


def test_line_18_trailing_exhaustion_is_detected_on_multiple_signals():
    edges = detect_saturation(line_18_aligned(), WINDOW, confidence=0.82)
    trail = edges["trail"]
    assert trail["exhausted"] is True
    # Every independent signal the archived evidence actually contains.
    assert "stacked_at_edge" in trail["signals"]
    assert "degenerate_duration" in trail["signals"]
    assert "edge_pinned" in trail["signals"]
    assert "displacement_ramp" in trail["signals"]
    assert "compressed_run" in trail["signals"]
    # The whole stack is implicated, not just the final word.
    assert "s18:11" in trail["word_keys"]
    assert "s18:7" in trail["word_keys"]
    # The leading edge is healthy and must not be dragged into the verdict.
    assert edges["lead"]["exhausted"] is False


def test_expansion_is_bounded_by_neighbouring_speech_not_the_asr_boundary():
    edges = detect_saturation(line_18_aligned(), WINDOW, confidence=0.82)
    plan = expansion_plan(WINDOW, edges, pass_number=1)
    ceiling = WINDOW["next_speech_start_ms"] - NEIGHBOUR_GUARD_MS - WINDOW["end_ms"]
    assert plan["trail_ceiling_ms"] == ceiling
    assert plan["trail_expansion_ms"] == ceiling
    # The expanded slice stops short of the next utterance's audio — it can never
    # absorb speech belonging to the following line.
    assert WINDOW["end_ms"] + plan["trail_expansion_ms"] < WINDOW["next_speech_start_ms"]
    # A healthy edge is never expanded "just in case".
    assert plan["lead_expansion_ms"] == 0


def test_expansion_is_refused_when_the_neighbour_leaves_no_room():
    tight = {**WINDOW, "next_speech_start_ms": 94_100}
    edges = detect_saturation(line_18_aligned(), tight, confidence=0.82)
    plan = expansion_plan(tight, edges, pass_number=1)
    assert plan["trail_expansion_ms"] == 0
    assert "trail:no_headroom" in plan["blocked"]


def test_expansion_request_is_sized_by_what_the_words_need():
    """A short tail must not open a wide window just because room exists."""
    aligned = [
        word("s40:0", "Okay", 50_000, 50_300, 49_000, 49_300),
        word("s40:1", "sure.", 50_960, 50_961, 49_400, 49_520),
    ]
    window = {
        "start_ms": 49_500.0,
        "end_ms": 50_970.0,
        "previous_speech_end_ms": 48_000,
        "next_speech_start_ms": 90_000,  # 39 seconds of headroom available
    }
    edges = detect_saturation(aligned, window, confidence=0.7)
    assert edges["trail"]["exhausted"] is True
    plan = expansion_plan(window, edges, pass_number=1)
    assert 0 < plan["trail_expansion_ms"] < 2_000


def test_fast_legitimate_speech_is_not_expanded():
    """Plausible words that simply end near the edge are left alone."""
    aligned = [
        word("s5:0", "I", 10_000, 10_090),
        word("s5:1", "know", 10_100, 10_320),
        word("s5:2", "right", 10_330, 10_700),
    ]
    window = {"start_ms": 9_900.0, "end_ms": 11_000.0, "previous_speech_end_ms": 9_000, "next_speech_start_ms": 12_500}
    edges = detect_saturation(aligned, window, confidence=0.95)
    assert edges["trail"]["exhausted"] is False
    assert edges["lead"]["exhausted"] is False


def test_single_signal_never_triggers_expansion():
    """A word ending flush at the edge is ONE signal — not enough on its own."""
    aligned = [
        word("s9:0", "In", 20_000, 20_180),
        word("s9:1", "Mogadishu,", 20_200, 20_900),
        word("s9:2", "leadership.", 21_000, 21_800),
    ]
    window = {"start_ms": 19_900.0, "end_ms": 21_800.0, "previous_speech_end_ms": 19_000, "next_speech_start_ms": 24_000}
    edges = detect_saturation(aligned, window, confidence=0.93)
    assert edges["trail"]["signals"] == ["edge_pinned"]
    assert edges["trail"]["exhausted"] is False


def test_low_confidence_alone_cannot_justify_expansion():
    """Poor loss is corroboration only; it can never be one of the two signals."""
    aligned = [
        word("s7:0", "Yes", 30_000, 30_240),
        word("s7:1", "absolutely.", 30_300, 31_100),
    ]
    window = {"start_ms": 29_900.0, "end_ms": 31_100.0, "previous_speech_end_ms": 29_000, "next_speech_start_ms": 33_000}
    edges = detect_saturation(aligned, window, confidence=0.2)
    assert "low_confidence" in edges["trail"]["signals"]
    assert edges["trail"]["exhausted"] is False


def test_unresolved_reports_stacked_and_implausible_words():
    aligned = line_18_aligned()
    edges = detect_saturation(aligned, WINDOW, confidence=0.82)
    unresolved = unresolved_words(aligned, WINDOW, edges)
    # The zero-width stack is never presented as valid timing.
    assert "s18:11" in unresolved
    assert "s18:7" in unresolved
    # Words that placed plausibly are untouched.
    assert "s18:0" not in unresolved
    assert "s18:2" not in unresolved


def test_implausibly_long_word_is_unresolved_even_without_edge_saturation():
    """Silence/music absorption stays rejected — expansion must not reintroduce it."""
    aligned = [word("s16:0", "News.", 100_000, 102_000, 99_800, 100_010)]
    window = {"start_ms": 99_700.0, "end_ms": 103_000.0, "previous_speech_end_ms": 99_000, "next_speech_start_ms": 105_000}
    edges = detect_saturation(aligned, window, confidence=0.9)
    assert "s16:0" in unresolved_words(aligned, window, edges)


def test_leading_edge_exhaustion_is_detected_independently():
    aligned = [
        word("s22:0", "But", 60_000, 60_001, 61_200, 61_300),
        word("s22:1", "the", 60_000, 60_001, 61_350, 61_420),
        word("s22:2", "point", 60_400, 60_900, 61_500, 61_700),
    ]
    window = {"start_ms": 60_000.0, "end_ms": 63_000.0, "previous_speech_end_ms": 56_000, "next_speech_start_ms": 65_000}
    edges = detect_saturation(aligned, window, confidence=0.8)
    assert edges["lead"]["exhausted"] is True
    plan = expansion_plan(window, edges, pass_number=1)
    assert plan["lead_expansion_ms"] > 0
    assert window["start_ms"] - plan["lead_expansion_ms"] > window["previous_speech_end_ms"]
