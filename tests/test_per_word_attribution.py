"""Per-word expansion attribution + evidence-relative unresolved reasons (engine
expansion policy v3).

WHY THESE EXIST. Run-level totals ("1 chunk expanded, 2,098ms") forced the
consumer to INFER which words an expansion affected, and an inferred attribution
is not evidence. Likewise a bare unresolved COUNT is unactionable: an operator
told "1 unresolved word" has no way to determine which word failed or why.

The reasons are also derived from the SAME evidence-relative ceiling the worker's
arbitration uses, so the engine can never declare a window acceptable that the
downstream audit then rejects (or vice versa).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.expansion import unresolved_reasons, unresolved_words  # noqa: E402
from services.plausibility import (  # noqa: E402
    MIN_PLAUSIBLE_WORD_MS,
    evidence_ceiling_ms,
    max_word_duration_ms,
    provider_capture_credible,
)


def word(key, text, start, end, provider_start=None, provider_end=None):
    return {
        "key": key,
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "provider_start_ms": start if provider_start is None else provider_start,
        "provider_end_ms": end if provider_end is None else provider_end,
    }


HEALTHY_EDGES = {
    "lead": {"exhausted": False, "signals": [], "word_keys": []},
    "trail": {"exhausted": False, "signals": [], "word_keys": []},
}


def test_healthy_words_have_no_unresolved_reasons():
    aligned = [
        word("s:0", "hello", 1000, 1400),
        word("s:1", "there", 1450, 1800),
    ]
    assert unresolved_reasons(aligned, {"start_ms": 900, "end_ms": 1900}, HEALTHY_EDGES) == {}


def test_exhausted_edge_words_carry_the_edge_reason():
    aligned = [word("s:0", "leadership.", 93986, 93987)]
    edges = {
        "lead": {"exhausted": False, "word_keys": []},
        "trail": {"exhausted": True, "word_keys": ["s:0"]},
    }
    reasons = unresolved_reasons(aligned, {"start_ms": 90000, "end_ms": 93987}, edges)
    assert reasons["s:0"] == "search_region_exhausted_at_trail_edge"
    # Keys-only helper stays available for callers that do not need reasons.
    assert unresolved_words(aligned, {"start_ms": 90000, "end_ms": 93987}, edges) == ["s:0"]


def test_sub_floor_window_is_reported_as_below_the_evidence_floor():
    aligned = [word("s:0", "Officials", 153763, 153764, 156250, 156570)]
    reasons = unresolved_reasons(aligned, {"start_ms": 150000, "end_ms": 160000}, HEALTHY_EDGES)
    assert reasons["s:0"] == "aligned_window_below_evidence_floor"


def test_inflation_beyond_a_credible_capture_is_reported():
    # A short word measured at 210ms, aligned across 1,500ms. The generic rate
    # ceiling for this word IS 1,500ms, so only the evidence-relative bound sees it.
    aligned = [word("s:0", "News.", 85746, 87246, 85216, 85426)]
    assert max_word_duration_ms("News.") == 1500
    reasons = unresolved_reasons(aligned, {"start_ms": 84000, "end_ms": 88000}, HEALTHY_EDGES)
    assert reasons["s:0"] == "aligned_window_inflated_beyond_evidence"


def test_a_crushed_capture_never_bounds_the_aligner():
    # The mirror-image class: the provider window is too short to be believed, so
    # a legitimately longer acoustic window must NOT be reported unresolved.
    aligned = [word("s:0", "leadership.", 95366, 95886, 93549, 93664)]
    assert provider_capture_credible("leadership.", 115) is False
    assert evidence_ceiling_ms("leadership.", 115) == max_word_duration_ms("leadership.")
    assert unresolved_reasons(aligned, {"start_ms": 94000, "end_ms": 96200}, HEALTHY_EDGES) == {}


def test_long_and_fast_speech_are_not_falsely_reported():
    aligned = [
        word("s:0", "anticonstitutionnellement", 1000, 4000, 1020, 3820),
        word("s:1", "the", 4100, 4280, 4105, 4305),
    ]
    assert unresolved_reasons(aligned, {"start_ms": 900, "end_ms": 4400}, HEALTHY_EDGES) == {}


def test_evidence_floor_is_a_floor_on_evidence_not_on_word_length():
    # Both timelines agreeing on a brief window is corroboration, and the ceiling
    # only tightens when the capture is credible.
    assert MIN_PLAUSIBLE_WORD_MS == 40
    assert evidence_ceiling_ms("News.", None) == max_word_duration_ms("News.")
    assert evidence_ceiling_ms("News.", 210) < max_word_duration_ms("News.")
