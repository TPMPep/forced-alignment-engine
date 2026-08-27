"""Tokenization-reconciliation contract (tokenization policy v2).

The Japanese Duo failure this closes: 89 submitted tokens, 112 returned words,
HTTP 422, dual-model consensus impossible for every space-free script. These tests
pin BOTH halves of the fix — that space-delimited runs are unchanged, and that the
fail-closed behaviour survives.
"""

import pytest

from services.token_reconcile import alignable_chars, reconcile_returned_words


def word(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_one_to_one_is_byte_for_byte_unchanged():
    """A space-delimited run must resolve to exactly the provider's own windows.

    This is the regression guard on every existing English/European programme: the
    reconciliation may not perturb a single millisecond where the two tokenizations
    already agree, and nothing may be reported as estimated.
    """
    plan = reconcile_returned_words(
        ["This", "program", "is", "presented"],
        [word("This", 0.0, 0.30), word("program", 0.31, 0.90), word("is", 0.91, 1.00), word("presented", 1.01, 1.80)],
    )
    assert [(p["start"], p["end"]) for p in plan] == [(0.0, 0.30), (0.31, 0.90), (0.91, 1.00), (1.01, 1.80)]
    assert not any(p["estimated"] for p in plan)


def test_provider_split_takes_measured_span():
    """A token the provider split is MEASURED, not estimated: first start, last end."""
    plan = reconcile_returned_words(
        ["よろしくお願いいたします"],
        [word("よろしく", 1.0, 1.5), word("お願い", 1.5, 2.0), word("いたします", 2.0, 2.6)],
    )
    assert plan[0]["start"] == 1.0
    assert plan[0]["end"] == 2.6
    assert plan[0]["estimated"] is False
    assert plan[0]["returned_word_count"] == 3


def test_provider_merge_subdivides_and_discloses():
    """Tokens the provider merged share one window, split by character count.

    The subdivision is a derived value, so both halves must be flagged estimated —
    a derived window that looks measured is the failure this discloses.
    """
    plan = reconcile_returned_words(["あい", "うえお"], [word("あいうえお", 0.0, 1.0)])
    assert plan[0]["estimated"] is True and plan[1]["estimated"] is True
    assert plan[0]["start"] == 0.0
    assert plan[0]["end"] == pytest.approx(0.4)
    assert plan[1]["start"] == pytest.approx(0.4)
    assert plan[1]["end"] == 1.0


def test_japanese_count_divergence_no_longer_fails():
    """The exact production shape: more returned words than submitted tokens."""
    plan = reconcile_returned_words(
        ["私は", "長崎大学病院の", "高岡と申します"],
        [
            word("私", 0.0, 0.2), word("は", 0.2, 0.3),
            word("長崎", 0.3, 0.7), word("大学", 0.7, 1.1), word("病院", 1.1, 1.5), word("の", 1.5, 1.6),
            word("高岡", 1.6, 2.0), word("と", 2.0, 2.1), word("申します", 2.1, 2.8),
        ],
    )
    assert len(plan) == 3
    assert (plan[0]["start"], plan[0]["end"]) == (0.0, 0.3)
    assert (plan[2]["start"], plan[2]["end"]) == (1.6, 2.8)


def test_punctuation_only_tokens_are_ignored_on_both_sides():
    """Punctuation anchors nothing; the two tokenizers may disagree about it freely."""
    plan = reconcile_returned_words(
        ["します。", "はい"],
        [word("します", 0.0, 0.5), word("。", 0.5, 0.5), word("はい", 0.6, 0.9)],
    )
    assert (plan[0]["start"], plan[0]["end"]) == (0.0, 0.5)
    assert (plan[1]["start"], plan[1]["end"]) == (0.6, 0.9)


def test_token_with_no_alignable_characters_still_gets_a_window():
    """The 1:1 output contract holds even for a token carrying no speech."""
    plan = reconcile_returned_words(["はい", "。", "どうぞ"], [word("はい", 0.0, 0.4), word("どうぞ", 0.5, 1.0)])
    assert len(plan) == 3
    assert plan[1]["estimated"] is True
    assert plan[1]["end"] > plan[1]["start"]
    assert plan[2]["start"] == 0.5


def test_different_transcript_still_fails_closed():
    """The whole safety value: a provider that aligned other text must still raise."""
    with pytest.raises(ValueError) as error:
        reconcile_returned_words(["hello", "world"], [word("hello", 0.0, 0.4), word("planet", 0.4, 0.9)])
    assert "character-stream mismatch" in str(error.value)


def test_truncated_provider_result_still_fails_closed():
    with pytest.raises(ValueError):
        reconcile_returned_words(["hello", "world"], [word("hello", 0.0, 0.4)])


def test_windows_never_regress():
    """Reconciled output is monotonic, so the normalizer is never handed a regression."""
    plan = reconcile_returned_words(
        ["one", "two", "three"],
        [word("one", 1.0, 1.4), word("two", 0.9, 1.2), word("three", 1.5, 2.0)],
    )
    starts = [p["start"] for p in plan]
    assert starts == sorted(starts)
    assert plan[1]["end"] >= plan[1]["start"]


def test_alignable_chars_matches_normalized_token_semantics():
    assert alignable_chars("Med-scape,") == "medscape"
    assert alignable_chars("。") == ""
