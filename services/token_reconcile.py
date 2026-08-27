"""Reconcile the provider's OWN tokenization back onto our input tokens.

WHY THIS EXISTS (root cause, not a loosened check)
--------------------------------------------------
`align_chunk` sends one transcript string built by joining our input tokens with
spaces, and then assumed the provider returns exactly one word per input token,
in order, so it could `zip()` the two lists. That assumption is only true for
space-delimited languages.

ElevenLabs forced alignment re-tokenizes the transcript it is given. For Japanese
(and Chinese, and any script written without spaces) it segments by its own
morphology and ignores the spaces we inserted, so the returned word count has no
relationship to ours. Ground truth, project 6a7d8758ce3722efa0a0c73a
(source_language 'ja', 2026-08-27): 89 input tokens produced 112 returned words,
the count guard raised, the engine answered HTTP 422, and Duo verification failed
outright — meaning dual-model consensus was structurally impossible for Japanese
content, not merely unreliable.

Two rejected alternatives, and why:
  * DROP THE COUNT GUARD and zip anyway — that silently pairs word N of our
    transcript with an unrelated word N of theirs. It is the worst outcome
    available: every timing would be wrong and nothing would report it.
  * SEND ONE ALIGNMENT CALL PER TOKEN — correct in principle, but it multiplies a
    paid provider call by the word count (thousands per programme) and destroys
    the chunk-level acoustic context alignment depends on.

So the provider's tokenization is treated as WHAT IT ACTUALLY IS: an independent
segmentation of the same character stream. The reconciliation below is a pure,
deterministic mapping between the two segmentations, keyed on the characters
themselves — never on counts, never on positions.

THE CONTRACT
------------
The alignable characters of our tokens, concatenated in order, MUST equal the
alignable characters of the returned words, concatenated in order. That identity
is a STRICTER guarantee than the old per-token comparison it replaces: it proves
the provider aligned our transcript and nothing else. If the streams differ the
function raises and the run still fails closed — the failure mode is preserved,
only the false positive is removed.

Given that identity, each input token's window is derived from the returned words
that carry ITS characters:
  * one returned word per token (the space-delimited case) -> the window is that
    word's window, byte-for-byte identical to the previous behaviour, so no
    English/European run changes at all;
  * several returned words per token (a token the provider split) -> first start
    to last end, which is measured, not estimated;
  * one returned word spanning several tokens (tokens the provider merged) -> the
    shared window is subdivided in proportion to each token's character count.
    That is a DERIVED value, so it is disclosed per word as
    `token_window_estimated` and counted on the response, because a derived
    timing that is indistinguishable from a measured one is exactly the kind of
    silent fiction this pipeline refuses to ship.

Windows are clamped non-decreasing, so a reconciled timeline can never hand the
downstream normalizer a regression it would have to repair.
"""

# Tokens carrying no alignable character (a standalone Japanese "。", a stray
# dash) have no speech of their own. They are placed at the seam of their
# neighbours with this nominal width rather than being dropped, because dropping
# an input token would break the 1:1 output contract the caller depends on.
PUNCTUATION_WINDOW_SECONDS = 0.001


def alignable_chars(text: str) -> str:
    """The characters forced alignment can actually anchor on.

    Deliberately identical in spirit to app.normalized_token (casefolded
    alphanumerics only): punctuation and spacing are presentation, and the two
    tokenizers are free to disagree about them without that being an error.
    """
    return "".join(ch.casefold() for ch in str(text or "") if ch.isalnum())


def _mismatch_detail(expected_stream: str, returned_stream: str) -> str:
    limit = min(len(expected_stream), len(returned_stream))
    position = next((i for i in range(limit) if expected_stream[i] != returned_stream[i]), limit)
    return (
        f"Alignment character-stream mismatch at character {position}: "
        f"expected {len(expected_stream)} alignable characters, received {len(returned_stream)}"
    )


def reconcile_returned_words(expected_texts: list[str], returned_words: list[dict]) -> list[dict]:
    """Map provider-returned windows onto our input tokens, one window per token.

    `returned_words` items carry `text`, `start`, `end` (seconds, provider units).
    Returns one dict per entry of `expected_texts`: {start, end, estimated,
    returned_word_count}. Raises ValueError when the two character streams are not
    the same stream — the fail-closed path.
    """
    expected_chars = [alignable_chars(text) for text in expected_texts]
    carriers = []
    for word in returned_words:
        chars = alignable_chars(word.get("text"))
        if not chars:
            # A punctuation-only returned word anchors nothing; its time belongs to
            # the neighbouring speech and is absorbed by the windows around it.
            continue
        carriers.append({
            "chars": chars,
            "start": float(word["start"]),
            "end": float(word["end"]),
        })

    expected_stream = "".join(expected_chars)
    returned_stream = "".join(carrier["chars"] for carrier in carriers)
    if expected_stream != returned_stream:
        raise ValueError(_mismatch_detail(expected_stream, returned_stream))
    if not expected_stream:
        raise ValueError("Alignment chunk carries no alignable characters")

    windows: list[dict] = []
    carrier_index = 0
    char_offset = 0
    cursor = carriers[0]["start"]

    for chars in expected_chars:
        if not chars:
            windows.append({
                "start": cursor,
                "end": cursor + PUNCTUATION_WINDOW_SECONDS,
                "estimated": True,
                "returned_word_count": 0,
            })
            continue

        remaining = len(chars)
        start = None
        end = None
        estimated = False
        touched = 0
        while remaining > 0:
            carrier = carriers[carrier_index]
            available = len(carrier["chars"]) - char_offset
            consumed = min(remaining, available)
            span = carrier["end"] - carrier["start"]
            total = len(carrier["chars"])
            partial = consumed < total
            # Proportional subdivision of a SHARED returned word. Only reached when
            # the provider merged several of our tokens into one of its own.
            portion_start = carrier["start"] + (span * (char_offset / total))
            portion_end = carrier["start"] + (span * ((char_offset + consumed) / total))
            if start is None:
                start = portion_start
            end = portion_end
            estimated = estimated or partial
            touched += 1
            remaining -= consumed
            char_offset += consumed
            if char_offset >= total:
                carrier_index += 1
                char_offset = 0

        start = max(start, cursor)
        end = max(end, start + PUNCTUATION_WINDOW_SECONDS)
        cursor = end
        windows.append({
            "start": start,
            "end": end,
            "estimated": estimated,
            "returned_word_count": touched,
        })

    return windows
