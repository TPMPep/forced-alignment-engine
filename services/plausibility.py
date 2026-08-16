"""Rate-aware plausibility of a single word's timing window.

ONE SOURCE OF TRUTH, MIRRORED DOWNSTREAM. These constants are the same ones the
worker's timeline-integrity stage uses (bullmq-worker/src/timeline-integrity.ts:
WORD_MS_PER_CHAR, WORD_DURATION_SAFETY_FACTOR, WORD_MAX_DURATION_FLOOR_MS,
MIN_PLAUSIBLE_WORD_MS) and that lib/segment-shaping.js uses for the mirror-image
provider-padding defect. A drift between them means the engine would expand a
search window to place words the downstream audit then rejects, or vice versa —
two stages disagreeing about what "physically possible speech" means. The parity
test tests/test_plausibility_parity.py locks the numbers against the TypeScript.

14 characters per second is the conversational pace the translation pipeline
assumes. The 2.5x safety factor means only clearly pathological spans are judged
impossible: "anticonstitutionnellement" (25 chars) is allowed ~4.4s, while a
155-second single word is not.

The MINIMUM is a floor on EVIDENCE, not on word length. Observed on a real
734-word run the 5th-percentile aligned word was 41ms; collapsed words come back
at 1ms and some provider captures come back zero-width. A window under the floor
describes no speech at all, so it cannot be used as evidence of placement.
"""

WORD_MS_PER_CHAR = 1000 / 14
WORD_DURATION_SAFETY_FACTOR = 2.5
WORD_MAX_DURATION_FLOOR_MS = 1500
MIN_PLAUSIBLE_WORD_MS = 40
# Typical inter-word gap at conversational pace. Used only to size how much audio
# a run of unplaced words plausibly NEEDS — never to assert where a word is.
TYPICAL_INTER_WORD_GAP_MS = 80


def visible_chars(text: str) -> int:
    return len("".join(ch for ch in str(text or "") if not ch.isspace())) or 1


def max_word_duration_ms(text: str) -> float:
    """Realistic maximum spoken duration for one word, given its text."""
    return max(
        float(WORD_MAX_DURATION_FLOOR_MS),
        round(visible_chars(text) * WORD_MS_PER_CHAR * WORD_DURATION_SAFETY_FACTOR),
    )


def typical_word_duration_ms(text: str) -> float:
    """Duration this word would occupy at conversational pace.

    This is the sizing estimate for expansion — how much audio a run of words
    needs in order to be placeable at all. It is deliberately NOT a claim about
    where the word is; the aligner still measures that against the waveform.
    """
    return visible_chars(text) * WORD_MS_PER_CHAR


def provider_capture_credible(text: str, provider_duration_ms: float | None) -> bool:
    """Is the PROVIDER's measured duration credible as a reading of this text?

    Mirrors providerCaptureCredible in bullmq-worker/src/timeline-integrity.ts. A
    capture more than WORD_DURATION_SAFETY_FACTOR times faster than conversational
    pace is not a measurement of that word — it is a compressed or collapsed
    timestamp, which is precisely the case the aligner is expected to correct. The
    same safety factor bounds both directions so there is one documented tolerance.
    """
    if provider_duration_ms is None:
        return False
    try:
        duration = float(provider_duration_ms)
    except (TypeError, ValueError):
        return False
    if duration < MIN_PLAUSIBLE_WORD_MS:
        return False
    return duration >= typical_word_duration_ms(text) / WORD_DURATION_SAFETY_FACTOR


def evidence_ceiling_ms(text: str, provider_duration_ms: float | None) -> float:
    """Upper bound an ALIGNED window must respect given this word's evidence.

    Mirrors evidenceCeilingMs in the worker. The generic rate ceiling alone cannot
    detect absorption, because its floor grants every short word a 1,500ms
    allowance; a credible provider capture supplies the independent, word-specific
    bound. Degrades to the rate ceiling whenever no credible capture exists, so a
    too-short capture can never stop the aligner from correcting it.
    """
    rate_ceiling = max_word_duration_ms(text)
    if not provider_capture_credible(text, provider_duration_ms):
        return rate_ceiling
    return min(rate_ceiling, round(float(provider_duration_ms) * WORD_DURATION_SAFETY_FACTOR + MIN_PLAUSIBLE_WORD_MS))


def near_zero_corroborated(provider_duration_ms: float | None) -> bool:
    """Is a sub-floor aligned window corroborated as a genuinely brief word?

    Mirrors nearZeroCorroborated in the worker. A sub-floor duration is evidence
    of real (clipped) speech only when the other timeline independently reports a
    comparably brief word.
    """
    if provider_duration_ms is None:
        return False
    try:
        duration = float(provider_duration_ms)
    except (TypeError, ValueError):
        return False
    return 0 < duration <= MIN_PLAUSIBLE_WORD_MS * WORD_DURATION_SAFETY_FACTOR


def window_plausible(text: str, start_ms: float, end_ms: float, provider_duration_ms: float | None = None) -> bool:
    """Is this window a physically possible utterance of `text`?

    `provider_duration_ms` supplies the evidence-relative upper bound. Pass None
    when judging the provider's own window — a measurement cannot bound itself.
    """
    try:
        duration = float(end_ms) - float(start_ms)
    except (TypeError, ValueError):
        return False
    return MIN_PLAUSIBLE_WORD_MS <= duration <= evidence_ceiling_ms(text, provider_duration_ms)


def required_span_ms(texts: list[str]) -> float:
    """Audio a contiguous run of words needs to be placeable at conversational pace."""
    if not texts:
        return 0.0
    speech = sum(typical_word_duration_ms(text) for text in texts)
    gaps = TYPICAL_INTER_WORD_GAP_MS * max(0, len(texts) - 1)
    return speech + gaps
