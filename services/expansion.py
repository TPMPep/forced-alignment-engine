"""Adaptive search-window expansion — removes the ASR segment boundary as a HARD
limit on forced alignment.

═══ WHY THIS EXISTS ════════════════════════════════════════════════════════════
The timing chain was CIRCULAR. The ASR provider proposed each word's position,
the chunk's audio window was cut from those same provider positions (plus a
bounded pad), and forced alignment was then only allowed to search inside that
slice. So when the provider ended an utterance early, the words spoken after its
OUT point were never in the audio the aligner received — it could not possibly
place them correctly, and since it must account for every millisecond it was
given, it stacked them against the end of the clip.

GROUND TRUTH, project 6a7d874aa2ddd372f426a4df line 18 ("...a key role in
shaping his leadership."):
    provider last word end   93,664ms
    next chunk's speech      96,232ms   → measured gap 2,568ms
    pad taken                min(350, 1284) = 350ms
    audio slice ended        94,014ms
    aligned result           role/in/shaping/his/leadership. ALL at 93,986→93,987
The displacement ramp behind that wall (marshal +316ms, who +661, played +808,
a +890, key +944, then flat) is a saturation curve against a hard edge, not a
measurement. The words are really spoken out to ~96,200ms. No amount of
downstream repair can recover a position from audio that was never analysed.

═══ WHAT THIS MODULE DOES ══════════════════════════════════════════════════════
The provider window becomes an initial SEARCH HYPOTHESIS. Alignment starts
conservatively; if the result shows the search region was exhausted, the region
is expanded on evidence and alignment re-runs. Nothing is guessed: the expansion
only decides how much AUDIO the aligner may listen to, and the aligner then
measures where the words actually are inside it.

TWO INDEPENDENT SIGNALS ARE REQUIRED to declare exhaustion. A single signal is
not enough — legitimately fast speech, a genuinely clipped function word, or one
bad provider timestamp can each produce one signal on their own, and expanding on
that alone would hand the aligner silence it has no use for. The signals:
    stacked_at_edge      2+ words sharing one window at the boundary
    degenerate_duration  a word under the evidence floor at the boundary
    edge_pinned          the final word ends flush against the slice edge
    displacement_ramp    monotonically growing lateness into the boundary
    compressed_run       the boundary run occupies less audio than it needs
    low_confidence       supporting only; the aligner's own loss was poor
`low_confidence` can never be one of the two — it is corroboration, not evidence
of a boundary.

EXPANSION IS EVIDENCE-BOUNDED, NEVER "extend to the next segment". The ceiling is
independent of the provider's segmentation decision: it is the neighbouring
chunk's observed speech edge, held back by NEIGHBOUR_GUARD_MS so the expanded
window can never reach another utterance's audio. Within that ceiling, each pass
asks for only what the unplaced words need at conversational pace (with a margin,
doubling per pass) — so a two-word tail never opens a ten-second window. When the
ceiling gives no headroom, no pass is attempted: the condition is reported, and
the caller fails it safe rather than inventing timings.

Absorption protection is NOT weakened by this. Expansion is granted only at an
edge where exhaustion was proven, only up to what the words need, and every
resulting word still passes the same rate-aware plausibility checks used by the
downstream arbitration. An expanded edge that produces an implausible word is
reported as such, and the arbitration restores the safer evidence.
"""

from .plausibility import (
    MIN_PLAUSIBLE_WORD_MS,
    evidence_ceiling_ms,
    max_word_duration_ms,
    required_span_ms,
)

# A word ending within this distance of the slice edge is touching the boundary.
# Chosen below the evidence floor (40ms) so it cannot be satisfied by ordinary
# rounding of a genuine word end.
EDGE_PROXIMITY_MS = 30
# Words sharing a window this close together are one stack, not two measurements.
STACK_TOLERANCE_MS = 5
STACK_MIN_WORDS = 2
# How far back to look for a displacement ramp into the boundary.
RAMP_WINDOW_WORDS = 5
# A ramp only counts when it actually grows into something material.
RAMP_MIN_TOTAL_MS = 250
# The aligner's own confidence, as CORROBORATION only.
LOW_CONFIDENCE = 0.6
# Never place the expanded edge closer than this to a neighbour's observed speech.
NEIGHBOUR_GUARD_MS = 120
# Headroom below which an expansion pass is not worth an extra paid alignment call.
MIN_USEFUL_EXPANSION_MS = 150
# Margin over the words' plausible need, so the aligner has room to find the true
# edge rather than being pinned to a second artificial wall.
EXPANSION_NEED_MARGIN = 1.5
MAX_EXPANSION_PASSES = max(1, 3)


def _boundary_run(words: list[dict], edge_ms: float, trailing: bool) -> list[int]:
    """Indices of the contiguous run of words touching the given slice edge."""
    order = range(len(words) - 1, -1, -1) if trailing else range(len(words))
    run: list[int] = []
    for index in order:
        word = words[index]
        probe = float(word["end_ms"]) if trailing else float(word["start_ms"])
        if abs(probe - float(edge_ms)) <= EDGE_PROXIMITY_MS:
            run.append(index)
            continue
        # Also absorb neighbours stacked onto the same instant as the edge word.
        if run:
            anchor = words[run[-1]]
            same_window = (
                abs(float(word["start_ms"]) - float(anchor["start_ms"])) <= STACK_TOLERANCE_MS
                and abs(float(word["end_ms"]) - float(anchor["end_ms"])) <= STACK_TOLERANCE_MS
            )
            if same_window:
                run.append(index)
                continue
        break
    return sorted(run)


def detect_saturation(aligned: list[dict], window: dict, confidence: float) -> dict:
    """Evidence that alignment ran out of SEARCH REGION rather than out of speech.

    Returns one verdict per edge with the signals behind it, the words involved,
    and how much audio those words plausibly need. Pure and side-effect free, so
    the same evidence can be persisted for audit and asserted in tests.
    """
    edges: dict[str, dict] = {}
    for trailing in (False, True):
        side = "trail" if trailing else "lead"
        edge_ms = float(window["end_ms"] if trailing else window["start_ms"])
        run = _boundary_run(aligned, edge_ms, trailing)
        signals: list[str] = []
        if not run:
            edges[side] = {
                "exhausted": False,
                "signals": [],
                "word_keys": [],
                "required_span_ms": 0,
                "observed_span_ms": 0,
            }
            continue

        run_words = [aligned[index] for index in run]
        # A stack ANYWHERE inside the boundary run is evidence — it does not require
        # every word in the run to share the window. Demanding that suppressed the
        # signal precisely where the defect is most severe: on project
        # 6a7d874aa2ddd372f426a4df line 18 the word "key" ends 28ms from the slice
        # edge, inside EDGE_PROXIMITY_MS, so it legitimately joins the run — and the
        # five-word stack sitting behind it (all at 93,986→93,987) then went
        # unreported, because that one non-stacked member failed the all() test.
        # Exhaustion was still declared there on four other signals, so no verdict
        # was wrong; but on a shorter run this is the difference between finding the
        # second required signal and silently skipping a needed expansion.
        # This can never fire on a healthy run: it needs STACK_MIN_WORDS words whose
        # start AND end agree to within STACK_TOLERANCE_MS (5ms), which is a
        # degenerate aligner output by construction, not fast speech.
        if len(run) >= STACK_MIN_WORDS:
            stacked = any(
                sum(
                    1
                    for other in run_words
                    if abs(float(other["start_ms"]) - float(anchor["start_ms"])) <= STACK_TOLERANCE_MS
                    and abs(float(other["end_ms"]) - float(anchor["end_ms"])) <= STACK_TOLERANCE_MS
                )
                >= STACK_MIN_WORDS
                for anchor in run_words
            )
            if stacked:
                signals.append("stacked_at_edge")
        if any(float(w["end_ms"]) - float(w["start_ms"]) < MIN_PLAUSIBLE_WORD_MS for w in run_words):
            signals.append("degenerate_duration")
        signals.append("edge_pinned")

        # Displacement ramp: lateness (trailing) or earliness (leading) growing
        # monotonically toward the boundary is the fingerprint of a constraint.
        probe_index = run[0] if trailing else run[-1]
        history = (
            aligned[max(0, probe_index - RAMP_WINDOW_WORDS): probe_index + 1]
            if trailing
            else list(reversed(aligned[probe_index: probe_index + 1 + RAMP_WINDOW_WORDS]))
        )
        displacements = [
            (float(w["start_ms"]) - float(w["provider_start_ms"])) * (1 if trailing else -1)
            for w in history
        ]
        if len(displacements) >= 3:
            growing = all(b >= a for a, b in zip(displacements, displacements[1:]))
            if growing and displacements[-1] - displacements[0] >= RAMP_MIN_TOTAL_MS:
                signals.append("displacement_ramp")

        observed_span = max(float(w["end_ms"]) for w in run_words) - min(float(w["start_ms"]) for w in run_words)
        required_span = required_span_ms([str(w["text"]) for w in run_words])
        if observed_span < required_span:
            signals.append("compressed_run")
        if float(confidence) < LOW_CONFIDENCE:
            signals.append("low_confidence")

        # Two INDEPENDENT signals required; corroboration alone is never enough.
        independent = [s for s in signals if s != "low_confidence"]
        edges[side] = {
            "exhausted": len(independent) >= 2,
            "signals": signals,
            "word_keys": [str(w["key"]) for w in run_words],
            "required_span_ms": round(required_span),
            "observed_span_ms": round(observed_span),
        }
    return edges


def expansion_plan(window: dict, edges: dict, pass_number: int) -> dict:
    """How far each exhausted edge may move, bounded by NEIGHBOUR EVIDENCE.

    The ceiling comes from the adjacent chunk's observed speech edge, never from
    the provider's segment boundary, and never from a fixed number of seconds.
    Within it, the request is what the boundary words need at conversational pace
    plus a margin, doubled per pass so a first conservative attempt can grow
    without ever jumping straight to the ceiling.
    """
    plan = {
        "lead_expansion_ms": 0,
        "trail_expansion_ms": 0,
        "lead_ceiling_ms": 0,
        "trail_ceiling_ms": 0,
        "blocked": [],
    }
    growth = 2 ** max(0, pass_number - 1)

    for side in ("lead", "trail"):
        edge = edges.get(side) or {}
        if not edge.get("exhausted"):
            continue
        neighbour = window.get("previous_speech_end_ms") if side == "lead" else window.get("next_speech_start_ms")
        current = float(window["start_ms"] if side == "lead" else window["end_ms"])
        if neighbour is None:
            # Program edge: no neighbouring speech to collide with, so the only
            # bound is the media itself (clamped by the caller).
            ceiling = float("inf")
        elif side == "lead":
            ceiling = max(0.0, current - (float(neighbour) + NEIGHBOUR_GUARD_MS))
        else:
            ceiling = max(0.0, (float(neighbour) - NEIGHBOUR_GUARD_MS) - current)
        deficit = max(0.0, float(edge.get("required_span_ms", 0)) - float(edge.get("observed_span_ms", 0)))
        request = max(deficit, float(edge.get("required_span_ms", 0))) * EXPANSION_NEED_MARGIN * growth
        granted = min(request, ceiling)
        plan[f"{side}_ceiling_ms"] = None if ceiling == float("inf") else round(ceiling)
        if granted < MIN_USEFUL_EXPANSION_MS:
            plan["blocked"].append(f"{side}:no_headroom")
            continue
        plan[f"{side}_expansion_ms"] = round(granted)
    return plan


def unresolved_reasons(aligned: list[dict], window: dict, edges: dict) -> dict[str, str]:
    """Words with no credible placement after the final pass, each with its REASON.

    Fail-safe: these are never presented as valid timing. A word qualifies when it
    sits in an exhausted boundary run, or when its window is not a physically
    possible utterance of its own text — judged against the same evidence-relative
    ceiling the downstream arbitration uses, so the two stages cannot disagree
    about what "possible" means.

    The reason is carried per word because a count alone is unactionable: an
    operator handed "1 unresolved word" cannot tell which word failed or why.
    """
    reasons: dict[str, str] = {}
    for side in ("lead", "trail"):
        edge = edges.get(side) or {}
        if edge.get("exhausted"):
            for key in edge.get("word_keys") or []:
                reasons[str(key)] = f"search_region_exhausted_at_{side}_edge"
    for word in aligned:
        key = str(word["key"])
        duration = float(word["end_ms"]) - float(word["start_ms"])
        provider_duration = None
        if word.get("provider_start_ms") is not None and word.get("provider_end_ms") is not None:
            provider_duration = float(word["provider_end_ms"]) - float(word["provider_start_ms"])
        if duration < MIN_PLAUSIBLE_WORD_MS:
            reasons.setdefault(key, "aligned_window_below_evidence_floor")
        elif duration > evidence_ceiling_ms(str(word["text"]), provider_duration):
            reasons.setdefault(
                key,
                "aligned_window_inflated_beyond_evidence"
                if provider_duration is not None
                else "aligned_window_exceeds_rate_ceiling",
            )
        elif duration > max_word_duration_ms(str(word["text"])):
            reasons.setdefault(key, "aligned_window_exceeds_rate_ceiling")
    return reasons


def unresolved_words(aligned: list[dict], window: dict, edges: dict) -> list[str]:
    """Keys only, for callers that do not need the reasons."""
    return sorted(unresolved_reasons(aligned, window, edges).keys())
