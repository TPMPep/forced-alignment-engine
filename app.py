import asyncio
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from services.expansion import (
    MAX_EXPANSION_PASSES,
    NEIGHBOUR_GUARD_MS,
    detect_saturation,
    expansion_plan,
    unresolved_reasons,
)

BUILD_TAG = "2026-08-16-elevenlabs-forced-alignment-v9-stack-detection-anywhere-in-run"
# Policy generation of the ADAPTIVE SEARCH EXPANSION behaviour, pinned onto every
# response so a delivered alignment is never reinterpreted under later rules.
# v1 = provider window is a hard limit (circular timing: the ASR proposes the
#      window, the aligner may only search inside it).
# v2 = provider window is an initial HYPOTHESIS; an exhausted search region is
#      expanded on multi-signal evidence, bounded by neighbouring speech, and a
#      region that still cannot place its words is reported UNRESOLVED rather
#      than accepted.
# v3 = PER-WORD attribution. Every returned word carries the chunk it came from,
#      the pass that produced it, and the expansion granted to each edge of its
#      search region, plus a machine-readable reason on every unresolved word.
#      Run-level totals alone forced the consumer to INFER which words an
#      expansion affected, and an inferred attribution is not evidence — it is a
#      guess that happens to be checkable. Unresolved words are also judged
#      against the same evidence-relative ceiling the downstream arbitration
#      uses, so the two stages cannot disagree about what "possible" means.
# v4 = the stacked_at_edge signal is detected ANYWHERE inside a boundary run, not
#      only when EVERY word in the run shares one window. The old rule silently
#      suppressed the strongest, most diagnostic saturation signal whenever a
#      neighbouring word merely TOUCHED the slice edge and joined the run: on
#      project 6a7d874aa2ddd372f426a4df line 18 the word "key" ends 28ms from the
#      edge — inside EDGE_PROXIMITY_MS — so the five-word stack behind it (all at
#      93,986→93,987) was never reported. Exhaustion was still declared there on
#      four other signals, so no shipped verdict was wrong; but on a shorter run
#      the lost signal is the difference between reaching the two required signals
#      and silently skipping an expansion the words genuinely needed. Pinned as
#      its own generation because a run delivered under v3 reported a DIFFERENT
#      signal set for identical audio, and an auditor must read the evidence
#      against the rules that produced it.
EXPANSION_POLICY_VERSION = 4
PROVIDER = "elevenlabs_forced_alignment"
PROVIDER_API = "https://api.elevenlabs.io/v1/forced-alignment"
MAX_WORDS = int(os.getenv("ALIGNMENT_MAX_WORDS", "50000"))
MAX_AUDIO_BYTES = int(os.getenv("ALIGNMENT_MAX_AUDIO_BYTES", str(8 * 1024**3)))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("ALIGNMENT_DOWNLOAD_TIMEOUT_SECONDS", "1800"))
REQUEST_CONCURRENCY = max(1, int(os.getenv("ALIGNMENT_MAX_CONCURRENCY", "1")))
CHUNK_SECONDS = min(290, max(30, int(os.getenv("ALIGNMENT_CHUNK_SECONDS", "240"))))
# ─── Edge padding: the last unaccounted audio in the pipeline ───────────────────
# Padding exists ONLY to avoid clipping a word at a chunk edge. But padding is BY
# DEFINITION audio with no transcript, and forced alignment must account for every
# millisecond it is given — so the aligner assigns the padding to the edge word.
# That is not a theory. Ground truth, project 6a7d874aa2ddd372f426a4df row 17: the
# provider measured speech ending at 93,664ms, the chunk therefore ended at
# 93,664 + 2,000 = 95,664, and the final word "leadership." was aligned ending at
# 95,639 — it swallowed the ENTIRE trailing pad, to within 25ms. Every observed
# divergence on that program equalled the fixed pad almost exactly. A fixed pad is
# a fixed absorption budget handed to the aligner on every chunk edge.
#
# The pad cannot simply be removed: at an edge, the provider's own word boundary
# carries measurement error, and a zero pad would clip the word's onset or offset.
# So the pad is DERIVED FROM EVIDENCE instead of assumed: each edge is padded by at
# most HALF the silence the provider actually measured at that edge, capped at
# EDGE_PADDING_MS. Because chunk edges are cut at gaps wider than
# CHUNK_MAX_SPEECH_GAP_MS, there is provably silence to spend, and the pad can
# never reach into the neighbouring utterance's audio — so an edge word can never
# absorb another word's speech.
#
# WHY 350ms IS THE CAP, AND WHY IT IS THE WHOLE POINT: the maximum absorption an
# edge word can now suffer is EDGE_PADDING_MS. At 350ms that is below the 650ms
# breath boundary in lib/segment-shaping.js and far below the 1,500ms divergence
# trust threshold in the worker's timeline-integrity audit. Edge absorption is
# therefore structurally incapable of creating a spurious line break or of
# reassigning a word to the wrong speaker — the two consequential failures this
# whole control chain exists to prevent. It stops being a defect we detect and
# repair downstream and becomes one we cannot produce.
#
# The floor keeps a usable pad when a gap is small or when segment streams overlap
# (a negative measured gap), where clipping risk outweighs a sub-breath absorption.
EDGE_PADDING_MS = min(2_000, max(50, int(os.getenv("ALIGNMENT_EDGE_PADDING_MS", "350"))))
EDGE_PADDING_FLOOR_MS = min(EDGE_PADDING_MS, max(25, int(os.getenv("ALIGNMENT_EDGE_PADDING_FLOOR_MS", "100"))))
# ─── The absorption fix (root cause, not a repair) ──────────────────────────────
# Forced alignment must account for EVERY millisecond of audio it is given. If a
# chunk's audio window spans a stretch with no transcript — music, atmosphere,
# unintelligible overlap — the aligner has nowhere to put that time and stuffs it
# into the neighbouring words, either inside one word or across the gaps between
# them. Ground truth (project 6a6c561ef670f3992db756d0): "¿Aita, estás bien?" —
# 670ms of speech in the provider capture — was aligned across 28,200ms, because
# its chunk spanned a long non-dialogue stretch.
#
# Chunking previously split ONLY on accumulated duration (240s) or characters, so
# a single chunk routinely straddled minutes of silence. Splitting additionally at
# any provider-measured gap too long to be speech means every chunk's audio window
# tightly bounds real dialogue, and there is no untranscribed audio inside it to
# absorb. The defect becomes structurally impossible rather than something we
# detect and repair afterwards.
#
# 2000ms is well clear of natural speech rhythm (the segment shaper treats 650ms as
# a breath and 300ms as a pause), so this never fractures a continuous utterance —
# it only cuts where the provider itself measured a non-speech span. Residual
# absorption is bounded by the padding above.
CHUNK_MAX_SPEECH_GAP_MS = max(500, int(os.getenv("ALIGNMENT_CHUNK_MAX_SPEECH_GAP_MS", "2000")))
CHUNK_MAX_CHARS = max(1000, int(os.getenv("ALIGNMENT_CHUNK_MAX_CHARS", "18000")))
MAX_REGRESSION_MS = max(250, int(os.getenv("ALIGNMENT_MAX_REGRESSION_MS", "10000")))
MAX_REPAIR_RATIO = min(0.05, max(0.001, float(os.getenv("ALIGNMENT_MAX_REPAIR_RATIO", "0.01"))))
# Per-word tolerance used to CLASSIFY a provider disagreement as an outlier.
# It is a measurement threshold only — the release decision lives in the worker
# gate, which fails closed on SYSTEMIC drift (p99 + outlier ratio) instead of a
# single worst word. A 45-minute program holds ~8,000 words; one provider
# timestamp outlier must never veto an otherwise fully verified alignment.
OUTLIER_TOLERANCE_MS = max(1_000, int(os.getenv("ALIGNMENT_OUTLIER_TOLERANCE_MS", "30000")))
MAX_OUTLIER_SAMPLE = 25
SHARED_SECRET = os.getenv("ALIGNMENT_SHARED_SECRET", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
SUPPORTED_LANGUAGES = {"en", "ja", "zh", "de", "hi", "fr", "ko", "pt", "it", "es", "id", "nl", "tr", "fil", "pl", "sv", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms", "sk", "da", "ta", "uk", "ru"}
semaphore = asyncio.Semaphore(REQUEST_CONCURRENCY)


class InputWord(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500)
    provider_start_ms: float = Field(ge=0)
    provider_end_ms: float = Field(ge=0)


class AlignmentRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=200)
    audio_url: str = Field(min_length=8)
    language_code: str = Field(min_length=2, max_length=8)
    words: list[InputWord] = Field(min_length=1)


def normalized_token(value: str) -> str:
    return "".join(ch.casefold() for ch in value if ch.isalnum())


async def download_audio(url: str, target: Path) -> tuple[int, str]:
    total = 0
    digest = hashlib.sha256()
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS, connect=30)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > MAX_AUDIO_BYTES:
                raise ValueError("Audio exceeds the configured size limit")
            with target.open("wb") as output:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise ValueError("Audio exceeds the configured size limit")
                    digest.update(chunk)
                    output.write(chunk)
    if total == 0:
        raise ValueError("Downloaded audio is empty")
    return total, digest.hexdigest()


def chunk_words(words: list[InputWord]) -> list[list[InputWord]]:
    chunks: list[list[InputWord]] = []
    current: list[InputWord] = []
    chars = 0
    current_end_ms = 0.0
    for word in words:
        proposed_chars = chars + len(word.text) + 1
        proposed_end_ms = max(current_end_ms, word.provider_end_ms)
        proposed_duration = 0 if not current else (proposed_end_ms - current[0].provider_start_ms) / 1000
        # Gap to the previous word, as MEASURED BY THE PROVIDER. A gap this long is
        # not speech rhythm — it is audio with no transcript, which is exactly what
        # the aligner would otherwise absorb into a neighbouring word.
        speech_gap_ms = 0.0 if not current else (word.provider_start_ms - current_end_ms)
        if current and (proposed_duration > CHUNK_SECONDS or proposed_chars > CHUNK_MAX_CHARS or speech_gap_ms > CHUNK_MAX_SPEECH_GAP_MS):
            chunks.append(current)
            current = []
            chars = 0
            current_end_ms = 0.0
        current.append(word)
        chars += len(word.text) + 1
        current_end_ms = max(current_end_ms, word.provider_end_ms)
    if current:
        chunks.append(current)
    return chunks


def chunk_windows(groups: list[list[InputWord]]) -> list[dict]:
    """Audio window per chunk, with each edge pad bounded by MEASURED silence.

    A chunk's window is the group's provider-measured speech span plus a pad on
    each side. The pad is the minimum of EDGE_PADDING_MS and half the provider-
    measured gap to the adjacent chunk, floored at EDGE_PADDING_FLOOR_MS. Taking
    half guarantees two adjacent chunks can never pad into the same silence and
    never into each other's speech, so the audio an edge word could absorb is
    always genuine non-speech and always bounded. At the program's first and last
    edge there is no adjacent word to measure against, so the cap is used.

    Returned per chunk for audit: the window, both pads, and both measured gaps —
    so an auditor can reconstruct exactly how much unaccounted audio each edge
    word was exposed to, without re-deriving it from the word timings.
    """
    windows: list[dict] = []
    for index, group in enumerate(groups):
        speech_start = min(word.provider_start_ms for word in group)
        speech_end = max(word.provider_end_ms for word in group)
        previous_end = max(word.provider_end_ms for word in groups[index - 1]) if index > 0 else None
        next_start = min(word.provider_start_ms for word in groups[index + 1]) if index + 1 < len(groups) else None
        lead_gap_ms = None if previous_end is None else (speech_start - previous_end)
        trail_gap_ms = None if next_start is None else (next_start - speech_end)
        lead_pad_ms = float(EDGE_PADDING_MS) if lead_gap_ms is None else max(EDGE_PADDING_FLOOR_MS, min(EDGE_PADDING_MS, lead_gap_ms / 2))
        trail_pad_ms = float(EDGE_PADDING_MS) if trail_gap_ms is None else max(EDGE_PADDING_FLOOR_MS, min(EDGE_PADDING_MS, trail_gap_ms / 2))
        windows.append({
            "start_ms": max(0.0, speech_start - lead_pad_ms),
            "end_ms": speech_end + trail_pad_ms,
            "speech_start_ms": speech_start,
            "speech_end_ms": speech_end,
            "lead_pad_ms": round(lead_pad_ms),
            "trail_pad_ms": round(trail_pad_ms),
            "lead_gap_ms": None if lead_gap_ms is None else round(lead_gap_ms),
            "trail_gap_ms": None if trail_gap_ms is None else round(trail_gap_ms),
            # Neighbouring OBSERVED speech edges. These are the ceiling on adaptive
            # expansion — independent of where the ASR provider chose to end a
            # segment, and the guarantee that an expanded window can never reach
            # into another utterance's audio.
            "previous_speech_end_ms": None if previous_end is None else round(previous_end),
            "next_speech_start_ms": None if next_start is None else round(next_start),
            "word_count": len(group),
        })
    return windows


def normalize_aligned_timeline(words: list[dict]) -> tuple[list[dict], int, int]:
    normalized: list[dict] = []
    previous_start_by_stream: dict[str, int] = {}
    repair_count = 0
    max_regression_ms = 0
    repair_limit = max(2, int(len(words) * MAX_REPAIR_RATIO))
    for source in words:
        word = dict(source)
        stream_key = word["key"].rsplit(":", 1)[0]
        previous_start = previous_start_by_stream.get(stream_key, -1)
        if word["start_ms"] < previous_start:
            regression_ms = previous_start - word["start_ms"]
            if regression_ms > MAX_REGRESSION_MS:
                raise ValueError(f"Alignment regression exceeds safety bound at {word['key']}: {regression_ms}ms")
            repair_count += 1
            if repair_count > repair_limit:
                raise ValueError(f"Alignment repair ratio exceeds safety bound at {word['key']}")
            duration_ms = max(1, word["end_ms"] - word["start_ms"])
            word["raw_start_ms"] = word["start_ms"]
            word["raw_end_ms"] = word["end_ms"]
            word["start_ms"] = previous_start
            word["end_ms"] = previous_start + duration_ms
            word["confidence"] = 0.0
            word["timing_repaired"] = True
            max_regression_ms = max(max_regression_ms, regression_ms)
        previous_start_by_stream[stream_key] = word["start_ms"]
        normalized.append(word)
    return normalized, repair_count, max_regression_ms


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def shift_distribution(aligned: list[dict]) -> dict:
    """Provider-vs-acoustic disagreement as a DISTRIBUTION, not a single max.

    Returns the max (kept verbatim as evidence), the median/p95/p99, the count
    and ratio of words beyond OUTLIER_TOLERANCE_MS, and a bounded sample of the
    worst offenders so an auditor can inspect exactly which words disagreed and
    by how much. Systemic drift moves p99; an isolated bad provider timestamp
    only moves max.
    """
    shifts: list[tuple[int, str]] = []
    for word in aligned:
        shift = max(
            abs(word["start_ms"] - word["provider_start_ms"]),
            abs(word["end_ms"] - word["provider_end_ms"]),
        )
        shifts.append((int(shift), str(word["key"])))
    values = sorted(value for value, _ in shifts)
    outliers = sorted((entry for entry in shifts if entry[0] > OUTLIER_TOLERANCE_MS), reverse=True)
    return {
        "max_provider_shift_ms": values[-1] if values else 0,
        "median_provider_shift_ms": percentile(values, 0.5),
        "p95_provider_shift_ms": percentile(values, 0.95),
        "p99_provider_shift_ms": percentile(values, 0.99),
        "outlier_tolerance_ms": OUTLIER_TOLERANCE_MS,
        "outlier_word_count": len(outliers),
        "outlier_ratio": (len(outliers) / len(values)) if values else 0.0,
        "outlier_sample": [{"key": key, "shift_ms": value} for value, key in outliers[:MAX_OUTLIER_SAMPLE]],
    }


def extract_chunk(source: Path, target: Path, start_seconds: float, duration_seconds: float):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-ss", f"{start_seconds:.3f}", "-t", f"{duration_seconds:.3f}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=max(120, int(duration_seconds * 4)))
    if completed.returncode != 0 or not target.exists() or target.stat().st_size < 100:
        raise ValueError(f"Audio chunk extraction failed: {completed.stderr[-500:]}")


async def align_chunk(client: httpx.AsyncClient, audio_path: Path, words: list[InputWord], offset_ms: int) -> tuple[list[dict], float]:
    transcript = " ".join(word.text.strip() for word in words)
    response = None
    for attempt in range(1, 5):
        with audio_path.open("rb") as audio:
            response = await client.post(
                PROVIDER_API,
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files={"file": ("alignment.flac", audio, "audio/flac")},
                data={"text": transcript},
            )
        if response.status_code != 429 and response.status_code < 500:
            break
        if attempt == 4:
            raise RuntimeError(f"ElevenLabs forced alignment transient failure: HTTP {response.status_code}")
        await asyncio.sleep(min(20, 2 ** attempt))
    if response is None:
        raise RuntimeError("ElevenLabs forced alignment returned no response")
    if not response.is_success:
        raise ValueError(f"ElevenLabs forced alignment rejected chunk: {response.text[:500]}")
    body = response.json()
    raw_words = [item for item in body.get("words", []) if str(item.get("text", "")).strip()]
    if len(raw_words) != len(words):
        raise ValueError(f"Alignment word-count mismatch: expected {len(words)}, received {len(raw_words)}")
    loss = float(body.get("loss", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, 1.0 - loss))
    aligned = []
    for expected, actual in zip(words, raw_words):
        if normalized_token(expected.text) != normalized_token(str(actual.get("text", ""))):
            raise ValueError(f"Alignment token mismatch at {expected.key}")
        start_ms = offset_ms + round(float(actual["start"]) * 1000)
        end_ms = offset_ms + round(float(actual["end"]) * 1000)
        if end_ms <= start_ms:
            raise ValueError(f"Invalid alignment window at {expected.key}")
        aligned.append({
            "key": expected.key,
            "text": expected.text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": confidence,
            "provider_start_ms": expected.provider_start_ms,
            "provider_end_ms": expected.provider_end_ms,
        })
    return aligned, loss


async def align_group_adaptively(
    client: httpx.AsyncClient,
    source: Path,
    tmp: str,
    index: int,
    group: list[InputWord],
    window: dict,
) -> dict:
    """Align one chunk, expanding its SEARCH REGION while the region is exhausted.

    Pass 1 is the conservative provider-derived window — identical to the previous
    behaviour, so a chunk whose words all place plausibly costs exactly one paid
    alignment call and is bit-for-bit unchanged. Only a chunk carrying multi-signal
    evidence that it ran out of audio pays for another pass, and only up to the
    neighbouring-speech ceiling.

    Returns the final aligned words plus the full per-pass audit: the window used,
    the signals observed, what was granted, what was refused and why.
    """
    passes: list[dict] = []
    lead_extra = 0.0
    trail_extra = 0.0
    aligned: list[dict] = []
    loss = 0.0
    effective = dict(window)
    edges: dict = {}

    for pass_number in range(1, MAX_EXPANSION_PASSES + 1):
        start_ms = max(0.0, float(window["start_ms"]) - lead_extra)
        end_ms = float(window["end_ms"]) + trail_extra
        chunk_path = Path(tmp) / f"chunk-{index:05d}-p{pass_number}.flac"
        await asyncio.to_thread(
            extract_chunk, source, chunk_path, start_ms / 1000, max(1.0, (end_ms - start_ms) / 1000)
        )
        aligned, loss = await align_chunk(client, chunk_path, group, int(start_ms))
        confidence = max(0.0, min(1.0, 1.0 - loss))
        effective = {**window, "start_ms": start_ms, "end_ms": end_ms}
        edges = detect_saturation(aligned, effective, confidence)
        record = {
            "pass": pass_number,
            "window_start_ms": round(start_ms),
            "window_end_ms": round(end_ms),
            "lead_expansion_applied_ms": round(lead_extra),
            "trail_expansion_applied_ms": round(trail_extra),
            "confidence": round(confidence, 4),
            "lead": edges["lead"],
            "trail": edges["trail"],
        }
        passes.append(record)

        if not (edges["lead"]["exhausted"] or edges["trail"]["exhausted"]):
            break
        if pass_number == MAX_EXPANSION_PASSES:
            record["stopped"] = "pass_budget_exhausted"
            break
        plan = expansion_plan(effective, edges, pass_number)
        record["plan"] = plan
        if not plan["lead_expansion_ms"] and not plan["trail_expansion_ms"]:
            # Neighbouring speech leaves no room. Expanding further would steal
            # another utterance's audio, so the condition stands and the caller
            # reports it unresolved instead of inventing a placement.
            record["stopped"] = "neighbour_bound"
            break
        lead_extra += plan["lead_expansion_ms"]
        trail_extra += plan["trail_expansion_ms"]

    reasons = unresolved_reasons(aligned, effective, edges)
    exhausted_keys = set()
    for side in ("lead", "trail"):
        if edges.get(side, {}).get("exhausted"):
            exhausted_keys.update(edges[side].get("word_keys") or [])
    # PER-WORD ATTRIBUTION (policy v3). Stamped on EVERY word, resolved or not, so
    # a downstream consumer can attribute an expansion to the exact words it
    # affected instead of inferring it from chunk totals.
    for word in aligned:
        key = str(word["key"])
        word["chunk_index"] = index
        word["alignment_pass"] = len(passes)
        word["expansion_lead_ms"] = round(lead_extra)
        word["expansion_trail_ms"] = round(trail_extra)
        word["search_window_start_ms"] = round(float(effective["start_ms"]))
        word["search_window_end_ms"] = round(float(effective["end_ms"]))
        if key in exhausted_keys:
            word["search_window_exhausted"] = True
        if key in reasons:
            word["unresolved"] = True
            word["unresolved_reason"] = reasons[key]

    return {
        "aligned": aligned,
        "loss": loss,
        "passes": passes,
        "pass_count": len(passes),
        "lead_expansion_ms": round(lead_extra),
        "trail_expansion_ms": round(trail_extra),
        "final_window_start_ms": round(float(effective["start_ms"])),
        "final_window_end_ms": round(float(effective["end_ms"])),
        "unresolved_keys": sorted(reasons.keys()),
        "unresolved_reasons": reasons,
    }


def verify_cross_chunk_order(chunk_results: list[dict]) -> list[dict]:
    """No expanded chunk may claim audio the next chunk's words actually occupy.

    Expansion is bounded by the neighbour's PROVIDER-observed speech edge, which
    is itself only a hypothesis. This is the independent check against the
    neighbour's MEASURED result: if an expanded chunk's last word now ends after
    the next chunk's first word begins, the two disagree about the same audio, so
    both boundary words are marked unresolved rather than silently overlapping.
    """
    violations: list[dict] = []
    for index in range(len(chunk_results) - 1):
        current = chunk_results[index]["aligned"]
        following = chunk_results[index + 1]["aligned"]
        if not current or not following:
            continue
        current_end = max(float(word["end_ms"]) for word in current)
        next_start = min(float(word["start_ms"]) for word in following)
        if current_end <= next_start:
            continue
        overlap_ms = round(current_end - next_start)
        late = max(current, key=lambda word: float(word["end_ms"]))
        early = min(following, key=lambda word: float(word["start_ms"]))
        for word in (late, early):
            word["unresolved"] = True
            word["cross_chunk_overlap_ms"] = overlap_ms
        violations.append({
            "chunk_index": index,
            "overlap_ms": overlap_ms,
            "late_word_key": str(late["key"]),
            "early_word_key": str(early["key"]),
        })
    return violations


app = FastAPI(title="Media Creator Forced Alignment Engine")


@app.get("/health")
async def health():
    return {
        "ok": bool(SHARED_SECRET and ELEVENLABS_API_KEY),
        "ready": bool(SHARED_SECRET and ELEVENLABS_API_KEY),
        "build_tag": BUILD_TAG,
        "provider": PROVIDER,
        "max_concurrency": REQUEST_CONCURRENCY,
        "chunk_seconds": CHUNK_SECONDS,
        "chunk_max_speech_gap_ms": CHUNK_MAX_SPEECH_GAP_MS,
        "edge_padding_ms": EDGE_PADDING_MS,
        "edge_padding_floor_ms": EDGE_PADDING_FLOOR_MS,
        "expansion_policy_version": EXPANSION_POLICY_VERSION,
        "max_expansion_passes": MAX_EXPANSION_PASSES,
        "neighbour_guard_ms": NEIGHBOUR_GUARD_MS,
        "supported_language_count": len(SUPPORTED_LANGUAGES),
    }


@app.post("/align")
async def align(payload: AlignmentRequest, x_alignment_secret: str = Header(default="")):
    if not SHARED_SECRET or not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="Alignment service credentials are not configured")
    if not hmac.compare_digest(x_alignment_secret, SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if len(payload.words) > MAX_WORDS:
        raise HTTPException(status_code=413, detail="Word count exceeds the configured limit")
    language = payload.language_code.lower().replace("_", "-").split("-")[0]
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"Forced alignment is not commercially supported for language: {payload.language_code}")
    provider_cursor_by_segment: dict[str, float] = {}
    for word in payload.words:
        segment_key = word.key.rsplit(":", 1)[0]
        previous_provider_start = provider_cursor_by_segment.get(segment_key, -1.0)
        if word.provider_end_ms < word.provider_start_ms or word.provider_start_ms < previous_provider_start:
            raise HTTPException(status_code=422, detail=f"Invalid provider timeline at {word.key}")
        provider_cursor_by_segment[segment_key] = word.provider_start_ms
    started = time.monotonic()
    try:
        async with semaphore:
            with tempfile.TemporaryDirectory(prefix="media-align-") as tmp:
                source = Path(tmp) / "source_audio"
                audio_bytes, audio_sha256 = await download_audio(payload.audio_url, source)
                groups = chunk_words(payload.words)
                windows = chunk_windows(groups)
                aligned: list[dict] = []
                losses: list[float] = []
                chunk_results: list[dict] = []
                timeout = httpx.Timeout(600, connect=30)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    for index, group in enumerate(groups):
                        result = await align_group_adaptively(client, source, tmp, index, group, windows[index])
                        chunk_results.append(result)
                        losses.append(result["loss"])
                overlaps = verify_cross_chunk_order(chunk_results)
                for index, result in enumerate(chunk_results):
                    aligned.extend(result["aligned"])
                    windows[index] = {
                        **windows[index],
                        "final_start_ms": result["final_window_start_ms"],
                        "final_end_ms": result["final_window_end_ms"],
                        "lead_expansion_ms": result["lead_expansion_ms"],
                        "trail_expansion_ms": result["trail_expansion_ms"],
                        "pass_count": result["pass_count"],
                        "passes": result["passes"],
                    }
        if len(aligned) != len(payload.words):
            raise ValueError("Alignment result is incomplete")
        aligned, timing_repair_count, max_regression_ms = normalize_aligned_timeline(aligned)
        distribution = shift_distribution(aligned)
        mean_loss = sum(losses) / len(losses)
        response = {
            "ok": True,
            "verified": True,
            "request_id": payload.request_id,
            "provider": PROVIDER,
            "model": "forced-alignment-api",
            "model_revision": f"provider-managed-unversioned:{BUILD_TAG}",
            "language_code": payload.language_code,
            "audio_bytes": audio_bytes,
            "audio_sha256": audio_sha256,
            "word_count": len(aligned),
            "mean_confidence": max(0.0, min(1.0, 1.0 - mean_loss)),
            **distribution,
            "timing_repair_count": timing_repair_count,
            "max_regression_ms": max_regression_ms,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "chunk_count": len(groups),
            # The absorption budget this run actually granted. max_edge_padding_ms is
            # the ceiling on how much unaccounted audio ANY edge word could have taken,
            # so a reviewer can confirm from the response alone that it stayed below the
            # 650ms breath boundary and the 1,500ms divergence trust threshold.
            "edge_padding_ms": EDGE_PADDING_MS,
            "max_edge_padding_ms": max((max(w["lead_pad_ms"], w["trail_pad_ms"]) for w in windows), default=0),
            # ─── Adaptive search expansion evidence ──────────────────────────
            # The provider's segment window is now only the FIRST hypothesis. These
            # fields are the auditor's answer to "did the aligner have to look
            # beyond the transcriber's boundary, by how much, on what evidence, and
            # did anything remain unplaceable?" — without re-deriving it from the
            # word timings. unresolved_word_count > 0 means the run must NOT be
            # treated as fully verified timing: those words are surfaced for review
            # instead of being accepted at a slice edge.
            "expansion_policy_version": EXPANSION_POLICY_VERSION,
            "max_expansion_passes": MAX_EXPANSION_PASSES,
            "neighbour_guard_ms": NEIGHBOUR_GUARD_MS,
            "alignment_pass_count": sum(int(w.get("pass_count", 1)) for w in windows),
            "expanded_chunk_count": sum(
                1 for w in windows if (w.get("lead_expansion_ms") or 0) or (w.get("trail_expansion_ms") or 0)
            ),
            "total_expansion_ms": sum(
                int(w.get("lead_expansion_ms") or 0) + int(w.get("trail_expansion_ms") or 0) for w in windows
            ),
            "max_expansion_ms": max(
                (max(int(w.get("lead_expansion_ms") or 0), int(w.get("trail_expansion_ms") or 0)) for w in windows),
                default=0,
            ),
            "cross_chunk_overlaps": overlaps,
            "unresolved_word_count": sum(1 for word in aligned if word.get("unresolved")),
            "unresolved_sample": [
                {
                    "key": str(word["key"]),
                    "text": str(word["text"]),
                    "start_ms": word["start_ms"],
                    "end_ms": word["end_ms"],
                    "reason": word.get("unresolved_reason"),
                    "chunk_index": word.get("chunk_index"),
                    "alignment_pass": word.get("alignment_pass"),
                    "expansion_trail_ms": word.get("expansion_trail_ms"),
                }
                for word in aligned if word.get("unresolved")
            ][:MAX_OUTLIER_SAMPLE],
            "chunk_windows": windows,
            "request_hash": hashlib.sha256(json.dumps([word.model_dump() for word in payload.words], sort_keys=True).encode()).hexdigest(),
            "words": aligned,
        }
        return response
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, KeyError, subprocess.TimeoutExpired) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Alignment network failure: {error}") from error
