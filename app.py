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

BUILD_TAG = "2026-08-14-elevenlabs-forced-alignment-v4-distribution-gate"
PROVIDER = "elevenlabs_forced_alignment"
PROVIDER_API = "https://api.elevenlabs.io/v1/forced-alignment"
MAX_WORDS = int(os.getenv("ALIGNMENT_MAX_WORDS", "50000"))
MAX_AUDIO_BYTES = int(os.getenv("ALIGNMENT_MAX_AUDIO_BYTES", str(8 * 1024**3)))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("ALIGNMENT_DOWNLOAD_TIMEOUT_SECONDS", "1800"))
REQUEST_CONCURRENCY = max(1, int(os.getenv("ALIGNMENT_MAX_CONCURRENCY", "1")))
CHUNK_SECONDS = min(290, max(30, int(os.getenv("ALIGNMENT_CHUNK_SECONDS", "240"))))
CHUNK_PADDING_SECONDS = min(15, max(2, int(os.getenv("ALIGNMENT_CHUNK_PADDING_SECONDS", "5"))))
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
        if current and (proposed_duration > CHUNK_SECONDS or proposed_chars > CHUNK_MAX_CHARS):
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
                aligned: list[dict] = []
                losses: list[float] = []
                timeout = httpx.Timeout(600, connect=30)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    for index, group in enumerate(groups):
                        start_ms = max(0, int(min(word.provider_start_ms for word in group) - CHUNK_PADDING_SECONDS * 1000))
                        end_ms = int(max(word.provider_end_ms for word in group) + CHUNK_PADDING_SECONDS * 1000)
                        chunk_path = Path(tmp) / f"chunk-{index:05d}.flac"
                        await asyncio.to_thread(extract_chunk, source, chunk_path, start_ms / 1000, max(1.0, (end_ms - start_ms) / 1000))
                        chunk_aligned, loss = await align_chunk(client, chunk_path, group, start_ms)
                        aligned.extend(chunk_aligned)
                        losses.append(loss)
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
