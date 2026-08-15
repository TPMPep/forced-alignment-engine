"""Locks the engine's plausibility constants against the worker's audit stage.

The engine decides how much audio to search for a word; the worker's
timeline-integrity stage decides whether the resulting window is a physically
possible utterance. If those two disagree about "possible", the engine can expand
a region to place words the audit then rejects — a silent, self-inflicted defect
loop. This test reads the TypeScript source directly, so a drift is a failing
test rather than a production surprise.

Skipped when the worker source is not present (the engine also ships as its own
standalone repo, where the mirror is unavailable).
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import plausibility  # noqa: E402

WORKER_SOURCE = Path(__file__).resolve().parents[2] / "bullmq-worker" / "src" / "timeline-integrity.ts"


def ts_number(source: str, name: str) -> float:
    match = re.search(rf"export const {name}\s*=\s*([0-9./*\s]+);", source)
    assert match, f"{name} not found in timeline-integrity.ts"
    return float(eval(match.group(1).strip()))  # noqa: S307 - numeric literal from our own source


@pytest.fixture(scope="module")
def worker_source() -> str:
    if not WORKER_SOURCE.exists():
        pytest.skip("worker timeline-integrity.ts is not present in this checkout")
    return WORKER_SOURCE.read_text()


def test_word_ms_per_char_matches_worker(worker_source):
    assert plausibility.WORD_MS_PER_CHAR == ts_number(worker_source, "WORD_MS_PER_CHAR")


def test_safety_factor_matches_worker(worker_source):
    assert plausibility.WORD_DURATION_SAFETY_FACTOR == ts_number(worker_source, "WORD_DURATION_SAFETY_FACTOR")


def test_duration_floor_matches_worker(worker_source):
    assert plausibility.WORD_MAX_DURATION_FLOOR_MS == ts_number(worker_source, "WORD_MAX_DURATION_FLOOR_MS")


def test_min_plausible_word_matches_worker(worker_source):
    assert plausibility.MIN_PLAUSIBLE_WORD_MS == ts_number(worker_source, "MIN_PLAUSIBLE_WORD_MS")


def test_max_word_duration_matches_worker_formula():
    # Same worked examples the worker documents: a long word gets real headroom,
    # a short one is held to the floor.
    assert plausibility.max_word_duration_ms("anticonstitutionnellement") == pytest.approx(4464, abs=1)
    assert plausibility.max_word_duration_ms("a") == 1500
