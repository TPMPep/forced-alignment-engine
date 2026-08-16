# =============================================================================
# conftest.py — pytest import-path root for the forced-alignment engine.
# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS: the policy tests import BOTH `app` (app.py, at the repo
# root) and `services.*`. Under pytest's default import mode, only the test
# file's own directory (tests/) lands on sys.path — so `from app import ...`
# raises ModuleNotFoundError and pytest aborts during COLLECTION with exit
# code 2, meaning not one test runs. That is exactly how the first CI run
# failed: 27 tests collected, 1 collection error, whole suite reported red.
#
# Putting the repo root on sys.path here fixes it once, for every current and
# future test file, instead of each test carrying its own sys.path shim (which
# is what the other suites do today and is precisely the drift that let this
# gap sit unnoticed). Placing it at the root also makes pytest treat the root
# as rootdir unambiguously, so `pytest tests -v` behaves identically locally,
# in CI, and inside the Docker image.
# =============================================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
