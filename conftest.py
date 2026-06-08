"""Root conftest.py — ensures src/ is on sys.path before test collection."""

from __future__ import annotations

import sys
from pathlib import Path

# NOTE: Insert src/ at the front of sys.path so that the installed
# hybrid_doc_parser package (from src/) takes precedence over the
# tests/hybrid_doc_parser/ namespace package that pytest would otherwise
# discover first, causing ModuleNotFoundError for sub-modules like .models.
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
