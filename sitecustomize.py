from __future__ import annotations

import sys
from pathlib import Path


def _ensure_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


_ROOT = Path(__file__).resolve().parent
_PROJECT = _ROOT / "research" / "world_model"
_SRC = _PROJECT / "src"

if _PROJECT.is_dir():
    _ensure_path(_PROJECT)
if _SRC.is_dir():
    _ensure_path(_SRC)
