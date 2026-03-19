from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import OptionKind


def default_semantic_priors_path() -> Path:
    return Path(__file__).with_name("semantic_priors.toml")


@lru_cache(maxsize=1)
def load_semantic_priors(path: str | Path | None = None) -> dict[str, Any]:
    payload_path = Path(path) if path is not None else default_semantic_priors_path()
    return tomllib.loads(payload_path.read_text(encoding="utf-8"))


def option_kind_priority_map(section_path: tuple[str, ...]) -> dict[OptionKind, int]:
    payload: Any = load_semantic_priors()
    for key in section_path:
        payload = payload[key]
    return {OptionKind[name]: int(priority) for name, priority in payload.items()}


def action_quality_config() -> dict[str, Any]:
    return dict(load_semantic_priors()["action_quality"])
