from __future__ import annotations

from typing import Any


def episode_step_count(episode: Any) -> int:
    metadata = dict(getattr(episode, "metadata", {}) or {})
    raw_count = metadata.get("cached_step_count", metadata.get("step_count"))
    if raw_count is not None:
        try:
            return max(0, int(raw_count))
        except (TypeError, ValueError):
            pass
    return len(getattr(episode, "steps", ()))
