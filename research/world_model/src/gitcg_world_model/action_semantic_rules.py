from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any


def default_action_semantic_rules_path() -> Path:
    return Path(__file__).with_name("action_semantic_rules.toml")


@lru_cache(maxsize=1)
def load_action_semantic_rules(path: str | Path | None = None) -> dict[str, Any]:
    payload_path = Path(path) if path is not None else default_action_semantic_rules_path()
    return tomllib.loads(payload_path.read_text(encoding="utf-8"))


def active_action_semantic_profile() -> dict[str, Any]:
    payload = load_action_semantic_rules()
    profile_name = str(payload["active_profile"])
    return dict(payload[profile_name])


def active_action_categories() -> tuple[str, ...]:
    profile = active_action_semantic_profile()
    return tuple(str(value) for value in profile.get("action_categories", ()))


def active_action_priority_order() -> tuple[str, ...]:
    profile = active_action_semantic_profile()
    priority_map = {
        str(key): int(value)
        for key, value in dict(profile.get("priorities", {})).items()
    }
    return tuple(
        key
        for key, _ in sorted(priority_map.items(), key=lambda item: (-item[1], item[0]))
    )


def active_action_thresholds() -> dict[str, float]:
    profile = active_action_semantic_profile()
    return {
        str(key): float(value)
        for key, value in dict(profile.get("thresholds", {})).items()
    }


def active_action_card_groups() -> dict[str, frozenset[int]]:
    profile = active_action_semantic_profile()
    groups = dict(profile.get("card_groups", {}))
    return {
        str(group_name): frozenset(int(value) for value in values)
        for group_name, values in groups.items()
    }


def active_action_entity_groups() -> dict[str, frozenset[int]]:
    profile = active_action_semantic_profile()
    groups = dict(profile.get("entity_groups", {}))
    return {
        str(group_name): frozenset(int(value) for value in values)
        for group_name, values in groups.items()
    }

