from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any


def default_non_action_semantic_rules_path() -> Path:
    return Path(__file__).with_name("non_action_semantic_rules.toml")


@lru_cache(maxsize=1)
def load_non_action_semantic_rules(path: str | Path | None = None) -> dict[str, Any]:
    payload_path = Path(path) if path is not None else default_non_action_semantic_rules_path()
    return tomllib.loads(payload_path.read_text(encoding="utf-8"))


def _active_profile_payload() -> dict[str, Any]:
    payload = load_non_action_semantic_rules()
    return dict(payload[str(payload["active_profile"])])


def active_reroll_character_elements() -> dict[int, int]:
    payload = _active_profile_payload()
    block = dict(payload.get("reroll", {})).get("character_elements", {})
    return {int(definition_id): int(dice_type) for definition_id, dice_type in block.items()}


def active_choose_active_role_groups() -> dict[str, frozenset[int]]:
    payload = _active_profile_payload()
    role_block = dict(dict(payload.get("choose_active", {})).get("roles", {}))
    return {
        str(role_name): frozenset(int(value) for value in values)
        for role_name, values in role_block.items()
    }


def active_choose_active_role_priority() -> tuple[str, ...]:
    payload = _active_profile_payload()
    priority_block = dict(dict(payload.get("choose_active", {})).get("priority", {}))
    return tuple(str(value) for value in priority_block.get("order", ()))


def active_card_function_groups() -> dict[str, frozenset[int]]:
    payload = _active_profile_payload()
    group_block = dict(dict(payload.get("card_functions", {})).get("groups", {}))
    return {
        str(group_name): frozenset(int(value) for value in values)
        for group_name, values in group_block.items()
    }


def active_card_function_priority() -> tuple[str, ...]:
    payload = _active_profile_payload()
    priority_block = dict(dict(payload.get("card_functions", {})).get("priority", {}))
    return tuple(str(value) for value in priority_block.get("groups", ()))
