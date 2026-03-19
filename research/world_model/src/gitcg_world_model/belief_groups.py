from __future__ import annotations

import tomllib
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def default_belief_groups_path() -> Path:
    return Path(__file__).with_name("belief_groups.toml")


@lru_cache(maxsize=1)
def _load_belief_group_payload(path: str | Path | None = None) -> tuple[tuple[str, ...], dict[int, tuple[str, ...]]]:
    payload_path = Path(path) if path is not None else default_belief_groups_path()
    payload = tomllib.loads(payload_path.read_text(encoding="utf-8"))
    groups = tuple(str(value) for value in payload.get("groups", ()))
    card_groups_block = payload.get("card_groups", {})
    card_groups = {
        int(definition_id): tuple(str(group) for group in groups_payload)
        for definition_id, groups_payload in card_groups_block.items()
    }
    return groups, card_groups


def belief_groups() -> tuple[str, ...]:
    groups, _ = _load_belief_group_payload()
    return groups


def belief_group_dim() -> int:
    return len(belief_groups())


def groups_for_card(definition_id: int) -> tuple[str, ...]:
    _, card_groups = _load_belief_group_payload()
    return card_groups.get(int(definition_id), ())


def encode_group_presence(cards: Iterable[object]) -> tuple[float, ...]:
    groups = belief_groups()
    group_index = {name: index for index, name in enumerate(groups)}
    presence = [0.0] * len(groups)
    for card in cards:
        if hasattr(card, "definition_id"):
            definition_id = int(getattr(card, "definition_id"))
        else:
            definition_id = int(card)
        for group in groups_for_card(definition_id):
            index = group_index.get(group)
            if index is not None:
                presence[index] = 1.0
    return tuple(presence)
