from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .schema import (
    ActionChoice,
    CharacterSnapshot,
    DecisionType,
    EntitySnapshot,
    EpisodeRecord,
    PlayerSnapshot,
    StateSnapshot,
    TrajectoryStep,
)


def append_episode_records(path: str | Path, episodes: Iterable[EpisodeRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(serialize_episode_record(episode), sort_keys=True))
            handle.write("\n")


def read_episode_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_episode_record_payloads(path))


def iter_episode_record_payloads(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_episode_records(path: str | Path) -> list[EpisodeRecord]:
    return list(iter_episode_records(path))


def iter_episode_records(path: str | Path) -> Iterator[EpisodeRecord]:
    for payload in iter_episode_record_payloads(path):
        yield deserialize_episode_record(payload)


def load_episode_records_from_paths(paths: Sequence[str | Path]) -> list[EpisodeRecord]:
    return list(iter_episode_records_from_paths(paths))


def iter_episode_records_from_paths(
    paths: Sequence[str | Path],
    *,
    dedupe: bool = True,
) -> Iterator[EpisodeRecord]:
    for path in resolve_replay_paths(paths, dedupe=dedupe):
        yield from iter_episode_records(path)


def resolve_replay_paths(
    paths: Sequence[str | Path],
    *,
    dedupe: bool = True,
) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            resolved.append(path)
    if not dedupe:
        return resolved
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def sample_episode_records(
    episodes: Sequence[EpisodeRecord],
    count: int,
    *,
    seed: int,
) -> list[EpisodeRecord]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0 or not episodes:
        return []
    rng = random.Random(seed)
    if count >= len(episodes):
        sampled = list(episodes)
        rng.shuffle(sampled)
        return sampled[:count]
    return rng.sample(list(episodes), count)


def group_episode_records_by_matchup(
    episodes: Sequence[EpisodeRecord],
) -> dict[str, list[EpisodeRecord]]:
    grouped: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        grouped.setdefault(episode.matchup, []).append(episode)
    return grouped


def serialize_episode_record(episode: EpisodeRecord) -> dict[str, Any]:
    return _to_jsonable(episode)


def deserialize_episode_record(payload: dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        matchup=str(payload["matchup"]),
        seed=payload.get("seed"),
        winner=payload.get("winner"),
        steps=tuple(_deserialize_trajectory_step(step) for step in payload.get("steps", [])),
        final_state=_deserialize_state_snapshot(payload["final_state"]),
        final_state_json=payload.get("final_state_json"),
        metadata=dict(payload.get("metadata", {})),
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _deserialize_trajectory_step(payload: dict[str, Any]) -> TrajectoryStep:
    choice_payload = dict(payload.get("choice", {}))
    action_code = int(choice_payload.get("action_code", -1))
    return TrajectoryStep(
        acting_player=int(payload["acting_player"]),
        request_type=DecisionType(payload["request_type"]),
        choice=ActionChoice(action_code=action_code),
        pre_state=_deserialize_state_snapshot(payload["pre_state"]),
        post_state=_deserialize_state_snapshot(payload["post_state"]),
        reward=float(payload["reward"]),
        done=bool(payload["done"]),
        legal_low_level_codes=tuple(int(value) for value in payload.get("legal_low_level_codes", ())),
        legal_low_level_mask=tuple(bool(value) for value in payload.get("legal_low_level_mask", ())),
        legal_high_level_codes=tuple(int(value) for value in payload.get("legal_high_level_codes", ())),
        high_to_low_map=tuple(
            (
                int(entry[0]),
                tuple(int(code) for code in entry[1]),
            )
            for entry in payload.get("high_to_low_map", ())
        ),
        chosen_high_level_code=(
            int(payload["chosen_high_level_code"])
            if payload.get("chosen_high_level_code") is not None
            else None
        ),
        player_view=(
            _deserialize_state_snapshot(payload["player_view"])
            if payload.get("player_view") is not None
            else None
        ),
        full_state_json_before=payload.get("full_state_json_before"),
        metadata=dict(payload.get("metadata", {})),
    )

def _deserialize_state_snapshot(payload: dict[str, Any]) -> StateSnapshot:
    return StateSnapshot(
        phase=str(payload["phase"]),
        round_number=int(payload["round_number"]),
        current_turn=int(payload["current_turn"]),
        winner=payload.get("winner"),
        players=tuple(_deserialize_player_snapshot(player) for player in payload["players"]),  # type: ignore[arg-type]
    )


def _deserialize_player_snapshot(payload: dict[str, Any]) -> PlayerSnapshot:
    return PlayerSnapshot(
        active_character_id=payload.get("active_character_id"),
        characters=tuple(
            _deserialize_character_snapshot(character) for character in payload["characters"]
        ),
        combat_statuses=tuple(
            _deserialize_entity_snapshot(entity) for entity in payload["combat_statuses"]
        ),
        summons=tuple(_deserialize_entity_snapshot(entity) for entity in payload["summons"]),
        supports=tuple(_deserialize_entity_snapshot(entity) for entity in payload["supports"]),
        hand_cards=tuple(
            _deserialize_entity_snapshot(entity) for entity in payload["hand_cards"]
        ),
        pile_cards=tuple(
            _deserialize_entity_snapshot(entity) for entity in payload["pile_cards"]
        ),
        dice=tuple(int(value) for value in payload.get("dice", [])),
        declared_end=bool(payload["declared_end"]),
        legend_used=bool(payload["legend_used"]),
    )


def _deserialize_character_snapshot(payload: dict[str, Any]) -> CharacterSnapshot:
    return CharacterSnapshot(
        id=int(payload["id"]),
        definition_id=int(payload["definition_id"]),
        health=int(payload["health"]),
        max_health=int(payload["max_health"]),
        energy=int(payload["energy"]),
        max_energy=int(payload["max_energy"]),
        defeated=bool(payload["defeated"]),
        is_active=bool(payload["is_active"]),
        aura=int(payload.get("aura", 0)),
        entities=tuple(_deserialize_entity_snapshot(entity) for entity in payload["entities"]),
    )


def _deserialize_entity_snapshot(payload: dict[str, Any]) -> EntitySnapshot:
    return EntitySnapshot(
        id=int(payload["id"]),
        definition_id=int(payload["definition_id"]),
        variables={str(key): int(value) for key, value in payload.get("variables", {}).items()},
    )
