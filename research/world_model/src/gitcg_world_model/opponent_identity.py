from __future__ import annotations

import hashlib

_OPPONENT_TAGS = (
    "active_population_slot",
    "frozen_population",
    "meta_population",
    "baseline_random",
    "baseline_heuristic",
    "baseline_scripted",
    "unknown",
)

_KNOWN_BASELINE_DECKS = {
    "baseline_random": "baseline_random",
    "baseline_heuristic": "baseline_heuristic",
    "baseline_scripted": "baseline_scripted",
}


def opponent_tags() -> tuple[str, ...]:
    return _OPPONENT_TAGS


def opponent_tag_to_id(tag: str) -> int:
    normalized = tag.strip().lower()
    try:
        return _OPPONENT_TAGS.index(normalized)
    except ValueError:
        return _OPPONENT_TAGS.index("unknown")


def opponent_tag(kind: str, source: str) -> str:
    normalized_kind = kind.strip().lower()
    normalized = source.strip().lower()
    if normalized_kind in {"active_main_slot", "active_br_slot", "active_slot", "candidate_eval"}:
        return "active_population_slot"
    if normalized_kind in {"main", "current", "best_response", "br", "frozen_population"}:
        return "frozen_population"
    if normalized_kind == "meta_population":
        return "meta_population"
    if normalized in {"baseline_random", "random"}:
        return "baseline_random"
    if normalized in {"baseline_heuristic", "heuristic"}:
        return "baseline_heuristic"
    if normalized in {"baseline_scripted", "scripted"}:
        return "baseline_scripted"
    return "unknown"


def population_entity_id(*, kind: str, source: str) -> str:
    normalized_source = source.replace("\\", "/")
    normalized_kind = kind.strip().lower()
    if normalized_kind == "baseline":
        return normalized_source
    if normalized_kind in {"main", "current", "best_response", "br", "frozen_population"}:
        return f"frozen_population:{normalized_source}"
    if normalized_kind in {"active_main_slot", "active_br_slot", "active_slot", "candidate_eval"}:
        return f"active_population_slot:{normalized_source}"
    if normalized_kind == "meta_population":
        return f"meta_population:{normalized_source}"
    return f"{normalized_kind}:{normalized_source}"


def stable_hash_id(key: str, *, vocab_size: int, unknown_id: int = 0) -> int:
    if vocab_size <= 1:
        return unknown_id
    if not key or key == "unknown":
        return unknown_id
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket_count = max(1, vocab_size - 1)
    return 1 + (int.from_bytes(digest[:8], byteorder="big", signed=False) % bucket_count)


def infer_opponent_deck_name(
    *,
    matchup_key: str | None,
    acting_player: int,
    opponent_kind: str | None = None,
    opponent_source: str | None = None,
) -> str:
    normalized_kind = (opponent_kind or "").strip().lower()
    normalized_source = (opponent_source or "").strip().lower()
    if normalized_kind == "baseline":
        return _KNOWN_BASELINE_DECKS.get(normalized_source, "unknown")
    if normalized_source in _KNOWN_BASELINE_DECKS:
        return _KNOWN_BASELINE_DECKS[normalized_source]
    if matchup_key and "__vs__" in matchup_key:
        left, right = matchup_key.split("__vs__", 1)
        return right if acting_player == 0 else left
    return "unknown"
