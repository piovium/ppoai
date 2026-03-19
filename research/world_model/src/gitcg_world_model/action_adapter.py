from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from threading import RLock
from typing import Any

from .action_hierarchy import (
    build_action_legality_engine,
    classify_action_outcome_key,
    high_level_code_for_key,
    rebuild_encoded_action_set,
)
from .schema import (
    ActionChoice,
    DecisionContext,
    DecisionType,
    EnvConfig,
    LowLevelActionSpec,
    Matchup,
    OptionKind,
    PublicSlotTarget,
    StateSnapshot,
)

OMNI_DICE = 8
ELEMENTAL_DICE = (1, 2, 3, 4, 5, 6, 7)

REQ_VOID = 0
REQ_ALIGNED = 8
REQ_ENERGY = 9
REQ_LEGEND = 10
_RESULT_RELABEL_CACHE_LIMIT = 2048
_RESULT_RELABEL_CACHE_LOCK = RLock()
_RESULT_RELABEL_CACHE: dict[tuple[Any, ...], dict[int, int]] = {}
_DIRECT_ACTION_LABEL_KINDS = {
    OptionKind.ACTION_DECLARE_END,
    OptionKind.ACTION_SWITCH_ACTIVE,
    OptionKind.ACTION_ELEMENTAL_TUNING,
}


@dataclass(frozen=True)
class BuiltDecisionContext:
    context: DecisionContext
    payload_by_low_level_code: dict[int, dict[str, Any]]
    label_by_low_level_code: dict[int, str]
    low_to_high_code: dict[int, int]


@dataclass(frozen=True)
class _SpecPayload:
    spec: LowLevelActionSpec
    payload: dict[str, Any]


def encode_choice(built: BuiltDecisionContext, action_code: int) -> dict[str, Any]:
    if int(action_code) not in built.payload_by_low_level_code:
        raise KeyError(f"unknown action_code: {action_code}")
    return dict(built.payload_by_low_level_code[int(action_code)])


def build_decision_context(
    *,
    acting_player: int,
    request_type: DecisionType,
    request: Any,
    full_state: StateSnapshot,
    player_view: StateSnapshot | None,
    full_state_json: str | None,
    step_index: int,
    metadata: dict[str, Any] | None = None,
    enable_result_based_action_relabel: bool = True,
) -> BuiltDecisionContext:
    metadata = metadata or {}
    builder = {
        DecisionType.CHOOSE_ACTIVE: _choose_active_specs,
        DecisionType.SELECT_CARD: _select_card_specs,
        DecisionType.REROLL_DICE: _reroll_specs,
        DecisionType.SWITCH_HANDS: _switch_hands_specs,
        DecisionType.ACTION: _action_specs,
    }[request_type]
    spec_payloads, request_payload = builder(
        request=request,
        acting_player=acting_player,
        full_state=full_state,
        player_view=player_view,
    )
    legality_engine = build_action_legality_engine()
    encoded_actions, materialized_specs, low_to_high_code = legality_engine.encode_legal_actions(
        acting_player=acting_player,
        visible_state=player_view or full_state,
        legal_specs=tuple(item.spec for item in spec_payloads),
    )
    if (
        enable_result_based_action_relabel
        and request_type == DecisionType.ACTION
        and full_state_json
        and encoded_actions.legal_low_level_codes
    ):
        encoded_actions, low_to_high_code = _relabel_action_high_levels_by_result(
            acting_player=acting_player,
            full_state_json=full_state_json,
            full_state=full_state,
            legal_low_level_codes=encoded_actions.legal_low_level_codes,
            legal_low_level_mask=encoded_actions.legal_low_level_mask,
            low_to_high_code=low_to_high_code,
            legality_engine=legality_engine,
            metadata=metadata,
        )
    payload_by_low_level_code = {
        int(spec.action_code): dict(item.payload)
        for spec, item in zip(materialized_specs, spec_payloads, strict=False)
    }
    label_by_low_level_code = {
        int(spec.action_code): str(spec.label)
        for spec in materialized_specs
    }
    context = DecisionContext(
        acting_player=acting_player,
        request_type=request_type,
        step_index=step_index,
        full_state=full_state,
        legal_low_level_codes=encoded_actions.legal_low_level_codes,
        legal_low_level_mask=encoded_actions.legal_low_level_mask,
        legal_high_level_codes=encoded_actions.legal_high_level_codes,
        high_to_low_map=encoded_actions.high_to_low_map,
        player_view=player_view,
        full_state_json=full_state_json,
        request_payload=request_payload,
        metadata=metadata,
    )
    return BuiltDecisionContext(
        context=context,
        payload_by_low_level_code=payload_by_low_level_code,
        label_by_low_level_code=label_by_low_level_code,
        low_to_high_code=low_to_high_code,
    )


def _relabel_action_high_levels_by_result(
    *,
    acting_player: int,
    full_state_json: str,
    full_state: StateSnapshot,
    legal_low_level_codes: tuple[int, ...],
    legal_low_level_mask: tuple[bool, ...],
    low_to_high_code: dict[int, int],
    legality_engine,
    metadata: dict[str, Any] | None,
) -> tuple[Any, dict[int, int]]:
    from .env import GitcgDecisionEnv

    metadata = metadata or {}
    matchup = _matchup_from_metadata(metadata)
    cache_key = _result_relabel_cache_key(
        acting_player=acting_player,
        full_state_json=full_state_json,
        legal_low_level_codes=legal_low_level_codes,
    )
    with _RESULT_RELABEL_CACHE_LOCK:
        cached = _RESULT_RELABEL_CACHE.get(cache_key)
    if cached is not None:
        return rebuild_encoded_action_set(
            legal_low_level_codes=legal_low_level_codes,
            legal_low_level_mask=legal_low_level_mask,
            low_to_high_code=cached,
        ), dict(cached)
    env = GitcgDecisionEnv(
        _result_relabel_env_config(),
        matchup,
        enable_result_based_action_relabel=False,
    )
    codebook = legality_engine.codebook
    relabeled: dict[int, int] = {}
    try:
        for action_code in legal_low_level_codes:
            spec = codebook.low_spec(int(action_code))
            # These kinds still have real downstream game consequences, but under the
            # current ACTION taxonomy their primary high-level label is defined by the
            # action type itself:
            # - declare_end -> yield_initiative
            # - switch_active -> switch_character
            # - elemental_tuning -> elemental_tuning
            # We therefore skip the one-step relabel simulation here because the
            # simulated post-state would not change the chosen high-level bucket.
            if spec.kind in _DIRECT_ACTION_LABEL_KINDS:
                key = classify_action_outcome_key(
                    spec=spec,
                    pre_state=full_state,
                    post_state=full_state,
                    acting_player=acting_player,
                )
                relabeled[int(action_code)] = high_level_code_for_key(
                    codebook,
                    DecisionType.ACTION,
                    key,
                )
                continue
            env.reset(state_json=full_state_json)
            _, step, _ = env.step(ActionChoice(action_code=int(action_code)))
            key = classify_action_outcome_key(
                spec=spec,
                pre_state=full_state,
                post_state=step.post_state,
                acting_player=acting_player,
            )
            relabeled[int(action_code)] = high_level_code_for_key(
                codebook,
                DecisionType.ACTION,
                key,
            )
    except Exception:
        return rebuild_encoded_action_set(
            legal_low_level_codes=legal_low_level_codes,
            legal_low_level_mask=legal_low_level_mask,
            low_to_high_code=low_to_high_code,
        ), low_to_high_code
    finally:
        env.close()
    with _RESULT_RELABEL_CACHE_LOCK:
        if len(_RESULT_RELABEL_CACHE) >= _RESULT_RELABEL_CACHE_LIMIT:
            oldest_key = next(iter(_RESULT_RELABEL_CACHE))
            _RESULT_RELABEL_CACHE.pop(oldest_key, None)
        _RESULT_RELABEL_CACHE[cache_key] = dict(relabeled)
    return rebuild_encoded_action_set(
        legal_low_level_codes=legal_low_level_codes,
        legal_low_level_mask=legal_low_level_mask,
        low_to_high_code=relabeled,
    ), relabeled


def _matchup_from_metadata(metadata: dict[str, Any]) -> Matchup:
    matchup_key = str(metadata.get("matchup", "unknown__vs__unknown"))
    if "__vs__" not in matchup_key:
        return Matchup("unknown", "unknown")
    deck0, deck1 = matchup_key.split("__vs__", maxsplit=1)
    if not deck0 or not deck1:
        return Matchup("unknown", "unknown")
    return Matchup(deck0, deck1)


def _result_relabel_env_config() -> EnvConfig:
    return EnvConfig(
        deck_pool=(),
        record_full_state_json=False,
        record_player_view=True,
        draw_penalty=0.0,
    )


def _result_relabel_cache_key(
    *,
    acting_player: int,
    full_state_json: str,
    legal_low_level_codes: tuple[int, ...],
) -> tuple[Any, ...]:
    digest = hashlib.blake2b(full_state_json.encode("utf-8"), digest_size=16).hexdigest()
    return (
        int(acting_player),
        digest,
        tuple(int(code) for code in legal_low_level_codes),
    )


def _choose_active_specs(**kwargs):
    request = kwargs["request"]
    acting_player = kwargs["acting_player"]
    state = kwargs["player_view"] or kwargs["full_state"]
    slot_by_character_id = {
        int(character.id): index
        for index, character in enumerate(state.players[acting_player].characters)
    }
    definition_by_character_id = {
        int(character.id): int(character.definition_id)
        for character in state.players[acting_player].characters
    }
    specs = []
    for character_id in request.candidate_ids:
        specs.append(
            _SpecPayload(
                spec=LowLevelActionSpec(
                    action_code=-1,
                    request_type=DecisionType.CHOOSE_ACTIVE,
                    kind=OptionKind.CHOOSE_ACTIVE,
                    label=f"choose_active:{character_id}",
                    subject_definition_id=int(definition_by_character_id.get(int(character_id), 0)),
                    choose_active_slot=int(slot_by_character_id.get(int(character_id), -1)),
                ),
                payload={"active_character_id": character_id},
            )
        )
    return specs, {
        "candidate_ids": list(request.candidate_ids),
        "candidate_definition_ids": [
            int(definition_by_character_id.get(int(character_id), 0))
            for character_id in request.candidate_ids
        ],
    }


def _select_card_specs(**kwargs):
    request = kwargs["request"]
    specs = [
        _SpecPayload(
            spec=LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.SELECT_CARD,
                kind=OptionKind.SELECT_CARD,
                label=f"select_card:{definition_id}",
                subject_definition_id=int(definition_id),
                select_card_definition_id=int(definition_id),
            ),
            payload={"selected_definition_id": definition_id},
        )
        for definition_id in request.candidate_definition_ids
    ]
    return specs, {
        "candidate_definition_ids": list(request.candidate_definition_ids),
    }


def _reroll_specs(**kwargs):
    acting_player = kwargs["acting_player"]
    player_view = kwargs["player_view"]
    if player_view is None:
        raise ValueError("player_view is required for reroll enumeration")
    visible_dice = player_view.players[acting_player].dice
    specs = [
        _SpecPayload(
            spec=LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.REROLL_DICE,
                kind=OptionKind.REROLL_DICE,
                label=f"reroll:{','.join(map(str, dice_to_reroll)) or 'none'}",
                reroll_dice_mask=_reroll_mask_from_subset(visible_dice, dice_to_reroll),
            ),
            payload={"dice_to_reroll": list(dice_to_reroll)},
        )
        for dice_to_reroll in _enumerate_multiset_subsets(visible_dice)
    ]
    return specs, {"visible_dice": list(visible_dice)}


def _switch_hands_specs(**kwargs):
    acting_player = kwargs["acting_player"]
    player_view = kwargs["player_view"]
    if player_view is None:
        raise ValueError("player_view is required for switch-hands enumeration")
    hand_cards = tuple(player_view.players[acting_player].hand_cards)
    hand_ids = tuple(card.id for card in hand_cards)
    specs = []
    for removed_ids in _enumerate_id_subsets(hand_ids):
        switch_mask = _switch_hand_mask_from_removed_ids(hand_cards, removed_ids)
        removed_definitions = [
            int(card.definition_id)
            for index, card in enumerate(hand_cards)
            if switch_mask & (1 << index)
        ]
        subject_definition_id = removed_definitions[0] if removed_definitions else 0
        specs.append(
            _SpecPayload(
                spec=LowLevelActionSpec(
                    action_code=-1,
                    request_type=DecisionType.SWITCH_HANDS,
                    kind=OptionKind.SWITCH_HANDS,
                    label=f"switch_hands:{','.join(map(str, removed_ids)) or 'none'}",
                    subject_definition_id=int(subject_definition_id),
                    switch_hand_slot_mask=int(switch_mask),
                ),
                payload={"removed_hand_ids": list(removed_ids)},
            )
        )
    return specs, {
        "hand_ids": list(hand_ids),
        "hand_definition_ids": [int(card.definition_id) for card in hand_cards],
    }


def _action_specs(**kwargs):
    request = kwargs["request"]
    acting_player = kwargs["acting_player"]
    player_view = kwargs["player_view"]
    if player_view is None:
        raise ValueError("player_view is required for action enumeration")
    visible_dice = tuple(player_view.players[acting_player].dice)
    hand_card_definitions = {
        int(card.id): int(card.definition_id)
        for card in player_view.players[acting_player].hand_cards
    }
    specs: list[_SpecPayload] = []
    for action_index, action in enumerate(request.action):
        if getattr(action, "validity", None) != 0:
            continue
        kind, label, semantic = _classify_action(action, state=player_view, acting_player=acting_player)
        semantic = _enrich_action_semantics(
            kind=kind,
            semantic=semantic,
            hand_card_definitions=hand_card_definitions,
        )
        auto_selected = tuple(int(d) for d in getattr(action, "auto_selected_dice", ()))
        payments = (
            [auto_selected]
            if auto_selected
            else _enumerate_action_dice_payments(visible_dice, action.required_cost)
        )
        if kind == OptionKind.ACTION_DECLARE_END and not payments:
            payments = [tuple()]
        if not payments:
            continue
        for used_dice in payments:
            specs.append(
                _SpecPayload(
                    spec=LowLevelActionSpec(
                        action_code=-1,
                        request_type=DecisionType.ACTION,
                        kind=kind,
                        label=f"{label}:{action_index}:{','.join(map(str, used_dice)) or 'none'}",
                        subject_definition_id=int(semantic.get("subject_definition_id", 0)),
                        target_slots=tuple(semantic.get("target_slots", ())),
                        used_dice=tuple(used_dice),
                        auto_selected_dice=auto_selected,
                        discarded_hand_slot=int(semantic.get("discarded_hand_slot", -1)),
                        discarded_card_definition_id=int(semantic.get("discarded_card_definition_id", 0)),
                        target_dice=int(semantic.get("target_dice", 0)),
                        metadata={
                            "action_index": int(action_index),
                            "required_cost": tuple(
                                (int(req.type), int(req.count))
                                for req in action.required_cost
                            ),
                            "is_fast": bool(getattr(action, "is_fast", False)),
                        },
                    ),
                    payload={
                        "chosen_action_index": action_index,
                        "used_dice": list(used_dice),
                    },
                )
            )
    return specs, {"visible_dice": list(visible_dice), "legal_action_count": len(specs)}


def _classify_action(
    action: Any,
    *,
    state: StateSnapshot,
    acting_player: int,
) -> tuple[OptionKind, str, dict[str, Any]]:
    if action.HasField("use_skill"):
        value = action.use_skill
        return (
            OptionKind.ACTION_USE_SKILL,
            "use_skill",
            {
                "subject_definition_id": int(value.skill_definition_id),
                "target_slots": _target_slots_from_ids(tuple(int(target) for target in value.target_ids), state=state, acting_player=acting_player),
            },
        )
    if action.HasField("play_card"):
        value = action.play_card
        return (
            OptionKind.ACTION_PLAY_CARD,
            "play_card",
            {
                "subject_definition_id": int(value.card_definition_id),
                "target_slots": _target_slots_from_ids(tuple(int(target) for target in value.target_ids), state=state, acting_player=acting_player),
            },
        )
    if action.HasField("switch_active"):
        value = action.switch_active
        return (
            OptionKind.ACTION_SWITCH_ACTIVE,
            "switch_active",
            {
                "subject_definition_id": int(value.character_definition_id),
                "target_slots": _target_slots_for_switch_active(state=state, acting_player=acting_player, character_id=int(value.character_id)),
            },
        )
    if action.HasField("elemental_tuning"):
        value = action.elemental_tuning
        return (
            OptionKind.ACTION_ELEMENTAL_TUNING,
            "elemental_tuning",
            {
                "removed_card_id": int(value.removed_card_id),
                "target_dice": int(value.target_dice),
            },
        )
    return (
        OptionKind.ACTION_DECLARE_END,
        "declare_end",
        {},
    )


def _enrich_action_semantics(
    *,
    kind: OptionKind,
    semantic: dict[str, Any],
    hand_card_definitions: dict[int, int],
) -> dict[str, Any]:
    enriched = dict(semantic)
    if kind == OptionKind.ACTION_ELEMENTAL_TUNING:
        removed_card_id = int(semantic.get("removed_card_id", 0))
        removed_card_definition_id = int(hand_card_definitions.get(removed_card_id, 0))
        if removed_card_definition_id > 0:
            enriched["subject_definition_id"] = removed_card_definition_id
            enriched["discarded_card_definition_id"] = removed_card_definition_id
    return enriched


def _build_target_lookup(state: StateSnapshot) -> dict[int, dict[str, int | bool | str]]:
    lookup: dict[int, dict[str, int | bool | str]] = {}

    def register(
        entity_id: int,
        *,
        owner: int,
        kind: str,
        definition_id: int,
        public_index: int,
        is_active: bool = False,
        defeated: bool = False,
    ) -> None:
        lookup[int(entity_id)] = {
            "owner": owner,
            "kind": kind,
            "definition_id": int(definition_id),
            "public_index": int(public_index),
            "is_active": bool(is_active),
            "defeated": bool(defeated),
        }

    for owner, player in enumerate(state.players):
        for public_index, character in enumerate(player.characters):
            register(
                int(character.id),
                owner=owner,
                kind="character",
                definition_id=int(character.definition_id),
                public_index=public_index,
                is_active=bool(character.is_active),
                defeated=bool(character.defeated),
            )
            for entity_index, entity in enumerate(character.entities):
                register(
                    int(entity.id),
                    owner=owner,
                    kind="character_entity",
                    definition_id=int(entity.definition_id),
                    public_index=entity_index,
                    is_active=bool(character.is_active),
                )
        for public_index, entity in enumerate(player.combat_statuses):
            register(
                int(entity.id),
                owner=owner,
                kind="combat_status",
                definition_id=int(entity.definition_id),
                public_index=public_index,
            )
        for public_index, entity in enumerate(player.summons):
            register(
                int(entity.id),
                owner=owner,
                kind="summon",
                definition_id=int(entity.definition_id),
                public_index=public_index,
            )
        for public_index, entity in enumerate(player.supports):
            register(
                int(entity.id),
                owner=owner,
                kind="support",
                definition_id=int(entity.definition_id),
                public_index=public_index,
            )
    return lookup


def _target_slots_from_ids(
    target_ids: tuple[int, ...],
    *,
    state: StateSnapshot,
    acting_player: int,
) -> tuple[PublicSlotTarget, ...]:
    lookup = _build_target_lookup(state)
    targets: list[PublicSlotTarget] = []
    for target_id in target_ids:
        target = lookup.get(int(target_id))
        if target is None:
            continue
        owner = "self" if int(target["owner"]) == int(acting_player) else "opponent"
        zone = "combat_status" if str(target["kind"]) == "status" else str(target["kind"])
        targets.append(PublicSlotTarget(owner=owner, zone=zone, index=int(target["public_index"])))
    return tuple(targets)


def _target_slots_for_switch_active(
    *,
    state: StateSnapshot,
    acting_player: int,
    character_id: int,
) -> tuple[PublicSlotTarget, ...]:
    for index, character in enumerate(state.players[acting_player].characters):
        if int(character.id) == int(character_id):
            return (PublicSlotTarget(owner="self", zone="character", index=index),)
    return ()


def _enumerate_action_dice_payments(
    visible_dice: tuple[int, ...],
    required_cost: Any,
) -> list[tuple[int, ...]]:
    entries = [(int(req.type), int(req.count)) for req in required_cost]
    total_count = sum(
        count for req_type, count in entries if req_type not in (REQ_ENERGY, REQ_LEGEND)
    )
    if total_count == 0:
        return [tuple()]
    unique: set[tuple[int, ...]] = set()
    for indexes in combinations(range(len(visible_dice)), total_count):
        chosen = tuple(sorted(int(visible_dice[index]) for index in indexes))
        if chosen in unique:
            continue
        if _satisfies_cost(chosen, entries):
            unique.add(chosen)
    return sorted(unique)


def _satisfies_cost(chosen_dice: tuple[int, ...], entries: list[tuple[int, int]]) -> bool:
    exact_requirements = [(req_type, count) for req_type, count in entries if req_type in ELEMENTAL_DICE]
    aligned_count = sum(count for req_type, count in entries if req_type == REQ_ALIGNED)
    void_count = sum(count for req_type, count in entries if req_type == REQ_VOID)
    counter = Counter(chosen_dice)
    counter = _consume_exact(counter, exact_requirements)
    if counter is None:
        return False
    if aligned_count and _consume_aligned(counter, aligned_count) is None:
        return False
    return len(chosen_dice) == sum(count for _, count in exact_requirements) + aligned_count + void_count


def _consume_exact(
    counter: Counter[int],
    exact_requirements: list[tuple[int, int]],
) -> Counter[int] | None:
    current = counter.copy()
    for req_type, count in exact_requirements:
        exact_available = min(current[req_type], count)
        remaining = count - exact_available
        current[req_type] -= exact_available
        if current[OMNI_DICE] < remaining:
            return None
        current[OMNI_DICE] -= remaining
    return current


def _consume_aligned(counter: Counter[int], aligned_count: int) -> Counter[int] | None:
    for die_type in ELEMENTAL_DICE:
        exact_available = min(counter[die_type], aligned_count)
        remaining = aligned_count - exact_available
        if counter[OMNI_DICE] < remaining:
            continue
        next_counter = counter.copy()
        next_counter[die_type] -= exact_available
        next_counter[OMNI_DICE] -= remaining
        return next_counter
    if counter[OMNI_DICE] >= aligned_count:
        next_counter = counter.copy()
        next_counter[OMNI_DICE] -= aligned_count
        return next_counter
    return None


def _enumerate_multiset_subsets(values: tuple[int, ...]) -> list[tuple[int, ...]]:
    unique: set[tuple[int, ...]] = set()
    for subset_size in range(len(values) + 1):
        for indexes in combinations(range(len(values)), subset_size):
            unique.add(tuple(sorted(values[index] for index in indexes)))
    return sorted(unique)


def _enumerate_id_subsets(values: tuple[int, ...]) -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    for subset_size in range(len(values) + 1):
        for indexes in combinations(range(len(values)), subset_size):
            subsets.append(tuple(sorted(values[index] for index in indexes)))
    return subsets


def _switch_hand_mask_from_removed_ids(hand_cards: tuple[Any, ...], removed_ids: tuple[int, ...]) -> int:
    remaining = list(int(value) for value in removed_ids)
    mask = 0
    for index, card in enumerate(hand_cards):
        card_id = int(getattr(card, "id", 0))
        if card_id in remaining:
            mask |= 1 << index
            remaining.remove(card_id)
    return mask


def _reroll_mask_from_subset(visible_dice: tuple[int, ...], dice_to_reroll: tuple[int, ...]) -> int:
    remaining = list(int(value) for value in dice_to_reroll)
    mask = 0
    for index, die in enumerate(visible_dice):
        die = int(die)
        if die in remaining:
            mask |= 1 << index
            remaining.remove(die)
    return mask
