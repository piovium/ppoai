from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence

from .schema import (
    ActionTaxonomy,
    DecisionContext,
    DecisionType,
    EncodedLegalActionSet,
    HighLevelActionSpec,
    LowLevelActionSpec,
    OptionKind,
    PublicSlotTarget,
    StateSnapshot,
)
from .semantic_priors import action_hierarchy_config

_OWNER_VALUES = ("self", "opponent", "neutral")
_ZONE_VALUES = (
    "character",
    "character_entity",
    "combat_status",
    "summon",
    "support",
    "hand_card",
    "pile_card",
)
_ACTION_REQUEST_TYPES = (
    DecisionType.ACTION,
    DecisionType.CHOOSE_ACTIVE,
    DecisionType.REROLL_DICE,
    DecisionType.SELECT_CARD,
    DecisionType.SWITCH_HANDS,
)


def default_action_taxonomy_path() -> Path:
    return Path(__file__).with_name("action_taxonomy.toml")


@lru_cache(maxsize=1)
def load_action_taxonomy(path: str | Path | None = None) -> ActionTaxonomy:
    taxonomy_path = Path(path) if path is not None else default_action_taxonomy_path()
    payload = tomllib.loads(taxonomy_path.read_text(encoding="utf-8"))
    active = str(payload["active_taxonomy"])
    active_block = payload[active]
    request_type_to_categories: dict[str, tuple[str, ...]] = {}
    for request_name, config in active_block.items():
        request_type_to_categories[str(request_name)] = tuple(str(value) for value in config.get("categories", ()))
    return ActionTaxonomy(
        active_taxonomy=active,
        request_type_to_categories=request_type_to_categories,
    )


@dataclass(frozen=True)
class HierarchicalActionCodebook:
    taxonomy: ActionTaxonomy
    low_level_specs: dict[int, LowLevelActionSpec]
    high_level_specs: dict[int, HighLevelActionSpec]
    low_key_to_code: dict[tuple[Any, ...], int]
    high_key_to_code: dict[tuple[str, str], int]
    next_low_code: int = 0

    def low_spec(self, action_code: int) -> LowLevelActionSpec:
        return self.low_level_specs[int(action_code)]

    def high_spec(self, high_action_code: int) -> HighLevelActionSpec:
        return self.high_level_specs[int(high_action_code)]

    def ensure_low_spec(self, spec: LowLevelActionSpec) -> "HierarchicalActionCodebook":
        semantic_key = _low_level_semantic_key(spec)
        existing = self.low_key_to_code.get(semantic_key)
        if existing is not None:
            return self
        code = int(self.next_low_code)
        materialized = replace(spec, action_code=code)
        low_level_specs = dict(self.low_level_specs)
        low_level_specs[code] = materialized
        low_key_to_code = dict(self.low_key_to_code)
        low_key_to_code[semantic_key] = code
        return HierarchicalActionCodebook(
            taxonomy=self.taxonomy,
            low_level_specs=low_level_specs,
            high_level_specs=self.high_level_specs,
            low_key_to_code=low_key_to_code,
            high_key_to_code=self.high_key_to_code,
            next_low_code=code + 1,
        )

    def code_for(self, spec: LowLevelActionSpec) -> int:
        return int(self.low_key_to_code[_low_level_semantic_key(spec)])


class ActionLegalityEngine:
    def __init__(self, codebook: HierarchicalActionCodebook) -> None:
        self.codebook = codebook

    def encode_legal_actions(
        self,
        *,
        acting_player: int,
        visible_state: StateSnapshot,
        legal_specs: Iterable[LowLevelActionSpec],
    ) -> tuple[EncodedLegalActionSet, tuple[LowLevelActionSpec, ...], dict[int, int]]:
        low_codes: list[int] = []
        materialized_specs: list[LowLevelActionSpec] = []
        low_to_high: dict[int, int] = {}
        with _DEFAULT_CODEBOOK_LOCK:
            global _DEFAULT_CODEBOOK
            mutable_codebook = _DEFAULT_CODEBOOK or build_default_hierarchical_action_codebook()
            for spec in legal_specs:
                mutable_codebook = mutable_codebook.ensure_low_spec(spec)
                action_code = mutable_codebook.code_for(spec)
                materialized_spec = mutable_codebook.low_spec(action_code)
                low_codes.append(action_code)
                materialized_specs.append(materialized_spec)
                low_to_high[action_code] = high_level_code_for_spec(
                    mutable_codebook,
                    materialized_spec,
                    visible_state=visible_state,
                    acting_player=acting_player,
                )
            _DEFAULT_CODEBOOK = mutable_codebook
            self.codebook = mutable_codebook
        unique_high_codes: list[int] = []
        high_to_low: dict[int, list[int]] = {}
        for code in low_codes:
            high_code = int(low_to_high[code])
            if high_code not in high_to_low:
                unique_high_codes.append(high_code)
                high_to_low[high_code] = []
            high_to_low[high_code].append(int(code))
        encoded = EncodedLegalActionSet(
            legal_low_level_codes=tuple(int(code) for code in low_codes),
            legal_low_level_mask=tuple(True for _ in low_codes),
            legal_high_level_codes=tuple(int(code) for code in unique_high_codes),
            high_to_low_map=tuple(
                (int(high_code), tuple(int(code) for code in codes))
                for high_code, codes in high_to_low.items()
            ),
        )
        return encoded, tuple(materialized_specs), low_to_high


def build_default_hierarchical_action_codebook() -> HierarchicalActionCodebook:
    taxonomy = load_action_taxonomy()
    high_level_specs: dict[int, HighLevelActionSpec] = {}
    high_key_to_code: dict[tuple[str, str], int] = {}
    next_high_code = 0
    for request_type in _ACTION_REQUEST_TYPES:
        categories = taxonomy.request_type_to_categories.get(request_type.name, ())
        for key in categories:
            high_key = (request_type.name, str(key))
            high_key_to_code[high_key] = next_high_code
            high_level_specs[next_high_code] = HighLevelActionSpec(
                high_action_code=next_high_code,
                request_type=request_type,
                taxonomy_name=taxonomy.active_taxonomy,
                key=str(key),
                label=str(key),
            )
            next_high_code += 1
    return HierarchicalActionCodebook(
        taxonomy=taxonomy,
        low_level_specs={},
        high_level_specs=high_level_specs,
        low_key_to_code={},
        high_key_to_code=high_key_to_code,
        next_low_code=0,
    )


_DEFAULT_CODEBOOK_LOCK = RLock()
_DEFAULT_CODEBOOK: HierarchicalActionCodebook | None = None


def default_hierarchical_action_codebook() -> HierarchicalActionCodebook:
    global _DEFAULT_CODEBOOK
    with _DEFAULT_CODEBOOK_LOCK:
        if _DEFAULT_CODEBOOK is None:
            _DEFAULT_CODEBOOK = build_default_hierarchical_action_codebook()
        return _DEFAULT_CODEBOOK


def set_default_hierarchical_action_codebook(codebook: HierarchicalActionCodebook) -> None:
    global _DEFAULT_CODEBOOK
    with _DEFAULT_CODEBOOK_LOCK:
        _DEFAULT_CODEBOOK = codebook


def build_action_legality_engine() -> ActionLegalityEngine:
    return ActionLegalityEngine(default_hierarchical_action_codebook())

def high_level_code_for_spec(
    codebook: HierarchicalActionCodebook,
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> int:
    key = _high_level_key_for_spec(spec, visible_state=visible_state, acting_player=acting_player)
    high_code = codebook.high_key_to_code.get((spec.request_type.name, key))
    if high_code is None:
        categories = codebook.taxonomy.request_type_to_categories.get(spec.request_type.name, ())
        fallback_key = categories[0] if categories else "fallback"
        high_code = codebook.high_key_to_code[(spec.request_type.name, fallback_key)]
    return int(high_code)


def low_level_semantic_key_for_code(action_code: int) -> tuple[Any, ...]:
    return _low_level_semantic_key(default_hierarchical_action_codebook().low_spec(action_code))


def low_level_spec_for_code(action_code: int) -> LowLevelActionSpec:
    return default_hierarchical_action_codebook().low_spec(action_code)


def high_level_spec_for_code(high_action_code: int) -> HighLevelActionSpec:
    return default_hierarchical_action_codebook().high_spec(high_action_code)


def high_action_vocab_size() -> int:
    return len(default_hierarchical_action_codebook().high_level_specs)

def legal_low_level_specs(context: DecisionContext) -> tuple[LowLevelActionSpec, ...]:
    return tuple(low_level_spec_for_code(int(code)) for code in context.legal_low_level_codes)


def selected_low_level_code(step) -> int | None:
    action_code = int(getattr(step.choice, "action_code", -1))
    if action_code >= 0:
        return action_code
    return None


def selected_low_level_spec(step) -> LowLevelActionSpec | None:
    action_code = selected_low_level_code(step)
    if action_code is None:
        return None
    try:
        return low_level_spec_for_code(int(action_code))
    except KeyError:
        return None


def selected_high_level_code(step) -> int | None:
    value = getattr(step, "chosen_high_level_code", None)
    if value is not None:
        return int(value)
    if getattr(step, "high_to_low_map", ()):
        mapping = {int(high): tuple(int(code) for code in low_codes) for high, low_codes in step.high_to_low_map}
        action_code = selected_low_level_code(step)
        if action_code is not None:
            for high_code, low_codes in mapping.items():
                if int(action_code) in low_codes:
                    return int(high_code)
    spec = selected_low_level_spec(step)
    if spec is None:
        return None
    return None


def low_level_kind_for_code(action_code: int) -> OptionKind:
    return low_level_spec_for_code(int(action_code)).kind


def semantic_action_key_for_code(action_code: int) -> tuple[Any, ...]:
    spec = low_level_spec_for_code(int(action_code))
    return (
        ("request_type", spec.request_type.value),
        ("kind", spec.kind.value),
        ("subject_definition_id", int(spec.subject_definition_id)),
        (
            "skill_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_USE_SKILL else 0,
        ),
        (
            "card_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_PLAY_CARD else 0,
        ),
        ("candidate_definition_id", int(spec.select_card_definition_id)),
        ("removed_card_definition_id", int(spec.discarded_card_definition_id)),
        (
            "target_slots",
            tuple((target.owner, target.zone, int(target.index)) for target in spec.target_slots),
        ),
        ("used_dice", tuple(int(value) for value in sorted(spec.used_dice))),
        ("target_dice", int(spec.target_dice)),
    )


def public_spent_dice_for_code(action_code: int) -> int:
    spec = low_level_spec_for_code(int(action_code))
    if spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING:
        return 0
    return len(spec.used_dice)


def match_low_level_code_index(context: DecisionContext, reference_action_code: int) -> int | None:
    reference_key = semantic_action_key_for_code(int(reference_action_code))
    legal_codes = (
        tuple(int(code) for code in context.legal_low_level_codes)
        if context.legal_low_level_codes
        else tuple(int(spec.action_code) for spec in legal_low_level_specs(context))
    )
    for index, action_code in enumerate(legal_codes):
        candidate_key = semantic_action_key_for_code(int(action_code))
        if candidate_key == reference_key:
            return index
    return None


def legal_high_to_low_dict(context: DecisionContext) -> dict[int, tuple[int, ...]]:
    if context.high_to_low_map:
        return {int(high): tuple(int(code) for code in low_codes) for high, low_codes in context.high_to_low_map}
    return {}


def aggregate_high_policy(
    *,
    context: DecisionContext,
    low_level_policy: Sequence[float],
) -> tuple[float, ...]:
    high_dim = high_action_vocab_size()
    aggregated = [0.0] * high_dim
    high_to_low = legal_high_to_low_dict(context)
    probability_by_code = {
        int(code): float(low_level_policy[index])
        for index, code in enumerate(context.legal_low_level_codes)
    }
    for high_code, low_codes in high_to_low.items():
        aggregated[int(high_code)] = sum(probability_by_code.get(int(code), 0.0) for code in low_codes)
    total = sum(aggregated)
    if total > 0:
        aggregated = [value / total for value in aggregated]
    return tuple(aggregated)


def _low_level_semantic_key(spec: LowLevelActionSpec) -> tuple[Any, ...]:
    return (
        spec.request_type.value,
        spec.kind.value,
        int(spec.subject_definition_id),
        tuple((target.owner, target.zone, int(target.index)) for target in spec.target_slots),
        tuple(int(value) for value in sorted(spec.used_dice)),
        tuple(int(value) for value in sorted(spec.auto_selected_dice)),
        int(spec.choose_active_slot),
        int(spec.select_card_definition_id),
        int(spec.switch_hand_slot_mask),
        int(spec.reroll_dice_mask),
        int(spec.discarded_hand_slot),
        int(spec.discarded_card_definition_id),
        int(spec.target_dice),
    )

def _target_slots_from_metadata(
    metadata: dict[str, Any],
    *,
    acting_player: int,
) -> tuple[PublicSlotTarget, ...]:
    owner_ids = tuple(int(value) for value in metadata.get("target_owner_ids", ()))
    kinds = tuple(str(value) for value in metadata.get("target_kinds", ()))
    public_indices = tuple(int(value) for value in metadata.get("target_public_indices", ()))
    targets: list[PublicSlotTarget] = []
    for owner_id, kind, public_index in zip(owner_ids, kinds, public_indices):
        owner = "self" if int(owner_id) == int(acting_player) else "opponent"
        zone = "combat_status" if kind == "status" else str(kind)
        targets.append(PublicSlotTarget(owner=owner, zone=zone, index=int(public_index)))
    return tuple(targets)


def _choose_active_slot(
    metadata: dict[str, Any],
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> int:
    candidate_definition_id = int(metadata.get("candidate_definition_id", 0))
    for index, character in enumerate(visible_state.players[acting_player].characters):
        if int(character.definition_id) == candidate_definition_id:
            return index
    return -1


def _discarded_hand_slot(
    metadata: dict[str, Any],
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> int:
    removed_card_definition_id = int(metadata.get("removed_card_definition_id", 0))
    if removed_card_definition_id <= 0:
        return -1
    for index, card in enumerate(visible_state.players[acting_player].hand_cards):
        if int(card.definition_id) == removed_card_definition_id:
            return index
    return -1


def _switch_hand_slot_mask(
    metadata: dict[str, Any],
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> int:
    removed_definitions = [int(value) for value in metadata.get("removed_hand_definition_ids", ())]
    if not removed_definitions:
        return 0
    remaining = list(removed_definitions)
    mask = 0
    for index, card in enumerate(visible_state.players[acting_player].hand_cards):
        definition_id = int(card.definition_id)
        if definition_id in remaining:
            mask |= 1 << index
            remaining.remove(definition_id)
    return mask


def _reroll_dice_mask(
    metadata: dict[str, Any],
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> int:
    dice_to_reroll = [int(value) for value in metadata.get("dice_to_reroll", ())]
    if not dice_to_reroll:
        return 0
    remaining = list(dice_to_reroll)
    mask = 0
    for index, die in enumerate(visible_state.players[acting_player].dice):
        die = int(die)
        if die in remaining:
            mask |= 1 << index
            remaining.remove(die)
    return mask


def _high_level_key_for_spec(
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> str:
    priors = action_hierarchy_config()
    thresholds = priors["thresholds"]
    action_card_thresholds = priors["action_card_thresholds"]
    select_card_thresholds = priors["select_card_thresholds"]
    action_keys = priors["keys"]["action"]
    reroll_keys = priors["keys"]["reroll"]
    choose_keys = priors["keys"]["choose_active"]
    select_keys = priors["keys"]["select_card"]
    switch_keys = priors["keys"]["switch_hands"]
    if spec.request_type == DecisionType.ACTION:
        if spec.kind == OptionKind.ACTION_DECLARE_END:
            return str(action_keys["declare_end"])
        if spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING:
            return str(action_keys["elemental_tuning"])
        if spec.kind == OptionKind.ACTION_SWITCH_ACTIVE:
            return str(action_keys["switch_active"])
        if spec.kind == OptionKind.ACTION_USE_SKILL:
            if len(spec.target_slots) > 0 and len(spec.used_dice) >= int(thresholds["skill_commit_min_used_dice"]):
                return str(action_keys["skill_commit"])
            return str(action_keys["skill_setup"])
        if spec.kind == OptionKind.ACTION_PLAY_CARD:
            if spec.subject_definition_id >= int(action_card_thresholds["defense_heal_min"]):
                return str(action_keys["card_defense_heal"])
            if spec.subject_definition_id >= int(action_card_thresholds["board_setup_min"]):
                return str(action_keys["card_board_setup"])
            if spec.subject_definition_id >= int(action_card_thresholds["buff_equip_min"]):
                return str(action_keys["card_buff_equip"])
            if spec.subject_definition_id >= int(action_card_thresholds["resource_min"]):
                return str(action_keys["card_resource"])
            return str(action_keys["card_draw_search"])
    if spec.request_type == DecisionType.REROLL_DICE:
        reroll_count = _bit_count(spec.reroll_dice_mask)
        total_dice = max(1, len(visible_state.players[acting_player].dice))
        if reroll_count <= 0:
            return str(reroll_keys["keep_all"])
        if reroll_count >= total_dice:
            return str(reroll_keys["all"])
        if reroll_count <= int(thresholds["reroll_minor_max"]):
            return str(reroll_keys["minor"])
        return str(reroll_keys["major"])
    if spec.request_type == DecisionType.CHOOSE_ACTIVE:
        slot = int(spec.choose_active_slot)
        if 0 <= slot < len(visible_state.players[acting_player].characters):
            character = visible_state.players[acting_player].characters[slot]
            health_ratio = float(character.health) / max(1.0, float(character.max_health))
            if health_ratio <= float(thresholds["choose_safety_health_ratio"]):
                return str(choose_keys["safety"])
            if int(character.energy) >= max(1, int(character.max_energy) - int(thresholds["choose_setup_energy_gap"])):
                return str(choose_keys["setup"])
        return str(choose_keys["mainline"])
    if spec.request_type == DecisionType.SELECT_CARD:
        definition_id = int(spec.select_card_definition_id or spec.subject_definition_id)
        if definition_id >= int(select_card_thresholds["resource_min"]):
            return str(select_keys["resource"])
        if definition_id >= int(select_card_thresholds["curve_min"]):
            return str(select_keys["curve"])
        if definition_id >= int(select_card_thresholds["core_min"]):
            return str(select_keys["core"])
        return str(select_keys["flex"])
    if spec.request_type == DecisionType.SWITCH_HANDS:
        removed = _bit_count(spec.switch_hand_slot_mask)
        hand_size = max(1, len(visible_state.players[acting_player].hand_cards))
        if removed <= 0:
            return str(switch_keys["none"])
        if removed >= hand_size:
            return str(switch_keys["all"])
        if removed <= int(thresholds["switch_minor_max"]):
            return str(switch_keys["minor"])
        return str(switch_keys["major"])
    return codebook_default_high_key(spec.request_type)


def codebook_default_high_key(request_type: DecisionType) -> str:
    categories = load_action_taxonomy().request_type_to_categories.get(request_type.name, ())
    return categories[0] if categories else "fallback"


def _bit_count(mask: int) -> int:
    return int(mask).bit_count()
