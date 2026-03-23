from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence

from .action_semantic_rules import (
    active_action_card_groups,
    active_action_categories,
    active_action_entity_groups,
    active_action_priority_order,
    active_action_thresholds,
)
from .non_action_semantic_rules import (
    active_card_function_groups as active_non_action_card_function_groups,
    active_card_function_priority,
    active_choose_active_role_groups,
    active_choose_active_role_priority,
    active_reroll_character_elements,
)
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


@dataclass(frozen=True)
class ActionOutcomeFeatures:
    opponent_frontline_hp_loss: int
    opponent_backline_hp_loss_total: int
    opponent_frontline_lethal: bool
    self_frontline_tank_delta: int
    self_team_tank_delta: int
    self_team_hp_gain_count: int
    self_frontline_energy_delta: int
    self_burst_ready_delta: int
    self_hand_count_delta: int
    self_support_delta: int
    self_summon_delta: int
    self_durable_status_delta: int
    future_damage_asset_delta: int


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
    return high_level_code_for_key(codebook, spec.request_type, key)


def high_level_code_for_key(
    codebook: HierarchicalActionCodebook,
    request_type: DecisionType,
    key: str,
) -> int:
    high_code = codebook.high_key_to_code.get((request_type.name, str(key)))
    if high_code is None:
        categories = codebook.taxonomy.request_type_to_categories.get(request_type.name, ())
        fallback_key = categories[0] if categories else "fallback"
        high_code = codebook.high_key_to_code[(request_type.name, fallback_key)]
    return int(high_code)


def rebuild_encoded_action_set(
    *,
    legal_low_level_codes: Sequence[int],
    legal_low_level_mask: Sequence[bool],
    low_to_high_code: dict[int, int],
) -> EncodedLegalActionSet:
    unique_high_codes: list[int] = []
    high_to_low: dict[int, list[int]] = {}
    for action_code in legal_low_level_codes:
        high_code = int(low_to_high_code[int(action_code)])
        if high_code not in high_to_low:
            unique_high_codes.append(high_code)
            high_to_low[high_code] = []
        high_to_low[high_code].append(int(action_code))
    return EncodedLegalActionSet(
        legal_low_level_codes=tuple(int(code) for code in legal_low_level_codes),
        legal_low_level_mask=tuple(bool(value) for value in legal_low_level_mask),
        legal_high_level_codes=tuple(int(code) for code in unique_high_codes),
        high_to_low_map=tuple(
            (int(high_code), tuple(int(code) for code in codes))
            for high_code, codes in high_to_low.items()
        ),
    )


def low_level_semantic_key_for_code(action_code: int) -> tuple[Any, ...]:
    return _low_level_semantic_key(default_hierarchical_action_codebook().low_spec(action_code))


def low_level_spec_for_code(action_code: int) -> LowLevelActionSpec:
    return default_hierarchical_action_codebook().low_spec(action_code)


def try_low_level_spec_for_code(action_code: int) -> LowLevelActionSpec | None:
    try:
        return low_level_spec_for_code(int(action_code))
    except KeyError:
        return None


def high_level_spec_for_code(high_action_code: int) -> HighLevelActionSpec:
    return default_hierarchical_action_codebook().high_spec(high_action_code)


def high_action_vocab_size() -> int:
    return len(default_hierarchical_action_codebook().high_level_specs)

def legal_low_level_specs(context: DecisionContext) -> tuple[LowLevelActionSpec, ...]:
    if context.legal_low_level_specs:
        return tuple(context.legal_low_level_specs)
    recovered: list[LowLevelActionSpec] = []
    for code in context.legal_low_level_codes:
        spec = try_low_level_spec_for_code(int(code))
        if spec is not None:
            recovered.append(spec)
    return tuple(recovered)


def selected_low_level_code(step) -> int | None:
    action_code = int(getattr(step.choice, "action_code", -1))
    if action_code >= 0:
        return action_code
    return None


def selected_low_level_spec(step) -> LowLevelActionSpec | None:
    stored = getattr(step, "chosen_low_level_spec", None)
    if stored is not None:
        return stored
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

def try_low_level_kind_for_code(action_code: int) -> OptionKind | None:
    spec = try_low_level_spec_for_code(int(action_code))
    return spec.kind if spec is not None else None


def semantic_action_key_for_code(action_code: int) -> tuple[Any, ...]:
    spec = low_level_spec_for_code(int(action_code))
    return semantic_action_key_for_spec(spec)

def try_semantic_action_key_for_code(action_code: int) -> tuple[Any, ...] | None:
    spec = try_low_level_spec_for_code(int(action_code))
    if spec is None:
        return None
    return semantic_action_key_for_spec(spec)

def semantic_action_key_for_spec(spec: LowLevelActionSpec) -> tuple[Any, ...]:
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
        ("auto_selected_dice", tuple(int(value) for value in sorted(spec.auto_selected_dice))),
        ("choose_active_slot", int(spec.choose_active_slot)),
        ("switch_hand_slot_mask", int(spec.switch_hand_slot_mask)),
        ("reroll_dice_mask", int(spec.reroll_dice_mask)),
        ("discarded_hand_slot", int(spec.discarded_hand_slot)),
        ("target_dice", int(spec.target_dice)),
    )


def public_spent_dice_for_code(action_code: int) -> int:
    spec = low_level_spec_for_code(int(action_code))
    return public_spent_dice_for_spec(spec)


def try_public_spent_dice_for_code(action_code: int) -> int | None:
    spec = try_low_level_spec_for_code(int(action_code))
    if spec is None:
        return None
    return public_spent_dice_for_spec(spec)


def public_spent_dice_for_spec(spec: LowLevelActionSpec) -> int:
    if spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING:
        return 0
    return len(spec.used_dice)


def match_low_level_code_index(context: DecisionContext, reference_action_code: int) -> int | None:
    reference_key = semantic_action_key_for_code(int(reference_action_code))
    specs = legal_low_level_specs(context)
    if specs:
        for index, spec in enumerate(specs):
            candidate_key = semantic_action_key_for_spec(spec)
            if candidate_key == reference_key:
                return index
        return None
    for index, action_code in enumerate(context.legal_low_level_codes):
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
    legal_codes = tuple(int(code) for code in context.legal_low_level_codes)
    if len(low_level_policy) != len(legal_codes):
        print(
            "[aggregate-high-policy] "
            f"length_mismatch policy={len(low_level_policy)} legal={len(legal_codes)}",
            flush=True,
        )
    usable = min(len(low_level_policy), len(legal_codes))
    probability_by_code = {
        int(legal_codes[index]): float(low_level_policy[index])
        for index in range(usable)
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
    if spec.request_type == DecisionType.ACTION:
        return _provisional_action_high_level_key(
            spec,
            visible_state=visible_state,
            acting_player=acting_player,
        )
    if spec.request_type == DecisionType.REROLL_DICE:
        return _reroll_high_level_key(spec, visible_state=visible_state, acting_player=acting_player)
    if spec.request_type == DecisionType.CHOOSE_ACTIVE:
        return _choose_active_high_level_key(spec, visible_state=visible_state, acting_player=acting_player)
    if spec.request_type == DecisionType.SELECT_CARD:
        return _select_card_high_level_key(spec)
    if spec.request_type == DecisionType.SWITCH_HANDS:
        return _switch_hands_high_level_key(spec, visible_state=visible_state, acting_player=acting_player)
    return codebook_default_high_key(spec.request_type)


def classify_action_outcome_key(
    *,
    spec: LowLevelActionSpec,
    pre_state: StateSnapshot,
    post_state: StateSnapshot,
    acting_player: int,
) -> str:
    if spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING:
        return "elemental_tuning"
    if spec.kind == OptionKind.ACTION_SWITCH_ACTIVE:
        return "switch_character"
    if spec.kind == OptionKind.ACTION_DECLARE_END:
        return "yield_initiative"
    features = action_outcome_features(
        pre_state=pre_state,
        post_state=post_state,
        acting_player=acting_player,
    )
    return _classify_action_from_features(spec=spec, features=features)


def action_outcome_features(
    *,
    pre_state: StateSnapshot,
    post_state: StateSnapshot,
    acting_player: int,
) -> ActionOutcomeFeatures:
    opponent_player = 1 - int(acting_player)
    pre_self = pre_state.players[acting_player]
    post_self = post_state.players[acting_player]
    pre_opponent = pre_state.players[opponent_player]
    post_opponent = post_state.players[opponent_player]
    pre_self_active_index = _active_character_index(pre_self)
    post_self_active_index = _active_character_index(post_self)
    pre_opp_active_index = _active_character_index(pre_opponent)
    post_opp_active_index = _active_character_index(post_opponent)
    pre_frontline = _character_at(pre_opponent, pre_opp_active_index)
    post_frontline = _character_at(post_opponent, post_opp_active_index)
    pre_self_frontline = _character_at(pre_self, pre_self_active_index)
    post_self_frontline = _character_at(post_self, post_self_active_index)
    pre_frontline_hp = int(pre_frontline.health) if pre_frontline is not None else 0
    post_frontline_hp = int(post_frontline.health) if post_frontline is not None else 0
    opponent_frontline_hp_loss = max(0, pre_frontline_hp - post_frontline_hp)
    opponent_backline_hp_loss_total = _backline_hp_loss(
        pre_player=pre_opponent,
        post_player=post_opponent,
        active_index=pre_opp_active_index,
    )
    opponent_frontline_lethal = bool(
        pre_frontline is not None
        and not bool(pre_frontline.defeated)
        and post_frontline is not None
        and bool(post_frontline.defeated)
    )
    self_frontline_tank_delta = _player_frontline_tankiness(post_self) - _player_frontline_tankiness(pre_self)
    self_team_tank_delta = _player_team_tankiness(post_self) - _player_team_tankiness(pre_self)
    self_team_hp_gain_count = _team_hp_gain_count(pre_self, post_self)
    self_frontline_energy_delta = _frontline_energy(post_self_frontline) - _frontline_energy(pre_self_frontline)
    self_burst_ready_delta = _burst_ready_count(post_self) - _burst_ready_count(pre_self)
    self_hand_count_delta = len(post_self.hand_cards) - len(pre_self.hand_cards)
    self_support_delta = len(post_self.supports) - len(pre_self.supports)
    self_summon_delta = len(post_self.summons) - len(pre_self.summons)
    self_durable_status_delta = _durable_status_count(post_self) - _durable_status_count(pre_self)
    future_damage_asset_delta = _future_damage_asset_score(post_self) - _future_damage_asset_score(pre_self)
    return ActionOutcomeFeatures(
        opponent_frontline_hp_loss=int(opponent_frontline_hp_loss),
        opponent_backline_hp_loss_total=int(opponent_backline_hp_loss_total),
        opponent_frontline_lethal=bool(opponent_frontline_lethal),
        self_frontline_tank_delta=int(self_frontline_tank_delta),
        self_team_tank_delta=int(self_team_tank_delta),
        self_team_hp_gain_count=int(self_team_hp_gain_count),
        self_frontline_energy_delta=int(self_frontline_energy_delta),
        self_burst_ready_delta=int(self_burst_ready_delta),
        self_hand_count_delta=int(self_hand_count_delta),
        self_support_delta=int(self_support_delta),
        self_summon_delta=int(self_summon_delta),
        self_durable_status_delta=int(self_durable_status_delta),
        future_damage_asset_delta=int(future_damage_asset_delta),
    )


def _provisional_action_high_level_key(
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> str:
    categories = set(active_action_categories())
    card_groups = active_action_card_groups()
    if spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING and "elemental_tuning" in categories:
        return "elemental_tuning"
    if spec.kind == OptionKind.ACTION_SWITCH_ACTIVE and "switch_character" in categories:
        return "switch_character"
    if spec.kind == OptionKind.ACTION_DECLARE_END and "yield_initiative" in categories:
        return "yield_initiative"
    if spec.kind == OptionKind.ACTION_USE_SKILL:
        if _targets_opponent_frontline(spec):
            return "chip_frontline_hp"
        if _targets_opponent_backline(spec):
            return "chip_backline_hp"
        if _targets_self_character(spec, acting_player=acting_player):
            return "fortify_frontline"
        return "burst_setup"
    if spec.kind == OptionKind.ACTION_PLAY_CARD:
        subject_definition_id = int(spec.subject_definition_id)
        if subject_definition_id in card_groups.get("cycle_card_ids", frozenset()):
            return "cycle_for_cards"
        if subject_definition_id in card_groups.get("engine_card_ids", frozenset()):
            return "invest_engine_card"
        if subject_definition_id in card_groups.get("delayed_damage_card_ids", frozenset()):
            return "invest_delayed_damage"
        if subject_definition_id in card_groups.get("tech_card_ids", frozenset()):
            return "invest_tech"
        if _targets_self_character(spec, acting_player=acting_player):
            return "fortify_frontline"
    return "other_action"


def _classify_action_from_features(
    *,
    spec: LowLevelActionSpec,
    features: ActionOutcomeFeatures,
) -> str:
    thresholds = active_action_thresholds()
    card_groups = active_action_card_groups()
    priority_order = active_action_priority_order()
    subject_definition_id = int(spec.subject_definition_id)
    for key in priority_order:
        if key == "force_frontline_lethal" and features.opponent_frontline_lethal:
            return key
        if key == "chip_frontline_hp" and features.opponent_frontline_hp_loss >= int(thresholds["min_frontline_hp_loss"]):
            return key
        if key == "chip_backline_hp" and features.opponent_backline_hp_loss_total >= int(thresholds["min_backline_hp_loss"]):
            return key
        if key == "fortify_team":
            if (
                features.self_team_tank_delta >= int(thresholds["min_team_tank_gain"])
                and features.self_team_hp_gain_count >= int(thresholds["min_team_hp_gain_count"])
            ):
                return key
        if key == "fortify_frontline" and features.self_frontline_tank_delta >= int(thresholds["min_frontline_tank_gain"]):
            return key
        if key == "switch_character" and spec.kind == OptionKind.ACTION_SWITCH_ACTIVE:
            return key
        if key == "cycle_for_cards":
            if (
                features.self_hand_count_delta >= int(thresholds["min_hand_count_gain"])
                or subject_definition_id in card_groups.get("cycle_card_ids", frozenset())
            ):
                return key
        if key == "burst_setup":
            if (
                features.self_frontline_energy_delta >= int(thresholds["min_frontline_energy_gain"])
                or features.self_burst_ready_delta >= int(thresholds["min_burst_ready_gain"])
                or subject_definition_id in card_groups.get("burst_setup_card_ids", frozenset())
            ):
                return key
        if key == "invest_delayed_damage":
            if (
                features.future_damage_asset_delta >= int(thresholds["min_future_damage_asset_gain"])
                or subject_definition_id in card_groups.get("delayed_damage_card_ids", frozenset())
            ):
                return key
        if key == "invest_engine_card" and subject_definition_id in card_groups.get("engine_card_ids", frozenset()):
            return key
        if key == "invest_tech":
            if (
                subject_definition_id in card_groups.get("tech_card_ids", frozenset())
                or features.self_support_delta >= int(thresholds["min_support_gain"])
                or features.self_durable_status_delta >= int(thresholds["min_durable_status_gain"])
            ):
                return key
        if key == "elemental_tuning" and spec.kind == OptionKind.ACTION_ELEMENTAL_TUNING:
            return key
        if key == "yield_initiative" and spec.kind == OptionKind.ACTION_DECLARE_END:
            return key
    return "other_action"


def _reroll_high_level_key(
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> str:
    if int(spec.reroll_dice_mask) == 0:
        return "keep_all"
    player = visible_state.players[acting_player]
    visible_dice = tuple(int(value) for value in player.dice)
    if not visible_dice:
        return "keep_all"
    rerolled = [
        die
        for index, die in enumerate(visible_dice)
        if int(spec.reroll_dice_mask) & (1 << index)
    ]
    kept = [
        die
        for index, die in enumerate(visible_dice)
        if not (int(spec.reroll_dice_mask) & (1 << index))
    ]
    active_color, backline_colors = _team_color_profile(visible_state=visible_state, acting_player=acting_player)
    team_colors = {color for color in (active_color, *backline_colors) if int(color) > 0}
    kept_counts = Counter(int(die) for die in kept)
    rerolled_counts = Counter(int(die) for die in rerolled)
    active_kept = int(kept_counts.get(int(active_color), 0)) if active_color > 0 else 0
    active_rerolled = int(rerolled_counts.get(int(active_color), 0)) if active_color > 0 else 0
    backline_pairs = [
        (
            int(kept_counts.get(int(color), 0)),
            int(rerolled_counts.get(int(color), 0)),
        )
        for color in backline_colors
    ]
    off_kept = sum(
        int(count)
        for die, count in kept_counts.items()
        if int(die) not in team_colors
    )
    off_rerolled = sum(
        int(count)
        for die, count in rerolled_counts.items()
        if int(die) not in team_colors
    )
    candidates = (
        ("keep_active_and_two_backline", "keep", True, 2),
        ("keep_active_and_one_backline", "keep", True, 1),
        ("keep_active_only", "keep", True, 0),
        ("keep_two_backline", "keep", False, 2),
        ("keep_one_backline", "keep", False, 1),
        ("reroll_active_and_two_backline", "reroll", True, 2),
        ("reroll_active_and_one_backline", "reroll", True, 1),
        ("reroll_active_only", "reroll", True, 0),
        ("reroll_two_backline", "reroll", False, 2),
        ("reroll_one_backline", "reroll", False, 1),
    )
    best_label = "keep_all"
    best_score: float | None = None
    for label, family, include_active, backline_target_count in candidates:
        score = 0.0
        if active_color > 0:
            score += _alignment_score(
                target_keep=(include_active if family == "keep" else not include_active),
                kept=active_kept,
                rerolled=active_rerolled,
            )
        selected_backline = min(int(backline_target_count), len(backline_pairs))
        if family == "keep":
            deltas = [
                _alignment_score(target_keep=True, kept=kept_count, rerolled=rerolled_count)
                - _alignment_score(target_keep=False, kept=kept_count, rerolled=rerolled_count)
                for kept_count, rerolled_count in backline_pairs
            ]
            chosen = set(sorted(range(len(deltas)), key=lambda index: deltas[index], reverse=True)[:selected_backline])
            for index, (kept_count, rerolled_count) in enumerate(backline_pairs):
                score += _alignment_score(
                    target_keep=index in chosen,
                    kept=kept_count,
                    rerolled=rerolled_count,
                )
            score += _alignment_score(target_keep=False, kept=off_kept, rerolled=off_rerolled) * 0.5
        else:
            deltas = [
                _alignment_score(target_keep=False, kept=kept_count, rerolled=rerolled_count)
                - _alignment_score(target_keep=True, kept=kept_count, rerolled=rerolled_count)
                for kept_count, rerolled_count in backline_pairs
            ]
            chosen = set(sorted(range(len(deltas)), key=lambda index: deltas[index], reverse=True)[:selected_backline])
            for index, (kept_count, rerolled_count) in enumerate(backline_pairs):
                score += _alignment_score(
                    target_keep=index not in chosen,
                    kept=kept_count,
                    rerolled=rerolled_count,
                )
            score += _alignment_score(target_keep=True, kept=off_kept, rerolled=off_rerolled) * 0.5
        if best_score is None or score > best_score:
            best_label = label
            best_score = score
    return str(best_label)


def _choose_active_high_level_key(
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> str:
    slot = int(spec.choose_active_slot)
    if not (0 <= slot < len(visible_state.players[acting_player].characters)):
        return "choose_other"
    definition_id = int(visible_state.players[acting_player].characters[slot].definition_id)
    role_groups = active_choose_active_role_groups()
    for role in active_choose_active_role_priority():
        if definition_id in role_groups.get(role, frozenset()):
            return f"choose_{role}"
    return "choose_other"


def _select_card_high_level_key(spec: LowLevelActionSpec) -> str:
    definition_id = int(spec.select_card_definition_id or spec.subject_definition_id)
    group = _card_function_group_for_definition_id(definition_id)
    if group is None:
        return "select_other"
    if group == "core":
        return "select_core"
    return f"select_{group}"


def _switch_hands_high_level_key(
    spec: LowLevelActionSpec,
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> str:
    removed_definitions = _removed_hand_definition_ids(
        spec=spec,
        visible_state=visible_state,
        acting_player=acting_player,
    )
    if not removed_definitions:
        return "switch_keep_all"
    group_counts: Counter[str] = Counter()
    unknown_count = 0
    for definition_id in removed_definitions:
        group = _card_function_group_for_definition_id(int(definition_id))
        if group is None:
            unknown_count += 1
            continue
        group_counts[group] += 1
    if not group_counts:
        return "switch_other"
    top_group, top_count = max(
        group_counts.items(),
        key=lambda item: (int(item[1]), -_card_function_priority_index(item[0])),
    )
    total_count = sum(group_counts.values()) + int(unknown_count)
    if total_count == int(top_count):
        if top_group == "core":
            return "switch_core"
        return f"switch_{top_group}"
    return "switch_mixed"


def _removed_hand_definition_ids(
    *,
    spec: LowLevelActionSpec,
    visible_state: StateSnapshot,
    acting_player: int,
) -> tuple[int, ...]:
    removed: list[int] = []
    for index, card in enumerate(visible_state.players[acting_player].hand_cards):
        if int(spec.switch_hand_slot_mask) & (1 << index):
            removed.append(int(card.definition_id))
    return tuple(removed)


def _card_function_group_for_definition_id(definition_id: int) -> str | None:
    groups = active_non_action_card_function_groups()
    for group in active_card_function_priority():
        if int(definition_id) in groups.get(group, frozenset()):
            return str(group)
    return None


def _card_function_priority_index(group: str) -> int:
    order = active_card_function_priority()
    try:
        return int(order.index(str(group)))
    except ValueError:
        return len(order)


def _team_color_profile(
    *,
    visible_state: StateSnapshot,
    acting_player: int,
) -> tuple[int, tuple[int, ...]]:
    element_map = active_reroll_character_elements()
    player = visible_state.players[acting_player]
    active_index = _active_character_index(player)
    active_color = int(element_map.get(int(player.characters[active_index].definition_id), 0))
    backline_colors: list[int] = []
    seen = {active_color} if active_color > 0 else set()
    for index, character in enumerate(player.characters):
        if int(index) == int(active_index):
            continue
        color = int(element_map.get(int(character.definition_id), 0))
        if color <= 0 or color in seen:
            continue
        seen.add(color)
        backline_colors.append(color)
    return active_color, tuple(backline_colors)


def _alignment_score(*, target_keep: bool, kept: int, rerolled: int) -> float:
    if target_keep:
        return float(int(kept) - int(rerolled))
    return float(int(rerolled) - int(kept))


def _targets_opponent_frontline(spec: LowLevelActionSpec) -> bool:
    return any(
        target.owner == "opponent" and target.zone == "character" and int(target.index) == 0
        for target in spec.target_slots
    )


def _targets_opponent_backline(spec: LowLevelActionSpec) -> bool:
    return any(
        target.owner == "opponent" and target.zone == "character" and int(target.index) > 0
        for target in spec.target_slots
    )


def _targets_self_character(spec: LowLevelActionSpec, *, acting_player: int) -> bool:
    del acting_player
    return any(target.owner == "self" and target.zone == "character" for target in spec.target_slots)


def _active_character_index(player) -> int:
    for index, character in enumerate(player.characters):
        if bool(character.is_active):
            return int(index)
    return 0


def _character_at(player, index: int):
    if 0 <= int(index) < len(player.characters):
        return player.characters[int(index)]
    return None


def _frontline_energy(character) -> int:
    if character is None:
        return 0
    return int(character.energy)


def _burst_ready_count(player) -> int:
    return sum(
        1
        for character in player.characters
        if int(character.energy) >= int(character.max_energy)
    )


def _backline_hp_loss(*, pre_player, post_player, active_index: int) -> int:
    loss = 0
    for index, (pre_character, post_character) in enumerate(zip(pre_player.characters, post_player.characters, strict=False)):
        if int(index) == int(active_index):
            continue
        loss += max(0, int(pre_character.health) - int(post_character.health))
    return int(loss)


def _player_frontline_tankiness(player) -> int:
    active_index = _active_character_index(player)
    character = _character_at(player, active_index)
    if character is None:
        return 0
    defensive_entity_ids = active_action_entity_groups().get("defensive_entity_definition_ids", frozenset())
    value = int(character.health)
    value += _entity_bonus(tuple(character.entities), definition_ids=defensive_entity_ids)
    value += _entity_bonus(player.combat_statuses, definition_ids=defensive_entity_ids)
    return int(value)


def _player_team_tankiness(player) -> int:
    defensive_entity_ids = active_action_entity_groups().get("defensive_entity_definition_ids", frozenset())
    value = sum(int(character.health) for character in player.characters)
    value += sum(_entity_bonus(tuple(character.entities), definition_ids=defensive_entity_ids) for character in player.characters)
    value += _entity_bonus(player.combat_statuses, definition_ids=defensive_entity_ids)
    value += _entity_bonus(player.supports, definition_ids=defensive_entity_ids)
    return int(value)


def _team_hp_gain_count(pre_player, post_player) -> int:
    count = 0
    for pre_character, post_character in zip(pre_player.characters, post_player.characters, strict=False):
        if int(post_character.health) > int(pre_character.health):
            count += 1
    return int(count)


def _durable_status_count(player) -> int:
    return int(len(player.combat_statuses) + len(player.supports) + sum(len(character.entities) for character in player.characters))


def _future_damage_asset_score(player) -> int:
    offensive_entity_ids = active_action_entity_groups().get("offensive_entity_definition_ids", frozenset())
    value = len(player.summons)
    value += _entity_bonus(player.combat_statuses, definition_ids=offensive_entity_ids)
    value += sum(_entity_bonus(tuple(character.entities), definition_ids=offensive_entity_ids) for character in player.characters)
    return int(value)


def _entity_bonus(entities: Sequence[Any], *, definition_ids: frozenset[int]) -> int:
    if not definition_ids:
        return 0
    bonus = 0
    for entity in entities:
        if int(entity.definition_id) not in definition_ids:
            continue
        variables = getattr(entity, "variables", {})
        if not variables:
            bonus += 1
            continue
        bonus += max(1, sum(max(0, int(value)) for value in variables.values()))
    return int(bonus)


def codebook_default_high_key(request_type: DecisionType) -> str:
    categories = load_action_taxonomy().request_type_to_categories.get(request_type.name, ())
    return categories[0] if categories else "fallback"


def _bit_count(mask: int) -> int:
    return int(mask).bit_count()
