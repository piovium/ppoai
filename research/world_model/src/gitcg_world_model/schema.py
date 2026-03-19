from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionType(str, Enum):
    ACTION = "action"
    CHOOSE_ACTIVE = "choose_active"
    REROLL_DICE = "reroll_dice"
    SELECT_CARD = "select_card"
    SWITCH_HANDS = "switch_hands"
    TERMINAL = "terminal"


class OptionKind(str, Enum):
    ACTION_USE_SKILL = "action_use_skill"
    ACTION_PLAY_CARD = "action_play_card"
    ACTION_SWITCH_ACTIVE = "action_switch_active"
    ACTION_ELEMENTAL_TUNING = "action_elemental_tuning"
    ACTION_DECLARE_END = "action_declare_end"
    CHOOSE_ACTIVE = "choose_active"
    SELECT_CARD = "select_card"
    REROLL_DICE = "reroll_dice"
    SWITCH_HANDS = "switch_hands"


@dataclass(frozen=True)
class DeckSpec:
    name: str
    characters: tuple[int, ...]
    cards: tuple[int, ...]


@dataclass(frozen=True)
class Matchup:
    deck0: str
    deck1: str

    @property
    def key(self) -> str:
        return f"{self.deck0}__vs__{self.deck1}"


@dataclass(frozen=True)
class EnvConfig:
    deck_pool: tuple[DeckSpec, ...]
    version: str | None = None
    max_rounds: int | None = None
    seed: int | None = None
    seed_policy: str = "provided_or_config"
    record_full_state_json: bool = True
    record_player_view: bool = True
    draw_penalty: float = 2.0


@dataclass(frozen=True)
class EntitySnapshot:
    id: int
    definition_id: int
    variables: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterSnapshot:
    id: int
    definition_id: int
    health: int
    max_health: int
    energy: int
    max_energy: int
    defeated: bool
    is_active: bool
    aura: int = 0
    entities: tuple[EntitySnapshot, ...] = ()


@dataclass(frozen=True)
class PlayerSnapshot:
    active_character_id: int | None
    characters: tuple[CharacterSnapshot, ...]
    combat_statuses: tuple[EntitySnapshot, ...]
    summons: tuple[EntitySnapshot, ...]
    supports: tuple[EntitySnapshot, ...]
    hand_cards: tuple[EntitySnapshot, ...]
    pile_cards: tuple[EntitySnapshot, ...]
    dice: tuple[int, ...]
    declared_end: bool
    legend_used: bool
    can_charged: bool = False
    can_plunging: bool = False
    has_defeated: bool = False


@dataclass(frozen=True)
class StateSnapshot:
    phase: str
    round_number: int
    current_turn: int
    winner: int | None
    players: tuple[PlayerSnapshot, PlayerSnapshot]


@dataclass(frozen=True)
class PublicSlotTarget:
    owner: str
    zone: str
    index: int


@dataclass(frozen=True)
class LowLevelActionSpec:
    action_code: int
    request_type: DecisionType
    kind: OptionKind
    label: str
    subject_definition_id: int = 0
    target_slots: tuple[PublicSlotTarget, ...] = ()
    used_dice: tuple[int, ...] = ()
    auto_selected_dice: tuple[int, ...] = ()
    choose_active_slot: int = -1
    select_card_definition_id: int = 0
    switch_hand_slot_mask: int = 0
    reroll_dice_mask: int = 0
    discarded_hand_slot: int = -1
    discarded_card_definition_id: int = 0
    target_dice: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HighLevelActionSpec:
    high_action_code: int
    request_type: DecisionType
    taxonomy_name: str
    key: str
    label: str


@dataclass(frozen=True)
class ActionTaxonomy:
    active_taxonomy: str
    request_type_to_categories: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class EncodedLegalActionSet:
    legal_low_level_codes: tuple[int, ...]
    legal_low_level_mask: tuple[bool, ...]
    legal_high_level_codes: tuple[int, ...]
    high_to_low_map: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class DecisionContext:
    acting_player: int
    request_type: DecisionType
    step_index: int
    full_state: StateSnapshot
    legal_low_level_codes: tuple[int, ...] = ()
    legal_low_level_mask: tuple[bool, ...] = ()
    legal_high_level_codes: tuple[int, ...] = ()
    high_to_low_map: tuple[tuple[int, tuple[int, ...]], ...] = ()
    player_view: StateSnapshot | None = None
    full_state_json: str | None = None
    terminal: bool = False
    reward: float = 0.0
    request_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionChoice:
    action_code: int = -1


@dataclass(frozen=True)
class TrajectoryStep:
    acting_player: int
    request_type: DecisionType
    choice: ActionChoice
    pre_state: StateSnapshot
    post_state: StateSnapshot
    reward: float
    done: bool
    legal_low_level_codes: tuple[int, ...] = ()
    legal_low_level_mask: tuple[bool, ...] = ()
    legal_high_level_codes: tuple[int, ...] = ()
    high_to_low_map: tuple[tuple[int, tuple[int, ...]], ...] = ()
    chosen_high_level_code: int | None = None
    player_view: StateSnapshot | None = None
    full_state_json_before: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeRecord:
    matchup: str
    seed: int | None
    winner: int | None
    steps: tuple[TrajectoryStep, ...]
    final_state: StateSnapshot
    final_state_json: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
