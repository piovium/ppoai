from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .action_hierarchy import (
    public_spent_dice_for_code,
    public_spent_dice_for_spec,
    selected_low_level_code,
    selected_low_level_spec,
    semantic_action_key_for_code,
    semantic_action_key_for_spec,
)
from .schema import (
    CharacterSnapshot,
    DecisionContext,
    DecisionType,
    EntitySnapshot,
    PlayerSnapshot,
    StateSnapshot,
    TrajectoryStep,
)


@dataclass(frozen=True)
class PublicActionRecord:
    acting_player: int
    request_type: DecisionType
    round_number: int
    phase: str
    payload: dict[str, Any] | None
    semantic_key: tuple[tuple[str, object], ...] | None
    option_kind: str | None
    used_dice_count: int
    known_payload: bool
    selected_legal_index: int | None
    legal_option_count: int


@dataclass(frozen=True)
class ExplicitRangeHypothesis:
    hand_definition_ids: tuple[int, ...]
    deck_definition_ids: tuple[int, ...]
    hidden_dice: tuple[int, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class ExplicitRange:
    hypotheses: tuple[ExplicitRangeHypothesis, ...]

    @property
    def count(self) -> int:
        return len(self.hypotheses)


@dataclass(frozen=True)
class PublicBeliefState:
    root_player: int
    acting_player: int
    public_state: StateSnapshot
    round_number: int
    phase: str
    opponent_public_dice_count: float
    opponent_public_hand_count: float
    self_range: ExplicitRange
    opponent_range: ExplicitRange
    public_history: tuple[PublicActionRecord, ...]
    revealed_opponent_card_definition_ids: tuple[int, ...]


class PublicStateTracker:
    def __init__(self, *, player_id: int, initial_dice_per_round: int = 8) -> None:
        self.player_id = int(player_id)
        self.initial_dice_per_round = int(initial_dice_per_round)
        self._history: list[PublicActionRecord] = []
        self._opponent_spent_by_round: dict[int, int] = {}

    def reset(self) -> None:
        self._history.clear()
        self._opponent_spent_by_round.clear()

    def clone(self, *, player_id: int | None = None) -> "PublicStateTracker":
        copied = PublicStateTracker(
            player_id=self.player_id if player_id is None else int(player_id),
            initial_dice_per_round=self.initial_dice_per_round,
        )
        for record in self._history:
            copied.observe_public_record(record)
        return copied

    def observe_public_record(self, record: PublicActionRecord) -> None:
        self._history.append(record)
        if record.acting_player == self.player_id or record.request_type != DecisionType.ACTION:
            return
        spent = int(record.used_dice_count)
        if spent <= 0:
            return
        self._opponent_spent_by_round[record.round_number] = (
            self._opponent_spent_by_round.get(record.round_number, 0) + spent
        )

    def observe_step(self, step: TrajectoryStep) -> None:
        selected_spec = selected_low_level_spec(step)
        action_code = selected_low_level_code(step)
        payload: dict[str, Any] | None = None
        known_payload = False
        selected_legal_index: int | None = None
        if action_code is not None:
            for index, candidate_code in enumerate(step.legal_low_level_codes):
                if int(candidate_code) == int(action_code):
                    selected_legal_index = index
                    break
            if step.acting_player == self.player_id or step.request_type in {
                DecisionType.ACTION,
                DecisionType.CHOOSE_ACTIVE,
            }:
                payload = {"action_code": int(action_code)}
                known_payload = True
        round_number = int(step.player_view.round_number) if step.player_view is not None else 0
        phase = str(step.player_view.phase) if step.player_view is not None else ""
        record = PublicActionRecord(
            acting_player=int(step.acting_player),
            request_type=step.request_type,
            round_number=round_number,
            phase=phase,
            payload=payload,
            semantic_key=(
                semantic_action_key_for_spec(selected_spec)
                if selected_spec is not None
                else (semantic_action_key_for_code(action_code) if action_code is not None else None)
            ),
            option_kind=(selected_spec.kind.value if selected_spec is not None else None),
            used_dice_count=(
                public_spent_dice_for_spec(selected_spec)
                if selected_spec is not None
                else (public_spent_dice_for_code(action_code) if action_code is not None else 0)
            ),
            known_payload=known_payload,
            selected_legal_index=selected_legal_index,
            legal_option_count=len(step.legal_low_level_codes),
        )
        self.observe_public_record(record)

    def public_history(self) -> tuple[PublicActionRecord, ...]:
        return tuple(self._history)

    def opponent_public_dice_count(self, context: DecisionContext) -> float:
        if context.player_view is None:
            return 0.0
        return self.opponent_public_dice_count_from_state(context.player_view)

    def opponent_public_dice_count_from_state(self, state: StateSnapshot) -> float:
        phase = str(state.phase)
        if phase not in {"roll", "action", "end"}:
            return 0.0
        round_number = int(state.round_number)
        spent = int(self._opponent_spent_by_round.get(round_number, 0))
        return float(max(0, min(self.initial_dice_per_round, self.initial_dice_per_round - spent)))

    def opponent_public_hand_count(self, context: DecisionContext) -> float:
        if context.player_view is None:
            return 0.0
        return self.opponent_public_hand_count_from_state(context.player_view)

    def opponent_public_hand_count_from_state(self, state: StateSnapshot) -> float:
        opponent = state.players[1 - self.player_id]
        return float(len(opponent.hand_cards))

    def revealed_opponent_card_definition_ids(self) -> tuple[int, ...]:
        definitions: list[int] = []
        for record in self._history:
            if record.acting_player == self.player_id or record.semantic_key is None:
                continue
            semantic = dict(record.semantic_key)
            for key in ("card_definition_id", "removed_card_definition_id", "candidate_definition_id"):
                value = semantic.get(key)
                if isinstance(value, int) and value > 0:
                    definitions.append(int(value))
            target_values = semantic.get("target_definition_ids")
            if isinstance(target_values, tuple):
                for value in target_values:
                    if isinstance(value, int) and value > 0:
                        definitions.append(int(value))
        return tuple(definitions)

    def revealed_opponent_card_counts(self) -> dict[int, int]:
        return dict(Counter(self.revealed_opponent_card_definition_ids()))

    @classmethod
    def from_public_history(
        cls,
        *,
        player_id: int,
        history: tuple[PublicActionRecord, ...],
        initial_dice_per_round: int = 8,
    ) -> "PublicStateTracker":
        tracker = cls(player_id=player_id, initial_dice_per_round=initial_dice_per_round)
        for record in history:
            tracker.observe_public_record(record)
        return tracker


def mask_state_for_player(state: StateSnapshot, *, perspective_player: int) -> StateSnapshot:
    players: list[PlayerSnapshot] = []
    for player_index, player in enumerate(state.players):
        if player_index == perspective_player:
            players.append(player)
            continue
        players.append(
            PlayerSnapshot(
                active_character_id=player.active_character_id,
                characters=tuple(
                    CharacterSnapshot(
                        id=character.id,
                        definition_id=character.definition_id,
                        health=character.health,
                        max_health=character.max_health,
                        energy=character.energy,
                        max_energy=character.max_energy,
                        defeated=character.defeated,
                        is_active=character.is_active,
                        aura=character.aura,
                        entities=character.entities,
                    )
                    for character in player.characters
                ),
                combat_statuses=player.combat_statuses,
                summons=player.summons,
                supports=player.supports,
                hand_cards=tuple(
                    EntitySnapshot(id=0, definition_id=0, variables={})
                    for _ in player.hand_cards
                ),
                pile_cards=tuple(
                    EntitySnapshot(id=0, definition_id=0, variables={})
                    for _ in player.pile_cards
                ),
                dice=tuple(),
                declared_end=player.declared_end,
                legend_used=player.legend_used,
                can_charged=player.can_charged,
                can_plunging=player.can_plunging,
                has_defeated=player.has_defeated,
            )
        )
    return StateSnapshot(
        phase=state.phase,
        round_number=state.round_number,
        current_turn=state.current_turn,
        winner=state.winner,
        players=(players[0], players[1]),
    )
