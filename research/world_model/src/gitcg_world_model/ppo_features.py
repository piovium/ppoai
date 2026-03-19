from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .action_hierarchy import (
    aggregate_high_policy,
    default_hierarchical_action_codebook,
    high_action_vocab_size,
    legal_high_to_low_dict,
    low_level_spec_for_code,
    selected_low_level_spec,
)
from .belief_groups import belief_group_dim, encode_group_presence
from .event_features import EventFeatureEncoder
from .opponent_identity import (
    infer_opponent_deck_name,
    opponent_tag_to_id,
    opponent_tags,
    stable_hash_id,
)
from .schema import DecisionContext, DecisionType, OptionKind, StateSnapshot, TrajectoryStep

_TOKEN_TYPES = (
    "global",
    "player_summary",
    "character",
    "character_entity",
    "status",
    "summon",
    "support",
    "hand_card",
    "dice",
    "history",
)
_OWNER_TYPES = ("self", "opponent", "neutral")
_PHASES = ("init_hands", "init_actives", "roll", "action", "end", "game_end")
_REQUEST_TYPES = tuple(DecisionType)
_OPTION_KINDS = tuple(OptionKind)
_OPPONENT_TAGS = opponent_tags()
_AURA_TYPES = (0, 1, 2, 3, 4, 5, 6)
_SCALAR_DIM = 24


@dataclass(frozen=True)
class TokenObservationEncoderConfig:
    card_vocabulary: tuple[int, ...]
    character_vocabulary: tuple[int, ...]
    opponent_entity_vocab_size: int = 257
    deck_vocab_size: int = 65
    max_history_tokens: int = 32
    max_character_entities_per_character: int = 3
    max_status_tokens_per_player: int = 6
    max_summon_tokens_per_player: int = 4
    max_support_tokens_per_player: int = 4
    max_hand_tokens: int = 10
    max_dice_tokens: int = 8
    include_belief_targets: bool = True
    include_opponent_tag: bool = True


@dataclass(frozen=True)
class BeliefTarget:
    hand_histogram: tuple[float, ...]
    hand_size: float
    remaining_deck_histogram: tuple[float, ...]
    burst_ready_count: float
    group_hand_presence: tuple[float, ...]
    group_remaining_deck_presence: tuple[float, ...]


@dataclass(frozen=True)
class EncodedObservation:
    token_features: tuple[tuple[float, ...], ...]
    token_mask: tuple[bool, ...]
    opponent_token_mask: tuple[bool, ...]
    option_features: tuple[tuple[float, ...], ...]
    option_mask: tuple[bool, ...]
    low_level_action_codes: tuple[int, ...]
    low_to_high_codes: tuple[int, ...]
    high_level_mask: tuple[bool, ...]
    privileged_state: tuple[float, ...]
    belief_target: BeliefTarget
    opponent_tag_id: int
    opponent_entity_id: int
    opponent_deck_id: int


class TokenObservationEncoder:
    def __init__(self, config: TokenObservationEncoderConfig | None = None) -> None:
        if config is None:
            base = EventFeatureEncoder()
            config = TokenObservationEncoderConfig(
                card_vocabulary=base.card_vocabulary,
                character_vocabulary=base.character_vocabulary,
            )
        self.config = config
        self.event = EventFeatureEncoder(
            card_vocabulary=config.card_vocabulary,
            character_vocabulary=config.character_vocabulary,
        )
        self.codebook = default_hierarchical_action_codebook()
        self.token_dim = (
            len(_TOKEN_TYPES)
            + len(_OWNER_TYPES)
            + len(_PHASES)
            + len(_REQUEST_TYPES)
            + len(_OPTION_KINDS)
            + (len(_OPPONENT_TAGS) if config.include_opponent_tag else 0)
            + _SCALAR_DIM
        )
        self.option_dim = self.event.option_dim
        self.high_action_dim = high_action_vocab_size()
        self.privileged_state_dim = self.event.privileged_observation_dim
        self.belief_histogram_dim = len(config.card_vocabulary)
        self.belief_remaining_deck_histogram_dim = self.belief_histogram_dim
        self.belief_dim = self.belief_histogram_dim + 1
        self.belief_group_dim = belief_group_dim()
        self.max_tokens = (
            1
            + 2
            + 6
            + (6 * config.max_character_entities_per_character)
            + (2 * config.max_status_tokens_per_player)
            + (2 * config.max_summon_tokens_per_player)
            + (2 * config.max_support_tokens_per_player)
            + config.max_hand_tokens
            + config.max_dice_tokens
            + config.max_history_tokens
        )
        self._card_index = {value: index for index, value in enumerate(config.card_vocabulary)}

    @property
    def opponent_tag_vocab_size(self) -> int:
        return len(_OPPONENT_TAGS)

    @property
    def opponent_entity_vocab_size(self) -> int:
        return int(self.config.opponent_entity_vocab_size)

    @property
    def deck_vocab_size(self) -> int:
        return int(self.config.deck_vocab_size)

    def to_dict(self) -> dict[str, object]:
        return asdict(self.config)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TokenObservationEncoder":
        return cls(
            TokenObservationEncoderConfig(
                card_vocabulary=tuple(int(value) for value in payload["card_vocabulary"]),
                character_vocabulary=tuple(int(value) for value in payload["character_vocabulary"]),
                opponent_entity_vocab_size=int(payload.get("opponent_entity_vocab_size", 257)),
                deck_vocab_size=int(payload.get("deck_vocab_size", 65)),
                max_history_tokens=int(payload.get("max_history_tokens", 32)),
                max_character_entities_per_character=int(payload.get("max_character_entities_per_character", 3)),
                max_status_tokens_per_player=int(payload.get("max_status_tokens_per_player", 6)),
                max_summon_tokens_per_player=int(payload.get("max_summon_tokens_per_player", 4)),
                max_support_tokens_per_player=int(payload.get("max_support_tokens_per_player", 4)),
                max_hand_tokens=int(payload.get("max_hand_tokens", 10)),
                max_dice_tokens=int(payload.get("max_dice_tokens", 8)),
                include_belief_targets=bool(payload.get("include_belief_targets", True)),
                include_opponent_tag=bool(payload.get("include_opponent_tag", False)),
            )
        )

    def encode_context(
        self,
        context: DecisionContext,
        *,
        history: Sequence[TrajectoryStep] = (),
        opponent_tag: str = "unknown",
        opponent_entity_key: str | None = None,
        opponent_deck_name: str | None = None,
        opponent_public_dice_count: float | None = None,
        include_training_targets: bool = True,
    ) -> EncodedObservation:
        if context.player_view is None:
            raise ValueError("player_view is required for PPO observation encoding")
        tokens: list[tuple[float, ...]] = []
        opponent_token_mask: list[bool] = []
        acting_player = context.acting_player
        visible_state = context.player_view
        tokens.append(self._encode_global_token(context, opponent_tag=opponent_tag))
        opponent_token_mask.append(False)
        for player_index, player in enumerate(visible_state.players):
            owner = self._owner_label(acting_player, player_index)
            visible_dice_count = float(len(player.dice))
            if owner == "opponent" and opponent_public_dice_count is not None:
                visible_dice_count = max(0.0, min(8.0, float(opponent_public_dice_count)))
            tokens.append(
                self._build_token(
                    token_type="player_summary",
                    owner=owner,
                    request_type=context.request_type,
                    round_number=visible_state.round_number,
                    scalars=(
                        self._safe_ratio(len(player.characters), 3.0),
                        self._safe_ratio(len(player.hand_cards), 10.0),
                        self._safe_ratio(visible_dice_count, 8.0),
                        self._safe_ratio(len(player.combat_statuses), 8.0),
                        self._safe_ratio(len(player.summons), 8.0),
                        self._safe_ratio(len(player.supports), 8.0),
                        1.0 if player.declared_end else 0.0,
                        1.0 if player.legend_used else 0.0,
                    ),
                )
            )
            opponent_token_mask.append(owner == "opponent")
            for character in player.characters[:3]:
                tokens.append(
                    self._build_token(
                        token_type="character",
                        owner=owner,
                        definition_id=character.definition_id,
                        request_type=context.request_type,
                        round_number=visible_state.round_number,
                        scalars=(
                            self._safe_ratio(character.health, max(1, character.max_health)),
                            self._safe_ratio(character.max_health, 10.0),
                            self._safe_ratio(character.energy, max(1, character.max_energy)),
                            self._safe_ratio(character.max_energy, 5.0),
                            1.0 if character.defeated else 0.0,
                            1.0 if character.is_active else 0.0,
                            self._safe_ratio(len(character.entities), 8.0),
                            *(
                                1.0 if character.aura == aura_value else 0.0
                                for aura_value in _AURA_TYPES
                            ),
                        ),
                    )
                )
                opponent_token_mask.append(owner == "opponent")
                for entity in character.entities[: self.config.max_character_entities_per_character]:
                    tokens.append(
                        self._encode_entity_token(
                            token_type="character_entity",
                            owner=owner,
                            definition_id=entity.definition_id,
                            variables=entity.variables,
                            round_number=visible_state.round_number,
                            request_type=context.request_type,
                        )
                    )
                    opponent_token_mask.append(owner == "opponent")
            for entity in player.combat_statuses[: self.config.max_status_tokens_per_player]:
                tokens.append(
                    self._encode_entity_token(
                        token_type="status",
                        owner=owner,
                        definition_id=entity.definition_id,
                        variables=entity.variables,
                        round_number=visible_state.round_number,
                        request_type=context.request_type,
                    )
                )
                opponent_token_mask.append(owner == "opponent")
            for entity in player.summons[: self.config.max_summon_tokens_per_player]:
                tokens.append(
                    self._encode_entity_token(
                        token_type="summon",
                        owner=owner,
                        definition_id=entity.definition_id,
                        variables=entity.variables,
                        round_number=visible_state.round_number,
                        request_type=context.request_type,
                    )
                )
                opponent_token_mask.append(owner == "opponent")
            for entity in player.supports[: self.config.max_support_tokens_per_player]:
                tokens.append(
                    self._encode_entity_token(
                        token_type="support",
                        owner=owner,
                        definition_id=entity.definition_id,
                        variables=entity.variables,
                        round_number=visible_state.round_number,
                        request_type=context.request_type,
                    )
                )
                opponent_token_mask.append(owner == "opponent")
        acting_snapshot = visible_state.players[acting_player]
        for card in acting_snapshot.hand_cards[: self.config.max_hand_tokens]:
            tokens.append(
                self._build_token(
                    token_type="hand_card",
                    owner="self",
                    definition_id=card.definition_id,
                    request_type=context.request_type,
                    round_number=visible_state.round_number,
                )
            )
            opponent_token_mask.append(False)
        for die in acting_snapshot.dice[: self.config.max_dice_tokens]:
            tokens.append(
                self._build_token(
                    token_type="dice",
                    owner="self",
                    request_type=context.request_type,
                    round_number=visible_state.round_number,
                    scalars=(self._safe_ratio(int(die), 8.0),),
                )
            )
            opponent_token_mask.append(False)
        for step in history[-self.config.max_history_tokens :]:
            selected_low_spec = self._selected_low_spec(step)
            option_kind = selected_low_spec.kind if selected_low_spec is not None else OptionKind.ACTION_DECLARE_END
            target_count = len(selected_low_spec.target_slots) if selected_low_spec is not None else 0
            used_dice_count = len(selected_low_spec.used_dice) if selected_low_spec is not None else 0
            tokens.append(
                self._build_token(
                    token_type="history",
                    owner=self._owner_label(acting_player, step.acting_player),
                    option_kind=option_kind,
                    request_type=step.request_type,
                    round_number=step.pre_state.round_number,
                    scalars=(
                        self._safe_ratio(len(step.legal_low_level_codes), 32.0),
                        self._safe_ratio(target_count, 4.0),
                        self._safe_ratio(used_dice_count, 8.0),
                        step.reward,
                        1.0 if step.done else 0.0,
                    ),
                )
            )
            opponent_token_mask.append(self._owner_label(acting_player, step.acting_player) == "opponent")
        tokens = tokens[: self.max_tokens]
        opponent_token_mask = opponent_token_mask[: self.max_tokens]
        low_level_codes = tuple(int(code) for code in context.legal_low_level_codes)
        low_to_high = {
            int(code): int(high_code)
            for high_code, low_codes in context.high_to_low_map
            for code in low_codes
        }
        option_features = tuple(
            self.event.encode_public_low_level_spec(low_level_spec_for_code(code), context)
            for code in low_level_codes
        )
        option_mask = tuple(bool(value) for value in context.legal_low_level_mask)
        high_mask = [False] * high_action_vocab_size()
        for high_code in context.legal_high_level_codes:
            if 0 <= int(high_code) < len(high_mask):
                high_mask[int(high_code)] = True
        if include_training_targets:
            privileged_state = self.event.encode_privileged_state(context.full_state)
            belief_target = self.encode_belief_target(context.full_state, acting_player=acting_player)
        else:
            privileged_state = tuple(0.0 for _ in range(self.privileged_state_dim))
            belief_target = BeliefTarget(
                hand_histogram=tuple(0.0 for _ in range(self.belief_histogram_dim)),
                hand_size=0.0,
                remaining_deck_histogram=tuple(0.0 for _ in range(self.belief_remaining_deck_histogram_dim)),
                burst_ready_count=0.0,
                group_hand_presence=tuple(0.0 for _ in range(self.belief_group_dim)),
                group_remaining_deck_presence=tuple(0.0 for _ in range(self.belief_group_dim)),
            )
        entity_key = opponent_entity_key or str(context.metadata.get("opponent_entity_key") or "unknown")
        deck_name = opponent_deck_name or str(
            context.metadata.get("opponent_deck_name")
            or infer_opponent_deck_name(
                matchup_key=str(context.metadata.get("matchup", "")),
                acting_player=acting_player,
                opponent_kind=str(context.metadata.get("opponent_kind", "")),
                opponent_source=str(context.metadata.get("opponent_source", "")),
            )
        )
        return EncodedObservation(
            token_features=tuple(tokens),
            token_mask=tuple(True for _ in tokens),
            opponent_token_mask=tuple(opponent_token_mask),
            option_features=option_features,
            option_mask=option_mask,
            low_level_action_codes=low_level_codes,
            low_to_high_codes=tuple(int(low_to_high.get(code, -1)) for code in low_level_codes),
            high_level_mask=tuple(high_mask),
            privileged_state=privileged_state,
            belief_target=belief_target,
            opponent_tag_id=self.opponent_tag_id(opponent_tag),
            opponent_entity_id=self.opponent_entity_id(entity_key),
            opponent_deck_id=self.opponent_deck_id(deck_name),
        )

    def encode_belief_target(self, state: StateSnapshot, *, acting_player: int) -> BeliefTarget:
        if not self.config.include_belief_targets:
            return BeliefTarget(
                hand_histogram=tuple(0.0 for _ in range(self.belief_histogram_dim)),
                hand_size=0.0,
                remaining_deck_histogram=tuple(0.0 for _ in range(self.belief_remaining_deck_histogram_dim)),
                burst_ready_count=0.0,
                group_hand_presence=tuple(0.0 for _ in range(self.belief_group_dim)),
                group_remaining_deck_presence=tuple(0.0 for _ in range(self.belief_group_dim)),
            )
        opponent = state.players[1 - acting_player]
        hand_histogram = [0.0] * self.belief_histogram_dim
        remaining_deck_histogram = [0.0] * self.belief_histogram_dim
        for card in opponent.hand_cards:
            index = self._card_index.get(int(card.definition_id))
            if index is not None:
                hand_histogram[index] += 1.0 / 10.0
        for card in opponent.pile_cards:
            index = self._card_index.get(int(card.definition_id))
            if index is not None:
                remaining_deck_histogram[index] += 1.0 / 30.0
        burst_ready = sum(
            1
            for character in opponent.characters
            if not character.defeated and character.energy >= character.max_energy
        )
        return BeliefTarget(
            hand_histogram=tuple(hand_histogram),
            hand_size=self._safe_ratio(len(opponent.hand_cards), 10.0),
            remaining_deck_histogram=tuple(remaining_deck_histogram),
            burst_ready_count=self._safe_ratio(burst_ready, 3.0),
            group_hand_presence=encode_group_presence(opponent.hand_cards),
            group_remaining_deck_presence=encode_group_presence(opponent.pile_cards),
        )

    def _encode_global_token(self, context: DecisionContext, *, opponent_tag: str) -> tuple[float, ...]:
        visible_state = context.player_view
        assert visible_state is not None
        return self._build_token(
            token_type="global",
            owner="neutral",
            phase=visible_state.phase,
            request_type=context.request_type,
            round_number=visible_state.round_number,
            current_turn=visible_state.current_turn,
            acting_player=context.acting_player,
            opponent_tag=opponent_tag,
            scalars=(
                self._safe_ratio(len(context.legal_low_level_codes), 64.0),
                self._safe_ratio(len(context.legal_high_level_codes), 32.0),
                1.0 if context.terminal else 0.0,
                1.0 if visible_state.winner is None else 0.0,
                1.0 if visible_state.winner == context.acting_player else 0.0,
                1.0 if visible_state.winner == (1 - context.acting_player) else 0.0,
            ),
        )

    def _encode_entity_token(
        self,
        *,
        token_type: str,
        owner: str,
        definition_id: int,
        variables: dict[str, int],
        round_number: int,
        request_type: DecisionType,
    ) -> tuple[float, ...]:
        values = sorted((int(value) for value in variables.values()), key=abs, reverse=True)
        total = sum(values)
        nonzero = sum(1 for value in values if value != 0)
        top1 = values[0] if len(values) > 0 else 0
        top2 = values[1] if len(values) > 1 else 0
        top3 = values[2] if len(values) > 2 else 0
        return self._build_token(
            token_type=token_type,
            owner=owner,
            definition_id=definition_id,
            request_type=request_type,
            round_number=round_number,
            scalars=(
                self._safe_ratio(len(values), 8.0),
                self._safe_ratio(nonzero, 8.0),
                self._safe_ratio(total, 20.0),
                self._signed_log_ratio(total, 20.0),
                self._safe_ratio(top1, 20.0),
                self._signed_log_ratio(top1, 20.0),
                self._safe_ratio(top2, 20.0),
                self._signed_log_ratio(top2, 20.0),
                self._safe_ratio(top3, 20.0),
                self._signed_log_ratio(top3, 20.0),
            ),
        )

    def _build_token(
        self,
        *,
        token_type: str,
        owner: str,
        definition_id: int | None = None,
        phase: str | None = None,
        request_type: DecisionType | None = None,
        option_kind: OptionKind | None = None,
        opponent_tag: str | None = None,
        round_number: int = 0,
        current_turn: int | None = None,
        acting_player: int | None = None,
        scalars: Sequence[float] = (),
    ) -> tuple[float, ...]:
        features: list[float] = []
        features.extend(1.0 if token_type == value else 0.0 for value in _TOKEN_TYPES)
        features.extend(1.0 if owner == value else 0.0 for value in _OWNER_TYPES)
        features.extend(1.0 if phase == value else 0.0 for value in _PHASES)
        features.extend(1.0 if request_type == value else 0.0 for value in _REQUEST_TYPES)
        features.extend(1.0 if option_kind == value else 0.0 for value in _OPTION_KINDS)
        if self.config.include_opponent_tag:
            features.extend(1.0 if opponent_tag == value else 0.0 for value in _OPPONENT_TAGS)
        scalar_values = [
            self._safe_ratio(round_number, 15.0),
            self._safe_ratio(definition_id or 0, 400000.0),
            1.0 if current_turn == 0 else 0.0,
            1.0 if current_turn == 1 else 0.0,
            1.0 if acting_player == 0 else 0.0,
            1.0 if acting_player == 1 else 0.0,
            *[float(value) for value in scalars[: (_SCALAR_DIM - 6)]],
        ]
        if len(scalar_values) < _SCALAR_DIM:
            scalar_values.extend(0.0 for _ in range(_SCALAR_DIM - len(scalar_values)))
        features.extend(scalar_values[:_SCALAR_DIM])
        return tuple(features)

    def opponent_tag_id(self, opponent_tag: str) -> int:
        return opponent_tag_to_id(opponent_tag)

    def opponent_entity_id(self, opponent_entity_key: str) -> int:
        return stable_hash_id(
            opponent_entity_key,
            vocab_size=self.opponent_entity_vocab_size,
            unknown_id=0,
        )

    def opponent_deck_id(self, opponent_deck_name: str) -> int:
        return stable_hash_id(
            opponent_deck_name,
            vocab_size=self.deck_vocab_size,
            unknown_id=0,
        )

    def _owner_label(self, acting_player: int, entity_owner: int) -> str:
        return "self" if acting_player == entity_owner else "opponent"

    def _selected_option(self, step: TrajectoryStep):
        return selected_low_level_spec(step)

    def _selected_low_spec(self, step: TrajectoryStep):
        if step.choice.action_code >= 0:
            try:
                return low_level_spec_for_code(int(step.choice.action_code))
            except KeyError:
                pass
        return None

    def _safe_ratio(self, value: float, denom: float) -> float:
        if denom <= 0:
            return 0.0
        return max(-1.0, min(1.0, float(value) / float(denom)))

    def _signed_log_ratio(self, value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        magnitude = math.log1p(abs(float(value)))
        denom = math.log1p(float(scale))
        if denom <= 0:
            return 0.0
        signed = math.copysign(magnitude / denom, float(value))
        return max(-1.0, min(1.0, signed))
