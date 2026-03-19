from __future__ import annotations

from .decks import initial_card_vocabulary, initial_character_vocabulary
from .schema import (
    DecisionContext,
    DecisionType,
    EntitySnapshot,
    LowLevelActionSpec,
    OptionKind,
    PlayerSnapshot,
    PublicSlotTarget,
    StateSnapshot,
)

_PHASES = ("init_hands", "init_actives", "roll", "action", "end", "game_end")
_DECISION_TYPES = tuple(DecisionType)
_OPTION_KINDS = tuple(OptionKind)
_DICE_TYPES = (1, 2, 3, 4, 5, 6, 7, 8)
_AURA_TYPES = (0, 1, 2, 3, 4, 5, 6)
_MAX_CHARACTERS = 3
_TARGET_SLOTS = 2
_TARGET_SLOT_FEATURE_DIM = 6
_TARGET_AGGREGATE_FEATURE_DIM = 4
_COST_BUCKETS = ("exact", "aligned", "void", "energy", "legend")
_REMOVED_HAND_SLOTS = 2


class FeatureEncoder:
    def __init__(
        self,
        card_vocabulary: tuple[int, ...] | None = None,
        character_vocabulary: tuple[int, ...] | None = None,
    ):
        self.card_vocabulary = card_vocabulary or initial_card_vocabulary()
        self.character_vocabulary = character_vocabulary or initial_character_vocabulary()
        self._card_index = {value: index for index, value in enumerate(self.card_vocabulary)}
        self._character_index = {
            value: index for index, value in enumerate(self.character_vocabulary)
        }
        self._player_dim = self._compute_player_dim()
        self.state_dim = self._compute_state_dim()
        self.option_dim = self._compute_option_dim()

    def encode_context(self, context: DecisionContext) -> list[float]:
        features = self.encode_state(context.full_state)
        features.extend(self._one_hot(context.request_type.value, [value.value for value in _DECISION_TYPES]))
        features.extend([1.0 if context.acting_player == 0 else 0.0, 1.0 if context.acting_player == 1 else 0.0])
        if context.player_view is None:
            features.extend([0.0] * len(self.encode_state(context.full_state)))
        else:
            features.extend(self.encode_state(context.player_view))
        return features

    def encode_state(self, state: StateSnapshot) -> list[float]:
        features: list[float] = []
        features.extend(self._one_hot(state.phase, list(_PHASES)))
        features.append(min(float(state.round_number), 20.0) / 20.0)
        features.extend([1.0 if state.current_turn == 0 else 0.0, 1.0 if state.current_turn == 1 else 0.0])
        features.extend(
            [
                1.0 if state.winner is None else 0.0,
                1.0 if state.winner == 0 else 0.0,
                1.0 if state.winner == 1 else 0.0,
            ]
        )
        for player in state.players:
            features.extend(self._encode_player(player))
        return features

    def encode_low_level_spec(
        self,
        spec: LowLevelActionSpec,
        *,
        state: StateSnapshot | None = None,
        acting_player: int | None = None,
    ) -> list[float]:
        features: list[float] = []
        features.extend(self._one_hot(spec.kind.value, [value.value for value in _OPTION_KINDS]))
        features.extend(self._dice_histogram(spec.used_dice))
        features.extend(self._dice_histogram(spec.auto_selected_dice))
        features.extend(
            [
                min(len(spec.used_dice), 8) / 8.0,
                min(len(spec.auto_selected_dice), 8) / 8.0,
                1.0 if bool(spec.metadata.get("is_fast", False)) else 0.0,
                min(len(spec.target_slots), 4) / 4.0,
                min(float(spec.target_dice), 8.0) / 8.0,
                self._normalize_definition(spec.subject_definition_id),
                self._normalize_definition(spec.select_card_definition_id),
                self._normalize_definition(spec.discarded_card_definition_id),
                max(-1.0, min(float(spec.choose_active_slot), 2.0)) / 2.0,
                min(float(_bit_count(spec.switch_hand_slot_mask)), 10.0) / 10.0,
            ]
        )
        features.extend(self._encode_required_cost(spec.metadata.get("required_cost", ())))
        features.extend(
            self._encode_target_slots(
                target_slots=spec.target_slots,
                state=state,
                acting_player=acting_player,
            )
        )
        features.extend(
            self._encode_removed_hand_from_mask(
                switch_hand_slot_mask=spec.switch_hand_slot_mask,
                state=state,
                acting_player=acting_player,
            )
        )
        features.extend(
            self._encode_reroll_mask(
                reroll_dice_mask=spec.reroll_dice_mask,
                state=state,
                acting_player=acting_player,
            )
        )
        return features

    def _encode_player(self, player: PlayerSnapshot) -> list[float]:
        features: list[float] = []
        active_character = next(
            (character for character in player.characters if character.id == player.active_character_id),
            None,
        )
        features.extend(
            self._character_one_hot(
                active_character.definition_id if active_character is not None else None
            )
        )
        for index in range(_MAX_CHARACTERS):
            if index < len(player.characters):
                character = player.characters[index]
                features.extend(self._character_one_hot(character.definition_id))
                features.extend(
                    [
                        min(character.health, 10) / 10.0,
                        min(character.max_health, 10) / 10.0,
                        min(character.energy, 5) / 5.0,
                        min(character.max_energy, 5) / 5.0,
                        1.0 if character.defeated else 0.0,
                        1.0 if character.is_active else 0.0,
                        min(len(character.entities), 8) / 8.0,
                    ]
                )
                features.extend(self._one_hot(character.aura, list(_AURA_TYPES)))
            else:
                features.extend([0.0] * (len(self.character_vocabulary) + 7 + len(_AURA_TYPES)))
        features.extend(self._dice_histogram(player.dice))
        features.extend(self._card_histogram(card.definition_id for card in player.hand_cards))
        features.extend(self._card_histogram(card.definition_id for card in player.pile_cards))
        features.extend(
            [
                min(len(player.combat_statuses), 8) / 8.0,
                min(len(player.summons), 8) / 8.0,
                min(len(player.supports), 8) / 8.0,
                min(len(player.hand_cards), 30) / 30.0,
                min(len(player.pile_cards), 30) / 30.0,
                1.0 if player.declared_end else 0.0,
                1.0 if player.legend_used else 0.0,
            ]
        )
        return features

    def _compute_player_dim(self) -> int:
        return (
            len(self.character_vocabulary)
            + _MAX_CHARACTERS * (len(self.character_vocabulary) + 7 + len(_AURA_TYPES))
            + len(_DICE_TYPES)
            + len(self.card_vocabulary)
            + len(self.card_vocabulary)
            + 7
        )

    def _compute_state_dim(self) -> int:
        full_state_dim = len(_PHASES) + 1 + 2 + 3 + 2 * self._player_dim
        context_extra_dim = len(_DECISION_TYPES) + 2 + full_state_dim
        return full_state_dim + context_extra_dim

    def _compute_option_dim(self) -> int:
        return (
            len(_OPTION_KINDS)
            + len(_DICE_TYPES)
            + len(_DICE_TYPES)
            + 10
            + len(_COST_BUCKETS)
            + _TARGET_SLOTS * _TARGET_SLOT_FEATURE_DIM
            + _TARGET_AGGREGATE_FEATURE_DIM
            + 1
            + _REMOVED_HAND_SLOTS
            + 1
            + len(_DICE_TYPES)
        )

    def _one_hot(self, value: str | int | None, vocabulary: list[str | int]) -> list[float]:
        return [1.0 if entry == value else 0.0 for entry in vocabulary]

    def _character_one_hot(self, definition_id: int | None) -> list[float]:
        features = [0.0] * len(self.character_vocabulary)
        if definition_id in self._character_index:
            features[self._character_index[definition_id]] = 1.0
        return features

    def _card_histogram(self, definition_ids) -> list[float]:
        features = [0.0] * len(self.card_vocabulary)
        for definition_id in definition_ids:
            index = self._card_index.get(int(definition_id))
            if index is not None:
                features[index] += 1.0 / 30.0
        return features

    def _dice_histogram(self, dice) -> list[float]:
        features = [0.0] * len(_DICE_TYPES)
        for die in dice:
            die = int(die)
            if die in _DICE_TYPES:
                features[_DICE_TYPES.index(die)] += 1.0 / 8.0
        return features

    def _normalize_definition(self, definition_id: int | None) -> float:
        if definition_id is None or definition_id <= 0:
            return 0.0
        return min(float(definition_id), 400000.0) / 400000.0

    def _encode_required_cost(self, required_cost) -> list[float]:
        totals = {key: 0.0 for key in _COST_BUCKETS}
        for entry in required_cost:
            if isinstance(entry, dict):
                req_type = int(entry.get("type", 0))
                count = float(entry.get("count", 0))
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                req_type = int(entry[0])
                count = float(entry[1])
            else:
                continue
            if req_type in _DICE_TYPES[:-1]:
                totals["exact"] += count
            elif req_type == 8:
                totals["aligned"] += count
            elif req_type == 0:
                totals["void"] += count
            elif req_type == 9:
                totals["energy"] += count
            elif req_type == 10:
                totals["legend"] += count
        return [min(totals[key], 8.0) / 8.0 for key in _COST_BUCKETS]

    def _encode_targets(
        self,
        *,
        target_ids: tuple[int, ...],
        state: StateSnapshot | None,
        acting_player: int | None,
    ) -> list[float]:
        if state is None or acting_player is None:
            return [0.0] * (_TARGET_SLOTS * _TARGET_SLOT_FEATURE_DIM + _TARGET_AGGREGATE_FEATURE_DIM)
        target_lookup = _build_target_lookup(state)
        features: list[float] = []
        aggregate = {
            "self": 0.0,
            "opponent": 0.0,
            "character": 0.0,
            "active": 0.0,
        }
        for index in range(_TARGET_SLOTS):
            target = target_lookup.get(target_ids[index]) if index < len(target_ids) else None
            if target is None:
                features.extend([0.0] * _TARGET_SLOT_FEATURE_DIM)
                continue
            owner = int(target["owner"])
            is_self = 1.0 if owner == acting_player else 0.0
            is_opponent = 1.0 if owner != acting_player else 0.0
            is_character = 1.0 if target["kind"] == "character" else 0.0
            is_active = 1.0 if bool(target["is_active"]) else 0.0
            is_defeated = 1.0 if bool(target["defeated"]) else 0.0
            definition = self._normalize_definition(int(target["definition_id"]))
            features.extend(
                [
                    is_self,
                    is_opponent,
                    is_character,
                    is_active,
                    is_defeated,
                    definition,
                ]
            )
            aggregate["self"] += is_self
            aggregate["opponent"] += is_opponent
            aggregate["character"] += is_character
            aggregate["active"] += is_active
        features.extend(
            min(aggregate[key], 4.0) / 4.0
            for key in ("self", "opponent", "character", "active")
        )
        return features

    def _encode_target_slots(
        self,
        *,
        target_slots: tuple[PublicSlotTarget, ...],
        state: StateSnapshot | None,
        acting_player: int | None,
    ) -> list[float]:
        features: list[float] = []
        aggregate = {
            "self": 0.0,
            "opponent": 0.0,
            "character": 0.0,
            "active": 0.0,
        }
        for index in range(_TARGET_SLOTS):
            target = target_slots[index] if index < len(target_slots) else None
            if target is None:
                features.extend([0.0] * _TARGET_SLOT_FEATURE_DIM)
                continue
            is_self = 1.0 if target.owner == "self" else 0.0
            is_opponent = 1.0 if target.owner == "opponent" else 0.0
            is_character = 1.0 if target.zone == "character" else 0.0
            is_active = 0.0
            definition = 0.0
            if state is not None and acting_player is not None:
                resolved = self._resolve_target_slot_definition(
                    target=target,
                    state=state,
                    acting_player=acting_player,
                )
                definition = self._normalize_definition(resolved["definition_id"])
                is_active = 1.0 if resolved["is_active"] else 0.0
            features.extend(
                [
                    is_self,
                    is_opponent,
                    is_character,
                    is_active,
                    0.0,
                    definition,
                ]
            )
            aggregate["self"] += is_self
            aggregate["opponent"] += is_opponent
            aggregate["character"] += is_character
            aggregate["active"] += is_active
        features.extend(
            min(aggregate[key], 4.0) / 4.0
            for key in ("self", "opponent", "character", "active")
        )
        return features

    def _encode_removed_hand(
        self,
        *,
        removed_hand_definition_ids: tuple[int, ...],
    ) -> list[float]:
        features = [min(len(removed_hand_definition_ids), 10) / 10.0]
        normalized = [
            self._normalize_definition(definition_id)
            for definition_id in removed_hand_definition_ids[:_REMOVED_HAND_SLOTS]
        ]
        if len(normalized) < _REMOVED_HAND_SLOTS:
            normalized.extend([0.0] * (_REMOVED_HAND_SLOTS - len(normalized)))
        features.extend(normalized)
        return features

    def _encode_reroll_subset(
        self,
        *,
        dice_to_reroll: tuple[int, ...],
    ) -> list[float]:
        features = [min(len(dice_to_reroll), 8) / 8.0]
        features.extend(self._dice_histogram(dice_to_reroll))
        return features

    def _encode_removed_hand_from_mask(
        self,
        *,
        switch_hand_slot_mask: int,
        state: StateSnapshot | None,
        acting_player: int | None,
    ) -> list[float]:
        if state is None or acting_player is None:
            return [0.0] * (1 + _REMOVED_HAND_SLOTS)
        definitions: list[int] = []
        for index, card in enumerate(state.players[acting_player].hand_cards):
            if int(switch_hand_slot_mask) & (1 << index):
                definitions.append(int(card.definition_id))
        return self._encode_removed_hand(removed_hand_definition_ids=tuple(definitions))

    def _encode_reroll_mask(
        self,
        *,
        reroll_dice_mask: int,
        state: StateSnapshot | None,
        acting_player: int | None,
    ) -> list[float]:
        if state is None or acting_player is None:
            features = [min(float(_bit_count(reroll_dice_mask)), 8.0) / 8.0]
            features.extend(
                1.0 if int(reroll_dice_mask) & (1 << index) else 0.0 for index in range(len(_DICE_TYPES))
            )
            return features
        dice = tuple(
            int(die)
            for index, die in enumerate(state.players[acting_player].dice)
            if int(reroll_dice_mask) & (1 << index)
        )
        return self._encode_reroll_subset(dice_to_reroll=dice)

    def _resolve_target_slot_definition(
        self,
        *,
        target: PublicSlotTarget,
        state: StateSnapshot,
        acting_player: int,
    ) -> dict[str, int | bool]:
        owner_index = acting_player if target.owner == "self" else (1 - acting_player)
        player = state.players[owner_index]
        zone = str(target.zone)
        index = int(target.index)
        if zone == "character" and 0 <= index < len(player.characters):
            character = player.characters[index]
            return {
                "definition_id": int(character.definition_id),
                "is_active": bool(character.is_active),
            }
        zone_collection = {
            "character_entity": tuple(entity for character in player.characters for entity in character.entities),
            "combat_status": player.combat_statuses,
            "summon": player.summons,
            "support": player.supports,
            "hand_card": player.hand_cards,
            "pile_card": player.pile_cards,
        }.get(zone, ())
        if 0 <= index < len(zone_collection):
            entity = zone_collection[index]
            return {
                "definition_id": int(entity.definition_id),
                "is_active": False,
            }
        return {
            "definition_id": 0,
            "is_active": False,
        }


def _build_target_lookup(state: StateSnapshot) -> dict[int, dict[str, int | bool | str]]:
    lookup: dict[int, dict[str, int | bool | str]] = {}

    def register(
        entity: EntitySnapshot | None,
        *,
        owner: int,
        kind: str,
        definition_id: int,
        is_active: bool = False,
        defeated: bool = False,
        entity_id: int | None = None,
    ) -> None:
        resolved_id = entity_id if entity_id is not None else (entity.id if entity is not None else None)
        if resolved_id is None:
            return
        lookup[int(resolved_id)] = {
            "owner": owner,
            "kind": kind,
            "definition_id": int(definition_id),
            "is_active": bool(is_active),
            "defeated": bool(defeated),
        }

    for owner, player in enumerate(state.players):
        for character in player.characters:
            register(
                None,
                owner=owner,
                kind="character",
                definition_id=character.definition_id,
                is_active=character.id == player.active_character_id,
                defeated=character.defeated,
                entity_id=character.id,
            )
            for entity in character.entities:
                register(
                    entity,
                    owner=owner,
                    kind="character_entity",
                    definition_id=entity.definition_id,
                    is_active=character.id == player.active_character_id,
                )
        for entity in player.combat_statuses:
            register(entity, owner=owner, kind="combat_status", definition_id=entity.definition_id)
        for entity in player.summons:
            register(entity, owner=owner, kind="summon", definition_id=entity.definition_id)
        for entity in player.supports:
            register(entity, owner=owner, kind="support", definition_id=entity.definition_id)
        for entity in player.hand_cards:
            register(entity, owner=owner, kind="hand_card", definition_id=entity.definition_id)
        for entity in player.pile_cards:
            register(entity, owner=owner, kind="pile_card", definition_id=entity.definition_id)
    return lookup
def _bit_count(mask: int) -> int:
    return int(mask).bit_count()
