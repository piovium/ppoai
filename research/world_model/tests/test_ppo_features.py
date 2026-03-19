from __future__ import annotations

import unittest

from gitcg_world_model.schema import CharacterSnapshot, DecisionContext, DecisionType, EntitySnapshot, PlayerSnapshot, StateSnapshot
from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.event_features import EventFeatureEncoder


class TokenObservationEncoderTests(unittest.TestCase):
    def test_encode_context_can_skip_privileged_targets_without_touching_full_state(self):
        encoder = TokenObservationEncoder(config=None)
        context = _context_with_full_state_trap()

        encoded = encoder.encode_context(context, include_training_targets=False)

        self.assertTrue(all(value == 0.0 for value in encoded.privileged_state))
        self.assertTrue(all(value == 0.0 for value in encoded.belief_target.hand_histogram))
        self.assertTrue(all(value == 0.0 for value in encoded.belief_target.remaining_deck_histogram))
        self.assertEqual(encoded.belief_target.hand_size, 0.0)

    def test_opponent_public_dice_count_uses_count_but_not_color(self):
        encoder = TokenObservationEncoder(config=None)
        context_a = _context_with_opponent_dice((1, 2, 3, 4, 8))
        context_b = _context_with_opponent_dice((6, 6, 6, 6, 6))

        encoded_a = encoder.encode_context(
            context_a,
            include_training_targets=False,
            opponent_public_dice_count=5.0,
        )
        encoded_b = encoder.encode_context(
            context_b,
            include_training_targets=False,
            opponent_public_dice_count=5.0,
        )

        self.assertEqual(encoded_a.token_features, encoded_b.token_features)

    def test_entity_token_distinguishes_small_and_large_variable_values(self):
        base = EventFeatureEncoder()
        encoder = TokenObservationEncoder(
            config=None,
        )
        small = encoder._encode_entity_token(
            token_type="status",
            owner="self",
            definition_id=base.character_vocabulary[0],
            variables={"shield": 1},
            round_number=1,
            request_type=None,
        )
        large = encoder._encode_entity_token(
            token_type="status",
            owner="self",
            definition_id=base.character_vocabulary[0],
            variables={"shield": 10},
            round_number=1,
            request_type=None,
        )

        self.assertNotEqual(small, large)

    def test_character_token_distinguishes_aura(self):
        encoder = TokenObservationEncoder(config=None)
        context_no_aura = _context_with_aura(0)
        context_pyro = _context_with_aura(3)

        encoded_no_aura = encoder.encode_context(context_no_aura)
        encoded_pyro = encoder.encode_context(context_pyro)

        self.assertNotEqual(encoded_no_aura.token_features, encoded_pyro.token_features)

    def test_character_entities_are_encoded_as_separate_tokens(self):
        encoder = TokenObservationEncoder(config=None)
        context_plain = _context_with_character_entity(None)
        context_with_equipment = _context_with_character_entity(
            EntitySnapshot(id=101, definition_id=312001, variables={"usage": 2})
        )

        encoded_plain = encoder.encode_context(context_plain)
        encoded_with_equipment = encoder.encode_context(context_with_equipment)

        self.assertGreater(len(encoded_with_equipment.token_features), len(encoded_plain.token_features))
        self.assertNotEqual(encoded_plain.token_features, encoded_with_equipment.token_features)


def _context_with_aura(aura: int) -> DecisionContext:
    state = StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            PlayerSnapshot(
                active_character_id=1,
                characters=(
                    CharacterSnapshot(
                        id=1,
                        definition_id=1411,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=aura,
                        entities=(),
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=(1, 2, 8),
                declared_end=False,
                legend_used=False,
            ),
            PlayerSnapshot(
                active_character_id=11,
                characters=(
                    CharacterSnapshot(
                        id=11,
                        definition_id=1510,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=0,
                        entities=(),
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=(3, 3, 8),
                declared_end=False,
                legend_used=False,
            ),
        ),
    )
    return DecisionContext(
        acting_player=0,
        request_type=DecisionType.ACTION,
        step_index=0,
        full_state=state,
        player_view=state,
    )


def _context_with_character_entity(entity: EntitySnapshot | None) -> DecisionContext:
    entities = (() if entity is None else (entity,))
    state = StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            PlayerSnapshot(
                active_character_id=1,
                characters=(
                    CharacterSnapshot(
                        id=1,
                        definition_id=1411,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=0,
                        entities=entities,
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=(1, 2, 8),
                declared_end=False,
                legend_used=False,
            ),
            PlayerSnapshot(
                active_character_id=11,
                characters=(
                    CharacterSnapshot(
                        id=11,
                        definition_id=1510,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=0,
                        entities=(),
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=(3, 3, 8),
                declared_end=False,
                legend_used=False,
            ),
        ),
    )
    return DecisionContext(
        acting_player=0,
        request_type=DecisionType.ACTION,
        step_index=0,
        full_state=state,
        player_view=state,
    )


def _context_with_full_state_trap() -> DecisionContext:
    state = _context_with_aura(0).player_view
    assert state is not None
    return DecisionContext(
        acting_player=0,
        request_type=DecisionType.ACTION,
        step_index=0,
        full_state=_FullStateTrap(),
        player_view=state,
    )


def _context_with_opponent_dice(opponent_dice: tuple[int, ...]) -> DecisionContext:
    state = StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            PlayerSnapshot(
                active_character_id=1,
                characters=(
                    CharacterSnapshot(
                        id=1,
                        definition_id=1411,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=0,
                        entities=(),
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=(1, 2, 8),
                declared_end=False,
                legend_used=False,
            ),
            PlayerSnapshot(
                active_character_id=11,
                characters=(
                    CharacterSnapshot(
                        id=11,
                        definition_id=1510,
                        health=10,
                        max_health=10,
                        energy=0,
                        max_energy=2,
                        defeated=False,
                        is_active=True,
                        aura=0,
                        entities=(),
                    ),
                ),
                combat_statuses=(),
                summons=(),
                supports=(),
                hand_cards=(),
                pile_cards=(),
                dice=opponent_dice,
                declared_end=False,
                legend_used=False,
            ),
        ),
    )
    return DecisionContext(
        acting_player=0,
        request_type=DecisionType.ACTION,
        step_index=0,
        full_state=state,
        player_view=state,
    )


class _FullStateTrap:
    def __getattr__(self, name):
        raise AssertionError(f"full_state should not be accessed in live encoding path: {name}")


if __name__ == "__main__":
    unittest.main()
