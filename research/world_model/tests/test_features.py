from __future__ import annotations

import unittest

from gitcg_world_model.features import FeatureEncoder
from gitcg_world_model.schema import (
    CharacterSnapshot,
    DecisionType,
    LowLevelActionSpec,
    OptionKind,
    PlayerSnapshot,
    PublicSlotTarget,
    StateSnapshot,
)


class FeatureEncoderTests(unittest.TestCase):
    def test_active_character_encoding_uses_definition_id(self):
        encoder = FeatureEncoder()
        player = _player_snapshot(active_id=11, active_definition=1609, reserve_definition=1411)

        encoded = encoder._encode_player(player)
        active_slice = encoded[: len(encoder.character_vocabulary)]

        self.assertEqual(sum(active_slice), 1.0)
        self.assertEqual(active_slice[encoder.character_vocabulary.index(1609)], 1.0)

    def test_encode_low_level_spec_distinguishes_target_owner_and_matches_option_dim(self):
        encoder = FeatureEncoder()
        state = _state_snapshot()
        self_target = LowLevelActionSpec(
            action_code=0,
            request_type=DecisionType.ACTION,
            kind=OptionKind.ACTION_USE_SKILL,
            label="self",
            used_dice=(3, 3),
            subject_definition_id=1101,
            target_slots=(PublicSlotTarget(owner="self", zone="character", index=0),),
        )
        enemy_target = LowLevelActionSpec(
            action_code=1,
            request_type=DecisionType.ACTION,
            kind=OptionKind.ACTION_USE_SKILL,
            label="enemy",
            used_dice=(3, 3),
            subject_definition_id=1101,
            target_slots=(PublicSlotTarget(owner="opponent", zone="character", index=0),),
        )

        self_features = encoder.encode_low_level_spec(self_target, state=state, acting_player=0)
        enemy_features = encoder.encode_low_level_spec(enemy_target, state=state, acting_player=0)
        trailing_dim = 3 + 9  # removed-hand block + reroll block
        target_block_start = len(self_features) - trailing_dim - 16
        self_slot = self_features[target_block_start : target_block_start + 6]
        enemy_slot = enemy_features[target_block_start : target_block_start + 6]

        self.assertEqual(len(self_features), encoder.option_dim)
        self.assertNotEqual(self_features, enemy_features)
        self.assertEqual(self_slot[:4], [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(enemy_slot[:4], [0.0, 1.0, 1.0, 1.0])

    def test_encode_low_level_spec_includes_removed_hand_definition_ids(self):
        encoder = FeatureEncoder()
        option_a = LowLevelActionSpec(
            action_code=0,
            request_type=DecisionType.SWITCH_HANDS,
            kind=OptionKind.SWITCH_HANDS,
            label="remove_a",
            subject_definition_id=312004,
            switch_hand_slot_mask=0b01,
        )
        option_b = LowLevelActionSpec(
            action_code=1,
            request_type=DecisionType.SWITCH_HANDS,
            kind=OptionKind.SWITCH_HANDS,
            label="remove_b",
            subject_definition_id=332025,
            switch_hand_slot_mask=0b10,
        )

        features_a = encoder.encode_low_level_spec(option_a)
        features_b = encoder.encode_low_level_spec(option_b)

        self.assertEqual(len(features_a), encoder.option_dim)
        self.assertNotEqual(features_a, features_b)

    def test_encode_player_distinguishes_character_aura(self):
        encoder = FeatureEncoder()
        player_a = _player_snapshot(active_id=1, active_definition=1411, reserve_definition=1510, active_aura=0)
        player_b = _player_snapshot(active_id=1, active_definition=1411, reserve_definition=1510, active_aura=3)

        features_a = encoder._encode_player(player_a)
        features_b = encoder._encode_player(player_b)

        self.assertNotEqual(features_a, features_b)

    def test_encode_low_level_spec_distinguishes_reroll_subset(self):
        encoder = FeatureEncoder()
        reroll_a = LowLevelActionSpec(
            action_code=0,
            request_type=DecisionType.REROLL_DICE,
            kind=OptionKind.REROLL_DICE,
            label="reroll_a",
            reroll_dice_mask=0b101,
        )
        reroll_b = LowLevelActionSpec(
            action_code=1,
            request_type=DecisionType.REROLL_DICE,
            kind=OptionKind.REROLL_DICE,
            label="reroll_b",
            reroll_dice_mask=0b111,
        )

        features_a = encoder.encode_low_level_spec(reroll_a)
        features_b = encoder.encode_low_level_spec(reroll_b)

        self.assertEqual(len(features_a), encoder.option_dim)
        self.assertNotEqual(features_a, features_b)


def _state_snapshot() -> StateSnapshot:
    return StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            _player_snapshot(active_id=1, active_definition=1411, reserve_definition=1510),
            _player_snapshot(active_id=11, active_definition=1609, reserve_definition=2203),
        ),
    )


def _player_snapshot(
    *,
    active_id: int,
    active_definition: int,
    reserve_definition: int,
    active_aura: int = 0,
) -> PlayerSnapshot:
    return PlayerSnapshot(
        active_character_id=active_id,
        characters=(
            CharacterSnapshot(
                id=active_id,
                definition_id=active_definition,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=True,
                aura=active_aura,
                entities=(),
            ),
            CharacterSnapshot(
                id=active_id + 1,
                definition_id=reserve_definition,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=False,
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
    )


if __name__ == "__main__":
    unittest.main()
