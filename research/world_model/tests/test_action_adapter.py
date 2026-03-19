from __future__ import annotations

import unittest
from dataclasses import dataclass

from gitcg_world_model.action_adapter import build_decision_context, encode_choice
from gitcg_world_model.action_hierarchy import legal_low_level_specs
from gitcg_world_model.schema import (
    CharacterSnapshot,
    DecisionType,
    EntitySnapshot,
    PlayerSnapshot,
    StateSnapshot,
)


@dataclass
class FakeRequirement:
    type: int
    count: int


class FakeAction:
    def __init__(
        self,
        *,
        field_name: str,
        field_value,
        required_cost=(),
        auto_selected_dice=(),
        validity: int = 0,
        is_fast: bool = False,
    ):
        self.required_cost = list(required_cost)
        self.auto_selected_dice = list(auto_selected_dice)
        self.validity = validity
        self.is_fast = is_fast
        setattr(self, field_name, field_value)
        self._field_name = field_name

    def HasField(self, field_name: str) -> bool:
        return field_name == self._field_name


class FakeChooseActiveRequest:
    candidate_ids = (1, 2)


class FakeSelectCardRequest:
    candidate_definition_ids = (312004, 332025)


class FakeActionRequest:
    def __init__(self, action):
        self.action = list(action)


class ActionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.full_state = _state_snapshot(
            player0_dice=(3, 3, 8),
            player1_dice=(1, 2, 3),
            player0_hand=(501, 502),
        )
        self.player_view = self.full_state

    def test_choose_active_context(self):
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.CHOOSE_ACTIVE,
            request=FakeChooseActiveRequest(),
            full_state=self.full_state,
            player_view=None,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        specs = legal_low_level_specs(context)
        self.assertEqual([spec.choose_active_slot for spec in specs], [0, 1])
        self.assertEqual(
            [spec.subject_definition_id for spec in specs],
            [1411, 1510],
        )
        self.assertEqual(
            context.request_payload["candidate_definition_ids"],
            [1411, 1510],
        )
        self.assertEqual(encode_choice(built, context.legal_low_level_codes[1]), {"active_character_id": 2})

    def test_select_card_context(self):
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.SELECT_CARD,
            request=FakeSelectCardRequest(),
            full_state=self.full_state,
            player_view=None,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        specs = legal_low_level_specs(context)
        self.assertEqual([spec.select_card_definition_id for spec in specs], [312004, 332025])
        self.assertEqual(encode_choice(built, context.legal_low_level_codes[0]), {"selected_definition_id": 312004})

    def test_reroll_context_enumerates_dice_subsets(self):
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.REROLL_DICE,
            request=object(),
            full_state=self.full_state,
            player_view=self.player_view,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        specs = legal_low_level_specs(context)
        reroll_masks = {int(spec.reroll_dice_mask) for spec in specs}
        self.assertIn(0, reroll_masks)
        self.assertIn(0b001, reroll_masks)
        self.assertIn(0b101, reroll_masks)
        self.assertIn(0b111, reroll_masks)

    def test_switch_hands_context_enumerates_card_subsets(self):
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.SWITCH_HANDS,
            request=object(),
            full_state=self.full_state,
            player_view=self.player_view,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        specs = legal_low_level_specs(context)
        semantic_choices = {
            tuple(
                int(card.definition_id)
                for index, card in enumerate(self.player_view.players[0].hand_cards)
                if spec.switch_hand_slot_mask & (1 << index)
            )
            for spec in specs
        }
        self.assertEqual(
            semantic_choices,
            {(), (312004,), (332025,), (312004, 332025)},
        )
        self.assertEqual(context.request_payload["hand_definition_ids"], [312004, 332025])

    def test_action_context_enumerates_real_dice_payments(self):
        use_skill = type(
            "UseSkill",
            (),
            {"skill_definition_id": 1101, "target_ids": ()},
        )()
        burst_skill = type(
            "UseSkill",
            (),
            {"skill_definition_id": 2201, "target_ids": ()},
        )()
        declare_end = type("DeclareEnd", (), {})()
        request = FakeActionRequest(
            [
                FakeAction(
                    field_name="use_skill",
                    field_value=use_skill,
                    required_cost=(FakeRequirement(3, 1), FakeRequirement(8, 1)),
                ),
                FakeAction(
                    field_name="declare_end",
                    field_value=declare_end,
                    required_cost=(),
                ),
                FakeAction(
                    field_name="use_skill",
                    field_value=burst_skill,
                    required_cost=(FakeRequirement(9, 2),),
                ),
            ]
        )
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.ACTION,
            request=request,
            full_state=self.full_state,
            player_view=self.player_view,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        specs = legal_low_level_specs(context)
        use_skill_specs = [spec for spec in specs if spec.subject_definition_id == 1101]
        used_dice = {tuple(spec.used_dice) for spec in use_skill_specs}
        self.assertEqual(used_dice, {(3, 3), (3, 8)})
        burst_specs = [spec for spec in specs if spec.subject_definition_id == 2201]
        self.assertEqual([tuple(spec.used_dice) for spec in burst_specs], [()])
        declare_end_payloads = [
            encode_choice(built, action_code)
            for spec, action_code in zip(specs, context.legal_low_level_codes, strict=False)
            if spec.kind.value == "action_declare_end"
        ]
        self.assertEqual(declare_end_payloads, [{"chosen_action_index": 1, "used_dice": []}])

    def test_elemental_tuning_option_includes_removed_card_definition(self):
        tuning = type(
            "ElementalTuning",
            (),
            {"removed_card_id": 501, "target_dice": 3},
        )()
        request = FakeActionRequest(
            [
                FakeAction(
                    field_name="elemental_tuning",
                    field_value=tuning,
                    required_cost=(),
                ),
            ]
        )
        built = build_decision_context(
            acting_player=0,
            request_type=DecisionType.ACTION,
            request=request,
            full_state=self.full_state,
            player_view=self.player_view,
            full_state_json=None,
            step_index=0,
        )
        context = built.context
        spec = legal_low_level_specs(context)[0]
        self.assertEqual(spec.discarded_card_definition_id, 312004)
        self.assertEqual(spec.subject_definition_id, 312004)


def _state_snapshot(
    *,
    player0_dice=(),
    player1_dice=(),
    player0_hand=(),
) -> StateSnapshot:
    def _player(dice, hand) -> PlayerSnapshot:
        characters = (
            CharacterSnapshot(
                id=1,
                definition_id=1411,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=True,
                entities=(),
            ),
            CharacterSnapshot(
                id=2,
                definition_id=1510,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=False,
                entities=(),
            ),
            CharacterSnapshot(
                id=3,
                definition_id=2103,
                health=10,
                max_health=10,
                energy=0,
                max_energy=3,
                defeated=False,
                is_active=False,
                entities=(),
            ),
        )
        return PlayerSnapshot(
            active_character_id=1,
            characters=characters,
            combat_statuses=(),
            summons=(),
            supports=(),
            hand_cards=tuple(
                EntitySnapshot(
                    id=value,
                    definition_id=312004 if index == 0 else 332025,
                    variables={},
                )
                for index, value in enumerate(hand)
            ),
            pile_cards=(),
            dice=tuple(dice),
            declared_end=False,
            legend_used=False,
        )

    return StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(_player(player0_dice, player0_hand), _player(player1_dice, ())),
    )


if __name__ == "__main__":
    unittest.main()
