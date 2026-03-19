from __future__ import annotations

import unittest

from gitcg_world_model.action_hierarchy import (
    build_default_hierarchical_action_codebook,
    classify_action_outcome_key,
    high_level_code_for_spec,
    load_action_taxonomy,
)
from gitcg_world_model.schema import (
    CharacterSnapshot,
    DecisionType,
    EntitySnapshot,
    LowLevelActionSpec,
    OptionKind,
    PlayerSnapshot,
    PublicSlotTarget,
    StateSnapshot,
)


class ActionHierarchyOutcomeTests(unittest.TestCase):
    def test_action_taxonomy_uses_result_action_categories(self):
        taxonomy = load_action_taxonomy()
        self.assertEqual(
            taxonomy.request_type_to_categories["ACTION"],
            (
                "chip_frontline_hp",
                "chip_backline_hp",
                "force_frontline_lethal",
                "fortify_frontline",
                "fortify_team",
                "switch_character",
                "cycle_for_cards",
                "yield_initiative",
                "burst_setup",
                "invest_tech",
                "invest_delayed_damage",
                "invest_engine_card",
                "elemental_tuning",
                "other_action",
            ),
        )
        self.assertEqual(
            taxonomy.request_type_to_categories["CHOOSE_ACTIVE"],
            (
                "choose_attack",
                "choose_defense",
                "choose_heal",
                "choose_absorb",
                "choose_support",
                "choose_other",
            ),
        )
        self.assertEqual(
            taxonomy.request_type_to_categories["REROLL_DICE"],
            (
                "keep_all",
                "keep_active_and_two_backline",
                "keep_active_and_one_backline",
                "keep_active_only",
                "keep_two_backline",
                "keep_one_backline",
                "reroll_active_and_two_backline",
                "reroll_active_and_one_backline",
                "reroll_active_only",
                "reroll_two_backline",
                "reroll_one_backline",
            ),
        )
        self.assertEqual(
            taxonomy.request_type_to_categories["SELECT_CARD"],
            (
                "select_resource",
                "select_draw_search",
                "select_defense_heal",
                "select_buff_equip",
                "select_board_setup",
                "select_engine_card",
                "select_damage_pressure",
                "select_core",
                "select_other",
            ),
        )
        self.assertEqual(
            taxonomy.request_type_to_categories["SWITCH_HANDS"],
            (
                "switch_keep_all",
                "switch_resource",
                "switch_draw_search",
                "switch_defense_heal",
                "switch_buff_equip",
                "switch_board_setup",
                "switch_engine_card",
                "switch_damage_pressure",
                "switch_core",
                "switch_mixed",
                "switch_other",
            ),
        )

    def test_force_frontline_lethal_has_priority_over_frontline_chip(self):
        spec = _spec(
            kind=OptionKind.ACTION_USE_SKILL,
            subject_definition_id=1101,
            target_slots=(PublicSlotTarget(owner="opponent", zone="character", index=0),),
        )
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(opponent_healths=(3, 10, 10)),
            post_state=_state(opponent_healths=(0, 10, 10), opponent_defeated=(True, False, False)),
            acting_player=0,
        )
        self.assertEqual(key, "force_frontline_lethal")

    def test_chip_backline_hp(self):
        spec = _spec(
            kind=OptionKind.ACTION_USE_SKILL,
            subject_definition_id=1102,
            target_slots=(PublicSlotTarget(owner="opponent", zone="character", index=1),),
        )
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(opponent_healths=(10, 10, 10)),
            post_state=_state(opponent_healths=(10, 7, 10)),
            acting_player=0,
        )
        self.assertEqual(key, "chip_backline_hp")

    def test_fortify_frontline(self):
        spec = _spec(
            kind=OptionKind.ACTION_PLAY_CARD,
            subject_definition_id=333003,
            target_slots=(PublicSlotTarget(owner="self", zone="character", index=0),),
        )
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(self_healths=(5, 10, 10)),
            post_state=_state(self_healths=(7, 10, 10)),
            acting_player=0,
        )
        self.assertEqual(key, "fortify_frontline")

    def test_fortify_team(self):
        spec = _spec(
            kind=OptionKind.ACTION_PLAY_CARD,
            subject_definition_id=333004,
            target_slots=(PublicSlotTarget(owner="self", zone="character", index=0),),
        )
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(self_healths=(4, 4, 10)),
            post_state=_state(self_healths=(6, 6, 10)),
            acting_player=0,
        )
        self.assertEqual(key, "fortify_team")

    def test_switch_character_and_choose_active_are_separate(self):
        spec = _spec(kind=OptionKind.ACTION_SWITCH_ACTIVE, subject_definition_id=1411)
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(),
            post_state=_state(self_active_index=1),
            acting_player=0,
        )
        self.assertEqual(key, "switch_character")

    def test_choose_active_uses_role_groups(self):
        codebook = build_default_hierarchical_action_codebook()
        visible_state = _state(self_definition_ids=(1411, 1510, 2103))
        spec = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.CHOOSE_ACTIVE,
            kind=OptionKind.CHOOSE_ACTIVE,
            label="choose_active",
            choose_active_slot=0,
        )
        high_code = high_level_code_for_spec(codebook, spec, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(high_code).key, "choose_heal")

        absorb_spec = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.CHOOSE_ACTIVE,
            kind=OptionKind.CHOOSE_ACTIVE,
            label="choose_active",
            choose_active_slot=2,
        )
        absorb_code = high_level_code_for_spec(codebook, absorb_spec, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(absorb_code).key, "choose_absorb")

    def test_reroll_uses_team_color_semantics(self):
        codebook = build_default_hierarchical_action_codebook()
        visible_state = _state(
            self_definition_ids=(1411, 1510, 2103),
            self_dice=(4, 4, 5, 1, 8),
        )
        keep_active_only = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.REROLL_DICE,
            kind=OptionKind.REROLL_DICE,
            label="reroll",
            reroll_dice_mask=(1 << 2) | (1 << 3) | (1 << 4),
        )
        keep_code = high_level_code_for_spec(codebook, keep_active_only, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(keep_code).key, "keep_active_only")

        reroll_active_only = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.REROLL_DICE,
            kind=OptionKind.REROLL_DICE,
            label="reroll",
            reroll_dice_mask=(1 << 0) | (1 << 1),
        )
        reroll_code = high_level_code_for_spec(codebook, reroll_active_only, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(reroll_code).key, "reroll_active_only")

    def test_select_card_uses_card_function_group(self):
        codebook = build_default_hierarchical_action_codebook()
        visible_state = _state()
        spec = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.SELECT_CARD,
            kind=OptionKind.SELECT_CARD,
            label="select_card",
            select_card_definition_id=332004,
            subject_definition_id=332004,
        )
        high_code = high_level_code_for_spec(codebook, spec, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(high_code).key, "select_draw_search")

    def test_switch_hands_uses_removed_card_function_group(self):
        codebook = build_default_hierarchical_action_codebook()
        visible_state = _state(
            self_hand_definition_ids=(332004, 332003, 311503, 331601),
        )
        spec = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.SWITCH_HANDS,
            kind=OptionKind.SWITCH_HANDS,
            label="switch_hands",
            switch_hand_slot_mask=(1 << 0) | (1 << 1),
        )
        high_code = high_level_code_for_spec(codebook, spec, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(high_code).key, "switch_draw_search")

        mixed_spec = LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.SWITCH_HANDS,
            kind=OptionKind.SWITCH_HANDS,
            label="switch_hands",
            switch_hand_slot_mask=(1 << 0) | (1 << 2),
        )
        mixed_code = high_level_code_for_spec(codebook, mixed_spec, visible_state=visible_state, acting_player=0)
        self.assertEqual(codebook.high_spec(mixed_code).key, "switch_mixed")

    def test_yield_initiative_and_cycle_for_cards_are_separate(self):
        end_spec = _spec(kind=OptionKind.ACTION_DECLARE_END)
        end_key = classify_action_outcome_key(
            spec=end_spec,
            pre_state=_state(),
            post_state=_state(),
            acting_player=0,
        )
        self.assertEqual(end_key, "yield_initiative")

        cycle_spec = _spec(kind=OptionKind.ACTION_PLAY_CARD, subject_definition_id=332004)
        cycle_key = classify_action_outcome_key(
            spec=cycle_spec,
            pre_state=_state(self_hand_count=2),
            post_state=_state(self_hand_count=3),
            acting_player=0,
        )
        self.assertEqual(cycle_key, "cycle_for_cards")

    def test_burst_setup(self):
        spec = _spec(kind=OptionKind.ACTION_USE_SKILL, subject_definition_id=2201)
        key = classify_action_outcome_key(
            spec=spec,
            pre_state=_state(self_energies=(0, 0, 0)),
            post_state=_state(self_energies=(1, 0, 0)),
            acting_player=0,
        )
        self.assertEqual(key, "burst_setup")

    def test_investment_categories(self):
        delayed = classify_action_outcome_key(
            spec=_spec(kind=OptionKind.ACTION_PLAY_CARD, subject_definition_id=323004),
            pre_state=_state(self_summon_count=0),
            post_state=_state(self_summon_count=1),
            acting_player=0,
        )
        self.assertEqual(delayed, "invest_delayed_damage")

        engine = classify_action_outcome_key(
            spec=_spec(kind=OptionKind.ACTION_PLAY_CARD, subject_definition_id=321011),
            pre_state=_state(),
            post_state=_state(),
            acting_player=0,
        )
        self.assertEqual(engine, "invest_engine_card")

        tech = classify_action_outcome_key(
            spec=_spec(kind=OptionKind.ACTION_PLAY_CARD, subject_definition_id=331601),
            pre_state=_state(),
            post_state=_state(),
            acting_player=0,
        )
        self.assertEqual(tech, "invest_tech")

    def test_elemental_tuning_and_other_action(self):
        tuning = classify_action_outcome_key(
            spec=_spec(kind=OptionKind.ACTION_ELEMENTAL_TUNING, subject_definition_id=312004),
            pre_state=_state(),
            post_state=_state(),
            acting_player=0,
        )
        self.assertEqual(tuning, "elemental_tuning")

        other = classify_action_outcome_key(
            spec=_spec(kind=OptionKind.ACTION_PLAY_CARD, subject_definition_id=399999),
            pre_state=_state(),
            post_state=_state(),
            acting_player=0,
        )
        self.assertEqual(other, "other_action")


def _spec(
    *,
    kind: OptionKind,
    subject_definition_id: int = 0,
    target_slots: tuple[PublicSlotTarget, ...] = (),
) -> LowLevelActionSpec:
    return LowLevelActionSpec(
        action_code=-1,
        request_type=DecisionType.ACTION,
        kind=kind,
        label=kind.value,
        subject_definition_id=subject_definition_id,
        target_slots=target_slots,
    )


def _state(
    *,
    self_active_index: int = 0,
    opponent_active_index: int = 0,
    self_definition_ids: tuple[int, int, int] = (1400, 1401, 1402),
    opponent_definition_ids: tuple[int, int, int] = (1500, 1501, 1502),
    self_healths: tuple[int, int, int] = (10, 10, 10),
    opponent_healths: tuple[int, int, int] = (10, 10, 10),
    self_energies: tuple[int, int, int] = (0, 0, 0),
    opponent_energies: tuple[int, int, int] = (0, 0, 0),
    self_defeated: tuple[bool, bool, bool] = (False, False, False),
    opponent_defeated: tuple[bool, bool, bool] = (False, False, False),
    self_hand_count: int = 2,
    self_hand_definition_ids: tuple[int, ...] | None = None,
    self_summon_count: int = 0,
    self_dice: tuple[int, ...] = (1, 2, 3),
) -> StateSnapshot:
    return StateSnapshot(
        phase="action",
        round_number=4,
        current_turn=0,
        winner=None,
        players=(
            _player(
                active_index=self_active_index,
                definition_ids=self_definition_ids,
                healths=self_healths,
                energies=self_energies,
                defeated=self_defeated,
                hand_count=self_hand_count,
                hand_definition_ids=self_hand_definition_ids,
                summon_count=self_summon_count,
                dice=self_dice,
            ),
            _player(
                active_index=opponent_active_index,
                definition_ids=opponent_definition_ids,
                healths=opponent_healths,
                energies=opponent_energies,
                defeated=opponent_defeated,
                hand_count=0,
                hand_definition_ids=None,
                summon_count=0,
                dice=(1, 2, 3),
            ),
        ),
    )


def _player(
    *,
    active_index: int,
    definition_ids: tuple[int, int, int],
    healths: tuple[int, int, int],
    energies: tuple[int, int, int],
    defeated: tuple[bool, bool, bool],
    hand_count: int,
    hand_definition_ids: tuple[int, ...] | None,
    summon_count: int,
    dice: tuple[int, ...],
) -> PlayerSnapshot:
    characters = tuple(
        CharacterSnapshot(
            id=index + 1 + definition_ids[index],
            definition_id=definition_ids[index],
            health=int(health),
            max_health=10,
            energy=int(energy),
            max_energy=2,
            defeated=bool(defeated_flag),
            is_active=index == active_index,
            entities=(),
        )
        for index, (health, energy, defeated_flag) in enumerate(zip(healths, energies, defeated, strict=True))
    )
    if hand_definition_ids is None:
        hand_definition_ids = tuple(300000 + index for index in range(hand_count))
    hand_cards = tuple(
        EntitySnapshot(id=2000 + index, definition_id=int(definition_id), variables={})
        for index, definition_id in enumerate(hand_definition_ids)
    )
    summons = tuple(
        EntitySnapshot(id=4000 + index, definition_id=500000 + index, variables={"usages": 1})
        for index in range(summon_count)
    )
    return PlayerSnapshot(
        active_character_id=characters[active_index].id,
        characters=characters,
        combat_statuses=(),
        summons=summons,
        supports=(),
        hand_cards=hand_cards,
        pile_cards=(),
        dice=tuple(int(value) for value in dice),
        declared_end=False,
        legend_used=False,
    )


if __name__ == "__main__":
    unittest.main()
