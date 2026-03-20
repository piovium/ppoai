from __future__ import annotations

import unittest

from gitcg_world_model.action_hierarchy import high_level_spec_for_code
from gitcg_world_model.bootstrap_agent import ScriptedBootstrapAgent
from gitcg_world_model.schema import DecisionContext, DecisionType, LowLevelActionSpec, OptionKind
from gitcg_world_model.testsupport import materialize_legal_specs, synthetic_episode_record


def _spec(kind: OptionKind, label: str, *, is_fast: bool = False) -> LowLevelActionSpec:
    return LowLevelActionSpec(
        action_code=-1,
        request_type=DecisionType.ACTION,
        kind=kind,
        label=label,
        metadata={"is_fast": is_fast},
    )


class BootstrapAgentTests(unittest.TestCase):
    def test_skill_is_preferred_over_card_and_end(self):
        episode = synthetic_episode_record()
        visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
        encoded, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=visible_state,
            legal_specs=(
                _spec(OptionKind.ACTION_DECLARE_END, "end"),
                _spec(OptionKind.ACTION_PLAY_CARD, "card"),
                _spec(OptionKind.ACTION_USE_SKILL, "skill"),
            ),
        )
        context = DecisionContext(
            acting_player=0,
            request_type=DecisionType.ACTION,
            step_index=0,
            full_state=episode.steps[0].pre_state,
            legal_low_level_codes=encoded.legal_low_level_codes,
            legal_low_level_mask=encoded.legal_low_level_mask,
            legal_high_level_codes=encoded.legal_high_level_codes,
            high_to_low_map=encoded.high_to_low_map,
            player_view=episode.steps[0].player_view,
        )

        choice = ScriptedBootstrapAgent().choose_action(context)
        expected_code = next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_USE_SKILL)
        self.assertEqual(choice.action_code, expected_code)

    def test_declare_end_is_delayed_when_non_end_exists(self):
        episode = synthetic_episode_record()
        visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
        encoded, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=visible_state,
            legal_specs=(
                _spec(OptionKind.ACTION_DECLARE_END, "end"),
                _spec(OptionKind.ACTION_SWITCH_ACTIVE, "switch"),
            ),
        )
        context = DecisionContext(
            acting_player=0,
            request_type=DecisionType.ACTION,
            step_index=0,
            full_state=episode.steps[0].pre_state,
            legal_low_level_codes=encoded.legal_low_level_codes,
            legal_low_level_mask=encoded.legal_low_level_mask,
            legal_high_level_codes=encoded.legal_high_level_codes,
            high_to_low_map=encoded.high_to_low_map,
            player_view=episode.steps[0].player_view,
        )

        choice = ScriptedBootstrapAgent().choose_action(context)
        expected_code = next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_SWITCH_ACTIVE)
        self.assertEqual(choice.action_code, expected_code)

    def test_reroll_is_not_forced_to_keep_all(self):
        episode = synthetic_episode_record()
        visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
        legal_specs = (
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.REROLL_DICE,
                kind=OptionKind.REROLL_DICE,
                label="reroll:none",
                reroll_dice_mask=0,
            ),
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.REROLL_DICE,
                kind=OptionKind.REROLL_DICE,
                label="reroll:3",
                reroll_dice_mask=1,
            ),
        )
        encoded, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=visible_state,
            legal_specs=legal_specs,
        )
        context = DecisionContext(
            acting_player=0,
            request_type=DecisionType.REROLL_DICE,
            step_index=0,
            full_state=episode.steps[0].pre_state,
            legal_low_level_codes=encoded.legal_low_level_codes,
            legal_low_level_specs=materialized,
            legal_low_level_mask=encoded.legal_low_level_mask,
            legal_high_level_codes=encoded.legal_high_level_codes,
            high_to_low_map=encoded.high_to_low_map,
            player_view=episode.steps[0].player_view,
        )

        choice = ScriptedBootstrapAgent().choose_action(context)
        chosen_spec = next(spec for spec in materialized if spec.action_code == choice.action_code)
        chosen_high_key = None
        for high_code, low_codes in context.high_to_low_map:
            if choice.action_code in low_codes:
                chosen_high_key = high_level_spec_for_code(high_code).key
                break
        self.assertIsNotNone(chosen_high_key)
        self.assertNotEqual(chosen_spec.reroll_dice_mask, 0)
        self.assertNotEqual(chosen_high_key, "keep_all")


if __name__ == "__main__":
    unittest.main()
