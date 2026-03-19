from __future__ import annotations

import unittest
from dataclasses import replace

from gitcg_world_model.event_analysis import analyze_episode_records
from gitcg_world_model.schema import ActionChoice, DecisionType, EntitySnapshot, LowLevelActionSpec, OptionKind
from gitcg_world_model.testsupport import encode_step_actions, materialize_legal_specs, synthetic_episode_record, synthetic_legal_specs


def _declare_end_code(episode, legal_specs):
    visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
    _, materialized, _ = materialize_legal_specs(
        acting_player=episode.steps[0].acting_player,
        visible_state=visible_state,
        legal_specs=legal_specs,
    )
    return next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_DECLARE_END)


class EventAnalysisTests(unittest.TestCase):
    def test_dice_waste_ignores_normal_actions(self):
        episode = synthetic_episode_record(winner=0)
        analytics = analyze_episode_records((episode,))
        self.assertEqual(analytics.dice_waste_ratio, 0.0)
        self.assertEqual(analytics.declare_end_while_non_end_exists_rate, 0.0)

    def test_dice_waste_and_premature_end_only_count_productive_endings(self):
        episode = synthetic_episode_record(winner=0)
        premature_end = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=_declare_end_code(episode, synthetic_legal_specs())),
        )
        premature_end = encode_step_actions(premature_end)
        analytics = analyze_episode_records((replace(episode, steps=(premature_end,)),))
        self.assertAlmostEqual(analytics.dice_waste_ratio, 3.0 / 8.0)
        self.assertEqual(analytics.declare_end_while_non_end_exists_rate, 1.0)

    def test_elemental_tuning_does_not_count_as_productive_non_end(self):
        episode = synthetic_episode_record(winner=0)
        legal_specs = (
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_DECLARE_END,
                label="declare_end",
            ),
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_ELEMENTAL_TUNING,
                label="tune",
                subject_definition_id=312004,
                discarded_card_definition_id=312004,
                target_dice=3,
            ),
        )
        tuning_end = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=_declare_end_code(episode, legal_specs)),
        )
        tuning_end = encode_step_actions(tuning_end, legal_specs=legal_specs)
        analytics = analyze_episode_records((replace(episode, steps=(tuning_end,)),))
        self.assertEqual(analytics.dice_waste_ratio, 0.0)
        self.assertEqual(analytics.declare_end_while_non_end_exists_rate, 0.0)

    def test_switch_active_does_not_count_as_productive_non_end(self):
        episode = synthetic_episode_record(winner=0)
        legal_specs = (
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_DECLARE_END,
                label="declare_end",
            ),
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_SWITCH_ACTIVE,
                label="switch",
                used_dice=(3,),
                subject_definition_id=1510,
            ),
        )
        switch_end = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=_declare_end_code(episode, legal_specs)),
        )
        switch_end = encode_step_actions(switch_end, legal_specs=legal_specs)
        analytics = analyze_episode_records((replace(episode, steps=(switch_end,)),))
        self.assertEqual(analytics.dice_waste_ratio, 0.0)
        self.assertEqual(analytics.declare_end_while_non_end_exists_rate, 0.0)

    def test_elemental_tuning_counts_when_it_has_card_followup_potential(self):
        episode = synthetic_episode_record(winner=0)
        player0 = replace(
            episode.steps[0].pre_state.players[0],
            hand_cards=(EntitySnapshot(id=101, definition_id=999001),),
            dice=(3, 4, 8),
        )
        pre_state = replace(
            episode.steps[0].pre_state,
            players=(player0, episode.steps[0].pre_state.players[1]),
        )
        post_state = replace(
            episode.steps[0].post_state,
            players=(player0, episode.steps[0].post_state.players[1]),
        )
        legal_specs = (
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_DECLARE_END,
                label="declare_end",
            ),
            LowLevelActionSpec(
                action_code=-1,
                request_type=DecisionType.ACTION,
                kind=OptionKind.ACTION_ELEMENTAL_TUNING,
                label="tune",
                used_dice=(4,),
                subject_definition_id=999001,
                discarded_card_definition_id=999001,
                target_dice=3,
            ),
        )
        tuning_end = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=_declare_end_code(episode, legal_specs)),
            pre_state=pre_state,
            post_state=post_state,
        )
        tuning_end = encode_step_actions(tuning_end, legal_specs=legal_specs)
        analytics = analyze_episode_records((replace(episode, steps=(tuning_end,)),))
        self.assertAlmostEqual(analytics.dice_waste_ratio, 3.0 / 8.0)
        self.assertEqual(analytics.declare_end_while_non_end_exists_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
