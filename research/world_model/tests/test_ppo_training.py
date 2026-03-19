from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.ppo_model import masked_log_softmax
from gitcg_world_model.ppo_training import (
    _action_efficiency_reward,
    _resolve_micro_batch_size,
    _select_aux_replay_chunks,
    build_ppo_chunks,
    build_ppo_samples,
)
from gitcg_world_model.schema import ActionChoice, DecisionType, EntitySnapshot, EpisodeRecord, LowLevelActionSpec, OptionKind
from gitcg_world_model.testsupport import encode_step_actions, materialize_legal_specs, synthetic_episode_record, synthetic_legal_specs


class PpoTrainingTests(unittest.TestCase):
    def test_resolve_micro_batch_size_caps_large_cuda_models(self):
        size = _resolve_micro_batch_size(
            batch_size=64,
            requested_micro_batch_size=None,
            param_count=31_000_000,
            device=torch.device("cuda"),
        )
        self.assertEqual(size, 8)

    def test_resolve_micro_batch_size_keeps_small_cpu_batches(self):
        size = _resolve_micro_batch_size(
            batch_size=32,
            requested_micro_batch_size=None,
            param_count=5_000_000,
            device=torch.device("cpu"),
        )
        self.assertEqual(size, 32)

    def test_masked_log_softmax_supports_float16_masks(self):
        logits = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        mask = torch.tensor([[True, False]])
        output = masked_log_softmax(logits, mask)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output[:, :1]).all())

    def test_build_ppo_samples_only_uses_trainable_steps(self):
        episode = synthetic_episode_record()
        first = replace(
            episode.steps[0],
            metadata={
                "ppo_trainable": True,
                "selected_logprob": -0.2,
                "partial_value": 0.1,
                "opponent_tag": "baseline_random",
            },
        )
        second = replace(
            first,
            acting_player=1,
            choice=ActionChoice(action_code=first.choice.action_code),
            request_type=DecisionType.ACTION,
            reward=0.0,
            done=False,
            post_state=first.pre_state,
            metadata={
                "ppo_trainable": False,
                "selected_logprob": -0.3,
                "partial_value": 0.2,
                "opponent_tag": "baseline_random",
            },
        )
        second = encode_step_actions(second)
        final = replace(
            first,
            acting_player=0,
            reward=1.0,
            done=True,
            metadata={
                "ppo_trainable": True,
                "selected_logprob": -0.1,
                "partial_value": 0.4,
                "opponent_tag": "baseline_random",
            },
        )
        final = encode_step_actions(final)
        custom = EpisodeRecord(
            matchup=episode.matchup,
            seed=episode.seed,
            winner=0,
            steps=(second, final),
            final_state=episode.final_state,
        )
        samples = build_ppo_samples((custom,), TokenObservationEncoder())
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].action_index, 0)
        self.assertGreaterEqual(samples[0].sample_weight, 1.0)
        self.assertTrue(samples[0].opponent_token_mask)
        self.assertGreaterEqual(samples[0].opponent_tag_id, 0)
        self.assertGreaterEqual(samples[0].opponent_entity_id, 0)
        self.assertGreaterEqual(samples[0].opponent_deck_id, 0)
        self.assertEqual(
            len(samples[0].belief_remaining_deck_histogram_target),
            len(samples[0].belief_histogram_target),
        )
        self.assertEqual(len(samples[0].belief_group_hand_target), 10)
        self.assertEqual(len(samples[0].belief_group_deck_target), 10)

    def test_aux_replay_includes_late_round_winning_chunks(self):
        episode = synthetic_episode_record(winner=0)
        late_pre_state = replace(episode.steps[0].pre_state, round_number=14)
        late_post_state = replace(episode.steps[0].post_state, round_number=15, winner=0)
        late_step = replace(
            episode.steps[0],
            pre_state=late_pre_state,
            post_state=late_post_state,
            metadata={
                "ppo_trainable": True,
                "selected_logprob": -0.1,
                "partial_value": 0.2,
                "opponent_tag": "baseline_random",
            },
        )
        late_episode = replace(episode, steps=(late_step,), final_state=late_post_state, winner=0)

        chunks = build_ppo_chunks((late_episode,), TokenObservationEncoder())
        replay_chunks = _select_aux_replay_chunks(chunks)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].has_late_round)
        self.assertFalse(chunks[0].has_terminal_loss)
        self.assertIn(chunks[0], replay_chunks)

    def test_action_efficiency_reward_penalizes_premature_end(self):
        episode = synthetic_episode_record(winner=0)
        visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
        _, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=visible_state,
            legal_specs=synthetic_legal_specs(),
        )
        end_code = next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_DECLARE_END)
        step = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=end_code),
        )
        step = encode_step_actions(step)
        self.assertLess(_action_efficiency_reward(step, perspective_player=0), 0.0)

    def test_action_efficiency_reward_does_not_penalize_normal_skill_use(self):
        episode = synthetic_episode_record(winner=0)
        self.assertEqual(_action_efficiency_reward(episode.steps[0], perspective_player=0), 0.0)

    def test_action_efficiency_reward_ignores_declare_end_when_only_tuning_remains(self):
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
        visible_state = episode.steps[0].player_view or episode.steps[0].pre_state
        _, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=visible_state,
            legal_specs=legal_specs,
        )
        end_code = next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_DECLARE_END)
        tuning_only = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=end_code),
        )
        tuning_only = encode_step_actions(tuning_only, legal_specs=legal_specs)
        self.assertEqual(_action_efficiency_reward(tuning_only, perspective_player=0), 0.0)

    def test_action_efficiency_reward_lightly_penalizes_tuning_followup_potential(self):
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
        _, materialized, _ = materialize_legal_specs(
            acting_player=0,
            visible_state=pre_state,
            legal_specs=legal_specs,
        )
        end_code = next(spec.action_code for spec in materialized if spec.kind == OptionKind.ACTION_DECLARE_END)
        tuning_then_end = replace(
            episode.steps[0],
            choice=ActionChoice(action_code=end_code),
            pre_state=pre_state,
            post_state=post_state,
        )
        tuning_then_end = encode_step_actions(tuning_then_end, legal_specs=legal_specs)
        reward = _action_efficiency_reward(tuning_then_end, perspective_player=0)
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -0.05)


if __name__ == "__main__":
    unittest.main()
