from __future__ import annotations

import unittest

import torch

from gitcg_world_model.ppo_agent import PpoAgent
from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.ppo_model import PpoModelConfig, PpoTransformerPolicy
from gitcg_world_model.public_state import PublicStateTracker
from gitcg_world_model.schema import DecisionContext
from gitcg_world_model.sepot_search import (
    SePotSearchController,
    SePotSearchConfig,
    build_public_belief_state,
    phase_budget_for_context,
    range_summary,
    state_search_context_summary,
)
from gitcg_world_model.testsupport import synthetic_episode_record


class SePotSearchTests(unittest.TestCase):
    def test_public_belief_state_contains_explicit_ranges(self):
        episode = synthetic_episode_record()
        step = episode.steps[0]
        context = DecisionContext(
            acting_player=step.acting_player,
            request_type=step.request_type,
            step_index=0,
            full_state=step.pre_state,
            legal_low_level_codes=step.legal_low_level_codes,
            legal_low_level_mask=step.legal_low_level_mask,
            legal_high_level_codes=step.legal_high_level_codes,
            high_to_low_map=step.high_to_low_map,
            player_view=step.player_view,
            metadata={
                "matchup": episode.matchup,
                "opponent_deck_name": "sample_b",
            },
        )
        tracker = PublicStateTracker(player_id=0)
        encoder = TokenObservationEncoder()
        budget = phase_budget_for_context(context, config=SePotSearchConfig())

        belief = build_public_belief_state(
            context=context,
            tracker=tracker,
            card_vocabulary=encoder.config.card_vocabulary,
            opponent_deck_name="sample_b",
            belief_histogram=(0.0,) * encoder.belief_histogram_dim,
            belief_remaining_deck_histogram=(0.0,) * encoder.belief_remaining_deck_histogram_dim,
            belief_samples=budget.belief_samples,
        )

        self.assertGreaterEqual(belief.self_range.count, 1)
        self.assertGreaterEqual(belief.opponent_range.count, 1)
        self.assertEqual(len(belief.self_range.hypotheses[0].hidden_dice), 3)
        self.assertEqual(len(state_search_context_summary(context=context, budget=budget)), 16)

    def test_search_heads_accept_range_and_context_summaries(self):
        encoder = TokenObservationEncoder()
        model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=encoder.option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
                belief_group_dim=encoder.belief_group_dim,
                opponent_tag_vocab_size=encoder.opponent_tag_vocab_size,
                opponent_entity_vocab_size=encoder.opponent_entity_vocab_size,
                deck_vocab_size=encoder.deck_vocab_size,
                search_range_summary_dim=(encoder.belief_histogram_dim * 2) + 12,
                search_context_dim=16,
            )
        )
        batch_size = 2
        option_count = 3
        option_features = torch.zeros((batch_size, option_count, encoder.option_dim), dtype=torch.float32)
        self_range_summary = torch.zeros((batch_size, (encoder.belief_histogram_dim * 2) + 12), dtype=torch.float32)
        opponent_range_summary = torch.zeros_like(self_range_summary)
        search_context_summary = torch.zeros((batch_size, 16), dtype=torch.float32)

        logits = model.search_option_logits(
            option_features=option_features,
            self_range_summary=self_range_summary,
            opponent_range_summary=opponent_range_summary,
            search_context_summary=search_context_summary,
        )
        values = model.search_state_value(
            self_range_summary=self_range_summary,
            opponent_range_summary=opponent_range_summary,
            search_context_summary=search_context_summary,
        )

        self.assertEqual(tuple(logits.shape), (batch_size, option_count))
        self.assertEqual(tuple(values.shape), (batch_size,))

    def test_ppo_agent_marks_search_triggered_metadata(self):
        episode = synthetic_episode_record()
        step = episode.steps[0]
        context = DecisionContext(
            acting_player=step.acting_player,
            request_type=step.request_type,
            step_index=0,
            full_state=step.pre_state,
            legal_low_level_codes=step.legal_low_level_codes,
            legal_low_level_mask=step.legal_low_level_mask,
            legal_high_level_codes=step.legal_high_level_codes,
            high_to_low_map=step.high_to_low_map,
            player_view=step.player_view,
            metadata={
                "matchup": episode.matchup,
                "opponent_deck_name": "sample_b",
                "opponent_tag": "unknown",
                "opponent_entity_key": "baseline:sample_b",
            },
        )
        encoder = TokenObservationEncoder()
        model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=encoder.option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
                belief_group_dim=encoder.belief_group_dim,
                opponent_tag_vocab_size=encoder.opponent_tag_vocab_size,
                opponent_entity_vocab_size=encoder.opponent_entity_vocab_size,
                deck_vocab_size=encoder.deck_vocab_size,
                search_range_summary_dim=(encoder.belief_histogram_dim * 2) + 12,
                search_context_dim=16,
            )
        )
        agent = PpoAgent(
            model=model,
            encoder=encoder,
            device="cpu",
            sample=False,
            temperature=1.0,
            player_id=0,
            opponent_tag="unknown",
            opponent_entity_key="baseline:sample_b",
            opponent_deck_name="sample_b",
        )

        agent.choose_action(context)
        metadata = agent.pop_last_decision_metadata()

        self.assertTrue(bool(metadata["sepot_triggered"]))
        self.assertIn("search_teacher_weight", metadata)
        self.assertGreaterEqual(float(metadata["sepot_root_candidate_count"]), 1.0)

    def test_runtime_search_path_ignores_live_root_truth_when_player_view_matches(self):
        episode = synthetic_episode_record()
        step = episode.steps[0]
        context_a = DecisionContext(
            acting_player=step.acting_player,
            request_type=step.request_type,
            step_index=0,
            full_state=step.pre_state,
            legal_low_level_codes=step.legal_low_level_codes,
            legal_low_level_mask=step.legal_low_level_mask,
            legal_high_level_codes=step.legal_high_level_codes,
            high_to_low_map=step.high_to_low_map,
            player_view=step.player_view,
            metadata={
                "matchup": episode.matchup,
                "opponent_deck_name": "sample_b",
                "opponent_tag": "unknown",
                "opponent_entity_key": "baseline:sample_b",
            },
            full_state_json="{\"root\":\"live_a_should_be_ignored\"}",
        )
        context_b = DecisionContext(
            acting_player=step.acting_player,
            request_type=step.request_type,
            step_index=0,
            full_state=step.post_state,
            legal_low_level_codes=step.legal_low_level_codes,
            legal_low_level_mask=step.legal_low_level_mask,
            legal_high_level_codes=step.legal_high_level_codes,
            high_to_low_map=step.high_to_low_map,
            player_view=step.player_view,
            metadata=dict(context_a.metadata),
            full_state_json="{\"root\":\"live_b_should_be_ignored\"}",
        )
        encoder = TokenObservationEncoder()
        model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=encoder.option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
                belief_group_dim=encoder.belief_group_dim,
                opponent_tag_vocab_size=encoder.opponent_tag_vocab_size,
                opponent_entity_vocab_size=encoder.opponent_entity_vocab_size,
                deck_vocab_size=encoder.deck_vocab_size,
                search_range_summary_dim=(encoder.belief_histogram_dim * 2) + 12,
                search_context_dim=16,
            )
        )
        controller = SePotSearchController(
            card_vocabulary=encoder.config.card_vocabulary,
            config=SePotSearchConfig(
                opening_timeout_ms=5000,
                midgame_timeout_ms=5000,
                endgame_timeout_ms=5000,
            ),
        )
        agent_a = PpoAgent(
            model=model,
            encoder=encoder,
            device="cpu",
            sample=False,
            player_id=0,
            opponent_deck_name="sample_b",
            sepot_controller=controller,
        )
        agent_b = PpoAgent(
            model=model,
            encoder=encoder,
            device="cpu",
            sample=False,
            player_id=0,
            opponent_deck_name="sample_b",
            sepot_controller=controller,
        )

        choice_a = agent_a.choose_action(context_a)
        metadata_a = agent_a.pop_last_decision_metadata()
        choice_b = agent_b.choose_action(context_b)
        metadata_b = agent_b.pop_last_decision_metadata()

        self.assertEqual(choice_a.action_code, choice_b.action_code)
        self.assertEqual(metadata_a["sepot_triggered"], metadata_b["sepot_triggered"])
        self.assertEqual(metadata_a["sepot_fallback_reason"], metadata_b["sepot_fallback_reason"])
        self.assertEqual(metadata_a.get("search_teacher_policy"), metadata_b.get("search_teacher_policy"))


if __name__ == "__main__":
    unittest.main()
