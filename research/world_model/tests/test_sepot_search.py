from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import gitcg_world_model.sepot_search as sepot_search_module
from gitcg_world_model.ppo_agent import PpoAgent
from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.ppo_model import PpoModelConfig, PpoTransformerPolicy
from gitcg_world_model.public_state import PublicStateTracker
from gitcg_world_model.schema import DecisionContext
from gitcg_world_model.sepot_search import (
    SePotPhaseBudget,
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

    def test_rollout_keeps_partial_branch_scores_when_budget_is_exhausted(self):
        episode = synthetic_episode_record()
        step = next(item for item in episode.steps if item.request_type.value == "action")
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
            full_state_json="{}",
        )
        self.assertGreaterEqual(len(context.legal_low_level_codes), 2)

        encoder = TokenObservationEncoder()
        controller = SePotSearchController(card_vocabulary=encoder.config.card_vocabulary)
        tracker = PublicStateTracker(player_id=0)
        budget = phase_budget_for_context(context, config=controller.config)
        belief = build_public_belief_state(
            context=context,
            tracker=tracker,
            card_vocabulary=encoder.config.card_vocabulary,
            opponent_deck_name="sample_b",
            belief_histogram=(0.0,) * encoder.belief_histogram_dim,
            belief_remaining_deck_histogram=(0.0,) * encoder.belief_remaining_deck_histogram_dim,
            belief_samples=budget.belief_samples,
            root_player=0,
        )

        call_count = {"value": 0}

        def fake_advance_until_action(**kwargs):
            return (
                kwargs["current_context"],
                kwargs["current_sampled_state_json"],
                kwargs["current_public_belief"],
                kwargs["current_history"],
                kwargs["sampled_self_hypothesis"],
                kwargs["sampled_opponent_hypothesis"],
                kwargs["current_tracker"],
            )

        def fake_internal_candidate(**kwargs):
            del kwargs
            call_count["value"] += 1
            if call_count["value"] == 1:
                return 0.42
            return None

        controller._advance_until_action = fake_advance_until_action  # type: ignore[method-assign]
        controller._evaluate_internal_self_candidate = fake_internal_candidate  # type: ignore[method-assign]

        dummy_env = type("DummyEnv", (), {"_config": None, "_matchup": None})()
        sampled_self = belief.self_range.hypotheses[0]
        sampled_opponent = belief.opponent_range.hypotheses[0]
        matchup = sepot_search_module._matchup_from_key(episode.matchup)
        env_config = sepot_search_module._env_config_from_matchup(matchup)
        fake_log_probs = tuple(
            float(len(context.legal_low_level_codes) - index)
            for index in range(len(context.legal_low_level_codes))
        )

        with patch.object(sepot_search_module, "_policy_log_probs", return_value=fake_log_probs):
            value = controller._rollout_value(
                env=dummy_env,
                current_context=context,
                current_sampled_state_json="{}",
                current_public_belief=belief,
                current_history=[],
                root_player=0,
                budget=budget,
                depth_remaining=1,
                model=None,
                encoder=encoder,
                device=torch.device("cpu"),
                opponent_deck_name="sample_b",
                search_state_value_fn=lambda **_: torch.tensor([0.0], dtype=torch.float32),
                deadline=1.0e9,
                updater=None,
                sampled_self_hypothesis=sampled_self,
                sampled_opponent_hypothesis=sampled_opponent,
                current_tracker=tracker,
                policy_cache={},
                leaf_value_cache={},
                env_config=env_config,
                matchup=matchup,
            )

        self.assertAlmostEqual(value, 0.42, places=6)
        self.assertEqual(call_count["value"], 2)

    def test_evaluate_root_candidate_handles_terminal_branch_step_without_context(self):
        episode = synthetic_episode_record()
        step = next(item for item in episode.steps if item.request_type.value == "action")
        context = DecisionContext(
            acting_player=step.acting_player,
            request_type=step.request_type,
            step_index=0,
            full_state=step.pre_state,
            legal_low_level_codes=step.legal_low_level_codes,
            legal_low_level_specs=step.legal_low_level_specs,
            legal_low_level_mask=step.legal_low_level_mask,
            legal_high_level_codes=step.legal_high_level_codes,
            high_to_low_map=step.high_to_low_map,
            player_view=step.player_view,
            metadata={
                "matchup": episode.matchup,
                "opponent_deck_name": "sample_b",
            },
            full_state_json=step.full_state_json_before or "{}",
        )
        encoder = TokenObservationEncoder()
        controller = SePotSearchController(card_vocabulary=encoder.config.card_vocabulary)
        tracker = PublicStateTracker(player_id=int(step.acting_player))
        budget = SePotPhaseBudget(depth=2, root_top_k=1, belief_samples=1, timeout_ms=1000)
        belief = build_public_belief_state(
            context=context,
            tracker=tracker,
            card_vocabulary=encoder.config.card_vocabulary,
            opponent_deck_name="sample_b",
            belief_histogram=(0.0,) * encoder.belief_histogram_dim,
            belief_remaining_deck_histogram=(0.0,) * encoder.belief_remaining_deck_histogram_dim,
            belief_samples=budget.belief_samples,
            root_player=int(step.acting_player),
        )
        matchup = sepot_search_module._matchup_from_key(episode.matchup)
        env_config = sepot_search_module._env_config_from_matchup(matchup)
        captured: dict[str, object] = {}

        def fake_rollout_value(**kwargs):
            captured["current_context"] = kwargs["current_context"]
            captured["current_sampled_state_json"] = kwargs["current_sampled_state_json"]
            return 0.5

        controller._rollout_value = fake_rollout_value  # type: ignore[method-assign]

        class FakeEnv:
            def step(self, _choice):
                return None, step, True

            def close(self):
                return None

        class FakeUpdater:
            def update_after_step(self, **_kwargs):
                return type(
                    "UpdateResult",
                    (),
                    {
                        "updated_public_belief": belief,
                        "sampled_self_hypothesis": belief.self_range.hypotheses[0],
                        "sampled_opponent_hypothesis": belief.opponent_range.hypotheses[0],
                    },
                )()

        with patch.object(sepot_search_module, "_try_reset_branch_env", return_value=(FakeEnv(), context)):
            value = controller._evaluate_root_candidate(
                root_context=context,
                root_history=[],
                root_tracker=tracker,
                root_public_belief=belief,
                root_action_code=int(context.legal_low_level_codes[0]),
                sampled_opponent_hypothesis=belief.opponent_range.hypotheses[0],
                model=None,
                encoder=encoder,
                device=torch.device("cpu"),
                budget=budget,
                deadline=1.0e9,
                reconstructor=None,
                updater=FakeUpdater(),
                opponent_deck_name="sample_b",
                search_state_value_fn=lambda **_: torch.tensor([0.0], dtype=torch.float32),
                policy_cache={},
                leaf_value_cache={},
                env_config=env_config,
                matchup=matchup,
            )

        self.assertAlmostEqual(value, 0.5, places=6)
        self.assertIsNone(captured["current_context"])
        self.assertIsNone(captured["current_sampled_state_json"])


if __name__ == "__main__":
    unittest.main()
