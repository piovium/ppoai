from __future__ import annotations

import unittest

import torch

from gitcg_world_model.agents import LegalRandomAgent
from gitcg_world_model.decks import SMALL_DECK_MATCHUPS, SMALL_DECK_POOL
from gitcg_world_model.event_rollout import run_event_episode
from gitcg_world_model.lookahead_search import (
    _evaluate_root_option,
    _evaluate_root_options,
    _policy_log_probs,
    _policy_log_probs_batch,
    _rebuild_root_context,
)
from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.ppo_model import PpoModelConfig, PpoTransformerPolicy
from gitcg_world_model.schema import EnvConfig


def _tiny_model_and_encoder() -> tuple[PpoTransformerPolicy, TokenObservationEncoder, torch.device]:
    encoder = TokenObservationEncoder()
    torch.manual_seed(0)
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
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            option_hidden_dim=64,
            privileged_hidden_dim=64,
            recurrent_hidden_dim=0,
        )
    )
    model.eval()
    return model, encoder, torch.device("cpu")


class LookaheadSearchTests(unittest.TestCase):
    def test_policy_log_probs_batch_matches_single_context_calls(self):
        model, encoder, device = _tiny_model_and_encoder()
        env_config = EnvConfig(deck_pool=SMALL_DECK_POOL, record_full_state_json=True, record_player_view=True)

        from gitcg_world_model.env import GitcgDecisionEnv

        env_a = GitcgDecisionEnv(env_config, SMALL_DECK_MATCHUPS[0], enable_result_based_action_relabel=False)
        env_b = GitcgDecisionEnv(env_config, SMALL_DECK_MATCHUPS[1], enable_result_based_action_relabel=False)
        try:
            context_a = env_a.reset(seed=17)
            context_b = env_b.reset(seed=23)
        finally:
            env_a.close()
            env_b.close()

        batch = _policy_log_probs_batch(
            contexts=(context_a, context_b),
            model=model,
            encoder=encoder,
            device=device,
        )
        single_a = _policy_log_probs(context=context_a, model=model, encoder=encoder, device=device)
        single_b = _policy_log_probs(context=context_b, model=model, encoder=encoder, device=device)

        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), len(single_a))
        self.assertEqual(len(batch[1]), len(single_b))
        for actual, expected in zip(batch[0], single_a, strict=False):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(batch[1], single_b, strict=False):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_batched_root_option_evaluation_matches_single_option_path(self):
        model, encoder, device = _tiny_model_and_encoder()
        env_config = EnvConfig(deck_pool=SMALL_DECK_POOL, record_full_state_json=True, record_player_view=True)
        matchup = SMALL_DECK_MATCHUPS[0]
        episode = run_event_episode(
            config=env_config,
            matchup=matchup,
            agent0=LegalRandomAgent(seed=1),
            agent1=LegalRandomAgent(seed=2),
            seed=101,
            max_decisions=48,
        )
        step = next(
            item
            for item in episode.steps
            if item.full_state_json_before is not None
            and item.player_view is not None
            and len(item.legal_low_level_codes) >= 3
        )
        root_context = _rebuild_root_context(
            state_json=step.full_state_json_before,
            matchup=matchup,
            env_config=env_config,
            seed=episode.seed,
        )
        self.assertIsNotNone(root_context)
        assert root_context is not None
        root_action_codes = tuple(
            (index, int(root_context.legal_low_level_codes[index]))
            for index in range(min(3, len(root_context.legal_low_level_codes)))
        )

        batched = _evaluate_root_options(
            step=step,
            matchup=matchup,
            env_config=env_config,
            model=model,
            encoder=encoder,
            device=device,
            root_action_codes=root_action_codes,
            depth=3,
            seed=episode.seed,
        )
        sequential = {
            int(index): _evaluate_root_option(
                step=step,
                matchup=matchup,
                env_config=env_config,
                model=model,
                encoder=encoder,
                device=device,
                action_code=int(action_code),
                depth=3,
                seed=episode.seed,
            )
            for index, action_code in root_action_codes
        }

        self.assertEqual(set(batched), set(sequential))
        for index in sequential:
            self.assertAlmostEqual(batched[index], sequential[index], places=6)


if __name__ == "__main__":
    unittest.main()
