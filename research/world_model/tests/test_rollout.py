from __future__ import annotations

import unittest
from unittest.mock import patch

from gitcg_world_model.rollout import run_episode
from gitcg_world_model.schema import (
    ActionChoice,
    CharacterSnapshot,
    DecisionContext,
    DecisionType,
    EpisodeRecord,
    EnvConfig,
    Matchup,
    PlayerSnapshot,
    StateSnapshot,
)


class RolloutTests(unittest.TestCase):
    def test_run_episode_returns_truncated_record_when_max_decisions_is_reached(self):
        context = DecisionContext(
            acting_player=0,
            request_type=DecisionType.ACTION,
            step_index=0,
            full_state=_state_snapshot(),
        )
        truncated = EpisodeRecord(
            matchup="sample_a__vs__sample_b",
            seed=7,
            winner=None,
            steps=(),
            final_state=_state_snapshot(),
            final_state_json=None,
            metadata={
                "truncated": True,
                "truncation_reason": "max_decisions=0",
                "terminal_value": 0.0,
            },
        )

        class FakeEnv:
            def reset(self, seed=None):
                del seed
                return context

            def step(self, choice):
                raise AssertionError(f"unexpected step call: {choice}")

            def partial_episode_record(self, *, reason=None):
                self.reason = reason
                return truncated

            def close(self):
                return None

        with patch("gitcg_world_model.rollout.GitcgDecisionEnv", return_value=FakeEnv()):
            episode = run_episode(
                config=EnvConfig(deck_pool=()),
                matchup=Matchup("sample_a", "sample_b"),
                agent0=_FixedAgent(),
                agent1=_FixedAgent(),
                seed=7,
                max_decisions=0,
            )

        self.assertTrue(episode.metadata["truncated"])
        self.assertEqual(episode.metadata["truncation_reason"], "max_decisions=0")
        self.assertEqual(episode.metadata["terminal_value"], 0.0)


class _FixedAgent:
    def choose_action(self, _context):
        return ActionChoice(action_code=0)


def _state_snapshot() -> StateSnapshot:
    return StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            _player_snapshot(1, 1411),
            _player_snapshot(11, 1609),
        ),
    )


def _player_snapshot(active_id: int, definition_id: int) -> PlayerSnapshot:
    return PlayerSnapshot(
        active_character_id=active_id,
        characters=(
            CharacterSnapshot(
                id=active_id,
                definition_id=definition_id,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=True,
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
