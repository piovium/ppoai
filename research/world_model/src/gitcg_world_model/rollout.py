from __future__ import annotations

from .agents import PolicyAgent
from .env import GitcgDecisionEnv
from .schema import ActionChoice, EnvConfig, EpisodeRecord, Matchup


def run_episode(
    config: EnvConfig,
    matchup: Matchup,
    agent0: PolicyAgent,
    agent1: PolicyAgent,
    seed: int | None = None,
    max_decisions: int | None = None,
) -> EpisodeRecord:
    env = GitcgDecisionEnv(config=config, matchup=matchup)
    try:
        context = env.reset(seed=seed)
        decision_count = 0
        while True:
            if max_decisions is not None and decision_count >= max_decisions:
                return env.partial_episode_record(reason=f"max_decisions={max_decisions}")
            agent = agent0 if context.acting_player == 0 else agent1
            choice = agent.choose_action(context)
            next_context, _, done = env.step(choice)
            decision_count += 1
            if done:
                return env.episode_record()
            assert next_context is not None
            context = next_context
    finally:
        env.close()
