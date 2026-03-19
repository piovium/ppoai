from __future__ import annotations

from dataclasses import replace

from .agents import PolicyAgent
from .env import GitcgDecisionEnv
from .schema import ActionChoice, EnvConfig, EpisodeRecord, Matchup


def run_event_episode(
    config: EnvConfig,
    matchup: Matchup,
    agent0: PolicyAgent,
    agent1: PolicyAgent,
    *,
    seed: int | None = None,
    max_decisions: int | None = None,
) -> EpisodeRecord:
    env = GitcgDecisionEnv(config=config, matchup=matchup)
    try:
        context = env.reset(seed=seed)
        decision_count = 0
        updated_steps = []
        while True:
            if max_decisions is not None and decision_count >= max_decisions:
                partial = env.partial_episode_record(reason=f"max_decisions={max_decisions}")
                return replace(partial, steps=tuple(updated_steps))
            agent = agent0 if context.acting_player == 0 else agent1
            choice = agent.choose_action(context)
            metadata = {}
            if hasattr(agent, "pop_last_decision_metadata"):
                metadata = dict(getattr(agent, "pop_last_decision_metadata")() or {})
            next_context, trajectory_step, done = env.step(choice)
            if metadata:
                trajectory_step = replace(
                    trajectory_step,
                    metadata={**trajectory_step.metadata, **metadata},
                )
            updated_steps.append(trajectory_step)
            if hasattr(agent0, "observe_step"):
                getattr(agent0, "observe_step")(trajectory_step)
            if hasattr(agent1, "observe_step"):
                getattr(agent1, "observe_step")(trajectory_step)
            decision_count += 1
            if done:
                episode = env.episode_record()
                return replace(episode, steps=tuple(updated_steps))
            assert next_context is not None
            context = next_context
    finally:
        env.close()
