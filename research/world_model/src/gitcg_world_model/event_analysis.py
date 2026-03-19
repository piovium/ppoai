from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .action_quality import (
    declare_end_waste_ratio,
    declare_end_with_productive_options,
    is_declare_end_option,
    selected_option,
)
from .schema import EpisodeRecord


@dataclass(frozen=True)
class EpisodeAnalytics:
    perspective_player: int
    episodes: int
    wins: int
    losses: int
    draws: int
    truncated: int
    average_steps: float
    draw_rate: float
    late_round_rate: float
    average_round_number: float
    average_actions_per_round: float
    dice_waste_ratio: float
    declare_end_while_non_end_exists_rate: float
    sepot_trigger_rate: float
    sepot_fallback_rate: float
    sepot_average_latency_ms: float
    sepot_average_root_k: float
    sepot_average_belief_samples: float
    win_step_counts: tuple[int, ...]
    loss_step_counts: tuple[int, ...]
    draw_step_counts: tuple[int, ...]
    examples: tuple[dict[str, object], ...]

    @property
    def player0_wins(self) -> int:
        return self.wins if self.perspective_player == 0 else self.losses

    @property
    def player1_wins(self) -> int:
        return self.losses if self.perspective_player == 0 else self.wins

    def to_dict(self) -> dict[str, object]:
        return {
            "perspective_player": self.perspective_player,
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "player0_wins": self.player0_wins,
            "player1_wins": self.player1_wins,
            "draws": self.draws,
            "truncated": self.truncated,
            "average_steps": self.average_steps,
            "draw_rate": self.draw_rate,
            "late_round_rate": self.late_round_rate,
            "average_round_number": self.average_round_number,
            "average_actions_per_round": self.average_actions_per_round,
            "dice_waste_ratio": self.dice_waste_ratio,
            "declare_end_while_non_end_exists_rate": self.declare_end_while_non_end_exists_rate,
            "sepot_trigger_rate": self.sepot_trigger_rate,
            "sepot_fallback_rate": self.sepot_fallback_rate,
            "sepot_average_latency_ms": self.sepot_average_latency_ms,
            "sepot_average_root_k": self.sepot_average_root_k,
            "sepot_average_belief_samples": self.sepot_average_belief_samples,
            "win_step_counts": list(self.win_step_counts),
            "loss_step_counts": list(self.loss_step_counts),
            "draw_step_counts": list(self.draw_step_counts),
            "examples": list(self.examples),
        }


def analyze_episode_records(
    episodes: Sequence[EpisodeRecord],
    *,
    player: int = 0,
    max_examples: int = 8,
) -> EpisodeAnalytics:
    wins = 0
    losses = 0
    draws = 0
    truncated = 0
    step_counts: list[int] = []
    round_numbers: list[int] = []
    actions_per_round: list[float] = []
    win_steps: list[int] = []
    loss_steps: list[int] = []
    draw_steps: list[int] = []
    examples: list[dict[str, object]] = []
    late_round_episodes = 0
    dice_waste_values: list[float] = []
    declare_end_count = 0
    declare_end_with_non_end = 0
    sepot_action_steps = 0
    sepot_triggered_steps = 0
    sepot_fallback_steps = 0
    sepot_latencies: list[float] = []
    sepot_root_candidates: list[float] = []
    sepot_belief_samples: list[float] = []

    for episode in episodes:
        steps = len(episode.steps)
        step_counts.append(steps)
        round_number = int(episode.final_state.round_number)
        round_numbers.append(round_number)
        actions_per_round.append((steps / round_number) if round_number > 0 else 0.0)
        if round_number >= 13:
            late_round_episodes += 1
        is_truncated = bool(episode.metadata.get("truncated"))
        if is_truncated:
            truncated += 1
        if episode.winner == player:
            wins += 1
            win_steps.append(steps)
        elif episode.winner is None:
            draws += 1
            draw_steps.append(steps)
        else:
            losses += 1
            loss_steps.append(steps)

        for step in episode.steps:
            waste_ratio = declare_end_waste_ratio(step)
            if waste_ratio > 0.0:
                dice_waste_values.append(waste_ratio)
            if is_declare_end_option(selected_option(step)):
                declare_end_count += 1
            if declare_end_with_productive_options(step):
                declare_end_with_non_end += 1
            if step.request_type.value == "action":
                sepot_action_steps += 1
                if bool(step.metadata.get("sepot_triggered")):
                    sepot_triggered_steps += 1
                    if str(step.metadata.get("sepot_fallback_reason") or "") not in {"", "none", "not_triggered"}:
                        sepot_fallback_steps += 1
                    sepot_latencies.append(float(step.metadata.get("sepot_latency_ms", 0.0)))
                    sepot_root_candidates.append(float(step.metadata.get("sepot_root_candidate_count", 0.0)))
                    sepot_belief_samples.append(float(step.metadata.get("sepot_belief_sample_count", 0.0)))

        if len(examples) < max_examples:
            examples.append(
                {
                    "matchup": episode.matchup,
                    "seed": episode.seed,
                    "winner": episode.winner,
                    "steps": steps,
                    "truncated": is_truncated,
                    "truncation_reason": episode.metadata.get("truncation_reason"),
                }
            )

    return EpisodeAnalytics(
        perspective_player=player,
        episodes=len(episodes),
        wins=wins,
        losses=losses,
        draws=draws,
        truncated=truncated,
        average_steps=mean(step_counts) if step_counts else 0.0,
        draw_rate=(draws / len(episodes)) if episodes else 0.0,
        late_round_rate=(late_round_episodes / len(episodes)) if episodes else 0.0,
        average_round_number=mean(round_numbers) if round_numbers else 0.0,
        average_actions_per_round=mean(actions_per_round) if actions_per_round else 0.0,
        dice_waste_ratio=mean(dice_waste_values) if dice_waste_values else 0.0,
        declare_end_while_non_end_exists_rate=(
            float(declare_end_with_non_end) / float(declare_end_count) if declare_end_count else 0.0
        ),
        sepot_trigger_rate=(float(sepot_triggered_steps) / float(sepot_action_steps) if sepot_action_steps else 0.0),
        sepot_fallback_rate=(float(sepot_fallback_steps) / float(sepot_triggered_steps) if sepot_triggered_steps else 0.0),
        sepot_average_latency_ms=mean(sepot_latencies) if sepot_latencies else 0.0,
        sepot_average_root_k=mean(sepot_root_candidates) if sepot_root_candidates else 0.0,
        sepot_average_belief_samples=mean(sepot_belief_samples) if sepot_belief_samples else 0.0,
        win_step_counts=tuple(win_steps),
        loss_step_counts=tuple(loss_steps),
        draw_step_counts=tuple(draw_steps),
        examples=tuple(examples),
    )
