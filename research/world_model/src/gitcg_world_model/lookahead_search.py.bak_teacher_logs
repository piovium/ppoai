from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .action_hierarchy import (
    low_level_kind_for_code,
    match_low_level_code_index,
)
from .decks import SMALL_DECK_MATCHUPS
from .env import GitcgDecisionEnv
from .ppo_features import TokenObservationEncoder
from .ppo_model import PpoTransformerPolicy, masked_log_softmax
from .schema import ActionChoice, DecisionContext, EnvConfig, EpisodeRecord, Matchup, TrajectoryStep


@dataclass(frozen=True)
class LookaheadSearchConfig:
    enabled: bool = True
    round_threshold: int = 6
    legal_option_threshold: int = 6
    root_top_k: int = 8
    depth: int = 3
    policy_temperature: float = 0.75
    policy_distill_coef: float = 0.20
    value_distill_coef: float = 0.10
    max_states_per_episode: int = 24


@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    value_target: float
    weight: float


_MATCHUP_BY_KEY = {matchup.key: matchup for matchup in SMALL_DECK_MATCHUPS}


def annotate_episodes_with_search_targets(
    episodes: Sequence[EpisodeRecord],
    *,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    env_config: EnvConfig,
    config: LookaheadSearchConfig,
) -> tuple[EpisodeRecord, ...]:
    # Training-only full-information teacher path. Runtime acting must not call
    # this module as its search implementation.
    if not config.enabled:
        return tuple(episodes)
    annotated: list[EpisodeRecord] = []
    for episode in episodes:
        annotated.append(
            _annotate_episode(
                episode,
                model=model,
                encoder=encoder,
                device=device,
                env_config=env_config,
                config=config,
            )
        )
    return tuple(annotated)


def _annotate_episode(
    episode: EpisodeRecord,
    *,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    env_config: EnvConfig,
    config: LookaheadSearchConfig,
) -> EpisodeRecord:
    matchup = _MATCHUP_BY_KEY.get(episode.matchup)
    if matchup is None:
        return episode
    remaining_budget = int(config.max_states_per_episode)
    steps: list[TrajectoryStep] = []
    for step in episode.steps:
        target: SearchTeacherTarget | None = None
        if float(step.metadata.get("search_teacher_weight", 0.0)) > 0.0:
            steps.append(step)
            continue
        if remaining_budget > 0 and _is_search_state(step, config=config):
            target = _search_teacher_target(
                step=step,
                matchup=matchup,
                model=model,
                encoder=encoder,
                device=device,
                env_config=env_config,
                config=config,
                seed=episode.seed,
            )
            if target is not None:
                remaining_budget -= 1
        if target is None:
            steps.append(step)
            continue
        metadata = dict(step.metadata)
        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
        steps.append(replace(step, metadata=metadata))
    return replace(episode, steps=tuple(steps))


def _is_search_state(step: TrajectoryStep, *, config: LookaheadSearchConfig) -> bool:
    if step.request_type.name != "ACTION":
        return False
    if step.player_view is None or step.full_state_json_before is None:
        return False
    if step.pre_state.round_number >= config.round_threshold:
        return True
    if len(step.legal_low_level_codes) >= config.legal_option_threshold:
        return True
    productive_non_end = any(
        low_level_kind_for_code(int(action_code)).value != "action_declare_end"
        for action_code in step.legal_low_level_codes
    )
    has_end = any(
        low_level_kind_for_code(int(action_code)).value == "action_declare_end"
        for action_code in step.legal_low_level_codes
    )
    return productive_non_end and has_end


def _search_teacher_target(
    *,
    step: TrajectoryStep,
    matchup: Matchup,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    env_config: EnvConfig,
    config: LookaheadSearchConfig,
    seed: int | None,
) -> SearchTeacherTarget | None:
    root_context = _rebuild_root_context(
        state_json=step.full_state_json_before,
        matchup=matchup,
        env_config=env_config,
        seed=seed,
    )
    if root_context is None or root_context.player_view is None or not root_context.legal_low_level_codes:
        return None
    root_option_indices = _candidate_indices(
        context=root_context,
        model=model,
        encoder=encoder,
        device=device,
        top_k=config.root_top_k,
    )
    if not root_option_indices:
        return None
    score_by_index: dict[int, float] = {}
    for index in root_option_indices:
        root_action_code = int(root_context.legal_low_level_codes[index])
        score_by_index[index] = _evaluate_root_option(
            step=step,
            matchup=matchup,
            env_config=env_config,
            model=model,
            encoder=encoder,
            device=device,
            action_code=root_action_code,
            depth=config.depth,
            seed=seed,
        )
    if not score_by_index:
        return None
    base_log_probs = _policy_log_probs(
        context=root_context,
        model=model,
        encoder=encoder,
        device=device,
    )
    teacher_logits = list(base_log_probs)
    for index, score in score_by_index.items():
        teacher_logits[index] = float(score) / max(1e-6, config.policy_temperature)
    policy_target = _softmax_tuple(teacher_logits)
    best_score = max(score_by_index.values())
    teacher_weight = max(config.policy_distill_coef, config.value_distill_coef)
    return SearchTeacherTarget(
        policy_target=policy_target,
        value_target=float(best_score),
        weight=float(teacher_weight),
    )


def _evaluate_root_option(
    *,
    step: TrajectoryStep,
    matchup: Matchup,
    env_config: EnvConfig,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    action_code: int,
    depth: int,
    seed: int | None,
) -> float:
    env = GitcgDecisionEnv(
        _teacher_env_config(env_config),
        matchup,
        enable_result_based_action_relabel=False,
    )
    try:
        context = env.reset(seed=seed, state_json=step.full_state_json_before)
        option_index = match_low_level_code_index(context, action_code)
        if option_index is None:
            return -1.0
        next_context, transition, done = env.step(ActionChoice(action_code=context.legal_low_level_codes[option_index]))
        root_player = int(step.acting_player)
        if done:
            return _terminal_value(transition.post_state.winner, root_player=root_player)
        current_context = next_context
        remaining_depth = max(0, depth - 1)
        while remaining_depth > 0 and current_context is not None and not current_context.terminal:
            greedy_index = _greedy_action_index(
                context=current_context,
                model=model,
                encoder=encoder,
                device=device,
            )
            if greedy_index is None:
                break
            current_context, transition, done = env.step(
                ActionChoice(action_code=current_context.legal_low_level_codes[greedy_index])
            )
            if done:
                return _terminal_value(transition.post_state.winner, root_player=root_player)
            remaining_depth -= 1
        if current_context is None:
            return 0.0
        privileged_state = torch.as_tensor(
            encoder.event.encode_privileged_state(current_context.full_state),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.inference_mode():
            return float(model.oracle_value(privileged_state)[0].item())
    finally:
        env.close()


def _candidate_indices(
    *,
    context: DecisionContext,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    top_k: int,
) -> tuple[int, ...]:
    if not context.legal_low_level_codes:
        return ()
    log_probs = _policy_log_probs(context=context, model=model, encoder=encoder, device=device)
    ranked = sorted(range(len(log_probs)), key=lambda index: log_probs[index], reverse=True)
    limit = max(1, min(len(ranked), int(top_k)))
    return tuple(ranked[:limit])


def _policy_log_probs(
    *,
    context: DecisionContext,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> tuple[float, ...]:
    encoded = encoder.encode_context(context, history=(), include_training_targets=False)
    token_features = torch.as_tensor(encoded.token_features, dtype=torch.float32, device=device).unsqueeze(0)
    token_mask = torch.as_tensor(encoded.token_mask, dtype=torch.bool, device=device).unsqueeze(0)
    opponent_token_mask = torch.as_tensor(encoded.opponent_token_mask, dtype=torch.bool, device=device).unsqueeze(0)
    option_features = torch.as_tensor(encoded.option_features, dtype=torch.float32, device=device).unsqueeze(0)
    option_mask = torch.as_tensor(encoded.option_mask, dtype=torch.bool, device=device).unsqueeze(0)
    opponent_tag_ids = torch.as_tensor([encoded.opponent_tag_id], dtype=torch.long, device=device)
    opponent_entity_ids = torch.as_tensor([encoded.opponent_entity_id], dtype=torch.long, device=device)
    opponent_deck_ids = torch.as_tensor([encoded.opponent_deck_id], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            token_features=token_features,
            token_mask=token_mask,
            option_features=option_features,
            option_mask=option_mask,
            opponent_token_mask=opponent_token_mask,
            opponent_tag_ids=opponent_tag_ids,
            opponent_entity_ids=opponent_entity_ids,
            opponent_deck_ids=opponent_deck_ids,
            recurrent_state=None,
        )
        log_probs = masked_log_softmax(output.policy_logits[0], option_mask[0])
    return tuple(float(value) for value in log_probs.detach().cpu().tolist())


def _greedy_action_index(
    *,
    context: DecisionContext,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> int | None:
    if context.player_view is None or not context.legal_low_level_codes:
        return None
    log_probs = _policy_log_probs(context=context, model=model, encoder=encoder, device=device)
    return max(range(len(log_probs)), key=lambda index: log_probs[index])


def _rebuild_root_context(
    *,
    state_json: str | None,
    matchup: Matchup,
    env_config: EnvConfig,
    seed: int | None,
) -> DecisionContext | None:
    if not state_json:
        return None
    env = GitcgDecisionEnv(
        _teacher_env_config(env_config),
        matchup,
        enable_result_based_action_relabel=False,
    )
    try:
        return env.reset(seed=seed, state_json=state_json)
    except Exception:
        return None
    finally:
        env.close()


def _teacher_env_config(env_config: EnvConfig) -> EnvConfig:
    return EnvConfig(
        deck_pool=env_config.deck_pool,
        version=env_config.version,
        max_rounds=env_config.max_rounds,
        seed=env_config.seed,
        seed_policy=env_config.seed_policy,
        record_full_state_json=True,
        record_player_view=True,
        draw_penalty=env_config.draw_penalty,
    )



def _softmax_tuple(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        return ()
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    if total <= 0.0:
        return tuple(1.0 / float(len(logits)) for _ in logits)
    return tuple(value / total for value in exp_values)


def _terminal_value(winner: int | None, *, root_player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == int(root_player) else -1.0
