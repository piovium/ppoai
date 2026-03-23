from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .action_hierarchy import (
    legal_low_level_specs,
    match_low_level_code_index,
    semantic_action_key_for_spec,
    try_low_level_kind_for_code,
    try_semantic_action_key_for_code,
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
    # episode: only per-episode summary logs (default)
    # state: add per-state start/done logs
    # root: add per-root summary logs
    # option: full detailed per-option logs
    trace_level: str = "episode"


@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    action_semantic_keys: tuple[str, ...]
    value_target: float
    weight: float


_MATCHUP_BY_KEY = {matchup.key: matchup for matchup in SMALL_DECK_MATCHUPS}


_TRACE_LEVEL_ORDER = {"episode": 0, "state": 1, "root": 2, "option": 3}


def _jsonable_semantic_value(value):
    if isinstance(value, tuple):
        return [_jsonable_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_semantic_value(item) for item in value]
    return value


def _semantic_key_token_for_spec(spec) -> str:
    import json

    key = semantic_action_key_for_spec(spec)
    return json.dumps(_jsonable_semantic_value(key), ensure_ascii=False, separators=(",", ":"))


def _semantic_key_tokens_for_context(context: DecisionContext) -> tuple[str, ...]:
    specs = legal_low_level_specs(context)
    if specs and len(specs) == len(context.legal_low_level_codes):
        return tuple(_semantic_key_token_for_spec(spec) for spec in specs)
    tokens: list[str] = []
    for code in context.legal_low_level_codes:
        key = try_semantic_action_key_for_code(int(code))
        if key is None:
            tokens.append(f"code:{int(code)}")
            continue
        import json
        tokens.append(json.dumps(_jsonable_semantic_value(key), ensure_ascii=False, separators=(",", ":")))
    return tuple(tokens)


def _normalized_trace_level(config: LookaheadSearchConfig) -> str:
    value = os.environ.get("SEARCH_TEACHER_TRACE_LEVEL", config.trace_level)
    level = str(value).strip().lower()
    return level if level in _TRACE_LEVEL_ORDER else "episode"


def _trace_enabled(config: LookaheadSearchConfig, level: str) -> bool:
    current = _TRACE_LEVEL_ORDER[_normalized_trace_level(config)]
    wanted = _TRACE_LEVEL_ORDER.get(str(level).strip().lower(), 0)
    return current >= wanted


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
    started_at = time.perf_counter()
    total_episodes = len(episodes)
    print(
        (
            "[search-teacher] start "
            f"episodes={total_episodes} "
            f"root_top_k={config.root_top_k} depth={config.depth} "
            f"max_states_per_episode={config.max_states_per_episode} "
            f"round_threshold={config.round_threshold} "
            f"legal_option_threshold={config.legal_option_threshold} "
            f"trace_level={_normalized_trace_level(config)}"
        ),
        flush=True,
    )
    annotated: list[EpisodeRecord] = []
    for episode_index, episode in enumerate(episodes, start=1):
        episode_started_at = time.perf_counter()
        if _trace_enabled(config, "state"):
            print(
                (
                    "[search-teacher-episode] "
                    f"{episode_index}/{total_episodes} "
                    f"matchup={episode.matchup} seed={episode.seed} steps={len(episode.steps)}"
                ),
                flush=True,
            )
        annotated_episode = _annotate_episode(
            episode,
            model=model,
            encoder=encoder,
            device=device,
            env_config=env_config,
            config=config,
            episode_index=episode_index,
            episode_count=total_episodes,
        )
        annotated.append(annotated_episode)
        target_count = sum(
            1
            for step in annotated_episode.steps
            if float(step.metadata.get("search_teacher_weight", 0.0)) > 0.0
        )
        if _trace_enabled(config, "state"):
            print(
                (
                    "[search-teacher-episode-done] "
                    f"{episode_index}/{total_episodes} "
                    f"targets={target_count} elapsed_s={time.perf_counter() - episode_started_at:.2f}"
                ),
                flush=True,
            )
    print(
        f"[search-teacher] done episodes={total_episodes} elapsed_s={time.perf_counter() - started_at:.2f}",
        flush=True,
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
    episode_index: int | None = None,
    episode_count: int | None = None,
) -> EpisodeRecord:
    matchup = _MATCHUP_BY_KEY.get(episode.matchup)
    if matchup is None:
        print(
            f"[search-teacher-episode-skip] matchup_unavailable matchup={episode.matchup}",
            flush=True,
        )
        return episode
    remaining_budget = int(config.max_states_per_episode)
    steps: list[TrajectoryStep] = []
    considered_states = 0
    annotated_states = 0
    total_steps = len(episode.steps)
    episode_label = (
        f"{episode_index}/{episode_count}"
        if episode_index is not None and episode_count is not None
        else "?"
    )
    for step_index, step in enumerate(episode.steps, start=1):
        target: SearchTeacherTarget | None = None
        if float(step.metadata.get("search_teacher_weight", 0.0)) > 0.0:
            steps.append(step)
            continue
        if remaining_budget > 0 and _is_search_state(step, config=config):
            considered_states += 1
            trace_prefix = (
                f"episode={episode_label} step={step_index}/{total_steps} "
                f"round={step.pre_state.round_number} legal={len(step.legal_low_level_codes)} "
                f"budget_left={remaining_budget}"
            )
            state_started_at = time.perf_counter()
            if _trace_enabled(config, "state"):
                print(f"[search-teacher-state] {trace_prefix}", flush=True)
            target = _search_teacher_target(
                step=step,
                matchup=matchup,
                model=model,
                encoder=encoder,
                device=device,
                env_config=env_config,
                config=config,
                seed=episode.seed,
                trace_prefix=(trace_prefix if _trace_enabled(config, "root") else None),
            )
            state_elapsed_ms = (time.perf_counter() - state_started_at) * 1000.0
            if target is not None:
                remaining_budget -= 1
                annotated_states += 1
            if _trace_enabled(config, "state"):
                print(
                    (
                        "[search-teacher-state-done] "
                        f"{trace_prefix} "
                        f"status={'hit' if target is not None else 'miss'} "
                        f"elapsed_ms={state_elapsed_ms:.1f} "
                        f"remaining_budget={remaining_budget}"
                    ),
                    flush=True,
                )
        if target is None:
            steps.append(step)
            continue
        metadata = dict(step.metadata)
        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_action_semantic_keys"] = list(target.action_semantic_keys)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
        steps.append(replace(step, metadata=metadata))
    print(
        (
            "[search-teacher-episode-summary] "
            f"{episode_label} matchup={episode.matchup} seed={episode.seed} "
            f"considered_states={considered_states} "
            f"annotated_states={annotated_states} remaining_budget={remaining_budget}"
        ),
        flush=True,
    )
    return replace(episode, steps=tuple(steps))


def _legal_action_kind_values_for_step(step: TrajectoryStep) -> tuple[str, ...]:
    if step.legal_low_level_specs:
        return tuple(spec.kind.value for spec in step.legal_low_level_specs)
    kinds: list[str] = []
    for action_code in step.legal_low_level_codes:
        kind = try_low_level_kind_for_code(int(action_code))
        if kind is not None:
            kinds.append(kind.value)
    return tuple(kinds)


def _is_search_state(step: TrajectoryStep, *, config: LookaheadSearchConfig) -> bool:
    if step.request_type.name != "ACTION":
        return False
    if step.player_view is None or step.full_state_json_before is None:
        return False
    if step.pre_state.round_number >= config.round_threshold:
        return True
    if len(step.legal_low_level_codes) >= config.legal_option_threshold:
        return True
    legal_kind_values = _legal_action_kind_values_for_step(step)
    if not legal_kind_values:
        return False
    productive_non_end = any(kind_value != "action_declare_end" for kind_value in legal_kind_values)
    has_end = any(kind_value == "action_declare_end" for kind_value in legal_kind_values)
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
    trace_prefix: str | None = None,
) -> SearchTeacherTarget | None:
    root_context = _rebuild_root_context(
        state_json=step.full_state_json_before,
        matchup=matchup,
        env_config=env_config,
        seed=seed,
    )
    if root_context is None or root_context.player_view is None or not root_context.legal_low_level_codes:
        if trace_prefix is not None and _trace_enabled(config, "root"):
            print(f"[search-teacher-root-skip] {trace_prefix} reason=rebuild_failed", flush=True)
        return None
    root_option_indices = _candidate_indices(
        context=root_context,
        model=model,
        encoder=encoder,
        device=device,
        top_k=config.root_top_k,
    )
    if not root_option_indices:
        if trace_prefix is not None and _trace_enabled(config, "root"):
            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_candidates", flush=True)
        return None
    if trace_prefix is not None and _trace_enabled(config, "root"):
        print(
            f"[search-teacher-root] {trace_prefix} root_candidates={len(root_option_indices)}",
            flush=True,
        )
    score_by_index: dict[int, float] = {}
    option_count = len(root_option_indices)
    for option_position, index in enumerate(root_option_indices, start=1):
        root_action_code = int(root_context.legal_low_level_codes[index])
        option_started_at = time.perf_counter()
        if trace_prefix is not None and _trace_enabled(config, "option"):
            print(
                (
                    "[search-teacher-root-option] "
                    f"{trace_prefix} option={option_position}/{option_count} "
                    f"action_code={root_action_code}"
                ),
                flush=True,
            )
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
        if trace_prefix is not None and _trace_enabled(config, "option"):
            print(
                (
                    "[search-teacher-root-option-done] "
                    f"{trace_prefix} option={option_position}/{option_count} "
                    f"score={score_by_index[index]:.4f} "
                    f"elapsed_ms={(time.perf_counter() - option_started_at) * 1000.0:.1f}"
                ),
                flush=True,
            )
    if not score_by_index:
        if trace_prefix is not None and _trace_enabled(config, "root"):
            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_scores", flush=True)
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
    if trace_prefix is not None and _trace_enabled(config, "root"):
        print(
            (
                "[search-teacher-root-done] "
                f"{trace_prefix} best_score={best_score:.4f} "
                f"teacher_weight={teacher_weight:.3f}"
            ),
            flush=True,
        )
    return SearchTeacherTarget(
        policy_target=policy_target,
        action_semantic_keys=_semantic_key_tokens_for_context(root_context),
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
