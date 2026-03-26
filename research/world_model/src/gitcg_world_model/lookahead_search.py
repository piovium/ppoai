from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    rollout_workers: int = 8
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


@dataclass
class _TeacherProfile:
    states: int = 0
    root_candidates: int = 0
    rollout_envs: int = 0
    rollout_steps: int = 0
    policy_batches: int = 0
    policy_contexts: int = 0
    oracle_batches: int = 0
    oracle_contexts: int = 0
    root_rebuild_s: float = 0.0
    root_policy_s: float = 0.0
    rollout_reset_s: float = 0.0
    rollout_step_s: float = 0.0
    rollout_policy_s: float = 0.0
    leaf_value_s: float = 0.0

    def add(self, other: "_TeacherProfile") -> None:
        self.states += int(other.states)
        self.root_candidates += int(other.root_candidates)
        self.rollout_envs += int(other.rollout_envs)
        self.rollout_steps += int(other.rollout_steps)
        self.policy_batches += int(other.policy_batches)
        self.policy_contexts += int(other.policy_contexts)
        self.oracle_batches += int(other.oracle_batches)
        self.oracle_contexts += int(other.oracle_contexts)
        self.root_rebuild_s += float(other.root_rebuild_s)
        self.root_policy_s += float(other.root_policy_s)
        self.rollout_reset_s += float(other.rollout_reset_s)
        self.rollout_step_s += float(other.rollout_step_s)
        self.rollout_policy_s += float(other.rollout_policy_s)
        self.leaf_value_s += float(other.leaf_value_s)


@dataclass
class _RootOptionRollout:
    option_index: int
    env: GitcgDecisionEnv
    current_context: DecisionContext | None
    score: float | None = None


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
            f"rollout_workers={_rollout_worker_cap(config=config, candidate_count=config.root_top_k)} "
            f"round_threshold={config.round_threshold} "
            f"legal_option_threshold={config.legal_option_threshold} "
            f"trace_level={_normalized_trace_level(config)}"
        ),
        flush=True,
    )
    annotated: list[EpisodeRecord] = []
    profile = _TeacherProfile()
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
            profile=profile,
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
    _print_search_teacher_profile(
        profile=profile,
        elapsed_s=time.perf_counter() - started_at,
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
    profile: _TeacherProfile | None = None,
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
                profile=profile,
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
    profile: _TeacherProfile | None = None,
) -> SearchTeacherTarget | None:
    if profile is not None:
        profile.states += 1
    root_rebuild_started_at = time.perf_counter()
    root_context = _rebuild_root_context(
        state_json=step.full_state_json_before,
        matchup=matchup,
        env_config=env_config,
        seed=seed,
    )
    if profile is not None:
        profile.root_rebuild_s += time.perf_counter() - root_rebuild_started_at
    if root_context is None or root_context.player_view is None or not root_context.legal_low_level_codes:
        if trace_prefix is not None and _trace_enabled(config, "root"):
            print(f"[search-teacher-root-skip] {trace_prefix} reason=rebuild_failed", flush=True)
        return None
    root_policy_started_at = time.perf_counter()
    base_log_probs = _policy_log_probs(
        context=root_context,
        model=model,
        encoder=encoder,
        device=device,
    )
    if profile is not None:
        profile.root_policy_s += time.perf_counter() - root_policy_started_at
    root_option_indices = _top_candidate_indices(
        base_log_probs,
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
    if profile is not None:
        profile.root_candidates += len(root_option_indices)
    score_by_index: dict[int, float] = {}
    option_count = len(root_option_indices)
    if _trace_enabled(config, "option"):
        for option_position, index in enumerate(root_option_indices, start=1):
            root_action_code = int(root_context.legal_low_level_codes[index])
            option_started_at = time.perf_counter()
            if trace_prefix is not None:
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
                profile=profile,
            )
            if trace_prefix is not None:
                print(
                    (
                        "[search-teacher-root-option-done] "
                        f"{trace_prefix} option={option_position}/{option_count} "
                        f"score={score_by_index[index]:.4f} "
                        f"elapsed_ms={(time.perf_counter() - option_started_at) * 1000.0:.1f}"
                    ),
                    flush=True,
                )
    else:
        score_by_index = _evaluate_root_options(
            step=step,
            matchup=matchup,
            env_config=env_config,
            model=model,
            encoder=encoder,
            device=device,
            root_action_codes=tuple(
                (int(index), int(root_context.legal_low_level_codes[index]))
                for index in root_option_indices
            ),
            depth=config.depth,
            seed=seed,
            profile=profile,
            rollout_workers=_rollout_worker_cap(config=config, candidate_count=len(root_option_indices)),
        )
    if not score_by_index:
        if trace_prefix is not None and _trace_enabled(config, "root"):
            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_scores", flush=True)
        return None
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
    profile: _TeacherProfile | None = None,
) -> float:
    env = GitcgDecisionEnv(
        _teacher_env_config(env_config),
        matchup,
        enable_result_based_action_relabel=False,
    )
    try:
        reset_started_at = time.perf_counter()
        context = env.reset(seed=seed, state_json=step.full_state_json_before)
        if profile is not None:
            profile.rollout_envs += 1
            profile.rollout_reset_s += time.perf_counter() - reset_started_at
        option_index = match_low_level_code_index(context, action_code)
        if option_index is None:
            return -1.0
        step_started_at = time.perf_counter()
        next_context, transition, done = env.step(ActionChoice(action_code=context.legal_low_level_codes[option_index]))
        if profile is not None:
            profile.rollout_steps += 1
            profile.rollout_step_s += time.perf_counter() - step_started_at
        root_player = int(step.acting_player)
        if done:
            return _terminal_value(transition.post_state.winner, root_player=root_player)
        current_context = next_context
        remaining_depth = max(0, depth - 1)
        while remaining_depth > 0 and current_context is not None and not current_context.terminal:
            policy_started_at = time.perf_counter()
            greedy_index = _greedy_action_index(
                context=current_context,
                model=model,
                encoder=encoder,
                device=device,
            )
            if profile is not None:
                profile.policy_batches += 1
                profile.policy_contexts += 1
                profile.rollout_policy_s += time.perf_counter() - policy_started_at
            if greedy_index is None:
                break
            step_started_at = time.perf_counter()
            current_context, transition, done = env.step(
                ActionChoice(action_code=current_context.legal_low_level_codes[greedy_index])
            )
            if profile is not None:
                profile.rollout_steps += 1
                profile.rollout_step_s += time.perf_counter() - step_started_at
            if done:
                return _terminal_value(transition.post_state.winner, root_player=root_player)
            remaining_depth -= 1
        if current_context is None:
            return 0.0
        value_started_at = time.perf_counter()
        value = _oracle_values_for_contexts(
            contexts=(current_context,),
            model=model,
            encoder=encoder,
            device=device,
        )[0]
        if profile is not None:
            profile.oracle_batches += 1
            profile.oracle_contexts += 1
            profile.leaf_value_s += time.perf_counter() - value_started_at
        return value
    finally:
        env.close()


def _evaluate_root_options(
    *,
    step: TrajectoryStep,
    matchup: Matchup,
    env_config: EnvConfig,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    root_action_codes: Sequence[tuple[int, int]],
    depth: int,
    seed: int | None,
    profile: _TeacherProfile | None = None,
    rollout_workers: int = 1,
) -> dict[int, float]:
    score_by_index: dict[int, float] = {}
    rollouts: list[_RootOptionRollout] = []
    root_player = int(step.acting_player)
    resolved_rollout_workers = max(1, min(int(rollout_workers), max(1, len(root_action_codes))))

    def _initialize_rollout(option_payload: tuple[int, int]) -> tuple[_RootOptionRollout, float, float, float | None]:
        option_index, action_code = option_payload
        env = GitcgDecisionEnv(
            _teacher_env_config(env_config),
            matchup,
            enable_result_based_action_relabel=False,
        )
        reset_started_at = time.perf_counter()
        context = env.reset(seed=seed, state_json=step.full_state_json_before)
        reset_elapsed = time.perf_counter() - reset_started_at
        matched_index = match_low_level_code_index(context, int(action_code))
        if matched_index is None:
            return _RootOptionRollout(option_index=int(option_index), env=env, current_context=None), reset_elapsed, 0.0, -1.0
        step_started_at = time.perf_counter()
        next_context, transition, done = env.step(
            ActionChoice(action_code=context.legal_low_level_codes[matched_index])
        )
        step_elapsed = time.perf_counter() - step_started_at
        if done:
            terminal_score = _terminal_value(transition.post_state.winner, root_player=root_player)
            return _RootOptionRollout(option_index=int(option_index), env=env, current_context=None), reset_elapsed, step_elapsed, terminal_score
        return _RootOptionRollout(option_index=int(option_index), env=env, current_context=next_context), reset_elapsed, step_elapsed, None

    def _apply_rollout_step(
        rollout_payload: tuple[_RootOptionRollout, int | None],
    ) -> tuple[_RootOptionRollout, float, float | None]:
        rollout, greedy_index = rollout_payload
        current_context = rollout.current_context
        if current_context is None or greedy_index is None:
            return rollout, 0.0, None
        step_started_at = time.perf_counter()
        next_context, transition, done = rollout.env.step(
            ActionChoice(action_code=current_context.legal_low_level_codes[greedy_index])
        )
        step_elapsed = time.perf_counter() - step_started_at
        if done:
            return rollout, step_elapsed, _terminal_value(transition.post_state.winner, root_player=root_player)
        rollout.current_context = next_context
        if next_context is None:
            return rollout, step_elapsed, 0.0
        return rollout, step_elapsed, None

    try:
        if resolved_rollout_workers > 1 and len(root_action_codes) > 1:
            with ThreadPoolExecutor(max_workers=resolved_rollout_workers) as executor:
                init_results = list(executor.map(_initialize_rollout, root_action_codes))
                for rollout, reset_elapsed, step_elapsed, terminal_score in init_results:
                    rollouts.append(rollout)
                    if profile is not None:
                        profile.rollout_envs += 1
                        profile.rollout_reset_s += reset_elapsed
                    if step_elapsed > 0.0 and profile is not None:
                        profile.rollout_steps += 1
                        profile.rollout_step_s += step_elapsed
                    if terminal_score is not None:
                        score_by_index[rollout.option_index] = float(terminal_score)
                active_rollouts = [
                    rollout
                    for rollout in rollouts
                    if rollout.option_index not in score_by_index
                    and rollout.current_context is not None
                    and not rollout.current_context.terminal
                ]
                remaining_depth = max(0, depth - 1)
                while remaining_depth > 0 and active_rollouts:
                    policy_started_at = time.perf_counter()
                    greedy_indices = _greedy_action_indices(
                        contexts=tuple(
                            rollout.current_context for rollout in active_rollouts if rollout.current_context is not None
                        ),
                        model=model,
                        encoder=encoder,
                        device=device,
                    )
                    if profile is not None:
                        profile.policy_batches += 1
                        profile.policy_contexts += len(greedy_indices)
                        profile.rollout_policy_s += time.perf_counter() - policy_started_at
                    step_results = list(executor.map(_apply_rollout_step, zip(active_rollouts, greedy_indices, strict=False)))
                    next_active_rollouts: list[_RootOptionRollout] = []
                    for rollout, step_elapsed, terminal_score in step_results:
                        if step_elapsed > 0.0 and profile is not None:
                            profile.rollout_steps += 1
                            profile.rollout_step_s += step_elapsed
                        if terminal_score is not None:
                            score_by_index[rollout.option_index] = float(terminal_score)
                            rollout.current_context = None
                            continue
                        if rollout.current_context is not None and not rollout.current_context.terminal:
                            next_active_rollouts.append(rollout)
                    active_rollouts = next_active_rollouts
                    remaining_depth -= 1
        else:
            for option_index, action_code in root_action_codes:
                rollout, reset_elapsed, step_elapsed, terminal_score = _initialize_rollout((option_index, action_code))
                rollouts.append(rollout)
                if profile is not None:
                    profile.rollout_envs += 1
                    profile.rollout_reset_s += reset_elapsed
                if step_elapsed > 0.0 and profile is not None:
                    profile.rollout_steps += 1
                    profile.rollout_step_s += step_elapsed
                if terminal_score is not None:
                    score_by_index[int(option_index)] = float(terminal_score)
                    continue

            active_rollouts = [
                rollout
                for rollout in rollouts
                if rollout.option_index not in score_by_index
                and rollout.current_context is not None
                and not rollout.current_context.terminal
            ]
            remaining_depth = max(0, depth - 1)
            while remaining_depth > 0 and active_rollouts:
                policy_started_at = time.perf_counter()
                greedy_indices = _greedy_action_indices(
                    contexts=tuple(
                        rollout.current_context for rollout in active_rollouts if rollout.current_context is not None
                    ),
                    model=model,
                    encoder=encoder,
                    device=device,
                )
                if profile is not None:
                    profile.policy_batches += 1
                    profile.policy_contexts += len(greedy_indices)
                    profile.rollout_policy_s += time.perf_counter() - policy_started_at
                next_active_rollouts: list[_RootOptionRollout] = []
                for rollout, greedy_index in zip(active_rollouts, greedy_indices, strict=False):
                    rollout, step_elapsed, terminal_score = _apply_rollout_step((rollout, greedy_index))
                    if step_elapsed > 0.0 and profile is not None:
                        profile.rollout_steps += 1
                        profile.rollout_step_s += step_elapsed
                    if terminal_score is not None:
                        score_by_index[rollout.option_index] = float(terminal_score)
                        rollout.current_context = None
                        continue
                    if rollout.current_context is not None and not rollout.current_context.terminal:
                        next_active_rollouts.append(rollout)
                active_rollouts = next_active_rollouts
                remaining_depth -= 1

        leaf_rollouts = [
            rollout
            for rollout in rollouts
            if rollout.option_index not in score_by_index
            and rollout.current_context is not None
        ]
        if leaf_rollouts:
            value_started_at = time.perf_counter()
            leaf_values = _oracle_values_for_contexts(
                contexts=tuple(rollout.current_context for rollout in leaf_rollouts if rollout.current_context is not None),
                model=model,
                encoder=encoder,
                device=device,
            )
            if profile is not None:
                profile.oracle_batches += 1
                profile.oracle_contexts += len(leaf_values)
                profile.leaf_value_s += time.perf_counter() - value_started_at
            for rollout, leaf_value in zip(leaf_rollouts, leaf_values, strict=False):
                score_by_index[rollout.option_index] = float(leaf_value)

        for rollout in rollouts:
            if rollout.option_index not in score_by_index:
                score_by_index[rollout.option_index] = 0.0
        return score_by_index
    finally:
        for rollout in rollouts:
            rollout.env.close()


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
    return _top_candidate_indices(log_probs, top_k=top_k)


def _top_candidate_indices(log_probs: Sequence[float], *, top_k: int) -> tuple[int, ...]:
    if not log_probs:
        return ()
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
    return _policy_log_probs_batch(
        contexts=(context,),
        model=model,
        encoder=encoder,
        device=device,
    )[0]


def _policy_log_probs_batch(
    *,
    contexts: Sequence[DecisionContext],
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> tuple[tuple[float, ...], ...]:
    if not contexts:
        return ()
    results: list[tuple[float, ...]] = [() for _ in contexts]
    encoded_contexts: list[tuple[int, object]] = []
    for index, context in enumerate(contexts):
        if context.player_view is None or not context.legal_low_level_codes:
            continue
        encoded_contexts.append(
            (index, encoder.encode_context(context, history=(), include_training_targets=False))
        )
    if not encoded_contexts:
        return tuple(results)

    max_tokens = max(len(encoded.token_features) for _, encoded in encoded_contexts)
    max_options = max(len(encoded.option_features) for _, encoded in encoded_contexts)
    token_dim = len(encoded_contexts[0][1].token_features[0])
    option_dim = len(encoded_contexts[0][1].option_features[0])
    batch_size = len(encoded_contexts)
    token_features = torch.zeros((batch_size, max_tokens, token_dim), dtype=torch.float32, device=device)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    opponent_token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    option_features = torch.zeros((batch_size, max_options, option_dim), dtype=torch.float32, device=device)
    option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool, device=device)
    opponent_tag_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)
    opponent_entity_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)
    opponent_deck_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for batch_index, (_, encoded) in enumerate(encoded_contexts):
        token_count = len(encoded.token_features)
        option_count = len(encoded.option_features)
        token_features[batch_index, :token_count] = torch.as_tensor(
            encoded.token_features,
            dtype=torch.float32,
            device=device,
        )
        token_mask[batch_index, :token_count] = torch.as_tensor(encoded.token_mask, dtype=torch.bool, device=device)
        opponent_token_mask[batch_index, :token_count] = torch.as_tensor(
            encoded.opponent_token_mask,
            dtype=torch.bool,
            device=device,
        )
        option_features[batch_index, :option_count] = torch.as_tensor(
            encoded.option_features,
            dtype=torch.float32,
            device=device,
        )
        option_mask[batch_index, :option_count] = torch.as_tensor(
            encoded.option_mask,
            dtype=torch.bool,
            device=device,
        )
        opponent_tag_ids[batch_index] = int(encoded.opponent_tag_id)
        opponent_entity_ids[batch_index] = int(encoded.opponent_entity_id)
        opponent_deck_ids[batch_index] = int(encoded.opponent_deck_id)

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
        log_probs = masked_log_softmax(output.policy_logits, option_mask).detach().cpu()

    for batch_index, (context_index, encoded) in enumerate(encoded_contexts):
        option_count = len(encoded.option_features)
        results[context_index] = tuple(float(value) for value in log_probs[batch_index, :option_count].tolist())
    return tuple(results)


def _greedy_action_index(
    *,
    context: DecisionContext,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> int | None:
    return _greedy_action_indices(
        contexts=(context,),
        model=model,
        encoder=encoder,
        device=device,
    )[0]


def _greedy_action_indices(
    *,
    contexts: Sequence[DecisionContext],
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> tuple[int | None, ...]:
    log_probs_batch = _policy_log_probs_batch(
        contexts=contexts,
        model=model,
        encoder=encoder,
        device=device,
    )
    indices: list[int | None] = []
    for log_probs in log_probs_batch:
        if not log_probs:
            indices.append(None)
            continue
        indices.append(max(range(len(log_probs)), key=lambda index: log_probs[index]))
    return tuple(indices)


def _oracle_values_for_contexts(
    *,
    contexts: Sequence[DecisionContext],
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
) -> tuple[float, ...]:
    if not contexts:
        return ()
    privileged_state = torch.as_tensor(
        [encoder.event.encode_privileged_state(context.full_state) for context in contexts],
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        values = model.oracle_value(privileged_state).detach().cpu().tolist()
    return tuple(float(value) for value in values)


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
        record_full_state_json=False,
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


def _print_search_teacher_profile(*, profile: _TeacherProfile, elapsed_s: float) -> None:
    if profile.states <= 0:
        return
    avg_candidates = float(profile.root_candidates) / float(profile.states)
    avg_policy_batch = (
        float(profile.policy_contexts) / float(profile.policy_batches)
        if profile.policy_batches > 0
        else 0.0
    )
    avg_oracle_batch = (
        float(profile.oracle_contexts) / float(profile.oracle_batches)
        if profile.oracle_batches > 0
        else 0.0
    )
    print(
        (
            "[search-teacher-profile] "
            f"wall_elapsed_s={elapsed_s:.2f} "
            f"states={profile.states} "
            f"avg_candidates={avg_candidates:.2f} "
            f"rollout_envs={profile.rollout_envs} "
            f"rollout_steps={profile.rollout_steps} "
            f"policy_batches={profile.policy_batches} "
            f"avg_policy_batch={avg_policy_batch:.2f} "
            f"oracle_batches={profile.oracle_batches} "
            f"avg_oracle_batch={avg_oracle_batch:.2f} "
            f"work_s=root_rebuild:{profile.root_rebuild_s:.2f} "
            f"root_policy:{profile.root_policy_s:.2f} "
            f"rollout_reset:{profile.rollout_reset_s:.2f} "
            f"rollout_step:{profile.rollout_step_s:.2f} "
            f"rollout_policy:{profile.rollout_policy_s:.2f} "
            f"leaf_value:{profile.leaf_value_s:.2f}"
        ),
        flush=True,
    )


def _rollout_worker_cap(*, config: LookaheadSearchConfig, candidate_count: int) -> int:
    raw_override = os.environ.get("SEARCH_TEACHER_ROLLOUT_WORKERS")
    if raw_override is not None:
        try:
            requested = max(1, int(raw_override.strip()))
        except ValueError:
            requested = max(1, int(config.rollout_workers))
    else:
        requested = max(1, int(config.rollout_workers))
    cpu_limit = max(1, int(os.cpu_count() or requested))
    return max(1, min(int(candidate_count), requested, cpu_limit))
