
from pathlib import Path

TARGET = Path("research/world_model/src/gitcg_world_model/lookahead_search.py")
BACKUP_SUFFIX = ".bak_teacher_logs"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"未找到需要替换的片段: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"找不到文件: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "import math\nfrom dataclasses import dataclass, replace\n",
        "import math\nimport time\nfrom dataclasses import dataclass, replace\n",
        "import time",
    )

    old_annotate_episodes = """def annotate_episodes_with_search_targets(
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
"""

    new_annotate_episodes = """def annotate_episodes_with_search_targets(
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
            f"legal_option_threshold={config.legal_option_threshold}"
        ),
        flush=True,
    )
    annotated: list[EpisodeRecord] = []
    for episode_index, episode in enumerate(episodes, start=1):
        episode_started_at = time.perf_counter()
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
"""

    text = replace_once(text, old_annotate_episodes, new_annotate_episodes, "annotate_episodes_with_search_targets")

    old_annotate_episode = """def _annotate_episode(
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
"""

    new_annotate_episode = """def _annotate_episode(
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
                trace_prefix=trace_prefix,
            )
            state_elapsed_ms = (time.perf_counter() - state_started_at) * 1000.0
            if target is not None:
                remaining_budget -= 1
                annotated_states += 1
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
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
        steps.append(replace(step, metadata=metadata))
    print(
        (
            "[search-teacher-episode-summary] "
            f"{episode_label} considered_states={considered_states} "
            f"annotated_states={annotated_states} remaining_budget={remaining_budget}"
        ),
        flush=True,
    )
    return replace(episode, steps=tuple(steps))
"""

    text = replace_once(text, old_annotate_episode, new_annotate_episode, "_annotate_episode")

    old_search_teacher_target = """def _search_teacher_target(
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
"""

    new_search_teacher_target = """def _search_teacher_target(
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
        if trace_prefix is not None:
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
        if trace_prefix is not None:
            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_candidates", flush=True)
        return None
    if trace_prefix is not None:
        print(
            f"[search-teacher-root] {trace_prefix} root_candidates={len(root_option_indices)}",
            flush=True,
        )
    score_by_index: dict[int, float] = {}
    option_count = len(root_option_indices)
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
    if not score_by_index:
        if trace_prefix is not None:
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
    if trace_prefix is not None:
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
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

    text = replace_once(text, old_search_teacher_target, new_search_teacher_target, "_search_teacher_target")

    backup = TARGET.with_name(TARGET.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"已修改: {TARGET}")
    print(f"备份文件: {backup}")


if __name__ == "__main__":
    main()
