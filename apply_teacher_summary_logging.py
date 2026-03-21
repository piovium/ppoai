
from pathlib import Path

TARGET = Path("research/world_model/src/gitcg_world_model/lookahead_search.py")

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing snippet for {label}")
    return text.replace(old, new, 1)

def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "import math\nimport time\n",
        "import math\nimport os\nimport time\n",
        "add os import",
    )

    text = replace_once(
        text,
        '    max_states_per_episode: int = 24\n',
        '    max_states_per_episode: int = 24\n'
        '    # episode: only per-episode summary logs (default)\n'
        '    # state: add per-state start/done logs\n'
        '    # root: add per-root summary logs\n'
        '    # option: full detailed per-option logs\n'
        '    trace_level: str = "episode"\n',
        "add trace_level config",
    )

    text = replace_once(
        text,
        '_MATCHUP_BY_KEY = {matchup.key: matchup for matchup in SMALL_DECK_MATCHUPS}\n\n\n',
        '_MATCHUP_BY_KEY = {matchup.key: matchup for matchup in SMALL_DECK_MATCHUPS}\n\n\n'
        '_TRACE_LEVEL_ORDER = {"episode": 0, "state": 1, "root": 2, "option": 3}\n\n\n'
        'def _normalized_trace_level(config: LookaheadSearchConfig) -> str:\n'
        '    value = os.environ.get("SEARCH_TEACHER_TRACE_LEVEL", config.trace_level)\n'
        '    level = str(value).strip().lower()\n'
        '    return level if level in _TRACE_LEVEL_ORDER else "episode"\n\n\n'
        'def _trace_enabled(config: LookaheadSearchConfig, level: str) -> bool:\n'
        '    current = _TRACE_LEVEL_ORDER[_normalized_trace_level(config)]\n'
        '    wanted = _TRACE_LEVEL_ORDER.get(str(level).strip().lower(), 0)\n'
        '    return current >= wanted\n\n\n',
        "add trace helpers",
    )

    text = replace_once(
        text,
        '            f"max_states_per_episode={config.max_states_per_episode} "\n'
        '            f"round_threshold={config.round_threshold} "\n'
        '            f"legal_option_threshold={config.legal_option_threshold}"\n',
        '            f"max_states_per_episode={config.max_states_per_episode} "\n'
        '            f"round_threshold={config.round_threshold} "\n'
        '            f"legal_option_threshold={config.legal_option_threshold} "\n'
        '            f"trace_level={_normalized_trace_level(config)}"\n',
        "add trace_level to start log",
    )

    text = replace_once(
        text,
        '        print(\n'
        '            (\n'
        '                "[search-teacher-episode] "\n'
        '                f"{episode_index}/{total_episodes} "\n'
        '                f"matchup={episode.matchup} seed={episode.seed} steps={len(episode.steps)}"\n'
        '            ),\n'
        '            flush=True,\n'
        '        )\n',
        '        if _trace_enabled(config, "state"):\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-episode] "\n'
        '                    f"{episode_index}/{total_episodes} "\n'
        '                    f"matchup={episode.matchup} seed={episode.seed} steps={len(episode.steps)}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        "guard episode-start log",
    )

    text = replace_once(
        text,
        '        print(\n'
        '            (\n'
        '                "[search-teacher-episode-done] "\n'
        '                f"{episode_index}/{total_episodes} "\n'
        '                f"targets={target_count} elapsed_s={time.perf_counter() - episode_started_at:.2f}"\n'
        '            ),\n'
        '            flush=True,\n'
        '        )\n',
        '        if _trace_enabled(config, "state"):\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-episode-done] "\n'
        '                    f"{episode_index}/{total_episodes} "\n'
        '                    f"targets={target_count} elapsed_s={time.perf_counter() - episode_started_at:.2f}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        "guard episode-done log",
    )

    text = replace_once(
        text,
        '            trace_prefix = (\n'
        '                f"episode={episode_label} step={step_index}/{total_steps} "\n'
        '                f"round={step.pre_state.round_number} legal={len(step.legal_low_level_codes)} "\n'
        '                f"budget_left={remaining_budget}"\n'
        '            )\n'
        '            state_started_at = time.perf_counter()\n'
        '            print(f"[search-teacher-state] {trace_prefix}", flush=True)\n'
        '            target = _search_teacher_target(\n'
        '                step=step,\n'
        '                matchup=matchup,\n'
        '                model=model,\n'
        '                encoder=encoder,\n'
        '                device=device,\n'
        '                env_config=env_config,\n'
        '                config=config,\n'
        '                seed=episode.seed,\n'
        '                trace_prefix=trace_prefix,\n'
        '            )\n'
        '            state_elapsed_ms = (time.perf_counter() - state_started_at) * 1000.0\n'
        '            if target is not None:\n'
        '                remaining_budget -= 1\n'
        '                annotated_states += 1\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-state-done] "\n'
        '                    f"{trace_prefix} "\n'
        '                    f"status={\'hit\' if target is not None else \'miss\'} "\n'
        '                    f"elapsed_ms={state_elapsed_ms:.1f} "\n'
        '                    f"remaining_budget={remaining_budget}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        '            trace_prefix = (\n'
        '                f"episode={episode_label} step={step_index}/{total_steps} "\n'
        '                f"round={step.pre_state.round_number} legal={len(step.legal_low_level_codes)} "\n'
        '                f"budget_left={remaining_budget}"\n'
        '            )\n'
        '            state_started_at = time.perf_counter()\n'
        '            if _trace_enabled(config, "state"):\n'
        '                print(f"[search-teacher-state] {trace_prefix}", flush=True)\n'
        '            target = _search_teacher_target(\n'
        '                step=step,\n'
        '                matchup=matchup,\n'
        '                model=model,\n'
        '                encoder=encoder,\n'
        '                device=device,\n'
        '                env_config=env_config,\n'
        '                config=config,\n'
        '                seed=episode.seed,\n'
        '                trace_prefix=(trace_prefix if _trace_enabled(config, "root") else None),\n'
        '            )\n'
        '            state_elapsed_ms = (time.perf_counter() - state_started_at) * 1000.0\n'
        '            if target is not None:\n'
        '                remaining_budget -= 1\n'
        '                annotated_states += 1\n'
        '            if _trace_enabled(config, "state"):\n'
        '                print(\n'
        '                    (\n'
        '                        "[search-teacher-state-done] "\n'
        '                        f"{trace_prefix} "\n'
        '                        f"status={\'hit\' if target is not None else \'miss\'} "\n'
        '                        f"elapsed_ms={state_elapsed_ms:.1f} "\n'
        '                        f"remaining_budget={remaining_budget}"\n'
        '                    ),\n'
        '                    flush=True,\n'
        '                )\n',
        "guard state logs and root trace prefix",
    )

    text = replace_once(
        text,
        '    print(\n'
        '        (\n'
        '            "[search-teacher-episode-summary] "\n'
        '            f"{episode_label} considered_states={considered_states} "\n'
        '            f"annotated_states={annotated_states} remaining_budget={remaining_budget}"\n'
        '        ),\n'
        '        flush=True,\n'
        '    )\n',
        '    print(\n'
        '        (\n'
        '            "[search-teacher-episode-summary] "\n'
        '            f"{episode_label} matchup={episode.matchup} seed={episode.seed} "\n'
        '            f"considered_states={considered_states} "\n'
        '            f"annotated_states={annotated_states} remaining_budget={remaining_budget}"\n'
        '        ),\n'
        '        flush=True,\n'
        '    )\n',
        "improve episode summary",
    )

    text = replace_once(
        text,
        '    if root_context is None or root_context.player_view is None or not root_context.legal_low_level_codes:\n'
        '        if trace_prefix is not None:\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=rebuild_failed", flush=True)\n'
        '        return None\n',
        '    if root_context is None or root_context.player_view is None or not root_context.legal_low_level_codes:\n'
        '        if trace_prefix is not None and _trace_enabled(config, "root"):\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=rebuild_failed", flush=True)\n'
        '        return None\n',
        "guard root-skip rebuild_failed",
    )

    text = replace_once(
        text,
        '    if not root_option_indices:\n'
        '        if trace_prefix is not None:\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_candidates", flush=True)\n'
        '        return None\n',
        '    if not root_option_indices:\n'
        '        if trace_prefix is not None and _trace_enabled(config, "root"):\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_candidates", flush=True)\n'
        '        return None\n',
        "guard root-skip no_candidates",
    )

    text = replace_once(
        text,
        '    if trace_prefix is not None:\n'
        '        print(\n'
        '            f"[search-teacher-root] {trace_prefix} root_candidates={len(root_option_indices)}",\n'
        '            flush=True,\n'
        '        )\n',
        '    if trace_prefix is not None and _trace_enabled(config, "root"):\n'
        '        print(\n'
        '            f"[search-teacher-root] {trace_prefix} root_candidates={len(root_option_indices)}",\n'
        '            flush=True,\n'
        '        )\n',
        "guard root summary",
    )

    text = replace_once(
        text,
        '        if trace_prefix is not None:\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-root-option] "\n'
        '                    f"{trace_prefix} option={option_position}/{option_count} "\n'
        '                    f"action_code={root_action_code}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        '        if trace_prefix is not None and _trace_enabled(config, "option"):\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-root-option] "\n'
        '                    f"{trace_prefix} option={option_position}/{option_count} "\n'
        '                    f"action_code={root_action_code}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        "guard option-start log",
    )

    text = replace_once(
        text,
        '        if trace_prefix is not None:\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-root-option-done] "\n'
        '                    f"{trace_prefix} option={option_position}/{option_count} "\n'
        '                    f"score={score_by_index[index]:.4f} "\n'
        '                    f"elapsed_ms={(time.perf_counter() - option_started_at) * 1000.0:.1f}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        '        if trace_prefix is not None and _trace_enabled(config, "option"):\n'
        '            print(\n'
        '                (\n'
        '                    "[search-teacher-root-option-done] "\n'
        '                    f"{trace_prefix} option={option_position}/{option_count} "\n'
        '                    f"score={score_by_index[index]:.4f} "\n'
        '                    f"elapsed_ms={(time.perf_counter() - option_started_at) * 1000.0:.1f}"\n'
        '                ),\n'
        '                flush=True,\n'
        '            )\n',
        "guard option-done log",
    )

    text = replace_once(
        text,
        '    if not score_by_index:\n'
        '        if trace_prefix is not None:\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_scores", flush=True)\n'
        '        return None\n',
        '    if not score_by_index:\n'
        '        if trace_prefix is not None and _trace_enabled(config, "root"):\n'
        '            print(f"[search-teacher-root-skip] {trace_prefix} reason=no_scores", flush=True)\n'
        '        return None\n',
        "guard root-skip no_scores",
    )

    text = replace_once(
        text,
        '    if trace_prefix is not None:\n'
        '        print(\n'
        '            (\n'
        '                "[search-teacher-root-done] "\n'
        '                f"{trace_prefix} best_score={best_score:.4f} "\n'
        '                f"teacher_weight={teacher_weight:.3f}"\n'
        '            ),\n'
        '            flush=True,\n'
        '        )\n',
        '    if trace_prefix is not None and _trace_enabled(config, "root"):\n'
        '        print(\n'
        '            (\n'
        '                "[search-teacher-root-done] "\n'
        '                f"{trace_prefix} best_score={best_score:.4f} "\n'
        '                f"teacher_weight={teacher_weight:.3f}"\n'
        '            ),\n'
        '            flush=True,\n'
        '        )\n',
        "guard root-done log",
    )

    backup = TARGET.with_suffix(".py.bak_teacher_summary")
    backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"updated {TARGET}")
    print(f"backup  {backup}")
    print("default: only episode summaries")
    print("override detail with SEARCH_TEACHER_TRACE_LEVEL=state|root|option")

if __name__ == "__main__":
    main()
