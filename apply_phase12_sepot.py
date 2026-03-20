from __future__ import annotations

from pathlib import Path
import re
import sys

TARGET = Path("research/world_model/src/gitcg_world_model/sepot_search.py")
BACKUP_SUFFIX = ".bak_phase12"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing snippet for {label}")
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, repl: str, label: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"expected 1 regex replacement for {label}, got {count}")
    return new_text


def main() -> None:
    if not TARGET.exists():
        print(f"Target file not found: {TARGET}")
        sys.exit(1)

    original = TARGET.read_text(encoding="utf-8")
    text = original

    text = replace_once(
        text,
        """
        if not _should_search_for_policy(context=context, policy_log_probs=policy_log_probs):
            return None
""",
        """
        # Phase 1: policy is a prior for ordering / scoring, not a veto over
        # whether search may run at all. Triggering is handled by tactical /
        # structural conditions above.
""",
        "remove policy veto gate",
    )

    text = replace_once(
        text,
        """
        if reconstructor.ensure_template(acting_player=root_player) is None:
            return _fallback_decision(
                reason=\"template_unavailable\",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
""",
        """
        # Phase 2: do not require a synthetic ACTION template before root search.
        # Root branches should start from the live root state's full_state_json
        # whenever available. Posterior updates may still fall back to the
        # reconstructor until later phases remove it completely.
""",
        "remove eager template requirement",
    )

    text = replace_once(
        text,
        """        root_indices = _select_root_candidate_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
""",
        """        root_indices = _prioritized_root_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
""",
        "switch to prioritized root indices",
    )

    text = replace_once(
        text,
        """            return _fallback_decision(
                reason=\"no_root_candidates\",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
""",
        """            return _fallback_decision(
                reason=\"no_root_candidates\",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
                root_candidate_count=0,
            )
""",
        "no_root_candidates fallback count",
    )

    text = replace_once(
        text,
        """            return _fallback_decision(
                reason=(\"timeout\" if timed_out else \"reconstruction_failed\"),
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
""",
        """            return _fallback_decision(
                reason=(\"timeout\" if timed_out else \"reconstruction_failed\"),
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
                root_candidate_count=len(root_indices),
            )
""",
        "root_scores empty fallback count",
    )

    text = replace_once(
        text,
        """            sampled_self_hypothesis = root_public_belief.self_range.hypotheses[0]
            env, current_context = reconstructor.reconstruct(
                public_belief=root_public_belief,
                self_hypothesis=sampled_self_hypothesis,
                opponent_hypothesis=sampled_opponent_hypothesis,
            )
            if env is None or current_context is None:
                return None
""",
        """            sampled_self_hypothesis = root_public_belief.self_range.hypotheses[0]
            env, current_context = _reset_branch_env_from_state_json(
                state_json=root_context.full_state_json,
                env_config=reconstructor.env_config,
                matchup=reconstructor.matchup,
                seed=reconstructor.config.template_seed,
            )
            if env is None or current_context is None:
                env, current_context = reconstructor.reconstruct(
                    public_belief=root_public_belief,
                    self_hypothesis=sampled_self_hypothesis,
                    opponent_hypothesis=sampled_opponent_hypothesis,
                )
            if env is None or current_context is None:
                return None
""",
        "root branch direct reset from live state",
    )

    prioritized_pattern = r"""
def _prioritized_root_indices\(
    \*,
    context: DecisionContext,
    policy_log_probs: Sequence\[float\],
    budget: SePotPhaseBudget,
\) -> tuple\[int, \.\.\.\]:
    if not context\.legal_low_level_codes:
        return \(\)
    tactical_reason = _tactical_trigger_reason\(context\)
    margin = _top_policy_margin\(policy_log_probs\)
    legal_count = len\(context\.legal_low_level_codes\)
    limit = max\(1, min\(len\(context\.legal_low_level_codes\), int\(budget\.root_top_k\)\)\)
    # Policy provides a prior only for non-tactical states\. Tactical states keep
    # their wider budget so search can correct confident but tactically-wrong policies\.
    if tactical_reason is None:
        if legal_count <= 4 and margin >= 1\.50:
            limit = 1
        elif margin >= 1\.00:
            limit = min\(limit, 2\)
        elif margin >= 0\.60:
            limit = min\(limit, 3\)
    else:
        limit = max\(limit, 3\)

    ranked_by_policy = sorted\(
        range\(len\(policy_log_probs\)\),
        key=lambda index: float\(policy_log_probs\[index\]\),
        reverse=True,
    \)
    candidate_pool = ranked_by_policy\[: max\(limit \* 3, limit\)\]
    tactical_ranked = sorted\(
        candidate_pool,
        key=lambda index: \(
            _tactical_priority_score\(context, int\(context\.legal_low_level_codes\[index\]\)\),
            float\(policy_log_probs\[index\]\),
        \),
        reverse=True,
    \)

    selected: list\[int\] = \[\]
    seen: set\[int\] = set\(\)
    declare_end_added = False

    def consider\(index: int\) -> None:
        nonlocal declare_end_added
        if index in seen:
            return
        action_code = int\(context\.legal_low_level_codes\[index\]\)
        kind = low_level_kind_for_code\(action_code\)
        if kind == OptionKind\.ACTION_DECLARE_END:
            if declare_end_added:
                return
            declare_end_added = True
        seen\.add\(index\)
        selected\.append\(index\)

    # Keep tactically meaningful actions first, then fill with policy prior\.
    for index in tactical_ranked:
        action_code = int\(context\.legal_low_level_codes\[index\]\)
        if _tactical_priority_score\(context, action_code\) > 0\.0:
            consider\(index\)
        if len\(selected\) >= limit:
            return tuple\(selected\[:limit\]\)

    for index in ranked_by_policy:
        consider\(index\)
        if len\(selected\) >= limit:
            break
    return tuple\(selected\[:limit\]\)
"""
    prioritized_repl = """
def _prioritized_root_indices(
    *,
    context: DecisionContext,
    policy_log_probs: Sequence[float],
    budget: SePotPhaseBudget,
) -> tuple[int, ...]:
    if not context.legal_low_level_codes:
        return ()
    tactical_reason = _tactical_trigger_reason(context)
    margin = _top_policy_margin(policy_log_probs)
    legal_count = len(context.legal_low_level_codes)
    limit = max(1, min(len(context.legal_low_level_codes), int(budget.root_top_k)))
    # Policy provides a prior only for non-tactical states. Tactical states keep
    # their wider budget so search can correct confident but tactically-wrong policies.
    if tactical_reason is None:
        if legal_count <= 4 and margin >= 1.50:
            limit = 1
        elif margin >= 1.00:
            limit = min(limit, 2)
        elif margin >= 0.60:
            limit = min(limit, 3)
    else:
        limit = max(limit, 3)

    ranked_by_policy = sorted(
        range(len(policy_log_probs)),
        key=lambda index: float(policy_log_probs[index]),
        reverse=True,
    )
    candidate_pool = ranked_by_policy[: max(limit * 3, limit)]
    tactical_ranked = sorted(
        candidate_pool,
        key=lambda index: (
            _tactical_priority_score(context, int(context.legal_low_level_codes[index])),
            float(policy_log_probs[index]),
        ),
        reverse=True,
    )

    selected: list[int] = []
    seen_indices: set[int] = set()
    seen_semantics: set[tuple[Any, ...]] = set()
    declare_end_added = False

    def consider(index: int) -> None:
        nonlocal declare_end_added
        if index in seen_indices:
            return
        action_code = int(context.legal_low_level_codes[index])
        semantic_key = semantic_action_key_for_code(action_code)
        if semantic_key in seen_semantics:
            return
        kind = low_level_kind_for_code(action_code)
        if kind == OptionKind.ACTION_DECLARE_END:
            if declare_end_added:
                return
            declare_end_added = True
        seen_indices.add(index)
        seen_semantics.add(semantic_key)
        selected.append(index)

    # Keep tactically meaningful actions first, then fill with policy prior.
    for index in tactical_ranked:
        action_code = int(context.legal_low_level_codes[index])
        if _tactical_priority_score(context, action_code) > 0.0:
            consider(index)
        if len(selected) >= limit:
            return tuple(selected[:limit])

    for index in ranked_by_policy:
        consider(index)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _reset_branch_env_from_state_json(
    *,
    state_json: str | None,
    env_config: EnvConfig,
    matchup: Matchup,
    seed: int,
) -> tuple[GitcgDecisionEnv | None, DecisionContext | None]:
    if not state_json:
        return None, None
    env = GitcgDecisionEnv(
        env_config,
        matchup,
        enable_result_based_action_relabel=False,
    )
    try:
        context = env.reset(seed=seed, state_json=state_json)
        return env, context
    except Exception:
        env.close()
        return None, None
"""
    text = regex_replace_once(text, prioritized_pattern, prioritized_repl, "prioritized root function")

    fallback_pattern = r"""
def _fallback_decision\(
    \*,
    reason: str,
    policy_log_probs: Sequence\[float\],
    phase_depth: int,
    belief_sample_count: int,
    root_candidate_count: int \| None = None,
\) -> SearchDecision:
    transformed_policy = _softmax_tuple\(policy_log_probs\)
    if transformed_policy:
        selected_action_index = int\(max\(range\(len\(transformed_policy\)\), key=lambda index: transformed_policy\[index\]\)\)
    else:
        selected_action_index = 0
    effective_root_candidate_count = \(
        int\(root_candidate_count\)
        if root_candidate_count is not None
        else len\(transformed_policy\)
    \)
    return SearchDecision\(
        triggered=True,
        fallback_reason=reason,
        selected_action_index=selected_action_index,
        transformed_policy=tuple\(transformed_policy\),
        root_value=0\.0,
        root_candidate_count=max\(0, effective_root_candidate_count\),
        belief_sample_count=int\(belief_sample_count\),
        phase_depth=int\(phase_depth\),
    \)
"""
    fallback_repl = """
def _fallback_decision(
    *,
    reason: str,
    policy_log_probs: Sequence[float],
    phase_depth: int,
    belief_sample_count: int,
    root_candidate_count: int | None = None,
) -> SearchDecision:
    transformed_policy = _softmax_tuple(policy_log_probs)
    if transformed_policy:
        selected_action_index = int(max(range(len(transformed_policy)), key=lambda index: transformed_policy[index]))
    else:
        selected_action_index = 0
    effective_root_candidate_count = int(root_candidate_count) if root_candidate_count is not None else 0
    return SearchDecision(
        triggered=True,
        fallback_reason=reason,
        selected_action_index=selected_action_index,
        transformed_policy=tuple(transformed_policy),
        root_value=0.0,
        root_candidate_count=max(0, effective_root_candidate_count),
        belief_sample_count=int(belief_sample_count),
        phase_depth=int(phase_depth),
    )
"""
    text = regex_replace_once(text, fallback_pattern, fallback_repl, "fallback decision cleanup")

    if text == original:
        raise RuntimeError("no changes applied")

    backup = TARGET.with_name(TARGET.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"Patched {TARGET}")
    print(f"Backup saved to {backup}")


if __name__ == "__main__":
    main()
