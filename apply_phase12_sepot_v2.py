from __future__ import annotations

from pathlib import Path
import re
import sys

TARGET = Path("research/world_model/src/gitcg_world_model/sepot_search.py")
BACKUP_SUFFIX = ".bak_phase12_v2"


def normalize_newlines(text: str) -> tuple[str, str]:
    if "\r\n" in text:
        return text.replace("\r\n", "\n"), "\r\n"
    return text, "\n"


def regex_replace_once(text: str, pattern: str, repl: str, label: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"expected 1 regex replacement for {label}, got {count}")
    return new_text


def insert_once_after(text: str, anchor: str, addition: str, label: str) -> str:
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError(f"missing anchor for {label}")
    insert_at = idx + len(anchor)
    return text[:insert_at] + addition + text[insert_at:]


def main() -> None:
    if not TARGET.exists():
        print(f"Target file not found: {TARGET}")
        sys.exit(1)

    original_raw = TARGET.read_text(encoding="utf-8")
    text, newline = normalize_newlines(original_raw)
    original = text

    # text = regex_replace_once(
    #     text,
    #     r'\n([ \t]*)if not _should_search_for_policy\(context=context, policy_log_probs=policy_log_probs\):\n\1    return None\n',
    #     '\n\\1# Phase 1: policy is a prior for ordering / scoring, not a veto over\n\\1# whether search may run at all. Triggering is handled by tactical /\n\\1# structural conditions above.\n',
    #     "remove policy veto gate",
    #     flags=re.M,
    # )

    text = regex_replace_once(
        text,
        r'\n([ \t]*)if reconstructor\.ensure_template\(acting_player=root_player\) is None:\n'
        r'\1    return _fallback_decision\(\n'
        r'\1        reason="template_unavailable",\n'
        r'\1        policy_log_probs=policy_log_probs,\n'
        r'\1        phase_depth=budget\.depth,\n'
        r'\1        belief_sample_count=public_belief\.opponent_range\.count,\n'
        r'\1    \)\n',
        '\n\\1# Phase 2: do not require a synthetic ACTION template before root search.\n'
        '\\1# Root branches should start from the live root state\'s full_state_json\n'
        '\\1# whenever available. Posterior updates may still fall back to the\n'
        '\\1# reconstructor until later phases remove it completely.\n',
        "remove eager template requirement",
        flags=re.M,
    )

    text = regex_replace_once(
        text,
        r'root_indices = _select_root_candidate_indices\(',
        'root_indices = _prioritized_root_indices(',
        "switch root selector",
    )

    text = regex_replace_once(
        text,
        r'return _fallback_decision\(\n([ \t]*)reason="no_root_candidates",\n'
        r'\1policy_log_probs=policy_log_probs,\n'
        r'\1phase_depth=budget\.depth,\n'
        r'\1belief_sample_count=public_belief\.opponent_range\.count,\n'
        r'([ \t]*)\)',
        'return _fallback_decision(\n\\1reason="no_root_candidates",\n\\1policy_log_probs=policy_log_probs,\n\\1phase_depth=budget.depth,\n\\1belief_sample_count=public_belief.opponent_range.count,\n\\1root_candidate_count=0,\n\\2)',
        "no_root_candidates fallback count",
        flags=re.M,
    )

    text = regex_replace_once(
        text,
        r'return _fallback_decision\(\n([ \t]*)reason=\("timeout" if timed_out else "reconstruction_failed"\),\n'
        r'\1policy_log_probs=policy_log_probs,\n'
        r'\1phase_depth=budget\.depth,\n'
        r'\1belief_sample_count=public_belief\.opponent_range\.count,\n'
        r'([ \t]*)\)',
        'return _fallback_decision(\n\\1reason=("timeout" if timed_out else "reconstruction_failed"),\n\\1policy_log_probs=policy_log_probs,\n\\1phase_depth=budget.depth,\n\\1belief_sample_count=public_belief.opponent_range.count,\n\\1root_candidate_count=len(root_indices),\n\\2)',
        "root_scores empty fallback count",
        flags=re.M,
    )

    text = regex_replace_once(
        text,
        r'''sampled_self_hypothesis = root_public_belief\.self_range\.hypotheses\[0\]\n([ \t]*)env, current_context = reconstructor\.reconstruct\(\n([ \t]*)public_belief=root_public_belief,\n([ \t]*)self_hypothesis=sampled_self_hypothesis,\n([ \t]*)opponent_hypothesis=sampled_opponent_hypothesis,\n([ \t]*)\)\n([ \t]*)if env is None or current_context is None:\n([ \t]*)    return None''',
        '''sampled_self_hypothesis = root_public_belief.self_range.hypotheses[0]
\\1env, current_context = _reset_branch_env_from_state_json(
\\2state_json=root_context.full_state_json,
\\2env_config=reconstructor.env_config,
\\2matchup=reconstructor.matchup,
\\2seed=reconstructor.config.template_seed,
\\1)
\\1if env is None or current_context is None:
\\2env, current_context = reconstructor.reconstruct(
\\3public_belief=root_public_belief,
\\3self_hypothesis=sampled_self_hypothesis,
\\3opponent_hypothesis=sampled_opponent_hypothesis,
\\2)
\\1if env is None or current_context is None:
\\2return None''',
        "root branch direct reset from live state",
        flags=re.M,
    )

    prioritized_pattern = r'''
def _prioritized_root_indices\(
    \*,
    context: DecisionContext,
    policy_log_probs: Sequence\[float\],
    budget: SePotPhaseBudget,
\) -> tuple\[int, \.\.\.\]:
(?:.*?\n)*?    return tuple\(selected\[:limit\]\)
'''
    prioritized_repl = '''def _prioritized_root_indices(
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

'''
    text = regex_replace_once(text, prioritized_pattern, prioritized_repl, "prioritized root function")

    helper = '''
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

'''
    if "_reset_branch_env_from_state_json(" not in text:
        anchor = "def _build_self_range_from_state(state: StateSnapshot, *, player_id: int) -> ExplicitRange:\n"
        text = insert_once_after(text, anchor, helper, "insert branch reset helper")

    text = regex_replace_once(
        text,
        r'''def _fallback_decision\(
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
    \)''',
        '''def _fallback_decision(
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
    effective_root_candidate_count = int(root_candidate_count) if root_candidate_count is not None else 0''',
        "fallback decision cleanup",
    )

    if text == original:
        raise RuntimeError("no changes applied")

    backup = TARGET.with_name(TARGET.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(original_raw, encoding="utf-8")

    out = text.replace("\n", newline)
    TARGET.write_text(out, encoding="utf-8")
    print(f"Patched {TARGET}")
    print(f"Backup saved to {backup}")


if __name__ == "__main__":
    main()
