from pathlib import Path

path = Path('research/world_model/src/gitcg_world_model/sepot_search.py')
text = path.read_text(encoding='utf-8')
original = text

replacements = []

old = """        if not _should_search_for_policy(context=context, policy_log_probs=policy_log_probs):
            return None
"""
new = """        # Paper-aligned runtime rule: the learned policy provides a prior for
        # ordering / pruning branches, but it does not veto search after a
        # tactical or structural trigger has fired.
"""
replacements.append((old, new))

old = """        root_indices = _select_root_candidate_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
"""
new = """        root_indices = _prioritized_root_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
"""
replacements.append((old, new))

old = """            return _fallback_decision(
                reason="no_root_candidates",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
"""
new = """            return _fallback_decision(
                reason="no_root_candidates",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
                root_candidate_count=0,
            )
"""
replacements.append((old, new))

old = """            return _fallback_decision(
                reason=("timeout" if timed_out else "reconstruction_failed"),
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
"""
new = """            return _fallback_decision(
                reason=("timeout" if timed_out else "reconstruction_failed"),
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
                root_candidate_count=len(root_indices),
            )
"""
replacements.append((old, new))

old = """def _prioritized_root_indices(
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
    seen: set[int] = set()
    declare_end_added = False

    def consider(index: int) -> None:
        nonlocal declare_end_added
        if index in seen:
            return
        action_code = int(context.legal_low_level_codes[index])
        kind = low_level_kind_for_code(action_code)
        if kind == OptionKind.ACTION_DECLARE_END:
            if declare_end_added:
                return
            declare_end_added = True
        seen.add(index)
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
"""

new = """def _semantic_dedup_ranked_indices(
    *,
    context: DecisionContext,
    ranked_indices: Sequence[int],
) -> tuple[int, ...]:
    deduped: list[int] = []
    seen_semantic_keys: set[tuple[Any, ...]] = set()
    for index in ranked_indices:
        action_code = int(context.legal_low_level_codes[index])
        try:
            semantic_key = semantic_action_key_for_code(action_code)
        except KeyError:
            semantic_key = (("action_code", action_code),)
        if semantic_key in seen_semantic_keys:
            continue
        seen_semantic_keys.add(semantic_key)
        deduped.append(int(index))
    return tuple(deduped)


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

    ranked_by_policy = tuple(
        sorted(
            range(len(policy_log_probs)),
            key=lambda index: float(policy_log_probs[index]),
            reverse=True,
        )
    )
    ranked_by_policy = _semantic_dedup_ranked_indices(
        context=context,
        ranked_indices=ranked_by_policy,
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
    seen: set[int] = set()
    declare_end_added = False

    def consider(index: int) -> None:
        nonlocal declare_end_added
        if index in seen:
            return
        action_code = int(context.legal_low_level_codes[index])
        kind = low_level_kind_for_code(action_code)
        if kind == OptionKind.ACTION_DECLARE_END:
            if declare_end_added:
                return
            declare_end_added = True
        seen.add(index)
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
"""
replacements.append((old, new))

old = """    effective_root_candidate_count = (
        int(root_candidate_count)
        if root_candidate_count is not None
        else len(transformed_policy)
    )
"""
new = """    effective_root_candidate_count = (
        int(root_candidate_count)
        if root_candidate_count is not None
        else 0
    )
"""
replacements.append((old, new))

for old, new in replacements:
    if old not in text:
        raise SystemExit(f'Missing expected snippet:\n{old[:200]}')
    text = text.replace(old, new, 1)

if text == original:
    raise SystemExit('No changes made')

backup = path.with_suffix(path.suffix + '.bak_semantic_prioritized')
backup.write_text(original, encoding='utf-8')
path.write_text(text, encoding='utf-8')
print(f'Patched {path}')
print(f'Backup  {backup}')
