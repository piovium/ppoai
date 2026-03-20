from pathlib import Path
import shutil
import sys

FILE = Path('research/world_model/src/gitcg_world_model/sepot_search.py')


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f'未找到待替换片段: {label}')
    return text.replace(old, new, 1)


def main() -> None:
    if not FILE.exists():
        raise SystemExit(f'找不到文件: {FILE.resolve()}')

    text = FILE.read_text(encoding='utf-8')
    original = text

    old_budget = '''def phase_budget_for_context(context: DecisionContext, *, config: SePotSearchConfig) -> SePotPhaseBudget:\n    round_number = int(context.player_view.round_number) if context.player_view is not None else 0\n    if round_number <= 3:\n        return SePotPhaseBudget(depth=1, root_top_k=3, belief_samples=2, timeout_ms=config.opening_timeout_ms)\n    if round_number <= 8:\n        return SePotPhaseBudget(depth=2, root_top_k=5, belief_samples=4, timeout_ms=config.midgame_timeout_ms)\n    return SePotPhaseBudget(depth=3, root_top_k=8, belief_samples=6, timeout_ms=config.endgame_timeout_ms)\n\n\n'''

    new_budget = '''def phase_budget_for_context(context: DecisionContext, *, config: SePotSearchConfig) -> SePotPhaseBudget:\n    round_number = int(context.player_view.round_number) if context.player_view is not None else 0\n    if round_number <= 3:\n        return SePotPhaseBudget(depth=1, root_top_k=2, belief_samples=2, timeout_ms=config.opening_timeout_ms)\n    if round_number <= 8:\n        return SePotPhaseBudget(depth=2, root_top_k=3, belief_samples=4, timeout_ms=config.midgame_timeout_ms)\n    return SePotPhaseBudget(depth=3, root_top_k=4, belief_samples=6, timeout_ms=config.endgame_timeout_ms)\n\n\ndef _select_root_candidate_indices(\n    *,\n    context: DecisionContext,\n    policy_log_probs: Sequence[float],\n    budget: SePotPhaseBudget,\n) -> tuple[int, ...]:\n    if not policy_log_probs or not context.legal_low_level_codes:\n        return ()\n    ranked = sorted(range(len(policy_log_probs)), key=lambda index: float(policy_log_probs[index]), reverse=True)\n    limit = max(1, min(len(ranked), int(budget.root_top_k)))\n    selected: list[int] = []\n    declare_end_index: int | None = None\n    for index in ranked:\n        action_code = int(context.legal_low_level_codes[index])\n        try:\n            kind = low_level_kind_for_code(action_code)\n        except Exception:\n            kind = None\n        if kind == OptionKind.ACTION_DECLARE_END:\n            if declare_end_index is None:\n                declare_end_index = index\n            continue\n        selected.append(index)\n        if len(selected) >= max(1, limit - 1):\n            break\n    if declare_end_index is not None and len(selected) < limit:\n        selected.append(declare_end_index)\n    if not selected:\n        selected = ranked[:limit]\n    elif len(selected) < limit:\n        for index in ranked:\n            if index in selected:\n                continue\n            selected.append(index)\n            if len(selected) >= limit:\n                break\n    return tuple(selected[:limit])\n\n\n'''

    text = replace_once(text, old_budget, new_budget, 'phase_budget_for_context + _select_root_candidate_indices')

    old_root = '''        ranked = sorted(range(len(policy_log_probs)), key=lambda index: float(policy_log_probs[index]), reverse=True)\n        root_indices = tuple(ranked[: max(1, min(len(ranked), budget.root_top_k))])\n'''
    new_root = '''        root_indices = _select_root_candidate_indices(\n            context=context,\n            policy_log_probs=policy_log_probs,\n            budget=budget,\n        )\n'''
    text = replace_once(text, old_root, new_root, 'choose_action root candidate selection')

    old_advance = '''    def _advance_opponent_range(\n        self,\n        *,\n        pre_public_belief: PublicBeliefState,\n        selected_action_code: int,\n        sampled_self_hypothesis: ExplicitRangeHypothesis,\n        sampled_opponent_hypothesis: ExplicitRangeHypothesis,\n        deadline: float,\n    ) -> ExplicitRange:\n        updated: list[ExplicitRangeHypothesis] = []\n        total_weight = 0.0\n        for hypothesis in pre_public_belief.opponent_range.hypotheses:\n            if time.perf_counter() > deadline:\n                raise _SearchTimeout()\n            env, context = self.reconstructor.reconstruct(\n                public_belief=pre_public_belief,\n                self_hypothesis=sampled_self_hypothesis,\n                opponent_hypothesis=hypothesis,\n            )\n            if env is None or context is None:\n                continue\n            try:\n                matched = match_low_level_code_index(context, selected_action_code)\n                if matched is None:\n                    continue\n                _, step, _ = env.step(ActionChoice(action_code=context.legal_low_level_codes[matched]))\n                next_hypothesis = _extract_player_hypothesis(\n                    step.post_state,\n                    player_id=1 - pre_public_belief.root_player,\n                    weight=float(hypothesis.weight),\n                )\n                updated.append(next_hypothesis)\n                total_weight += float(hypothesis.weight)\n            finally:\n                env.close()\n        if not updated or total_weight <= 0.0:\n            return ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))\n        normalized = tuple(\n            replace_weight(hypothesis, float(hypothesis.weight) / total_weight)\n            for hypothesis in updated\n        )\n        return ExplicitRange(normalized)\n'''

    new_advance = '''    def _advance_opponent_range(\n        self,\n        *,\n        pre_public_belief: PublicBeliefState,\n        selected_action_code: int,\n        sampled_self_hypothesis: ExplicitRangeHypothesis,\n        sampled_opponent_hypothesis: ExplicitRangeHypothesis,\n        deadline: float,\n    ) -> ExplicitRange:\n        updated: list[ExplicitRangeHypothesis] = []\n        total_weight = 0.0\n        for hypothesis in pre_public_belief.opponent_range.hypotheses:\n            now = time.perf_counter()\n            if now > deadline:\n                if updated and total_weight > 0.0:\n                    break\n                raise _SearchTimeout()\n            env, context = self.reconstructor.reconstruct(\n                public_belief=pre_public_belief,\n                self_hypothesis=sampled_self_hypothesis,\n                opponent_hypothesis=hypothesis,\n            )\n            if env is None or context is None:\n                continue\n            try:\n                if time.perf_counter() > deadline:\n                    if updated and total_weight > 0.0:\n                        break\n                    raise _SearchTimeout()\n                matched = match_low_level_code_index(context, selected_action_code)\n                if matched is None:\n                    continue\n                if time.perf_counter() > deadline:\n                    if updated and total_weight > 0.0:\n                        break\n                    raise _SearchTimeout()\n                _, step, _ = env.step(ActionChoice(action_code=context.legal_low_level_codes[matched]))\n                next_hypothesis = _extract_player_hypothesis(\n                    step.post_state,\n                    player_id=1 - pre_public_belief.root_player,\n                    weight=float(hypothesis.weight),\n                )\n                updated.append(next_hypothesis)\n                total_weight += float(hypothesis.weight)\n            finally:\n                env.close()\n        if not updated or total_weight <= 0.0:\n            return ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))\n        normalized = tuple(\n            replace_weight(hypothesis, float(hypothesis.weight) / total_weight)\n            for hypothesis in updated\n        )\n        return ExplicitRange(normalized)\n'''
    text = replace_once(text, old_advance, new_advance, '_advance_opponent_range')

    if text == original:
        raise RuntimeError('文件未发生变化')

    backup = FILE.with_suffix(FILE.suffix + '.bak_root_timeout')
    shutil.copy2(FILE, backup)
    FILE.write_text(text, encoding='utf-8')
    print(f'已更新: {FILE}')
    print(f'已备份: {backup}')


if __name__ == '__main__':
    main()
