from pathlib import Path
import re
import sys

FILE = Path('research/world_model/src/gitcg_world_model/sepot_search.py')
BACKUP = Path('research/world_model/src/gitcg_world_model/sepot_search.py.bak_paper_aligned')

NEW_SHOULD_TRIGGER = '''def should_trigger_search(
    context: DecisionContext,
    *,
    config: SePotSearchConfig,
) -> bool:
    if not config.enabled:
        return False
    if context.request_type != DecisionType.ACTION:
        return False
    if context.player_view is None:
        return False
    legal_count = len(context.legal_low_level_codes)
    round_number = int(context.player_view.round_number)
    has_end = any(
        low_level_kind_for_code(int(action_code)) == OptionKind.ACTION_DECLARE_END
        for action_code in context.legal_low_level_codes
    )
    productive_non_end = any(
        low_level_kind_for_code(int(action_code)) in {
            OptionKind.ACTION_USE_SKILL,
            OptionKind.ACTION_PLAY_CARD,
            OptionKind.ACTION_ELEMENTAL_TUNING,
        }
        for action_code in context.legal_low_level_codes
    )
    # Paper-aligned principle: search should be forced by tactical / structural
    # ambiguity, not vetoed by the learned policy. Tactical opportunities and
    # end-vs-continue conflicts must enter search even when the policy is very
    # confident.
    if _tactical_trigger_reason(context) is not None:
        return True
    if has_end and productive_non_end:
        return True
    if legal_count >= 8:
        return True
    if round_number >= 6 and legal_count >= 3:
        return True
    return False
'''

INSERT_HELPERS = '''

def _semantic_key_dict(action_code: int) -> dict[str, object]:
    return dict(semantic_action_key_for_code(int(action_code)))


def _active_character_index(player) -> int:
    for index, character in enumerate(player.characters):
        if bool(character.is_active):
            return int(index)
    return 0


def _targets_opponent_frontline(action_code: int) -> bool:
    target_slots = _semantic_key_dict(action_code).get("target_slots")
    if not isinstance(target_slots, tuple):
        return False
    return any(
        owner == "opponent" and zone == "character" and int(index) == 0
        for owner, zone, index in target_slots
    )


def _targets_opponent_backline(action_code: int) -> bool:
    target_slots = _semantic_key_dict(action_code).get("target_slots")
    if not isinstance(target_slots, tuple):
        return False
    return any(
        owner == "opponent" and zone == "character" and int(index) > 0
        for owner, zone, index in target_slots
    )


def _tactical_trigger_reason(context: DecisionContext) -> str | None:
    visible_state = context.player_view
    if visible_state is None or context.request_type != DecisionType.ACTION:
        return None
    acting_player = int(context.acting_player)
    opponent = visible_state.players[1 - acting_player]
    opponent_active = opponent.characters[_active_character_index(opponent)]
    opponent_front_hp = int(opponent_active.health)
    has_end = False
    has_productive_non_end = False
    frontline_pressure = False
    backline_pressure = False
    for action_code in context.legal_low_level_codes:
        kind = low_level_kind_for_code(int(action_code))
        if kind == OptionKind.ACTION_DECLARE_END:
            has_end = True
            continue
        if kind in {
            OptionKind.ACTION_USE_SKILL,
            OptionKind.ACTION_PLAY_CARD,
            OptionKind.ACTION_ELEMENTAL_TUNING,
            OptionKind.ACTION_SWITCH_ACTIVE,
        }:
            has_productive_non_end = True
        if kind in {OptionKind.ACTION_USE_SKILL, OptionKind.ACTION_PLAY_CARD}:
            if _targets_opponent_frontline(int(action_code)):
                frontline_pressure = True
            if _targets_opponent_backline(int(action_code)):
                backline_pressure = True
    if opponent_front_hp <= 3 and frontline_pressure:
        return "frontline_lethal_window"
    if opponent_front_hp <= 5 and frontline_pressure and has_end:
        return "frontline_pressure_vs_end"
    if backline_pressure and has_end:
        return "backline_pressure_vs_end"
    if has_end and has_productive_non_end:
        return "end_conflict"
    return None


def _top_policy_margin(policy_log_probs: Sequence[float]) -> float:
    if len(policy_log_probs) < 2:
        return float("inf")
    ranked = sorted((float(v) for v in policy_log_probs), reverse=True)
    return float(ranked[0] - ranked[1])


def _tactical_priority_score(context: DecisionContext, action_code: int) -> float:
    visible_state = context.player_view
    if visible_state is None:
        return 0.0
    acting_player = int(context.acting_player)
    opponent = visible_state.players[1 - acting_player]
    opponent_active = opponent.characters[_active_character_index(opponent)]
    opponent_front_hp = int(opponent_active.health)
    kind = low_level_kind_for_code(int(action_code))
    score = 0.0
    if kind == OptionKind.ACTION_DECLARE_END:
        return -10.0
    if kind in {OptionKind.ACTION_USE_SKILL, OptionKind.ACTION_PLAY_CARD}:
        if _targets_opponent_frontline(int(action_code)):
            score += 4.0
            if opponent_front_hp <= 5:
                score += 3.0
            if opponent_front_hp <= 3:
                score += 5.0
        if _targets_opponent_backline(int(action_code)):
            score += 1.5
    if kind == OptionKind.ACTION_SWITCH_ACTIVE and opponent_front_hp <= 3:
        score += 1.0
    if kind == OptionKind.ACTION_ELEMENTAL_TUNING and opponent_front_hp <= 3:
        score -= 1.0
    score -= 0.1 * float(public_spent_dice_for_code(int(action_code)))
    return score


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
'''

REPLACE_ROOT_SELECTION = '''        root_indices = _prioritized_root_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
        if not root_indices:
            return _fallback_decision(
                reason="no_root_candidates",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
'''


def replace_function(text: str, func_name: str, new_code: str) -> str:
    pattern = re.compile(rf"def {func_name}\(.*?\n(?=def |class |\Z)", re.S)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f'Could not find function {func_name}')
    return text[:m.start()] + new_code + "\n\n" + text[m.end():]


def main() -> None:
    if not FILE.exists():
        print(f'File not found: {FILE}')
        sys.exit(1)
    text = FILE.read_text(encoding='utf-8')
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding='utf-8')

    text = replace_function(text, 'should_trigger_search', NEW_SHOULD_TRIGGER.strip())

    marker = '\ndef _build_self_range_from_state('
    if marker not in text:
        raise RuntimeError('Could not find insertion marker for helper functions')
    if '_prioritized_root_indices(' not in text:
        text = text.replace(marker, INSERT_HELPERS + marker)

    old_block_pattern = re.compile(
        r"\s*ranked = sorted\(range\(len\(policy_log_probs\)\), key=lambda index: float\(policy_log_probs\[index\]\), reverse=True\)\n"
        r"\s*root_indices = tuple\(ranked\[: max\(1, min\(len\(ranked\), budget\.root_top_k\)\)\]\)\n"
        r"\s*if not root_indices:\n"
        r"\s*return _fallback_decision\(\n"
        r"\s*reason=\"no_root_candidates\",\n"
        r"\s*policy_log_probs=policy_log_probs,\n"
        r"\s*phase_depth=budget\.depth,\n"
        r"\s*belief_sample_count=public_belief\.opponent_range\.count,\n"
        r"\s*\)\n",
        re.S,
    )
    if old_block_pattern.search(text):
        text = old_block_pattern.sub(REPLACE_ROOT_SELECTION, text, count=1)
    elif '_prioritized_root_indices(' not in text:
        raise RuntimeError('Could not replace root selection block')

    FILE.write_text(text, encoding='utf-8')
    print(f'Patched: {FILE}')
    print(f'Backup:  {BACKUP}')


if __name__ == '__main__':
    main()
