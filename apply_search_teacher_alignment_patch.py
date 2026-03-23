
from __future__ import annotations

import argparse
from pathlib import Path


LOOKAHEAD_OLD_DATACLASS = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_NEW_DATACLASS = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    action_codes: tuple[int, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_OLD_ANNOTATE = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_NEW_ANNOTATE = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_action_codes"] = [int(code) for code in target.action_codes]
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_OLD_RETURN = """    return SearchTeacherTarget(
        policy_target=policy_target,
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

LOOKAHEAD_NEW_RETURN = """    return SearchTeacherTarget(
        policy_target=policy_target,
        action_codes=tuple(int(code) for code in root_context.legal_low_level_codes),
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

PPO_HELPER_ANCHOR = """@dataclass(frozen=True)
class PpoTrainSample:
"""

PPO_HELPER_BLOCK = """def _normalize_policy(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 1.0e-8:
        return tuple(0.0 for _ in values)
    return tuple(max(0.0, float(value)) / total for value in values)


def _align_search_teacher_policy_target(
    *,
    context: DecisionContext,
    policy: Sequence[float],
    teacher_action_codes: Sequence[int] | None = None,
) -> tuple[float, ...]:
    current_codes = tuple(int(code) for code in context.legal_low_level_codes)
    if not current_codes:
        return ()
    if not policy:
        return tuple(0.0 for _ in current_codes)

    if teacher_action_codes is not None and len(teacher_action_codes) == len(policy):
        aligned = [0.0] * len(current_codes)
        index_by_code = {int(code): index for index, code in enumerate(current_codes)}
        for source_index, action_code in enumerate(teacher_action_codes):
            target_index = index_by_code.get(int(action_code))
            if target_index is None:
                continue
            aligned[target_index] += float(policy[source_index])
        normalized = _normalize_policy(aligned)
        if any(value > 0.0 for value in normalized):
            return normalized
        print(
            "[search-teacher-align] "
            f"status=drop_all_unmatched current={len(current_codes)} teacher={len(policy)}",
            flush=True,
        )
        return tuple(0.0 for _ in current_codes)

    if len(policy) == len(current_codes):
        return _normalize_policy(tuple(float(value) for value in policy))

    print(
        "[search-teacher-align] "
        f"status=length_mismatch current={len(current_codes)} teacher={len(policy)}",
        flush=True,
    )
    return tuple(0.0 for _ in current_codes)


"""

PPO_OLD_SNIPPET = """            search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            result[player].append(
"""

PPO_NEW_SNIPPET = """            raw_search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            raw_search_teacher_action_codes = tuple(
                int(value) for value in transition.source_step.metadata.get("search_teacher_action_codes", ())
            )
            search_teacher_policy_target = _align_search_teacher_policy_target(
                context=transition_context,
                policy=raw_search_teacher_policy_target,
                teacher_action_codes=(raw_search_teacher_action_codes if raw_search_teacher_action_codes else None),
            )
            result[player].append(
"""

AGG_OLD = """def aggregate_high_policy(
    *,
    context: DecisionContext,
    low_level_policy: Sequence[float],
) -> tuple[float, ...]:
    high_dim = high_action_vocab_size()
    aggregated = [0.0] * high_dim
    high_to_low = legal_high_to_low_dict(context)
    probability_by_code = {
        int(code): float(low_level_policy[index])
        for index, code in enumerate(context.legal_low_level_codes)
    }
"""

AGG_NEW = """def aggregate_high_policy(
    *,
    context: DecisionContext,
    low_level_policy: Sequence[float],
) -> tuple[float, ...]:
    high_dim = high_action_vocab_size()
    aggregated = [0.0] * high_dim
    high_to_low = legal_high_to_low_dict(context)
    legal_codes = tuple(int(code) for code in context.legal_low_level_codes)
    if len(low_level_policy) != len(legal_codes):
        print(
            "[aggregate-high-policy] "
            f"length_mismatch policy={len(low_level_policy)} legal={len(legal_codes)}",
            flush=True,
        )
    usable = min(len(low_level_policy), len(legal_codes))
    probability_by_code = {
        int(legal_codes[index]): float(low_level_policy[index])
        for index in range(usable)
    }
"""


def patch_text(text: str, old: str, new: str, marker: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if marker in text:
        return text
    raise RuntimeError(f"Could not patch block: {marker}")


def patch_lookahead(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, LOOKAHEAD_OLD_DATACLASS, LOOKAHEAD_NEW_DATACLASS, "action_codes: tuple[int, ...]")
    text = patch_text(text, LOOKAHEAD_OLD_ANNOTATE, LOOKAHEAD_NEW_ANNOTATE, '"search_teacher_action_codes"')
    text = patch_text(
        text,
        LOOKAHEAD_OLD_RETURN,
        LOOKAHEAD_NEW_RETURN,
        "action_codes=tuple(int(code) for code in root_context.legal_low_level_codes)",
    )
    path.write_text(text, encoding="utf-8")


def patch_ppo_training(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "_align_search_teacher_policy_target(" not in text:
        if PPO_HELPER_ANCHOR not in text:
            raise RuntimeError("Could not find PpoTrainSample anchor in ppo_training.py")
        text = text.replace(PPO_HELPER_ANCHOR, PPO_HELPER_BLOCK + PPO_HELPER_ANCHOR, 1)
    text = patch_text(text, PPO_OLD_SNIPPET, PPO_NEW_SNIPPET, "raw_search_teacher_action_codes")
    path.write_text(text, encoding="utf-8")


def patch_action_hierarchy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, AGG_OLD, AGG_NEW, '[aggregate-high-policy]')
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch search teacher alignment across lookahead_search.py, ppo_training.py, action_hierarchy.py")
    parser.add_argument("--lookahead", required=True)
    parser.add_argument("--ppo-training", required=True)
    parser.add_argument("--action-hierarchy", required=True)
    args = parser.parse_args()

    lookahead = Path(args.lookahead)
    ppo_training = Path(args.ppo_training)
    action_hierarchy = Path(args.action_hierarchy)

    if not lookahead.exists():
        raise SystemExit(f"lookahead_search.py not found: {lookahead}")
    if not ppo_training.exists():
        raise SystemExit(f"ppo_training.py not found: {ppo_training}")
    if not action_hierarchy.exists():
        raise SystemExit(f"action_hierarchy.py not found: {action_hierarchy}")

    patch_lookahead(lookahead)
    patch_ppo_training(ppo_training)
    patch_action_hierarchy(action_hierarchy)
    print(f"patched: {lookahead}")
    print(f"patched: {ppo_training}")
    print(f"patched: {action_hierarchy}")


if __name__ == "__main__":
    main()
