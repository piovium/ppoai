
from __future__ import annotations

import argparse
from pathlib import Path


def patch_text(text: str, old: str, new: str, marker: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if marker in text:
        return text
    raise RuntimeError(f"Could not patch block: {marker}")


LOOKAHEAD_IMPORT_OLD = """from .action_hierarchy import (
    match_low_level_code_index,
    try_low_level_kind_for_code,
)"""

LOOKAHEAD_IMPORT_NEW = """from .action_hierarchy import (
    match_low_level_code_index,
    semantic_action_key_for_code,
    try_low_level_kind_for_code,
)"""

LOOKAHEAD_DATACLASS_ORIGINAL = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_DATACLASS_FINAL = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    action_semantic_keys: tuple[str, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_ANNOTATE_ORIGINAL = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_ANNOTATE_FINAL = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_action_semantic_keys"] = list(target.action_semantic_keys)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_RETURN_ORIGINAL = """    return SearchTeacherTarget(
        policy_target=policy_target,
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

LOOKAHEAD_RETURN_FINAL = """    return SearchTeacherTarget(
        policy_target=policy_target,
        action_semantic_keys=tuple(
            _semantic_key_token_for_code(int(code))
            for code in root_context.legal_low_level_codes
        ),
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

LOOKAHEAD_HELPER_ANCHOR = "def _softmax_tuple(logits: Sequence[float]) -> tuple[float, ...]:\n"

LOOKAHEAD_HELPER_BLOCK = """def _jsonable_semantic_value(value):
    if isinstance(value, tuple):
        return [_jsonable_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_semantic_value(item) for item in value]
    return value


def _semantic_key_token_for_code(action_code: int) -> str:
    import json

    key = semantic_action_key_for_code(int(action_code))
    return json.dumps(_jsonable_semantic_value(key), ensure_ascii=False, separators=(",", ":"))


"""

PPO_IMPORT_OLD = """from .action_hierarchy import (
    aggregate_high_policy,
    high_action_vocab_size,
    legal_low_level_specs,
    selected_low_level_code,
)"""

PPO_IMPORT_FINAL = """from .action_hierarchy import (
    aggregate_high_policy,
    high_action_vocab_size,
    legal_low_level_specs,
    selected_low_level_code,
    semantic_action_key_for_code,
)"""

PPO_HELPER_ANCHOR = "@dataclass(frozen=True)\nclass PpoTrainSample:\n"

PPO_HELPER_BLOCK = """def _jsonable_semantic_value(value):
    if isinstance(value, tuple):
        return [_jsonable_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_semantic_value(item) for item in value]
    return value


def _semantic_key_token_for_code(action_code: int) -> str:
    import json

    key = semantic_action_key_for_code(int(action_code))
    return json.dumps(_jsonable_semantic_value(key), ensure_ascii=False, separators=(",", ":"))


def _normalize_policy(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 1.0e-8:
        return tuple(0.0 for _ in values)
    return tuple(max(0.0, float(value)) / total for value in values)


def _align_search_teacher_policy_target(
    *,
    context: DecisionContext,
    policy: Sequence[float],
    teacher_action_semantic_keys: Sequence[str] | None = None,
) -> tuple[float, ...]:
    current_codes = tuple(int(code) for code in context.legal_low_level_codes)
    if not current_codes:
        return ()
    if not policy:
        return tuple(0.0 for _ in current_codes)

    current_tokens = tuple(_semantic_key_token_for_code(code) for code in current_codes)

    if teacher_action_semantic_keys is not None and len(teacher_action_semantic_keys) == len(policy):
        aligned = [0.0] * len(current_codes)
        index_by_token = {token: index for index, token in enumerate(current_tokens)}
        for source_index, token in enumerate(teacher_action_semantic_keys):
            target_index = index_by_token.get(str(token))
            if target_index is None:
                continue
            aligned[target_index] += float(policy[source_index])
        normalized = _normalize_policy(aligned)
        if any(value > 0.0 for value in normalized):
            return normalized
        print(
            "[search-teacher-align] "
            f"status=drop_all_unmatched semantic current={len(current_codes)} teacher={len(policy)}",
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

PPO_SNIPPET_ORIGINAL = """            search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            result[player].append(
"""

PPO_SNIPPET_FINAL = """            raw_search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            raw_search_teacher_action_semantic_keys = tuple(
                str(value)
                for value in transition.source_step.metadata.get("search_teacher_action_semantic_keys", ())
            )
            search_teacher_policy_target = _align_search_teacher_policy_target(
                context=transition_context,
                policy=raw_search_teacher_policy_target,
                teacher_action_semantic_keys=(
                    raw_search_teacher_action_semantic_keys
                    if raw_search_teacher_action_semantic_keys
                    else None
                ),
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

AGG_FINAL = """def aggregate_high_policy(
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


def patch_lookahead(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, LOOKAHEAD_IMPORT_OLD, LOOKAHEAD_IMPORT_NEW, "semantic_action_key_for_code")
    text = patch_text(text, LOOKAHEAD_DATACLASS_ORIGINAL, LOOKAHEAD_DATACLASS_FINAL, "action_semantic_keys: tuple[str, ...]")
    text = patch_text(text, LOOKAHEAD_ANNOTATE_ORIGINAL, LOOKAHEAD_ANNOTATE_FINAL, '"search_teacher_action_semantic_keys"')
    text = patch_text(text, LOOKAHEAD_RETURN_ORIGINAL, LOOKAHEAD_RETURN_FINAL, "_semantic_key_token_for_code(int(code))")
    if "_semantic_key_token_for_code(" not in text:
        if LOOKAHEAD_HELPER_ANCHOR not in text:
            raise RuntimeError("Could not find helper anchor in lookahead_search.py")
        text = text.replace(LOOKAHEAD_HELPER_ANCHOR, LOOKAHEAD_HELPER_BLOCK + LOOKAHEAD_HELPER_ANCHOR, 1)
    path.write_text(text, encoding="utf-8")


def patch_ppo_training(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, PPO_IMPORT_OLD, PPO_IMPORT_FINAL, "semantic_action_key_for_code")
    if "_align_search_teacher_policy_target(" not in text or "_semantic_key_token_for_code(" not in text:
        if PPO_HELPER_ANCHOR not in text:
            raise RuntimeError("Could not find helper anchor in ppo_training.py")
        text = text.replace(PPO_HELPER_ANCHOR, PPO_HELPER_BLOCK + PPO_HELPER_ANCHOR, 1)
    text = patch_text(text, PPO_SNIPPET_ORIGINAL, PPO_SNIPPET_FINAL, "raw_search_teacher_action_semantic_keys")
    path.write_text(text, encoding="utf-8")


def patch_action_hierarchy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, AGG_OLD, AGG_FINAL, '[aggregate-high-policy]')
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final consolidated patch for stable search-teacher alignment and aggregate-high-policy guard.")
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
