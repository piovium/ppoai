
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

LOOKAHEAD_DATACLASS_OLD = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    action_codes: tuple[int, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_DATACLASS_NEW = """@dataclass(frozen=True)
class SearchTeacherTarget:
    policy_target: tuple[float, ...]
    action_semantic_keys: tuple[str, ...]
    value_target: float
    weight: float
"""

LOOKAHEAD_ANNOTATE_OLD = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_action_codes"] = [int(code) for code in target.action_codes]
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_ANNOTATE_NEW = """        metadata["search_teacher_policy"] = list(target.policy_target)
        metadata["search_teacher_action_semantic_keys"] = list(target.action_semantic_keys)
        metadata["search_teacher_value"] = float(target.value_target)
        metadata["search_teacher_weight"] = float(target.weight)
"""

LOOKAHEAD_RETURN_OLD = """    return SearchTeacherTarget(
        policy_target=policy_target,
        action_codes=tuple(int(code) for code in root_context.legal_low_level_codes),
        value_target=float(best_score),
        weight=float(teacher_weight),
    )
"""

LOOKAHEAD_RETURN_NEW = """    return SearchTeacherTarget(
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

PPO_IMPORT_NEW = """from .action_hierarchy import (
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
    teacher_action_codes: Sequence[int] | None = None,
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

    if len(policy) == len(current_codes):
        return _normalize_policy(tuple(float(value) for value in policy))

    print(
        "[search-teacher-align] "
        f"status=length_mismatch current={len(current_codes)} teacher={len(policy)}",
        flush=True,
    )
    return tuple(0.0 for _ in current_codes)


"""

PPO_SNIPPET_OLD = """            raw_search_teacher_policy_target = tuple(
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
"""

PPO_SNIPPET_NEW = """            raw_search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            raw_search_teacher_action_semantic_keys = tuple(
                str(value)
                for value in transition.source_step.metadata.get("search_teacher_action_semantic_keys", ())
            )
            raw_search_teacher_action_codes = tuple(
                int(value) for value in transition.source_step.metadata.get("search_teacher_action_codes", ())
            )
            search_teacher_policy_target = _align_search_teacher_policy_target(
                context=transition_context,
                policy=raw_search_teacher_policy_target,
                teacher_action_semantic_keys=(
                    raw_search_teacher_action_semantic_keys
                    if raw_search_teacher_action_semantic_keys
                    else None
                ),
                teacher_action_codes=(
                    raw_search_teacher_action_codes
                    if raw_search_teacher_action_codes
                    else None
                ),
            )
"""


def patch_lookahead(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, LOOKAHEAD_IMPORT_OLD, LOOKAHEAD_IMPORT_NEW, "semantic_action_key_for_code")
    text = patch_text(text, LOOKAHEAD_DATACLASS_OLD, LOOKAHEAD_DATACLASS_NEW, "action_semantic_keys: tuple[str, ...]")
    text = patch_text(text, LOOKAHEAD_ANNOTATE_OLD, LOOKAHEAD_ANNOTATE_NEW, '"search_teacher_action_semantic_keys"')
    text = patch_text(text, LOOKAHEAD_RETURN_OLD, LOOKAHEAD_RETURN_NEW, "_semantic_key_token_for_code(int(code))")
    if "_semantic_key_token_for_code(" not in text:
        if LOOKAHEAD_HELPER_ANCHOR not in text:
            raise RuntimeError("Could not find helper anchor in lookahead_search.py")
        text = text.replace(LOOKAHEAD_HELPER_ANCHOR, LOOKAHEAD_HELPER_BLOCK + LOOKAHEAD_HELPER_ANCHOR, 1)
    path.write_text(text, encoding="utf-8")


def patch_ppo_training(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = patch_text(text, PPO_IMPORT_OLD, PPO_IMPORT_NEW, "semantic_action_key_for_code")
    if "_align_search_teacher_policy_target(" not in text or "_semantic_key_token_for_code(" not in text:
        if PPO_HELPER_ANCHOR not in text:
            raise RuntimeError("Could not find helper anchor in ppo_training.py")
        text = text.replace(PPO_HELPER_ANCHOR, PPO_HELPER_BLOCK + PPO_HELPER_ANCHOR, 1)
    text = patch_text(text, PPO_SNIPPET_OLD, PPO_SNIPPET_NEW, "raw_search_teacher_action_semantic_keys")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch search teacher alignment to use semantic action keys instead of unstable action codes.")
    parser.add_argument("--lookahead", required=True)
    parser.add_argument("--ppo-training", required=True)
    args = parser.parse_args()

    lookahead = Path(args.lookahead)
    ppo_training = Path(args.ppo_training)

    if not lookahead.exists():
        raise SystemExit(f"lookahead_search.py not found: {lookahead}")
    if not ppo_training.exists():
        raise SystemExit(f"ppo_training.py not found: {ppo_training}")

    patch_lookahead(lookahead)
    patch_ppo_training(ppo_training)
    print(f"patched: {lookahead}")
    print(f"patched: {ppo_training}")


if __name__ == "__main__":
    main()
