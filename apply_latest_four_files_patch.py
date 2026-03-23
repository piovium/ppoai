from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


PPO_HELPERS = '''def _jsonable_semantic_value(value):
    if isinstance(value, tuple):
        return [_jsonable_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_semantic_value(item) for item in value]
    return value


def _semantic_key_token_for_code(action_code: int) -> str:
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
'''

PPO_BUILD_SEQ_BLOCK = '''            raw_search_teacher_policy_target = tuple(
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
'''

LOOKAHEAD_HELPERS = '''def _jsonable_semantic_value(value):
    if isinstance(value, tuple):
        return [_jsonable_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_semantic_value(item) for item in value]
    return value


def _semantic_key_token_for_code(action_code: int) -> str:
    import json

    key = semantic_action_key_for_code(int(action_code))
    return json.dumps(_jsonable_semantic_value(key), ensure_ascii=False, separators=(",", ":"))
'''


def _replace_once(text: str, pattern: str, replacement: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected one replacement for pattern: {pattern}")
    return new_text


def patch_action_adapter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required_markers = [
        "def _dedupe_spec_payloads(",
        "Keep the first payload silently",
        "[action-payload-collision]",
    ]
    for marker in required_markers:
        if marker not in text:
            raise RuntimeError(f"action_adapter.py is missing expected marker: {marker}")
    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def patch_action_hierarchy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required_markers = [
        '("auto_selected_dice", tuple(int(value) for value in sorted(spec.auto_selected_dice)))',
        '("switch_hand_slot_mask", int(spec.switch_hand_slot_mask))',
        '("reroll_dice_mask", int(spec.reroll_dice_mask))',
        '("discarded_hand_slot", int(spec.discarded_hand_slot))',
        '[aggregate-high-policy]',
    ]
    for marker in required_markers:
        if marker not in text:
            raise RuntimeError(f"action_hierarchy.py is missing expected marker: {marker}")
    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def patch_lookahead_search(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "semantic_action_key_for_code" not in text:
        text = text.replace(
            "from .action_hierarchy import (\n    match_low_level_code_index,\n    try_low_level_kind_for_code,\n)\n",
            "from .action_hierarchy import (\n    match_low_level_code_index,\n    semantic_action_key_for_code,\n    try_low_level_kind_for_code,\n)\n",
        )

    if "action_semantic_keys: tuple[str, ...]" not in text:
        text = text.replace(
            "@dataclass(frozen=True)\nclass SearchTeacherTarget:\n    policy_target: tuple[float, ...]\n    value_target: float\n    weight: float\n",
            "@dataclass(frozen=True)\nclass SearchTeacherTarget:\n    policy_target: tuple[float, ...]\n    action_semantic_keys: tuple[str, ...]\n    value_target: float\n    weight: float\n",
        )

    if 'metadata["search_teacher_action_semantic_keys"]' not in text:
        text = text.replace(
            '        metadata["search_teacher_policy"] = list(target.policy_target)\n        metadata["search_teacher_value"] = float(target.value_target)\n        metadata["search_teacher_weight"] = float(target.weight)\n',
            '        metadata["search_teacher_policy"] = list(target.policy_target)\n        metadata["search_teacher_action_semantic_keys"] = list(target.action_semantic_keys)\n        metadata["search_teacher_value"] = float(target.value_target)\n        metadata["search_teacher_weight"] = float(target.weight)\n',
        )

    if "_semantic_key_token_for_code(" not in text:
        anchor = "\ndef _softmax_tuple(logits: Sequence[float]) -> tuple[float, ...]:\n"
        if anchor not in text:
            raise RuntimeError("Could not find _softmax_tuple anchor in lookahead_search.py")
        text = text.replace(anchor, "\n" + LOOKAHEAD_HELPERS + "\n\n" + anchor.lstrip("\n"), 1)

    if "action_semantic_keys=tuple(" not in text:
        text = text.replace(
            "    return SearchTeacherTarget(\n        policy_target=policy_target,\n        value_target=float(best_score),\n        weight=float(teacher_weight),\n    )\n",
            '    return SearchTeacherTarget(\n        policy_target=policy_target,\n        action_semantic_keys=tuple(\n            _semantic_key_token_for_code(int(code))\n            for code in root_context.legal_low_level_codes\n        ),\n        value_target=float(best_score),\n        weight=float(teacher_weight),\n    )\n',
        )

    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def patch_ppo_training(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "semantic_action_key_for_code" not in text:
        text = text.replace(
            "from .action_hierarchy import (\n    aggregate_high_policy,\n    high_action_vocab_size,\n    legal_low_level_specs,\n    selected_low_level_code,\n)\n",
            "from .action_hierarchy import (\n    aggregate_high_policy,\n    high_action_vocab_size,\n    legal_low_level_specs,\n    selected_low_level_code,\n    semantic_action_key_for_code,\n)\n",
        )

    dataclass_anchor = "\n\n@dataclass(frozen=True)\nclass PpoTrainSample:\n"
    if dataclass_anchor not in text:
        raise RuntimeError("Could not find PpoTrainSample anchor in ppo_training.py")

    import_anchor = "from .schema import DecisionContext, EnvConfig, EpisodeRecord, TrajectoryStep\n"
    if import_anchor not in text:
        raise RuntimeError("Could not find schema import anchor in ppo_training.py")

    prefix, suffix = text.split(dataclass_anchor, 1)
    import_prefix, _ = prefix.split(import_anchor, 1)
    new_prefix = import_prefix + import_anchor + "\n\n" + PPO_HELPERS + "\n"
    text = new_prefix + dataclass_anchor.lstrip("\n") + suffix

    seq_pattern = r'''            raw_search_teacher_policy_target = tuple\(
                float\(value\) for value in transition\.source_step\.metadata\.get\("search_teacher_policy", \(\)\)
            \)
(?:            .*?\n)*?            search_teacher_policy_target = _align_search_teacher_policy_target\(
                context=transition_context,
                policy=raw_search_teacher_policy_target,
(?:                .*?\n)*?            \)
'''
    text = _replace_once(text, seq_pattern, PPO_BUILD_SEQ_BLOCK)

    if "raw_search_teacher_action_codes" in text:
        text = text.replace(
            '            raw_search_teacher_action_codes = tuple(\n                int(value) for value in transition.source_step.metadata.get("search_teacher_action_codes", ())\n            )\n',
            "",
        )
        text = text.replace(
            "                teacher_action_codes=(\n                    raw_search_teacher_action_codes\n                    if raw_search_teacher_action_codes\n                    else None\n                ),\n",
            "",
        )

    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate the latest 4 files into one self-consistent version.")
    parser.add_argument("--action-adapter", required=True)
    parser.add_argument("--action-hierarchy", required=True)
    parser.add_argument("--ppo-training", required=True)
    parser.add_argument("--lookahead-search", required=True)
    args = parser.parse_args()

    files = {
        "action_adapter.py": Path(args.action_adapter),
        "action_hierarchy.py": Path(args.action_hierarchy),
        "ppo_training.py": Path(args.ppo_training),
        "lookahead_search.py": Path(args.lookahead_search),
    }
    for name, path in files.items():
        if not path.exists():
            raise SystemExit(f"{name} not found: {path}")

    patch_action_adapter(files["action_adapter.py"])
    patch_action_hierarchy(files["action_hierarchy.py"])
    patch_lookahead_search(files["lookahead_search.py"])
    patch_ppo_training(files["ppo_training.py"])

    for path in files.values():
        ast.parse(path.read_text(encoding="utf-8"))

    for name, path in files.items():
        print(f"patched: {name} -> {path}")


if __name__ == "__main__":
    main()
