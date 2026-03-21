
from __future__ import annotations

from pathlib import Path


def read_text_keep_newline(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw, newline


def write_text_keep_newline(path: Path, text: str, newline: str) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        normalized = normalized.replace("\n", newline)
    path.write_text(normalized, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing snippet for {label}")
    return text.replace(old, new, 1)


def main() -> None:
    repo_root = Path(".").resolve()
    teacher_path = repo_root / "research/world_model/src/gitcg_world_model/lookahead_search.py"

    path = teacher_path
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")


    # lookahead_search.py: avoid hard dependency on reverse lookup from historical low-level codes
    teacher_text, teacher_nl = read_text_keep_newline(teacher_path)
    teacher_backup = teacher_path.with_suffix(teacher_path.suffix + ".bak_alignment_teacher")
    if not teacher_backup.exists():
        teacher_backup.write_text(teacher_text, encoding="utf-8")

    old_import = """from .action_hierarchy import (
    low_level_kind_for_code,
    match_low_level_code_index,
)
"""
    new_import = """from .action_hierarchy import (
    match_low_level_code_index,
    try_low_level_kind_for_code,
)
"""
    if old_import in teacher_text:
        teacher_text = replace_once(teacher_text, old_import, new_import, "lookahead import fix")

    old_is_search = """def _is_search_state(step: TrajectoryStep, *, config: LookaheadSearchConfig) -> bool:
    if step.request_type.name != "ACTION":
        return False
    if step.player_view is None or step.full_state_json_before is None:
        return False
    if step.pre_state.round_number >= config.round_threshold:
        return True
    if len(step.legal_low_level_codes) >= config.legal_option_threshold:
        return True
    productive_non_end = any(
        low_level_kind_for_code(int(action_code)).value != "action_declare_end"
        for action_code in step.legal_low_level_codes
    )
    has_end = any(
        low_level_kind_for_code(int(action_code)).value == "action_declare_end"
        for action_code in step.legal_low_level_codes
    )
    return productive_non_end and has_end
"""
    new_is_search = """def _legal_action_kind_values_for_step(step: TrajectoryStep) -> tuple[str, ...]:
    if step.legal_low_level_specs:
        return tuple(spec.kind.value for spec in step.legal_low_level_specs)
    kinds: list[str] = []
    for action_code in step.legal_low_level_codes:
        kind = try_low_level_kind_for_code(int(action_code))
        if kind is not None:
            kinds.append(kind.value)
    return tuple(kinds)


def _is_search_state(step: TrajectoryStep, *, config: LookaheadSearchConfig) -> bool:
    if step.request_type.name != "ACTION":
        return False
    if step.player_view is None or step.full_state_json_before is None:
        return False
    if step.pre_state.round_number >= config.round_threshold:
        return True
    if len(step.legal_low_level_codes) >= config.legal_option_threshold:
        return True
    legal_kind_values = _legal_action_kind_values_for_step(step)
    if not legal_kind_values:
        return False
    productive_non_end = any(kind_value != "action_declare_end" for kind_value in legal_kind_values)
    has_end = any(kind_value == "action_declare_end" for kind_value in legal_kind_values)
    return productive_non_end and has_end
"""
    teacher_text = replace_once(teacher_text, old_is_search, new_is_search, "lookahead _is_search_state safe kinds")
    write_text_keep_newline(teacher_path, teacher_text, teacher_nl)

    print("Patched:")
    print(f"  - {teacher_path}")
    print("Backups:")
    print(f"  - {teacher_backup}")


if __name__ == "__main__":
    main()
