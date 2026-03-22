
from __future__ import annotations

import argparse
from pathlib import Path


OLD_SNIPPET = """
def semantic_action_key_for_spec(spec: LowLevelActionSpec) -> tuple[Any, ...]:
    return (
        ("request_type", spec.request_type.value),
        ("kind", spec.kind.value),
        ("subject_definition_id", int(spec.subject_definition_id)),
        (
            "skill_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_USE_SKILL else 0,
        ),
        (
            "card_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_PLAY_CARD else 0,
        ),
        ("candidate_definition_id", int(spec.select_card_definition_id)),
        ("removed_card_definition_id", int(spec.discarded_card_definition_id)),
        (
            "target_slots",
            tuple((target.owner, target.zone, int(target.index)) for target in spec.target_slots),
        ),
        ("used_dice", tuple(int(value) for value in sorted(spec.used_dice))),
        ("target_dice", int(spec.target_dice)),
    )
""".strip("\n")

NEW_SNIPPET = """
def semantic_action_key_for_spec(spec: LowLevelActionSpec) -> tuple[Any, ...]:
    return (
        ("request_type", spec.request_type.value),
        ("kind", spec.kind.value),
        ("subject_definition_id", int(spec.subject_definition_id)),
        (
            "skill_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_USE_SKILL else 0,
        ),
        (
            "card_definition_id",
            int(spec.subject_definition_id) if spec.kind == OptionKind.ACTION_PLAY_CARD else 0,
        ),
        ("candidate_definition_id", int(spec.select_card_definition_id)),
        ("removed_card_definition_id", int(spec.discarded_card_definition_id)),
        (
            "target_slots",
            tuple((target.owner, target.zone, int(target.index)) for target in spec.target_slots),
        ),
        ("used_dice", tuple(int(value) for value in sorted(spec.used_dice))),
        ("auto_selected_dice", tuple(int(value) for value in sorted(spec.auto_selected_dice))),
        ("choose_active_slot", int(spec.choose_active_slot)),
        ("switch_hand_slot_mask", int(spec.switch_hand_slot_mask)),
        ("reroll_dice_mask", int(spec.reroll_dice_mask)),
        ("discarded_hand_slot", int(spec.discarded_hand_slot)),
        ("target_dice", int(spec.target_dice)),
    )
""".strip("\n")


def patch_action_hierarchy(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if OLD_SNIPPET in text:
        text = text.replace(OLD_SNIPPET, NEW_SNIPPET, 1)
    elif '"auto_selected_dice"' not in text or '"switch_hand_slot_mask"' not in text:
        raise RuntimeError("Could not patch semantic_action_key_for_spec in action_hierarchy.py")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch action_hierarchy.py to make semantic action keys collision-resistant.")
    parser.add_argument(
        "--action-hierarchy",
        required=True,
        help="Path to research/world_model/src/gitcg_world_model/action_hierarchy.py",
    )
    args = parser.parse_args()

    target = Path(args.action_hierarchy)
    if not target.exists():
        raise SystemExit(f"action_hierarchy.py not found: {target}")

    patch_action_hierarchy(target)
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
