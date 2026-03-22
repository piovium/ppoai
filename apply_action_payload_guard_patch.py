
from __future__ import annotations

import argparse
from pathlib import Path


HELPER_BLOCK = """
def _spec_payload_dedup_key(item: _SpecPayload) -> tuple[Any, ...]:
    spec = item.spec
    return (
        spec.request_type.value,
        spec.kind.value,
        int(spec.subject_definition_id),
        tuple((target.owner, target.zone, int(target.index)) for target in spec.target_slots),
        tuple(int(value) for value in sorted(spec.used_dice)),
        tuple(int(value) for value in sorted(spec.auto_selected_dice)),
        int(spec.choose_active_slot),
        int(spec.select_card_definition_id),
        int(spec.switch_hand_slot_mask),
        int(spec.reroll_dice_mask),
        int(spec.discarded_hand_slot),
        int(spec.discarded_card_definition_id),
        int(spec.target_dice),
    )


def _dedupe_spec_payloads(spec_payloads: list[_SpecPayload]) -> tuple[_SpecPayload, ...]:
    unique: list[_SpecPayload] = []
    seen: dict[tuple[Any, ...], _SpecPayload] = {}
    for item in spec_payloads:
        key = _spec_payload_dedup_key(item)
        existing = seen.get(key)
        if existing is not None:
            if dict(existing.payload) != dict(item.payload):
                print(
                    "[action-dedupe] "
                    f"label={item.spec.label} "
                    f"first_payload={existing.payload} "
                    f"second_payload={item.payload}",
                    flush=True,
                )
            continue
        seen[key] = item
        unique.append(item)
    return tuple(unique)
""".strip("\n")

OLD_BUILD_SNIPPET = """
    spec_payloads, request_payload = builder(
        request=request,
        acting_player=acting_player,
        full_state=full_state,
        player_view=player_view,
    )
""".strip("\n")

NEW_BUILD_SNIPPET = """
    spec_payloads, request_payload = builder(
        request=request,
        acting_player=acting_player,
        full_state=full_state,
        player_view=player_view,
    )
    spec_payloads = list(_dedupe_spec_payloads(spec_payloads))
""".strip("\n")

OLD_PAYLOAD_SNIPPET = """
    payload_by_low_level_code = {
        int(spec.action_code): dict(item.payload)
        for spec, item in zip(materialized_specs, spec_payloads, strict=False)
    }
    label_by_low_level_code = {
        int(spec.action_code): str(spec.label)
        for spec in materialized_specs
    }
""".strip("\n")

NEW_PAYLOAD_SNIPPET = """
    payload_by_low_level_code: dict[int, dict[str, Any]] = {}
    label_by_low_level_code: dict[int, str] = {}
    for spec, item in zip(materialized_specs, spec_payloads, strict=False):
        action_code = int(spec.action_code)
        payload = dict(item.payload)
        existing_payload = payload_by_low_level_code.get(action_code)
        if existing_payload is not None:
            if existing_payload != payload:
                print(
                    "[action-payload-collision] "
                    f"action_code={action_code} "
                    f"label={spec.label} "
                    f"kept_payload={existing_payload} "
                    f"dropped_payload={payload}",
                    flush=True,
                )
            continue
        payload_by_low_level_code[action_code] = payload
        label_by_low_level_code[action_code] = str(spec.label)
""".strip("\n")


def patch_action_adapter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "_dedupe_spec_payloads(" not in text:
        anchor = "\n\ndef encode_choice("
        if anchor not in text:
            raise RuntimeError("Could not find encode_choice anchor in action_adapter.py")
        text = text.replace(anchor, "\n\n" + HELPER_BLOCK + "\n\n\ndef encode_choice(", 1)

    if OLD_BUILD_SNIPPET in text:
        text = text.replace(OLD_BUILD_SNIPPET, NEW_BUILD_SNIPPET, 1)
    elif "spec_payloads = list(_dedupe_spec_payloads(spec_payloads))" not in text:
        raise RuntimeError("Could not patch spec_payload dedupe block")

    if OLD_PAYLOAD_SNIPPET in text:
        text = text.replace(OLD_PAYLOAD_SNIPPET, NEW_PAYLOAD_SNIPPET, 1)
    elif "[action-payload-collision]" not in text:
        raise RuntimeError("Could not patch payload collision block")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch action_adapter.py to dedupe colliding semantic actions.")
    parser.add_argument(
        "--action-adapter",
        required=True,
        help="Path to research/world_model/src/gitcg_world_model/action_adapter.py",
    )
    args = parser.parse_args()

    target = Path(args.action_adapter)
    if not target.exists():
        raise SystemExit(f"action_adapter.py not found: {target}")

    patch_action_adapter(target)
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
