
from __future__ import annotations

import argparse
from pathlib import Path


OLD_BLOCK = """        if existing is not None:
            if dict(existing.payload) != dict(item.payload):
                print(
                    "[action-dedupe] "
                    f"label={item.spec.label} "
                    f"first_payload={existing.payload} "
                    f"second_payload={item.payload}",
                    flush=True,
                )
            continue
"""

NEW_BLOCK = """        if existing is not None:
            # Benign duplicate raw options are common (for example identical card
            # copies / equivalent tuning branches). Keep the first payload silently
            # to avoid flooding stdout with index-only dedupe noise.
            continue
"""

ALT_BLOCK = """        if existing is not None:
            if dict(existing.payload) != dict(item.payload):
                if bool(int(os.environ.get("GITCG_ACTION_DEDUPE_TRACE", "0") or "0")):
                    print(
                        "[action-dedupe] "
                        f"label={item.spec.label} "
                        f"first_payload={existing.payload} "
                        f"second_payload={item.payload}",
                        flush=True,
                    )
            continue
"""


def patch_action_adapter(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif ALT_BLOCK in text:
        text = text.replace(ALT_BLOCK, NEW_BLOCK, 1)
    elif "[action-dedupe]" in text:
        raise RuntimeError("Found action-dedupe logging, but block shape did not match expected patch target.")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Silence noisy [action-dedupe] logs in action_adapter.py")
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
