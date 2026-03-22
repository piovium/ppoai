
from __future__ import annotations

import argparse
from pathlib import Path


HELPER_BLOCK = """
def _validate_terminal_outcome(full_state, winner: int | None, *, status: str | None, error: str | None) -> tuple[int | None, str | None]:
    players = tuple(getattr(full_state, "players", ()) or ())
    if len(players) != 2:
        if status and status != "FINISHED":
            return None, f"terminal_status={status}"
        return winner, None

    def _all_defeated(player) -> bool:
        characters = tuple(getattr(player, "characters", ()) or ())
        return bool(characters) and all(bool(getattr(ch, "defeated", False)) for ch in characters)

    player0_all_defeated = _all_defeated(players[0])
    player1_all_defeated = _all_defeated(players[1])

    expected_winner: int | None
    if player1_all_defeated and not player0_all_defeated:
        expected_winner = 0
    elif player0_all_defeated and not player1_all_defeated:
        expected_winner = 1
    elif player0_all_defeated and player1_all_defeated:
        expected_winner = None
    else:
        expected_winner = None

    if status and status != "FINISHED":
        return None, f"terminal_status={status}"
    if error:
        return None, f"terminal_error={error}"
    if winner in (0, 1) and expected_winner is None:
        return None, f"inconsistent_terminal_winner={winner}"
    if winner in (0, 1) and expected_winner is not None and int(winner) != int(expected_winner):
        return None, f"mismatched_terminal_winner={winner}_expected={expected_winner}"
    return winner, None
""".strip("\n")


OLD_TERMINAL_BLOCK = """
                terminal_context = DecisionContext(
                    acting_player=-1,
                    request_type=DecisionType.TERMINAL,
                    step_index=step_index_ref["value"],
                    full_state=final_state,
                    legal_low_level_codes=(),
                    legal_low_level_mask=(),
                    legal_high_level_codes=(),
                    high_to_low_map=(),
                    player_view=None,
                    full_state_json=final_state_json,
                    terminal=True,
                    metadata={
                        "matchup": self._matchup.key,
                        "seed": self._seed,
                        "status": game.status().name,
                        "winner": game.winner(),
                        "error": game.error() if game.status().name == "ABORTED" else None,
                    },
                )
""".strip("\n")

NEW_TERMINAL_BLOCK = """
                status_name = game.status().name
                raw_winner = game.winner()
                error_message = game.error() if status_name == "ABORTED" else None
                sanitized_winner, invalid_terminal_reason = _validate_terminal_outcome(
                    final_state,
                    raw_winner,
                    status=status_name,
                    error=error_message,
                )
                if sanitized_winner != final_state.winner:
                    final_state = replace(final_state, winner=sanitized_winner)
                terminal_context = DecisionContext(
                    acting_player=-1,
                    request_type=DecisionType.TERMINAL,
                    step_index=step_index_ref["value"],
                    full_state=final_state,
                    legal_low_level_codes=(),
                    legal_low_level_mask=(),
                    legal_high_level_codes=(),
                    high_to_low_map=(),
                    player_view=None,
                    full_state_json=final_state_json,
                    terminal=True,
                    metadata={
                        "matchup": self._matchup.key,
                        "seed": self._seed,
                        "status": status_name,
                        "winner": sanitized_winner,
                        "raw_winner": raw_winner,
                        "error": error_message,
                        "invalid_terminal": bool(invalid_terminal_reason),
                        "invalid_terminal_reason": invalid_terminal_reason,
                    },
                )
                if invalid_terminal_reason:
                    print(
                        "[terminal-guard] "
                        f"matchup={self._matchup.key} "
                        f"seed={self._seed} "
                        f"reason={invalid_terminal_reason} "
                        f"raw_winner={raw_winner} "
                        f"sanitized_winner={sanitized_winner}",
                        flush=True,
                    )
""".strip("\n")


def patch_env_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "_validate_terminal_outcome(" not in text:
        anchor = "\n\ndef _build_create_param("
        if anchor not in text:
            raise RuntimeError("Could not find _build_create_param anchor in env.py")
        text = text.replace(anchor, "\n\n" + HELPER_BLOCK + "\n\n\ndef _build_create_param(", 1)

    if OLD_TERMINAL_BLOCK in text:
        text = text.replace(OLD_TERMINAL_BLOCK, NEW_TERMINAL_BLOCK, 1)
    elif "invalid_terminal_reason = _validate_terminal_outcome(" not in text:
        raise RuntimeError("Could not find terminal_context block to patch in env.py")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch env.py with terminal-outcome guards.")
    parser.add_argument(
        "--env",
        required=True,
        help="Path to research/world_model/src/gitcg_world_model/env.py",
    )
    args = parser.parse_args()

    env_path = Path(args.env)
    if not env_path.exists():
        raise SystemExit(f"env.py not found: {env_path}")

    patch_env_file(env_path)
    print(f"patched: {env_path}")


if __name__ == "__main__":
    main()
