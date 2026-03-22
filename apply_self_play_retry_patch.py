
from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPER_BLOCK = """
_SELF_PLAY_INVALID_TERMINAL_RETRIES_ENV = "GITCG_SELF_PLAY_INVALID_TERMINAL_RETRIES"
_SELF_PLAY_INVALID_TERMINAL_RETRIES_DEFAULT = 8
_SELF_PLAY_RETRY_SEED_STRIDE = 1_000_003


def _self_play_retry_limit(job_metadata: dict[str, Any]) -> int:
    if not bool(job_metadata.get("retry_invalid_terminal")):
        return 0
    explicit = job_metadata.get("max_invalid_terminal_retries")
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    raw = os.environ.get(_SELF_PLAY_INVALID_TERMINAL_RETRIES_ENV)
    if raw is not None:
        try:
            return max(0, int(raw.strip()))
        except ValueError:
            pass
    return _SELF_PLAY_INVALID_TERMINAL_RETRIES_DEFAULT


def _all_player_characters_defeated(player: Any) -> bool:
    characters = tuple(getattr(player, "characters", ()) or ())
    return bool(characters) and all(bool(getattr(character, "defeated", False)) for character in characters)


def _episode_requires_self_play_retry(episode: EpisodeRecord) -> tuple[bool, str | None]:
    metadata = dict(episode.metadata or {})
    if bool(metadata.get("invalid_terminal")):
        return True, str(metadata.get("invalid_terminal_reason") or "invalid_terminal")
    if bool(metadata.get("truncated")):
        return True, str(metadata.get("truncation_reason") or "truncated_episode")

    final_state = episode.final_state
    phase = str(getattr(final_state, "phase", "") or "")
    if phase != "game_end":
        return True, f"terminal_phase={phase or 'unknown'}"

    players = tuple(getattr(final_state, "players", ()) or ())
    if len(players) != 2:
        return False, None

    winner = episode.winner
    if winner not in (0, 1):
        return False, None

    player0_all_defeated = _all_player_characters_defeated(players[0])
    player1_all_defeated = _all_player_characters_defeated(players[1])

    if winner == 0 and player1_all_defeated and not player0_all_defeated:
        return False, None
    if winner == 1 and player0_all_defeated and not player1_all_defeated:
        return False, None
    return True, f"inconsistent_final_winner={winner}"
""".strip("\n")


NEW_RUN_EPISODE_JOB = """
def _run_episode_job(
    *,
    config: EnvConfig,
    max_decisions: int | None,
    job: _EpisodeJob,
) -> _EpisodeJobResult:
    requested_seed = job.seed
    retry_budget = _self_play_retry_limit(job.metadata)
    attempt = 0

    while True:
        effective_seed = requested_seed if attempt == 0 else requested_seed + (attempt * _SELF_PLAY_RETRY_SEED_STRIDE)
        episode = _run_episode(
            config=config,
            matchup=job.matchup,
            agent0=job.agent0_factory(effective_seed),
            agent1=job.agent1_factory(effective_seed + 17),
            seed=effective_seed,
            max_decisions=max_decisions,
        )
        merged_metadata = {**episode.metadata, **job.metadata}
        retry_probe_episode = replace(
            episode,
            seed=requested_seed,
            metadata=merged_metadata,
        )
        should_retry, retry_reason = _episode_requires_self_play_retry(retry_probe_episode)

        retry_metadata: dict[str, Any] = {}
        if attempt > 0:
            retry_metadata.update(
                {
                    "requested_seed": requested_seed,
                    "actual_seed": effective_seed,
                    "retry_attempt": attempt,
                    "resampled_episode": True,
                }
            )
        if retry_reason is not None:
            retry_metadata["self_play_retry_reason"] = retry_reason

        final_episode = replace(
            episode,
            seed=requested_seed,
            metadata={**merged_metadata, **retry_metadata},
        )

        if not should_retry:
            return _EpisodeJobResult(job=job, episode=final_episode)

        if attempt >= retry_budget:
            exhausted_episode = replace(
                final_episode,
                metadata={
                    **final_episode.metadata,
                    "retry_exhausted": True,
                    "retry_budget": retry_budget,
                },
            )
            return _EpisodeJobResult(job=job, episode=exhausted_episode)

        print(
            "[self-play-retry] "
            f"matchup={job.matchup.key} "
            f"seed={requested_seed} "
            f"actual_seed={effective_seed} "
            f"attempt={attempt + 1}/{retry_budget} "
            f"reason={retry_reason}",
            flush=True,
        )
        attempt += 1
""".strip("\n")


OLD_METADATA_SNIPPET = """                    metadata={
                        "opponent_kind": "frozen_population",
                        "opponent_policy_id": opponent_entry.policy_id,
                        "opponent_source": opponent_entry.checkpoint_path,
                        "matchup": matchup.key,
                    },"""

NEW_METADATA_SNIPPET = """                    metadata={
                        "opponent_kind": "frozen_population",
                        "opponent_policy_id": opponent_entry.policy_id,
                        "opponent_source": opponent_entry.checkpoint_path,
                        "matchup": matchup.key,
                        "retry_invalid_terminal": True,
                    },"""


def patch_ppo_pipeline(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    helper_anchor = "\n\ndef collect_bootstrap_episodes("
    if "_episode_requires_self_play_retry(" not in text:
        if helper_anchor not in text:
            raise RuntimeError("Could not find helper anchor in ppo_pipeline.py")
        text = text.replace(helper_anchor, "\n\n" + HELPER_BLOCK + "\n\n\ndef collect_bootstrap_episodes(", 1)

    run_episode_job_pattern = re.compile(
        r"def _run_episode_job\(\n(?:.*?\n)*?\n(?=def _execute_episode_jobs\()",
        re.S,
    )
    if not run_episode_job_pattern.search(text):
        raise RuntimeError("Could not find _run_episode_job block in ppo_pipeline.py")
    text = run_episode_job_pattern.sub(NEW_RUN_EPISODE_JOB + "\n\n", text, count=1)

    if OLD_METADATA_SNIPPET in text and '"retry_invalid_terminal": True' not in text:
        text = text.replace(OLD_METADATA_SNIPPET, NEW_METADATA_SNIPPET, 1)

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch ppo_pipeline.py to retry invalid self-play episodes.")
    parser.add_argument(
        "--ppo",
        required=True,
        help="Path to research/world_model/src/gitcg_world_model/ppo_pipeline.py",
    )
    args = parser.parse_args()

    ppo_path = Path(args.ppo)
    if not ppo_path.exists():
        raise SystemExit(f"ppo_pipeline.py not found: {ppo_path}")

    patch_ppo_pipeline(ppo_path)
    print(f"patched: {ppo_path}")


if __name__ == "__main__":
    main()
