from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gitcg_world_model.decks import SMALL_DECK_MATCHUPS
from gitcg_world_model.progress import RunStatus, load_run_status, reconstruct_bootstrap_status, render_progress_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the current local training status.")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-dir")
    parser.add_argument("--episodes-per-matchup", type=int, default=128)
    parser.add_argument("--rollout-workers", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        status = resolve_status(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            bootstrap_dir=Path(args.bootstrap_dir) if args.bootstrap_dir else None,
            episodes_per_matchup=args.episodes_per_matchup,
            worker_count=int(args.rollout_workers or 0),
        )
        if status is None:
            raise SystemExit("no status or reconstructable artifacts found")
        if args.json:
            print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_status(status))
        if not args.watch:
            return
        time.sleep(args.interval)


def resolve_status(
    *,
    output_dir: Path | None,
    bootstrap_dir: Path | None,
    episodes_per_matchup: int,
    worker_count: int,
) -> RunStatus | None:
    if output_dir is not None:
        live_status = load_run_status(output_dir / "event_loop_status.json")
        if live_status is not None:
            return live_status
        summary_path = output_dir / "event_loop_result.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(summary_path.stat().st_mtime))
            return RunStatus(
                phase="complete",
                phase_percent=100.0,
                overall_percent=100.0,
                completed_units=1,
                total_units=1,
                current_task="run finished",
                worker_count=0,
                started_at=timestamp,
                updated_at=timestamp,
                artifacts={
                    "summary_path": str(summary_path),
                    "deployment_checkpoint": str(payload.get("deployment_checkpoint", "")),
                },
                last_error=None,
                overall_completed_units=1,
                overall_total_units=1,
                details={
                    "round_count": int(len(payload.get("rounds", ()))),
                },
            )
        inferred_bootstrap_dir = output_dir / "round_0001" / "bootstrap"
        if inferred_bootstrap_dir.exists():
            bootstrap_dir = inferred_bootstrap_dir
    if bootstrap_dir is not None:
        return reconstruct_bootstrap_status(
            bootstrap_dir=bootstrap_dir,
            expected_shard_names=_expected_bootstrap_shard_names(),
            episodes_per_shard=episodes_per_matchup,
            worker_count=worker_count,
        )
    return None


def render_status(status: RunStatus) -> str:
    return (
        f"{status.phase} {render_progress_bar(status.phase_percent)} "
        f"{status.phase_percent:6.2f}% overall={status.overall_percent:6.2f}% "
        f"units={status.completed_units}/{status.total_units} "
        f"workers={status.worker_count} task={status.current_task or '-'}"
    )


def _expected_bootstrap_shard_names() -> tuple[str, ...]:
    baselines = ("baseline_random", "baseline_heuristic", "baseline_scripted")
    shard_names: list[str] = []
    for matchup in SMALL_DECK_MATCHUPS:
        for baseline in baselines:
            shard_names.append(f"{matchup.key}__{baseline}")
    return tuple(shard_names)


if __name__ == "__main__":
    main()
