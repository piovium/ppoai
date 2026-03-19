from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .progress import atomic_write_json


def auto_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def write_episode_analysis(
    path: str | Path,
    *,
    phase: str,
    round_index: int,
    analytics: Any,
) -> None:
    atomic_write_json(
        path,
        {
            "round": round_index,
            "phase": phase,
            **analytics.to_dict(),
        },
    )


def print_episode_analysis(
    *,
    label: str,
    round_index: int,
    analytics_path: str | Path,
    analytics: Any,
) -> None:
    outcome_summary = (
        f"player0_wins={analytics.player0_wins} "
        f"player1_wins={analytics.player1_wins} "
    )
    print(
        f"[{label}] "
        f"round={round_index:04d} "
        f"path={analytics_path} "
        f"episodes={analytics.episodes} "
        f"{outcome_summary}"
        f"draws={analytics.draws} "
        f"truncated={analytics.truncated} "
        f"avg_steps={analytics.average_steps:.2f}",
        flush=True,
    )


def checkpoint_size_mb(path: str | Path) -> float:
    candidate = Path(path)
    if not candidate.exists():
        return 0.0
    return candidate.stat().st_size / (1024.0 * 1024.0)


def cleanup_completed_round_training_temps(
    workspace_path: Path,
    *,
    completed_round: int,
) -> list[str]:
    if completed_round <= 0:
        return []
    cleaned: list[str] = []
    for round_index in range(1, completed_round + 1):
        checkpoint_dir = workspace_path / f"round_{round_index:04d}" / "train" / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if not checkpoint_dir.exists():
            cleaned.append(f"round_{round_index:04d}")
    return cleaned


def prune_old_round_directories(
    workspace_path: Path,
    *,
    current_round: int,
    retain_round_directories: int | None,
    protected_paths: Sequence[str | Path | None] = (),
) -> list[str]:
    if retain_round_directories is None or current_round <= 0:
        return []
    cutoff_round = current_round - retain_round_directories
    if cutoff_round <= 0:
        return []
    protected = tuple(
        Path(path).resolve()
        for path in protected_paths
        if path is not None
    )
    pruned: list[str] = []
    for round_dir in sorted(workspace_path.glob("round_*")):
        if not round_dir.is_dir():
            continue
        try:
            candidate_round = int(round_dir.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        if candidate_round > cutoff_round:
            continue
        resolved_round_dir = round_dir.resolve()
        if any(path_is_within(path, resolved_round_dir) for path in protected):
            continue
        shutil.rmtree(round_dir, ignore_errors=True)
        if not round_dir.exists():
            pruned.append(round_dir.name)
    return pruned


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def release_cuda_memory(*modules: Any) -> None:
    for module in modules:
        if isinstance(module, torch.nn.Module):
            try:
                module.to("cpu")
            except RuntimeError:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_runtime_manifest(path: Path, *, workspace: Path) -> None:
    atomic_write_json(
        path,
        {
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "workspace": str(workspace),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
