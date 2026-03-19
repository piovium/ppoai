from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RunStatus:
    phase: str
    phase_percent: float
    overall_percent: float
    completed_units: int
    total_units: int
    current_task: str | None
    worker_count: int
    started_at: str
    updated_at: str
    artifacts: dict[str, str]
    last_error: str | None
    overall_completed_units: int
    overall_total_units: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressTracker:
    def __init__(
        self,
        *,
        status_path: str | Path,
        event_log_path: str | Path,
        overall_total_units: int,
        worker_count: int,
        enable_print: bool = True,
    ):
        self._status_path = Path(status_path)
        self._event_log_path = Path(event_log_path)
        self._overall_total_units = max(1, overall_total_units)
        self._worker_count = worker_count
        self._enable_print = enable_print
        self._started_at = utc_now_iso()
        self._last_status: RunStatus | None = None

    @property
    def status_path(self) -> Path:
        return self._status_path

    @property
    def event_log_path(self) -> Path:
        return self._event_log_path

    @property
    def last_status(self) -> RunStatus | None:
        return self._last_status

    def start_phase(
        self,
        phase: str,
        *,
        total_units: int,
        current_task: str | None = None,
        artifacts: Mapping[str, str] | None = None,
        overall_completed_units: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> RunStatus:
        return self._write(
            event_type="phase_started",
            phase=phase,
            completed_units=0,
            total_units=total_units,
            current_task=current_task,
            artifacts=artifacts,
            overall_completed_units=overall_completed_units,
            details=details,
        )

    def update_phase(
        self,
        phase: str,
        *,
        completed_units: int,
        total_units: int,
        current_task: str | None = None,
        artifacts: Mapping[str, str] | None = None,
        overall_completed_units: int = 0,
        details: Mapping[str, Any] | None = None,
        last_error: str | None = None,
    ) -> RunStatus:
        return self._write(
            event_type="phase_updated",
            phase=phase,
            completed_units=completed_units,
            total_units=total_units,
            current_task=current_task,
            artifacts=artifacts,
            overall_completed_units=overall_completed_units,
            details=details,
            last_error=last_error,
        )

    def complete_phase(
        self,
        phase: str,
        *,
        total_units: int,
        current_task: str | None = None,
        artifacts: Mapping[str, str] | None = None,
        overall_completed_units: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> RunStatus:
        return self._write(
            event_type="phase_completed",
            phase=phase,
            completed_units=total_units,
            total_units=total_units,
            current_task=current_task,
            artifacts=artifacts,
            overall_completed_units=overall_completed_units,
            details=details,
        )

    def fail(
        self,
        phase: str,
        *,
        completed_units: int,
        total_units: int,
        error: str,
        current_task: str | None = None,
        artifacts: Mapping[str, str] | None = None,
        overall_completed_units: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> RunStatus:
        return self._write(
            event_type="phase_failed",
            phase=phase,
            completed_units=completed_units,
            total_units=total_units,
            current_task=current_task,
            artifacts=artifacts,
            overall_completed_units=overall_completed_units,
            details=details,
            last_error=error,
        )

    def _write(
        self,
        *,
        event_type: str,
        phase: str,
        completed_units: int,
        total_units: int,
        current_task: str | None,
        artifacts: Mapping[str, str] | None,
        overall_completed_units: int,
        details: Mapping[str, Any] | None,
        last_error: str | None = None,
    ) -> RunStatus:
        total_units = max(1, total_units)
        completed_units = min(max(0, completed_units), total_units)
        overall_completed_units = min(max(0, overall_completed_units), self._overall_total_units)
        status = RunStatus(
            phase=phase,
            phase_percent=(completed_units / total_units) * 100.0,
            overall_percent=(overall_completed_units / self._overall_total_units) * 100.0,
            completed_units=completed_units,
            total_units=total_units,
            current_task=current_task,
            worker_count=self._worker_count,
            started_at=self._started_at,
            updated_at=utc_now_iso(),
            artifacts=dict(artifacts or {}),
            last_error=last_error,
            overall_completed_units=overall_completed_units,
            overall_total_units=self._overall_total_units,
            details=dict(details or {}),
        )
        atomic_write_json(self._status_path, status.to_dict())
        append_jsonl(
            self._event_log_path,
            {
                "event_type": event_type,
                "timestamp": status.updated_at,
                "status": status.to_dict(),
            },
        )
        if self._enable_print:
            print(format_status_line(status), flush=True)
        self._last_status = status
        return status


def load_run_status(path: str | Path) -> RunStatus | None:
    status_path = Path(path)
    if not status_path.exists():
        return None
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    return RunStatus(
        phase=str(payload["phase"]),
        phase_percent=float(payload["phase_percent"]),
        overall_percent=float(payload["overall_percent"]),
        completed_units=int(payload["completed_units"]),
        total_units=int(payload["total_units"]),
        current_task=payload.get("current_task"),
        worker_count=int(payload["worker_count"]),
        started_at=str(payload["started_at"]),
        updated_at=str(payload["updated_at"]),
        artifacts={str(key): str(value) for key, value in payload.get("artifacts", {}).items()},
        last_error=payload.get("last_error"),
        overall_completed_units=int(payload["overall_completed_units"]),
        overall_total_units=int(payload["overall_total_units"]),
        details=dict(payload.get("details", {})),
    )


def reconstruct_bootstrap_status(
    *,
    bootstrap_dir: str | Path,
    expected_shard_names: Sequence[str],
    episodes_per_shard: int,
    worker_count: int = 0,
) -> RunStatus | None:
    directory = Path(bootstrap_dir)
    if not directory.exists():
        return None
    total_shards = len(expected_shard_names)
    if total_shards == 0:
        return None
    completed_shards = 0
    for shard_name in expected_shard_names:
        if (directory / f"{shard_name}.jsonl").exists() and (directory / f"{shard_name}.meta.json").exists():
            completed_shards += 1
    total_units = total_shards * episodes_per_shard
    completed_units = completed_shards * episodes_per_shard
    if completed_units == 0 and total_units > 0:
        current_task = "waiting_for_first_shard"
    else:
        current_task = f"{completed_shards}/{total_shards} shards complete"
    timestamp = utc_now_iso()
    return RunStatus(
        phase="bootstrap_rollout",
        phase_percent=(completed_units / total_units) * 100.0 if total_units else 0.0,
        overall_percent=(completed_units / total_units) * 100.0 if total_units else 0.0,
        completed_units=completed_units,
        total_units=total_units,
        current_task=current_task,
        worker_count=worker_count,
        started_at=timestamp,
        updated_at=timestamp,
        artifacts={"bootstrap_dir": str(directory)},
        last_error=None,
        overall_completed_units=completed_units,
        overall_total_units=total_units,
        details={
            "completed_shards": completed_shards,
            "total_shards": total_shards,
            "episodes_per_shard": episodes_per_shard,
        },
    )


def format_status_line(status: RunStatus) -> str:
    return (
        f"[{status.phase}] "
        f"phase={status.phase_percent:6.2f}% "
        f"overall={status.overall_percent:6.2f}% "
        f"units={status.completed_units}/{status.total_units} "
        f"workers={status.worker_count} "
        f"task={status.current_task or '-'}"
    )


def render_progress_bar(percent: float, *, width: int = 24) -> str:
    bounded = max(0.0, min(100.0, percent))
    filled = int(round((bounded / 100.0) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(20):
        try:
            temp_path.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True))
        handle.write("\n")
