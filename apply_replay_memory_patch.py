from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"patch anchor not found for {label}")
    return text.replace(old, new, 1)


def patch_replay(text: str) -> str:
    pattern = r'def load_episode_records\(path: str \| Path\) -> list\[EpisodeRecord\]:\n    return list\(iter_episode_records\(path\)\)'
    replacement = """def load_episode_records(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[EpisodeRecord]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    records: list[EpisodeRecord] = []
    for index, episode in enumerate(iter_episode_records(path), start=1):
        records.append(episode)
        if limit is not None and index >= limit:
            break
    return records"""
    patched, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError("failed to patch replay.load_episode_records")
    ast.parse(patched)
    return patched


def patch_ppo_pipeline(text: str) -> str:
    patched = text

    patched = _replace_once(
        patched,
        "import json\nimport math\nimport random\nimport shutil\nimport time\n",
        "import ctypes\nimport gc\nimport json\nimport math\nimport os\nimport random\nimport shutil\nimport time\n",
        label="ppo imports",
    )
    patched = _replace_once(
        patched,
        "from .replay import append_episode_records, load_episode_records",
        "from .replay import append_episode_records, iter_episode_records, load_episode_records",
        label="ppo replay import",
    )

    helper_block = """

_REPLAY_MEMORY_CHECK_INTERVAL = 64
_REPLAY_DEFAULT_MAX_PER_FILE = 768
_REPLAY_DEFAULT_MAX_TOTAL = 2048
_REPLAY_ENV_MAX_PER_FILE = "GITCG_REPLAY_MAX_PER_FILE"
_REPLAY_ENV_MAX_TOTAL = "GITCG_REPLAY_MAX_TOTAL"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _process_rss_bytes() -> int | None:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        process_handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize)

    try:
        statm_path = Path("/proc/self/statm")
        if statm_path.exists():
            fields = statm_path.read_text(encoding="utf-8").split()
            if len(fields) >= 2:
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(fields[1]) * int(page_size)
    except Exception:
        pass
    return None


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if not ok:
            return None
        return int(status.ullAvailPhys)

    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (ValueError, OSError, AttributeError):
        return None


def _select_replay_load_limits() -> tuple[int, int]:
    override_per_file = _env_int(_REPLAY_ENV_MAX_PER_FILE)
    override_total = _env_int(_REPLAY_ENV_MAX_TOTAL)
    if override_per_file is not None or override_total is not None:
        max_per_file = max(
            0,
            override_per_file if override_per_file is not None else _REPLAY_DEFAULT_MAX_PER_FILE,
        )
        max_total = max(
            0,
            override_total if override_total is not None else _REPLAY_DEFAULT_MAX_TOTAL,
        )
        return max_per_file, max_total

    gib = 1024 ** 3
    rss = _process_rss_bytes()
    available = _available_memory_bytes()

    if available is not None and available < (2 * gib):
        return 64, 256
    if rss is not None and rss >= (28 * gib):
        return 64, 256
    if available is not None and available < (4 * gib):
        return 128, 512
    if rss is not None and rss >= (24 * gib):
        return 128, 512
    if available is not None and available < (8 * gib):
        return 256, 1024
    if rss is not None and rss >= (20 * gib):
        return 256, 1024
    if available is not None and available < (12 * gib):
        return 512, 1536
    if rss is not None and rss >= (16 * gib):
        return 512, 1536
    return _REPLAY_DEFAULT_MAX_PER_FILE, _REPLAY_DEFAULT_MAX_TOTAL


def _replay_memory_pressure_hard() -> bool:
    gib = 1024 ** 3
    rss = _process_rss_bytes()
    available = _available_memory_bytes()
    if available is not None and available < int(1.5 * gib):
        return True
    if rss is not None and rss >= (29 * gib):
        return True
    return False


def _collect_recent_replay_paths(
    workspace_path: Path,
    *,
    current_round: int,
    replay_round_window: int,
    include_br: bool,
) -> list[Path]:
    paths: list[Path] = []
    round_start = max(1, current_round - replay_round_window)
    for round_index in range(current_round - 1, round_start - 1, -1):
        round_dir = workspace_path / f"round_{round_index:04d}"
        primary_path = round_dir / "self_play.jsonl"
        if primary_path.exists():
            paths.append(primary_path)
        if include_br:
            br_path = round_dir / "br_self_play.jsonl"
            if br_path.exists():
                paths.append(br_path)
    return paths


def _load_replay_episodes_with_budget(
    paths: Sequence[Path],
    *,
    label: str,
) -> tuple[EpisodeRecord, ...]:
    if not paths:
        return ()

    gc.collect()
    max_per_file, max_total = _select_replay_load_limits()
    if max_per_file <= 0 or max_total <= 0:
        return ()

    replay_episodes: list[EpisodeRecord] = []
    stop_reason: str | None = None
    loaded_paths = 0

    for path in paths:
        remaining_total = max_total - len(replay_episodes)
        if remaining_total <= 0:
            stop_reason = "total_budget"
            break

        dynamic_per_file, dynamic_total = _select_replay_load_limits()
        remaining_dynamic_total = max(0, dynamic_total - len(replay_episodes))
        effective_per_file = min(
            max_per_file,
            dynamic_per_file,
            remaining_total,
            remaining_dynamic_total,
        )
        if effective_per_file <= 0:
            stop_reason = "dynamic_budget"
            break

        loaded_from_path = 0
        for episode in iter_episode_records(path):
            replay_episodes.append(episode)
            loaded_from_path += 1

            if loaded_from_path >= effective_per_file:
                stop_reason = "per_file_budget"
                break
            if len(replay_episodes) >= max_total:
                stop_reason = "total_budget"
                break
            if (
                loaded_from_path % _REPLAY_MEMORY_CHECK_INTERVAL == 0
                and _replay_memory_pressure_hard()
            ):
                stop_reason = "memory_pressure"
                break

        loaded_paths += 1
        if stop_reason in {"total_budget", "dynamic_budget", "memory_pressure"}:
            break
        stop_reason = None

    if stop_reason is not None or loaded_paths < len(paths):
        print(
            "[replay-budget] "
            f"label={label} "
            f"episodes={len(replay_episodes)} "
            f"files={loaded_paths}/{len(paths)} "
            f"max_per_file={max_per_file} "
            f"max_total={max_total} "
            f"stop={stop_reason or 'completed'}",
            flush=True,
        )

    return tuple(replay_episodes)


def _compose_replay_buffer(
    *episode_groups: Sequence[EpisodeRecord],
    label: str,
) -> tuple[EpisodeRecord, ...]:
    _, max_total = _select_replay_load_limits()
    if max_total <= 0:
        return ()

    combined: list[EpisodeRecord] = []
    total_requested = 0
    for group in episode_groups:
        total_requested += len(group)
        remaining = max_total - len(combined)
        if remaining <= 0:
            break
        if len(group) <= remaining:
            combined.extend(group)
        else:
            combined.extend(group[:remaining])
            break

    if len(combined) < total_requested:
        print(
            "[replay-budget] "
            f"label={label} "
            f"episodes={len(combined)} "
            f"requested={total_requested} "
            f"max_total={max_total} "
            "stop=compose_budget",
            flush=True,
        )

    return tuple(combined)
"""
    patched = _replace_once(
        patched,
        "def _load_recent_replay_episodes(\n",
        helper_block + "\n" + "def _load_recent_replay_episodes(\n",
        label="replay helper insertion",
    )

    old_recent = """def _load_recent_replay_episodes(
    workspace_path: Path,
    *,
    current_round: int,
    replay_round_window: int = 3,
) -> tuple[EpisodeRecord, ...]:
    replay_episodes: list[EpisodeRecord] = []
    round_start = max(1, current_round - replay_round_window)
    for round_index in range(round_start, current_round):
        self_play_path = workspace_path / f"round_{round_index:04d}" / "self_play.jsonl"
        if not self_play_path.exists():
            continue
        replay_episodes.extend(load_episode_records(self_play_path))
    return tuple(replay_episodes)


def _load_recent_p2sro_replay_episodes(
    workspace_path: Path,
    *,
    current_round: int,
    replay_round_window: int = 3,
) -> tuple[EpisodeRecord, ...]:
    replay_episodes: list[EpisodeRecord] = []
    round_start = max(1, current_round - replay_round_window)
    for round_index in range(round_start, current_round):
        for relative_path in ("self_play.jsonl", "br_self_play.jsonl"):
            path = workspace_path / f"round_{round_index:04d}" / relative_path
            if path.exists():
                replay_episodes.extend(load_episode_records(path))
    return tuple(replay_episodes)"""
    new_recent = """def _load_recent_replay_episodes(
    workspace_path: Path,
    *,
    current_round: int,
    replay_round_window: int = 3,
) -> tuple[EpisodeRecord, ...]:
    return _load_replay_episodes_with_budget(
        _collect_recent_replay_paths(
            workspace_path,
            current_round=current_round,
            replay_round_window=replay_round_window,
            include_br=False,
        ),
        label="recent_replay",
    )


def _load_recent_p2sro_replay_episodes(
    workspace_path: Path,
    *,
    current_round: int,
    replay_round_window: int = 3,
) -> tuple[EpisodeRecord, ...]:
    return _load_replay_episodes_with_budget(
        _collect_recent_replay_paths(
            workspace_path,
            current_round=current_round,
            replay_round_window=replay_round_window,
            include_br=True,
        ),
        label="recent_p2sro_replay",
    )"""
    patched = _replace_once(patched, old_recent, new_recent, label="recent replay loaders")

    patched = _replace_once(
        patched,
        '        analysis_path = round_dir / "analysis.json"\n',
        '        analysis_path = round_dir / "analysis.json"\n'
        '        bootstrap_episodes: tuple[EpisodeRecord, ...] = ()\n'
        '        self_play_episodes: tuple[EpisodeRecord, ...] = ()\n'
        '        replay_episodes: tuple[EpisodeRecord, ...] = ()\n'
        '        br_self_play_episodes: tuple[EpisodeRecord, ...] = ()\n'
        '        br_replay_episodes: tuple[EpisodeRecord, ...] = ()\n',
        label="round local episode buffers",
    )

    patched = _replace_once(
        patched,
        '            br_replay_episodes = tuple(replay_episodes) + tuple(self_play_episodes)\n',
        '            br_replay_episodes = _compose_replay_buffer(\n'
        '                self_play_episodes,\n'
        '                replay_episodes,\n'
        '                label="br_train_replay",\n'
        '            )\n',
        label="br replay composition",
    )

    patched = _replace_once(
        patched,
        '        cleanup_completed_round_training_temps(workspace_path, completed_round=round_index - 1)\n',
        '        bootstrap_episodes = ()\n'
        '        self_play_episodes = ()\n'
        '        replay_episodes = ()\n'
        '        br_self_play_episodes = ()\n'
        '        br_replay_episodes = ()\n'
        '        gc.collect()\n'
        '        cleanup_completed_round_training_temps(workspace_path, completed_round=round_index - 1)\n',
        label="end-of-round gc cleanup",
    )

    ast.parse(patched)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch replay loading to use dynamic memory budgets.")
    parser.add_argument("--replay", required=True, help="Path to replay.py")
    parser.add_argument("--ppo", required=True, help="Path to ppo_pipeline.py")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input files in place")
    parser.add_argument("--suffix", default=".patched", help="Suffix for patched files when not using --in-place")
    args = parser.parse_args()

    replay_path = Path(args.replay)
    ppo_path = Path(args.ppo)

    replay_text = replay_path.read_text(encoding="utf-8")
    ppo_text = ppo_path.read_text(encoding="utf-8")

    replay_patched = patch_replay(replay_text)
    ppo_patched = patch_ppo_pipeline(ppo_text)

    if args.in_place:
        replay_path.write_text(replay_patched, encoding="utf-8")
        ppo_path.write_text(ppo_patched, encoding="utf-8")
        print(f"Patched in place: {replay_path}")
        print(f"Patched in place: {ppo_path}")
        return

    replay_out = replay_path.with_name(replay_path.stem + args.suffix + replay_path.suffix)
    ppo_out = ppo_path.with_name(ppo_path.stem + args.suffix + ppo_path.suffix)
    replay_out.write_text(replay_patched, encoding="utf-8")
    ppo_out.write_text(ppo_patched, encoding="utf-8")
    print(f"Wrote: {replay_out}")
    print(f"Wrote: {ppo_out}")


if __name__ == "__main__":
    main()
