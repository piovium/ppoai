from pathlib import Path
import json
import shutil
import sys


def strict_replace(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"[{name}] expected exactly 1 match, found {count}")
    return source.replace(old, new, 1)


def strict_remove(source: str, old: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"[{name}] expected exactly 1 match, found {count}")
    return source.replace(old, "", 1)


def rewrite(text: str) -> tuple[str, dict]:
    ops: list[str] = []

    old = '''_TRACE_LOG_PATH_ENV = "GITCG_IO_TRACE_PATH"
_TRACE_LOG_DEFAULT = "gitcg_io_trace.jsonl"


def _trace_log_path() -> str:
    configured = os.environ.get(_TRACE_LOG_PATH_ENV)
    if configured and configured.strip():
        return configured.strip()
    return os.path.join(os.getcwd(), _TRACE_LOG_DEFAULT)


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return repr(value)


def _append_trace_event(event: dict[str, Any]) -> None:
    try:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        path = _trace_log_path()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\\n")
    except Exception:
        pass


def _state_brief(full_state: Any) -> dict[str, Any]:
    return {
        "phase": getattr(full_state, "phase", None),
        "round_number": getattr(full_state, "round_number", None),
        "current_turn": getattr(full_state, "current_turn", None),
        "winner": getattr(full_state, "winner", None),
    }
'''
    new = '''_TRACE_LOG_PATH_ENV = "GITCG_IO_TRACE_PATH"
_TRACE_LOG_DEFAULT = "gitcg_io_trace.jsonl"
_PROBE_TOKEN = "116092.02"


def _trace_log_path() -> str:
    configured = os.environ.get(_TRACE_LOG_PATH_ENV)
    if configured and configured.strip():
        return configured.strip()
    return os.path.join(os.getcwd(), _TRACE_LOG_DEFAULT)


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return repr(value)


def _contains_probe_token(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _PROBE_TOKEN in value
    if isinstance(value, (int, bool)):
        return False
    if isinstance(value, float):
        return _PROBE_TOKEN in repr(value)
    if isinstance(value, (list, tuple)):
        return any(_contains_probe_token(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_probe_token(key) or _contains_probe_token(val) for key, val in value.items())
    return _PROBE_TOKEN in repr(value)


def _find_probe_snippets(text: str, *, radius: int = 140, limit: int = 2) -> list[str]:
    snippets: list[str] = []
    start = 0
    while len(snippets) < limit:
        index = text.find(_PROBE_TOKEN, start)
        if index < 0:
            break
        left = max(0, index - radius)
        right = min(len(text), index + len(_PROBE_TOKEN) + radius)
        snippets.append(text[left:right].replace("\\n", " "))
        start = index + len(_PROBE_TOKEN)
    return snippets


def _state_focus(raw_state_json: str | None) -> dict[str, Any]:
    text = raw_state_json or ""
    return {
        "length": len(text),
        "contains_probe_token": _PROBE_TOKEN in text,
        "snippets": _find_probe_snippets(text),
    }


def _state_brief(full_state: Any) -> dict[str, Any]:
    return {
        "phase": getattr(full_state, "phase", None),
        "round_number": getattr(full_state, "round_number", None),
        "current_turn": getattr(full_state, "current_turn", None),
        "winner": getattr(full_state, "winner", None),
    }


def _append_trace_event(event: dict[str, Any]) -> None:
    try:
        payload = dict(event)
        if not _contains_probe_token(payload):
            return
        payload.setdefault("ts", time.time())
        path = _trace_log_path()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\\n")
    except Exception:
        pass
'''
    text = strict_replace(text, old, new, 'helpers_dot_only')
    ops.append('helpers_dot_only')

    old = '''        self._worker = Thread(
            target=self._run_worker,
            args=(self._seed, initial_state_json),
            daemon=True,
            name=f"gitcg-world-model-{self._matchup.key}",
        )
        self._worker.start()
'''
    new = '''        self._worker = Thread(
            target=self._run_worker,
            args=(self._seed, initial_state_json),
            daemon=True,
            name=f"gitcg-world-model-{self._matchup.key}",
        )
        if initial_state_json is not None and _PROBE_TOKEN in initial_state_json:
            _append_trace_event(
                {
                    "kind": "reset_state_json_dot",
                    "matchup": self._matchup.key,
                    "seed": self._seed,
                    "focus": _state_focus(initial_state_json),
                }
            )
        self._worker.start()
'''
    text = strict_replace(text, old, new, 'reset_dot_only_probe')
    ops.append('reset_dot_only_probe')

    old = '''                def __init__(self, who: int):
                    self.who = who
                    self.last_notification = None
                    self.last_error: str | None = None
                    self.last_request_type: str | None = None
                    self.last_request_summary: str | None = None
                    self.last_response_payload: dict[str, Any] | None = None
                    self.trace_buffer: deque[dict[str, Any]] = deque(maxlen=32)
'''
    new = '''                def __init__(self, who: int):
                    self.who = who
                    self.last_notification = None
                    self.last_error: str | None = None
                    self.last_request_type: str | None = None
                    self.last_request_summary: str | None = None
                    self.last_response_payload: dict[str, Any] | None = None
                    self.trace_buffer: deque[dict[str, Any]] = deque(maxlen=32)
'''
    text = strict_replace(text, old, new, 'bridge_init_anchor')
    ops.append('bridge_init_anchor')

    old = '''                def on_io_error(self, error_msg: str):
                    self.last_error = error_msg
                    event = {
                        "kind": "io_error",
                        "matchup": env._matchup.key,
                        "seed": env._seed,
                        "player": self.who,
                        "request_type": self.last_request_type,
                        "request_summary": self.last_request_summary,
                        "response_payload": self.last_response_payload,
                        "error": error_msg,
                        "recent_trace": list(self.trace_buffer),
                    }
                    self._push_trace(event)
                    print(
                        "[io-error-file] "
                        f"path={_trace_log_path()} matchup={env._matchup.key} seed={env._seed} player={self.who}",
                        flush=True,
                    )
'''
    new = '''                def on_io_error(self, error_msg: str):
                    self.last_error = error_msg
                    if _PROBE_TOKEN not in str(error_msg):
                        return
                    event = {
                        "kind": "io_error",
                        "matchup": env._matchup.key,
                        "seed": env._seed,
                        "player": self.who,
                        "request_type": self.last_request_type,
                        "request_summary": self.last_request_summary,
                        "response_payload": self.last_response_payload,
                        "error": error_msg,
                        "recent_trace": list(self.trace_buffer),
                    }
                    self._push_trace(event)
'''
    text = strict_replace(text, old, new, 'io_error_dot_only')
    ops.append('io_error_dot_only')

    old = '''                    full_state_json = raw_state_json if env._config.record_full_state_json else None
                    visible_snapshot = None
'''
    new = '''                    full_state_json = raw_state_json if env._config.record_full_state_json else None
                    if _PROBE_TOKEN in raw_state_json:
                        _append_trace_event(
                            {
                                "kind": "request_state_dot",
                                "matchup": env._matchup.key,
                                "seed": env._seed,
                                "player": self.who,
                                "step_index": step_index_ref["value"],
                                "request_type": request_type.name,
                                "request_summary": self.last_request_summary,
                                "focus": _state_focus(raw_state_json),
                            }
                        )
                    visible_snapshot = None
'''
    text = strict_replace(text, old, new, 'request_state_dot_probe')
    ops.append('request_state_dot_probe')

    old = '''                    if request_type == DecisionType.SELECT_CARD:
                        raw_candidates = list(getattr(request, "candidate_definition_ids", ()))
                        if any(
                            (
                                isinstance(value, float) and not value.is_integer()
                            ) or (
                                not isinstance(value, (int, bool, float))
                            )
                            for value in raw_candidates
                        ):
                            print(
                                "[select-card-trace] "
                                f"stage=env.request matchup={env._matchup.key} seed={env._seed} player={self.who} "
                                f"step={step_index_ref['value']} candidates={[repr(v) + ':' + type(v).__name__ for v in raw_candidates]}",
                                flush=True,
                            )
'''
    text = strict_remove(text, old, 'remove_select_card_trace_stdout')
    ops.append('remove_select_card_trace_stdout')

    old = '''                    _append_trace_event(
                        {
                            "kind": "terminal_guard",
                            "matchup": self._matchup.key,
                            "seed": self._seed,
                            "reason": invalid_terminal_reason,
                            "status": status_name,
                            "raw_winner": raw_winner,
                            "sanitized_winner": sanitized_winner,
                            "error": error_message,
                        }
                    )
'''
    new = '''                    _append_trace_event(
                        {
                            "kind": "terminal_guard",
                            "matchup": self._matchup.key,
                            "seed": self._seed,
                            "reason": invalid_terminal_reason,
                            "status": status_name,
                            "raw_winner": raw_winner,
                            "sanitized_winner": sanitized_winner,
                            "error": error_message,
                            "player0_io_error": player0.last_error,
                            "player1_io_error": player1.last_error,
                            "focus": _state_focus(raw_state_json),
                        }
                    )
'''
    text = strict_replace(text, old, new, 'terminal_guard_focus')
    ops.append('terminal_guard_focus')

    required_present = [
        '_PROBE_TOKEN = "116092.02"',
        'def _contains_probe_token(value: Any) -> bool:',
        'def _state_focus(raw_state_json: str | None) -> dict[str, Any]:',
        '"kind": "reset_state_json_dot"',
        'if _PROBE_TOKEN not in str(error_msg):',
        '"kind": "request_state_dot"',
        '"player0_io_error": player0.last_error,',
        '"focus": _state_focus(raw_state_json),',
    ]
    required_absent = [
        '[io-error-file]',
        '[select-card-trace]',
    ]
    missing = [m for m in required_present if m not in text]
    leftovers = [m for m in required_absent if m in text]
    report = {
        'operations_applied': ops,
        'missing_required_new_markers': missing,
        'leftover_old_markers': leftovers,
        'success': not missing and not leftovers,
    }
    return text, report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: python rewrite_env_dot_only_probe_current.py <path-to-env.py>')
    src = Path(sys.argv[1])
    text = src.read_text(encoding='utf-8')
    new_text, report = rewrite(text)
    backup = src.with_suffix(src.suffix + '.bak')
    shutil.copyfile(src, backup)
    src.write_text(new_text, encoding='utf-8')
    preview = src.with_name(src.stem + '.dot_only.current.py')
    preview.write_text(new_text, encoding='utf-8')
    report.update({
        'source_file': str(src),
        'backup_file': str(backup),
        'preview_file': str(preview),
    })
    report_path = src.with_name(src.stem + '.dot_only_current_report.txt')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report_path)
    if not report['success']:
        raise SystemExit('rewrite finished but verification failed; inspect report')


if __name__ == '__main__':
    main()
