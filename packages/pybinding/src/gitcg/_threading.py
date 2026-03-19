from __future__ import annotations

import threading

from . import low_level as ll

_library_lock = threading.Lock()
_library_initialized = False
_thread_state = threading.local()


def current_thread_id() -> int:
    return threading.get_ident()


def initialize_library() -> None:
    global _library_initialized
    if _library_initialized:
        return
    with _library_lock:
        if not _library_initialized:
            ll.initialize()
            _library_initialized = True


def ensure_thread_initialized() -> None:
    initialize_library()
    if getattr(_thread_state, "initialized", False):
        return
    ll.thread_initialize()
    _thread_state.initialized = True
    _thread_state.thread_id = current_thread_id()


def thread_initialize() -> None:
    ensure_thread_initialized()


def thread_cleanup() -> None:
    if not getattr(_thread_state, "initialized", False):
        return
    ll.thread_cleanup()
    _thread_state.initialized = False
    _thread_state.thread_id = None


class ThreadBoundResource:
    _thread_handle_attr: str
    _resource_label: str
    _creator_thread_id: int
    _closed: bool

    def _thread_bind(self, handle_attr: str, resource_label: str) -> None:
        ensure_thread_initialized()
        self._thread_handle_attr = handle_attr
        self._resource_label = resource_label
        self._creator_thread_id = current_thread_id()
        self._closed = False

    def _raw_handle(self):
        return getattr(self, self._thread_handle_attr, ll.NULL)

    def _assert_thread(self) -> None:
        if current_thread_id() != self._creator_thread_id:
            raise RuntimeError(
                f"{self._resource_label} cannot be used across threads; "
                f"created on thread {self._creator_thread_id}, "
                f"accessed on thread {current_thread_id()}"
            )

    def _assert_access(self) -> None:
        ensure_thread_initialized()
        if self._closed or self._raw_handle() == ll.NULL:
            raise RuntimeError(f"{self._resource_label} is already closed")
        self._assert_thread()

    def _free_handle(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        handle = self._raw_handle()
        if handle == ll.NULL:
            self._closed = True
            return
        self._assert_access()
        self._free_handle()
        setattr(self, self._thread_handle_attr, ll.NULL)
        self._closed = True

    def _finalize(self) -> None:
        if getattr(self, "_closed", True):
            return
        handle = self._raw_handle()
        if handle == ll.NULL:
            self._closed = True
            return
        if current_thread_id() != getattr(self, "_creator_thread_id", None):
            return
        try:
            self._free_handle()
        except Exception:
            pass
        finally:
            setattr(self, self._thread_handle_attr, ll.NULL)
            self._closed = True

    def __enter__(self):
        self._assert_access()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
