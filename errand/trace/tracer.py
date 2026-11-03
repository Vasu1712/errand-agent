"""Per-run JSONL traces: one file per run id under traces/, one JSON event per line.

A run is the agent's handling of one user message or one proactive event. Its trace records state
transitions, model calls, tool calls (name, arguments, sanitized result), decisions, the context
sent to the model, latency, token usage and estimated cost.
"""

import secrets
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from errand.ids import is_valid_id


class EventKind(StrEnum):
    RUN_START = "run.start"
    RUN_END = "run.end"
    STATE = "state.transition"
    MODEL_CALL = "model.call"
    TOOL_CALL = "tool.call"
    DECISION = "decision"
    CONTEXT = "context"
    APPROVAL = "approval"
    LEDGER = "ledger"
    INJECTION = "injection"
    MEMORY = "memory"
    # The agent told the user a side effect happened (reply, notice or outbox).
    CLAIM = "claim"
    BUDGET = "budget"
    ERROR = "error"


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    user_id: str = ""  # whose run this is; every event carries it
    seq: int
    ts: AwareDatetime
    kind: str
    latency_ms: float | None = None
    data: dict[str, JsonValue] = {}


REDACTED = "[redacted]"
# Exact key names, so that e.g. "input_tokens" isn't caught by "token".
_SECRET_KEYS = frozenset(
    {"access_token", "api_key", "authorization", "cookie", "password", "secret", "token"}
)
_MAX_DEPTH = 8


# The run the current thread is working on, as (run_id, user_id): lets a wrapped model client say
# whose request it is serving when several users' runs overlap. Set by Agent.tracer.
current_run: ContextVar[tuple[str, str] | None] = ContextVar("current_run", default=None)


def new_run_id() -> str:
    return f"run_{datetime.now(UTC):%Y%m%dT%H%M%S}_{secrets.token_hex(4)}"


def sanitize(value: Any, *, max_str: int = 2000, _depth: int = 0) -> JsonValue:
    """`value` made JSON-ready for a trace: models dumped, long text cut, secret keys redacted."""
    if _depth > _MAX_DEPTH:
        return "[max depth]"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return f"{value[:max_str]}…[+{len(value) - max_str} chars]"
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return f"[{len(value)} bytes]"
    if isinstance(value, BaseException):
        return sanitize(f"{type(value).__name__}: {value}", max_str=max_str)
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED
            if str(key).lower() in _SECRET_KEYS
            else sanitize(item, max_str=max_str, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, set | frozenset):
        value = sorted(value, key=repr)
    if isinstance(value, list | tuple):
        return [sanitize(item, max_str=max_str, _depth=_depth + 1) for item in value]
    return sanitize(repr(value), max_str=max_str)


class Tracer:
    """Appends one run's events to ``<traces_dir>/<run_id>.jsonl``. Safe to share across threads."""

    def __init__(
        self, run_id: str, traces_dir: Path, *, max_str: int = 2000, user_id: str = ""
    ) -> None:
        if not is_valid_id(run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        traces_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.user_id = user_id
        self.path = traces_dir / f"{run_id}.jsonl"
        self._max_str = max_str
        self._lock = threading.Lock()
        self._seq = 0
        # Mode "x": a run id names exactly one run, so reusing one fails instead of appending.
        self._file = self.path.open("x", encoding="utf-8")

    def emit(self, kind: str, /, *, latency_ms: float | None = None, **data: Any) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                run_id=self.run_id,
                user_id=self.user_id,
                seq=self._seq,
                ts=datetime.now(UTC),
                kind=kind,
                latency_ms=None if latency_ms is None else round(latency_ms, 2),
                data=sanitize(data, max_str=self._max_str),
            )
            # One write per complete line, flushed, so a reader polling mid-run sees whole events.
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()
        return event

    @contextmanager
    def span(self, kind: str, /, **data: Any) -> Iterator[dict[str, Any]]:
        """Time the block and emit one event when it ends.

        Keys added to the yielded dict are recorded with the event. An exception is recorded as
        ``error`` and re-raised.
        """
        extra: dict[str, Any] = {}
        started = perf_counter()
        try:
            yield extra
        except BaseException as exc:
            elapsed = (perf_counter() - started) * 1000
            self.emit(kind, latency_ms=elapsed, **{**data, **extra, "error": exc})
            raise
        self.emit(kind, latency_ms=(perf_counter() - started) * 1000, **{**data, **extra})

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_trace(traces_dir: Path, run_id: str) -> list[TraceEvent]:
    """One run's events in order. A trailing partial line (the run is still writing) is skipped.

    Raises ``FileNotFoundError`` for an unknown run and ``ValueError`` for a malformed run id.
    """
    if not is_valid_id(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    text = (traces_dir / f"{run_id}.jsonl").read_text(encoding="utf-8")
    *complete, _partial = text.split("\n")
    return [TraceEvent.model_validate_json(line) for line in complete if line]
