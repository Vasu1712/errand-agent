"""Record a live model's replies, and replay them offline.

`RecordingModel` wraps the live model (Ollama) and keeps every call: purpose, schema, a digest of
the messages sent, and the raw reply or the error. `ReplayModel` answers from those records in
order per purpose, so a run recorded against the live stack can be re-run with no network.
"""

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ValidationError

from errand.llm.base import (
    ChatMessage,
    ModelClient,
    ModelError,
    ModelOutputError,
    ModelReply,
    ModelUnavailable,
    ScriptExhausted,
    T,
    TokenUsage,
)

_ERRORS: dict[str, type[ModelError]] = {
    "ModelUnavailable": ModelUnavailable,
    "ModelOutputError": ModelOutputError,
    "ModelError": ModelError,
}


class ModelCall(BaseModel):
    """One recorded call: what was asked (as a digest) and what came back."""

    purpose: str
    schema_name: str
    request_digest: str
    model: str
    raw: str | None = None
    usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    error: str | None = None  # exception class name, when the call failed
    error_message: str | None = None


def request_digest(messages: Sequence[ChatMessage]) -> str:
    payload = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class RecordingModel:
    """Passes every call to `inner` and records it in `calls`."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner
        self.name = inner.name
        self.calls: list[ModelCall] = []

    def complete(
        self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str
    ) -> ModelReply[T]:
        base = {
            "purpose": purpose,
            "schema_name": schema.__name__,
            "request_digest": request_digest(messages),
            "model": self.inner.name,
        }
        try:
            reply = self.inner.complete(messages, schema, purpose=purpose)
        except ModelError as exc:
            raw = exc.raw if isinstance(exc, ModelOutputError) else None
            self.calls.append(
                ModelCall(**base, raw=raw, error=type(exc).__name__, error_message=str(exc))
            )
            raise
        self.calls.append(
            ModelCall(**base, raw=reply.raw, usage=reply.usage, latency_ms=reply.latency_ms)
        )
        return reply


class ReplayModel:
    """Answers each call with the next recorded call for the same purpose.

    A call the recording doesn't have, or one for a different schema, raises `ScriptExhausted`:
    the replayed run has diverged from the recorded one. A request whose messages differ from the
    recorded ones is still answered, and counted in `drift`, so a report can say so.
    """

    def __init__(self, calls: Iterable[ModelCall], *, name: str = "replay") -> None:
        self.name = name
        self._by_purpose: dict[str, deque[ModelCall]] = {}
        for call in calls:
            self._by_purpose.setdefault(call.purpose, deque()).append(call)
        self.drift: list[str] = []

    @property
    def remaining(self) -> int:
        return sum(len(queue) for queue in self._by_purpose.values())

    def complete(
        self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str
    ) -> ModelReply[T]:
        queue = self._by_purpose.get(purpose)
        if not queue:
            raise ScriptExhausted(f"the recording has no {purpose} call left")
        call = queue.popleft()
        if call.schema_name != schema.__name__:
            raise ScriptExhausted(
                f"the recording's {purpose} call was for {call.schema_name}, not {schema.__name__}"
            )
        if call.request_digest != request_digest(messages):
            self.drift.append(purpose)
        if call.error:
            error = _ERRORS.get(call.error, ModelError)
            message = call.error_message or call.error
            if error is ModelOutputError:
                raise ModelOutputError(message, raw=call.raw or "")
            raise error(message)
        try:
            output = schema.model_validate_json(call.raw or "")
        except ValidationError as exc:
            raise ModelOutputError(
                f"recorded reply for {purpose} doesn't fit {schema.__name__}: {exc}",
                raw=call.raw or "",
            ) from exc
        return ModelReply[schema](
            output=output,
            raw=call.raw or "",
            model=call.model,
            usage=call.usage,
            latency_ms=call.latency_ms,
        )
