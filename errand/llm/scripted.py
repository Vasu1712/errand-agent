"""A model that follows a script: deterministic replies for tests and offline evals."""

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from errand.llm.base import (
    ChatMessage,
    ModelOutputError,
    ModelReply,
    ModelRequest,
    ScriptExhausted,
    T,
    TokenUsage,
    estimate_tokens,
)

Reply = Mapping[str, Any] | BaseModel
Step = Reply | BaseException | Callable[[ModelRequest], Reply]


class ScriptedModel:
    """Answers each call with the next step of its script.

    A step is the reply itself (a dict or a model, validated against the requested schema just as a
    live reply would be), a callable that computes the reply from the `ModelRequest`, or an
    exception to raise. The script is one ordered list, or a mapping from purpose to a list (used
    in order) or to a single callable (used for every call with that purpose). Every request is
    kept in `requests`.
    """

    def __init__(
        self,
        steps: Iterable[Step] | Mapping[str, Iterable[Step] | Callable[[ModelRequest], Reply]] = (),
        *,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._steps: deque[Step] = deque()
        self._by_purpose: dict[str, deque[Step]] = {}
        self._rules: dict[str, Callable[[ModelRequest], Reply]] = {}
        if isinstance(steps, Mapping):
            for purpose, entry in steps.items():
                if callable(entry):
                    self._rules[purpose] = entry
                else:
                    self._by_purpose[purpose] = deque(entry)
        else:
            self._steps.extend(steps)
        self.requests: list[ModelRequest] = []

    @property
    def remaining(self) -> int:
        return len(self._steps) + sum(len(queue) for queue in self._by_purpose.values())

    def _next_step(self, purpose: str) -> Step | None:
        if purpose in self._rules:
            return self._rules[purpose]
        queue = self._by_purpose.get(purpose)
        if queue:
            return queue.popleft()
        return self._steps.popleft() if self._steps else None

    def complete(
        self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str
    ) -> ModelReply[T]:
        request = ModelRequest(
            purpose=purpose, schema_name=schema.__name__, messages=tuple(messages)
        )
        self.requests.append(request)
        step = self._next_step(purpose)
        if step is None:
            raise ScriptExhausted(
                f"the script has no reply for call {len(self.requests)} ({purpose})"
            )
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            step = step(request)
        payload = step.model_dump(mode="json") if isinstance(step, BaseModel) else dict(step)
        raw = json.dumps(payload, ensure_ascii=False)
        try:
            output = schema.model_validate(payload)
        except ValidationError as exc:
            raise ModelOutputError(
                f"scripted reply for {purpose} doesn't fit {schema.__name__}: {exc}", raw=raw
            ) from exc
        prompt = "\n".join(message.content for message in messages)
        usage = TokenUsage(input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(raw))
        return ModelReply[schema](
            output=output, raw=raw, model=self.name, usage=usage, latency_ms=0.0
        )
