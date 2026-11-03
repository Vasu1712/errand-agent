"""Hard per-run limits on model calls, Vega decisions, tool calls and wall-clock time.

Execute and verify draw on a reserve that no other step can touch, so a run that reaches a side
effect always has the calls and the time to settle it.
"""

import time
from collections.abc import Callable
from typing import Self

from errand.config import Settings


class BudgetExceeded(Exception):
    def __init__(self, kind: str, used: float, limit: float) -> None:
        super().__init__(f"{kind} budget exhausted: {used:g} of {limit:g}")
        self.kind = kind
        self.used = used
        self.limit = limit


class RunBudget:
    def __init__(
        self,
        *,
        max_model_calls: int,
        max_decisions: int,
        max_tool_calls: int,
        timeout_s: float,
        reserved_tool_calls: int = 0,
        reserved_time_s: float = 0.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_model_calls = max_model_calls
        self.max_decisions = max_decisions
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s
        self.reserved_tool_calls = reserved_tool_calls
        self.reserved_time_s = reserved_time_s
        self._monotonic = monotonic
        self._started = monotonic()
        self.model_calls = 0
        self.decisions = 0
        self.tool_calls = 0

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> Self:
        kwargs: dict = {
            "max_model_calls": settings.max_model_calls,
            "max_decisions": settings.max_decisions,
            "max_tool_calls": settings.max_tool_calls,
            "timeout_s": settings.turn_timeout_s,
            "reserved_tool_calls": settings.reserved_tool_calls,
            "reserved_time_s": settings.reserved_time_s,
        }
        return cls(**{**kwargs, **overrides})

    def elapsed_s(self) -> float:
        return self._monotonic() - self._started

    def remaining_s(self, *, reserved: bool = False) -> float:
        limit = self.timeout_s if reserved else self.timeout_s - self.reserved_time_s
        return limit - self.elapsed_s()

    def check_time(self, *, reserved: bool = False) -> None:
        if self.remaining_s(reserved=reserved) <= 0:
            limit = self.timeout_s if reserved else self.timeout_s - self.reserved_time_s
            raise BudgetExceeded("time_s", round(self.elapsed_s(), 1), limit)

    def spend_model(self) -> None:
        self.check_time()
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("model_calls", self.model_calls, self.max_model_calls)
        self.model_calls += 1

    def spend_decision(self) -> None:
        self.check_time()
        if self.decisions >= self.max_decisions:
            raise BudgetExceeded("decisions", self.decisions, self.max_decisions)
        self.decisions += 1

    def check_tool(self, *, reserved: bool = False) -> None:
        """Raise as `spend_tool` would, without counting a call."""
        self.check_time(reserved=reserved)
        limit = self.max_tool_calls if reserved else self.max_tool_calls - self.reserved_tool_calls
        if self.tool_calls >= limit:
            raise BudgetExceeded("tool_calls", self.tool_calls, limit)

    def spend_tool(self, *, reserved: bool = False) -> None:
        """Count one tool call. Only execute and verify (``reserved=True``) may use the reserve."""
        self.check_tool(reserved=reserved)
        self.tool_calls += 1

    def snapshot(self) -> dict[str, float]:
        return {
            "model_calls": self.model_calls,
            "max_model_calls": self.max_model_calls,
            "decisions": self.decisions,
            "max_decisions": self.max_decisions,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "elapsed_s": round(self.elapsed_s(), 2),
            "timeout_s": self.timeout_s,
        }
