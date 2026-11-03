"""Time. Everything that reads the clock or waits takes a `Clock`, so tests can use a FakeClock."""

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware."""
        ...

    def sleep(self, seconds: float) -> None: ...

    def monotonic(self) -> float:
        """Seconds on a clock that never goes back, for run budgets: time spent sleeping or
        waiting on a provider counts against them on whichever clock the agent runs."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def monotonic(self) -> float:
        return time.monotonic()
