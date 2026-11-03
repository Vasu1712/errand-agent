"""Deterministic time for the mock world and tests."""

import threading
from datetime import datetime, timedelta


class FakeClock:
    """Time that only moves when told to, so hold expiry and event timing are reproducible.

    ``sleep`` advances the clock instead of blocking, so retry backoff costs nothing in tests.
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs a timezone-aware start")
        self._now = start
        self._lock = threading.Lock()
        self._waited = threading.local()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> datetime:
        """Move time forward and return the new now."""
        with self._lock:
            self._now += timedelta(seconds=seconds)
            return self._now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        self._waited.seconds = getattr(self._waited, "seconds", 0.0) + seconds

    def monotonic(self) -> float:
        """The simulated waiting done on this thread (its slow providers, its backoffs): what a
        run's time budget spends. Another thread's sleep moves the world's clock (holds expire,
        events fall due) but is not time this run spent (docs/bugs.md #22)."""
        return getattr(self._waited, "seconds", 0.0)
