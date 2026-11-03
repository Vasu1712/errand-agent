"""A turnstile: lets named parties through one point in a fixed order, once all have arrived.

Concurrency evals use it to pin an interleaving: two users' turns run on real threads, at the
same time, up to the point that matters (claiming a slot, taking a thread's lock), and then go
through it in the order the scenario names. The race is real; only its winner is chosen, so
the result is the same on every run.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager


class Turnstile:
    def __init__(self, order: Sequence[str], *, timeout_s: float = 10.0) -> None:
        self.order = list(order)
        self.timeout_s = timeout_s
        self.passed: list[str] = []
        self._arrived: set[str] = set()
        self._cond = threading.Condition()

    @contextmanager
    def turn(self, party: str | None) -> Iterator[None]:
        """Block until every party has arrived and it is `party`'s turn, run the block, then let
        the next one through. A party not in the order, or one already through, passes freely."""
        with self._cond:
            free = party not in self.order or party in self.passed
            if not free:
                self._arrived.add(party)
                self._cond.notify_all()
                ready = self._cond.wait_for(
                    lambda: (
                        len(self._arrived) == len(self.order)
                        and self.order[len(self.passed)] == party
                    ),
                    timeout=self.timeout_s,
                )
                if not ready:
                    missing = sorted(set(self.order) - self._arrived)
                    raise TimeoutError(f"turnstile: {party} waited for {missing or 'its turn'}")
        try:
            yield
        finally:
            if not free:
                with self._cond:
                    self.passed.append(party)
                    self._cond.notify_all()
