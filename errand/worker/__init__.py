"""Background work: the reconciler and the provider-cancel poller, run on a timer."""

import logging
import threading

from errand.worker.poller import handle_provider_cancelled, poll_once
from errand.worker.reconciler import reconcile

log = logging.getLogger(__name__)


class Worker:
    """Reconciles open ledger rows and handles due provider events, every `interval_s`."""

    def __init__(self, agent, *, interval_s: float) -> None:
        self._agent = agent
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> dict[str, int]:
        return {"reconciled": len(reconcile(self._agent)), "events": poll_once(self._agent)}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="errand-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.run_once()
            except Exception:
                log.exception("worker tick failed")


__all__ = ["Worker", "handle_provider_cancelled", "poll_once", "reconcile"]
