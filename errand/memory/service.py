"""Stored preferences: explicit vs inferred facts, correction, confirmation and forgetting.

Explicit facts come from the user and are active at once. Inferred facts (from past bookings)
start unconfirmed: they are never used, or shown to a model, until the user answers a
code-generated yes/no question. A correction supersedes the old fact. Forgetting scrubs the values
from stored messages first, then tombstones every row for the key and erases its values.
"""

import re
from collections.abc import Iterable

from pydantic import JsonValue

from errand.store import FactSource, FactStatus, MemoryFact, Store
from errand.trace import EventKind, Tracer

# Which stored keys can fill which task slot.
SLOT_FOR_KEY = {
    "preferred_area": "area",
    "earliest_time": "earliest_time",
    "preferred_time_window": "time_window",
    "budget_max_inr": "budget_max_inr",
    "technician": "provider_id",
}
KEY_FOR_SLOT = {slot: key for key, slot in SLOT_FOR_KEY.items()}

# The preferences that matter to each kind of model call. Everything else stays out of context.
RELEVANT_KEYS = {
    "book": (
        "preferred_area",
        "earliest_time",
        "preferred_time_window",
        "budget_max_inr",
        "technician",
    ),
    "reschedule": ("earliest_time", "preferred_time_window"),
    "cancel": (),
    "smalltalk": tuple(SLOT_FOR_KEY),
}

KEY_LABELS = {
    "preferred_area": "preferred area",
    "earliest_time": "earliest start time",
    "preferred_time_window": "preferred time of day",
    "budget_max_inr": "budget (₹)",
    "technician": "preferred provider",
}


def display_value(value: JsonValue) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("provider_id") or value)
    return str(value)


def _texts(value: JsonValue) -> Iterable[str]:
    """Every string inside a stored value, for scrubbing it out of messages."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)
    elif value is not None:
        yield str(value)


class MemoryService:
    def __init__(self, store: Store, tracer: Tracer | None = None) -> None:
        self._store = store
        self._tracer = tracer

    def relevant(
        self, user_id: str, *, purpose: str, service: str | None, limit: int
    ) -> list[MemoryFact]:
        """Active facts for this kind of call, global or for `service`, newest first."""
        keys = RELEVANT_KEYS.get(purpose, ())
        return self._store.memory.relevant(user_id, keys, service_tag=service, limit=limit)

    def for_slot(self, user_id: str, slot: str, *, service: str | None) -> MemoryFact | None:
        """The current fact (active or unconfirmed) that could fill `slot`, preferring one tagged
        with this service over a global one."""
        key = KEY_FOR_SLOT.get(slot)
        if key is None:
            return None
        if service:
            tagged = self._store.memory.current(user_id, key, service_tag=service)
            if tagged is not None:
                return tagged
        return self._store.memory.current(user_id, key)

    def remember(
        self,
        user_id: str,
        key: str,
        value: JsonValue,
        *,
        service: str | None = None,
        evidence: str,
    ) -> MemoryFact:
        fact = self._store.memory.remember(
            user_id, key, value, source=FactSource.EXPLICIT, service_tag=service, evidence=evidence
        )
        self._emit("remembered", fact)
        return fact

    def infer(
        self, user_id: str, key: str, value: JsonValue, *, service: str | None = None, evidence: str
    ) -> MemoryFact:
        """Record a guess. It stays unconfirmed (and unused) until the user confirms it."""
        fact = self._store.memory.remember(
            user_id, key, value, source=FactSource.INFERRED, service_tag=service, evidence=evidence
        )
        self._emit("inferred", fact)
        return fact

    def confirm(self, user_id: str, fact_id: str) -> MemoryFact:
        """The user said yes to an inferred fact: it becomes an explicit, active one."""
        fact = self._require(user_id, fact_id)
        confirmed = self._store.memory.remember(
            fact.user_id,
            fact.key,
            fact.value,
            source=FactSource.EXPLICIT,
            service_tag=fact.service_tag,
            evidence=f"confirmed by the user (was inferred: {fact.evidence or 'no evidence'})",
        )
        self._emit("confirmed", confirmed)
        return confirmed

    def reject(self, user_id: str, fact_id: str) -> int:
        """The user said no to an inferred fact: forget it."""
        fact = self._require(user_id, fact_id)
        return self.forget(fact.user_id, fact.key, service=fact.service_tag)

    def forget(self, user_id: str, key: str, *, service: str | None = None) -> int:
        """Scrub the key's values from stored messages, then tombstone and erase every row."""
        values = set()
        for row in self._store.memory.list_for_user(user_id, include_inactive=True):
            if row.key == key and row.service_tag == service and row.value is not None:
                for text in _texts(row.value):
                    if len(text) >= 2:
                        values.add(text)
                    # People rarely repeat a value word for word ("the one ending 4410" for "a
                    # second number ending 4410"), but numbers survive paraphrase: scrub them
                    # on their own too (docs/bugs.md #3).
                    values.update(re.findall(r"\S*\d{3,}\S*", text))
        scrubbed = self._store.messages.scrub(user_id, values)
        rows = self._store.memory.forget(user_id, key, service_tag=service)
        if self._tracer and rows:
            self._tracer.emit(
                EventKind.MEMORY,
                event="forgotten",
                key=key,
                service=service,
                rows=rows,
                scrubbed_messages=scrubbed,
            )
        return rows

    def describe(self, user_id: str) -> str:
        """Everything currently stored, for the user to inspect (never sent to a model)."""
        facts = self._store.memory.list_for_user(user_id)
        if not facts:
            return "I don't have any preferences stored for you."
        lines = []
        for fact in facts:
            scope = f" for {fact.service_tag}" if fact.service_tag else ""
            if fact.status is FactStatus.UNCONFIRMED:
                origin = "my guess, not yet confirmed, so I don't use it"
            else:
                origin = "you told me" if fact.source is FactSource.EXPLICIT else "confirmed by you"
            lines.append(
                f"- {KEY_LABELS.get(fact.key, fact.key)}{scope}: {display_value(fact.value)} ({origin})"
            )
        return "Here is what I remember about you:\n" + "\n".join(lines)

    def _require(self, user_id: str, fact_id: str) -> MemoryFact:
        fact = self._store.memory.get(fact_id, user_id=user_id)
        if fact is None:
            raise KeyError(fact_id)
        return fact

    def _emit(self, event: str, fact: MemoryFact) -> None:
        if self._tracer:
            self._tracer.emit(
                EventKind.MEMORY,
                event=event,
                fact_id=fact.fact_id,
                key=fact.key,
                service=fact.service_tag,
                source=fact.source,
                status=fact.status,
            )
