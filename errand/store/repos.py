"""Repositories over the agent DB: tasks, approvals, ledger, memory facts, outbox, messages, holds,
runs.

Every row belongs to one user, and every method takes that user's id and filters by it: a query
without one is refused by the database's scope guard (errand/store/db.py). The few scans that
really span users (the reconciler's sweep) say so with `Database.cross_user`.

Every state change is one conditional UPDATE or INSERT inside a write transaction, so a check and
the change it guards can't be separated by a concurrent writer.
"""

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import JsonValue

from errand.contracts import Hold
from errand.ids import new_id
from errand.store.db import Database, utcnow
from errand.store.records import (
    CURRENT_FACT_STATUSES,
    Action,
    Approval,
    ApprovalStatus,
    FactSource,
    FactStatus,
    HoldRecord,
    LedgerEntry,
    LedgerStatus,
    MemoryFact,
    Message,
    OutboxMessage,
    Run,
    Task,
    canonical_json,
    material_hash,
)
from errand.store.schema import USER_TABLES

FORGOTTEN_MARK = "[forgotten]"


class StoreError(Exception):
    """A repository refused an operation that would break one of its invariants."""


class ApprovalError(StoreError):
    """The approval is missing, no longer in the expected status, or covers different material."""


class LedgerConflict(StoreError):
    """An idempotency key was reused for a different operation."""


def _load(row: sqlite3.Row, *json_columns: str) -> dict[str, Any]:
    data = dict(row)
    for column in json_columns:
        if data[column] is not None:
            data[column] = json.loads(data[column])
    return data


_CURRENT_FACT = "status IN ('active', 'unconfirmed')"
_SAME_KEY = "user_id = ? AND key = ? AND IFNULL(service_tag, '') = IFNULL(?, '')"


class TaskRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        user_id: str,
        thread_id: str,
        intent: Action,
        phase: str,
        details: Mapping[str, JsonValue] | None = None,
    ) -> Task:
        task_id = new_id("task")
        now = self._db.timestamp()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, user_id, thread_id, intent, phase, details,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    user_id,
                    thread_id,
                    intent,
                    phase,
                    canonical_json(details or {}),
                    now,
                    now,
                ),
            )
        return self._require(task_id, user_id)

    def get(self, task_id: str, *, user_id: str) -> Task | None:
        row = self._db.fetch_one(
            "SELECT * FROM tasks WHERE task_id = ? AND user_id = ?", (task_id, user_id)
        )
        return Task.model_validate(_load(row, "details")) if row else None

    def update(
        self,
        task_id: str,
        *,
        user_id: str,
        phase: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
        booking_id: str | None = None,
    ) -> Task:
        changes = {
            "phase": phase,
            "details": None if details is None else canonical_json(details),
            "booking_id": booking_id,
        }
        changes = {column: value for column, value in changes.items() if value is not None}
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)
            with self._db.transaction() as conn:
                cur = conn.execute(
                    f"UPDATE tasks SET {assignments}, updated_at = ?"
                    " WHERE task_id = ? AND user_id = ?",
                    (*changes.values(), self._db.timestamp(), task_id, user_id),
                )
            if cur.rowcount == 0:
                raise KeyError(task_id)
        return self._require(task_id, user_id)

    def for_thread(self, thread_id: str, *, user_id: str) -> list[Task]:
        rows = self._db.fetch_all(
            "SELECT * FROM tasks WHERE thread_id = ? AND user_id = ? ORDER BY created_at, rowid",
            (thread_id, user_id),
        )
        return [Task.model_validate(_load(row, "details")) for row in rows]

    def _require(self, task_id: str, user_id: str) -> Task:
        task = self.get(task_id, user_id=user_id)
        if task is None:
            raise KeyError(task_id)
        return task


class ApprovalRepo:
    """Approvals bind a consequential action to the exact material the user confirmed.

    The material (provider, slot, price, ...) is hashed when the approval is requested and can't
    change afterwards; granting and consuming both require the caller to present the same hash,
    and the approval's own user.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def request(
        self, task_id: str, action: Action, material: Mapping[str, JsonValue], *, user_id: str
    ) -> Approval:
        """Open a pending approval for exactly `material`. Any live approval for the task is
        invalidated in the same transaction."""
        approval_id = new_id("apr")
        now = self._db.timestamp()
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE approvals SET status = 'invalidated', reason = ?, invalidated_at = ?"
                " WHERE task_id = ? AND user_id = ? AND status IN ('pending', 'granted')",
                (f"superseded by {approval_id}", now, task_id, user_id),
            )
            conn.execute(
                "INSERT INTO approvals (approval_id, user_id, task_id, action, material,"
                " material_hash, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    approval_id,
                    user_id,
                    task_id,
                    action,
                    canonical_json(material),
                    material_hash(material),
                    now,
                ),
            )
        return self._require(approval_id, user_id)

    def grant(self, approval_id: str, confirmed_hash: str, *, user_id: str) -> Approval:
        """pending -> granted, only if the user confirmed exactly this material."""
        return self._advance(
            approval_id,
            confirmed_hash,
            user_id,
            ApprovalStatus.PENDING,
            ApprovalStatus.GRANTED,
            "granted_at",
        )

    def consume(self, approval_id: str, confirmed_hash: str, *, user_id: str) -> Approval:
        """granted -> consumed. Spend the approval once, just before its side effect."""
        return self._advance(
            approval_id,
            confirmed_hash,
            user_id,
            ApprovalStatus.GRANTED,
            ApprovalStatus.CONSUMED,
            "consumed_at",
        )

    def invalidate(self, task_id: str, reason: str, *, user_id: str) -> list[str]:
        """Invalidate the task's live approval, e.g. because the provider, slot or price changed.
        Returns the ids invalidated (empty when nothing was live)."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                "UPDATE approvals SET status = 'invalidated', reason = ?, invalidated_at = ?"
                " WHERE task_id = ? AND user_id = ? AND status IN ('pending', 'granted')"
                " RETURNING approval_id",
                (reason, self._db.timestamp(), task_id, user_id),
            ).fetchall()
        return [row["approval_id"] for row in rows]

    def get(self, approval_id: str, *, user_id: str) -> Approval | None:
        row = self._db.fetch_one(
            "SELECT * FROM approvals WHERE approval_id = ? AND user_id = ?",
            (approval_id, user_id),
        )
        return Approval.model_validate(_load(row, "material")) if row else None

    def live(self, task_id: str, *, user_id: str) -> Approval | None:
        row = self._db.fetch_one(
            "SELECT * FROM approvals WHERE task_id = ? AND user_id = ?"
            " AND status IN ('pending', 'granted')",
            (task_id, user_id),
        )
        return Approval.model_validate(_load(row, "material")) if row else None

    def _advance(
        self,
        approval_id: str,
        confirmed_hash: str,
        user_id: str,
        from_status: ApprovalStatus,
        to_status: ApprovalStatus,
        stamp_column: str,
    ) -> Approval:
        with self._db.transaction() as conn:
            cur = conn.execute(
                f"UPDATE approvals SET status = ?, {stamp_column} = ?"
                " WHERE approval_id = ? AND user_id = ? AND status = ? AND material_hash = ?",
                (
                    to_status,
                    self._db.timestamp(),
                    approval_id,
                    user_id,
                    from_status,
                    confirmed_hash,
                ),
            )
        if cur.rowcount != 1:
            raise ApprovalError(
                self._refusal(approval_id, confirmed_hash, user_id, from_status, to_status)
            )
        return self._require(approval_id, user_id)

    def _refusal(
        self,
        approval_id: str,
        confirmed_hash: str,
        user_id: str,
        from_status: ApprovalStatus,
        to_status: ApprovalStatus,
    ) -> str:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None:
            return f"approval {approval_id} does not exist for {user_id}"
        if approval.material_hash != confirmed_hash:
            return f"approval {approval_id} covers different material from what was confirmed"
        return (
            f"approval {approval_id} is {approval.status}, not {from_status}, so it can't become"
            f" {to_status}"
        )

    def _require(self, approval_id: str, user_id: str) -> Approval:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None:
            raise KeyError(approval_id)
        return approval


class LedgerRepo:
    """One entry per side effect, keyed by its idempotency key, written before the tool is called.

    After a crash or timeout the entry is how the agent knows a call may have happened, so it asks
    the source of truth instead of calling again.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def begin(
        self,
        idempotency_key: str,
        op: Action,
        *,
        user_id: str,
        request: Mapping[str, JsonValue],
        task_id: str | None = None,
        approval_id: str | None = None,
    ) -> tuple[LedgerEntry, bool]:
        """Open the entry for a side effect and return ``(entry, created)``.

        If the key already exists, the stored entry comes back unchanged with ``created=False``.
        The caller must act on its status (a confirmed op is never repeated), not start over.
        """
        now = self._db.timestamp()
        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO ledger (idempotency_key, user_id, op, task_id, approval_id, request,"
                " status, attempts, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'in_flight', 0, ?, ?)"
                " ON CONFLICT (idempotency_key) DO NOTHING",
                (
                    idempotency_key,
                    user_id,
                    op,
                    task_id,
                    approval_id,
                    canonical_json(request),
                    now,
                    now,
                ),
            )
            entry = self._require(idempotency_key, user_id)
        created = cur.rowcount == 1
        if not created and (
            entry.op != op or canonical_json(entry.request) != canonical_json(request)
        ):
            raise LedgerConflict(
                f"idempotency key {idempotency_key} already belongs to a different {entry.op}"
                " request"
            )
        return entry, created

    def record_attempt(self, idempotency_key: str, *, user_id: str) -> LedgerEntry:
        """Count one more call of the tool under this key; the entry is in flight again."""
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE ledger SET attempts = attempts + 1, status = 'in_flight', updated_at = ?"
                " WHERE idempotency_key = ? AND user_id = ?",
                (self._db.timestamp(), idempotency_key, user_id),
            )
        if cur.rowcount == 0:
            raise KeyError(idempotency_key)
        return self._require(idempotency_key, user_id)

    def mark(
        self,
        idempotency_key: str,
        status: LedgerStatus,
        *,
        user_id: str,
        result: Mapping[str, JsonValue] | None = None,
        error: str | None = None,
    ) -> LedgerEntry:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE ledger SET status = ?, result = COALESCE(?, result), last_error = ?,"
                " updated_at = ? WHERE idempotency_key = ? AND user_id = ?",
                (
                    status,
                    None if result is None else canonical_json(result),
                    error,
                    self._db.timestamp(),
                    idempotency_key,
                    user_id,
                ),
            )
        if cur.rowcount == 0:
            raise KeyError(idempotency_key)
        return self._require(idempotency_key, user_id)

    def get(self, idempotency_key: str, *, user_id: str) -> LedgerEntry | None:
        row = self._db.fetch_one(
            "SELECT * FROM ledger WHERE idempotency_key = ? AND user_id = ?",
            (idempotency_key, user_id),
        )
        return LedgerEntry.model_validate(_load(row, "request", "result")) if row else None

    def for_approval(self, approval_id: str, *, user_id: str) -> LedgerEntry | None:
        """The one side effect an approval was spent on, if it has been spent."""
        row = self._db.fetch_one(
            "SELECT * FROM ledger WHERE approval_id = ? AND user_id = ?", (approval_id, user_id)
        )
        return LedgerEntry.model_validate(_load(row, "request", "result")) if row else None

    def for_booking(self, booking_id: str, *, user_id: str) -> LedgerEntry | None:
        """The confirmed booking op that made `booking_id`, if the agent made it."""
        row = self._db.fetch_one(
            "SELECT * FROM ledger WHERE user_id = ? AND status = 'confirmed' AND op = 'book'"
            " AND json_extract(result, '$.booking.booking_id') = ?",
            (user_id, booking_id),
        )
        return LedgerEntry.model_validate(_load(row, "request", "result")) if row else None

    def unresolved(self, *, user_id: str) -> list[LedgerEntry]:
        rows = self._db.fetch_all(
            "SELECT * FROM ledger WHERE user_id = ? AND status IN ('in_flight', 'unknown')"
            " ORDER BY created_at, rowid",
            (user_id,),
        )
        return [LedgerEntry.model_validate(_load(row, "request", "result")) for row in rows]

    def unresolved_everywhere(self) -> list[LedgerEntry]:
        """Every user's entries whose outcome isn't known yet: the reconciler's sweep, and the one
        ledger read that spans users."""
        rows = self._db.scan(
            "SELECT * FROM ledger WHERE status IN ('in_flight', 'unknown')"
            " ORDER BY created_at, rowid",
            reason="reconciler: open ledger rows of every user",
        )
        return [LedgerEntry.model_validate(_load(row, "request", "result")) for row in rows]

    def _require(self, idempotency_key: str, user_id: str) -> LedgerEntry:
        entry = self.get(idempotency_key, user_id=user_id)
        if entry is None:
            raise KeyError(idempotency_key)
        return entry


class MemoryRepo:
    """Stable user preferences. Each (user, key, service tag) has at most one current fact."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def remember(
        self,
        user_id: str,
        key: str,
        value: JsonValue,
        *,
        source: FactSource,
        service_tag: str | None = None,
        evidence: str | None = None,
    ) -> MemoryFact:
        """Make `value` the current fact for the key, superseding the previous one.

        Explicit facts are active at once and inferred ones start unconfirmed. An inferred value
        never replaces an explicit one: the explicit fact is returned unchanged.
        """
        if value is None:
            raise ValueError("a remembered value can't be null; use forget() to drop a fact")
        fact_id = new_id("mem")
        now = self._db.timestamp()
        status = FactStatus.ACTIVE if source is FactSource.EXPLICIT else FactStatus.UNCONFIRMED
        with self._db.transaction() as conn:
            current = self.current(user_id, key, service_tag=service_tag)
            if current is not None and (
                (current.source, current.value) == (source, value)
                or (current.source is FactSource.EXPLICIT and source is FactSource.INFERRED)
            ):
                return current
            if current is not None:
                conn.execute(
                    "UPDATE memory_facts SET status = 'superseded', superseded_by = ?,"
                    " updated_at = ? WHERE fact_id = ? AND user_id = ?",
                    (fact_id, now, current.fact_id, user_id),
                )
            conn.execute(
                "INSERT INTO memory_facts (fact_id, user_id, key, value, service_tag, source,"
                " status, evidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id,
                    user_id,
                    key,
                    canonical_json(value),
                    service_tag,
                    source,
                    status,
                    evidence,
                    now,
                    now,
                ),
            )
        return self._require(fact_id, user_id)

    def forget(self, user_id: str, key: str, *, service_tag: str | None = None) -> int:
        """Forget the key: every row for it, history included, is tombstoned and its value and
        evidence erased. Returns the number of rows forgotten (0 if nothing was stored)."""
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE memory_facts SET status = 'forgotten', value = NULL, evidence = NULL,"
                f" updated_at = ? WHERE {_SAME_KEY} AND status <> 'forgotten'",
                (self._db.timestamp(), user_id, key, service_tag),
            )
        return cur.rowcount

    def get(self, fact_id: str, *, user_id: str) -> MemoryFact | None:
        row = self._db.fetch_one(
            "SELECT * FROM memory_facts WHERE fact_id = ? AND user_id = ?", (fact_id, user_id)
        )
        return MemoryFact.model_validate(_load(row, "value")) if row else None

    def stored_values(self, user_id: str, key: str, *, service_tag: str | None = None) -> list[str]:
        """Every value still stored for the key, history included, as text (used for scrubbing)."""
        rows = self._db.fetch_all(
            f"SELECT value FROM memory_facts WHERE {_SAME_KEY} AND value IS NOT NULL",
            (user_id, key, service_tag),
        )
        values = {_as_text(json.loads(row["value"])) for row in rows}
        return sorted(value for value in values if value)

    def relevant(
        self, user_id: str, keys: Sequence[str], *, service_tag: str | None, limit: int
    ) -> list[MemoryFact]:
        """Active facts for these keys that apply to every service or to `service_tag`, newest
        first. Unconfirmed, superseded and forgotten facts never match."""
        if not keys or limit <= 0:
            return []
        placeholders = ", ".join("?" for _ in keys)
        rows = self._db.fetch_all(
            f"SELECT * FROM memory_facts WHERE user_id = ? AND status = 'active'"
            f" AND key IN ({placeholders}) AND (service_tag IS NULL OR service_tag = ?)"
            " ORDER BY updated_at DESC, rowid DESC LIMIT ?",
            (user_id, *keys, service_tag, limit),
        )
        return [MemoryFact.model_validate(_load(row, "value")) for row in rows]

    def status_counts(self, user_id: str) -> dict[str, int]:
        rows = self._db.fetch_all(
            "SELECT status, COUNT(*) AS n FROM memory_facts WHERE user_id = ? GROUP BY status",
            (user_id,),
        )
        return {row["status"]: row["n"] for row in rows}

    def current(
        self, user_id: str, key: str, *, service_tag: str | None = None
    ) -> MemoryFact | None:
        row = self._db.fetch_one(
            f"SELECT * FROM memory_facts WHERE {_SAME_KEY} AND {_CURRENT_FACT}",
            (user_id, key, service_tag),
        )
        return MemoryFact.model_validate(_load(row, "value")) if row else None

    def list_for_user(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryFact]:
        """The user's current facts, or every row (superseded and forgotten too) for inspection."""
        statuses = list(FactStatus) if include_inactive else list(CURRENT_FACT_STATUSES)
        placeholders = ", ".join("?" for _ in statuses)
        rows = self._db.fetch_all(
            f"SELECT * FROM memory_facts WHERE user_id = ? AND status IN ({placeholders})"
            " ORDER BY key, IFNULL(service_tag, ''), created_at, rowid",
            (user_id, *statuses),
        )
        return [MemoryFact.model_validate(_load(row, "value")) for row in rows]

    def _require(self, fact_id: str, user_id: str) -> MemoryFact:
        fact = self.get(fact_id, user_id=user_id)
        if fact is None:
            raise KeyError(fact_id)
        return fact


class OutboxRepo:
    """Messages the agent sends without being asked, such as "your provider cancelled"."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def enqueue(
        self,
        user_id: str,
        kind: str,
        body: str,
        *,
        task_id: str | None = None,
        payload: Mapping[str, JsonValue] | None = None,
        dedupe_key: str | None = None,
    ) -> tuple[OutboxMessage, bool]:
        """Queue a message and return ``(message, created)``. A repeated `dedupe_key` returns the
        original message, so one event never notifies the user twice."""
        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO outbox (user_id, task_id, kind, body, payload, dedupe_key, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (dedupe_key) DO NOTHING",
                (
                    user_id,
                    task_id,
                    kind,
                    body,
                    canonical_json(payload or {}),
                    dedupe_key,
                    self._db.timestamp(),
                ),
            )
            if cur.rowcount == 1:
                row = self._db.fetch_one(
                    "SELECT * FROM outbox WHERE message_id = ? AND user_id = ?",
                    (cur.lastrowid, user_id),
                )
            else:
                row = self._db.fetch_one(
                    "SELECT * FROM outbox WHERE dedupe_key = ? AND user_id = ?",
                    (dedupe_key, user_id),
                )
        if row is None:  # the key is taken by another user's message: never return theirs
            raise StoreError(f"dedupe key {dedupe_key!r} belongs to another user's message")
        return OutboxMessage.model_validate(_load(row, "payload")), cur.rowcount == 1

    def exists(self, user_id: str, dedupe_key: str) -> bool:
        row = self._db.fetch_one(
            "SELECT 1 FROM outbox WHERE user_id = ? AND dedupe_key = ?", (user_id, dedupe_key)
        )
        return row is not None

    def since(self, user_id: str, after_id: int = 0, *, limit: int = 100) -> list[OutboxMessage]:
        rows = self._db.fetch_all(
            "SELECT * FROM outbox WHERE user_id = ? AND message_id > ? ORDER BY message_id LIMIT ?",
            (user_id, after_id, limit),
        )
        return [OutboxMessage.model_validate(_load(row, "payload")) for row in rows]

    def mark_delivered(self, user_id: str, message_ids: Iterable[int]) -> int:
        ids = list(message_ids)
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._db.transaction() as conn:
            cur = conn.execute(
                f"UPDATE outbox SET delivered_at = ? WHERE user_id = ? AND delivered_at IS NULL"
                f" AND message_id IN ({placeholders})",
                (self._db.timestamp(), user_id, *ids),
            )
        return cur.rowcount


def _as_text(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class MessageRepo:
    """The conversation, one row per user or assistant message."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(
        self,
        user_id: str,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        *,
        run_id: str | None = None,
        created_at: datetime | None = None,
        for_model: bool = True,
    ) -> Message:
        stamp = (
            created_at.astimezone(UTC).isoformat(timespec="milliseconds")
            if created_at
            else self._db.timestamp()
        )
        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO messages (user_id, thread_id, run_id, role, content, created_at,"
                " for_model) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, thread_id, run_id, role, content, stamp, int(for_model)),
            )
        return self._require(cur.lastrowid, user_id)

    def recent(
        self, user_id: str, thread_id: str, limit: int, *, before_id: int | None = None
    ) -> list[Message]:
        """The thread's last `limit` messages a model may see (before `before_id`), oldest
        first."""
        if limit <= 0:
            return []
        rows = self._db.fetch_all(
            "SELECT * FROM messages WHERE user_id = ? AND thread_id = ? AND message_id < ?"
            " AND for_model = 1 ORDER BY message_id DESC LIMIT ?",
            (user_id, thread_id, before_id or 2**62, limit),
        )
        return [Message.model_validate(dict(row)) for row in reversed(rows)]

    def history(self, user_id: str, *, exclude_ids: Iterable[int], limit: int) -> list[Message]:
        """The user's other messages a model may see, any thread, newest first: the rolling
        summary's source."""
        excluded = sorted(set(exclude_ids)) or [0]
        placeholders = ", ".join("?" for _ in excluded)
        rows = self._db.fetch_all(
            f"SELECT * FROM messages WHERE user_id = ? AND message_id NOT IN ({placeholders})"
            " AND for_model = 1 ORDER BY message_id DESC LIMIT ?",
            (user_id, *excluded, limit),
        )
        return [Message.model_validate(dict(row)) for row in rows]

    def count(self, user_id: str) -> int:
        row = self._db.fetch_one("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
        return row[0] if row else 0

    def scrub(self, user_id: str, values: Iterable[str]) -> int:
        """Replace every occurrence of `values` (case-insensitive) in the user's messages."""
        patterns = [
            re.compile(re.escape(value), re.IGNORECASE)
            for value in values
            if len(value.strip()) >= 2
        ]
        if not patterns:
            return 0
        changed = 0
        with self._db.transaction() as conn:
            for row in conn.execute(
                "SELECT message_id, content FROM messages WHERE user_id = ?", (user_id,)
            ).fetchall():
                content = row["content"]
                for pattern in patterns:
                    content = pattern.sub(FORGOTTEN_MARK, content)
                if content != row["content"]:
                    conn.execute(
                        "UPDATE messages SET content = ? WHERE message_id = ? AND user_id = ?",
                        (content, row["message_id"], user_id),
                    )
                    changed += 1
        return changed

    def _require(self, message_id: int | None, user_id: str) -> Message:
        row = self._db.fetch_one(
            "SELECT * FROM messages WHERE message_id = ? AND user_id = ?", (message_id, user_id)
        )
        if row is None:
            raise KeyError(message_id)
        return Message.model_validate(dict(row))


class HoldRepo:
    """The gateway's own copy of each hold it obtained. Written once, never changed."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, hold: Hold, *, user_id: str, task_id: str | None) -> HoldRecord:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO holds (hold_id, user_id, task_id, provider_id, slot_start, slot_end,"
                " quoted_price_inr, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (hold_id) DO NOTHING",
                (
                    hold.hold_id,
                    user_id,
                    task_id,
                    hold.provider_id,
                    hold.slot.start.astimezone(UTC).isoformat(),
                    hold.slot.end.astimezone(UTC).isoformat(),
                    hold.quoted_price_inr,
                    hold.expires_at.astimezone(UTC).isoformat(),
                    self._db.timestamp(),
                ),
            )
        record = self.get(hold.hold_id, user_id=user_id)
        assert record is not None, f"hold {hold.hold_id} belongs to another user"
        return record

    def get(self, hold_id: str, *, user_id: str) -> HoldRecord | None:
        row = self._db.fetch_one(
            "SELECT * FROM holds WHERE hold_id = ? AND user_id = ?", (hold_id, user_id)
        )
        return HoldRecord.model_validate(dict(row)) if row else None


class RunRepo:
    """Whose each run (and so each trace file) is."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def register(self, run_id: str, *, user_id: str, thread_id: str | None, kind: str) -> None:
        if not user_id:
            raise ValueError("a run belongs to a user")
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, user_id, thread_id, kind, created_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT (run_id) DO NOTHING",
                (run_id, user_id, thread_id, kind, self._db.timestamp()),
            )

    def get(self, run_id: str, *, user_id: str) -> Run | None:
        row = self._db.fetch_one(
            "SELECT * FROM runs WHERE run_id = ? AND user_id = ?", (run_id, user_id)
        )
        return Run.model_validate(dict(row)) if row else None


class Store:
    """The agent DB, migrated on open, with one repository per table. Every query names a user."""

    def __init__(self, path: Path | str, *, now: Callable[[], datetime] = utcnow) -> None:
        self.db = Database(path, now=now, scoped_tables=USER_TABLES)
        self.tasks = TaskRepo(self.db)
        self.approvals = ApprovalRepo(self.db)
        self.ledger = LedgerRepo(self.db)
        self.memory = MemoryRepo(self.db)
        self.outbox = OutboxRepo(self.db)
        self.messages = MessageRepo(self.db)
        self.holds = HoldRepo(self.db)
        self.runs = RunRepo(self.db)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
