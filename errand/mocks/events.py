"""Simulated world events, such as a provider cancelling a booking, queued in the world DB."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from errand.store.db import Database


class WorldEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: int | None = None
    kind: Literal["provider_cancelled"]
    booking_id: str
    user_id: str
    due_at: AwareDatetime
    reason: str = ""


class EventQueue:
    """Events wait in the world DB until the clock reaches them; each is handed out once."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def push(self, event: WorldEvent) -> WorldEvent:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO events (kind, booking_id, user_id, due_at, reason)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    event.kind,
                    event.booking_id,
                    event.user_id,
                    event.due_at.astimezone(UTC).isoformat(),
                    event.reason,
                ),
            )
        return event.model_copy(update={"event_id": cur.lastrowid})

    def pop_due(self, now: datetime) -> list[WorldEvent]:
        """Remove and return the events due at or before `now`, oldest first."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE consumed_at IS NULL AND due_at <= ?"
                " ORDER BY due_at, event_id",
                (now.astimezone(UTC).isoformat(),),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE events SET consumed_at = ? WHERE event_id = ?",
                    (now.astimezone(UTC).isoformat(), row["event_id"]),
                )
        return [_event(row) for row in rows]

    def pending(self) -> list[WorldEvent]:
        rows = self._db.fetch_all(
            "SELECT * FROM events WHERE consumed_at IS NULL ORDER BY due_at, event_id"
        )
        return [_event(row) for row in rows]


def _event(row) -> WorldEvent:
    return WorldEvent(
        event_id=row["event_id"],
        kind=row["kind"],
        booking_id=row["booking_id"],
        user_id=row["user_id"],
        due_at=datetime.fromisoformat(row["due_at"]),
        reason=row["reason"],
    )
