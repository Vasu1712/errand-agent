"""Pydantic records for the agent DB rows. The enums mirror the schema's CHECK constraints."""

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue


class Action(StrEnum):
    """The consequential actions: each needs an approval and goes through the ledger."""

    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


class LedgerStatus(StrEnum):
    IN_FLIGHT = "in_flight"
    CONFIRMED = "confirmed"
    NOT_BOOKED = "not_booked"
    UNKNOWN = "unknown"


class FactSource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class FactStatus(StrEnum):
    ACTIVE = "active"
    UNCONFIRMED = "unconfirmed"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


CURRENT_FACT_STATUSES = (FactStatus.ACTIVE, FactStatus.UNCONFIRMED)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class Task(Record):
    task_id: str
    user_id: str
    thread_id: str
    intent: Action
    phase: str
    details: dict[str, JsonValue]
    booking_id: str | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class Approval(Record):
    approval_id: str
    user_id: str
    task_id: str
    action: Action
    material: dict[str, JsonValue]
    material_hash: str
    status: ApprovalStatus
    reason: str | None
    created_at: AwareDatetime
    granted_at: AwareDatetime | None
    consumed_at: AwareDatetime | None
    invalidated_at: AwareDatetime | None


class LedgerEntry(Record):
    idempotency_key: str
    user_id: str
    op: Action
    task_id: str | None
    approval_id: str | None
    request: dict[str, JsonValue]
    status: LedgerStatus
    attempts: int
    result: dict[str, JsonValue] | None
    last_error: str | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class MemoryFact(Record):
    fact_id: str
    user_id: str
    key: str
    value: JsonValue
    service_tag: str | None
    source: FactSource
    status: FactStatus
    evidence: str | None
    superseded_by: str | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class OutboxMessage(Record):
    message_id: int
    user_id: str
    task_id: str | None
    kind: str
    body: str
    payload: dict[str, JsonValue]
    dedupe_key: str | None
    created_at: AwareDatetime
    delivered_at: AwareDatetime | None


class Message(Record):
    message_id: int
    user_id: str
    thread_id: str
    run_id: str | None
    role: Literal["user", "assistant"]
    content: str
    created_at: AwareDatetime
    for_model: bool = True  # False: shown to the user, never sent to a model


class HoldRecord(Record):
    hold_id: str
    user_id: str
    task_id: str | None
    provider_id: str
    slot_start: AwareDatetime
    slot_end: AwareDatetime
    quoted_price_inr: int
    expires_at: AwareDatetime
    created_at: AwareDatetime


class Run(Record):
    """One agent run (one user message or one background event): whose trace it is."""

    run_id: str
    user_id: str
    thread_id: str | None
    kind: str
    created_at: AwareDatetime


def canonical_json(value: Any) -> str:
    """Key-sorted, whitespace-free JSON: equal content always gives equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def material_hash(material: Mapping[str, Any]) -> str:
    """Fingerprint of what the user is asked to approve. Any change to it means a new approval."""
    return hashlib.sha256(canonical_json(material).encode()).hexdigest()
