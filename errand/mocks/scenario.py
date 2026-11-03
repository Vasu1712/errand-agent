"""Scenario files (scenarios/*.yaml): how the mock world behaves, including forced outcomes."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from errand.contracts import ToolName


class BookOutcome(StrEnum):
    SUCCESS = "success"
    SLOT_TAKEN = "slot_taken"
    TIMEOUT_BOOKED = "timeout_booked"  # the call times out, but the booking went through
    TIMEOUT_NOT_BOOKED = "timeout_not_booked"
    # Fault injection for restart tests: commit the booking, then die like a killed process.
    CRASH_AFTER_COMMIT = "crash_after_commit"
    # ... or die after the agent's ledger write, before the provider has seen the request.
    CRASH_BEFORE_SEND = "crash_before_send"


CRASHES = (BookOutcome.CRASH_AFTER_COMMIT, BookOutcome.CRASH_BEFORE_SEND)


class Fault(BaseModel):
    """One broken tool call: the provider's API raises, answers with unreadable JSON, or answers
    with the wrong shape. With ``after``, the call takes effect first and only its reply is lost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: ToolName
    call: int = Field(1, ge=1)  # which call to this tool, counted over the world's life
    kind: Literal["exception", "malformed_json", "wrong_type"]
    after: bool = False


# The spec's split: success 70%, slot_taken 15%, timeout 15% with half of those booked anyway.
DEFAULT_BOOK_WEIGHTS = {
    BookOutcome.SUCCESS: 70.0,
    BookOutcome.SLOT_TAKEN: 15.0,
    BookOutcome.TIMEOUT_BOOKED: 7.5,
    BookOutcome.TIMEOUT_NOT_BOOKED: 7.5,
}


class Scenario(BaseModel):
    """One mock world. Defaults follow the spec; any field can be overridden per scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default"
    description: str = ""
    # None: use the app's seed.
    seed: int | None = None

    # book(): weighted random outcomes, after any forced ones (used up in order, one per call
    # that reaches an outcome; calls answered from the idempotency key don't use one).
    book_weights: dict[BookOutcome, float] = Field(
        default_factory=lambda: dict(DEFAULT_BOOK_WEIGHTS)
    )
    book_forced: list[BookOutcome] = []
    timeout_latency_s: float = Field(2.0, ge=0)

    # hold_slot(): the quote is the search price plus a delta. A fixed delta per provider id
    # ("*" for all) wins; otherwise a seeded share of quotes drift upwards.
    quote_delta_inr: dict[str, int] = {}
    quote_drift_probability: float = Field(0.2, ge=0, le=1)
    quote_drift_max_pct: int = Field(15, ge=0, le=100)
    hold_ttl_s: int = Field(600, gt=0)
    # Prices move: holding a slot again quotes this much more than its previous hold did.
    rehold_delta_inr: int = 0

    # get_booking_status(): this many calls fail with a 503 before it answers again.
    status_unavailable_calls: int = Field(0, ge=0)

    # cancel(): this many calls per booking fail with a transient 503 before one succeeds.
    cancel_fail_first: int = Field(1, ge=0)

    # reschedule(): slots that always conflict, as "HH:MM" (any day) or "YYYY-MM-DDTHH:MM" (IST).
    reschedule_conflicts: list[str] = []

    # Providers whose reviews carry a prompt injection; empty means one per service and area.
    injection_providers: list[str] = []
    # Replacement name, about or attributes for some providers (for injections in other fields).
    provider_overrides: dict[str, dict[str, Any]] = {}

    # Tool calls that break: a crash, unreadable JSON or a reply of the wrong shape.
    faults: list[Fault] = []

    @model_validator(mode="after")
    def _sane_weights(self) -> Self:
        if any(weight < 0 for weight in self.book_weights.values()):
            raise ValueError("book_weights must not be negative")
        if sum(self.book_weights.values()) <= 0:
            raise ValueError("book_weights must not all be zero")
        if any(crash in self.book_weights for crash in CRASHES):
            raise ValueError("crash outcomes can only be forced, never drawn at random")
        unknown = {key for o in self.provider_overrides.values() for key in o} - {
            "name",
            "about",
            "attributes",
        }
        if unknown:
            raise ValueError(
                f"provider_overrides can set name, about and attributes, not {unknown}"
            )
        return self


def load_scenario(name_or_path: str | Path, *, scenarios_dir: Path) -> Scenario:
    """Load ``scenarios_dir/<name>.yaml``, or an explicit path to a YAML file.

    The name "default" without a file on disk is the spec's default world.
    """
    candidate = Path(name_or_path)
    path = (
        candidate
        if candidate.suffix in {".yaml", ".yml"}
        else scenarios_dir / f"{name_or_path}.yaml"
    )
    if not path.is_file():
        if str(name_or_path) == "default":
            return Scenario()
        raise FileNotFoundError(f"no scenario file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("name", path.stem)
    return Scenario.model_validate(data)
