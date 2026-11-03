"""Tool contracts: inputs, outputs and failures of the eight provider tools.

The spec fixes their names, inputs, outputs and failure behaviour. The mock world implements
`ProviderTools`; the gateway is its only caller. Free text in tool output (reviews, "about") is
untrusted data, never instructions.

A tool returns its output model or raises `ToolError`. A ``timeout`` means the outcome is unknown:
the side effect may have happened, so ask ``get_booking_status`` before retrying or reporting.
"""

from datetime import date, timedelta, timezone
from enum import StrEnum
from typing import Protocol, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

# The mock world runs on Indian Standard Time (no DST, so a fixed offset is exact).
IST = timezone(timedelta(hours=5, minutes=30), "IST")


class ToolName(StrEnum):
    SEARCH_PROVIDERS = "search_providers"
    GET_PROVIDER = "get_provider"
    HOLD_SLOT = "hold_slot"
    BOOK = "book"
    GET_BOOKING_STATUS = "get_booking_status"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"
    LIST_BOOKINGS = "list_bookings"


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DateRange(Contract):
    """Calendar days, both ends inclusive."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("date_range ends before it starts")
        return self


class Slot(Contract):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("a slot must end after it starts")
        return self


Attributes = dict[str, str | int | float | bool]


class ProviderSummary(Contract):
    provider_id: str
    name: str
    service: str
    area: str
    price_inr: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)
    slots: list[Slot]
    attributes: Attributes = {}


class SearchResult(Contract):
    providers: list[ProviderSummary] = Field(min_length=3, max_length=8)


class Review(Contract):
    review_id: str
    author: str
    rating: int = Field(ge=1, le=5)
    text: str  # untrusted
    posted_on: date


class ProviderDetails(Contract):
    provider_id: str
    name: str
    service: str
    area: str
    price_inr: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    about: str  # untrusted
    technicians: list[str] = []
    attributes: Attributes = {}
    reviews: list[Review]


class Hold(Contract):
    hold_id: str
    provider_id: str
    slot: Slot
    quoted_price_inr: int = Field(ge=0)  # may differ from the price search_providers showed
    expires_at: AwareDatetime


class BookingState(StrEnum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    CANCELLED_BY_PROVIDER = "cancelled_by_provider"
    COMPLETED = "completed"


class Booking(Contract):
    booking_id: str
    user_id: str
    provider_id: str
    provider_name: str
    service: str
    slot: Slot
    price_inr: int = Field(ge=0)
    status: BookingState
    technician: str | None = None


class BookingStatus(Contract):
    """The source of truth for an ambiguous booking: did this hold or key produce a booking?"""

    booked: bool
    booking: Booking | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.booked != (self.booking is not None):
            raise ValueError("booked must be true exactly when a booking is returned")
        return self


class CancelResult(Contract):
    booking: Booking


class RescheduleResult(Contract):
    booking: Booking
    previous_slot: Slot


class BookingList(Contract):
    bookings: list[Booking]


class ToolErrorCode(StrEnum):
    NOT_FOUND = "not_found"
    INVALID_ARGUMENT = "invalid_argument"
    SLOT_TAKEN = "slot_taken"  # book
    HOLD_EXPIRED = "hold_expired"  # book after the hold's expiry
    TIMEOUT = "timeout"  # book: outcome unknown
    SERVICE_UNAVAILABLE = "service_unavailable"  # cancel's transient 503
    SLOT_CONFLICT = "slot_conflict"  # reschedule
    # Set by the gateway, never by a tool: the call raised, or answered outside its contract
    # (unreadable JSON, the wrong shape). For a side effect, the outcome is then unknown.
    BAD_RESPONSE = "bad_response"


class ToolFailure(Contract):
    tool: ToolName
    code: ToolErrorCode
    message: str = ""
    # Repeating the same call (same idempotency key) is safe and may succeed.
    retryable: bool = False
    # The side effect may have happened; check the source of truth before anything else.
    outcome_unknown: bool = False
    http_status: int | None = None


class ToolError(Exception):
    def __init__(self, failure: ToolFailure) -> None:
        detail = f": {failure.message}" if failure.message else ""
        super().__init__(f"{failure.tool} failed with {failure.code}{detail}")
        self.failure = failure

    @classmethod
    def of(cls, tool: ToolName, code: ToolErrorCode, message: str = "") -> Self:
        """A failure with the flags the contract implies for its code."""
        unavailable = code is ToolErrorCode.SERVICE_UNAVAILABLE
        return cls(
            ToolFailure(
                tool=tool,
                code=code,
                message=message,
                retryable=unavailable,
                outcome_unknown=code is ToolErrorCode.TIMEOUT,
                http_status=503 if unavailable else None,
            )
        )


class ProviderTools(Protocol):
    """The eight tools, exactly as the spec names them."""

    def search_providers(self, service: str, area: str, date_range: DateRange) -> SearchResult: ...

    def get_provider(self, provider_id: str) -> ProviderDetails: ...

    def hold_slot(self, provider_id: str, slot: Slot) -> Hold: ...

    def book(self, hold_id: str, idempotency_key: str) -> Booking: ...

    def get_booking_status(
        self, *, hold_id: str | None = None, idempotency_key: str | None = None
    ) -> BookingStatus: ...

    def cancel(self, booking_id: str, idempotency_key: str) -> CancelResult: ...

    def reschedule(self, booking_id: str, slot: Slot, idempotency_key: str) -> RescheduleResult: ...

    def list_bookings(self, user_id: str) -> BookingList: ...
