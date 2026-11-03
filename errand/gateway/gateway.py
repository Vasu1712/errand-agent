"""The tool gateway: the only path from the agent to a provider tool.

Read tools pass through budgeted and traced. Side effects (book, cancel, reschedule) need a
granted, unconsumed approval whose material hash matches what the call would actually do. The
material is worked out here from the gateway's own records of tool data (the hold it obtained,
the booking as the provider lists it now), never from model output or caller-supplied details.
A hold needs a granted approval for its provider and slot too; it doesn't spend the approval.

The approval is spent and a ledger row opened in one transaction before the tool is called. The
row's key, derived from the approval id, is the idempotency key for every retry, and the row ends
``confirmed`` only once the source of truth (get_booking_status, or list_bookings for changes) has
shown the result.
"""

from typing import Any, NoReturn

from pydantic import BaseModel

from errand.budget import BudgetExceeded, RunBudget
from errand.clock import Clock
from errand.contracts import (
    Booking,
    BookingList,
    BookingState,
    BookingStatus,
    CancelResult,
    DateRange,
    Hold,
    ProviderDetails,
    ProviderTools,
    RescheduleResult,
    SearchResult,
    Slot,
    ToolError,
    ToolErrorCode,
    ToolFailure,
    ToolName,
)
from errand.gateway.material import (
    booking_material,
    cancel_material,
    canonical_slot,
    reschedule_material,
)
from errand.gateway.untrusted import sanitize_text
from errand.store import Action, ApprovalError, ApprovalStatus, LedgerEntry, LedgerStatus, Store
from errand.store.records import canonical_json, material_hash
from errand.trace import EventKind, Tracer


class ApprovalRequired(Exception):
    """A side effect was attempted without a granted, unconsumed approval for exactly these
    details. The tool was not called."""


class Execution(BaseModel):
    """Where one side effect stands, as the ledger records it."""

    idempotency_key: str
    op: Action
    status: LedgerStatus
    booking: Booking | None = None
    failure: ToolFailure | None = None
    attempts: int = 0


def idempotency_key_for(approval_id: str) -> str:
    return f"idem_{approval_id}"


class ToolGateway:
    def __init__(
        self,
        tools: ProviderTools,
        store: Store,
        tracer: Tracer,
        budget: RunBudget,
        *,
        user_id: str,
        clock: Clock,
        backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0),
        max_book_attempts: int = 2,
        max_cancel_attempts: int = 3,
    ) -> None:
        self._tools = tools
        self._store = store
        self._tracer = tracer
        self._budget = budget
        self.user_id = user_id
        self._clock = clock
        self._backoff = backoff_s
        self._max_attempts = {
            Action.BOOK: max_book_attempts,
            Action.CANCEL: max_cancel_attempts,
            Action.RESCHEDULE: max_cancel_attempts,
        }

    # Read tools

    def search_providers(self, service: str, area: str, date_range: DateRange) -> SearchResult:
        """Providers with their names and attribute text sanitized, like get_provider's."""
        result = self._call(
            ToolName.SEARCH_PROVIDERS,
            self._tools.search_providers,
            service=service,
            area=area,
            date_range=date_range,
        )
        return result.model_copy(
            update={
                "providers": [
                    p.model_copy(
                        update={
                            "name": sanitize_text(p.name, max_chars=80),
                            "attributes": _clean_attributes(p.attributes),
                        }
                    )
                    for p in result.providers
                ]
            }
        )

    def get_provider(self, provider_id: str) -> ProviderDetails:
        """Provider details with every free-text field sanitized. Callers still wrap the text in
        an untrusted block before any model sees it."""
        details = self._call(
            ToolName.GET_PROVIDER, self._tools.get_provider, provider_id=provider_id
        )
        return details.model_copy(
            update={
                "name": sanitize_text(details.name, max_chars=80),
                "about": sanitize_text(details.about),
                "attributes": _clean_attributes(details.attributes),
                "reviews": [
                    review.model_copy(update={"text": sanitize_text(review.text)})
                    for review in details.reviews
                ],
            }
        )

    def list_bookings(self) -> BookingList:
        return self._call(ToolName.LIST_BOOKINGS, self._tools.list_bookings, user_id=self.user_id)

    def booking_status(self, *, idempotency_key: str) -> BookingStatus:
        return self._call(
            ToolName.GET_BOOKING_STATUS,
            self._tools.get_booking_status,
            idempotency_key=idempotency_key,
        )

    # Holds

    def hold_slot(
        self, provider_id: str, slot: Slot, *, approval_id: str, task_id: str | None
    ) -> Hold:
        """Hold a slot for a booking the user has approved: the approval must be granted, unspent,
        and for this provider and slot. It is not spent here. book() checks the price again, against
        the hold's quote, so a quote that differs from the approved price can't be booked on it."""
        wanted = {"provider_id": provider_id, "slot": canonical_slot(slot)}
        approval = self._store.approvals.get(approval_id, user_id=self.user_id)
        if approval is None:
            self._refuse(approval_id, ToolName.HOLD_SLOT, "there is no such approval", wanted)
        if approval.action is not Action.BOOK:
            self._refuse(
                approval_id, ToolName.HOLD_SLOT, f"the approval is for {approval.action}", wanted
            )
        if approval.status is not ApprovalStatus.GRANTED:
            self._refuse(
                approval_id, ToolName.HOLD_SLOT, f"the approval is {approval.status}", wanted
            )
        approved = {key: approval.material.get(key) for key in wanted}
        if approved != wanted:
            self._refuse(
                approval_id,
                ToolName.HOLD_SLOT,
                "the provider or slot differs from the approved ones",
                wanted,
            )
        hold = self._call(
            ToolName.HOLD_SLOT,
            self._tools.hold_slot,
            approval_id=approval_id,
            provider_id=provider_id,
            slot=slot,
        )
        self._store.holds.record(hold, user_id=self.user_id, task_id=task_id)
        return hold

    # Side effects

    def book(self, hold_id: str, *, approval_id: str) -> Execution:
        hold = self._store.holds.get(hold_id, user_id=self.user_id)
        if hold is None:
            self._refuse(
                approval_id, Action.BOOK, f"hold {hold_id} was not obtained through the gateway"
            )
        material = booking_material(
            provider_id=hold.provider_id,
            slot=Slot(start=hold.slot_start, end=hold.slot_end),
            price_inr=hold.quoted_price_inr,
        )
        return self._side_effect(Action.BOOK, approval_id, material, {"hold_id": hold_id})

    def cancel(self, booking_id: str, *, approval_id: str) -> Execution:
        existing = self._store.ledger.for_approval(approval_id, user_id=self.user_id)
        if existing is not None:
            return self._resume(existing, Action.CANCEL, {"booking_id": booking_id})
        material = cancel_material(self._current_booking(booking_id))
        return self._side_effect(Action.CANCEL, approval_id, material, {"booking_id": booking_id})

    def reschedule(self, booking_id: str, slot: Slot, *, approval_id: str) -> Execution:
        request = {"booking_id": booking_id, "slot": canonical_slot(slot)}
        existing = self._store.ledger.for_approval(approval_id, user_id=self.user_id)
        if existing is not None:
            return self._resume(existing, Action.RESCHEDULE, request)
        material = reschedule_material(self._current_booking(booking_id), slot)
        return self._side_effect(Action.RESCHEDULE, approval_id, material, request)

    def verify(self, idempotency_key: str) -> Execution:
        """Settle a ledger row from the source of truth.

        Status queries back off when the source of truth itself fails. An ambiguous attempt that
        turns out not to have happened is retried under the same key, a bounded number of times.
        Anything still unclear, including a budget that runs out, leaves the row ``unknown`` for
        the reconciler.
        """
        entry = self._entry(idempotency_key)
        if entry.status in (LedgerStatus.CONFIRMED, LedgerStatus.NOT_BOOKED):
            return self._execution(entry)  # both are final
        try:
            if entry.op is Action.BOOK:
                return self._verify_booking(entry)
            return self._verify_change(entry)
        except BudgetExceeded as exc:
            return self._settle(self._entry(idempotency_key), LedgerStatus.UNKNOWN, error=str(exc))

    # Internals

    def _side_effect(
        self, op: Action, approval_id: str, material: dict[str, Any], request: dict[str, Any]
    ) -> Execution:
        existing = self._store.ledger.for_approval(approval_id, user_id=self.user_id)
        if existing is not None:
            return self._resume(existing, op, request)
        digest = material_hash(material)
        approval = self._store.approvals.get(approval_id, user_id=self.user_id)
        if approval is None:
            self._refuse(approval_id, op, "there is no such approval", material)
        if approval.action is not op:
            self._refuse(approval_id, op, f"the approval is for {approval.action}", material)
        if approval.status is not ApprovalStatus.GRANTED:
            self._refuse(approval_id, op, f"the approval is {approval.status}", material)
        if approval.material_hash != digest:
            self._refuse(
                approval_id, op, "the call's details differ from the approved ones", material
            )
        key = idempotency_key_for(approval_id)
        try:
            # One BEGIN IMMEDIATE transaction: the approval is spent if and only if the ledger
            # row exists, and the schema's unique indexes refuse a second row for it.
            with self._store.db.transaction():
                self._store.approvals.consume(approval_id, digest, user_id=self.user_id)
                self._store.ledger.begin(
                    key,
                    op,
                    user_id=self.user_id,
                    request=request,
                    task_id=approval.task_id,
                    approval_id=approval_id,
                )
        except ApprovalError as exc:
            self._refuse(approval_id, op, str(exc), material)
        self._tracer.emit(
            EventKind.APPROVAL,
            event="consumed",
            approval_id=approval_id,
            material_hash=digest,
            material=material,
        )
        self._tracer.emit(
            EventKind.LEDGER, event="in_flight", idempotency_key=key, op=op, request=request
        )
        return self._attempt(key)

    def _resume(self, entry: LedgerEntry, op: Action, request: dict[str, Any]) -> Execution:
        """The approval was already spent on one operation: settle that row, never open another."""
        if entry.op is not op or canonical_json(entry.request) != canonical_json(request):
            self._refuse(
                entry.approval_id or "", op, "the approval was spent on a different request"
            )
        self._tracer.emit(
            EventKind.LEDGER,
            event="resumed",
            idempotency_key=entry.idempotency_key,
            status=entry.status,
        )
        return self.verify(entry.idempotency_key)

    def _attempt(self, key: str) -> Execution:
        """Call the tool under the row's key, retrying transient failures with backoff."""
        entry = self._entry(key)
        for retry in range(self._max_attempts[entry.op]):
            # Only count an attempt the budget lets through: a refused one must not use up a
            # retry the reconciler could still make (docs/bugs.md #17).
            self._budget.check_tool(reserved=True)
            entry = self._store.ledger.record_attempt(key, user_id=self.user_id)
            try:
                booking = self._perform(entry)
            except ToolError as exc:
                failure = exc.failure
                last = retry + 1 >= self._max_attempts[entry.op]
                if failure.retryable and not last:
                    self._tracer.emit(
                        EventKind.LEDGER,
                        event="retry",
                        idempotency_key=key,
                        attempt=entry.attempts,
                        failure=failure,
                    )
                    self._clock.sleep(self._backoff[min(retry, len(self._backoff) - 1)])
                    continue
                # A timeout or an exhausted transient error may still have taken effect.
                unclear = failure.outcome_unknown or failure.retryable
                status = LedgerStatus.UNKNOWN if unclear else LedgerStatus.NOT_BOOKED
                return self._settle(entry, status, failure=failure)
            # A success response is recorded but not yet trusted: verify() confirms it.
            entry = self._store.ledger.mark(
                key,
                LedgerStatus.IN_FLIGHT,
                user_id=self.user_id,
                result={"booking": booking.model_dump(mode="json")},
            )
            return self._execution(entry)
        raise AssertionError("unreachable: the last attempt always settles")

    def _perform(self, entry: LedgerEntry) -> Booking:
        request = entry.request
        if entry.op is Action.BOOK:
            return self._call(
                ToolName.BOOK,
                self._tools.book,
                reserved=True,
                approval_id=entry.approval_id,
                hold_id=request["hold_id"],
                idempotency_key=entry.idempotency_key,
            )
        if entry.op is Action.CANCEL:
            return self._call(
                ToolName.CANCEL,
                self._tools.cancel,
                reserved=True,
                approval_id=entry.approval_id,
                booking_id=request["booking_id"],
                idempotency_key=entry.idempotency_key,
            ).booking
        return self._call(
            ToolName.RESCHEDULE,
            self._tools.reschedule,
            reserved=True,
            approval_id=entry.approval_id,
            booking_id=request["booking_id"],
            slot=Slot.model_validate(request["slot"]),
            idempotency_key=entry.idempotency_key,
        ).booking

    def _verify_booking(self, entry: LedgerEntry) -> Execution:
        key = entry.idempotency_key
        while True:
            status = self._with_backoff(
                lambda: self._call(
                    ToolName.GET_BOOKING_STATUS,
                    self._tools.get_booking_status,
                    reserved=True,
                    idempotency_key=key,
                )
            )
            if status is None:
                return self._settle(
                    entry, LedgerStatus.UNKNOWN, error="get_booking_status unreachable"
                )
            if status.booked:
                return self._settle(entry, LedgerStatus.CONFIRMED, booking=status.booking)
            if entry.status in _UNSETTLED and entry.attempts < self._max_attempts[entry.op]:
                # Verified not booked after an ambiguous attempt (a timeout, or a crash that left
                # the row in flight): retry under the same key, which the provider dedupes, so a
                # retry can never book twice.
                retried = self._attempt(key)
                if retried.status is LedgerStatus.NOT_BOOKED:
                    return retried
                entry = self._entry(key)
                continue
            return self._settle(
                entry,
                LedgerStatus.NOT_BOOKED,
                error=entry.last_error or "the provider has no booking for this key",
            )

    def _verify_change(self, entry: LedgerEntry) -> Execution:
        key = entry.idempotency_key
        booking_id = entry.request["booking_id"]
        while True:
            listing = self._with_backoff(
                lambda: self._call(
                    ToolName.LIST_BOOKINGS,
                    self._tools.list_bookings,
                    reserved=True,
                    user_id=self.user_id,
                )
            )
            if listing is None:
                return self._settle(entry, LedgerStatus.UNKNOWN, error="list_bookings unreachable")
            booking = next((b for b in listing.bookings if b.booking_id == booking_id), None)
            if booking is None:
                return self._settle(entry, LedgerStatus.NOT_BOOKED, error="the booking is gone")
            if entry.op is Action.CANCEL:
                done = booking.status in (
                    BookingState.CANCELLED,
                    BookingState.CANCELLED_BY_PROVIDER,
                )
            else:
                done = canonical_slot(booking.slot) == entry.request["slot"]
            if done:
                return self._settle(entry, LedgerStatus.CONFIRMED, booking=booking)
            if entry.status in _UNSETTLED and entry.attempts < self._max_attempts[entry.op]:
                retried = self._attempt(key)
                if retried.status is LedgerStatus.NOT_BOOKED:
                    return retried
                entry = self._entry(key)
                continue
            return self._settle(
                entry, LedgerStatus.NOT_BOOKED, error=entry.last_error or "the change did not apply"
            )

    def _with_backoff(self, ask):
        """The source of truth's answer, asking again with backoff while it errors; None if it
        never answers."""
        for delay in (0.0, *self._backoff):
            if delay:
                self._clock.sleep(delay)
            try:
                return ask()
            except ToolError:
                continue
        return None

    def _current_booking(self, booking_id: str) -> Booking:
        listing = self._call(
            ToolName.LIST_BOOKINGS, self._tools.list_bookings, reserved=True, user_id=self.user_id
        )
        booking = next((b for b in listing.bookings if b.booking_id == booking_id), None)
        if booking is None:
            raise ApprovalRequired(f"booking {booking_id} isn't one of this user's bookings")
        return booking

    def _settle(
        self,
        entry: LedgerEntry,
        status: LedgerStatus,
        *,
        booking: Booking | None = None,
        failure: ToolFailure | None = None,
        error: str | None = None,
    ) -> Execution:
        reason = error or (f"{failure.code}: {failure.message}" if failure else None)
        result = {"booking": booking.model_dump(mode="json")} if booking else None
        if failure is not None:
            result = {"failure": failure.model_dump(mode="json")}
        entry = self._store.ledger.mark(
            entry.idempotency_key, status, user_id=self.user_id, result=result, error=reason
        )
        self._tracer.emit(
            EventKind.LEDGER,
            event=status,
            idempotency_key=entry.idempotency_key,
            op=entry.op,
            attempts=entry.attempts,
            booking_id=booking.booking_id if booking else None,
            error=reason,
        )
        return self._execution(entry)

    def _execution(self, entry: LedgerEntry) -> Execution:
        result = entry.result or {}
        return Execution(
            idempotency_key=entry.idempotency_key,
            op=entry.op,
            status=entry.status,
            booking=Booking.model_validate(result["booking"]) if "booking" in result else None,
            failure=ToolFailure.model_validate(result["failure"]) if "failure" in result else None,
            attempts=entry.attempts,
        )

    def _entry(self, key: str) -> LedgerEntry:
        entry = self._store.ledger.get(key, user_id=self.user_id)
        if entry is None:
            raise KeyError(key)
        return entry

    def _refuse(
        self, approval_id: str, op: str, reason: str, material: dict[str, Any] | None = None
    ) -> NoReturn:
        self._tracer.emit(
            EventKind.APPROVAL,
            event="refused",
            approval_id=approval_id,
            op=op,
            reason=reason,
            material=material,
        )
        raise ApprovalRequired(reason)

    def _call(
        self,
        tool: ToolName,
        fn,
        *,
        reserved: bool = False,
        approval_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """One traced, budgeted tool call. `approval_id` (for holds and side effects) is recorded
        with the call, not passed to the tool."""
        self._budget.spend_tool(reserved=reserved)
        if hasattr(self._tools, "set_call_timeout"):
            # The client gives up when the run's time does: no single call may outlast the
            # run's hard limit (docs/bugs.md #18). The mock world honours it.
            self._tools.set_call_timeout(max(self._budget.remaining_s(reserved=reserved), 0.0))
        under = {"approval_id": approval_id} if approval_id else {}
        with self._tracer.span(EventKind.TOOL_CALL, tool=tool, args=kwargs, **under) as span:
            try:
                result = fn(**kwargs)
            except ToolError as exc:
                span["failure"] = exc.failure
                raise
            except Exception as exc:  # the tool broke its contract: a crash, unreadable JSON
                failure = _bad_response(tool, f"{type(exc).__name__}: {exc}")
                span["failure"] = failure
                raise ToolError(failure) from exc
            if not isinstance(result, RETURNS[tool]):
                failure = _bad_response(
                    tool, f"answered with {type(result).__name__}, not {RETURNS[tool].__name__}"
                )
                span["failure"] = failure
                raise ToolError(failure)
            span["result"] = _trace_view(tool, result)
        return result


def _clean_attributes(attributes: dict) -> dict:
    return {
        sanitize_text(str(key), max_chars=40): sanitize_text(value, max_chars=120)
        if isinstance(value, str)
        else value
        for key, value in attributes.items()
    }


RETURNS: dict[ToolName, type[BaseModel]] = {
    ToolName.SEARCH_PROVIDERS: SearchResult,
    ToolName.GET_PROVIDER: ProviderDetails,
    ToolName.HOLD_SLOT: Hold,
    ToolName.BOOK: Booking,
    ToolName.GET_BOOKING_STATUS: BookingStatus,
    ToolName.CANCEL: CancelResult,
    ToolName.RESCHEDULE: RescheduleResult,
    ToolName.LIST_BOOKINGS: BookingList,
}
# Rows whose last attempt may or may not have reached the provider.
_UNSETTLED = (LedgerStatus.UNKNOWN, LedgerStatus.IN_FLIGHT)
# A broken reply to one of these doesn't say whether it took effect.
_MAY_HAVE_ACTED = {ToolName.HOLD_SLOT, ToolName.BOOK, ToolName.CANCEL, ToolName.RESCHEDULE}


def _bad_response(tool: ToolName, message: str) -> ToolFailure:
    return ToolFailure(
        tool=tool,
        code=ToolErrorCode.BAD_RESPONSE,
        message=message[:300],
        outcome_unknown=tool in _MAY_HAVE_ACTED,
    )


def _trace_view(tool: ToolName, result: BaseModel) -> Any:
    """What the trace keeps of a result: search results are summarised, the rest kept whole
    (the tracer sanitizes and truncates them)."""
    if isinstance(result, SearchResult):
        return [
            {
                "provider_id": p.provider_id,
                "name": p.name,
                "price_inr": p.price_inr,
                "rating": p.rating,
                "slots": len(p.slots),
            }
            for p in result.providers
        ]
    return result
