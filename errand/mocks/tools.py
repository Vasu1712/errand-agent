"""The mock provider world behind the eight tools.

Deterministic: every random outcome comes from an RNG seeded with the world seed plus the call's
identity (tool, idempotency key, attempt), so results don't depend on call order and are the same
after a restart. A scenario can force any outcome. State lives in the world's own SQLite file and
time comes from an injected clock.

Failure results that must survive the failure (a timeout that booked anyway, a 503 that still
counts as an attempt) are committed before the error is raised.
"""

import copy
import dataclasses
import json
import random
from contextlib import nullcontext
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Self

from errand.clock import Clock
from errand.contracts import (
    IST,
    Booking,
    BookingList,
    BookingState,
    BookingStatus,
    CancelResult,
    DateRange,
    Hold,
    ProviderDetails,
    ProviderSummary,
    RescheduleResult,
    Review,
    SearchResult,
    Slot,
    ToolError,
    ToolErrorCode,
    ToolName,
)
from errand.mocks.catalog import (
    AREAS,
    SERVICES,
    SLOT_LENGTH,
    base_open,
    generate_catalog,
    grid_starts,
    normalize_area,
    on_grid,
)
from errand.mocks.events import EventQueue, WorldEvent
from errand.mocks.faults import FaultyTools
from errand.mocks.scenario import BookOutcome, Scenario
from errand.mocks.schema import WORLD_MIGRATIONS
from errand.mocks.turnstile import Turnstile
from errand.store.db import Database

MAX_SEARCH_DAYS = 14
MAX_SLOTS_PER_PROVIDER = 12

_SERVICE_ALIASES = {
    "plumbing": "plumber",
    "electrical": "electrician",
    "ac": "ac_service",
    "ac_servicing": "ac_service",
    "ac_repair": "ac_service",
    "carpentry": "carpenter",
    "cleaner": "cleaning",
    "home_cleaning": "cleaning",
    "painting": "painter",
    "pest": "pest_control",
}


class SimulatedCrash(BaseException):
    """The crash_after_commit outcome: the booking is committed, then the process "dies".

    A BaseException, so no handler written for ordinary errors can swallow it.
    """


def _ts(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _slot(start: datetime) -> Slot:
    return Slot(start=start.astimezone(IST), end=(start + SLOT_LENGTH).astimezone(IST))


class MockWorld:
    """Implements `errand.contracts.ProviderTools` against a scenario.

    hold_slot and book carry no user id, so each view of the world is bound to one user
    (`for_user`); cancel and reschedule only touch that user's bookings.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        db_path: Path | str,
        clock: Clock,
        seed: int = 0,
        user_id: str = "anonymous",
    ) -> None:
        self.scenario = scenario
        self.clock = clock
        self.seed = scenario.seed if scenario.seed is not None else seed
        self.user_id = user_id
        # The caller's client timeout for its next call (an HTTP client's deadline, in a real
        # deployment); None waits as long as the provider takes.
        self.call_timeout_s: float | None = None
        # Turnstiles by point ("hold_slot", "book"), shared by every user's view: tests pin which
        # concurrent caller claims a slot first. Empty in the app.
        self.gates: dict[str, Turnstile] = {}
        self._db = Database(db_path, now=clock.now, migrations=WORLD_MIGRATIONS)
        self.events = EventQueue(self._db)
        self._initialise()

    def for_user(self, user_id: str) -> Self:
        """The same world (shared state and clock), acting for `user_id`. With faults in the
        scenario, a view whose tools break on cue."""
        view = copy.copy(self)
        view.user_id = user_id
        return FaultyTools(view) if self.scenario.faults else view

    def close(self) -> None:
        self._db.close()

    def set_call_timeout(self, seconds: float | None) -> None:
        self.call_timeout_s = seconds

    # Read-only tools

    def search_providers(self, service: str, area: str, date_range: DateRange) -> SearchResult:
        """3 to 8 providers with ids, prices, ratings, available slots and attributes."""
        tool = ToolName.SEARCH_PROVIDERS
        service_key = self._service(service, tool)
        canonical_area = normalize_area(area)
        if canonical_area is None:
            raise ToolError.of(
                tool,
                ToolErrorCode.INVALID_ARGUMENT,
                f"unknown area {area!r}; known areas: {', '.join(AREAS)}",
            )
        days = self._days(date_range, tool)
        rows = self._db.fetch_all(
            "SELECT * FROM providers WHERE service = ? AND area = ? ORDER BY provider_id",
            (service_key, canonical_area),
        )
        blocked = self._blocked([row["provider_id"] for row in rows])
        return SearchResult(providers=[self._summary(row, days, blocked) for row in rows])

    def get_provider(self, provider_id: str) -> ProviderDetails:
        """Details and free-text reviews; at least one review carries a prompt-injection attempt."""
        row = self._provider_row(provider_id, ToolName.GET_PROVIDER)
        reviews = self._db.fetch_all(
            "SELECT * FROM reviews WHERE provider_id = ? ORDER BY review_id", (provider_id,)
        )
        return ProviderDetails(
            provider_id=row["provider_id"],
            name=row["name"],
            service=row["service"],
            area=row["area"],
            price_inr=row["price_inr"],
            rating=row["rating"],
            about=row["about"],
            technicians=json.loads(row["technicians"]),
            attributes=json.loads(row["attributes"]),
            reviews=[
                Review(
                    review_id=review["review_id"],
                    author=review["author"],
                    rating=review["rating"],
                    text=review["text"],
                    posted_on=date.fromisoformat(review["posted_on"]),
                )
                for review in reviews
            ],
        )

    def get_booking_status(
        self, *, hold_id: str | None = None, idempotency_key: str | None = None
    ) -> BookingStatus:
        """Source of truth for ambiguous booking outcomes. Takes exactly one of the two ids."""
        if (hold_id is None) == (idempotency_key is None):
            raise ToolError.of(
                ToolName.GET_BOOKING_STATUS,
                ToolErrorCode.INVALID_ARGUMENT,
                "pass exactly one of hold_id and idempotency_key",
            )
        if self._bump("status", "calls") <= self.scenario.status_unavailable_calls:
            raise ToolError.of(
                ToolName.GET_BOOKING_STATUS,
                ToolErrorCode.SERVICE_UNAVAILABLE,
                "the booking status service is unavailable",
            )
        if hold_id is not None:
            booking = self._booking_where("b.hold_id = ?", hold_id)
        else:
            booking = self._booking_where("b.idempotency_key = ?", idempotency_key)
        return BookingStatus(booked=booking is not None, booking=booking)

    def list_bookings(self, user_id: str) -> BookingList:
        """Past and upcoming bookings with id, provider, service, slot and status."""
        rows = self._db.fetch_all(
            "SELECT b.*, p.name AS provider_name FROM bookings b"
            " JOIN providers p USING (provider_id) WHERE b.user_id = ?"
            " ORDER BY b.slot_start, b.booking_id",
            (user_id,),
        )
        return BookingList(bookings=[self._booking(row) for row in rows])

    def booking_rows(self) -> list[dict]:
        """Every booking row of every user, as stored (not a tool): for eval invariants that
        check the provider's side, such as "never two bookings for one idempotency key"."""
        return [dict(row) for row in self._db.fetch_all("SELECT * FROM bookings ORDER BY rowid")]

    # Holds and side effects

    def hold_slot(self, provider_id: str, slot: Slot) -> Hold:
        """Hold for 10 minutes. The quoted price may differ from the search price (scenario)."""
        tool = ToolName.HOLD_SLOT
        provider = self._provider_row(provider_id, tool)
        if not on_grid(slot.start, slot.end):
            raise ToolError.of(
                tool, ToolErrorCode.INVALID_ARGUMENT, "not one of the provider's slots"
            )
        now = self.clock.now()
        with self._gate("hold_slot"), self._db.transaction() as conn:
            if slot.start <= now or not base_open(self.seed, provider_id, slot.start):
                raise ToolError.of(
                    tool, ToolErrorCode.SLOT_TAKEN, "that slot is no longer available"
                )
            held_before = conn.execute(
                "SELECT COUNT(*) FROM holds WHERE provider_id = ? AND slot_start = ?",
                (provider_id, _ts(slot.start)),
            ).fetchone()[0]
            hold_id = f"hold_{self._next('hold')}"
            expires_at = now + timedelta(seconds=self.scenario.hold_ttl_s)
            # The claim: one conditional UPDATE; whoever changes the row has the slot.
            if not self._claim(
                conn, provider_id, slot.start, "held", hold_id=hold_id, held_until=expires_at
            ):
                raise ToolError.of(
                    tool, ToolErrorCode.SLOT_TAKEN, "that slot is no longer available"
                )
            quote = provider["price_inr"] + self._quote_delta(
                provider_id, provider["price_inr"], slot
            )
            if held_before:  # prices moved since the slot was last held
                quote += self.scenario.rehold_delta_inr
            conn.execute(
                "INSERT INTO holds (hold_id, provider_id, slot_start, slot_end, quoted_price_inr,"
                " expires_at, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'active', ?)",
                (
                    hold_id,
                    provider_id,
                    _ts(slot.start),
                    _ts(slot.end),
                    quote,
                    _ts(expires_at),
                    _ts(now),
                ),
            )
        return Hold(
            hold_id=hold_id,
            provider_id=provider_id,
            slot=_slot(slot.start),
            quoted_price_inr=quote,
            expires_at=expires_at.astimezone(IST),
        )

    def book(self, hold_id: str, idempotency_key: str) -> Booking:
        """success 70%, slot_taken 15%, timeout 15% (half of them booked anyway); hold_expired
        after the hold's expiry. A key that already booked returns that booking."""
        with self._gate("book"):
            return self._book(hold_id, idempotency_key)

    def _book(self, hold_id: str, idempotency_key: str) -> Booking:
        tool = ToolName.BOOK
        failure: ToolError | None = None
        crash = False
        if self._take_forced(BookOutcome.CRASH_BEFORE_SEND):
            raise SimulatedCrash(f"process killed before book was sent for {idempotency_key}")
        with self._db.transaction() as conn:
            existing = self._booking_where("b.idempotency_key = ?", idempotency_key)
            if existing is not None:
                return existing
            hold = conn.execute("SELECT * FROM holds WHERE hold_id = ?", (hold_id,)).fetchone()
            if hold is None:
                raise ToolError.of(tool, ToolErrorCode.NOT_FOUND, f"no hold {hold_id}")
            if hold["status"] != "active":
                raise ToolError.of(
                    tool, ToolErrorCode.SLOT_TAKEN, f"hold {hold_id} is no longer usable"
                )
            if self.clock.now() >= _dt(hold["expires_at"]):
                conn.execute("UPDATE holds SET status = 'released' WHERE hold_id = ?", (hold_id,))
                self._settle_claim(conn, hold, "free")
                failure = ToolError.of(
                    tool, ToolErrorCode.HOLD_EXPIRED, f"hold {hold_id} has expired"
                )
            else:
                attempt = self._bump("book", idempotency_key)
                outcome = self._book_outcome(idempotency_key, attempt)
                if outcome in (
                    BookOutcome.SUCCESS,
                    BookOutcome.TIMEOUT_BOOKED,
                    BookOutcome.CRASH_AFTER_COMMIT,
                ) and not self._create_booking(conn, hold, idempotency_key):
                    # The claim moved on: this caller's hold no longer owns the slot.
                    raise ToolError.of(
                        tool, ToolErrorCode.SLOT_TAKEN, "someone else has this slot now"
                    )
                if outcome is BookOutcome.SLOT_TAKEN:
                    conn.execute(
                        "UPDATE holds SET status = 'released' WHERE hold_id = ?", (hold_id,)
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO taken (provider_id, slot_start) VALUES (?, ?)",
                        (hold["provider_id"], hold["slot_start"]),
                    )
                    self._settle_claim(conn, hold, "taken")
                    failure = ToolError.of(
                        tool, ToolErrorCode.SLOT_TAKEN, "someone else booked this slot first"
                    )
                elif outcome in (BookOutcome.TIMEOUT_BOOKED, BookOutcome.TIMEOUT_NOT_BOOKED):
                    failure = ToolError.of(
                        tool, ToolErrorCode.TIMEOUT, "the provider did not answer in time"
                    )
                crash = outcome is BookOutcome.CRASH_AFTER_COMMIT
        if crash:
            raise SimulatedCrash(f"process killed after committing booking for {idempotency_key}")
        if failure is not None:
            if failure.failure.code is ToolErrorCode.TIMEOUT:
                # The caller's client gives up at its own deadline, if that comes first.
                wait = self.scenario.timeout_latency_s
                if self.call_timeout_s is not None:
                    wait = min(wait, self.call_timeout_s)
                self.clock.sleep(wait)
            raise failure
        booking = self._booking_where("b.idempotency_key = ?", idempotency_key)
        assert booking is not None
        return booking

    def cancel(self, booking_id: str, idempotency_key: str) -> CancelResult:
        """The first call for a booking fails with a transient 503; the retry succeeds."""
        tool = ToolName.CANCEL
        failure: ToolError | None = None
        with self._gate("cancel"), self._db.transaction() as conn:
            if self._replayed(conn, "cancel", booking_id, idempotency_key, tool) is not None:
                return CancelResult(booking=self._owned_booking(booking_id, tool))
            booking = self._owned_booking(booking_id, tool)
            if booking.status is BookingState.COMPLETED:
                raise ToolError.of(
                    tool, ToolErrorCode.INVALID_ARGUMENT, "that visit already happened"
                )
            if booking.status is BookingState.CONFIRMED:
                attempt = self._bump("cancel", booking_id)
                if attempt <= self.scenario.cancel_fail_first:
                    failure = ToolError.of(
                        tool, ToolErrorCode.SERVICE_UNAVAILABLE, "503 Service Unavailable"
                    )
                else:
                    conn.execute(
                        "UPDATE bookings SET status = 'cancelled', updated_at = ?"
                        " WHERE booking_id = ?",
                        (_ts(self.clock.now()), booking_id),
                    )
                    self._free_booked_slot(conn, booking_id)
            if failure is None:
                self._record_op(conn, idempotency_key, "cancel", booking_id, {})
        if failure is not None:
            raise failure
        return CancelResult(booking=self._owned_booking(booking_id, tool))

    def reschedule(self, booking_id: str, slot: Slot, idempotency_key: str) -> RescheduleResult:
        """slot_conflict on the scenario's configured slots (and on slots that aren't free)."""
        tool = ToolName.RESCHEDULE
        with self._db.transaction() as conn:
            replay = self._replayed(conn, "reschedule", booking_id, idempotency_key, tool)
            if replay is not None:
                return RescheduleResult(
                    booking=self._owned_booking(booking_id, tool),
                    previous_slot=Slot.model_validate(replay["previous_slot"]),
                )
            booking = self._owned_booking(booking_id, tool)
            if booking.status is not BookingState.CONFIRMED:
                raise ToolError.of(
                    tool,
                    ToolErrorCode.INVALID_ARGUMENT,
                    "only an upcoming, confirmed booking can be rescheduled",
                )
            if not on_grid(slot.start, slot.end):
                raise ToolError.of(
                    tool, ToolErrorCode.INVALID_ARGUMENT, "not one of the provider's slots"
                )
            open_slot = slot.start > self.clock.now() and base_open(
                self.seed, booking.provider_id, slot.start
            )
            if (
                self._conflicts(slot)
                or not open_slot
                or not self._claim(
                    conn, booking.provider_id, slot.start, "booked", booking_id=booking_id
                )
            ):
                raise ToolError.of(
                    tool, ToolErrorCode.SLOT_CONFLICT, "the provider can't take that slot"
                )
            conn.execute(
                "UPDATE slot_claims SET status = 'free', booking_id = NULL, user_id = NULL,"
                " updated_at = ? WHERE provider_id = ? AND slot_start = ? AND booking_id = ?",
                (_ts(self.clock.now()), booking.provider_id, _ts(booking.slot.start), booking_id),
            )
            conn.execute(
                "UPDATE bookings SET slot_start = ?, slot_end = ?, updated_at = ?"
                " WHERE booking_id = ?",
                (_ts(slot.start), _ts(slot.end), _ts(self.clock.now()), booking_id),
            )
            self._record_op(
                conn,
                idempotency_key,
                "reschedule",
                booking_id,
                {"previous_slot": booking.slot.model_dump(mode="json")},
            )
        return RescheduleResult(
            booking=self._owned_booking(booking_id, tool), previous_slot=booking.slot
        )

    # Simulation hooks (not tools)

    def provider_cancelled(
        self, booking_id: str, *, reason: str = "the provider is unavailable", due_in_s: float = 0
    ) -> WorldEvent:
        """The provider calls off a booking; an event is queued for the agent to notice."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)
            ).fetchone()
            if row is None:
                raise KeyError(booking_id)
            if row["status"] == "confirmed":
                conn.execute(
                    "UPDATE bookings SET status = 'cancelled_by_provider', updated_at = ?"
                    " WHERE booking_id = ?",
                    (_ts(self.clock.now()), booking_id),
                )
                # The provider can't do that slot after all: it isn't offered to anyone else.
                conn.execute(
                    "UPDATE slot_claims SET status = 'taken', booking_id = NULL, updated_at = ?"
                    " WHERE booking_id = ?",
                    (_ts(self.clock.now()), booking_id),
                )
        return self.events.push(
            WorldEvent(
                kind="provider_cancelled",
                booking_id=booking_id,
                user_id=row["user_id"],
                due_at=self.clock.now() + timedelta(seconds=due_in_s),
                reason=reason,
            )
        )

    def seed_booking(
        self,
        *,
        user_id: str,
        provider_id: str,
        slot: Slot,
        status: BookingState = BookingState.CONFIRMED,
        price_inr: int | None = None,
    ) -> Booking:
        """Put a booking straight into the world (for past bookings in demo data and tests)."""
        provider = self._provider_row(provider_id, ToolName.LIST_BOOKINGS)
        stored_status = "confirmed" if status is BookingState.COMPLETED else status.value
        with self._db.transaction() as conn:
            booking_id = f"b_{self._next('booking')}"
            now = _ts(self.clock.now())
            conn.execute(
                "INSERT INTO bookings (booking_id, user_id, provider_id, service, slot_start,"
                " slot_end, price_inr, status, technician, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    booking_id,
                    user_id,
                    provider_id,
                    provider["service"],
                    _ts(slot.start),
                    _ts(slot.end),
                    price_inr if price_inr is not None else provider["price_inr"],
                    stored_status,
                    json.loads(provider["technicians"])[0],
                    now,
                    now,
                ),
            )
            if stored_status == "confirmed":
                conn.execute(
                    "INSERT OR REPLACE INTO slot_claims (provider_id, slot_start, status,"
                    " booking_id, user_id, updated_at) VALUES (?, ?, 'booked', ?, ?, ?)",
                    (provider_id, _ts(slot.start), booking_id, user_id, now),
                )
        booking = self._booking_where("b.booking_id = ?", booking_id)
        assert booking is not None
        return booking

    def provider_cancels_all(
        self, provider_id: str, *, reason: str = "the provider is unavailable"
    ) -> list[WorldEvent]:
        """The provider calls off every upcoming booking they have, with every user: one event per
        booking, each for its own user."""
        rows = self._db.fetch_all(
            "SELECT booking_id FROM bookings WHERE provider_id = ? AND status = 'confirmed'"
            " AND slot_start > ? ORDER BY slot_start, booking_id",
            (provider_id, _ts(self.clock.now())),
        )
        return [self.provider_cancelled(row["booking_id"], reason=reason) for row in rows]

    # Internals

    def _free_booked_slot(self, conn, booking_id: str) -> None:
        conn.execute(
            "UPDATE slot_claims SET status = 'free', booking_id = NULL, user_id = NULL,"
            " updated_at = ? WHERE booking_id = ?",
            (_ts(self.clock.now()), booking_id),
        )

    def _initialise(self) -> None:
        """Generate the catalogue and load forced outcomes, once per world DB."""
        with self._db.transaction() as conn:
            stored = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if stored:
                expected = {"scenario": self.scenario.name, "seed": str(self.seed)}
                if {key: stored.get(key) for key in expected} != expected:
                    raise ValueError(
                        f"the world DB {self._db.path} was made for scenario {stored.get('scenario')!r}"
                        f" with seed {stored.get('seed')}; delete it to start a"
                        f" {self.scenario.name!r} world"
                    )
                return
            providers, reviews = generate_catalog(
                self.seed,
                today=self.clock.now().astimezone(IST).date(),
                injection_providers=frozenset(self.scenario.injection_providers),
            )
            overrides = self.scenario.provider_overrides
            providers = [
                dataclasses.replace(
                    p,
                    **{
                        **overrides[p.provider_id],
                        "attributes": {
                            **p.attributes,
                            **overrides[p.provider_id].get("attributes", {}),
                        },
                    },
                )
                if p.provider_id in overrides
                else p
                for p in providers
            ]
            conn.executemany(
                "INSERT INTO providers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        p.provider_id,
                        p.service,
                        p.area,
                        p.name,
                        p.price_inr,
                        p.rating,
                        p.review_count,
                        p.about,
                        json.dumps(list(p.technicians)),
                        json.dumps(p.attributes),
                    )
                    for p in providers
                ],
            )
            conn.executemany(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.review_id,
                        r.provider_id,
                        r.author,
                        r.rating,
                        r.text,
                        r.posted_on.isoformat(),
                    )
                    for r in reviews
                ],
            )
            conn.executemany(
                "INSERT INTO forced (tool, outcome) VALUES ('book', ?)",
                [(outcome.value,) for outcome in self.scenario.book_forced],
            )
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("scenario", self.scenario.name),
                    ("seed", str(self.seed)),
                    ("next_hold", "101"),
                    ("next_booking", "101"),
                ],
            )

    def _service(self, service: str, tool: ToolName) -> str:
        key = service.strip().lower().replace(" ", "_").replace("-", "_")
        key = _SERVICE_ALIASES.get(key, key)
        if key not in SERVICES:
            raise ToolError.of(
                tool,
                ToolErrorCode.INVALID_ARGUMENT,
                f"unknown service {service!r}; known services: {', '.join(SERVICES)}",
            )
        return key

    def _days(self, date_range: DateRange, tool: ToolName) -> list[date]:
        span = (date_range.end - date_range.start).days + 1
        if span > MAX_SEARCH_DAYS:
            raise ToolError.of(
                tool,
                ToolErrorCode.INVALID_ARGUMENT,
                f"date_range covers {span} days; the most is {MAX_SEARCH_DAYS}",
            )
        return [date_range.start + timedelta(days=offset) for offset in range(span)]

    def _provider_row(self, provider_id: str, tool: ToolName):
        row = self._db.fetch_one("SELECT * FROM providers WHERE provider_id = ?", (provider_id,))
        if row is None:
            raise ToolError.of(tool, ToolErrorCode.NOT_FOUND, f"no provider {provider_id}")
        return row

    def _blocked(self, provider_ids: list[str]) -> set[tuple[str, str]]:
        """(provider, slot start) pairs that are booked, taken or under an active hold."""
        if not provider_ids:
            return set()
        marks = ", ".join("?" for _ in provider_ids)
        now = _ts(self.clock.now())
        rows = self._db.fetch_all(
            f"SELECT provider_id, slot_start FROM slot_claims WHERE provider_id IN ({marks})"
            " AND (status IN ('booked', 'taken') OR (status = 'held' AND held_until > ?))",
            (*provider_ids, now),
        )
        return {(row["provider_id"], row["slot_start"]) for row in rows}

    def _is_open(self, provider_id: str, start: datetime) -> bool:
        if start <= self.clock.now() or not base_open(self.seed, provider_id, start):
            return False
        return (provider_id, _ts(start)) not in self._blocked([provider_id])

    def _summary(self, row, days: list[date], blocked: set[tuple[str, str]]) -> ProviderSummary:
        now = self.clock.now()
        pid = row["provider_id"]
        slots = [
            _slot(start)
            for day in days
            for start in grid_starts(day)
            if start > now and base_open(self.seed, pid, start) and (pid, _ts(start)) not in blocked
        ]
        return ProviderSummary(
            provider_id=pid,
            name=row["name"],
            service=row["service"],
            area=row["area"],
            price_inr=row["price_inr"],
            rating=row["rating"],
            review_count=row["review_count"],
            slots=slots[:MAX_SLOTS_PER_PROVIDER],
            attributes=json.loads(row["attributes"]),
        )

    def _quote_delta(self, provider_id: str, price: int, slot: Slot) -> int:
        fixed = self.scenario.quote_delta_inr.get(
            provider_id, self.scenario.quote_delta_inr.get("*")
        )
        if fixed is not None:
            return fixed
        rng = random.Random(f"{self.seed}|quote|{provider_id}|{_ts(slot.start)}")
        if rng.random() >= self.scenario.quote_drift_probability:
            return 0
        pct = rng.uniform(5, max(5, self.scenario.quote_drift_max_pct)) / 100
        return round(price * pct / 10) * 10

    def _take_forced(self, outcome: BookOutcome) -> bool:
        """Use up the next forced book outcome if it is `outcome`."""
        row = self._db.fetch_one(
            "SELECT seq, outcome FROM forced WHERE tool = 'book' AND used_at IS NULL"
            " ORDER BY seq LIMIT 1"
        )
        if row is None or row["outcome"] != outcome:
            return False
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE forced SET used_at = ? WHERE seq = ?", (_ts(self.clock.now()), row["seq"])
            )
        return True

    def _book_outcome(self, idempotency_key: str, attempt: int) -> BookOutcome:
        row = self._db.fetch_one(
            "SELECT seq, outcome FROM forced WHERE tool = 'book' AND used_at IS NULL"
            " ORDER BY seq LIMIT 1"
        )
        if row is not None:
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE forced SET used_at = ? WHERE seq = ?",
                    (_ts(self.clock.now()), row["seq"]),
                )
            return BookOutcome(row["outcome"])
        rng = random.Random(f"{self.seed}|book|{idempotency_key}|{attempt}")
        outcomes = list(self.scenario.book_weights)
        return rng.choices(outcomes, weights=[self.scenario.book_weights[o] for o in outcomes])[0]

    def _create_booking(self, conn, hold, idempotency_key: str) -> bool:
        """Book the held slot: turn this caller's claim into a booking (row count checked), then
        write the booking. False, and nothing written, if the claim is no longer theirs."""
        provider = self._provider_row(hold["provider_id"], ToolName.BOOK)
        booking_id = f"b_{self._next('booking')}"
        technicians = json.loads(provider["technicians"])
        technician = random.Random(f"{self.seed}|tech|{booking_id}").choice(technicians)
        now = _ts(self.clock.now())
        if not self._settle_claim(conn, hold, "booked", booking_id=booking_id):
            return False
        conn.execute(
            "INSERT INTO bookings (booking_id, user_id, provider_id, service, slot_start,"
            " slot_end, price_inr, status, technician, hold_id, idempotency_key, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'confirmed', ?, ?, ?, ?, ?)",
            (
                booking_id,
                self.user_id,
                hold["provider_id"],
                provider["service"],
                hold["slot_start"],
                hold["slot_end"],
                hold["quoted_price_inr"],
                technician,
                hold["hold_id"],
                idempotency_key,
                now,
                now,
            ),
        )
        conn.execute("UPDATE holds SET status = 'used' WHERE hold_id = ?", (hold["hold_id"],))
        return True

    def _claim(
        self,
        conn,
        provider_id: str,
        start: datetime,
        status: str,
        *,
        hold_id: str | None = None,
        held_until: datetime | None = None,
        booking_id: str | None = None,
    ) -> bool:
        """Take a slot that is free (or whose hold has lapsed) for this caller: one conditional
        UPDATE, whose row count says whether they got it. Two callers can't both change the row."""
        now = _ts(self.clock.now())
        at = _ts(start)
        conn.execute(
            "INSERT OR IGNORE INTO slot_claims (provider_id, slot_start, status, updated_at)"
            " VALUES (?, ?, 'free', ?)",
            (provider_id, at, now),
        )
        cur = conn.execute(
            "UPDATE slot_claims SET status = ?, hold_id = ?, held_until = ?, booking_id = ?,"
            " user_id = ?, updated_at = ? WHERE provider_id = ? AND slot_start = ?"
            " AND (status = 'free' OR (status = 'held' AND held_until <= ?))",
            (
                status,
                hold_id,
                _ts(held_until) if held_until else None,
                booking_id,
                self.user_id,
                now,
                provider_id,
                at,
                now,
            ),
        )
        return cur.rowcount == 1

    def _settle_claim(self, conn, hold, status: str, *, booking_id: str | None = None) -> bool:
        """Move this caller's hold on its slot to booked, taken or free; False if the slot isn't
        held under that hold by this caller any more."""
        free = status == "free"
        cur = conn.execute(
            "UPDATE slot_claims SET status = ?, booking_id = ?, held_until = NULL,"
            " hold_id = CASE WHEN ? THEN NULL ELSE hold_id END,"
            " user_id = CASE WHEN ? THEN NULL ELSE user_id END, updated_at = ?"
            " WHERE provider_id = ? AND slot_start = ? AND status = 'held' AND hold_id = ?"
            " AND user_id = ?",
            (
                status,
                booking_id,
                free,
                free,
                _ts(self.clock.now()),
                hold["provider_id"],
                hold["slot_start"],
                hold["hold_id"],
                self.user_id,
            ),
        )
        return cur.rowcount == 1

    def _gate(self, point: str):
        gate = self.gates.get(point)
        return gate.turn(self.user_id) if gate else nullcontext()

    def _replayed(
        self, conn, op: str, booking_id: str, idempotency_key: str, tool: ToolName
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM ops WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if row is None:
            return None
        if (row["op"], row["booking_id"]) != (op, booking_id):
            raise ToolError.of(
                tool, ToolErrorCode.INVALID_ARGUMENT, "idempotency key reused for another request"
            )
        return json.loads(row["result"])

    def _record_op(
        self, conn, idempotency_key: str, op: str, booking_id: str, result: dict
    ) -> None:
        conn.execute(
            "INSERT INTO ops (idempotency_key, op, booking_id, result, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (idempotency_key, op, booking_id, json.dumps(result), _ts(self.clock.now())),
        )

    def _conflicts(self, slot: Slot) -> bool:
        local = slot.start.astimezone(IST)
        for spec in self.scenario.reschedule_conflicts:
            if len(spec) == 5 and local.strftime("%H:%M") == spec:
                return True
            if len(spec) > 5:
                moment = datetime.fromisoformat(spec)
                moment = moment if moment.tzinfo else moment.replace(tzinfo=IST)
                if moment == slot.start:
                    return True
        return False

    def _next(self, kind: str) -> int:
        with self._db.transaction() as conn:
            row = conn.execute(
                "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = ?"
                " RETURNING CAST(value AS INTEGER) - 1 AS n",
                (f"next_{kind}",),
            ).fetchone()
        return row["n"]

    def _bump(self, scope: str, ref: str) -> int:
        with self._db.transaction() as conn:
            row = conn.execute(
                "INSERT INTO attempts (scope, ref, count) VALUES (?, ?, 1)"
                " ON CONFLICT (scope, ref) DO UPDATE SET count = count + 1 RETURNING count",
                (scope, ref),
            ).fetchone()
        return row["count"]

    def _owned_booking(self, booking_id: str, tool: ToolName) -> Booking:
        booking = self._booking_where("b.booking_id = ?", booking_id)
        if booking is None or booking.user_id != self.user_id:
            raise ToolError.of(tool, ToolErrorCode.NOT_FOUND, f"no booking {booking_id}")
        return booking

    def _booking_where(self, condition: str, value: str) -> Booking | None:
        row = self._db.fetch_one(
            "SELECT b.*, p.name AS provider_name FROM bookings b"
            f" JOIN providers p USING (provider_id) WHERE {condition}",
            (value,),
        )
        return self._booking(row) if row else None

    def _booking(self, row) -> Booking:
        status = BookingState(row["status"])
        if status is BookingState.CONFIRMED and _dt(row["slot_end"]) <= self.clock.now():
            status = BookingState.COMPLETED
        return Booking(
            booking_id=row["booking_id"],
            user_id=row["user_id"],
            provider_id=row["provider_id"],
            provider_name=row["provider_name"],
            service=row["service"],
            slot=_slot(_dt(row["slot_start"])),
            price_inr=row["price_inr"],
            status=status,
            technician=row["technician"],
        )
