import sqlite3
import threading
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from functools import partial

import pytest

from errand.contracts import IST, BookingState, DateRange, Hold, Slot, ToolError, ToolErrorCode
from errand.gateway.untrusted import injection_hint
from errand.mocks import (
    AREAS,
    SERVICES,
    BookOutcome,
    FakeClock,
    MockWorld,
    Scenario,
    SimulatedCrash,
    Turnstile,
)

START = datetime(2026, 11, 3, 9, 0, tzinfo=IST)
WEEK = DateRange(start=date(2026, 11, 3), end=date(2026, 11, 9))


@pytest.fixture
def clock():
    return FakeClock(START)


@pytest.fixture
def make_world(tmp_path, clock):
    def make(name="w", **scenario):
        return MockWorld(
            Scenario(**scenario), db_path=tmp_path / f"{name}.db", clock=clock, seed=42
        ).for_user("u1")

    return make


def first_option(world, service="plumber", area="Koramangala"):
    provider = world.search_providers(service, area, WEEK).providers[0]
    return provider, provider.slots[0]


def code_of(excinfo) -> ToolErrorCode:
    return excinfo.value.failure.code


# search_providers and get_provider


def test_search_returns_three_to_eight_providers_with_future_grid_slots(make_world):
    world = make_world()
    for service in SERVICES:
        for area in AREAS:
            providers = world.search_providers(service, area, WEEK).providers
            assert 3 <= len(providers) <= 8
            for provider in providers:
                assert provider.service == service and provider.area == area
                for slot in provider.slots:
                    assert slot.start > START
                    assert WEEK.start <= slot.start.date() <= WEEK.end
                    assert slot.end - slot.start == timedelta(hours=2)


def test_the_catalogue_is_a_function_of_the_seed(tmp_path, clock):
    def snapshot(name, seed):
        world = MockWorld(Scenario(), db_path=tmp_path / f"{name}.db", clock=clock, seed=seed)
        return world.search_providers("electrician", "HSR Layout", WEEK).model_dump()

    assert snapshot("a", 7) == snapshot("b", 7)
    assert snapshot("a2", 7) != snapshot("c", 8)


@pytest.mark.parametrize(
    ("service", "area", "date_range"),
    [
        ("astrologer", "Koramangala", WEEK),
        ("plumber", "Atlantis", WEEK),
        ("plumber", "Koramangala", DateRange(start=date(2026, 11, 3), end=date(2026, 11, 30))),
    ],
)
def test_search_rejects_bad_arguments(make_world, service, area, date_range):
    with pytest.raises(ToolError) as excinfo:
        make_world().search_providers(service, area, date_range)
    assert code_of(excinfo) is ToolErrorCode.INVALID_ARGUMENT


def test_every_service_and_area_has_a_review_carrying_a_prompt_injection(make_world):
    world = make_world()
    for service in SERVICES:
        for area in AREAS:
            reviews = [
                review.text
                for provider in world.search_providers(service, area, WEEK).providers
                for review in world.get_provider(provider.provider_id).reviews
            ]
            assert any(injection_hint(text) for text in reviews), (service, area)


# hold_slot


def test_a_hold_lasts_ten_minutes_and_takes_the_slot_off_the_market(make_world):
    world = make_world()
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)

    assert hold.expires_at - START == timedelta(minutes=10)
    again = world.search_providers("plumber", "Koramangala", WEEK).providers[0]
    assert slot not in again.slots
    with pytest.raises(ToolError) as excinfo:
        world.hold_slot(provider.provider_id, slot)
    assert code_of(excinfo) is ToolErrorCode.SLOT_TAKEN


def test_the_quote_can_differ_from_the_search_price(make_world):
    world = make_world(quote_delta_inr={"*": 150})
    provider, slot = first_option(world)
    assert world.hold_slot(provider.provider_id, slot).quoted_price_inr == provider.price_inr + 150


def test_holds_only_cover_real_slots(make_world):
    world = make_world()
    provider, slot = first_option(world)
    odd = Slot(start=slot.start + timedelta(minutes=30), end=slot.end + timedelta(minutes=30))
    with pytest.raises(ToolError) as excinfo:
        world.hold_slot(provider.provider_id, odd)
    assert code_of(excinfo) is ToolErrorCode.INVALID_ARGUMENT


# book and get_booking_status


def test_a_successful_booking_is_answered_again_for_the_same_key(make_world):
    world = make_world(book_forced=[BookOutcome.SUCCESS])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    booking = world.book(hold.hold_id, "idem_a")

    assert (booking.provider_id, booking.slot, booking.status) == (
        provider.provider_id,
        slot,
        BookingState.CONFIRMED,
    )
    assert booking.price_inr == hold.quoted_price_inr
    assert world.book(hold.hold_id, "idem_a") == booking
    assert len(world.list_bookings("u1").bookings) == 1


def test_a_timeout_can_hide_a_booking_that_went_through(make_world, clock):
    world = make_world(book_forced=[BookOutcome.TIMEOUT_BOOKED])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(ToolError) as excinfo:
        world.book(hold.hold_id, "idem_t")

    assert code_of(excinfo) is ToolErrorCode.TIMEOUT
    assert excinfo.value.failure.outcome_unknown
    assert clock.now() == START + timedelta(seconds=2)  # the timeout took a while
    by_key = world.get_booking_status(idempotency_key="idem_t")
    by_hold = world.get_booking_status(hold_id=hold.hold_id)
    assert by_key.booked and by_hold == by_key
    assert world.book(hold.hold_id, "idem_t") == by_key.booking


def test_a_timeout_that_did_not_book_leaves_the_hold_usable(make_world):
    world = make_world(book_forced=[BookOutcome.TIMEOUT_NOT_BOOKED, BookOutcome.SUCCESS])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(ToolError):
        world.book(hold.hold_id, "idem_n")
    assert not world.get_booking_status(idempotency_key="idem_n").booked
    assert world.book(hold.hold_id, "idem_n").status is BookingState.CONFIRMED


def test_slot_taken_removes_the_slot_for_good(make_world):
    world = make_world(book_forced=[BookOutcome.SLOT_TAKEN])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(ToolError) as excinfo:
        world.book(hold.hold_id, "idem_s")

    assert code_of(excinfo) is ToolErrorCode.SLOT_TAKEN
    assert not world.get_booking_status(idempotency_key="idem_s").booked
    with pytest.raises(ToolError):
        world.hold_slot(provider.provider_id, slot)


def test_booking_after_the_hold_expired_is_refused(make_world, clock):
    world = make_world()
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    clock.advance(10 * 60)
    with pytest.raises(ToolError) as excinfo:
        world.book(hold.hold_id, "idem_e")
    assert code_of(excinfo) is ToolErrorCode.HOLD_EXPIRED


def test_random_outcomes_follow_the_spec_split(make_world):
    world = make_world()
    outcomes = Counter(world._book_outcome(f"key_{n}", 1) for n in range(2000))
    share = {outcome: count / 2000 for outcome, count in outcomes.items()}
    assert share[BookOutcome.SUCCESS] == pytest.approx(0.70, abs=0.04)
    assert share[BookOutcome.SLOT_TAKEN] == pytest.approx(0.15, abs=0.03)
    timeouts = share[BookOutcome.TIMEOUT_BOOKED] + share[BookOutcome.TIMEOUT_NOT_BOOKED]
    assert timeouts == pytest.approx(0.15, abs=0.03)
    assert share[BookOutcome.TIMEOUT_BOOKED] == pytest.approx(timeouts / 2, abs=0.03)


def test_get_booking_status_takes_exactly_one_id(make_world):
    world = make_world()
    for kwargs in ({}, {"hold_id": "h", "idempotency_key": "k"}):
        with pytest.raises(ToolError) as excinfo:
            world.get_booking_status(**kwargs)
        assert code_of(excinfo) is ToolErrorCode.INVALID_ARGUMENT


def test_a_crash_after_commit_leaves_the_booking_behind(make_world):
    world = make_world(book_forced=[BookOutcome.CRASH_AFTER_COMMIT])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(SimulatedCrash):
        world.book(hold.hold_id, "idem_c")
    assert world.get_booking_status(idempotency_key="idem_c").booked


# cancel, reschedule, list_bookings


@pytest.fixture
def booked(make_world):
    world = make_world(book_forced=[BookOutcome.SUCCESS], reschedule_conflicts=["10:00"])
    provider, slot = first_option(world)
    booking = world.book(world.hold_slot(provider.provider_id, slot).hold_id, "idem_b")
    return world, booking


def test_cancel_fails_once_with_a_503_then_succeeds_with_the_same_key(booked):
    world, booking = booked
    with pytest.raises(ToolError) as excinfo:
        world.cancel(booking.booking_id, "cancel_1")
    assert code_of(excinfo) is ToolErrorCode.SERVICE_UNAVAILABLE
    assert excinfo.value.failure.retryable and excinfo.value.failure.http_status == 503

    result = world.cancel(booking.booking_id, "cancel_1")
    assert result.booking.status is BookingState.CANCELLED
    assert world.cancel(booking.booking_id, "cancel_1") == result


def test_reschedule_conflicts_on_configured_slots_and_moves_otherwise(booked):
    world, booking = booked
    free = [
        slot
        for provider in world.search_providers("plumber", "Koramangala", WEEK).providers
        if provider.provider_id == booking.provider_id
        for slot in provider.slots
    ]
    conflicting = next(s for s in free if s.start.astimezone(IST).strftime("%H:%M") == "10:00")
    fine = next(s for s in free if s.start.astimezone(IST).strftime("%H:%M") != "10:00")

    with pytest.raises(ToolError) as excinfo:
        world.reschedule(booking.booking_id, conflicting, "resched_1")
    assert code_of(excinfo) is ToolErrorCode.SLOT_CONFLICT

    moved = world.reschedule(booking.booking_id, fine, "resched_2")
    assert (moved.booking.slot, moved.previous_slot) == (fine, booking.slot)
    assert world.reschedule(booking.booking_id, fine, "resched_2") == moved


def test_one_user_cannot_touch_another_users_booking(booked):
    world, booking = booked
    intruder = world.for_user("u2")
    with pytest.raises(ToolError) as excinfo:
        intruder.cancel(booking.booking_id, "cancel_x")
    assert code_of(excinfo) is ToolErrorCode.NOT_FOUND
    assert intruder.list_bookings("u2").bookings == []


def test_past_bookings_read_as_completed(booked, clock):
    world, _ = booked
    clock.advance(30 * 24 * 3600)
    assert world.list_bookings("u1").bookings[0].status is BookingState.COMPLETED


# Persistence and events


def test_world_state_survives_a_new_world_instance(tmp_path, clock):
    first = MockWorld(
        Scenario(book_forced=[BookOutcome.SUCCESS]), db_path=tmp_path / "w.db", clock=clock
    ).for_user("u1")
    provider, slot = first_option(first)
    booking = first.book(first.hold_slot(provider.provider_id, slot).hold_id, "idem_p")
    first.close()

    second = MockWorld(
        Scenario(book_forced=[BookOutcome.SUCCESS]), db_path=tmp_path / "w.db", clock=clock
    )
    assert second.get_booking_status(idempotency_key="idem_p").booking == booking
    with pytest.raises(ValueError, match="delete it"):
        MockWorld(Scenario(name="other"), db_path=tmp_path / "w.db", clock=clock)


def test_a_provider_cancellation_queues_one_event(booked, clock):
    world, booking = booked
    event = world.provider_cancelled(booking.booking_id, due_in_s=60)
    assert world.list_bookings("u1").bookings[0].status is BookingState.CANCELLED_BY_PROVIDER

    assert world.events.pop_due(clock.now()) == []
    clock.advance(60)
    assert world.events.pop_due(clock.now()) == [event]
    assert world.events.pop_due(clock.now()) == []


# Fault controls the eval scenarios use


def test_a_slot_held_again_quotes_the_moved_price(make_world, clock):
    world = make_world(rehold_delta_inr=50, hold_ttl_s=60, quote_drift_probability=0)
    provider, slot = first_option(world)
    first = world.hold_slot(provider.provider_id, slot)
    clock.advance(61)  # the first hold lapses, so the slot is free to hold again
    assert world.hold_slot(provider.provider_id, slot).quoted_price_inr == (
        first.quoted_price_inr + 50
    )


def test_the_status_service_can_be_down_for_its_first_calls(make_world):
    world = make_world(status_unavailable_calls=2)
    for _ in range(2):
        with pytest.raises(ToolError) as excinfo:
            world.get_booking_status(idempotency_key="idem_x")
        assert code_of(excinfo) is ToolErrorCode.SERVICE_UNAVAILABLE
    assert world.get_booking_status(idempotency_key="idem_x").booked is False


def test_faults_break_chosen_calls_and_an_after_fault_still_takes_effect(make_world):
    world = make_world(
        book_forced=[BookOutcome.SUCCESS],
        faults=[
            {"tool": "get_provider", "kind": "wrong_type"},
            {"tool": "book", "kind": "malformed_json", "after": True},
        ],
    )
    provider, slot = first_option(world)
    assert world.get_provider(provider.provider_id) == {"status": "ok", "data": None}
    assert world.get_provider(provider.provider_id).provider_id == provider.provider_id
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(ValueError, match="Expecting value"):
        world.book(hold.hold_id, "idem_f")
    assert world.get_booking_status(
        idempotency_key="idem_f"
    ).booked  # the reply was lost, not the booking


def test_provider_overrides_replace_the_listing_text(make_world):
    world = make_world(
        provider_overrides={
            "prov_plumber_koramangala_1": {"name": "Pipes & Co", "attributes": {"note": "x"}}
        }
    )
    details = world.get_provider("prov_plumber_koramangala_1")
    assert details.name == "Pipes & Co" and details.attributes["note"] == "x"
    with pytest.raises(ValueError, match="name, about and attributes"):
        Scenario(provider_overrides={"prov_plumber_koramangala_1": {"price_inr": 1}})


def test_a_crash_before_send_reaches_no_provider_state(make_world):
    world = make_world(book_forced=[BookOutcome.CRASH_BEFORE_SEND, BookOutcome.SUCCESS])
    provider, slot = first_option(world)
    hold = world.hold_slot(provider.provider_id, slot)
    with pytest.raises(SimulatedCrash):
        world.book(hold.hold_id, "idem_c")
    assert not world.get_booking_status(idempotency_key="idem_c").booked
    assert world.book(hold.hold_id, "idem_c").status is BookingState.CONFIRMED  # a retry books


def test_one_threads_simulated_wait_is_not_another_threads_elapsed_time(clock):
    # Regression for docs/bugs.md #22: one run's 30 s provider wait used up a concurrent run's
    # whole time budget, because every run read the same fake clock.
    import threading

    started = clock.monotonic()
    other = threading.Thread(target=clock.sleep, args=(30,))
    other.start()
    other.join()
    assert clock.now() == START + timedelta(seconds=30)  # the world's time moved for everyone
    assert clock.monotonic() - started == 0  # but this thread didn't wait
    clock.sleep(2)
    assert clock.monotonic() - started == 2


# Concurrent callers


def race(*calls):
    """Run the calls on threads released together; each one's result, or the ToolError it raised."""
    barrier = threading.Barrier(len(calls))
    results: list = [None] * len(calls)

    def run(i, call):
        barrier.wait()
        try:
            results[i] = call()
        except ToolError as exc:
            results[i] = exc

    threads = [threading.Thread(target=run, args=(i, call)) for i, call in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


@pytest.mark.parametrize("connections", [1, 2], ids=["one process", "two processes"])
def test_two_users_holding_one_slot_at_once_leave_exactly_one_hold(tmp_path, clock, connections):
    worlds = [
        MockWorld(Scenario(), db_path=tmp_path / "w.db", clock=clock, seed=42)
        for _ in range(connections)
    ]
    asha, ravi = worlds[0].for_user("u_asha"), worlds[-1].for_user("u_ravi")
    provider = asha.search_providers("plumber", "Koramangala", WEEK).providers[0]
    for slot in provider.slots[:10]:  # ten races, each for a slot both want
        results = race(
            partial(asha.hold_slot, provider.provider_id, slot),
            partial(ravi.hold_slot, provider.provider_id, slot),
        )
        (won,) = [r for r in results if isinstance(r, Hold)]
        (lost,) = [r for r in results if isinstance(r, ToolError)]
        assert lost.failure.code is ToolErrorCode.SLOT_TAKEN
        claim = worlds[0]._db.fetch_one(
            "SELECT status, hold_id FROM slot_claims WHERE provider_id = ? AND slot_start = ?",
            (provider.provider_id, slot.start.astimezone(UTC).isoformat()),
        )
        assert (claim["status"], claim["hold_id"]) == ("held", won.hold_id)
    for world in worlds:
        world.close()


def test_a_lapsed_hold_cannot_book_or_free_a_slot_someone_else_now_holds(tmp_path, clock):
    world = MockWorld(
        Scenario(book_forced=[BookOutcome.SUCCESS]), db_path=tmp_path / "w.db", clock=clock
    )
    asha, ravi = world.for_user("u_asha"), world.for_user("u_ravi")
    provider, slot = first_option(asha)
    lapsed = asha.hold_slot(provider.provider_id, slot)
    clock.advance(11 * 60)
    fresh = ravi.hold_slot(provider.provider_id, slot)  # the lapsed hold no longer keeps it

    with pytest.raises(ToolError) as excinfo:
        asha.book(lapsed.hold_id, "idem_asha")
    assert code_of(excinfo) is ToolErrorCode.HOLD_EXPIRED
    assert slot not in asha.search_providers("plumber", "Koramangala", WEEK).providers[0].slots
    assert ravi.book(fresh.hold_id, "idem_ravi").user_id == "u_ravi"
    confirmed = [r["user_id"] for r in world.booking_rows() if r["status"] == "confirmed"]
    assert confirmed == ["u_ravi"]


def test_the_world_refuses_a_second_confirmed_booking_for_one_slot(make_world):
    world = make_world(book_forced=[BookOutcome.SUCCESS])
    provider, slot = first_option(world)
    booking = world.book(world.hold_slot(provider.provider_id, slot).hold_id, "idem_1")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"), world._db.transaction() as conn:
        conn.execute(
            "INSERT INTO bookings (booking_id, user_id, provider_id, service, slot_start,"
            " slot_end, price_inr, status, technician, created_at, updated_at)"
            " SELECT 'b_dup', 'u2', provider_id, service, slot_start, slot_end, price_inr,"
            " 'confirmed', technician, created_at, updated_at FROM bookings WHERE booking_id = ?",
            (booking.booking_id,),
        )


def test_a_turnstile_lets_parties_through_in_its_order_whatever_order_they_arrive_in():
    gate = Turnstile(["b", "a"])
    through: list[str] = []

    def go(party):
        with gate.turn(party):
            through.append(party)

    first = threading.Thread(target=go, args=("a",))
    first.start()  # "a" arrives first and waits: "b" goes first
    first.join(timeout=0.2)
    assert first.is_alive() and through == []
    second = threading.Thread(target=go, args=("b",))
    second.start()
    for thread in (first, second):
        thread.join()
    assert through == gate.passed == ["b", "a"]
    with gate.turn("a"), gate.turn("c"):  # through already, or never in the order: no wait
        pass


def test_a_turnstile_party_that_never_arrives_times_out_instead_of_hanging():
    gate = Turnstile(["a", "b"], timeout_s=0.1)
    with pytest.raises(TimeoutError, match=r"a waited for \['b'\]"), gate.turn("a"):
        pass
