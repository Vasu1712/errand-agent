"""Provider-cancel poller: notices a cancelled booking without a user prompt.

For each due provider_cancelled event it decides whether to act (only for an upcoming booking the
provider, not the user, cancelled, and only once), then opens a replacement task with the original
details plus active preferences. That task searches and ranks but holds nothing; rebooking needs
the user's pick and a fresh approval like any other booking, and only then is a slot held.
"""

from datetime import date, timedelta

from errand.contracts import IST, Booking, BookingState
from errand.graph import render
from errand.memory import MemoryService
from errand.mocks import WorldEvent
from errand.store import FactStatus, OutboxMessage
from errand.trace import EventKind, new_run_id

URGENT = timedelta(hours=24)


def poll_once(agent) -> int:
    """Handle the events that are due and return how many there were."""
    events = agent.world.events.pop_due(agent.clock.now())
    for event in events:
        handle_provider_cancelled(agent, event)
    return len(events)


def handle_provider_cancelled(agent, event: WorldEvent) -> OutboxMessage | None:
    run_id = new_run_id()
    with agent.tracer(
        run_id, user_id=event.user_id, thread_id=None, kind="provider_event"
    ) as tracer:
        tracer.emit(EventKind.RUN_START, kind="provider_event", user_id=event.user_id, event=event)
        budget = agent.budget(timeout_s=agent.settings.background_timeout_s)
        gateway = agent.gateway(tracer, event.user_id, budget)
        booking = next(
            (b for b in gateway.list_bookings().bookings if b.booking_id == event.booking_id), None
        )
        now = agent.clock.now()
        dedupe_key = f"provider_cancelled:{event.booking_id}"
        reason = None
        if booking is None:
            reason = "unknown booking"
        elif booking.status is not BookingState.CANCELLED_BY_PROVIDER:
            reason = f"booking is {booking.status}, not cancelled by the provider"
        elif booking.slot.start <= now:
            reason = "the visit time has already passed"
        elif agent.store.outbox.exists(event.user_id, dedupe_key):
            reason = "already handled"
        urgent = booking is not None and booking.slot.start - now <= URGENT
        tracer.emit(
            EventKind.STATE,
            node="provider_cancel_poller",
            decision="ignore" if reason else "notify",
            reason=reason,
            urgent=urgent,
        )
        if reason:
            tracer.emit(
                EventKind.RUN_END, outcome="ignored", reason=reason, budget=budget.snapshot()
            )
            return None
        slots, sources = _replacement_slots(
            agent, gateway, booking, MemoryService(agent.store, tracer)
        )
        tracer.emit(
            EventKind.RUN_END,
            outcome="replacement_opened",
            slots=slots,
            budget=budget.snapshot(),
        )

    heads_up = "Heads-up, this one is soon: " if urgent else ""
    intro = (
        f"{heads_up}{booking.provider_name} cancelled your {render.service_name(booking.service)}"
        f" visit on {render.when(booking.slot)} ({booking.booking_id}). You haven't been charged."
        " I looked for replacements within your preferences; nothing is held or booked yet."
    )
    return agent.open_replacement(
        user_id=event.user_id,
        intro=intro,
        slots=slots,
        slot_sources=sources,
        excluded_providers=[booking.provider_id],
        origin={"event": "provider_cancelled", "booking_id": booking.booking_id, "urgent": urgent},
        dedupe_key=dedupe_key,
    )


def _replacement_slots(
    agent, gateway, booking: Booking, memory: MemoryService
) -> tuple[dict, dict]:
    """The cancelled booking's details (from the task that made it, when the agent did), plus
    active preferences. Inferred, unconfirmed preferences are not used. The budget is the task's,
    else the user's own budget for the service, else what the cancelled visit cost plus a fifth
    (docs/bugs.md #23)."""
    made = agent.store.ledger.for_booking(booking.booking_id, user_id=booking.user_id)
    task = (
        agent.store.tasks.get(made.task_id, user_id=booking.user_id)
        if made and made.task_id
        else None
    )
    original = dict(task.details) if task else {}
    local = booking.slot.start.astimezone(IST)
    today = agent.clock.now().astimezone(IST).date()
    first = max(local.date(), today)
    slots = {
        "service": booking.service,
        "area": original.get("area") or gateway.get_provider(booking.provider_id).area,
        "date_from": first.isoformat(),
        "date_to": (first + timedelta(days=2)).isoformat(),
        "time_window": original.get("time_window")
        or ("morning" if local.hour < 12 else "afternoon" if local.hour < 17 else "evening"),
        "budget_max_inr": original.get("budget_max_inr")
        or round(booking.price_inr * 1.2 / 50) * 50,
        "budget_given": True,
    }
    sources = {key: f"booking:{booking.booking_id}" for key in slots}
    preferred = (
        ("earliest_time",)
        if original.get("budget_max_inr")
        else ("earliest_time", "budget_max_inr")
    )
    for slot in preferred:
        fact = memory.for_slot(booking.user_id, slot, service=booking.service)
        if fact is not None and fact.status == FactStatus.ACTIVE:
            slots[slot] = fact.value
            sources[slot] = f"memory:{fact.fact_id}"
    if isinstance(original.get("date_to"), str) and date.fromisoformat(original["date_to"]) > first:
        slots["date_to"] = min(
            date.fromisoformat(original["date_to"]), first + timedelta(days=6)
        ).isoformat()
    return slots, sources
