"""User-facing text rendered by code from tool data.

Anything with a price, a time, a provider or a commitment in it comes from here, never from the
model, so nothing the user is asked to approve can be invented and no outcome is claimed that the
ledger hasn't confirmed.
"""

from datetime import date, datetime

from errand.contracts import IST, Booking, Hold, Slot
from errand.mocks.catalog import SERVICES

ASK = {
    "service": "what you need done (plumber, electrician, AC servicing, cleaning, carpentry, painting or pest control)",
    "area": "which area to search (for example Koramangala, HSR Layout or Indiranagar)",
    "date_from": "which day or days suit you (for example tomorrow, this week, or Saturday)",
    "budget": 'any budget limit (or say "no limit")',
}


def money(amount: int) -> str:
    return f"₹{amount:,}"


def day(moment: datetime | date) -> str:
    if isinstance(moment, datetime):
        moment = moment.astimezone(IST)
    return f"{moment:%a} {moment.day} {moment:%b}"


def when(slot: Slot) -> str:
    start, end = slot.start.astimezone(IST), slot.end.astimezone(IST)
    return f"{day(start)}, {start:%H:%M}–{end:%H:%M}"


def service_name(service: str | None) -> str:
    """The service in running text: "plumber", "home cleaning", but "AC servicing"."""
    label = SERVICES.get(service or "", service or "service")
    return label if label[:2].isupper() else label[0].lower() + label[1:]


def ask_missing(missing: list[str], slots: dict) -> str:
    wanted = [ASK[item] for item in missing if item in ASK]
    lead = f"To find you a {service_name(slots['service'])}" if slots.get("service") else "To help"
    if len(wanted) == 1:
        return f"{lead}, I still need {wanted[0]}."
    return f"{lead}, I still need:\n" + "\n".join(f"- {item}" for item in wanted)


def known_details(slots: dict) -> str:
    parts = [service_name(slots.get("service"))]
    if slots.get("area"):
        parts.append(f"in {slots['area']}")
    if slots.get("date_from"):
        first, last = date.fromisoformat(slots["date_from"]), date.fromisoformat(slots["date_to"])
        parts.append(
            f"on {day(first)}" if first == last else f"between {day(first)} and {day(last)}"
        )
    if slots.get("time_window") and slots["time_window"] != "any":
        parts.append(f"{slots['time_window']}s")
    if slots.get("earliest_time"):
        parts.append(f"not before {slots['earliest_time']}")
    if slots.get("budget_max_inr"):
        parts.append(f"up to {money(slots['budget_max_inr'])}")
    if slots.get("provider_label"):
        parts.append(f"with {slots['provider_label']}")
    head, *rest = parts
    return f"{head} {', '.join(rest)}" if rest else head


def options(options: list[dict], slots: dict, notes: list[str]) -> str:
    lines = (
        [*notes, f"Options for {known_details(slots)}:"]
        if notes
        else [f"Options for {known_details(slots)}:"]
    )
    for option in options:
        slot = Slot.model_validate(option["slot"])
        flags = []
        if option.get("flagged_reviews"):
            flags.append(f"{option['flagged_reviews']} suspicious review ignored")
        detail = f" ({'; '.join(flags)})" if flags else ""
        lines.append(
            f"{option['option_id']}. {option['name']}: {when(slot)}, {money(option['price_inr'])},"
            f" rated {option['rating']}{detail}"
        )
    lines.append("Reply with a number to choose one, or tell me what to change.")
    return "\n".join(lines)


def no_options(slots: dict, notes: list[str]) -> str:
    return "\n".join(
        [
            *notes,
            f"I couldn't find any {known_details(slots)}.",
            "Want me to try other days, a later time, or a higher budget?",
        ]
    )


def booking_summary(
    option: dict, hold: Hold | None, slots: dict, notes: list[str], *, now: datetime
) -> str:
    """The summary the user approves. Before any hold it shows the listed price; once a hold has
    quoted a price, that quote."""
    slot = hold.slot if hold else Slot.model_validate(option["slot"])
    price_inr = hold.quoted_price_inr if hold else option["price_inr"]
    lines = [*notes, "Please confirm this booking:"]
    lines.append(f"- Service: {service_name(slots.get('service'))}")
    lines.append(f"- Provider: {option['name']} (rated {option['rating']})")
    lines.append(f"- When: {when(slot)}")
    price = money(price_inr)
    if price_inr != option["price_inr"]:
        price += f" (quoted when I held the slot; the listing said {money(option['price_inr'])})"
    lines.append(f"- Price: {price}")
    if hold and hold.expires_at > now:
        lines.append(f"- Slot held until {hold.expires_at.astimezone(IST):%H:%M}")
    else:
        lines.append(
            "- After your yes I hold the slot and book it. If the price changes, I ask again."
        )
    lines.append('Reply "yes" to book exactly this, or tell me what to change.')
    return "\n".join(lines)


def cancel_summary(booking: Booking, notes: list[str] = ()) -> str:
    return "\n".join(
        [
            *notes,
            "Please confirm this cancellation:",
            f"- Booking {booking.booking_id}: {service_name(booking.service)} with {booking.provider_name}",
            f"- When: {when(booking.slot)}",
            f"- Price: {money(booking.price_inr)}",
            'Reply "yes" to cancel exactly this booking, or "no" to keep it.',
        ]
    )


def reschedule_summary(booking: Booking, new_slot: Slot, notes: list[str] = ()) -> str:
    return "\n".join(
        [
            *notes,
            "Please confirm this change:",
            f"- Booking {booking.booking_id}: {service_name(booking.service)} with {booking.provider_name}",
            f"- From: {when(booking.slot)}",
            f"- To: {when(new_slot)}",
            f"- Price: {money(booking.price_inr)} (unchanged)",
            'Reply "yes" to move it, or tell me another day or time.',
        ]
    )


def done(intent: str, booking: Booking) -> str:
    if intent == "cancel":
        return f"Cancelled. Booking {booking.booking_id} with {booking.provider_name} on {when(booking.slot)} is off."
    if intent == "reschedule":
        return f"Rescheduled. Booking {booking.booking_id} with {booking.provider_name} is now {when(booking.slot)}."
    return (
        f"Booked. Confirmation {booking.booking_id}: {booking.provider_name},"
        f" {when(booking.slot)}, {money(booking.price_inr)}."
    )


def already(intent: str, booking: Booking) -> str:
    """A repeated confirmation: what the earlier one did, and where the booking stands now."""
    verb = {"cancel": "cancelled", "reschedule": "moved"}.get(intent, "booked")
    status = booking.status.value.replace("_", " ")
    return (
        f"That's already {verb}, so I didn't do it again. Booking {booking.booking_id} with"
        f" {booking.provider_name}, {when(booking.slot)}, is {status}."
    )


def unconfirmed(intent: str) -> str:
    action = {"book": "booking", "cancel": "cancellation", "reschedule": "change"}.get(
        intent, "request"
    )
    return (
        f"I couldn't confirm your {action} yet: the provider didn't give a clear answer. I'm"
        " checking with them and will update you here. I won't try it again until I know."
    )


def bookings_list(bookings: list[Booking]) -> str:
    return "\n".join(
        f"{n}. {service_name(b.service)} with {b.provider_name}, {when(b.slot)} ({b.booking_id})"
        for n, b in enumerate(bookings, start=1)
    )
