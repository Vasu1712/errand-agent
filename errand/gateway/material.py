"""What an approval covers: the material details of one consequential action.

The approval hash is sha256 over action, provider, slot, price and booking id, in canonical form.
Slots are normalised to IST at minute precision first, so the same slot always hashes the same
whichever timezone a tool happened to report it in.
"""

from typing import Any

from errand.contracts import IST, Booking, Slot
from errand.store.records import material_hash

__all__ = [
    "booking_material",
    "cancel_material",
    "canonical_slot",
    "material_hash",
    "reschedule_material",
]


def canonical_slot(slot: Slot) -> dict[str, str]:
    return {
        "start": slot.start.astimezone(IST).isoformat(timespec="minutes"),
        "end": slot.end.astimezone(IST).isoformat(timespec="minutes"),
    }


def booking_material(*, provider_id: str, slot: Slot, price_inr: int) -> dict[str, Any]:
    return {
        "action": "book",
        "provider_id": provider_id,
        "slot": canonical_slot(slot),
        "price_inr": price_inr,
        "booking_id": None,
    }


def cancel_material(booking: Booking) -> dict[str, Any]:
    return {
        "action": "cancel",
        "provider_id": booking.provider_id,
        "slot": canonical_slot(booking.slot),
        "price_inr": booking.price_inr,
        "booking_id": booking.booking_id,
    }


def reschedule_material(booking: Booking, new_slot: Slot) -> dict[str, Any]:
    return {
        "action": "reschedule",
        "provider_id": booking.provider_id,
        "slot": canonical_slot(new_slot),
        "price_inr": booking.price_inr,
        "booking_id": booking.booking_id,
    }
