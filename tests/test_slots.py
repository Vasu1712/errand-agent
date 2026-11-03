"""Grounding slot patches in the user's own words. The patches here are what qwen2.5:7b actually
returned for these messages during the live check (docs/bugs.md #6)."""

from datetime import date

from errand.graph.slots import booking_day, ground_patch
from errand.llm.schemas import SlotPatch

TODAY = date(2026, 11, 3)  # a Tuesday
CURRENT = {
    "service": "plumber",
    "date_from": "2026-11-03",
    "date_to": "2026-11-08",
    "time_window": "morning",
    "budget_max_inr": 1500,
}


def test_an_area_answer_changes_only_the_area():
    patch = SlotPatch(
        service="plumber",
        area="Koramangala",
        when="today",
        time_window="morning",
        earliest_time="09:00",
    )
    accepted, ignored = ground_patch(patch, "Koramangala", CURRENT, TODAY)
    assert accepted == {"service": "plumber", "area": "Koramangala"}
    assert set(ignored) == {"when", "time_window", "earliest_time"}


def test_a_named_weekday_beats_the_models_reading():
    patch = SlotPatch(when="this_weekend", booking_when="this_weekend")
    accepted, _ = ground_patch(patch, "make it Saturday instead", CURRENT, TODAY)
    assert (accepted["date_from"], accepted["date_to"]) == ("2026-11-07", "2026-11-07")


def test_only_what_was_said_survives():
    patch = SlotPatch(
        service="electrician",
        when="today",
        time_window="morning",
        earliest_time="10:00",
        same_provider_as_last_time=True,
    )
    message = "Book the same electrician as last time, but not before 10 a.m."
    accepted, ignored = ground_patch(patch, message, {}, TODAY)
    assert accepted == {"service": "electrician", "earliest_time": "10:00", "same_provider": True}
    assert set(ignored) == {"when", "time_window"}


def test_a_missed_service_is_read_from_the_users_words():
    patch = SlotPatch(when="this_weekend", booking_when="this_weekend")
    accepted, _ = ground_patch(patch, "Move my AC servicing to Saturday", {}, TODAY)
    assert accepted["service"] == "ac_service"
    assert accepted["date_from"] == accepted["date_to"] == "2026-11-07"


def test_hallucinated_amounts_areas_and_services_are_dropped():
    patch = SlotPatch(service="painter", area="Indiranagar", budget_max_inr=2000)
    accepted, ignored = ground_patch(patch, "tomorrow works", CURRENT, TODAY)
    assert accepted == {}
    assert set(ignored) == {"service", "area", "budget_max_inr"}


def test_the_booking_day_is_the_one_that_is_not_the_new_day():
    patch = SlotPatch(when="weekday", when_weekday="saturday")
    assert booking_day(patch, "Move my Thursday AC servicing to Saturday", "reschedule", TODAY) == (
        date(2026, 11, 5),
        date(2026, 11, 5),
    )
    assert booking_day(patch, "Move my AC servicing to Saturday", "reschedule", TODAY) is None
    cancel = SlotPatch(when="weekday", when_weekday="saturday")
    assert booking_day(cancel, "Cancel my Saturday cleaning", "cancel", TODAY) == (
        date(2026, 11, 7),
        date(2026, 11, 7),
    )
