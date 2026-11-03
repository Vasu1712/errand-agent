"""Relative days to dates, and the time filters rank_options applies. Pure code, no model."""

from datetime import date, datetime, time, timedelta

from errand.contracts import IST

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# A slot belongs to a window by its start time.
TIME_WINDOWS = {
    "morning": (time(0), time(12)),
    "afternoon": (time(12), time(17)),
    "evening": (time(17), time(23, 59)),
}
MAX_WINDOW_DAYS = 14
MAX_DAYS_AHEAD = 30


def resolve_when(
    kind: str | None, weekday: str | None, on: str | None, today: date
) -> tuple[date, date] | None:
    """(first day, last day) for a relative day, or None if it can't be worked out or is past."""
    if kind is None:
        return None
    if kind == "today":
        return today, today
    if kind == "tomorrow":
        return today + timedelta(days=1), today + timedelta(days=1)
    if kind == "this_week":
        return today, today + timedelta(days=6 - today.weekday())
    if kind == "next_week":
        monday = today + timedelta(days=7 - today.weekday())
        return monday, monday + timedelta(days=6)
    if kind == "this_weekend":
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        if today.weekday() == 6:  # it's Sunday: this weekend is today
            saturday = today
        return saturday, today + timedelta(days=(6 - today.weekday()) % 7)
    if kind == "weekday" and weekday in WEEKDAYS:
        day = today + timedelta(days=(WEEKDAYS.index(weekday) - today.weekday()) % 7)
        return day, day
    if kind == "date" and on:
        try:
            day = date.fromisoformat(on)
        except ValueError:
            return None
        if today <= day <= today + timedelta(days=MAX_DAYS_AHEAD):
            return day, day
    return None


def parse_clock(text: str | None) -> time | None:
    if not text:
        return None
    try:
        hour, minute = (int(part) for part in text.split(":"))
        return time(hour, minute)
    except (ValueError, TypeError):
        return None


def slot_fits(start: datetime, slots: dict) -> bool:
    """Whether a slot start passes the task's hard filters: days, time of day, earliest time."""
    local = start.astimezone(IST)
    if slots.get("date_from") and local.date() < date.fromisoformat(slots["date_from"]):
        return False
    if slots.get("date_to") and local.date() > date.fromisoformat(slots["date_to"]):
        return False
    window = TIME_WINDOWS.get(slots.get("time_window") or "")
    if window and not window[0] <= local.time() < window[1]:
        return False
    earliest = parse_clock(slots.get("earliest_time"))
    return not (earliest and local.time() < earliest)
