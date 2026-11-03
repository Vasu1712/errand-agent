"""Applying a model's slot patch: code accepts only what the user's message supports.

Slots change only from user messages. The model proposes a patch from the latest message, and code
keeps a value only if it can be grounded in that same message: the area must be named in it, a
budget or a time must appear in it as a number, and a different service needs a matching word.
Relative days are resolved here, not by the model.
"""

import re
from datetime import date

from errand.graph.dates import parse_clock, resolve_when
from errand.llm.schemas import SlotPatch
from errand.mocks.catalog import find_area, normalize_area

SERVICE_HINTS = {
    "plumber": r"plumb|\btap|leak|pipe|drain|toilet|sink|geyser|water",
    "electrician": r"electric|wiring|\bwire|switch|socket|\bfan\b|light|power|mcb",
    "ac_service": r"\bac\b|a\.c\.|air ?con|cooling",
    "carpenter": r"carpent|door|hinge|furniture|\bwood|cupboard|wardrobe",
    "cleaning": r"clean|maid|sofa",
    "painter": r"paint|\bwalls?\b",
    "pest_control": r"pest|cockroach|termite|\brats?\b|mosquito|bed ?bug",
}


WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
DAY_WORDS = re.compile(
    r"\b(today|tonight|tomorrow|week|weekend|" + "|".join(WEEKDAY_NAMES) + r"|\d{4}-\d{2}-\d{2})\b"
)
WINDOW_WORDS = re.compile(r"\b(morning|afternoon|evening|night)s?\b|\bany ?time\b")
SAME_WORDS = re.compile(r"\bsame\b|\blast time\b|\bagain\b|\bprevious\b")


def named_weekday(lowered: str) -> str | None:
    """The weekday the user named: the one after "to" when moving ("to Saturday"), or the only
    one mentioned."""
    target = re.search(r"\bto (?:this |next )?(" + "|".join(WEEKDAY_NAMES) + r")\b", lowered)
    if target:
        return target.group(1)
    days = set(re.findall(r"\b(" + "|".join(WEEKDAY_NAMES) + r")\b", lowered))
    return days.pop() if len(days) == 1 else None


def _numbers(text: str) -> set[int]:
    found = {int(n) for n in re.findall(r"\d+", text.replace(",", ""))}
    found |= {round(float(k) * 1000) for k in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", text.lower())}
    return found


def ground_patch(
    patch: SlotPatch, message: str, current: dict, today: date
) -> tuple[dict, dict[str, str]]:
    """The slot values the message supports, and what was dropped and why."""
    accepted: dict = {}
    ignored: dict[str, str] = {}
    lowered = message.lower()
    numbers = _numbers(message)

    if patch.service:
        same = current.get("service") in (None, patch.service)
        if same or re.search(SERVICE_HINTS[patch.service], lowered):
            accepted["service"] = patch.service
        else:
            ignored["service"] = f"{patch.service}: not named in the message"
    elif not current.get("service"):
        # The model missed it, but the user's own words may name exactly one service.
        hinted = [svc for svc, pattern in SERVICE_HINTS.items() if re.search(pattern, lowered)]
        if len(hinted) == 1:
            accepted["service"] = hinted[0]

    if patch.area:
        area = normalize_area(patch.area) or find_area(patch.area)
        mentioned = find_area(message)
        if area and mentioned == area:
            accepted["area"] = area
        elif area is None and mentioned is None:
            ignored["area"] = f"{patch.area}: not an area I cover"
        else:
            ignored["area"] = f"{patch.area}: not named in the message"

    if patch.when:
        kind, weekday = patch.when, patch.when_weekday
        named = named_weekday(lowered)
        if named and kind != "date":
            kind, weekday = "weekday", named  # the user's own word beats the model's reading
        window = resolve_when(kind, weekday, patch.when_date, today)
        if not DAY_WORDS.search(lowered):
            ignored["when"] = f"{patch.when}: the message names no day"
        elif window:
            accepted["date_from"], accepted["date_to"] = (d.isoformat() for d in window)
        else:
            ignored["when"] = f"{kind} {weekday or patch.when_date or ''}".strip()

    elif named := named_weekday(lowered):
        # The model left the day out (it put "Thursday afternoon" in the problem text, say), but
        # the user named exactly one weekday: their own word sets it (docs/bugs.md #25).
        window = resolve_when("weekday", named, None, today)
        if window:
            accepted["date_from"], accepted["date_to"] = (d.isoformat() for d in window)

    if patch.time_window:
        if WINDOW_WORDS.search(lowered):
            accepted["time_window"] = patch.time_window
        else:
            ignored["time_window"] = f"{patch.time_window}: not said in the message"
    else:
        windows = {w for w in re.findall(r"\b(morning|afternoon|evening)s?\b", lowered)}
        if len(windows) == 1:  # likewise, one time of day the user named
            accepted["time_window"] = windows.pop()

    if patch.earliest_time:
        clock = parse_clock(patch.earliest_time)
        if clock and numbers & {clock.hour, clock.hour - 12, clock.hour + 12}:
            accepted["earliest_time"] = f"{clock:%H:%M}"
        else:
            ignored["earliest_time"] = f"{patch.earliest_time}: no such time in the message"

    if patch.no_budget_limit:
        accepted["budget_max_inr"] = None
        accepted["budget_given"] = True
    elif patch.budget_max_inr:
        if patch.budget_max_inr in numbers:
            accepted["budget_max_inr"] = patch.budget_max_inr
            accepted["budget_given"] = True
        else:
            ignored["budget_max_inr"] = f"{patch.budget_max_inr}: no such amount in the message"

    if patch.same_provider_as_last_time:
        if SAME_WORDS.search(lowered):
            accepted["same_provider"] = True
        else:
            ignored["same_provider"] = "not asked for in the message"
    if patch.problem:
        accepted["problem"] = patch.problem[:80]
    return accepted, ignored


def booking_day(
    patch: SlotPatch, message: str, intent: str, today: date
) -> tuple[date, date] | None:
    """The day that points at an existing booking, if the user named one. For a cancel any day
    named does; for a reschedule only a day other than the new one ("move my Thursday visit to
    Saturday")."""
    lowered = message.lower()
    if not DAY_WORDS.search(lowered):
        return None
    if intent == "cancel":
        named = named_weekday(lowered)
        if named:
            return resolve_when("weekday", named, None, today)
        return resolve_when(
            patch.booking_when or patch.when,
            patch.booking_weekday or patch.when_weekday,
            patch.booking_date or patch.when_date,
            today,
        )
    target = named_weekday(lowered)
    others = [d for d in WEEKDAY_NAMES if re.search(rf"\b{d}\b", lowered) and d != target]
    return resolve_when("weekday", others[0], None, today) if others else None
