"""A regex stand-in for Qwen, for offline tests and evals.

It reads only the latest user message (the last message of every request), the same contract the
live prompts impose, and returns the same schemas. It is deliberately literal: what it can't
parse it leaves out, so missing details get asked for, as they would with a careful model.
"""

import re
from typing import Any

from errand.llm.base import ModelRequest
from errand.llm.scripted import ScriptedModel
from errand.mocks.catalog import find_area

_SERVICES = (
    (r"plumb|tap|leak|pipe|drain", "plumber"),
    (r"electric|wiring|switch|socket", "electrician"),
    (r"\bac\b|air ?con", "ac_service"),
    (r"carpent|door|hinge|furniture", "carpenter"),
    (r"clean", "cleaning"),
    (r"paint", "painter"),
    (r"pest|cockroach|termite", "pest_control"),
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _latest(request: ModelRequest) -> str:
    return request.messages[-1].content


def _when(text: str) -> dict[str, Any]:
    lowered = text.lower()
    if date := re.search(r"\b(20\d\d-\d\d-\d\d)\b", lowered):
        return {"when": "date", "when_date": date.group(1)}
    for pattern, kind in (
        (r"\btoday\b", "today"),
        (r"\btomorrow\b", "tomorrow"),
        (r"\bnext week\b", "next_week"),
        (r"\bthis week\b", "this_week"),
        (r"\bweekend\b", "this_weekend"),
    ):
        if re.search(pattern, lowered):
            return {"when": kind}
    # "move X to saturday": the day after "to" is the new day.
    target = re.search(r"\bto (?:this |next )?(" + "|".join(_WEEKDAYS) + r")\b", lowered)
    days = re.findall(r"\b(" + "|".join(_WEEKDAYS) + r")\b", lowered)
    if target:
        found: dict[str, Any] = {"when": "weekday", "when_weekday": target.group(1)}
        earlier = [day for day in days if day != target.group(1)]
        if earlier:
            found |= {"booking_when": "weekday", "booking_weekday": earlier[0]}
        return found
    if days:
        return {"when": "weekday", "when_weekday": days[0]}
    return {}


def _clock(hour: str, minute: str | None, meridiem: str | None) -> str:
    h = int(hour) % 12 if meridiem else int(hour)
    if meridiem and meridiem.startswith("p"):
        h += 12
    return f"{h:02d}:{int(minute or 0):02d}"


def slot_patch(request: ModelRequest) -> dict[str, Any]:
    text = _latest(request)
    lowered = text.lower()
    patch: dict[str, Any] = {}
    for pattern, service in _SERVICES:
        if re.search(pattern, lowered):
            patch["service"] = service
            break
    if area := find_area(text):
        patch["area"] = area
    patch |= _when(text)
    for window in ("morning", "afternoon", "evening"):
        if window in lowered:
            patch["time_window"] = window
    if earliest := re.search(
        r"(?:not before|after|no earlier than)\s+(\d{1,2})(?::(\d\d))?\s*(a\.?m\.?|p\.?m\.?)?",
        lowered,
    ):
        patch["earliest_time"] = _clock(*earliest.groups())
    if budget := re.search(
        r"(?:under|below|max(?:imum)?|budget(?: of| is)?|up to|upto)\s*(?:₹|rs\.?|inr)?\s*([\d,]{3,})",
        lowered,
    ):
        patch["budget_max_inr"] = int(budget.group(1).replace(",", ""))
    elif re.fullmatch(r"\s*(?:₹|rs\.?)?\s*([\d,]{3,})\s*(?:rupees|rs)?\s*", lowered):
        patch["budget_max_inr"] = int(re.sub(r"\D", "", lowered))
    if re.search(r"\bno (?:budget|limit|max)", lowered):
        patch["no_budget_limit"] = True
    if re.search(r"\bsame (?:\w+ )?(?:as last time|as before|one again)", lowered):
        patch["same_provider_as_last_time"] = True
    return patch


def pick(request: ModelRequest) -> dict[str, Any]:
    lowered = _latest(request).lower()
    number = re.search(r"(?:option|number|#)?\s*\b([1-8])\b", lowered)
    if number and not re.search(r"\b(am|pm|a\.m|p\.m)\b", lowered):
        return {"action": "propose_booking", "option_id": int(number.group(1))}
    for word, n in _ORDINALS.items():
        if re.search(rf"\b{word}\b", lowered):
            return {"action": "propose_booking", "option_id": n}
    if slot_patch(request):
        return {"action": "change_details", "option_id": None}
    return {"action": "none", "option_id": None}


def memory_command(request: ModelRequest) -> dict[str, Any]:
    text = _latest(request)
    lowered = text.lower()
    key = value = None
    if "area" in lowered or find_area(text):
        key, value = "preferred_area", find_area(text)
    elif re.search(r"earliest|not before|start time", lowered):
        key = "earliest_time"
        if found := re.search(r"(\d{1,2})(?::(\d\d))?\s*(a\.?m\.?|p\.?m\.?)?", lowered):
            value = _clock(*found.groups())
    elif "budget" in lowered:
        key = "budget_max_inr"
        if found := re.search(r"([\d,]{3,})", lowered):
            value = found.group(1).replace(",", "")
    elif re.search(r"technician|electrician|plumber", lowered):
        key = "technician"
    if re.search(r"\b(forget|delete|remove|stop (?:remembering|using))\b", lowered):
        return {"op": "forget", "key": key}
    if key and value:
        return {"op": "remember", "key": key, "value": value}
    return {"op": "inspect"}


def smalltalk(request: ModelRequest) -> dict[str, Any]:
    return {"text": "Happy to help. I can book, reschedule or cancel home services for you."}


def rule_model() -> ScriptedModel:
    return ScriptedModel(
        {
            "slot_patch": slot_patch,
            "find_booking": slot_patch,
            "pick": pick,
            "memory_command": memory_command,
            "smalltalk": smalltalk,
        },
        name="rules",
    )
