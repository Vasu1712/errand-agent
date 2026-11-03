"""Per-scenario expected outcomes, named by ``check:`` in a scenario file.

Every check reads the ledger, approvals, memory rows, the mock world, the trace or the requests
the model received, and renders what it expected and what it found, which become the eval
table's columns. None of them read hand-written reply wording; ``echoed`` compares a reply with
the renderer's own text for the resolved details.
"""

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from errand.contracts import IST, Booking, BookingState
from errand.eval.invariants import CONSTRAINT_SLOTS, started_slots
from errand.eval.runner import Observation
from errand.graph import render
from errand.graph.render import money


@dataclass
class CheckResult:
    check: str
    expected: str
    actual: str
    passed: bool


Check = Callable[[Observation, dict[str, Any]], CheckResult]
CHECKS: dict[str, Check] = {}


def check(name: str) -> Callable[[Check], Check]:
    def register(fn: Check) -> Check:
        CHECKS[name] = fn
        return fn

    return register


def run_check(obs: Observation, name: str, params: dict[str, Any]) -> CheckResult:
    if name not in CHECKS:
        return CheckResult(name, f"a check named {name!r}", "no such check", False)
    try:
        return CHECKS[name](obs, params)
    except Exception as exc:  # a broken check fails its scenario instead of the whole suite
        return CheckResult(name, name, f"the check failed: {type(exc).__name__}: {exc}", False)


def _booking(b: Booking) -> str:
    local = b.slot.start.astimezone(IST)
    return f"{b.booking_id} {b.provider_id} {local:%a %-d %b %H:%M} {money(b.price_inr)}"


def _day(text: str) -> str:
    return f"{date.fromisoformat(text):%a %-d %b}"


def _ref(obs: Observation, value: str) -> str:
    """A setup alias stands for its booking id."""
    return obs.aliases.get(value, value)


def _who(obs: Observation, p: dict) -> str:
    """The user a check is about: its ``user`` parameter, or the scenario's user."""
    return p.get("user") or obs.user_id


def _for(p: dict) -> str:
    return f"{p['user']}: " if p.get("user") else ""


# Bookings


@check("booked")
def booked(obs: Observation, p: dict) -> CheckResult:
    """New confirmed bookings made during the scenario (for one user), and their details."""
    new = [
        b
        for b in obs.bookings(_who(obs, p))
        if b.booking_id not in obs.initial_bookings
        and b.status in (BookingState.CONFIRMED, BookingState.COMPLETED)
    ]
    want = p.get("count", 1)
    rules, problems = [], []
    for b in new:
        local = b.slot.start.astimezone(IST)
        if (service := p.get("service")) and b.service != service:
            problems.append(f"{b.booking_id} is {b.service}")
        if (provider := p.get("provider")) and b.provider_id != provider:
            problems.append(f"{b.booking_id} is with {b.provider_id}")
        if (other := p.get("not_provider")) and b.provider_id == other:
            problems.append(f"{b.booking_id} is with {other} again")
        if (day := p.get("day")) and local.date().isoformat() != day:
            problems.append(f"{b.booking_id} is on {local:%a %-d %b}")
        if (days := p.get("days")) and not days[0] <= local.date().isoformat() <= days[1]:
            problems.append(f"{b.booking_id} is on {local:%a %-d %b}")
        if (low := p.get("min_hour")) is not None and local.hour < low:
            problems.append(f"{b.booking_id} starts {local:%H:%M}")
        if (high := p.get("max_hour")) is not None and local.hour >= high:
            problems.append(f"{b.booking_id} starts {local:%H:%M}")
        if (price := p.get("max_price")) and b.price_inr > price:
            problems.append(f"{b.booking_id} costs {money(b.price_inr)}")
        if (price := p.get("price")) and b.price_inr != price:
            problems.append(f"{b.booking_id} costs {money(b.price_inr)}")
        if (area := p.get("area")) and obs.world.get_provider(b.provider_id).area != area:
            problems.append(f"{b.booking_id} is outside {area}")
    for key, text in (
        ("service", "{}"),
        ("provider", "with {}"),
        ("not_provider", "not {}"),
        ("area", "in {}"),
    ):
        if key in p:
            rules.append(text.format(p[key]))
    if "day" in p:
        rules.append(f"on {_day(p['day'])}")
    if "days" in p:
        rules.append(f"{_day(p['days'][0])}–{_day(p['days'][1])}")
    if "min_hour" in p or "max_hour" in p:
        rules.append(f"starting {p.get('min_hour', 0):02d}:00–{p.get('max_hour', 24):02d}:00")
    if "max_price" in p:
        rules.append(f"≤ {money(p['max_price'])}")
    if "price" in p:
        rules.append(f"at {money(p['price'])}")
    noun = "booking" if want == 1 else "bookings"
    expected = _for(p) + f"{want} new {noun}" + (f" ({', '.join(rules)})" if rules else "")
    actual = f"{len(new)} new" + (": " + "; ".join(_booking(b) for b in new) if new else "")
    if problems:
        actual += " [" + "; ".join(problems) + "]"
    return CheckResult("booked", expected, actual, len(new) == want and not problems)


@check("booking")
def booking(obs: Observation, p: dict) -> CheckResult:
    """The final state of a booking that existed before the scenario."""
    select = p["select"]
    who = _who(obs, p)
    before = [
        b
        for b in obs.initial_bookings.values()
        if (
            b.booking_id == _ref(obs, select)
            if isinstance(select, str)
            else b.service == select["service"]
            and b.status is BookingState.CONFIRMED
            and b.user_id == who
        )
    ]
    label = select if isinstance(select, str) else f"{_for(p)}the {select['service']} booking"
    if len(before) != 1:
        return CheckResult("booking", label, f"{len(before)} bookings match", False)
    old = before[0]
    now = next(b for b in obs.bookings(old.user_id) if b.booking_id == old.booking_id)
    wants, problems = [], []
    if status := p.get("status"):
        wants.append(status)
        if now.status.value != status:
            problems.append(f"is {now.status.value}")
    local = now.slot.start.astimezone(IST)
    if day := p.get("day"):
        wants.append(f"on {_day(day)}")
        if local.date().isoformat() != day:
            problems.append(f"is on {local:%a %-d %b}")
    if (hour := p.get("hour")) is not None:
        wants.append(f"at {hour:02d}:00")
        if local.hour != hour:
            problems.append(f"is at {local:%H:%M}")
    if p.get("same_provider"):
        wants.append("same provider")
        if now.provider_id != old.provider_id:
            problems.append(f"moved to {now.provider_id}")
    if p.get("same_price"):
        wants.append("same price")
        if now.price_inr != old.price_inr:
            problems.append(f"now costs {money(now.price_inr)}")
    expected = f"{label} {old.booking_id}: {', '.join(wants)}"
    actual = f"{now.booking_id} {now.status.value}, {local:%a %-d %b %H:%M}, {money(now.price_inr)}"
    if problems:
        actual += " [" + "; ".join(problems) + "]"
    return CheckResult("booking", expected, actual, not problems)


# Ledger, tool calls, approvals, claims


@check("ledger")
def ledger(obs: Observation, p: dict) -> CheckResult:
    """The ledger rows for one operation, oldest first: their statuses and attempts."""
    op = p["op"]
    rows = obs.scan(
        "SELECT status, attempts FROM ledger WHERE op = ? AND user_id = ?"
        " ORDER BY created_at, rowid",
        (op, _who(obs, p)),
    )
    statuses = [row["status"] for row in rows]
    attempts = [row["attempts"] for row in rows]
    expected = _for(p) + f"{op} ledger rows: {', '.join(p['statuses']) or 'none'}"
    passed = statuses == p["statuses"]
    if "attempts" in p:
        expected += f" (attempts {p['attempts']})"
        passed = passed and attempts == p["attempts"]
    actual = f"{', '.join(statuses) or 'none'} (attempts {attempts})"
    return CheckResult("ledger", expected, actual, passed)


@check("calls")
def calls(obs: Observation, p: dict) -> CheckResult:
    """How often one tool was called (optionally in one step), under which keys, with what
    failures."""
    tool, step = p["tool"], p.get("step")
    found = [event.data for _n, _run, event in obs.tool_calls(tool, step, p.get("user"))]
    where = f" in step {step}" if step else ""
    expected = _for(p) + f"{tool} called {p['count']}×{where}"
    passed = len(found) == p["count"]
    keys = {call.get("args", {}).get("idempotency_key") for call in found}
    failures = [(call.get("failure") or {}).get("code") for call in found]
    if p.get("same_key"):
        expected += ", one idempotency key"
        passed = passed and len(keys) == 1
    if "failures" in p:
        expected += f", failures {p['failures']}"
        passed = passed and failures == p["failures"]
    actual = f"{len(found)}×, {len(keys)} key(s), failures {failures}"
    return CheckResult("calls", expected, actual, passed)


@check("sequence")
def sequence(obs: Observation, p: dict) -> CheckResult:
    """Exactly these tool calls, in this order, in one step."""
    step = p["step"]
    tools = [
        event.data["tool"] for _n, _run, event in obs.tool_calls(step=step, user=p.get("user"))
    ]
    expected = _for(p) + f"step {step} calls {' → '.join(p['tools']) or 'nothing'}"
    return CheckResult("sequence", expected, " → ".join(tools) or "no calls", tools == p["tools"])


@check("approvals")
def approvals(obs: Observation, p: dict) -> CheckResult:
    """Approvals by final status, optionally only those on one thread or requested after a step."""
    who = _who(obs, p)
    rows = obs.scan(
        "SELECT a.*, t.thread_id FROM approvals a JOIN tasks t USING (task_id) WHERE a.user_id = ?",
        (who,),
    )
    if thread := p.get("thread"):
        rows = [row for row in rows if row["thread_id"] == obs.thread(thread, who)]
    if (after := p.get("requested_after_step")) is not None:
        late = {
            event.data["approval_id"]
            for _n, run, event in obs.events("approval")
            if event.data.get("event") == "requested" and run.step > after
        }
        rows = [row for row in rows if row["approval_id"] in late]
    counts = Counter(row["status"] for row in rows)
    wanted = {
        status: p[status]
        for status in ("consumed", "invalidated", "granted", "pending")
        if status in p
    }
    scope = []
    if thread:
        scope.append(f"on the {thread} thread")
    if after is not None:
        scope.append(f"requested after step {after}")
    expected = (
        _for(p)
        + "approvals"
        + (f" {' '.join(scope)}" if scope else "")
        + ": "
        + ", ".join(f"{n} {status}" for status, n in wanted.items())
    )
    passed = all(counts.get(status, 0) == n for status, n in wanted.items())
    if p.get("distinct"):
        spent = [row["material"] for row in rows if row["status"] == "consumed"]
        expected += ", each spent on different details"
        passed = passed and len(set(spent)) == len(spent)
    actual = ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "none"
    return CheckResult("approvals", expected, actual, passed)


@check("approval_trail")
def approval_trail(obs: Observation, p: dict) -> CheckResult:
    """Every approval the agent asked for, oldest first: its final status and, if given, the price
    it covered; optionally that none covered more than a price."""
    rows = obs.scan(
        "SELECT status, material FROM approvals WHERE user_id = ? ORDER BY created_at, rowid",
        (_who(obs, p),),
    )
    got = [(row["status"], json.loads(row["material"])["price_inr"]) for row in rows]
    wanted = p.get("trail", [])
    passed = len(got) == len(wanted) and all(
        status == w["status"] and w.get("price") in (None, price)
        for (status, price), w in zip(got, wanted, strict=False)
    )
    expected = (
        _for(p)
        + "approvals: "
        + " → ".join(
            w["status"] + (f" at {money(w['price'])}" if "price" in w else "") for w in wanted
        )
    )
    if (ceiling := p.get("max_price")) is not None:
        expected += f"; none above {money(ceiling)}"
        passed = passed and all(price <= ceiling for _status, price in got)
    actual = " → ".join(f"{status} at {money(price)}" for status, price in got) or "none"
    return CheckResult("approval_trail", expected, actual, passed)


@check("same_proposal")
def same_proposal(obs: Observation, p: dict) -> CheckResult:
    """The approvals these users were asked for in one step cover the same provider and slot:
    they are about to compete for it."""
    step, users = p["step"], p["users"]
    found: dict[str, tuple[str, str]] = {}
    for _n, run, event in obs.events("approval"):
        if run.step == step and event.data.get("event") == "requested" and run.user_id in users:
            material = event.data["material"]
            found[run.user_id] = (material["provider_id"], material["slot"]["start"])
    same = len(found) == len(users) and len(set(found.values())) == 1
    expected = f"step {step}: {' and '.join(users)} asked to approve the same provider and slot"
    actual = "; ".join(f"{u}: {prov} at {start}" for u, (prov, start) in found.items()) or "none"
    return CheckResult("same_proposal", expected, actual, same)


@check("offered")
def offered(obs: Observation, p: dict) -> CheckResult:
    """The options shown in one step: how many, and providers that must not be among them."""
    step = p["step"]
    shown = [
        option
        for _n, run, event in obs.events("state.transition")
        if run.step == step and event.data.get("event") == "offered"
        for option in event.data.get("options", [])
    ]
    ids = [option["provider_id"] for option in shown]
    wants, passed = [], bool(shown) or p.get("count") == 0
    if "count" in p:
        wants.append(f"{p['count']} options")
        passed = len(shown) == p["count"]
    for provider in p.get("excludes", []):
        wants.append(f"not {provider}")
        passed = passed and provider not in ids
    expected = f"step {step} offers " + ", ".join(wants)
    return CheckResult("offered", expected, ", ".join(ids) or "no options", passed)


@check("claims")
def claims(obs: Observation, p: dict) -> CheckResult:
    """Successes reported to the user (optionally one user), by operation."""
    user = p.get("user")
    counts = Counter(
        event.data["op"]
        for _n, run, event in obs.events("claim")
        if user is None or run.user_id == user
    )
    wanted = {op: n for op, n in p.items() if op != "user"}
    expected = _for(p) + "reported: " + ", ".join(f"{op} ×{n}" for op, n in wanted.items())
    actual = ", ".join(f"{op} ×{n}" for op, n in sorted(counts.items())) or "nothing"
    return CheckResult("claims", expected, actual, dict(counts) == wanted)


@check("quiet")
def quiet(obs: Observation, p: dict) -> CheckResult:
    """No hold or side effect in runs of these kinds, or at all (nobody had said yes)."""
    kinds, step = p.get("kinds"), p.get("step")
    found = [
        f"{event.data['tool']} in a {run.kind} run, step {run.step}"
        for _n, run, event in obs.tool_calls(step=step, user=p.get("user"))
        if (kinds is None or run.kind in kinds)
        and event.data["tool"] in ("hold_slot", "book", "cancel", "reschedule")
    ]
    scope = f" in {' or '.join(kinds)} runs" if kinds else f" in step {step}" if step else " at all"
    expected = _for(p) + f"no hold or side effect{scope}"
    return CheckResult("quiet", expected, "; ".join(found) or "none", not found)


@check("tasks")
def tasks(obs: Observation, p: dict) -> CheckResult:
    """Tasks the agent opened (for one user, optionally on one session), in order of intent, and
    errands it queued for later."""
    who = _who(obs, p)
    rows = obs.scan(
        "SELECT intent, phase, thread_id FROM tasks WHERE user_id = ? ORDER BY created_at, rowid",
        (who,),
    )
    if session := p.get("session"):
        rows = [row for row in rows if row["thread_id"] == obs.thread(session, who)]
    intents = [row["intent"] for row in rows]
    queued = [
        item["intent"]
        for _n, run, event in obs.events("state.transition")
        if event.data.get("event") == "queued" and run.user_id == who
        for item in event.data.get("queued", [])
    ]
    wants, passed = [_for(p) + (f"session {session}" if session else "")], True
    if "intents" in p:
        wants.append(f"tasks {' then '.join(p['intents'])}")
        passed = intents == p["intents"]
    if "queued" in p:
        wants.append(f"queued {', '.join(p['queued']) or 'nothing'}")
        passed = passed and queued == p["queued"]
    actual = f"tasks {' then '.join(intents) or 'none'}; queued {', '.join(queued) or 'nothing'}"
    return CheckResult("tasks", "; ".join(w for w in wants if w), actual, passed)


# Conversation: questions and slots


def _questions(obs: Observation, user: str | None = None) -> list[tuple[int, list[str], str]]:
    out = []
    for _n, run, event in obs.events("state.transition"):
        if run.user_id != (user or obs.user_id):
            continue
        asked = event.data.get("asked") or {}
        if asked.get("kind") == "slots" and asked.get("missing"):
            out.append((run.step, list(asked["missing"]), "slots"))
        elif asked.get("kind") == "confirm_fact":
            out.append((run.step, [str(asked.get("slot"))], "confirm_fact"))
    return out


@check("asked")
def asked(obs: Observation, p: dict) -> CheckResult:
    """Which details the agent asked for, and how many times each."""
    questions = _questions(obs, p.get("user"))
    counts = Counter(slot for _step, slots, _kind in questions for slot in slots)
    actual = (
        "; ".join(f"step {step}: {', '.join(slots)}" for step, slots, _kind in questions)
        or "no questions"
    )
    if p.get("none"):
        return CheckResult("asked", "no questions asked", actual, not questions)
    wanted = p["slots"]
    expected = (
        "asked once each: " + ", ".join(wanted)
        if all(n == 1 for n in wanted.values())
        else "asked: " + ", ".join(f"{slot} ×{n}" for slot, n in wanted.items())
    )
    passed = dict(counts) == wanted
    if "questions" in p:
        expected += f", in {p['questions']} question(s)"
        passed = passed and len(questions) == p["questions"]
    return CheckResult("asked", expected, actual, passed)


def _slot_history(obs: Observation, thread: str | None):
    """(step, node, key, value, source) for every slot change on one thread, in order. A
    proactive run's starting details count as set by the worker that opened its task."""
    out = []
    for _n, run, event in obs.events():
        if thread and run.thread_id != thread:
            continue
        for key, change in started_slots(run, event).items():
            out.append((run.step, "open_replacement", key, change["value"], change["source"]))
        if event.kind != "state.transition":
            continue
        for key, change in (event.data.get("slots_changed") or {}).items():
            out.append(
                (run.step, event.data.get("node"), key, change.get("value"), change.get("source"))
            )
    return out


@check("slots")
def slots(obs: Observation, p: dict) -> CheckResult:
    """The task's details at the end (values and where each came from), or which details one
    step changed."""
    who = _who(obs, p)
    history = _slot_history(obs, obs.thread(p.get("thread", "main"), who))
    at = p.get("at_step")
    final: dict[str, tuple[Any, Any]] = {}
    for step, _node, key, value, source in history:
        if at is None or step <= at:
            final[key] = (value, source)
    wants, problems, found = [], [], []
    for key, value in (p.get("values") or {}).items():
        wants.append(f"{key} = {value}")
        got = final.get(key, (None, None))[0]
        found.append(f"{key} = {got}")
        if got != value:
            problems.append(key)
    for key, source in (p.get("sources") or {}).items():
        wants.append(f"{key} from {source}")
        got = str(final.get(key, (None, None))[1])
        found.append(f"{key} from {got}")
        if not _source_matches(obs, got, source, who):
            problems.append(f"{key} source")
    if changed := p.get("changed_in_step"):
        step, keys = changed["step"], sorted(changed["keys"])
        moved = sorted(
            {key for s, _node, key, _v, _src in history if s == step and key in CONSTRAINT_SLOTS}
        )
        wants.append(f"step {step} changes only {', '.join(keys)}")
        found.append(f"step {step} changed {', '.join(moved) or 'nothing'}")
        if moved != keys:
            problems.append("changed set")
    lead = _for(p) + (f"{p['thread']} session, " if p.get("thread") else "")
    lead += f"after step {at}: " if at else ""
    return CheckResult("slots", lead + "; ".join(wants), lead + "; ".join(found), not problems)


def _source_matches(obs: Observation, got: str, wanted: str, user: str) -> bool:
    """``memory`` (any fact), ``memory:<key>`` (a fact for that key), ``booking:<alias>``,
    or an exact source such as ``user``."""
    if wanted == "memory":
        return got.startswith("memory:")
    if wanted.startswith("memory:") and got.startswith("memory:"):
        row = obs.store.memory.get(got.removeprefix("memory:"), user_id=user)
        return row is not None and row.key == wanted.removeprefix("memory:")
    if wanted.startswith("booking:"):
        return got == f"booking:{_ref(obs, wanted.removeprefix('booking:'))}"
    return got == wanted


@check("search")
def search(obs: Observation, p: dict) -> CheckResult:
    """What every search_providers call (in one step, if given) searched for."""
    step = p.get("step")
    found = [
        event.data["args"]
        for _n, _run, event in obs.tool_calls("search_providers", step, p.get("user"))
    ]
    wants = {key: p[key] for key in ("service", "area") if key in p}
    expected = _for(p) + "searched " + ", ".join(f"{k} = {v}" for k, v in wants.items())
    if step:
        expected = f"step {step} {expected}"
    passed = bool(found) and all(all(a.get(k) == v for k, v in wants.items()) for a in found)
    if days := p.get("days"):
        expected += f", {_day(days[0])}–{_day(days[1])}"
        passed = passed and all(
            [a["date_range"]["start"], a["date_range"]["end"]] == days for a in found
        )
    actual = (
        "; ".join(
            f"{a['service']} in {a['area']}, {_day(a['date_range']['start'])}–"
            f"{_day(a['date_range']['end'])}"
            for a in found
        )
        or "no search"
    )
    return CheckResult("search", expected, actual, passed)


# Memory and context


@check("memory")
def memory(obs: Observation, p: dict) -> CheckResult:
    """Stored facts at the end: forgotten keys, current values and their source, superseding."""
    wants, found, problems = [], [], []
    rows = obs.store.memory.list_for_user(_who(obs, p), include_inactive=True)
    for fact in p["facts"]:
        key, service = fact["key"], fact.get("service")
        mine = [r for r in rows if r.key == key and r.service_tag == service]
        if fact.get("status") == "forgotten":
            wants.append(f"{key} forgotten")
            erased = mine and all(r.status.value == "forgotten" and r.value is None for r in mine)
            found.append(
                f"{key}: {len(mine)} rows, " + ("all forgotten" if erased else "not all forgotten")
            )
            if not erased:
                problems.append(key)
            continue
        current = next((r for r in mine if r.status.value in ("active", "unconfirmed")), None)
        want = f"{key} = {fact['value']} ({fact.get('source', 'explicit')})"
        got = (
            f"{key} = {current.value} ({current.source.value} {current.status.value})"
            if current
            else f"{key}: none"
        )
        if (
            current is None
            or current.value != fact["value"]
            or current.source.value != fact.get("source", "explicit")
        ):
            problems.append(key)
        if (old := fact.get("supersedes")) is not None:
            want += f", superseding {old}"
            previous = [r for r in mine if r.value == old and r.status.value == "superseded"]
            if not (current and previous and previous[-1].superseded_by == current.fact_id):
                problems.append(f"{key} history")
            got += ", superseding " + (str(previous[-1].value) if previous else "nothing")
        if (n := fact.get("inferred_rows")) is not None:
            inferred = [r for r in mine if r.source.value == "inferred"]
            want += f", {n} inferred rows"
            got += f", {len(inferred)} inferred rows"
            if len(inferred) != n:
                problems.append(f"{key} inferred")
        wants.append(want)
        found.append(got)
    return CheckResult(
        "memory", _for(p) + "; ".join(wants), _for(p) + "; ".join(found), not problems
    )


@check("not_sent")
def not_sent(obs: Observation, p: dict) -> CheckResult:
    """None of these values in anything sent to the model after a step, or before one."""
    values, user = p["values"], p.get("user")
    mine = [c for c in obs.captured if user is None or c.user_id == user]
    if "before_step" in p:
        before = p["before_step"]
        requests = [c.request for c in mine if c.step < before]
        when = f"before step {before}"
    else:
        after = p["after_step"]
        requests = [c.request for c in mine if c.step > after]
        when = f"after step {after}" if after else "in any step"
    when = _for(p) + when
    leaks = sorted(
        {
            value
            for request in requests
            for message in request.messages
            for value in values
            if value in message.content
        }
    )
    expected = f"{when}: never sent {', '.join(repr(v) for v in values)}"
    actual = f"{len(requests)} requests checked; " + (
        f"sent {', '.join(repr(v) for v in leaks)}" if leaks else "none sent"
    )
    return CheckResult("not_sent", expected, actual, bool(requests) and not leaks)


# Outcomes the user sees, structurally


@check("phases")
def phases(obs: Observation, p: dict) -> CheckResult:
    """The workflow phase after each user turn (null skips a turn), and optionally how each turn
    ended (ok, stopped by a limit or failure, or error). With ``user`` (and ``session``), only
    that user's turns (on that session)."""
    user, session = p.get("user"), p.get("session")
    turns = [
        turn
        for turn in obs.turns
        if (user is None or turn.user_id == user)
        and (session is None or turn.session_id == session)
    ]

    def matches(wanted: list, got: list) -> bool:
        if len(got) != len(wanted):
            return False
        return all(w is None or w == g for w, g in zip(wanted, got, strict=True))

    got = [
        turn.result.phase if turn.result else "crashed" if turn.crashed else "error"
        for turn in turns
    ]
    wanted = p["phases"]
    scope = _for(p) + (f"session {session}: " if session else "")
    expected, actual = scope + "phases " + " → ".join(w or "*" for w in wanted), " → ".join(got)
    passed = matches(wanted, got)
    if outcomes := p.get("outcomes"):
        ended = [turn.result.outcome if turn.result else "-" for turn in turns]
        expected += "; turns end " + ", ".join(w or "*" for w in outcomes)
        actual += "; ended " + ", ".join(ended)
        passed = passed and matches(outcomes, ended)
    return CheckResult("phases", expected, actual, passed)


@check("outbox")
def outbox(obs: Observation, p: dict) -> CheckResult:
    """Proactive messages: how many of a kind, with options, on a thread of their own."""
    who = _who(obs, p)
    messages = [m for m in obs.store.outbox.since(who) if m.kind == p["kind"]]
    expected = _for(p) + f"{p['count']} {p['kind']} message(s)"
    passed = len(messages) == p["count"]
    if p.get("has_options"):
        expected += " with options"
        passed = passed and all(m.payload.get("options") for m in messages)
    if p.get("new_thread"):
        expected += ", each on a new session of theirs"
        passed = passed and all(
            str(m.payload.get("thread_id")).startswith(f"{who}:")
            and m.payload.get("thread_id") != obs.thread("main", who)
            for m in messages
        )
    actual = (
        "; ".join(
            f"{m.kind}: {len(m.payload.get('options') or [])} options on {m.payload.get('thread_id')}"
            for m in messages
        )
        or "none"
    )
    return CheckResult("outbox", expected, actual, passed)


@check("echoed")
def echoed(obs: Observation, p: dict) -> CheckResult:
    """A step's reply restates the details as code resolved them (the renderer's own text for the
    slots at that point, so a wrong date shows up as a mismatch, not as wording)."""
    step = p["step"]
    record = obs.steps[step - 1]
    resolved: dict[str, Any] = {}
    for s, _node, key, value, _source in _slot_history(obs, record.thread_id):
        if s <= step:
            resolved[key] = value
    shown = render.known_details({k: v for k, v in resolved.items() if v is not None})
    reply = record.result.reply if record.result else ""
    found = "restated" if shown in reply else f"not restated in: {reply[:100]}"
    return CheckResult("echoed", f"step {step} restates: {shown}", found, shown in reply)


@check("context")
def context(obs: Observation, p: dict) -> CheckResult:
    """The facts the model was given (by key), in every manifest or one step's: keys that must be
    there, keys that must never be, the only keys allowed, and whether the cap made it drop
    anything."""
    step, after, user = p.get("step"), p.get("after_step", 0), p.get("user")
    sent: Counter[str] = Counter()
    dropped = most = 0
    for _n, run, event in obs.events("context"):
        if "cap" not in event.data or (step is not None and run.step != step):
            continue
        if run.step <= after or (user is not None and run.user_id != user):
            continue
        record = obs.steps[run.step - 1]
        for fact_id in event.data.get("fact_ids", []):
            row = record.memory_before.get(fact_id) or record.memory_after.get(fact_id)
            sent[row["key"] if row else fact_id] += 1
        dropped += len(event.data.get("dropped", []))
        most = max(most, event.data.get("total_tokens", 0))
    wants, problems = [], []
    for key in p.get("includes", []):
        wants.append(f"{key} sent")
        if key not in sent:
            problems.append(f"{key} missing")
    for key in p.get("excludes", []):
        wants.append(f"{key} never sent")
        if key in sent:
            problems.append(f"{key} sent")
    if (only := p.get("only")) is not None:
        wants.append(f"only {', '.join(only)}")
        if extra := sorted(set(sent) - set(only)):
            problems.append(f"also {', '.join(extra)}")
    if p.get("trimmed"):
        wants.append("the cap trimmed something")
        if not dropped:
            problems.append("nothing trimmed")
    scope = _for(p) + (
        f"step {step} context" if step else f"context after step {after}" if after else "context"
    )
    actual = (
        f"keys sent: {', '.join(f'{k} ×{n}' for k, n in sorted(sent.items())) or 'none'};"
        f" {dropped} items dropped; largest {most} tokens"
    )
    if problems:
        actual += " [" + "; ".join(problems) + "]"
    return CheckResult("context", f"{scope}: {'; '.join(wants)}", actual, not problems)


@check("stops")
def stops(obs: Observation, p: dict) -> CheckResult:
    """Runs a node stopped cleanly, by cause: budget:<limit>, model, decider, tool, internal
    (optionally only one user's runs)."""
    user = p.get("user")
    wanted = {k: n for k, n in p.items() if k != "user"}
    found: Counter[str] = Counter()
    for _n, run, event in obs.events():
        if event.kind not in ("budget", "error") or not event.data.get("node"):
            continue
        if user is not None and run.user_id != user:
            continue
        kind = str(event.data.get("kind"))
        if kind == "budget":
            limit = re.search(r"(\w+) budget exhausted", str(event.data.get("error")))
            kind = f"budget:{limit.group(1) if limit else '?'}"
        found[kind] += 1
    expected = (
        _for(p) + "stopped: " + (", ".join(f"{k} ×{n}" for k, n in wanted.items()) or "never")
    )
    actual = ", ".join(f"{k} ×{n}" for k, n in sorted(found.items())) or "no stops"
    return CheckResult("stops", expected, actual, dict(found) == wanted)


@check("poller")
def poller(obs: Observation, p: dict) -> CheckResult:
    """What the provider-cancel poller decided for each event, and why."""
    decisions = [
        (event.data["decision"], event.data.get("reason"))
        for _n, _run, event in obs.events("state.transition")
        if event.data.get("node") == "provider_cancel_poller"
    ]
    counts = Counter(decision for decision, _reason in decisions)
    wanted = {k: p[k] for k in ("ignore", "notify") if k in p}
    expected = "poller: " + ", ".join(f"{k} ×{n}" for k, n in wanted.items())
    actual = "; ".join(f"{d}" + (f" ({r})" if r else "") for d, r in decisions) or "no events"
    return CheckResult("poller", expected, actual, all(counts[k] == n for k, n in wanted.items()))


@check("injections")
def injections(obs: Observation, p: dict) -> CheckResult:
    """Prompt injections flagged in the trace, by the field they were found in."""
    found = Counter(
        str(event.data.get("field", "review")) for _n, _run, event in obs.events("injection")
    )
    expected = "flagged: " + ", ".join(f"{field} ×{n}" for field, n in p.items())
    actual = ", ".join(f"{field} ×{n}" for field, n in sorted(found.items())) or "nothing"
    return CheckResult("injections", expected, actual, all(found[f] == n for f, n in p.items()))


@check("tool_errors")
def tool_errors(obs: Observation, p: dict) -> CheckResult:
    """Tool calls that failed with a code (default: the gateway's bad_response), in order."""
    code = p.get("code", "bad_response")
    failed = [
        event.data["tool"]
        for _n, _run, event in obs.tool_calls(user=p.get("user"))
        if (event.data.get("failure") or {}).get("code") == code
    ]
    expected = _for(p) + f"{code} from {', '.join(p['tools']) or 'nothing'}"
    return CheckResult("tool_errors", expected, ", ".join(failed) or "none", failed == p["tools"])


@check("api_isolation")
def api_isolation(obs: Observation, p: dict) -> CheckResult:
    """Ask the app, as one user, for everything it would show them: none of the other user's
    facts, outbox messages or traces comes back, the other's runs look like no run at all (404),
    and a request that doesn't say whose it is is refused (400)."""
    user, other = p["user"], p["other"]
    their_facts = {f.fact_id for f in obs.store.memory.list_for_user(other, include_inactive=True)}
    their_messages = {m.message_id for m in obs.store.outbox.since(other)}
    their_runs = [r.run_id for r in obs.runs if r.user_id == other]
    my_runs = [r.run_id for r in obs.runs if r.user_id == user]

    async def ask() -> tuple:
        transport = httpx.ASGITransport(app=obs.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval") as client:
            me = {"user_id": user}
            memory = (await client.get("/memory", params={**me, "include_inactive": True})).json()
            outbox = (await client.get("/outbox", params=me)).json()
            theirs = [(await client.get(f"/trace/{r}", params=me)).status_code for r in their_runs]
            mine = [(await client.get(f"/trace/{r}", params=me)).json() for r in my_runs]
            nobody = await client.post("/chat", json={"message": "hi", "session_id": "main"})
        return memory["facts"], outbox["messages"], theirs, mine, nobody.status_code

    facts, messages, theirs, mine, nobody = asyncio.run(ask())
    problems = []
    if leaked := [
        f["fact_id"] for f in facts if f["user_id"] != user or f["fact_id"] in their_facts
    ]:
        problems.append(f"facts {leaked}")
    if leaked := [
        m["message_id"]
        for m in messages
        if m["user_id"] != user or m["message_id"] in their_messages
    ]:
        problems.append(f"outbox messages {leaked}")
    if any(code != 404 for code in theirs):
        problems.append(f"{other}'s traces answered {sorted(set(theirs))}")
    if any(event["user_id"] != user for trace in mine for event in trace["events"]):
        problems.append("own traces hold another user's events")
    if nobody != 400:
        problems.append(f"a request without a user got {nobody}")
    expected = (
        f"as {user}: none of {other}'s facts, outbox or traces ({len(their_runs)} runs → 404);"
        " no user → 400"
    )
    actual = (
        f"{len(facts)} facts, {len(messages)} outbox messages and {len(mine)} traces, all {user}'s;"
        f" {other}'s runs → {', '.join(map(str, sorted(set(theirs)))) or 'none'}; no user → {nobody}"
    )
    if problems:
        actual += " [" + "; ".join(problems) + "]"
    return CheckResult("api_isolation", expected, actual, not problems)


@check("run_budget")
def run_budget(obs: Observation, p: dict) -> CheckResult:
    """What one user's run in one step had used of its budget when it ended: its own calls and
    its own (simulated) time, whatever other users' runs were doing at the same moment."""
    user, step, kind = p["user"], p["step"], p.get("kind", "chat")
    budgets = [
        run.end.data.get("budget") or {}
        for run in obs.runs
        if run.user_id == user and run.step == step and run.kind == kind and run.end
    ]
    wants, problems = [], []
    if len(budgets) != 1:
        problems.append(f"{len(budgets)} runs")
    budget = budgets[0] if budgets else {}
    for field_name, limit in (
        ("elapsed_s", p.get("max_elapsed_s")),
        ("tool_calls", p.get("max_tool_calls")),
    ):
        if limit is not None:
            wants.append(f"{field_name} ≤ {limit}")
            if budget.get(field_name, 0) > limit:
                problems.append(f"{field_name} {budget.get(field_name)}")
    if (exact := p.get("elapsed_s")) is not None:
        wants.append(f"elapsed_s = {exact}")
        if budget.get("elapsed_s") != exact:
            problems.append(f"elapsed_s {budget.get('elapsed_s')}")
    actual = (
        f"tool_calls {budget.get('tool_calls')}/{budget.get('max_tool_calls')},"
        f" elapsed {budget.get('elapsed_s')}/{budget.get('timeout_s')} s"
    )
    if problems:
        actual += " [" + "; ".join(problems) + "]"
    expected = f"{user}'s {kind} run in step {step}: " + ", ".join(wants)
    return CheckResult("run_budget", expected, actual, not problems)


@check("trace_event")
def trace_event(obs: Observation, p: dict) -> CheckResult:
    """How many state events of one name the trace holds (optionally one user's, with given
    data), e.g. ``already_done`` for a confirmation that arrived after the booking."""
    name, user, data = p["event"], p.get("user"), p.get("data", {})
    found = [
        event.data
        for _n, run, event in obs.events("state.transition")
        if event.data.get("event") == name
        and (user is None or run.user_id == user)
        and all(event.data.get(k) == v for k, v in data.items())
    ]
    detail = f" with {data}" if data else ""
    expected = _for(p) + f"{p['count']} {name} event(s){detail}"
    return CheckResult("trace_event", expected, f"{len(found)}", len(found) == p["count"])


def automatic(obs: Observation) -> list[CheckResult]:
    """Checks every scenario gets: each step ran, and the script was used exactly."""
    errors = [f"step {s.index}: {s.error}" for s in obs.steps if s.error]
    out = [CheckResult("steps", "every step runs", "; ".join(errors) or "ok", not errors)]
    left = ", ".join(f"{n} {what}" for what, n in obs.leftovers.items())
    out.append(
        CheckResult(
            "script", "every scripted reply used", f"unused: {left}" if left else "ok", not left
        )
    )
    return out
