"""Graph nodes, following docs/errand_agent_hld.png (the LangGraph flowchart).

The model proposes, code decides. Vega answers typed questions inside nodes (intent, clear yes,
review quality, injection flag); Qwen extracts slot patches, reads picks and memory commands, and
writes small talk. Filters, holds, approvals, side effects and every factual reply are code.

Each node runs inside an error boundary: a budget, model, tool or unexpected failure becomes a
clear reply and a saved state instead of an exception. Interrupts pass through.
"""

import logging
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta

from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from errand.budget import BudgetExceeded
from errand.contracts import (
    IST,
    Booking,
    BookingState,
    DateRange,
    Hold,
    ProviderSummary,
    Slot,
    ToolError,
    ToolErrorCode,
)
from errand.decide import DecisionError
from errand.decide.questions import APPROVAL, INJECTION, INTENT, REVIEW_QUALITY
from errand.gateway import (
    ApprovalRequired,
    Execution,
    booking_material,
    cancel_material,
    canonical_slot,
    idempotency_key_for,
    injection_hint,
    material_hash,
    reschedule_material,
    untrusted_block,
)
from errand.graph import render
from errand.graph.dates import slot_fits
from errand.graph.deps import Deps
from errand.graph.slots import SERVICE_HINTS, booking_day, ground_patch
from errand.graph.state import AgentState, Phase
from errand.llm import ModelError
from errand.llm.prompts import MEMORY_SYSTEM, PICK_SYSTEM, SMALLTALK_SYSTEM, slot_patch_system
from errand.llm.schemas import MemoryCommand, PickProposal, ReplyText, SlotPatch
from errand.memory import KEY_LABELS, SLOT_FOR_KEY, display_value
from errand.mocks.catalog import normalize_area
from errand.store import Action, ApprovalStatus, FactStatus, LedgerStatus
from errand.trace import EventKind

log = logging.getLogger(__name__)

CONTINUE = re.compile(r"\s*(continue|go on|carry on|resume|try again|retry|keep going)\W*", re.I)
NEXT = re.compile(r"\s*(?:ok(?:ay)?,?\s*)?(?:next|the next one|(?:do|start) the next one)\W*", re.I)
# Words that turn a "yes" into something less than a clear yes: doubt, and shrugs (live Vega
# read "sure, whatever" as yes at 0.877, docs/bugs.md #12).
HEDGE = re.compile(
    r"\?|\b(not|don'?t|wait|maybe|hmm|unless|hold on|actually|whatever|i guess|guess so"
    r"|i suppose|suppose so|if you say so|if you must|i think so|probably|kind of|sort of)\b",
    re.I,
)
# Claims about approval, or orders aimed at the agent, are never the user's yes to a summary.
NOT_A_YES = re.compile(
    r"\balready (?:approved|confirmed|said yes)|\bapprov\w*\s*[=:]|\bsystem\s*:|\bignore (?:your|all"
    r"|the|previous|my) (?:rules|instructions)|\bwithout (?:asking|confirm)|\bskip (?:the )?confirm"
    r"|\bpre-?approved\b|\badmin mode\b|\boverride\b",
    re.I,
)
# Where a message joins two errands: "..., and cancel ...", "... then move ...", "...; book ...".
SPLIT = re.compile(
    r"(?:\s*;\s*|,?\s+(?:and|then|also|and then|and also)\s+)(?=(?:please\s+)?(?:cancel|move"
    r"|reschedule|book|find|get me|search for)\b)",
    re.I,
)
CHANGE = re.compile(
    r"\b(no|but|instead|change|cheaper|later|earlier|another|different|make it|rather)\b", re.I
)
PICK = re.compile(r"\s*(?:option\s*|number\s*|#)?([1-9])[.!]?\s*", re.I)
QUALITY_SCORE = {"positive": 1.0, "mixed": 0.5, "negative": 0.0}

Node = Callable[[AgentState, Deps], dict]


def node(name: str) -> Callable[[Node], Callable[[AgentState, Runtime[Deps]], dict]]:
    """Error boundary and state-transition trace around a node body."""

    def wrap(body: Node) -> Callable[[AgentState, Runtime[Deps]], dict]:
        # Not functools.wraps: it sets __wrapped__, and LangGraph reads the wrapped body's
        # (state, deps) signature through it and then never passes `runtime` (docs/bugs.md #2).
        def run(state: AgentState, runtime: Runtime[Deps]) -> dict:
            deps = runtime.context
            try:
                update = body(state, deps)
                if name in RESUMABLE:
                    update.setdefault("resume", None)  # handled: nothing left to resume
            except GraphBubbleUp:
                raise
            except BudgetExceeded as exc:
                update = _stopped(deps, state, name, "budget", exc)
            except ModelError as exc:
                update = _stopped(deps, state, name, "model", exc)
            except DecisionError as exc:
                update = _stopped(deps, state, name, "decider", exc)
            except (ToolError, ApprovalRequired) as exc:
                update = _stopped(deps, state, name, "tool", exc)
            except Exception as exc:
                log.exception("node %s failed", name)
                update = _stopped(deps, state, name, "internal", exc)
            extra: dict = {}
            if "slots" in update and (changed := _slot_changes(state, update)):
                extra["slots_changed"] = changed  # what changed, and where each value came from
            if pending := update.get("pending"):
                extra["asked"] = {k: pending.get(k) for k in ("kind", "missing", "slot")}
            deps.tracer.emit(
                EventKind.STATE,
                node=name,
                from_phase=state.phase,
                to_phase=update.get("phase", state.phase),
                goto=update.get("goto"),
                **extra,
            )
            return update

        run.__name__ = name
        run.__doc__ = body.__doc__
        return run

    return wrap


# Nodes that read the user's message: a run stopped in one of them is resumed there, with that
# message, by "continue". A stop anywhere later resumes from check_slots or find_booking.
RESUMABLE = {"extract_slots", "find_booking"}


def _stopped(deps: Deps, state: AgentState, name: str, kind: str, exc: BaseException) -> dict:
    nothing = "Nothing was booked, cancelled or changed."
    entry = (
        deps.store.ledger.get(
            idempotency_key_for(state.approval["approval_id"]), user_id=state.user_id
        )
        if state.approval
        else None
    )
    if entry and entry.status in (LedgerStatus.IN_FLIGHT, LedgerStatus.UNKNOWN):
        # A side effect may have happened: never say it didn't (docs/bugs.md #15).
        nothing = (
            "I couldn't confirm yet whether it went through; I'm checking and will update you here."
        )
    messages = {
        "budget": (
            "I reached my limit for one message before finishing, so I stopped there."
            f' {nothing} Say "continue" and I\'ll pick up where I left off.'
        ),
        "model": (
            "I couldn't reach my language model just now, so I stopped before doing anything."
            f" {nothing} Please try again in a moment."
        ),
        "decider": (
            "My decision model isn't working just now, so I stopped before doing anything."
            f" {nothing} Please try again in a moment."
        ),
        "tool": f"A provider system failed unexpectedly, so I stopped. {nothing} Please try again.",
        "internal": (
            f"Something went wrong on my side, so I stopped. {nothing} The details are in trace"
            f" {state.run_id}."
        ),
    }
    deps.tracer.emit(
        EventKind.BUDGET if kind == "budget" else EventKind.ERROR,
        node=name,
        kind=kind,
        error=exc,
        budget=deps.budget.snapshot(),
    )
    stop = {"kind": kind, "node": name, "error": str(exc)}
    update = {"stop": stop, "reply": messages[kind], "goto": "finish"}
    if name in RESUMABLE:
        # What "continue" needs to pick up exactly here, with this message (docs/bugs.md #16).
        update["resume"] = {
            "node": name,
            "message": state.user_message,
            "message_id": state.message_id,
        }
    return update


def _slot_changes(state: AgentState, update: dict) -> dict:
    before, after = state.slots, update["slots"]
    sources = update.get("slot_sources", state.slot_sources)
    return {
        key: {"value": after.get(key), "source": sources.get(key)}
        for key in sorted({*before, *after})
        if before.get(key) != after.get(key)
    }


def _key(provider_id: str, start: str) -> str:
    return f"{provider_id}|{start}"


def _invalidate(state: AgentState, deps: Deps, reason: str) -> list[str]:
    """Void the task's live approval, if any, and trace each one by id."""
    if not state.task_id:
        return []
    voided = deps.store.approvals.invalidate(state.task_id, reason, user_id=state.user_id)
    for approval_id in voided:
        deps.tracer.emit(
            EventKind.APPROVAL, event="invalidated", approval_id=approval_id, reason=reason
        )
    return voided


def _picked(state: AgentState) -> dict:
    return next(o for o in state.options if o["option_id"] == state.pick)


def _hold_for(state: AgentState, option: dict) -> Hold | None:
    """The hold in state, if it is for this option's provider and slot (it may have expired)."""
    if not state.hold:
        return None
    hold = Hold.model_validate(state.hold)
    wanted = canonical_slot(Slot.model_validate(option["slot"]))
    if hold.provider_id == option["provider_id"] and canonical_slot(hold.slot) == wanted:
        return hold
    return None


def _new_task(intent: str) -> dict:
    return {
        "task_id": None,
        "intent": intent,
        "phase": Phase.IDLE,
        "slots": {},
        "slot_sources": {},
        "asked": [],
        "pending": None,
        "candidates": [],
        "options": [],
        "excluded": [],
        "excluded_providers": [],
        "pick": None,
        "keep_provider": None,
        "hold": None,
        "approval": None,
        "target": None,
        "new_slot": None,
        "execution": None,
        "resume": None,
    }


def _abandon(state: AgentState, deps: Deps, reason: str) -> None:
    if state.task_id:
        _invalidate(state, deps, reason)
        deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.ABANDONED)


def _ensure_task(state: AgentState, deps: Deps, intent: str, phase: str) -> str:
    if state.task_id:
        return state.task_id
    return deps.store.tasks.create(
        user_id=state.user_id, thread_id=state.thread_id, intent=Action(intent), phase=phase
    ).task_id


# Entry


@node("load_context")
def load_context(state: AgentState, deps: Deps) -> dict:
    """Recover from interruptions, then note which stored facts are in play (SQL filter)."""
    update: dict = {
        "reply": "",
        "reply_private": False,
        "notice": "",
        "stop": None,
        "goto": "route_intent",
        "notes": [],
    }
    if state.phase in (Phase.EXECUTING, Phase.UNCONFIRMED) and state.approval:
        # The ledger, not the graph, knows what really happened to the last side effect.
        key = idempotency_key_for(state.approval["approval_id"])
        if deps.store.ledger.get(key, user_id=state.user_id) is None:
            _abandon(state, deps, "interrupted before anything was sent to the provider")
            update |= {
                "phase": Phase.PRESENTING if state.options else Phase.IDLE,
                "approval": None,
                "notice": "My last attempt stopped before anything was sent to the provider, so"
                " nothing was booked.",
            }
        else:
            settled = deps.gateway.verify(key)
            update |= _settled(state, deps, settled, as_notice=True)
    facts = deps.memory.relevant(
        state.user_id,
        purpose=state.intent or "smalltalk",
        service=state.slots.get("service"),
        limit=deps.settings.max_facts_in_context,
    )
    deps.tracer.emit(
        EventKind.CONTEXT,
        purpose="load_context",
        fact_ids=[fact.fact_id for fact in facts],
        stored_by_status=deps.store.memory.status_counts(state.user_id),
    )
    return update


@node("route_intent")
def route_intent(state: AgentState, deps: Deps) -> dict:
    if state.proactive:
        task_id = _ensure_task(state, deps, "book", Phase.COLLECTING)
        # In progress from the start, so a stopped follow-up can be resumed with "continue".
        return {"task_id": task_id, "phase": Phase.COLLECTING, "goto": "search"}
    text = state.user_message
    pending = state.pending or {}
    active = state.task_id is not None and state.phase not in Phase.FINISHED
    spent = idempotency_key_for((state.confirm or {}).get("approval_id", ""))
    if not active and state.execution and spent == state.execution.get("idempotency_key"):
        # A second Confirm for an approval already spent (another tab, a double click): say where
        # that side effect stands now, from the ledger and the provider. Never a second attempt.
        return {"reply": _already(state, deps, spent), "goto": "finish"}
    if active and CONTINUE.fullmatch(text):
        if resume := state.resume:
            # The last run stopped while handling a message: take that message up again.
            return {
                "user_message": resume["message"],
                "message_id": resume.get("message_id"),
                "resume": None,
                "goto": resume["node"],
            }
        return {
            "goto": "find_booking" if state.intent in ("cancel", "reschedule") else "check_slots"
        }
    if not active and state.queued and NEXT.fullmatch(text):
        item, *rest = state.queued
        deps.tracer.emit(EventKind.STATE, node="route_intent", event="dequeued", item=item)
        return {
            **_new_task(item["intent"]),
            "user_message": item["text"],
            "queued": rest,
            "queue_reminded": False,
            "goto": "extract_slots" if item["intent"] == "book" else "find_booking",
        }
    if pending.get("kind") == "confirm_fact":
        return {"goto": "confirm_fact"}

    if not active and (errands := _errands(text, deps)):
        # One errand becomes the task; the others wait their turn, and the user is told.
        first, *rest = errands
        deps.tracer.emit(EventKind.STATE, node="route_intent", event="queued", queued=rest)
        later = "; ".join(f'"{item["text"]}"' for item in rest)
        return {
            **_new_task(first["intent"]),
            "user_message": first["text"],
            "queued": [*state.queued, *rest],
            "queue_reminded": False,
            "notes": [f"You also asked: {later}. I'll do that next, once this is done."],
            "goto": "extract_slots" if first["intent"] == "book" else "find_booking",
        }

    decision = deps.decide(text, INTENT)
    choice = decision.choice if decision.is_confident(deps.threshold) else None
    if choice is None and decision.choice == "book" and _plainly_a_booking(text):
        # Routing to "book" only starts asking for details; nothing is held or booked without a
        # later yes. So a lean towards booking is enough when the message names a service and
        # nothing else (docs/bugs.md #24).
        deps.tracer.emit(
            EventKind.STATE,
            node="route_intent",
            event="intent_backstop",
            choice="book",
            confidence=decision.confidence,
            service=_service_named(text),
        )
        choice = "book"
    switching = choice in ("reschedule", "cancel", "memory_command") and choice != state.intent

    # While waiting on the user, a reply is an answer unless it clearly asks for something else.
    if active and not switching:
        if state.phase == Phase.AWAITING_APPROVAL:
            return {"goto": "approval_check"}
        if state.phase == Phase.PRESENTING:
            return {"goto": "parse_pick"}
        if state.phase in (Phase.COLLECTING, Phase.CHOOSING_BOOKING) and choice != "smalltalk":
            return {
                "goto": "find_booking"
                if state.intent in ("cancel", "reschedule")
                else "extract_slots"
            }

    if choice is None:
        return {"goto": "clarify"}
    if choice == "memory_command":
        return {"goto": "memory_ops"}
    if choice == "smalltalk":
        return {"goto": "smalltalk"}
    if choice == "book" and active and state.intent == "book":
        return {"goto": "extract_slots"}
    if active:
        _abandon(state, deps, f"the user moved on to a new {choice} request")
    return {**_new_task(choice), "goto": "extract_slots" if choice == "book" else "find_booking"}


def _already(state: AgentState, deps: Deps, key: str) -> str:
    entry = deps.store.ledger.get(key, user_id=state.user_id)
    intent = state.intent or (entry.op.value if entry else "book")
    status = entry.status if entry else None
    deps.tracer.emit(
        EventKind.STATE,
        node="route_intent",
        event="already_done",
        idempotency_key=key,
        status=status,
    )
    if status is LedgerStatus.CONFIRMED:
        booking_id = (entry.result or {}).get("booking", {}).get("booking_id")
        current = next(
            (b for b in deps.gateway.list_bookings().bookings if b.booking_id == booking_id), None
        )
        if current is not None:
            return render.already(intent, current)
    if status in (LedgerStatus.IN_FLIGHT, LedgerStatus.UNKNOWN):
        return render.unconfirmed(intent)
    return "That one didn't go through, so nothing was changed. Tell me if you'd like to try again."


def _errands(text: str, deps: Deps) -> list[dict]:
    """The errands in a message such as "book X and cancel Y", each with its own intent; empty
    when the message is one request. Two bookings count as two errands only if they name
    different services ("... and book it for tomorrow" is still the same one)."""
    clauses = [c.strip(" ,.;") for c in SPLIT.split(text)]
    clauses = [c for c in clauses if c]
    if len(clauses) < 2:
        return []
    found = []
    for clause in clauses:
        decision = deps.decide(clause, INTENT)
        if decision.is_confident(deps.threshold) and decision.choice in (
            "book",
            "cancel",
            "reschedule",
        ):
            found.append({"intent": decision.choice, "text": clause})
    if len(found) < 2:
        return []
    services = [_service_named(item["text"]) for item in found if item["intent"] == "book"]
    bookings_differ = len(services) > 1 and None not in services and len(set(services)) > 1
    if len({item["intent"] for item in found}) == 1 and not bookings_differ:
        return []
    return found


def _service_named(text: str) -> str | None:
    lowered = text.lower()
    return next((s for s, words in SERVICE_HINTS.items() if re.search(words, lowered)), None)


_OTHER_ERRANDS = re.compile(
    r"\b(cancel|call off|resched|move|postpone|shift|forget|remember|what do you know)\w*", re.I
)


def _plainly_a_booking(text: str) -> bool:
    """Names a service, and no cancel, reschedule or memory words."""
    return _service_named(text) is not None and not _OTHER_ERRANDS.search(text)


# Book path


@node("extract_slots")
def extract_slots(state: AgentState, deps: Deps) -> dict:
    patch = deps.ask_model(
        "slot_patch", SlotPatch, state=state, system=slot_patch_system(deps.today())
    )
    accepted, ignored = ground_patch(patch, state.user_message, state.slots, deps.today())
    changed = {k: v for k, v in accepted.items() if state.slots.get(k) != v}
    deps.tracer.emit(
        EventKind.STATE, node="extract_slots", event="slots", changed=changed, ignored=ignored
    )
    slots = {**state.slots, **accepted}
    sources = {**state.slot_sources, **{k: "user" for k in changed}}
    notes = list(state.notes)
    if "area" in ignored and "not an area I cover" in ignored["area"]:
        notes.append(f"I don't cover {ignored['area'].split(':')[0]} yet.")
    update: dict = {
        "slots": slots,
        "slot_sources": sources,
        "notes": notes,
        "pending": None,
        "task_id": _ensure_task(state, deps, "book", Phase.COLLECTING),
        "intent": "book",
        "phase": Phase.COLLECTING,  # in progress, so "continue" can resume it
        "goto": "check_slots",
    }
    material_change = {k for k in changed if k not in ("problem", "budget_given")}
    if not material_change:
        update["keep_provider"] = None  # a plain "no" changes nothing: nothing to propose again
    if material_change and (state.approval or state.hold or state.options):
        # Any change to the details voids the proposal built on the old ones.
        _invalidate(state, deps, f"details changed: {', '.join(sorted(material_change))}")
        update |= {
            "approval": None,
            "hold": None,
            "options": [],
            "candidates": [],
            "pick": None,
            "excluded": [],
        }
    if "service" in changed and state.slots.get("service"):
        # A provider picked for the old service says nothing about the new one.
        for stale in ("provider_id", "provider_label", "same_provider"):
            if stale not in accepted:
                slots.pop(stale, None)
    return update


@node("check_slots")
def check_slots(state: AgentState, deps: Deps) -> dict:
    """slots missing? Fill from active memory; confirm inferred memory before using it."""
    slots, sources = dict(state.slots), dict(state.slot_sources)
    notes, asked = list(state.notes), list(state.asked)
    service = slots.get("service")

    for slot in ("area", "budget_max_inr", "earliest_time", "time_window"):
        if slots.get(slot) is not None or (slot == "budget_max_inr" and slots.get("budget_given")):
            continue
        fact = deps.memory.for_slot(state.user_id, slot, service=service)
        if fact is None:
            continue
        if fact.status == FactStatus.ACTIVE:
            slots[slot] = fact.value
            sources[slot] = f"memory:{fact.fact_id}"
            if slot == "budget_max_inr":
                slots["budget_given"] = True
                sources["budget_given"] = sources[slot]
            notes.append(f"Using your saved {KEY_LABELS[fact.key]}: {display_value(fact.value)}.")
        elif slot in ("area", "budget_max_inr") and slot not in asked:
            # A guess is never used silently: ask a yes/no question first.
            label = KEY_LABELS[fact.key]
            question = (
                f"Should I use {display_value(fact.value)} as your {label}? I guessed it from"
                f" {fact.evidence or 'your history'}. (yes/no)"
            )
            return {
                "slots": slots,
                "slot_sources": sources,
                "pending": {
                    "kind": "confirm_fact",
                    "fact_id": fact.fact_id,
                    "slot": slot,
                    "question": question,
                },
                "phase": Phase.CONFIRMING_FACT,
                "asked": [*asked, slot],
                "reply": question,
                "goto": "finish",
            }

    if slots.get("same_provider") and not slots.get("provider_id") and service:
        past = [
            b
            for b in deps.gateway.list_bookings().bookings
            if b.service == service and b.status in (BookingState.COMPLETED, BookingState.CONFIRMED)
        ]
        if past:
            last = max(past, key=lambda b: b.slot.start)
            slots |= {"provider_id": last.provider_id, "provider_label": last.provider_name}
            sources["provider_id"] = sources["provider_label"] = f"booking:{last.booking_id}"
            if not slots.get("area"):
                slots["area"] = deps.gateway.get_provider(last.provider_id).area
                sources["area"] = f"booking:{last.booking_id}"
            notes.append(
                f"Looking for {last.provider_name} again, as on {render.day(last.slot.start)}."
            )
        else:
            slots["same_provider"] = False
            notes.append(
                f"I couldn't find an earlier {render.service_name(service)} booking, so I'll search everyone."
            )

    missing = [s for s in ("service", "area", "date_from") if not slots.get(s)]
    if missing and "budget" not in asked and not slots.get("budget_given"):
        missing.append("budget")  # asked once, alongside the other missing details
    if missing:
        return {
            "slots": slots,
            "slot_sources": sources,
            "notes": [],
            "pending": {"kind": "slots", "missing": missing},
            "phase": Phase.COLLECTING,
            "asked": sorted({*asked, *missing}),
            "reply": "\n".join([*notes, render.ask_missing(missing, slots)]),
            "goto": "finish",
        }
    return {
        "slots": slots,
        "slot_sources": sources,
        "notes": notes,
        "pending": None,
        "goto": "search",
    }


@node("confirm_fact")
def confirm_fact(state: AgentState, deps: Deps) -> dict:
    pending = state.pending or {}
    verdict = _yes_no(state.user_message, deps)
    if verdict == "unsure":
        return {
            "reply": f"Sorry, I need a yes or a no. {pending.get('question', '')}",
            "goto": "finish",
        }
    if verdict == "yes":
        deps.memory.confirm(state.user_id, pending["fact_id"])
        return {"pending": None, "goto": "check_slots"}
    deps.memory.reject(state.user_id, pending["fact_id"])
    # "No, Indiranagar" carries the right value: read it from the message.
    return {"pending": None, "goto": "extract_slots"}


@node("search")
def search(state: AgentState, deps: Deps) -> dict:
    slots = state.slots
    result = deps.gateway.search_providers(
        slots["service"],
        slots["area"],
        DateRange(
            start=date.fromisoformat(slots["date_from"]), end=date.fromisoformat(slots["date_to"])
        ),
    )
    return {
        "candidates": [p.model_dump(mode="json") for p in result.providers],
        "proactive": None,
        "goto": "rank_options",
    }


@node("rank_options")
def rank_options(state: AgentState, deps: Deps) -> dict:
    """Hard filters in code, then Vega rates the reviews of the best few providers. A provider
    whose own listing (name, attributes, description) carries instructions is left out."""
    slots = state.slots
    excluded, excluded_providers = set(state.excluded), set(state.excluded_providers)
    notes = list(state.notes)
    left_out = 0
    eligible: list[tuple[ProviderSummary, Slot]] = []
    for raw in state.candidates:
        provider = ProviderSummary.model_validate(raw)
        if provider.provider_id in excluded_providers:
            continue
        listing = {
            "name": provider.name,
            "attributes": " ".join(str(v) for v in provider.attributes.values()),
        }
        if _listing_flags(provider.provider_id, listing, deps):
            excluded_providers.add(provider.provider_id)
            left_out += 1
            continue
        if slots.get("provider_id") and provider.provider_id != slots["provider_id"]:
            continue
        if slots.get("budget_max_inr") and provider.price_inr > slots["budget_max_inr"]:
            continue
        fitting = [
            s
            for s in provider.slots
            if slot_fits(s.start, slots)
            and _key(provider.provider_id, canonical_slot(s)["start"]) not in excluded
        ]
        if fitting:
            eligible.append((provider, fitting[0]))

    scores = dict(state.review_scores)
    for provider, _ in sorted(eligible, key=lambda pair: -pair[0].rating)[
        : deps.settings.providers_to_review
    ]:
        if provider.provider_id not in scores:
            scores[provider.provider_id] = _rate_reviews(provider.provider_id, deps)
    for provider, _ in eligible:
        if scores.get(provider.provider_id, {}).get("listing_flagged"):
            excluded_providers.add(provider.provider_id)
            left_out += 1
    eligible = [pair for pair in eligible if pair[0].provider_id not in excluded_providers]
    if left_out:
        notes.append(
            f"I left out {left_out} provider{'s' if left_out > 1 else ''} whose listing contained"
            " instructions to me."
        )
    if not eligible:
        return {
            "options": [],
            "review_scores": scores,
            "excluded_providers": sorted(excluded_providers),
            "phase": Phase.COLLECTING,
            "pending": {"kind": "slots", "missing": []},
            "reply": render.no_options(slots, notes),
            "notes": [],
            "goto": "finish",
        }
    max_price = max(provider.price_inr for provider, _ in eligible) or 1

    def score(pair: tuple[ProviderSummary, Slot]) -> float:
        provider, _ = pair
        reviews = scores.get(provider.provider_id, {}).get("score", 0.5)
        return (
            0.45 * provider.rating / 5 + 0.35 * reviews + 0.2 * (1 - provider.price_inr / max_price)
        )

    options = []
    for n, (provider, slot) in enumerate(sorted(eligible, key=score, reverse=True)[:3], start=1):
        options.append(
            {
                "option_id": n,
                "provider_id": provider.provider_id,
                "name": provider.name,
                "rating": provider.rating,
                "price_inr": provider.price_inr,
                "slot": slot.model_dump(mode="json"),
                "review_score": round(scores.get(provider.provider_id, {}).get("score", 0.5), 2),
                "flagged_reviews": scores.get(provider.provider_id, {}).get("flagged", 0),
            }
        )
    kept = _kept_option(state, options, eligible, scores)
    if kept is None and state.keep_provider and options:
        name = next(
            (c["name"] for c in state.candidates if c["provider_id"] == state.keep_provider),
            "The provider you picked",
        )
        notes.append(f"{name} has no slot that fits the new details, so here are the others.")
    if kept is not None:
        options = [kept, *[o for o in options if o["provider_id"] != kept["provider_id"]]][:3]
        for n, option in enumerate(options, start=1):
            option["option_id"] = n
    deps.tracer.emit(
        EventKind.STATE,
        node="rank_options",
        event="offered",
        options=[
            {k: o[k] for k in ("option_id", "provider_id", "price_inr", "slot")} for o in options
        ],
    )
    if kept is not None:
        # The user changed a detail of a summary: same provider, new details, new approval.
        return {
            "options": options,
            "review_scores": scores,
            "excluded_providers": sorted(excluded_providers),
            "notes": notes,
            "pick": 1,
            "keep_provider": None,
            "pending": None,
            "goto": "build_summary",
        }
    return {
        "keep_provider": None,
        "options": options,
        "review_scores": scores,
        "excluded_providers": sorted(excluded_providers),
        "notes": notes,
        "phase": Phase.PRESENTING,
        "pending": None,
        "goto": "present_options",
    }


def _kept_option(
    state: AgentState, options: list[dict], eligible: list, scores: dict
) -> dict | None:
    """The kept provider's best slot under the new details, if it still fits them (and the user
    didn't ask for another provider); None otherwise, and the user gets the usual list."""
    kept = state.keep_provider
    if not kept or state.slots.get("provider_id") not in (None, kept):
        return None
    for option in options:
        if option["provider_id"] == kept:
            return dict(option)
    provider, slot = next(((p, s) for p, s in eligible if p.provider_id == kept), (None, None))
    if provider is None:
        return None
    return {
        "option_id": 1,
        "provider_id": provider.provider_id,
        "name": provider.name,
        "rating": provider.rating,
        "price_inr": provider.price_inr,
        "slot": slot.model_dump(mode="json"),
        "review_score": round(scores.get(provider.provider_id, {}).get("score", 0.5), 2),
        "flagged_reviews": scores.get(provider.provider_id, {}).get("flagged", 0),
    }


def _listing_flags(provider_id: str, fields: dict[str, str], deps: Deps) -> list[str]:
    """Provider-written fields that carry instructions to the agent. Unlike a customer's review,
    these come from the provider itself, so the provider is left out (docs/bugs.md #19)."""
    flagged = []
    for field, text in fields.items():
        if hint := injection_hint(text):
            flagged.append(field)
            deps.tracer.emit(
                EventKind.INJECTION,
                provider_id=provider_id,
                field=field,
                heuristic=hint,
                action="provider left out: its own listing gives instructions",
            )
    return flagged


def _rate_reviews(provider_id: str, deps: Deps) -> dict:
    """Review quality from Vega, with injection attempts flagged and left out of the score.

    Reviews only ever reach Vega inside an untrusted block, and a flag only changes what is
    logged and scored, never what the agent does. A provider whose details can't be read is
    ranked without reviews rather than failing the whole search.
    """
    try:
        details = deps.gateway.get_provider(provider_id)
    except ToolError as exc:
        deps.tracer.emit(
            EventKind.ERROR,
            where="rank_options",
            handled="ranked without its reviews",
            provider_id=provider_id,
            error=exc,
        )
        return {"score": 0.5, "flagged": 0, "unreadable": True}
    if _listing_flags(provider_id, {"about": details.about}, deps):
        return {"score": 0.0, "flagged": 0, "listing_flagged": True}
    ratings, flagged = [], 0
    for review in details.reviews:
        block = untrusted_block(
            review.text, source=f"get_provider:{provider_id}:{review.review_id}"
        )
        answers = deps.decide_many(block, [REVIEW_QUALITY, INJECTION])
        hint = injection_hint(review.text)
        vega_flag = (
            answers["injection"].choice == "injection" and answers["injection"].confidence >= 0.5
        )
        if hint or vega_flag:
            flagged += 1
            deps.tracer.emit(
                EventKind.INJECTION,
                provider_id=provider_id,
                field="review",
                review_id=review.review_id,
                vega=answers["injection"].choice,
                vega_confidence=round(answers["injection"].confidence, 3),
                heuristic=hint,
                action="ignored for ranking; never treated as an instruction",
            )
            continue
        ratings.append(QUALITY_SCORE[answers["review_quality"].choice])
    return {"score": sum(ratings) / len(ratings) if ratings else 0.5, "flagged": flagged}


@node("present_options")
def present_options(state: AgentState, deps: Deps) -> dict:
    text = render.options(state.options, state.slots, state.notes)
    answer = interrupt(
        {
            "kind": "options",
            "reply": text,
            "options": [
                {"option_id": o["option_id"], "label": f"{o['option_id']}. {o['name']}"}
                for o in state.options
            ],
        }
    )
    return {**_resumed(answer), "notes": [], "goto": "parse_pick"}


def _resumed(answer: dict) -> dict:
    return {
        "user_message": answer["message"],
        "message_id": answer.get("message_id"),
        "confirm": answer.get("confirm"),
        "run_id": answer.get("run_id", ""),
        "reply": "",
        "reply_private": False,
        "notice": "",
        "stop": None,
    }


@node("parse_pick")
def parse_pick(state: AgentState, deps: Deps) -> dict:
    ids = {o["option_id"] for o in state.options}
    quick = PICK.fullmatch(state.user_message)
    if quick:
        action, option_id = "propose_booking", int(quick.group(1))
    else:
        listing = "\n".join(
            f"{o['option_id']}. {o['name']}, {render.when(Slot.model_validate(o['slot']))},"
            f" {render.money(o['price_inr'])}"
            for o in state.options
        )
        proposal = deps.ask_model(
            "pick",
            PickProposal,
            state=state,
            system=PICK_SYSTEM,
            data=untrusted_block(listing, source="options"),
            recent=False,
        )
        action, option_id = proposal.action, proposal.option_id
    if action == "propose_booking" and option_id in ids:
        # Approval comes first: nothing is held until the user has said yes to this option.
        return {"pick": option_id, "goto": "build_summary"}
    if action == "change_details":
        return {"goto": "extract_slots"}
    # Neither a pick nor a change: maybe a different request altogether.
    decision = deps.decide(state.user_message, INTENT)
    if decision.is_confident(deps.threshold) and decision.choice == "memory_command":
        return {"goto": "memory_ops"}
    if decision.is_confident(deps.threshold) and decision.choice in ("cancel", "reschedule"):
        _abandon(state, deps, f"the user moved on to a {decision.choice} request")
        return {**_new_task(decision.choice), "goto": "find_booking"}
    return {
        "notes": [f"Please pick one of the options (1–{max(ids)}) or tell me what to change."],
        "goto": "present_options",
    }


@node("hold_slot")
def hold_slot(state: AgentState, deps: Deps) -> dict:
    """Hold the approved slot (the gateway refuses without the granted approval), then compare
    the quote with the approved price: the same price goes on to book, a different one voids the
    approval and asks again, and one above the budget drops the option."""
    option = _picked(state)
    approval = state.approval or {}
    slot = Slot.model_validate(option["slot"])
    key = _key(option["provider_id"], canonical_slot(slot)["start"])
    hold = _hold_for(state, option)
    if hold is None or hold.expires_at - deps.clock.now() <= timedelta(seconds=60):
        try:
            hold = deps.gateway.hold_slot(
                option["provider_id"],
                slot,
                approval_id=approval.get("approval_id", ""),
                task_id=state.task_id,
            )
        except ToolError as exc:
            if exc.failure.code is not ToolErrorCode.SLOT_TAKEN:
                raise
            _invalidate(state, deps, "the slot was taken before it could be held")
            return {
                "excluded": [*state.excluded, key],
                "hold": None,
                "approval": None,
                "notes": [
                    "That slot was taken a moment ago, so nothing was booked. Here are the"
                    " remaining options."
                ],
                "goto": "rank_options",
            }
        except ApprovalRequired:
            return {
                "notes": ["I need you to confirm the details again before I go ahead."],
                "goto": "build_summary",
            }
    held = hold.model_dump(mode="json")
    budget = state.slots.get("budget_max_inr")
    if budget and hold.quoted_price_inr > budget:
        _invalidate(state, deps, "the quoted price is above the budget")
        return {
            # The provider, not just this slot: its listed price is known to be wrong, so
            # offering another of its slots would ask for a yes to a false price (bugs.md #14).
            "excluded_providers": [*state.excluded_providers, option["provider_id"]],
            "excluded": [*state.excluded, key],
            "hold": None,
            "approval": None,
            "notes": [
                f"{option['name']} quoted {render.money(hold.quoted_price_inr)} when I held the"
                f" slot, above your {render.money(budget)} budget, so I left it and booked nothing."
            ],
            "goto": "rank_options",
        }
    if material_hash(_current_material(state.model_copy(update={"hold": held}))) != approval.get(
        "material_hash"
    ):
        _invalidate(state, deps, "the quoted price differs from the approved one")
        return {
            "hold": held,
            "approval": None,
            "notes": [
                f"{option['name']} quoted {render.money(hold.quoted_price_inr)} when I held the"
                f" slot, not the {render.money(approval.get('price_inr', option['price_inr']))}"
                " you approved, so nothing is booked yet."
            ],
            "goto": "build_summary",
        }
    return {"hold": held, "goto": "execute"}


# Approval


def _current_material(state: AgentState) -> dict:
    """The material of the proposal in state, from tool data only: the picked option as listed
    by search_providers, or as quoted by its hold once there is one."""
    if state.intent == "book":
        option = _picked(state)
        hold = _hold_for(state, option)
        return booking_material(
            provider_id=option["provider_id"],
            slot=hold.slot if hold else Slot.model_validate(option["slot"]),
            price_inr=hold.quoted_price_inr if hold else option["price_inr"],
        )
    target = Booking.model_validate(state.target)
    if state.intent == "cancel":
        return cancel_material(target)
    return reschedule_material(target, Slot.model_validate(state.new_slot))


@node("build_summary")
def build_summary(state: AgentState, deps: Deps) -> dict:
    """Render the summary from tool data and open a pending approval bound to its hash."""
    if state.intent == "book":
        option = _picked(state)
        summary = render.booking_summary(
            option, _hold_for(state, option), state.slots, state.notes, now=deps.clock.now()
        )
    elif state.intent == "cancel":
        summary = render.cancel_summary(Booking.model_validate(state.target), state.notes)
    else:
        summary = render.reschedule_summary(
            Booking.model_validate(state.target), Slot.model_validate(state.new_slot), state.notes
        )
    material = _current_material(state)
    _invalidate(state, deps, "superseded by a new summary")
    approval = deps.store.approvals.request(
        state.task_id, Action(state.intent), material, user_id=state.user_id
    )
    deps.tracer.emit(
        EventKind.APPROVAL,
        event="requested",
        approval_id=approval.approval_id,
        material_hash=approval.material_hash,
        material=material,
    )
    deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.AWAITING_APPROVAL)
    return {
        "approval": {
            "approval_id": approval.approval_id,
            "material_hash": approval.material_hash,
            "action": state.intent,
            "price_inr": material["price_inr"],
            "summary": summary,
        },
        "phase": Phase.AWAITING_APPROVAL,
        "notes": [],
        "goto": "await_approval",
    }


@node("await_approval")
def await_approval(state: AgentState, deps: Deps) -> dict:
    approval = state.approval or {}
    answer = interrupt(
        {
            "kind": "approval",
            "reply": approval.get("reprompt", "") + approval["summary"],
            "approval": {
                "approval_id": approval["approval_id"],
                "material_hash": approval["material_hash"],
            },
        }
    )
    return {**_resumed(answer), "goto": "approval_check"}


def _yes_no(text: str, deps: Deps) -> str:
    """Vega's clear-yes check, with a code backstop that only ever moves away from yes: hedges,
    shrugs, claims of approval and orders to the agent are never a yes, and a change of details is
    a no even when Vega isn't sure, so the change is applied (docs/bugs.md #11, #12)."""
    decision = deps.decide(text, APPROVAL)
    verdict = decision.choice if decision.is_confident(deps.threshold) else "unsure"
    if NOT_A_YES.search(text) or PICK.fullmatch(text):
        # An option number at a summary ("1", after a change brought a summary instead of a
        # list) picks nothing and approves nothing: ask for a clear yes.
        verdict = "unsure"
    elif CHANGE.search(text):
        verdict = "no"
    elif verdict == "yes" and HEDGE.search(text):
        verdict = "unsure"
    return verdict


@node("approval_check")
def approval_check(state: AgentState, deps: Deps) -> dict:
    approval = state.approval or {}
    stored = deps.store.approvals.get(approval.get("approval_id", ""), user_id=state.user_id)
    if stored is None or stored.status is not ApprovalStatus.PENDING:
        return {
            "notes": ["That proposal is no longer open, so here it is again."],
            "goto": "build_summary",
        }
    if state.confirm and state.confirm.get("material_hash") != approval["material_hash"]:
        return {
            "approval": {**approval, "reprompt": "That confirmation was for an earlier proposal. "},
            "goto": "await_approval",
        }
    if material_hash(_current_material(state)) != approval["material_hash"]:
        _invalidate(state, deps, "the details changed after the summary")
        return {
            "notes": ["Something changed since my last summary, so please check it again."],
            "goto": "build_summary",
        }

    verdict = _yes_no(state.user_message, deps)
    if verdict == "yes":
        deps.store.approvals.grant(
            approval["approval_id"], approval["material_hash"], user_id=state.user_id
        )
        deps.tracer.emit(
            EventKind.APPROVAL,
            event="granted",
            approval_id=approval["approval_id"],
            material_hash=approval["material_hash"],
            by_message=state.user_message,
        )
        deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.EXECUTING)
        # A booking's slot is held only now, under the granted approval.
        return {
            "phase": Phase.EXECUTING,
            "goto": "hold_slot" if state.intent == "book" else "execute",
        }
    if verdict == "unsure":
        return {
            "approval": {**approval, "reprompt": "I need a clear yes or no before I go ahead. "},
            "goto": "await_approval",
        }

    _invalidate(state, deps, "the user declined")
    switch = deps.decide(state.user_message, INTENT)
    if (
        switch.is_confident(deps.threshold)
        and switch.choice in ("cancel", "reschedule")
        and switch.choice != state.intent
    ):
        _abandon(state, deps, f"the user moved on to a {switch.choice} request")
        return {**_new_task(switch.choice), "goto": "find_booking"}
    if state.intent == "cancel":
        return {
            "approval": None,
            "phase": Phase.DONE,
            "reply": "Okay, I won't cancel it.",
            "goto": "finish",
        }
    if state.intent == "reschedule":
        return {
            "approval": None,
            "new_slot": None,
            "pending": {"kind": "slots", "missing": ["date_from"]},
            "phase": Phase.COLLECTING,
            "reply": 'Okay, I won\'t move it. Tell me another day or time, or say "cancel" to call it off.',
            "goto": "finish",
        }
    # A "no" that changes something ("make it Thursday afternoon, keep everything else"): keep
    # the picked provider in mind, so the new details come back as a summary to confirm.
    kept = _picked(state)["provider_id"] if state.pick is not None and state.options else None
    return {"approval": None, "keep_provider": kept, "goto": "extract_slots"}


# Execute and verify


@node("execute")
def execute(state: AgentState, deps: Deps) -> dict:
    approval_id = state.approval["approval_id"]
    try:
        if state.intent == "book":
            result = deps.gateway.book(state.hold["hold_id"], approval_id=approval_id)
        elif state.intent == "cancel":
            result = deps.gateway.cancel(state.target["booking_id"], approval_id=approval_id)
        else:
            result = deps.gateway.reschedule(
                state.target["booking_id"],
                Slot.model_validate(state.new_slot),
                approval_id=approval_id,
            )
    except BudgetExceeded:
        # Out of time or calls with the row open: verify (or the reconciler) settles it.
        key = idempotency_key_for(approval_id)
        if deps.store.ledger.get(key, user_id=state.user_id) is None:
            raise
        result = Execution(
            idempotency_key=key, op=Action(state.intent), status=LedgerStatus.IN_FLIGHT
        )
    except ApprovalRequired:
        return {
            "notes": ["I need you to confirm the details again before I go ahead."],
            "goto": "build_summary",
        }
    return {"execution": result.model_dump(mode="json"), "goto": "verify"}


@node("verify")
def verify(state: AgentState, deps: Deps) -> dict:
    settled = deps.gateway.verify(state.execution["idempotency_key"])
    return _settled(state, deps, settled, as_notice=False)


def _settled(state: AgentState, deps: Deps, settled: Execution, *, as_notice: bool) -> dict:
    """The next step after a side effect, from its ledger status. Only confirmed says done."""
    intent = state.intent or settled.op.value
    base = {"execution": settled.model_dump(mode="json")}
    if settled.status is LedgerStatus.CONFIRMED:
        text = render.done(intent, settled.booking)
        deps.tracer.emit(
            EventKind.CLAIM,
            op=intent,
            idempotency_key=settled.idempotency_key,
            booking_id=settled.booking.booking_id,
            channel="notice" if as_notice else "reply",
        )
        deps.store.tasks.update(
            state.task_id,
            user_id=state.user_id,
            phase=Phase.DONE,
            booking_id=settled.booking.booking_id,
        )
        if intent == "book" and not as_notice:
            # Guesses for next time; they stay unconfirmed until the user says yes to them.
            deps.memory.infer(
                state.user_id,
                "technician",
                {"provider_id": settled.booking.provider_id, "name": settled.booking.provider_name},
                service=settled.booking.service,
                evidence=f"booking {settled.booking.booking_id}",
            )
            if state.slot_sources.get("area") == "user":
                deps.memory.infer(
                    state.user_id,
                    "preferred_area",
                    state.slots["area"],
                    evidence=f"booking {settled.booking.booking_id} in {state.slots['area']}",
                )
        out = {**base, "phase": Phase.DONE, "approval": None, "hold": None}
        return out | ({"notice": text} if as_notice else {"reply": text, "goto": "finish"})
    if settled.status in (LedgerStatus.UNKNOWN, LedgerStatus.IN_FLIGHT):
        deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.UNCONFIRMED)
        text = render.unconfirmed(intent)
        out = {**base, "phase": Phase.UNCONFIRMED}
        return out | ({"notice": text} if as_notice else {"reply": text, "goto": "finish"})

    code = settled.failure.code if settled.failure else None
    if as_notice:
        deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.FAILED)
        return {
            **base,
            "phase": Phase.FAILED,
            "approval": None,
            "notice": "Your earlier request didn't go through.",
        }
    if code is ToolErrorCode.SLOT_TAKEN and intent == "book":
        option = _picked(state)
        key = _key(
            option["provider_id"], canonical_slot(Slot.model_validate(option["slot"]))["start"]
        )
        return {
            **base,
            "excluded": [*state.excluded, key],
            "approval": None,
            "hold": None,
            "notes": [
                "Someone else booked that slot just before me. Nothing was booked; here are the other options."
            ],
            "goto": "rank_options",
        }
    if code is ToolErrorCode.HOLD_EXPIRED and intent == "book":
        # The approval was spent on the failed attempt. The expired hold stays in state for its
        # quoted price; a fresh yes holds the slot again.
        return {
            **base,
            "approval": None,
            "notes": [
                "The hold ran out before I could book, so nothing was booked. Please confirm again"
                " and I'll hold the slot afresh."
            ],
            "goto": "build_summary",
        }
    if code is ToolErrorCode.SLOT_CONFLICT and intent == "reschedule":
        start = canonical_slot(Slot.model_validate(state.new_slot))["start"]
        return {
            **base,
            "excluded": [*state.excluded, _key(state.target["provider_id"], start)],
            "approval": None,
            "new_slot": None,
            "notes": ["The provider can't take that slot after all, so I picked the next one."],
            "goto": "plan_reschedule",
        }
    deps.store.tasks.update(state.task_id, user_id=state.user_id, phase=Phase.FAILED)
    reason = settled.failure.message if settled.failure else "the provider didn't apply it"
    return {
        **base,
        "phase": Phase.FAILED,
        "approval": None,
        "reply": f"That didn't go through ({reason}). Nothing was changed.",
        "goto": "finish",
    }


# Reschedule and cancel


@node("find_booking")
def find_booking(state: AgentState, deps: Deps) -> dict:
    """list_bookings, then pick the booking the user means (in code)."""
    pending = state.pending or {}
    text = state.user_message
    task_id = _ensure_task(state, deps, state.intent, Phase.CHOOSING_BOOKING)
    if pending.get("kind") == "which_booking" and (quick := PICK.fullmatch(text)):
        choices = pending["choices"]
        n = int(quick.group(1))
        if 1 <= n <= len(choices):
            return _target_chosen(state, Booking.model_validate(choices[n - 1]), task_id)

    patch = deps.ask_model(
        "find_booking", SlotPatch, state=state, system=slot_patch_system(deps.today())
    )
    accepted, _ = ground_patch(patch, text, {}, deps.today())
    slots, sources = dict(state.slots), dict(state.slot_sources)
    for slot in ("date_from", "date_to", "time_window"):
        if slot in accepted:
            slots[slot], sources[slot] = accepted[slot], "user"
    if state.target:  # the booking is known; this message only gives the new time
        return {
            "slots": slots,
            "slot_sources": sources,
            "task_id": task_id,
            "pending": None,
            "goto": "plan_reschedule",
        }

    upcoming = [
        b
        for b in deps.gateway.list_bookings().bookings
        if b.status is BookingState.CONFIRMED and b.slot.start > deps.clock.now()
    ]
    service = accepted.get("service") or patch.service
    matches = [b for b in upcoming if not service or b.service == service]
    on = booking_day(patch, text, state.intent, deps.today())
    if state.intent == "cancel":
        for slot in ("date_from", "date_to"):  # a cancel's day names the booking, not a new time
            slots.pop(slot, None)
            sources.pop(slot, None)
    if on:
        within = [b for b in matches if on[0] <= b.slot.start.astimezone(IST).date() <= on[1]]
        matches = within or matches
    if not matches:
        what = f"{render.service_name(service)} booking" if service else "booking"
        listing = render.bookings_list(upcoming) if upcoming else "You have no upcoming bookings."
        return {
            "task_id": task_id,
            "phase": Phase.FAILED,
            "reply": f"I couldn't find an upcoming {what} to {state.intent}.\n{listing}",
            "goto": "finish",
        }
    if len(matches) > 1:
        return {
            "task_id": task_id,
            "slots": slots,
            "slot_sources": sources,
            "phase": Phase.CHOOSING_BOOKING,
            "pending": {
                "kind": "which_booking",
                "choices": [b.model_dump(mode="json") for b in matches],
            },
            "reply": f"Which booking do you mean?\n{render.bookings_list(matches)}\nReply with its number.",
            "goto": "finish",
        }
    return _target_chosen(
        state.model_copy(update={"slots": slots, "slot_sources": sources}), matches[0], task_id
    )


def _target_chosen(state: AgentState, booking: Booking, task_id: str) -> dict:
    update = {
        "task_id": task_id,
        "target": booking.model_dump(mode="json"),
        "slots": {**state.slots, "service": booking.service},
        "slot_sources": {**state.slot_sources, "service": f"booking:{booking.booking_id}"},
        "pending": None,
        "phase": Phase.CHOOSING_BOOKING,
    }
    return update | {"goto": "build_summary" if state.intent == "cancel" else "plan_reschedule"}


@node("plan_reschedule")
def plan_reschedule(state: AgentState, deps: Deps) -> dict:
    """Pick the new slot with the same provider, in code: the requested day, then the closest
    time to the original visit."""
    target = Booking.model_validate(state.target)
    slots = state.slots
    if not slots.get("date_from"):
        return {
            "pending": {"kind": "slots", "missing": ["date_from"]},
            "phase": Phase.COLLECTING,
            "reply": f"When should I move your {render.service_name(target.service)} visit with {target.provider_name} to?",
            "goto": "finish",
        }
    area = deps.gateway.get_provider(target.provider_id).area
    result = deps.gateway.search_providers(
        target.service,
        area,
        DateRange(
            start=date.fromisoformat(slots["date_from"]), end=date.fromisoformat(slots["date_to"])
        ),
    )
    preferences = {**slots}
    earliest = deps.memory.for_slot(state.user_id, "earliest_time", service=target.service)
    if earliest and earliest.status == FactStatus.ACTIVE and not preferences.get("earliest_time"):
        preferences["earliest_time"] = earliest.value
    excluded = set(state.excluded)
    original = target.slot.start.astimezone(IST)
    candidates = [
        s
        for p in result.providers
        if p.provider_id == target.provider_id
        for s in p.slots
        if slot_fits(s.start, preferences)
        and _key(target.provider_id, canonical_slot(s)["start"]) not in excluded
    ]
    if not candidates:
        return {
            "pending": {"kind": "slots", "missing": ["date_from"]},
            "phase": Phase.COLLECTING,
            "reply": "\n".join(
                [
                    *state.notes,
                    f"{target.provider_name} has no free slots then. Is there another day that works?",
                ]
            ),
            "notes": [],
            "goto": "finish",
        }

    def distance(slot: Slot) -> tuple[int, datetime]:
        local = slot.start.astimezone(IST)
        return abs((local.hour * 60 + local.minute) - (original.hour * 60 + original.minute)), local

    best = min(candidates, key=distance)
    return {"new_slot": best.model_dump(mode="json"), "pick": None, "goto": "build_summary"}


# Memory, small talk, clarify, finish


@node("memory_ops")
def memory_ops(state: AgentState, deps: Deps) -> dict:
    """Inspect, correct or forget stored facts. The model reads the request; code validates it."""
    facts = deps.memory.relevant(
        state.user_id, purpose="smalltalk", service=None, limit=deps.settings.max_facts_in_context
    )
    command = deps.ask_model(
        "memory_command",
        MemoryCommand,
        state=state,
        system=MEMORY_SYSTEM,
        facts=facts,
        recent=False,
    )
    if command.op == "inspect" or not command.key:
        # It lists guesses too: for the user's eyes only, never model context (docs/bugs.md #13).
        return {
            "reply": deps.memory.describe(state.user_id),
            "reply_private": True,
            "goto": "finish",
        }
    label = KEY_LABELS[command.key]
    scope = command.service if command.key in ("technician", "budget_max_inr") else None
    if command.op == "forget":
        rows = deps.memory.forget(state.user_id, command.key, service=scope)
        if not rows and scope:
            rows = deps.memory.forget(state.user_id, command.key, service=None)
        slots, sources = dict(state.slots), dict(state.slot_sources)
        slot = SLOT_FOR_KEY[command.key]
        if sources.get(slot, "").startswith("memory:"):
            slots.pop(slot, None)
            sources.pop(slot, None)
        reply = (
            f"Done. I've forgotten your {label}." if rows else f"I didn't have a {label} stored."
        )
        return {"slots": slots, "slot_sources": sources, "reply": reply, "goto": "finish"}

    value = _memory_value(command.key, command.value, state.user_message)
    if value is None:
        return {
            "reply": f"I couldn't tell what to save as your {label}. Could you say it again?",
            "goto": "finish",
        }
    fact = deps.memory.remember(
        state.user_id,
        command.key,
        value,
        service=scope,
        evidence=f"user said: {state.user_message[:120]}",
    )
    slots, sources = dict(state.slots), dict(state.slot_sources)
    slot = SLOT_FOR_KEY[command.key]
    update: dict = {
        "reply": f"Saved: your {label} is now {display_value(fact.value)}.",
        "goto": "finish",
    }
    if sources.get(slot, "").startswith("memory:") and slots.get(slot) != fact.value:
        slots[slot] = fact.value
        sources[slot] = f"memory:{fact.fact_id}"
        update |= {"slots": slots, "slot_sources": sources}
        if _invalidate(state, deps, f"{command.key} corrected"):
            update |= {"approval": None, "hold": None, "options": []}
    return update


def _memory_value(key: str, value: str | None, message: str):
    """A stored value only if it's valid and stated in the user's own message."""
    if not value:
        return None
    if key == "preferred_area":
        area = normalize_area(value)
        return (
            area
            if area and area.lower().replace(" ", "") in message.lower().replace(" ", "")
            else None
        )
    if key == "earliest_time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
        if not match:
            return None
        hour = int(match.group(1))
        numbers = {int(n) for n in re.findall(r"\d+", message)}
        return f"{hour:02d}:{match.group(2)}" if numbers & {hour, hour - 12} else None
    if key == "budget_max_inr":
        digits = re.sub(r"\D", "", value)
        return int(digits) if digits and digits in re.sub(r"[,\s]", "", message) else None
    if key == "preferred_time_window":
        return (
            value
            if value in ("morning", "afternoon", "evening") and value in message.lower()
            else None
        )
    return {"name": value} if value.lower() in message.lower() else None


@node("smalltalk")
def smalltalk(state: AgentState, deps: Deps) -> dict:
    facts = deps.memory.relevant(
        state.user_id,
        purpose="smalltalk",
        service=state.slots.get("service"),
        limit=deps.settings.max_facts_in_context,
    )
    reply = deps.ask_model(
        "smalltalk", ReplyText, state=state, system=SMALLTALK_SYSTEM, facts=facts, summary=True
    )
    return {"reply": reply.text, "goto": "finish"}


@node("clarify")
def clarify(state: AgentState, deps: Deps) -> dict:
    if state.task_id and state.phase not in Phase.FINISHED and state.slots.get("service"):
        lead = f"Did you mean to carry on with your {render.service_name(state.slots['service'])} request?"
    else:
        lead = "I'm not sure what you'd like me to do."
    return {
        "reply": f"{lead} I can book, reschedule or cancel a home service, or show and edit what I remember about you.",
        "goto": "finish",
    }


@node("finish")
def finish(state: AgentState, deps: Deps) -> dict:
    if state.task_id:
        deps.store.tasks.update(
            state.task_id, user_id=state.user_id, phase=state.phase, details=state.slots
        )
    update: dict = {"goto": ""}
    finished = state.phase in (Phase.DONE, Phase.FAILED, Phase.ABANDONED)
    if state.queued and finished and not state.queue_reminded and state.reply:
        item = state.queued[0]
        update |= {
            "reply": f'{state.reply}\n\nNext on your list: "{item["text"]}". Say "next" when'
            " you're ready.",
            "queue_reminded": True,
        }
    return update
