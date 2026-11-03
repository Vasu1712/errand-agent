"""End-to-end flows through the real graph, gateway and mock world, with rule-based models."""

from datetime import datetime, timedelta

import pytest

from errand.contracts import IST, BookingState, Slot, ToolError, ToolErrorCode, ToolName
from errand.decide.rules import rule_decider
from errand.llm import ModelUnavailable
from errand.llm.rules import rule_model
from errand.memory import MemoryService
from errand.mocks import BookOutcome, MockWorld, SimulatedCrash
from errand.store import ApprovalStatus, FactSource, FactStatus, LedgerStatus
from errand.worker import poll_once, reconcile
from tests.support import Harness


@pytest.fixture
def make(tmp_path):
    harnesses = []

    def build(**kwargs):
        harness = Harness(tmp_path / f"h{len(harnesses)}", **kwargs)
        harnesses.append(harness)
        return harness

    yield build
    for harness in harnesses:
        harness.stop()


def test_a_booking_asks_only_for_what_is_missing_then_needs_a_clear_yes(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    asked = h.say("Find a plumber for my kitchen tap this week, mornings only, under ₹1,500")
    assert asked.phase == "collecting"
    assert "area" in asked.reply and "budget" not in asked.reply and "day" not in asked.reply

    options = h.say("Koramangala")
    assert options.phase == "presenting" and options.options
    assert "Options for plumber in Koramangala" in options.reply

    summary = h.say("1")
    assert summary.phase == "awaiting_approval" and summary.approval
    assert h.bookings() == []  # nothing booked without a yes

    done = h.say("yes")
    assert done.phase == "done" and done.reply.startswith("Booked. Confirmation b_")
    assert len(h.bookings()) == 1
    approval = h.store.approvals.get(summary.approval["approval_id"], user_id=h.user_id)
    assert approval.status is ApprovalStatus.CONSUMED
    ledger = h.store.ledger.for_approval(approval.approval_id, user_id=h.user_id)
    assert ledger.status is LedgerStatus.CONFIRMED and ledger.attempts == 1


SEARCH = "Find a plumber in Koramangala this week, mornings only, under ₹1,500"


def test_a_correction_keeps_every_other_detail_and_voids_the_proposal(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    h.say(SEARCH)
    summary = h.say("1")
    corrected = h.say("make it Saturday instead")

    state = h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values
    slots = state["slots"]
    assert (slots["service"], slots["area"], slots["budget_max_inr"], slots["time_window"]) == (
        "plumber",
        "Koramangala",
        1500,
        "morning",
    )
    assert slots["date_from"] == slots["date_to"] == "2026-11-07"
    # The same provider can do Saturday: a new summary to confirm, under a new approval.
    assert corrected.phase == "awaiting_approval" and "Sat 7 Nov" in corrected.reply
    assert corrected.approval["approval_id"] != summary.approval["approval_id"]
    assert (
        h.store.approvals.get(summary.approval["approval_id"], user_id=h.user_id).status
        is ApprovalStatus.INVALIDATED
    )
    assert h.bookings() == []


def test_a_correction_vega_is_unsure_about_is_still_applied(make):
    # Regression for docs/bugs.md #11: live Vega read this as "no" at 0.765, under the threshold.
    # The agent asked for a clear yes again, the change was lost, and the old slot got booked.
    decider = rule_decider()
    decider._rules["approval"] = lambda text: ("no", 0.765)
    h = make(book_forced=[BookOutcome.SUCCESS], decider=decider)
    h.say(SEARCH)
    summary = h.say("1")
    corrected = h.say("Actually, make it Saturday instead.")
    assert corrected.phase == "awaiting_approval" and "Sat 7 Nov" in corrected.reply
    status = h.store.approvals.get(summary.approval["approval_id"], user_id=h.user_id).status
    assert status is ApprovalStatus.INVALIDATED
    slots = h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values["slots"]
    assert slots["date_from"] == slots["date_to"] == "2026-11-07" and slots["area"] == "Koramangala"


def test_no_is_not_a_booking_and_hedges_are_not_a_yes(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    h.say(SEARCH)
    first = h.say("1")
    hedged = h.say("yes?")
    assert hedged.phase == "awaiting_approval" and hedged.reply.startswith(
        "I need a clear yes or no"
    )
    declined = h.say("no")
    assert declined.phase == "presenting" and h.bookings() == []
    assert (
        h.store.approvals.get(first.approval["approval_id"], user_id=h.user_id).status
        is ApprovalStatus.INVALIDATED
    )


def test_a_repeated_confirmation_never_books_twice(make):
    h = make(book_forced=[BookOutcome.SUCCESS, BookOutcome.SUCCESS])
    done = h.book_through(SEARCH)
    again = h.say("yes")
    yes_once_more = h.say("yes, book it")
    assert done.phase == "done" and len(h.bookings()) == 1
    assert "Booked" not in again.reply and "Booked" not in yes_once_more.reply
    assert len([e for e in h.all_events("tool.call") if e.data["tool"] == "book"]) == 1


def test_a_timeout_after_success_is_verified_and_booked_once(make):
    h = make(book_forced=[BookOutcome.TIMEOUT_BOOKED, BookOutcome.SUCCESS])
    done = h.book_through(SEARCH)
    assert done.phase == "done" and done.reply.startswith("Booked. Confirmation")
    assert len(h.bookings()) == 1
    tools = [e.data["tool"] for e in h.events(kind="tool.call")]
    assert tools == ["hold_slot", "book", "get_booking_status"]  # held only after the yes
    ledger = [e.data["event"] for e in h.events(kind="ledger")]
    assert ledger == ["in_flight", "unknown", "confirmed"]


def test_an_unverifiable_outcome_is_reported_unconfirmed_and_reconciled_later(make):
    h = make(book_forced=[BookOutcome.TIMEOUT_BOOKED])
    h.say(SEARCH)
    h.say("1")
    real = h.world.get_booking_status

    def unreachable(**kwargs):
        raise ToolError.of(ToolName.GET_BOOKING_STATUS, ToolErrorCode.SERVICE_UNAVAILABLE)

    h.world.get_booking_status = unreachable
    unsure = h.say("yes")
    assert unsure.phase == "unconfirmed" and "Booked" not in unsure.reply
    h.world.get_booking_status = real

    settled = reconcile(h.agent)
    assert [s.status for s in settled] == [LedgerStatus.CONFIRMED]
    update = h.store.outbox.since(h.user_id)[-1]
    assert update.kind == "reconciled" and update.body.startswith("Update: Booked.")
    later = h.say("thanks")
    assert "Booked" not in later.reply  # told once, not again


def test_a_provider_slower_than_the_turn_budget_is_settled_later_not_reported(make):
    # Regression for docs/bugs.md #9: the time budget ran on the real clock, so on the fake
    # clock a 40 s provider timeout cost nothing and this turn went on to report "Booked".
    h = make(book_forced=[BookOutcome.TIMEOUT_BOOKED], timeout_latency_s=40)
    h.say(SEARCH)
    h.say("1")
    slow = h.say("yes")
    assert slow.phase == "unconfirmed"
    (end,) = h.events(kind="run.end")
    # The simulated wait counted, and the client gave up at the turn's hard limit
    # (docs/bugs.md #18) instead of waiting the provider's full 40 s.
    assert end.data["budget"]["elapsed_s"] == end.data["budget"]["timeout_s"] == 30
    assert [e.data["event"] for e in h.events(kind="ledger")] == ["in_flight", "unknown", "unknown"]
    assert [s.status for s in reconcile(h.agent)] == [LedgerStatus.CONFIRMED]


def test_prompt_injection_in_reviews_changes_nothing(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    options = h.say(SEARCH)
    flags = h.events(kind="injection")
    assert flags and all(e.data["action"].startswith("ignored") for e in flags)
    assert options.phase == "presenting" and h.bookings() == []
    assert not [e for e in h.events(kind="approval") if e.data["event"] == "granted"]
    summary = h.say("1")
    assert summary.phase == "awaiting_approval"  # still needs the user's yes


def test_a_quote_that_differs_from_the_approved_price_needs_a_fresh_yes(make):
    h = make(book_forced=[BookOutcome.SUCCESS], quote_delta_inr={"*": 100})
    h.say("Find a plumber in Koramangala this week, mornings only, no budget limit")
    listed = h.say("1")
    assert listed.phase == "awaiting_approval"
    assert "hold_slot" not in {
        e.data["tool"] for e in h.all_events("tool.call")
    }  # not before a yes
    first = h.store.approvals.get(listed.approval["approval_id"], user_id=h.user_id)

    again = h.say("yes")  # the hold quotes ₹100 more than the listing
    assert again.phase == "awaiting_approval" and h.bookings() == []
    assert "quoted when I held the slot" in again.reply
    assert [e.data["tool"] for e in h.events(kind="tool.call")] == ["hold_slot"]
    assert h.store.approvals.get(first.approval_id, user_id=h.user_id).status is (
        ApprovalStatus.INVALIDATED
    )
    second = h.store.approvals.get(again.approval["approval_id"], user_id=h.user_id)
    assert second.material["price_inr"] == first.material["price_inr"] + 100

    done = h.say("yes")
    assert done.phase == "done" and h.bookings()[0].price_inr == second.material["price_inr"]
    # The hold from the first yes is still good, so it is booked as quoted.
    assert [e.data["tool"] for e in h.events(kind="tool.call")] == ["book", "get_booking_status"]


def test_a_quote_over_budget_is_never_booked_or_put_up_for_approval(make):
    h = make(quote_delta_inr={"*": 2000})
    h.say(SEARCH)
    listed = h.say("1")
    after = h.say("yes")
    assert after.phase == "presenting" and "above your ₹1,500 budget" in after.reply
    status = h.store.approvals.get(listed.approval["approval_id"], user_id=h.user_id).status
    assert status is ApprovalStatus.INVALIDATED and h.bookings() == []
    requested = [
        e.data["material"]["price_inr"]
        for e in h.all_events("approval")
        if e.data["event"] == "requested"
    ]
    assert requested and max(requested) <= 1500


def test_slot_taken_goes_back_to_the_other_options(make):
    h = make(book_forced=[BookOutcome.SLOT_TAKEN])
    h.say(SEARCH)
    h.say("1")
    after = h.say("yes")
    assert after.phase == "presenting" and "Someone else booked that slot" in after.reply
    assert h.bookings() == []


def test_a_hold_that_lapses_before_the_second_yes_is_taken_again(make):
    h = make(book_forced=[BookOutcome.SUCCESS], quote_delta_inr={"*": 100})
    h.say("Find a plumber in Koramangala this week, mornings only, no budget limit")
    h.say("1")
    repriced = h.say("yes")
    h.clock.advance(11 * 60)  # the first hold expires before the yes to the new price
    done = h.say("yes")
    assert done.phase == "done" and len(h.bookings()) == 1
    held = [e.data for e in h.events(kind="tool.call") if e.data["tool"] == "hold_slot"]
    assert [call["approval_id"] for call in held] == [repriced.approval["approval_id"]]
    assert len([e for e in h.all_events("tool.call") if e.data["tool"] == "hold_slot"]) == 2


def test_a_hold_that_expires_at_book_needs_a_fresh_yes(make, monkeypatch):
    h = make(book_forced=[BookOutcome.SUCCESS], hold_ttl_s=20)
    h.say(SEARCH)
    first = h.say("1")
    real_hold, slow = MockWorld.hold_slot, [True]

    def slow_hold(world, **kwargs):  # once, the provider answers after the hold has lapsed
        hold = real_hold(world, **kwargs)  # as the calling user's view: holds belong to a user
        if slow.pop() if slow else False:
            h.clock.advance(25)  # within the run's 30 s budget
        return hold

    monkeypatch.setattr(MockWorld, "hold_slot", slow_hold)
    again = h.say("yes")
    assert again.phase == "awaiting_approval" and "hold ran out" in again.reply
    assert again.approval["approval_id"] != first.approval["approval_id"] and h.bookings() == []
    spent = h.store.ledger.for_approval(first.approval["approval_id"], user_id=h.user_id)
    assert spent.status is LedgerStatus.NOT_BOOKED and "hold_expired" in spent.last_error
    assert h.say("yes").phase == "done" and len(h.bookings()) == 1


def seed_upcoming(h, service="ac_service", day=5, hour=10):
    provider = f"prov_{service}_koramangala_1"
    start = datetime(2026, 11, day, hour, tzinfo=IST)
    return h.world.seed_booking(
        user_id=h.user_id,
        provider_id=provider,
        slot=Slot(start=start, end=start + timedelta(hours=2)),
    )


def test_cancel_needs_a_yes_and_survives_the_first_503(make):
    h = make()
    booking = seed_upcoming(h)
    summary = h.say("Please cancel my AC servicing")
    assert summary.phase == "awaiting_approval" and booking.booking_id in summary.reply
    done = h.say("yes")
    assert done.phase == "done" and done.reply.startswith("Cancelled.")
    assert h.bookings()[0].status is BookingState.CANCELLED
    cancels = [e for e in h.events(kind="tool.call") if e.data["tool"] == "cancel"]
    assert len(cancels) == 2 and cancels[0].data["failure"]["code"] == "service_unavailable"
    assert len({e.data["args"]["idempotency_key"] for e in cancels}) == 1


def test_declining_a_cancellation_keeps_the_booking(make):
    h = make()
    seed_upcoming(h)
    h.say("cancel my AC servicing")
    kept = h.say("no")
    assert (
        kept.reply == "Okay, I won't cancel it."
        and h.bookings()[0].status is BookingState.CONFIRMED
    )


def test_reschedule_steps_past_a_conflicting_slot(make):
    h = make(reschedule_conflicts=["10:00"])
    booking = seed_upcoming(h, day=5, hour=10)
    summary = h.say("Move my AC servicing to Saturday")
    assert summary.phase == "awaiting_approval" and "Sat 7 Nov" in summary.reply
    new_slot = h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values[
        "new_slot"
    ]
    done = h.say("yes")
    if datetime.fromisoformat(new_slot["start"]).astimezone(IST).hour == 10:
        # The closest slot to the original 10:00 conflicts: back to planning, a fresh approval.
        assert done.phase == "awaiting_approval" and "can't take that slot" in done.reply
        done = h.say("yes")
    assert done.phase == "done" and done.reply.startswith("Rescheduled.")
    moved = h.bookings()[0]
    assert moved.booking_id == booking.booking_id and moved.slot.start.astimezone(IST).day == 7


def test_memory_can_be_inspected_corrected_and_forgotten_by_chat(make):
    h = make()
    h.store.memory.remember(h.user_id, "preferred_area", "HSR Layout", source=FactSource.EXPLICIT)
    shown = h.say("What do you remember about me?")
    assert "HSR Layout" in shown.reply and "you told me" in shown.reply
    h.say("Actually, update my area to Koramangala")
    assert h.store.memory.current(h.user_id, "preferred_area").value == "Koramangala"
    h.say("Forget my preferred area")
    assert h.store.memory.current(h.user_id, "preferred_area") is None
    asked = h.say("Book a plumber tomorrow, no budget limit")
    assert asked.phase == "collecting" and "area" in asked.reply  # no longer known, so asked


def test_forgotten_and_unconfirmed_facts_never_reach_the_model(make):
    model = rule_model()
    h = make(model=model)
    h.store.memory.remember(h.user_id, "preferred_area", "Indiranagar", source=FactSource.EXPLICIT)
    h.store.memory.remember(
        h.user_id, "budget_max_inr", 4321, source=FactSource.INFERRED, service_tag="plumber"
    )
    h.say("I live in Indiranagar, near the metro")
    h.say("Forget my area")
    before = len(model.requests)
    later_runs = [h.say("hello there").run_id, h.say("Book a plumber tomorrow").run_id]
    later = model.requests[before:]
    sent = "\n".join(m.content for request in later for m in request.messages)
    assert {r.purpose for r in later} == {"smalltalk", "slot_patch"}
    assert "Indiranagar" not in sent and "4321" not in sent
    assert "[forgotten]" in sent  # the old message was scrubbed, not just skipped
    contexts = [e for run in later_runs for e in h.events(run, kind="context")]
    forgotten_ids = {
        f.fact_id
        for f in h.store.memory.list_for_user(h.user_id, include_inactive=True)
        if f.status is FactStatus.FORGOTTEN
    }
    unconfirmed_ids = {
        f.fact_id
        for f in h.store.memory.list_for_user(h.user_id)
        if f.status is FactStatus.UNCONFIRMED
    }
    sent_ids = {fid for e in contexts for fid in e.data.get("fact_ids", [])}
    assert not sent_ids & (forgotten_ids | unconfirmed_ids)


def test_an_inferred_preference_is_confirmed_by_a_yes_no_question_before_use(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    h.store.memory.remember(
        h.user_id,
        "preferred_area",
        "Koramangala",
        source=FactSource.INFERRED,
        evidence="3 past bookings",
    )
    asked = h.say("Find a plumber this week, no budget limit")
    assert (
        asked.phase == "confirming_fact"
        and "Koramangala" in asked.reply
        and "(yes/no)" in asked.reply
    )
    options = h.say("yes")
    assert options.phase == "presenting"
    fact = h.store.memory.current(h.user_id, "preferred_area")
    assert (fact.source, fact.status) == (FactSource.EXPLICIT, FactStatus.ACTIVE)


def test_running_out_of_tool_calls_stops_cleanly_and_continue_resumes(make):
    h = make(settings={"max_tool_calls": 6})  # 5 are reserved for execute and verify
    stopped = h.say(SEARCH)  # search fits; the first review lookup doesn't
    assert stopped.outcome == "stopped" and 'Say "continue"' in stopped.reply
    assert [e.data["kind"] for e in h.events(kind="budget")] == ["budget"]
    assert h.bookings() == []
    h.agent.settings = h.agent.settings.model_copy(update={"max_tool_calls": 12})
    resumed = h.say("continue")
    assert resumed.phase == "presenting" and "Koramangala" in resumed.reply


def test_a_model_outage_ends_the_turn_cleanly(make):
    def down(request):
        raise ModelUnavailable("Ollama is not running")

    h = make(model=rule_model())
    h.model._rules["slot_patch"] = down
    stopped = h.say("Find a plumber in Koramangala this week")
    assert stopped.outcome == "stopped" and "couldn't reach my language model" in stopped.reply
    assert h.bookings() == []


def test_a_restart_mid_booking_is_settled_from_the_source_of_truth(make):
    h = make(book_forced=[BookOutcome.CRASH_AFTER_COMMIT, BookOutcome.SUCCESS])
    h.say(SEARCH)
    h.say("1")
    with pytest.raises(SimulatedCrash):
        h.say("yes")
    h.stop()
    h.start()  # a new process on the same files
    settled = reconcile(h.agent)
    assert [s.status for s in settled] == [LedgerStatus.CONFIRMED]
    assert len(h.bookings()) == 1
    assert h.store.outbox.since(h.user_id)[-1].body.startswith("Update: Booked.")
    after = h.say("did it work?")
    assert len(h.bookings()) == 1 and after.outcome != "error"


def test_a_provider_cancellation_is_noticed_and_rebooking_needs_a_fresh_yes(make):
    h = make(book_forced=[BookOutcome.SUCCESS, BookOutcome.SUCCESS])
    booked = h.book_through(SEARCH)
    original = h.bookings()[0]
    h.world.provider_cancelled(original.booking_id)
    assert poll_once(h.agent) == 1
    note = h.store.outbox.since(h.user_id)[-1]
    assert note.kind == "provider_cancelled" and "cancelled your plumber visit" in note.body
    assert "nothing is held or booked" in note.body and note.payload["options"]
    thread = note.payload["thread_id"]
    assert thread != h.thread_id and len(h.bookings()) == 1
    summary = h.say("1", thread_id=thread)
    assert summary.phase == "awaiting_approval"
    rebooked = h.say("yes", thread_id=thread)
    assert rebooked.phase == "done" and rebooked.reply.startswith("Booked")
    providers = {b.provider_id for b in h.bookings() if b.status is BookingState.CONFIRMED}
    assert original.provider_id not in providers
    assert poll_once(h.agent) == 0 and booked.phase == "done"


def test_provider_events_for_past_or_user_cancelled_bookings_are_ignored(make):
    h = make()
    cancelled_by_user = seed_upcoming(h)
    with h.world._db.transaction() as conn:
        conn.execute(
            "UPDATE bookings SET status = 'cancelled' WHERE booking_id = ?",
            (cancelled_by_user.booking_id,),
        )
    past = seed_upcoming(h, service="plumber", day=2)
    h.world.provider_cancelled(cancelled_by_user.booking_id)
    h.world.provider_cancelled(past.booking_id)
    assert poll_once(h.agent) == 2
    assert h.store.outbox.since(h.user_id) == []
    reasons = [
        e.data["reason"]
        for e in h.all_events("state.transition")
        if e.data.get("node") == "provider_cancel_poller"
    ]
    assert sorted(reasons) == [
        "booking is cancelled, not cancelled by the provider",
        "the visit time has already passed",
    ]


def test_the_same_electrician_as_last_time(make):
    h = make(book_forced=[BookOutcome.SUCCESS])
    past = datetime(2026, 10, 14, 10, tzinfo=IST)
    h.world.seed_booking(
        user_id=h.user_id,
        provider_id="prov_electrician_koramangala_3",
        slot=Slot(start=past, end=past + timedelta(hours=2)),
    )
    options = h.say(
        "Book the same electrician as last time, but not before 10 a.m., this week, no budget limit"
    )
    assert options.phase == "presenting"
    state = h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values
    assert {o["provider_id"] for o in state["options"]} == {"prov_electrician_koramangala_3"}
    assert all(
        datetime.fromisoformat(o["slot"]["start"]).astimezone(IST).hour >= 10
        for o in state["options"]
    )


def test_a_proactive_follow_up_that_runs_out_of_budget_can_be_continued(make):
    h = make(settings={"max_tool_calls": 7})  # two unreserved calls: search, one review lookup
    booking = seed_upcoming(h)
    h.world.provider_cancelled(booking.booking_id)
    poll_once(h.agent)
    note = h.store.outbox.since(h.user_id)[-1]
    assert note.kind == "provider_cancelled" and 'Say "continue"' in note.body
    h.agent.settings = h.agent.settings.model_copy(update={"max_tool_calls": 12})
    resumed = h.say("continue", thread_id=note.payload["thread_id"])
    assert resumed.phase == "presenting" and resumed.options


def test_a_replacement_searches_within_the_users_own_budget_for_the_service(make):
    # Regression for docs/bugs.md #23: a booking the agent didn't make got a replacement budget
    # of its price plus a fifth, though the user had said what they'd pay for the service.
    h = make()
    first, second = (
        seed_upcoming(h, service="cleaning"),
        seed_upcoming(h, service="cleaning", day=6),
    )

    def replacement(booking):
        h.world.provider_cancelled(booking.booking_id)
        poll_once(h.agent)
        thread = h.store.outbox.since(h.user_id)[-1].payload["thread_id"]
        state = h.agent.graph.get_state({"configurable": {"thread_id": thread}}).values
        return state["slots"]["budget_max_inr"], state["slot_sources"]["budget_max_inr"]

    # No budget known: what the cancelled visit cost, plus a fifth.
    assert replacement(first) == (
        round(first.price_inr * 1.2 / 50) * 50,
        f"booking:{first.booking_id}",
    )
    said = MemoryService(h.store).remember(
        h.user_id, "budget_max_inr", 2500, service="cleaning", evidence="said so"
    )
    assert replacement(second) == (2500, f"memory:{said.fact_id}")


class UnsureIntent:
    """The rule decider, except that intent comes back at Vega's 0.827 from the live trace."""

    def __init__(self):
        self._rules = rule_decider()

    def decide(self, text, question):
        decision = self._rules.decide(text, question)
        if question.id == "intent":
            return decision.model_copy(update={"confidence": 0.827})
        return decision

    def decide_many(self, text, questions):
        return {q.id: self.decide(text, q) for q in questions}


def test_a_plain_booking_request_vega_is_unsure_about_still_starts_a_booking(make):
    # Regression for docs/bugs.md #24: "book a plumber for me" at 0.827 got "I'm not sure what
    # you'd like me to do" on every try.
    h = make(decider=UnsureIntent())
    asked = h.say("book a plumber for me")
    assert asked.phase == "collecting" and "area" in asked.reply
    assert h.bookings() == []
    # Without a service named, or with another errand's words, an unsure intent still asks.
    assert make(decider=UnsureIntent()).say("can you help me").phase == "idle"
    unsure_cancel = make(decider=UnsureIntent()).say("book or cancel my plumber?")
    assert unsure_cancel.phase == "idle" and "not sure" in unsure_cancel.reply


def _misfiling_model():
    """The rule model, except that a "keep everything else" correction comes back the way Qwen
    gave it live: the day and time of day only in the problem text."""
    from errand.llm.rules import pick, slot_patch, smalltalk
    from errand.llm.scripted import ScriptedModel

    def misfiling(request):
        latest = request.messages[-1].content
        if "keep everything else" in latest:
            when = latest.split("make it ", 1)[1].split(",", 1)[0]
            return {"service": "plumber", "problem": f"kitchen tap on {when}"}
        return slot_patch(request)

    return ScriptedModel(
        {"slot_patch": misfiling, "find_booking": slot_patch, "pick": pick, "smalltalk": smalltalk}
    )


def _summary_then(h, correction):
    h.say("Find a plumber in Koramangala on Saturday, mornings only, under ₹1,500")
    first = h.say("1")
    assert first.phase == "awaiting_approval"
    state = h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values
    provider = state["options"][0]["provider_id"]
    changed = h.say(correction)
    return (
        first,
        changed,
        provider,
        h.agent.graph.get_state({"configurable": {"thread_id": h.thread_id}}).values,
    )


def test_a_change_to_a_summary_comes_back_as_a_new_summary_for_the_same_provider(make):
    # Regression for docs/bugs.md #25: Qwen put the new day in the problem text and left the day
    # and time empty, so the same Saturday-morning options came back.
    h = make(book_forced=[BookOutcome.SUCCESS], model=_misfiling_model())
    first, changed, provider, state = _summary_then(
        h, "Actually make it Thursday morning, but keep everything else."
    )
    assert state["slots"]["date_from"] == state["slots"]["date_to"] == "2026-11-05"
    assert state["slots"]["time_window"] == "morning"
    assert (state["slots"]["area"], state["slots"]["budget_max_inr"]) == ("Koramangala", 1500)
    # Straight to a summary for the same provider, under a new approval; nothing booked yet.
    assert changed.phase == "awaiting_approval" and "Thu 5 Nov" in changed.reply
    assert changed.approval["approval_id"] != first.approval["approval_id"]
    picked = next(o for o in state["options"] if o["option_id"] == state["pick"])
    assert picked["provider_id"] == provider
    assert h.bookings() == []
    assert h.say("yes").phase == "done" and len(h.bookings()) == 1


def test_a_change_the_same_provider_cannot_take_shows_the_others_and_says_why(make):
    h = make(model=_misfiling_model())
    _first, changed, _provider, state = _summary_then(
        h, "Actually make it Thursday afternoon, but keep everything else."
    )
    assert (state["slots"]["date_from"], state["slots"]["time_window"]) == (
        "2026-11-05",
        "afternoon",
    )
    assert (
        changed.phase == "presenting" and "has no slot that fits the new details" in changed.reply
    )
    assert "Thu 5 Nov" in changed.reply and h.bookings() == []
