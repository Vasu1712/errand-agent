"""Context management: seeded memories and history, the token cap, and what reaches the model."""

import pytest

from errand.llm.base import estimate_tokens
from errand.memory import RELEVANT_KEYS, ContextBuilder, MemoryService
from errand.memory.seed import seed_demo
from errand.mocks import FakeClock, MockWorld, Scenario
from errand.store import FactSource, FactStatus, Store
from tests.support import START

USER = "u_demo"


@pytest.fixture
def seeded(tmp_path):
    clock = FakeClock(START)
    store = Store(tmp_path / "agent.db", now=clock.now)
    world = MockWorld(Scenario(), db_path=tmp_path / "world.db", clock=clock, seed=42)
    assert seed_demo(store, world, clock) == ["u_demo", "u_asha", "u_ravi"]
    assert seed_demo(store, world, clock) == []  # seeding is once only
    yield store, world
    store.close()
    world.close()


def builder(store, cap=6000, turns=6, summary=600):
    return ContextBuilder(store, token_cap=cap, recent_turns=turns, summary_token_budget=summary)


def test_the_seed_has_fifty_plus_memories_of_every_kind_and_a_long_history(seeded):
    store, world = seeded
    facts = store.memory.list_for_user(USER, include_inactive=True)
    statuses = {fact.status for fact in facts}
    assert len(facts) >= 50
    assert statuses == {
        FactStatus.ACTIVE,
        FactStatus.UNCONFIRMED,
        FactStatus.SUPERSEDED,
        FactStatus.FORGOTTEN,
    }
    assert {fact.source for fact in facts} == {FactSource.EXPLICIT, FactSource.INFERRED}
    assert store.messages.count(USER) >= 100
    assert len(world.list_bookings(USER).bookings) == 4


def test_three_users_are_seeded_and_two_of_them_compete_for_the_same_slots(seeded):
    store, world = seeded

    def prefs(user):
        current = {(f.key, f.service_tag): f.value for f in store.memory.list_for_user(user)}
        keys = [("preferred_area", None), ("earliest_time", None), ("budget_max_inr", "plumber")]
        return [current.get(key) for key in keys] + [current.get(("usual_service", None))]

    assert prefs("u_asha") == [*prefs(USER)[:3], "plumber"]  # same area, prefs and service
    assert prefs("u_ravi")[0] == "HSR Layout"
    bookings = {
        u: {b.booking_id for b in world.list_bookings(u).bookings} for u in ("u_asha", "u_ravi")
    }
    assert all(bookings.values()) and not set.intersection(*bookings.values())
    assert all(store.messages.count(u) >= 8 for u in ("u_asha", "u_ravi"))
    cleaners = {
        u: {b.provider_id for b in world.list_bookings(u).bookings if b.service == "cleaning"}
        for u in (USER, "u_asha")
    }
    assert cleaners[USER] & cleaners["u_asha"]  # one provider, two users' bookings


def test_forgotten_seed_facts_are_scrubbed_from_the_history(seeded):
    store, _ = seeded
    history = store.messages.history(USER, exclude_ids=[], limit=500)
    text = " ".join(m.content for m in history)
    assert "Indiranagar 12th Main" not in text and "4410" not in text
    assert "[forgotten]" in text


def test_only_active_relevant_facts_are_selected(seeded):
    store, _ = seeded
    facts = MemoryService(store).relevant(USER, purpose="book", service="plumber", limit=50)
    assert facts
    for fact in facts:
        assert fact.status is FactStatus.ACTIVE
        assert fact.key in RELEVANT_KEYS["book"]
        assert fact.service_tag in (None, "plumber")
    keys = {(f.key, f.service_tag) for f in facts}
    assert ("budget_max_inr", "plumber") in keys and ("preferred_area", None) in keys
    assert ("earliest_time", "electrician") not in keys  # another service's preference
    assert all(f.key != "pets" for f in facts)  # stored, but irrelevant to booking
    assert not MemoryService(store).relevant(USER, purpose="cancel", service="plumber", limit=50)


def test_the_context_stays_under_the_cap_and_says_what_went_in(seeded):
    store, _ = seeded
    facts = MemoryService(store).relevant(USER, purpose="smalltalk", service=None, limit=12)
    pack = builder(store).build(
        purpose="smalltalk",
        system="You are a test.",
        latest="hello",
        user_id=USER,
        thread_id="u_demo:history",
        facts=facts,
        summary=True,
    )
    manifest = pack.manifest
    sent = sum(estimate_tokens(m.content) for m in pack.messages)
    assert manifest.total_tokens <= 6000 and sent <= 6000 + 8 * len(pack.messages)
    assert set(manifest.fact_ids) == {f.fact_id for f in facts}
    assert manifest.message_ids and manifest.summary_message_ids
    assert not set(manifest.message_ids) & set(manifest.summary_message_ids)
    included = {
        f.fact_id
        for f in store.memory.list_for_user(USER, include_inactive=True)
        if f.fact_id in manifest.fact_ids
    }
    assert all(store.memory.get(fid, user_id=USER).status is FactStatus.ACTIVE for fid in included)


def test_a_tight_cap_drops_history_before_facts(seeded):
    store, _ = seeded
    facts = MemoryService(store).relevant(USER, purpose="smalltalk", service=None, limit=12)
    roomy = (
        builder(store)
        .build(
            purpose="smalltalk",
            system="You are a test.",
            latest="hello",
            user_id=USER,
            thread_id="u_demo:history",
            facts=facts,
            summary=True,
        )
        .manifest
    )
    needed = roomy.sections["system"] + roomy.sections["latest"] + roomy.sections["facts"]
    tight = builder(store, cap=needed + 20).build(
        purpose="smalltalk",
        system="You are a test.",
        latest="hello",
        user_id=USER,
        thread_id="u_demo:history",
        facts=facts,
        summary=True,
    )
    assert tight.manifest.total_tokens <= needed + 20
    assert tight.manifest.fact_ids == roomy.fact_ids  # every fact kept
    assert len(tight.manifest.message_ids) < len(roomy.message_ids)  # history went first
    assert any(d.startswith("message:") for d in tight.manifest.dropped)
    assert tight.messages[-1].content == "hello"  # the latest message always goes in, last
