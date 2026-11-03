"""The eval harness: scenarios load, runs replay, and every invariant catches a broken agent.

An invariant that can't fail proves nothing, so each one is shown failing here against a real run
of a scenario with one deliberate defect (or, where a defect is hard to provoke, with the record
it would leave behind).
"""

import socket
from datetime import date

import pytest

from errand.config import Settings
from errand.contracts import Booking, DateRange, ToolName
from errand.decide.replay import RecordingDecider
from errand.eval.__main__ import forbid_network
from errand.eval.invariants import check_invariants
from errand.eval.recording import new_recording, recording_path
from errand.eval.runner import ScenarioRunner
from errand.eval.spec import EvalScenario, load_scenarios, scripted_decider, scripted_model
from errand.eval.suite import judge, run_scenario
from errand.gateway import Execution, ToolGateway
from errand.graph import builder
from errand.graph.nodes import node
from errand.llm.prompts import MIN_CONTEXT_ROOM_TOKENS, largest_system_prompt_tokens
from errand.llm.replay import RecordingModel
from errand.memory import ContextBuilder, MemoryService
from errand.store import LedgerStatus
from errand.trace import EventKind

REAL = Settings(_env_file=None)
SCENARIOS = {s.name: s for s in load_scenarios(REAL.evals_dir)}
FLOOR = largest_system_prompt_tokens() + MIN_CONTEXT_ROOM_TOKENS  # the smallest cap allowed


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, evals_dir=tmp_path / "evals", eval_traces_dir=tmp_path / "traces"
    )


@pytest.fixture
def play(tmp_path):
    """Play a scenario and keep its environment open, so a test can inspect or alter what it
    left behind before judging it."""
    runners = []

    def run(name: str, **overrides):
        scenario = SCENARIOS[name].model_copy(update=overrides)
        runner = ScenarioRunner(
            scenario,
            workdir=tmp_path / f"{name}{len(runners)}",
            model=scripted_model(scenario),
            decider=scripted_decider(scenario),
            scenarios_dir=REAL.scenarios_dir,
        )
        runners.append(runner)
        return runner.play()

    yield run
    for runner in runners:
        runner.close()


def violated(obs) -> dict[int, list[str]]:
    return {i.id: i.violations for i in check_invariants(obs) if i.violations}


def test_the_suite_has_the_core_scenarios_and_each_passes_offline(play):
    assert len(SCENARIOS) >= 12
    obs = play("happy_path_plumber")
    result = judge(obs, "scripted")
    assert result.passed, [c for c in result.checks if not c.passed]
    assert {i.id for i in result.invariants} == {1, 2, 3, 4, 5, 6, 7}
    assert all(i.checked for i in result.invariants)  # each one had something to check


def test_a_wrong_expectation_fails_its_scenario(play):
    expect = [
        c.model_copy(update={"day": "2026-11-08"}) if c.check == "booked" else c
        for c in SCENARIOS["correction_keeps_other_slots"].expect
    ]
    result = judge(play("correction_keeps_other_slots", expect=expect), "scripted")
    assert not result.passed
    (failed,) = [c for c in result.checks if not c.passed]
    assert failed.check == "booked" and "Sat 7 Nov" in failed.actual


# Each invariant fails when the agent breaks it


def test_invariant_1_catches_a_hold_without_an_approval(play, monkeypatch):
    def careless_hold(self, provider_id, slot, *, approval_id, task_id):
        hold = self._call(
            ToolName.HOLD_SLOT, self._tools.hold_slot, provider_id=provider_id, slot=slot
        )
        self._store.holds.record(hold, user_id=self.user_id, task_id=task_id)
        return hold

    monkeypatch.setattr(ToolGateway, "hold_slot", careless_hold)
    found = violated(play("happy_path_plumber"))
    assert list(found) == [1] and "hold_slot in step 3: called without an approval" in found[1]


def test_invariant_2_catches_a_second_booking_for_one_slot(play):
    obs = play("timeout_that_booked")
    (booking,) = [r for r in obs.world.booking_rows() if r["idempotency_key"]]
    # The provider's schema refuses a second booking under one key and a second confirmed
    # booking for one slot. Without the second guard, the duplicate a blind retry under a fresh
    # key would make: same user, provider and slot, no ledger row.
    with obs.world._db.transaction() as conn:
        conn.execute("DROP INDEX bookings_one_confirmed_per_slot")
        conn.execute(
            "INSERT INTO bookings (booking_id, user_id, provider_id, service, slot_start,"
            " slot_end, price_inr, status, technician, hold_id, idempotency_key, created_at,"
            " updated_at) SELECT 'b_dup', user_id, provider_id, service, slot_start, slot_end,"
            " price_inr, status, technician, NULL, 'idem_apr_fresh', created_at, updated_at"
            " FROM bookings WHERE booking_id = ?",
            (booking["booking_id"],),
        )
    found = violated(obs)
    assert list(found) == [2]
    assert any("idem_apr_fresh, which has no ledger row" in v for v in found[2])
    assert any("has 2 bookings with" in v for v in found[2])


def test_invariant_3_catches_success_reported_before_verification(play, monkeypatch):
    def optimistic_verify(self, idempotency_key):  # trusts the attempt instead of the source
        entry = self._entry(idempotency_key)
        booking: Booking = self._tools.list_bookings(self.user_id).bookings[-1]
        return Execution(
            idempotency_key=idempotency_key,
            op=entry.op,
            status=LedgerStatus.CONFIRMED,
            booking=booking,
        )

    monkeypatch.setattr(ToolGateway, "verify", optimistic_verify)
    found = violated(play("timeout_that_booked"))
    assert list(found) == [3] and "ledger row" in found[3][0] and "is unknown" in found[3][0]


def test_invariant_4_catches_a_run_that_dies_without_a_reply(play):
    found = violated(
        play("timeout_that_booked", world_overrides={"book_forced": ["crash_after_commit"]})
    )
    assert 4 in found
    assert any("raised SimulatedCrash" in v for v in found[4])
    assert any("never ended" in v for v in found[4])


def test_invariant_5_catches_an_unconfirmed_fact_in_the_context(play, monkeypatch):
    def everything_current(self, user_id, *, purpose, service, limit):  # forgets the status filter
        rows = self._store.memory.list_for_user(user_id)
        return [r for r in rows if r.key in ("preferred_area", "technician")][:limit]

    monkeypatch.setattr(MemoryService, "relevant", everything_current)
    found = violated(play("area_from_explicit_memory"))
    assert list(found) == [5]
    assert any("unconfirmed fact technician" in v for v in found[5])


def test_invariant_5_catches_a_context_over_the_token_cap(play, monkeypatch):
    real_build = ContextBuilder.build

    def uncapped(self, **kwargs):  # ignores the configured cap
        self._cap = 10**6
        return real_build(self, **kwargs)

    monkeypatch.setattr(ContextBuilder, "build", uncapped)
    found = violated(play("context_budget_long_history", settings={"context_token_cap": FLOOR}))
    assert list(found) == [5]
    assert any(f"tokens sent, cap {FLOOR}" in v for v in found[5])


def test_the_smallest_allowed_cap_holds_with_a_long_history(play):
    # Regression for docs/bugs.md #10: a cap below the largest prompt was accepted, then exceeded.
    obs = play("context_budget_long_history", settings={"context_token_cap": FLOOR})
    assert violated(obs) == {}
    dropped = [
        e.data["dropped"]
        for run in obs.runs
        for e in run.events
        if e.kind == "context" and "cap" in e.data
    ]
    assert any(dropped)  # the cap did bite: older history was left out to stay under it


def test_invariant_6_catches_slots_changed_by_tool_output(play, monkeypatch):
    @node("search")
    def search_that_trusts_tools(state, deps):  # lets the cheapest provider set the budget
        slots = state.slots
        result = deps.gateway.search_providers(
            slots["service"],
            slots["area"],
            DateRange(
                start=date.fromisoformat(slots["date_from"]),
                end=date.fromisoformat(slots["date_to"]),
            ),
        )
        cheapest = min(p.price_inr for p in result.providers)
        return {
            "candidates": [p.model_dump(mode="json") for p in result.providers],
            "slots": {**slots, "budget_max_inr": cheapest},
            "slot_sources": {**state.slot_sources, "budget_max_inr": "search_providers"},
            "goto": "rank_options",
        }

    monkeypatch.setitem(builder.NODES, "search", search_that_trusts_tools)
    found = violated(play("happy_path_plumber"))
    assert list(found) == [6]
    assert found[6] == [
        "budget_max_inr in search, step 1: changed by a node that handles tool output"
    ]


def test_invariant_7_catches_a_run_shown_another_users_bookings(play, monkeypatch):
    def demo_bookings(self):  # the old single-user agent: every caller is u_demo
        return self._call(ToolName.LIST_BOOKINGS, self._tools.list_bookings, user_id="u_demo")

    monkeypatch.setattr(ToolGateway, "list_bookings", demo_bookings)
    found = violated(play("user_isolation"))
    assert 7 in found
    shown = [v for v in found[7] if v.endswith("was shown another user's bookings")]
    assert shown and all("in step 6" in v for v in shown)  # Ravi's cancel; the demo user is fine


# Recording and replay


def test_a_recorded_run_replays_offline_with_the_same_outcome(settings):
    scenario = SCENARIOS["timeout_that_did_not_book"]
    # Stand-ins for the live model and Vega, wrapped by the same recorders the live run uses.
    model = RecordingModel(scripted_model(scenario))
    decider = RecordingDecider(scripted_decider(scenario))
    live = run_scenario(scenario, settings, model=model, decider=decider, mode="live")
    path = new_recording(scenario.name, model, decider).save(
        recording_path(settings.evals_dir, scenario.name)
    )
    assert path.is_file() and model.calls and decider.calls

    replayed = run_scenario(scenario.model_copy(update={"recording": scenario.name}), settings)
    assert live.passed and replayed.passed and replayed.mode == "replay"
    assert replayed.drift == []  # every request matched the recorded one
    assert [c.actual for c in replayed.checks] == [c.actual for c in live.checks]


def test_a_replay_that_diverges_from_its_recording_fails(settings):
    scenario = SCENARIOS["timeout_that_booked"]
    model = RecordingModel(scripted_model(scenario))
    decider = RecordingDecider(scripted_decider(scenario))
    run_scenario(scenario, settings, model=model, decider=decider, mode="live")
    new_recording(scenario.name, model, decider).save(
        recording_path(settings.evals_dir, scenario.name)
    )
    # The user now says something else after the options: no recorded decision covers it.
    steps = [*scenario.steps[:1], scenario.steps[1].model_copy(update={"say": "the cheapest one"})]
    diverged = scenario.model_copy(
        update={"recording": scenario.name, "steps": steps, "expect": []}
    )
    result = run_scenario(diverged, settings)
    assert not result.passed
    assert any(c.check == "script" or c.check == "steps" for c in result.checks if not c.passed)


# Scenario files and the offline guard


def test_scenario_labels_must_be_quoted_and_steps_do_one_thing():
    base = SCENARIOS["happy_path_plumber"].model_dump(exclude={"source"})
    with pytest.raises(ValueError, match="quoted"):
        scripted_decider(EvalScenario.model_validate({**base, "decider": {"approval": [True]}}))
    with pytest.raises(ValueError, match="exactly one"):
        EvalScenario.model_validate({**base, "steps": [{"say": "hi", "advance": 5}]})


def test_offline_evals_refuse_the_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)  # restored afterwards
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    forbid_network()
    with pytest.raises(RuntimeError, match="offline eval"):
        socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_claims_are_traced_where_success_is_reported():
    assert EventKind.CLAIM == "claim"
