"""Plays one scenario against the real agent (graph, gateway, store, mock world), on a fake clock.

Every user turn goes through the real ASGI app (``POST /chat`` with a user id and a session id),
so identity, scoping and the async handlers are part of what is tested. A ``together`` step fires
its turns at the same moment with ``asyncio.gather``; the app runs each on its own worker thread,
and a turnstile (errand/mocks/turnstile.py) can pin the order in which they reach a point (a slot
claim, a thread's lock), so the race is real and its result is the same on every run.

The model and decider are scripted or replayed offline, or live ones wrapped by recorders. What a
run leaves behind (traces, the ledger, approvals, memory rows, the mock world, every model
request) is collected into an `Observation` for the checks and invariants to read.
"""

import asyncio
import json
import logging
import secrets
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

from errand.agent import Agent, TurnResult
from errand.api import Services, create_app
from errand.config import Settings
from errand.contracts import BookingState, Slot
from errand.decide import DecisionClient
from errand.eval.spec import EvalScenario, Step
from errand.graph import open_checkpointer
from errand.ids import thread_id_for
from errand.llm import ChatMessage, ModelClient, ModelRequest
from errand.llm.base import T
from errand.memory import MemoryService
from errand.memory.seed import SEED_USERS, seed_demo
from errand.mocks import FakeClock, MockWorld, Scenario, SimulatedCrash, Turnstile, load_scenario
from errand.store import FactSource, Store
from errand.trace import TraceEvent, current_run, read_trace
from errand.worker import Worker

log = logging.getLogger(__name__)
ORACLE = "eval oracle: reads every user's rows to check the invariants"


@dataclass
class Captured:
    step: int
    user_id: str | None  # whose run made the request (several can overlap)
    run_id: str | None
    request: ModelRequest


class CapturingModel:
    """Keeps every request, with the step and the run (and so the user) it was made for."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner
        self.name = inner.name
        self.step = 0
        self.captured: list[Captured] = []

    def complete(self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str):
        request = ModelRequest(
            purpose=purpose, schema_name=schema.__name__, messages=tuple(messages)
        )
        run = current_run.get()
        self.captured.append(
            Captured(self.step, run[1] if run else None, run[0] if run else None, request)
        )
        return self.inner.complete(messages, schema, purpose=purpose)


@dataclass
class RunRecord:
    step: int
    run_id: str
    kind: str  # chat | proactive | reconcile | provider_event
    user_id: str | None
    thread_id: str | None
    events: list[TraceEvent]

    @property
    def end(self) -> TraceEvent | None:
        return next((e for e in reversed(self.events) if e.kind == "run.end"), None)


@dataclass
class TurnRecord:
    """One user turn: who sent it, on which thread, and what came back."""

    step: Step
    user_id: str
    session_id: str
    thread_id: str
    run_id: str
    result: TurnResult | None = None
    error: str | None = None
    crashed: bool = False


@dataclass
class StepRecord:
    index: int  # 1-based, as checks refer to steps
    step: Step
    turns: list[TurnRecord] = field(default_factory=list)
    other_error: str | None = None  # a non-turn step that failed
    runs: list[RunRecord] = field(default_factory=list)
    memory_before: dict[str, dict] = field(default_factory=dict)  # fact id -> row
    memory_after: dict[str, dict] = field(default_factory=dict)
    context_token_cap: int = 0  # the cap in force during this step

    @property
    def result(self) -> TurnResult | None:
        return self.turns[0].result if len(self.turns) == 1 else None

    @property
    def thread_id(self) -> str | None:
        return self.turns[0].thread_id if self.turns else None

    @property
    def crashed(self) -> bool:
        return any(turn.crashed for turn in self.turns)

    @property
    def error(self) -> str | None:
        errors = [t.error for t in self.turns if t.error] + [self.other_error] * bool(
            self.other_error
        )
        return "; ".join(errors) or None


@dataclass
class Observation:
    scenario: EvalScenario
    settings: Settings
    store: Store
    world: MockWorld
    app: Any  # the ASGI app the turns went through, for API checks
    user_id: str  # the scenario's main user
    main_thread: str
    steps: list[StepRecord]
    initial_bookings: dict[str, Any]  # booking id -> Booking, every user's, before step 1
    aliases: dict[str, str]  # setup alias -> booking id
    captured: list[Captured]
    leftovers: dict[str, int]  # scripted replies and decisions never used
    drift: list[str]  # replayed model calls whose messages differ from the recording

    @property
    def requests(self) -> list[tuple[int, ModelRequest]]:
        return [(c.step, c.request) for c in self.captured]

    @property
    def runs(self) -> list[RunRecord]:
        return [run for step in self.steps for run in step.runs]

    @property
    def turns(self) -> list[TurnRecord]:
        return [turn for step in self.steps for turn in step.turns]

    @property
    def users(self) -> list[str]:
        seen = dict.fromkeys([self.user_id, *(t.user_id for t in self.turns)])
        seen.update(dict.fromkeys(r.user_id for r in self.runs if r.user_id))
        return list(seen)

    def events(self, kind: str | None = None) -> list[tuple[int, RunRecord, TraceEvent]]:
        """Every trace event in the order it happened, with a global position."""
        out, n = [], 0
        for run in self.runs:
            for event in run.events:
                if kind is None or event.kind == kind:
                    out.append((n, run, event))
                n += 1
        return out

    def tool_calls(self, tool: str | None = None, step: int | None = None, user: str | None = None):
        return [
            (n, run, e)
            for n, run, e in self.events("tool.call")
            if (tool is None or e.data["tool"] == tool)
            and (step is None or run.step == step)
            and (user is None or run.user_id == user)
        ]

    def outbox_thread(self, user: str | None = None) -> str | None:
        messages = self.store.outbox.since(user or self.user_id)
        return messages[-1].payload.get("thread_id") if messages else None

    def thread(self, which: str, user: str | None = None) -> str | None:
        user = user or self.user_id
        if which == "outbox":
            return self.outbox_thread(user)
        return thread_id_for(user, "main" if which == "main" else which)

    def bookings(self, user: str | None = None):
        return self.world.list_bookings(user or self.user_id).bookings

    def scan(self, sql: str, params: Sequence[Any] = ()) -> list:
        return self.store.db.scan(sql, params, reason=ORACLE)


class ScenarioRunner:
    """One scenario's environment: built on fresh files in `workdir`, played step by step."""

    def __init__(
        self,
        scenario: EvalScenario,
        *,
        workdir: Path,
        model: ModelClient,
        decider: DecisionClient,
        scenarios_dir: Path,
        seed: int | None = None,
    ) -> None:
        self.scenario = scenario
        self.clock = FakeClock(scenario.start)
        self.settings = Settings(
            _env_file=None,
            agent_db_path=workdir / "agent.db",
            world_db_path=workdir / "world.db",
            traces_dir=workdir / "traces",
            warmup_models=False,
            seed_demo_data=False,
            seed=scenario.seed if seed is None else seed,
            **scenario.settings,
        )
        world = load_scenario(scenario.world, scenarios_dir=scenarios_dir)
        self.world_scenario = Scenario.model_validate(
            {**world.model_dump(), **scenario.world_overrides}
        )
        self.model = CapturingModel(model)
        self.decider = decider
        self.user_id = scenario.user
        self.main_thread = thread_id_for(scenario.user, "main")
        self._open()

    def _open(self) -> None:
        self.store = Store(self.settings.agent_db_path, now=self.clock.now)
        self.world = MockWorld(
            self.world_scenario,
            db_path=self.settings.world_db_path,
            clock=self.clock,
            seed=self.settings.seed,
        )
        self.saver = open_checkpointer(self.settings.agent_db_path)
        self.agent = Agent(
            settings=self.settings,
            store=self.store,
            world=self.world,
            model=self.model,
            decider=self.decider,
            clock=self.clock,
            checkpointer=self.saver,
        )
        self.worker = Worker(self.agent, interval_s=self.settings.worker_poll_interval_s)
        services = Services(
            store=self.store, world=self.world, agent=self.agent, worker=self.worker
        )
        self.app = create_app(self.settings, services=services, start_worker=False)

    def close(self) -> None:
        self.saver.conn.close()
        self.store.close()
        self.world.close()

    # Setup

    def setup(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        if self.scenario.seed_demo:
            seed_demo(self.store, self.world, self.clock)
        memory = MemoryService(self.store)
        for fact in self.scenario.memories:
            user = fact.user or self.user_id
            record = memory.remember if fact.source == FactSource.EXPLICIT else memory.infer
            record(
                user, fact.key, fact.value, service=fact.service, evidence=fact.evidence or "setup"
            )
        for booking in self.scenario.bookings:
            seeded = self.world.seed_booking(
                user_id=booking.user or self.user_id,
                provider_id=booking.provider,
                slot=Slot(start=booking.start, end=booking.start + timedelta(hours=2)),
                status=BookingState(booking.status),
            )
            if booking.alias:
                aliases[booking.alias] = seeded.booking_id
        return aliases

    def _all_bookings(self) -> dict[str, Any]:
        users = {self.user_id, *SEED_USERS, *(b.user for b in self.scenario.bookings if b.user)}
        return {b.booking_id: b for u in users for b in self.world.list_bookings(u).bookings}

    # Playing

    def play(self) -> Observation:
        aliases = self.setup()
        initial = self._all_bookings()
        self._sessions: dict[str, str] = {}  # user -> the session they write on now
        self._last: dict[str, TurnResult] = {}  # thread -> the last reply on it
        records: list[StepRecord] = []
        self._records = records
        for index, step in enumerate(self.scenario.steps, start=1):
            record = StepRecord(index=index, step=step, memory_before=self._memory())
            seen = set(self._trace_files())
            self.model.step = index
            try:
                if step.turns:
                    self._play_turns(step, record)
                else:
                    self._do(step, aliases)
            except Exception as exc:  # a step must never raise: record it
                log.exception("step %d of %s failed", index, self.scenario.name)
                record.other_error = f"{type(exc).__name__}: {exc}"
            record.context_token_cap = self.settings.context_token_cap
            record.memory_after = self._memory()
            record.runs = self._new_runs(index, seen)
            records.append(record)
        return Observation(
            scenario=self.scenario,
            settings=self.settings,
            store=self.store,
            world=self.world,
            app=self.app,
            user_id=self.user_id,
            main_thread=self.main_thread,
            steps=records,
            initial_bookings=initial,
            aliases=aliases,
            captured=self.model.captured,
            leftovers=self._leftovers(),
            drift=list(getattr(self.model.inner, "drift", [])),
        )

    def _play_turns(self, step: Step, record: StepRecord) -> None:
        record.turns = [self._plan(turn, record.index, n) for n, turn in enumerate(step.turns, 1)]
        if step.gate:
            gate = Turnstile(
                [record.turns[int(o) - 1].run_id for o in step.gate.order]
                if "lock" in step.gate.at
                else [str(o) for o in step.gate.order]
            )
            for point in step.gate.at:
                if point == "lock":
                    self.agent.gate = gate
                else:
                    self.world.gates[point] = gate
        try:
            outcomes = asyncio.run(self._post_all(record.turns))
        finally:
            self.agent.gate = None
            self.world.gates.clear()
        for turn, outcome in zip(record.turns, outcomes, strict=True):
            if isinstance(outcome, TurnResult):
                turn.result = outcome
                self._last[turn.thread_id] = outcome
            elif isinstance(outcome, SimulatedCrash) and turn.step.expect_crash:
                turn.crashed = True  # the process "died"; a restart step picks it up
            else:
                turn.error = f"{type(outcome).__name__}: {outcome}"
            if turn.step.expect_crash and not turn.crashed:
                turn.error = "the turn was expected to crash, but it finished"

    def _plan(self, turn: Step, index: int, n: int) -> TurnRecord:
        user = turn.user or self.user_id
        if turn.thread:
            self._sessions[user] = (
                self._outbox_session(user) if turn.thread == "outbox" else "main"
            ) or self._sessions.get(user, "main")
        session = turn.session or self._sessions.get(user, "main")
        if turn.session_from:
            session = self._outbox_session(turn.session_from) or session
        thread = thread_id_for(user, session)
        return TurnRecord(
            step=turn,
            user_id=user,
            session_id=session,
            thread_id=thread,
            run_id=f"run_e{index}t{n}_{secrets.token_hex(4)}",
        )

    async def _post_all(self, turns: list[TurnRecord]) -> list[Any]:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval") as client:
            return await asyncio.gather(
                *(self._post(client, turn) for turn in turns), return_exceptions=True
            )

    async def _post(self, client: httpx.AsyncClient, turn: TurnRecord) -> TurnResult:
        body: dict[str, Any] = {
            "message": turn.step.text,
            "user_id": turn.user_id,
            "session_id": turn.session_id,
            "run_id": turn.run_id,
        }
        if turn.step.action == "confirm":
            shown = (
                self._earlier_result(turn.step.approval_from_step)
                if turn.step.approval_from_step
                else self._last.get(turn.thread_id)
            )
            if shown is None or not shown.approval:
                raise RuntimeError("confirm turn, but no approval is on screen")
            body["confirm"] = dict(shown.approval)
        response = await client.post("/chat", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"/chat answered {response.status_code}: {response.text[:200]}")
        data = response.json()
        data.pop("session_id", None)
        return TurnResult(**data)

    def _earlier_result(self, step_number: int) -> TurnResult | None:
        if step_number > len(self._records):
            return None
        record = self._records[step_number - 1]
        return record.turns[0].result if record.turns else None

    def _do(self, step: Step, aliases: dict[str, str]) -> None:
        action = step.action
        if action == "advance":
            self.clock.advance(step.advance)
        elif action == "provider_cancels":
            if provider := step.provider_cancels.get("provider"):
                self.world.provider_cancels_all(provider)
            else:
                self.world.provider_cancelled(self._select_booking(step.provider_cancels, aliases))
        elif action == "worker":
            self.worker.run_once()
        elif action == "restart":
            self.close()
            self._open()
            self.worker.run_once()  # the startup reconciler, as the app's lifespan runs it
        elif action == "configure":
            self.settings = Settings(
                _env_file=None, **{**self.settings.model_dump(), **step.configure}
            )
            self.agent.settings = self.settings

    def _select_booking(self, selector: dict[str, str], aliases: dict[str, str]) -> str:
        if "alias" in selector:
            return aliases[selector["alias"]]
        user = selector.get("user", self.user_id)
        upcoming = [
            b
            for b in self.world.list_bookings(user).bookings
            if b.status is BookingState.CONFIRMED
            and b.slot.start > self.clock.now()
            and b.service == selector["service"]
        ]
        if not upcoming:
            raise RuntimeError(f"no upcoming booking matches {selector}")
        return min(upcoming, key=lambda b: b.slot.start).booking_id

    def _outbox_session(self, user: str) -> str | None:
        messages = self.store.outbox.since(user)
        return messages[-1].payload.get("session_id") if messages else None

    def _memory(self) -> dict[str, dict]:
        rows = self.store.db.scan("SELECT * FROM memory_facts ORDER BY rowid", reason=ORACLE)
        return {row["fact_id"]: dict(row) for row in rows}

    def _trace_files(self) -> list[Path]:
        return list(self.settings.traces_dir.glob("*.jsonl"))

    def _new_runs(self, index: int, seen: set[Path]) -> list[RunRecord]:
        runs = []
        for path in set(self._trace_files()) - seen:
            events = read_trace(self.settings.traces_dir, path.stem)
            start = events[0].data if events else {}
            runs.append(
                RunRecord(
                    step=index,
                    run_id=path.stem,
                    kind=str(start.get("kind", "?")),
                    user_id=(events[0].user_id if events else None) or start.get("user_id"),
                    thread_id=start.get("thread_id"),
                    events=events,
                )
            )
        # Several runs in one step (a worker tick, a together group): in the order they started.
        return sorted(
            runs, key=lambda run: (run.events[0].ts.timestamp() if run.events else 0.0, run.run_id)
        )

    def _leftovers(self) -> dict[str, int]:
        out = {}
        for label, client in (("model", self.model.inner), ("decider", self.decider)):
            remaining = getattr(client, "remaining", 0)
            if remaining:
                out[label] = remaining
        return out


def write_trace(observation: Observation, path: Path) -> Path:
    """The scenario's trace: every run's events in order, one JSON object per line, each tagged
    with the step that caused it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for run in observation.runs:
            for event in run.events:
                line = {"step": run.step, **json.loads(event.model_dump_json())}
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def workspace() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="errand-eval-")
