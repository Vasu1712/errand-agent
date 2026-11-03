"""HTTP BFF: serves the chat UI (static/index.html) and the JSON endpoints it calls.

    POST   /chat                         one user message -> one agent run
    GET    /trace/{run_id}?user_id=      that run's trace events, only to the run's own user
    GET    /memory?user_id=              the user's stored facts (current ones, or every row)
    PATCH  /memory                       set a fact explicitly: add, correct or confirm it
    DELETE /memory?user_id=&key=         forget a fact (scrubbed from history too)
    GET    /outbox?user_id=              the user's proactive messages after a cursor
    GET    /demo/users                   the seeded demo users, for the UI's user picker
    POST   /simulate/provider_cancelled  demo hook: a provider cancels one of the user's bookings

Identity is an input: every endpoint needs the user's id, and a missing one is a 400, never a
default. A chat turn also names a session; its LangGraph thread is ``f"{user_id}:{session_id}"``,
so the same session id from two users is two threads. (There is no authentication: the id is
trusted as given, as a session token's subject would be.)

Handlers are async and run the agent, Vega, Ollama and SQLite work in the thread pool, so a slow
turn never blocks the event loop or another user's request. Vega is never called here.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, JsonValue, field_validator
from starlette.concurrency import run_in_threadpool

from errand import __version__
from errand.agent import Agent
from errand.clock import Clock, SystemClock
from errand.config import Settings, get_settings
from errand.contracts import BookingState
from errand.decide import DecisionClient, DecisionUnavailable, VegaDecider
from errand.decide.questions import QUESTIONS
from errand.graph import open_checkpointer
from errand.ids import ID_PATTERN, session_of, thread_id_for
from errand.llm import ChatMessage, ModelClient, ModelError
from errand.llm.schemas import ReplyText
from errand.memory import MemoryService
from errand.memory.seed import SEED_USERS, seed_demo
from errand.mocks import MockWorld, load_scenario
from errand.store import MemoryFact, OutboxMessage, Store
from errand.trace import TraceEvent, read_trace
from errand.worker import Worker

log = logging.getLogger(__name__)

FACT_KEY_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
SafeId = Annotated[str, Field(pattern=ID_PATTERN)]
FactKey = Annotated[str, Field(pattern=FACT_KEY_PATTERN)]
UserIdQuery = Annotated[str | None, Query(pattern=ID_PATTERN)]
FactKeyQuery = Annotated[str, Query(pattern=FACT_KEY_PATTERN)]


def required(value: str | None, name: str) -> str:
    """Identity is never defaulted: a request that doesn't say whose it is is refused."""
    if not value:
        raise HTTPException(400, f"{name} is required")
    return value


class Confirm(BaseModel):
    approval_id: SafeId
    material_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: SafeId | None = None  # required; checked by the handler so a missing one is a 400
    session_id: SafeId | None = None  # required, likewise
    # The UI may choose the run id itself, so it can poll the trace while the run is in progress.
    run_id: SafeId | None = None
    # Sent by the UI's Confirm button: what the user saw. Code checks it against the open approval.
    confirm: Confirm | None = None


class ChatResponse(BaseModel):
    run_id: str
    user_id: str
    session_id: str
    thread_id: str
    reply: str
    phase: str
    approval: dict | None = None
    options: list[dict] = []
    outcome: str


class TraceResponse(BaseModel):
    run_id: str
    complete: bool
    events: list[TraceEvent]


class MemoryList(BaseModel):
    user_id: str
    facts: list[MemoryFact]


class MemoryPatch(BaseModel):
    user_id: SafeId | None = None
    key: FactKey
    value: JsonValue
    service_tag: FactKey | None = None

    @field_validator("value")
    @classmethod
    def _not_null(cls, value: JsonValue) -> JsonValue:
        if value is None:
            raise ValueError("value can't be null; DELETE /memory forgets a fact")
        return value


class Forgotten(BaseModel):
    user_id: str
    key: str
    service_tag: str | None
    rows_forgotten: int


class OutboxList(BaseModel):
    user_id: str
    messages: list[OutboxMessage]


class SimulateCancel(BaseModel):
    user_id: SafeId | None = None
    booking_id: SafeId | None = None  # default: the user's next upcoming booking
    due_in_s: float = Field(0, ge=0, le=3600)


@dataclass
class Services:
    """Everything a running app holds. The eval harness builds its own and hands it in."""

    store: Store
    world: MockWorld
    agent: Agent
    worker: Worker


def warm_up(model: ModelClient, decider: DecisionClient) -> None:
    """Load Vega (and fit its heads) and wake Ollama, so the first turn doesn't pay for it."""
    if isinstance(decider, VegaDecider):
        log.info("loading Vega and fitting %d question heads", len(QUESTIONS))
        try:
            decider.load()
        except DecisionUnavailable as exc:
            # Serve anyway: each turn says the decision model isn't working, and loading is
            # tried again on the next decision (docs/bugs.md #20).
            log.warning("Vega didn't load (%s); turns will say so until it does", exc)
    try:
        model.complete([ChatMessage(role="user", content="Say hi.")], ReplyText, purpose="warmup")
    except ModelError as exc:
        log.warning("the model isn't answering yet (%s); turns will say so until it does", exc)


def create_app(
    settings: Settings | None = None,
    *,
    model: ModelClient | None = None,
    decider: DecisionClient | None = None,
    clock: Clock | None = None,
    start_worker: bool = True,
    services: Services | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:  # built and owned by the caller
            yield
            return
        now = clock or SystemClock()
        store = Store(settings.agent_db_path, now=now.now)
        scenario = load_scenario(settings.scenario, scenarios_dir=settings.scenarios_dir)
        world = MockWorld(scenario, db_path=settings.world_db_path, clock=now, seed=settings.seed)
        if settings.seed_demo_data and (seeded := seed_demo(store, world, now)):
            log.info("seeded demo data for %s", ", ".join(seeded))
        live_model = model
        if live_model is None:
            from errand.llm.ollama import OllamaModel

            live_model = OllamaModel.from_settings(settings)
        live_decider = decider or VegaDecider.from_settings(QUESTIONS, settings)
        if settings.warmup_models:
            await run_in_threadpool(warm_up, live_model, live_decider)
        checkpointer = open_checkpointer(settings.agent_db_path)
        agent = Agent(
            settings=settings,
            store=store,
            world=world,
            model=live_model,
            decider=live_decider,
            clock=now,
            checkpointer=checkpointer,
        )
        worker = Worker(agent, interval_s=settings.worker_poll_interval_s)
        startup = await run_in_threadpool(worker.run_once)  # settle rows a crash left open
        if startup["reconciled"]:
            log.info("startup reconciler settled %d ledger rows", startup["reconciled"])
        if start_worker:
            worker.start()
        app.state.services = Services(store=store, world=world, agent=agent, worker=worker)
        try:
            yield
        finally:
            worker.stop()
            if model is None and hasattr(live_model, "close"):
                live_model.close()
            checkpointer.conn.close()
            store.close()
            world.close()

    app = FastAPI(title="Errand Agent", version=__version__, lifespan=lifespan)
    if services is not None:
        app.state.services = services
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    def running(request: Request) -> Services:
        return request.app.state.services

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.static_dir / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/demo/users")
    def demo_users() -> dict[str, list[dict[str, str]]]:
        return {"users": [{"user_id": u, "name": name} for u, name in SEED_USERS.items()]}

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        user_id = required(body.user_id, "user_id")
        session_id = required(body.session_id, "session_id")
        thread_id = thread_id_for(user_id, session_id)
        agent = running(request).agent
        try:
            result = await run_in_threadpool(
                lambda: agent.handle(
                    user_id=user_id,
                    thread_id=thread_id,
                    message=body.message,
                    run_id=body.run_id,
                    confirm=body.confirm.model_dump() if body.confirm else None,
                )
            )
        except FileExistsError:
            raise HTTPException(409, f"run id {body.run_id} has already been used") from None
        return ChatResponse(
            **result.model_dump(exclude={"task_id", "private"}),
            session_id=session_of(result.thread_id),
        )

    @app.get("/trace/{run_id}")
    async def trace(run_id: str, request: Request, user_id: UserIdQuery = None) -> TraceResponse:
        user_id = required(user_id, "user_id")
        store = running(request).store
        try:
            owned = await run_in_threadpool(store.runs.get, run_id, user_id=user_id)
        except ValueError:
            raise HTTPException(400, "malformed run id") from None
        if owned is None:  # another user's run looks exactly like no run
            raise HTTPException(404, f"no trace for run {run_id}")
        try:
            events = await run_in_threadpool(read_trace, settings.traces_dir, run_id)
        except ValueError:
            raise HTTPException(400, "malformed run id") from None
        except FileNotFoundError:
            raise HTTPException(404, f"no trace for run {run_id}") from None
        complete = any(event.kind == "run.end" for event in events)
        return TraceResponse(run_id=run_id, complete=complete, events=events)

    @app.get("/memory")
    async def list_memory(
        request: Request, user_id: UserIdQuery = None, include_inactive: bool = False
    ) -> MemoryList:
        user_id = required(user_id, "user_id")
        store = running(request).store
        facts = await run_in_threadpool(
            store.memory.list_for_user, user_id, include_inactive=include_inactive
        )
        return MemoryList(user_id=user_id, facts=facts)

    @app.patch("/memory")
    async def set_memory(body: MemoryPatch, request: Request) -> MemoryFact:
        user_id = required(body.user_id, "user_id")
        service = MemoryService(running(request).store)
        return await run_in_threadpool(
            lambda: service.remember(
                user_id,
                body.key,
                body.value,
                service=body.service_tag,
                evidence="set by the user through PATCH /memory",
            )
        )

    @app.delete("/memory")
    async def forget_memory(
        request: Request,
        key: FactKeyQuery,
        user_id: UserIdQuery = None,
        service_tag: Annotated[str | None, Query(pattern=FACT_KEY_PATTERN)] = None,
    ) -> Forgotten:
        user_id = required(user_id, "user_id")
        service = MemoryService(running(request).store)
        rows = await run_in_threadpool(service.forget, user_id, key, service=service_tag)
        if not rows:
            raise HTTPException(404, f"nothing is stored under {key!r} for {user_id}")
        return Forgotten(user_id=user_id, key=key, service_tag=service_tag, rows_forgotten=rows)

    @app.get("/outbox")
    async def outbox(
        request: Request,
        user_id: UserIdQuery = None,
        after_id: Annotated[int, Query(ge=0)] = 0,
    ) -> OutboxList:
        user_id = required(user_id, "user_id")
        store = running(request).store
        messages = await run_in_threadpool(store.outbox.since, user_id, after_id)
        return OutboxList(user_id=user_id, messages=messages)

    @app.post("/simulate/provider_cancelled")
    async def simulate_provider_cancelled(body: SimulateCancel, request: Request) -> dict:
        user_id = required(body.user_id, "user_id")
        world = running(request).world

        def cancel() -> dict:
            mine = world.list_bookings(user_id).bookings
            upcoming = [
                b
                for b in mine
                if b.status is BookingState.CONFIRMED and b.slot.start > world.clock.now()
            ]
            booking_id = body.booking_id or (
                min(upcoming, key=lambda b: b.slot.start).booking_id if upcoming else None
            )
            if booking_id is None or booking_id not in {b.booking_id for b in mine}:
                raise HTTPException(404, "no such upcoming booking for this user")
            event = world.provider_cancelled(booking_id, due_in_s=body.due_in_s)
            return {"event": event.model_dump(mode="json")}

        return await run_in_threadpool(cancel)

    return app
