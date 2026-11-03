"""The agent runtime: one run per user message or proactive event, over the LangGraph graph.

A thread waiting at an interrupt (options shown, approval asked) is resumed from its SQLite
checkpoint with the new message; otherwise the message starts a fresh run at load_context. Runs
on one thread are serialized. Every run writes its own trace and ends with a reply, whatever fails.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import BaseModel

from errand.budget import RunBudget
from errand.clock import Clock
from errand.config import Settings
from errand.decide import DecisionClient
from errand.gateway import ToolGateway
from errand.graph import Deps, Phase, build_graph
from errand.ids import new_id, thread_id_for
from errand.llm import ModelClient
from errand.memory import ContextBuilder, MemoryService
from errand.mocks import MockWorld
from errand.store import OutboxMessage, Store
from errand.trace import EventKind, Tracer, current_run, new_run_id

log = logging.getLogger(__name__)

FALLBACK = (
    "I'm not sure how to help with that. I can book, reschedule or cancel a home service, or show"
    " what I remember about you."
)


class TurnResult(BaseModel):
    run_id: str
    thread_id: str
    user_id: str
    reply: str
    phase: str
    task_id: str | None = None
    approval: dict | None = None  # approval id and hash, while a confirmation is pending
    options: list[dict] = []
    outcome: str = "ok"  # ok | stopped | error
    private: bool = False  # the reply lists stored memory: never sent back to a model


class Agent:
    def __init__(
        self,
        *,
        settings: Settings,
        store: Store,
        world: MockWorld,
        model: ModelClient,
        decider: DecisionClient,
        clock: Clock,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self.settings = settings
        self.store = store
        self.world = world
        self.model = model
        self.decider = decider
        self.clock = clock
        self.graph = build_graph(checkpointer)
        # Per-thread locks: runs on one thread are serialized; different threads (different
        # users, or one user's sessions) run in parallel. No per-user state lives on the agent.
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        # A test hook (errand.mocks.Turnstile): pins the order concurrent turns take a thread's
        # lock, so a concurrency eval is deterministic. None in the app.
        self.gate: Any = None

    # Running

    def handle(
        self,
        *,
        user_id: str,
        thread_id: str,
        message: str,
        run_id: str | None = None,
        confirm: dict | None = None,
    ) -> TurnResult:
        run_id = run_id or new_run_id()
        if not thread_id.startswith(f"{user_id}:"):
            # A thread is one user's session (errand.ids.thread_id_for): never another user's.
            raise ValueError(f"thread {thread_id!r} doesn't belong to {user_id!r}")
        lock = self.thread_lock(thread_id)
        if self.gate is not None:
            with self.gate.turn(run_id):  # tests pin which concurrent turn takes the lock first
                lock.acquire()
        else:
            lock.acquire()
        try:
            return self._handle(user_id, thread_id, message, run_id, confirm)
        finally:
            lock.release()

    def _handle(
        self, user_id: str, thread_id: str, message: str, run_id: str, confirm: dict | None
    ) -> TurnResult:
        with self.tracer(run_id, user_id=user_id, thread_id=thread_id, kind="chat") as tracer:
            deps = self.deps(tracer, user_id)
            tracer.emit(
                EventKind.RUN_START,
                kind="chat",
                thread_id=thread_id,
                user_id=user_id,
                message=message,
                confirm=confirm,
                model=self.model.name,
                budgets=self._budgets(),
            )
            started = perf_counter()
            stored = self.store.messages.add(user_id, thread_id, "user", message, run_id=run_id)
            config = {"configurable": {"thread_id": thread_id}}
            if self.graph.get_state(config).interrupts:
                payload: Any = Command(
                    resume={
                        "message": message,
                        "message_id": stored.message_id,
                        "confirm": confirm,
                        "run_id": run_id,
                    }
                )
            else:
                payload = {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "run_id": run_id,
                    "user_message": message,
                    "message_id": stored.message_id,
                    "confirm": confirm,
                    "proactive": None,
                }
            result = self._invoke(payload, config, deps)
            self.store.messages.add(
                user_id,
                thread_id,
                "assistant",
                result.reply,
                run_id=run_id,
                for_model=not result.private,
            )
            tracer.emit(
                EventKind.RUN_END,
                latency_ms=(perf_counter() - started) * 1000,
                outcome=result.outcome,
                phase=result.phase,
                reply=result.reply,
                reply_for_model=not result.private,
                budget=deps.budget.snapshot(),
            )
            return result

    def open_replacement(
        self,
        *,
        user_id: str,
        intro: str,
        slots: dict,
        slot_sources: dict,
        excluded_providers: list[str],
        origin: dict,
        dedupe_key: str,
    ) -> OutboxMessage:
        """A worker-opened task: search and rank replacements (nothing is held), then write the
        options to the outbox on a new session of the user's, which they can reply on."""
        session_id = new_id("ses")
        thread_id = thread_id_for(user_id, session_id)
        run_id = new_run_id()
        with (
            self.thread_lock(thread_id),
            self.tracer(run_id, user_id=user_id, thread_id=thread_id, kind="proactive") as tracer,
        ):
            budget = self.budget(timeout_s=self.settings.background_timeout_s)
            deps = self.deps(tracer, user_id, budget)
            tracer.emit(
                EventKind.RUN_START,
                kind="proactive",
                thread_id=thread_id,
                user_id=user_id,
                origin=origin,
                slots=slots,
                slot_sources=slot_sources,
            )
            started = perf_counter()
            payload = {
                "thread_id": thread_id,
                "user_id": user_id,
                "run_id": run_id,
                "user_message": "",
                "proactive": origin,
                "intent": "book",
                "slots": slots,
                "slot_sources": slot_sources,
                "excluded_providers": excluded_providers,
            }
            result = self._invoke(payload, {"configurable": {"thread_id": thread_id}}, deps)
            body = f"{intro}\n\n{result.reply}"
            self.store.messages.add(user_id, thread_id, "assistant", body, run_id=run_id)
            message, _ = self.store.outbox.enqueue(
                user_id,
                "provider_cancelled",
                body,
                task_id=result.task_id,
                payload={
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "run_id": run_id,
                    "options": result.options,
                },
                dedupe_key=dedupe_key,
            )
            tracer.emit(
                EventKind.RUN_END,
                latency_ms=(perf_counter() - started) * 1000,
                outcome=result.outcome,
                phase=result.phase,
                reply=body,
                outbox_message_id=message.message_id,
                budget=deps.budget.snapshot(),
            )
            return message

    def budget(self, *, timeout_s: float | None = None) -> RunBudget:
        """A run's budget, timed on the agent's clock (so a fake clock's simulated waits count)."""
        overrides = {} if timeout_s is None else {"timeout_s": timeout_s}
        return RunBudget.from_settings(self.settings, monotonic=self.clock.monotonic, **overrides)

    def deps(self, tracer: Tracer, user_id: str, budget: RunBudget | None = None) -> Deps:
        budget = budget or self.budget()
        return Deps(
            settings=self.settings,
            store=self.store,
            gateway=self.gateway(tracer, user_id, budget),
            model=self.model,
            decider=self.decider,
            memory=MemoryService(self.store, tracer),
            contexts=ContextBuilder(
                self.store,
                token_cap=self.settings.context_token_cap,
                recent_turns=self.settings.recent_turns,
                summary_token_budget=self.settings.summary_token_budget,
            ),
            tracer=tracer,
            budget=budget,
            clock=self.clock,
        )

    def gateway(self, tracer: Tracer, user_id: str, budget: RunBudget) -> ToolGateway:
        return ToolGateway(
            self.world.for_user(user_id),
            self.store,
            tracer,
            budget,
            user_id=user_id,
            clock=self.clock,
            backoff_s=self.settings.status_backoff_s,
            max_book_attempts=self.settings.max_book_attempts,
            max_cancel_attempts=self.settings.max_cancel_attempts,
        )

    def thread_lock(self, thread_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(thread_id, threading.Lock())

    # Internals

    @contextmanager
    def tracer(
        self, run_id: str, *, user_id: str, thread_id: str | None, kind: str
    ) -> Iterator[Tracer]:
        """A run's tracer. The run is registered to its user first, so its trace is only ever
        served to that user."""
        self.store.runs.register(run_id, user_id=user_id, thread_id=thread_id, kind=kind)
        token = current_run.set((run_id, user_id))
        try:
            with Tracer(
                run_id,
                self.settings.traces_dir,
                max_str=self.settings.trace_max_str_chars,
                user_id=user_id,
            ) as tracer:
                yield tracer
        finally:
            current_run.reset(token)

    def _budgets(self) -> dict[str, float]:
        s = self.settings
        return {
            "max_model_calls": s.max_model_calls,
            "max_decisions": s.max_decisions,
            "max_tool_calls": s.max_tool_calls,
            "reserved_tool_calls": s.reserved_tool_calls,
            "turn_timeout_s": s.turn_timeout_s,
            "context_token_cap": s.context_token_cap,
        }

    def _invoke(self, payload: Any, config: dict, deps: Deps) -> TurnResult:
        ids = {
            "run_id": deps.tracer.run_id,
            "thread_id": config["configurable"]["thread_id"],
            "user_id": deps.gateway.user_id,
        }
        try:
            out = self.graph.invoke(payload, config, context=deps, durability="sync")
        except Exception as exc:
            # The node boundaries catch failures inside nodes; this is anything else.
            log.exception("run %s failed outside the nodes", ids["run_id"])
            deps.tracer.emit(EventKind.ERROR, where="graph", error=exc)
            values = self.graph.get_state(config).values
            return TurnResult(
                **ids,
                reply=(
                    "Something went wrong on my side, so I stopped. Nothing was booked, cancelled"
                    f" or changed. The details are in trace {ids['run_id']}."
                ),
                phase=values.get("phase", Phase.IDLE),
                task_id=values.get("task_id"),
                outcome="error",
            )
        interrupts = out.get("__interrupt__") or []
        approval, options = None, []
        if interrupts:
            value = interrupts[0].value
            reply, approval, options = (
                value["reply"],
                value.get("approval"),
                value.get("options", []),
            )
        else:
            reply = out.get("reply") or FALLBACK
        if notice := out.get("notice"):
            reply = f"{notice}\n\n{reply}"
        return TurnResult(
            **ids,
            reply=reply,
            phase=out.get("phase", Phase.IDLE),
            task_id=out.get("task_id"),
            approval=approval,
            options=options,
            outcome="stopped" if out.get("stop") else "ok",
            private=bool(out.get("reply_private")),
        )
