"""What graph nodes get to work with, per run (LangGraph runtime context, never checkpointed).

`ask_model` and `decide` are the only ways a node reaches a model, so every call is budgeted,
built by the context builder (with its manifest traced) and logged.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TypeVar

from pydantic import BaseModel

from errand.budget import RunBudget
from errand.clock import Clock
from errand.config import Settings
from errand.contracts import IST
from errand.decide import Decision, DecisionClient, Question
from errand.gateway import ToolGateway
from errand.graph.state import AgentState
from errand.llm import ModelClient
from errand.memory import ContextBuilder, MemoryService
from errand.store import MemoryFact, Store
from errand.trace import EventKind, Tracer

T = TypeVar("T", bound=BaseModel)


@dataclass
class Deps:
    settings: Settings
    store: Store
    gateway: ToolGateway
    model: ModelClient
    decider: DecisionClient
    memory: MemoryService
    contexts: ContextBuilder
    tracer: Tracer
    budget: RunBudget
    clock: Clock

    @property
    def threshold(self) -> float:
        return self.settings.vega_confidence_threshold

    def today(self) -> date:
        return self.clock.now().astimezone(IST).date()

    def ask_model(
        self,
        purpose: str,
        schema: type[T],
        *,
        state: AgentState,
        system: str,
        facts: Sequence[MemoryFact] = (),
        data: str | None = None,
        recent: bool = True,
        summary: bool = False,
    ) -> T:
        self.budget.spend_model()
        pack = self.contexts.build(
            purpose=purpose,
            system=system,
            latest=state.user_message,
            user_id=state.user_id,
            thread_id=state.thread_id,
            latest_message_id=state.message_id,
            facts=facts,
            data=data,
            recent=recent,
            summary=summary,
        )
        self.tracer.emit(EventKind.CONTEXT, **pack.manifest.model_dump())
        with self.tracer.span(EventKind.MODEL_CALL, purpose=purpose, model=self.model.name) as span:
            reply = self.model.complete(pack.messages, schema, purpose=purpose)
            span.update(
                input_tokens=reply.usage.input_tokens,
                output_tokens=reply.usage.output_tokens,
                cost_usd=reply.usage.cost_usd(
                    self.settings.llm_usd_per_1k_input_tokens,
                    self.settings.llm_usd_per_1k_output_tokens,
                ),
                model_latency_ms=round(reply.latency_ms, 1),
                output=reply.output,
            )
        return reply.output

    def decide(self, text: str, question: Question) -> Decision:
        return self.decide_many(text, [question])[question.id]

    def decide_many(self, text: str, questions: Sequence[Question]) -> dict[str, Decision]:
        self.budget.spend_decision()
        decisions = self.decider.decide_many(text, questions)
        for question in questions:
            decision = decisions[question.id]
            self.tracer.emit(
                EventKind.DECISION,
                question=question.id,
                choice=decision.choice,
                confidence=round(decision.confidence, 3),
                abstain=decision.abstain,
                readout=decision.readout,
                latency_ms=round(decision.latency_ms, 1),
                text=text[:200],
            )
        return decisions
