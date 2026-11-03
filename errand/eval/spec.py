"""Eval scenarios: evals/scenarios/*.yaml, one simulated user each.

A scenario is the user's scripted turns (and clock steps, provider events, worker ticks and
restarts), what the model and decider answer (scripted outputs, or a recording of a live run),
a mock-world scenario from scenarios/, a seed, and the outcomes to check. evals/defaults.yaml
supplies decider rules every scenario shares (rating reviews); a scenario's own entries win.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from errand.contracts import IST
from errand.decide.base import DecisionUnavailable
from errand.decide.scripted import ScriptedDecider
from errand.llm.base import ModelOutputError, ModelUnavailable
from errand.llm.scripted import ScriptedModel

START = datetime(2026, 11, 3, 8, 0, tzinfo=IST)  # a Tuesday morning
ACTIONS = (
    "say",
    "confirm",
    "together",
    "advance",
    "provider_cancels",
    "worker",
    "restart",
    "configure",
)


class Gate(BaseModel):
    """Pins the order of a `together` group: at world points (``hold_slot``, ``book``,
    ``cancel``) by user id, or at ``lock`` (taking the thread's lock) by turn number (1-based)."""

    model_config = ConfigDict(extra="forbid")

    at: list[Literal["hold_slot", "book", "cancel", "lock"]]
    order: list[str | int]


class Step(BaseModel):
    """One thing that happens: a user says something or presses Confirm (alone, or with others at
    the same moment), time passes, a provider cancels, the worker ticks, the process restarts, or
    the app's limits are changed."""

    model_config = ConfigDict(extra="forbid")

    say: str | None = None
    # The UI's Confirm button: this text, plus the id and hash of the approval on screen.
    confirm: str | None = None
    # Who says it (default: the scenario's user), and in which of their sessions (default
    # "main"). The thread is f"{user}:{session}".
    user: str | None = None
    session: str | None = None
    # Press the Confirm button shown after an earlier step instead (a stale screen, a double
    # submit, another user's screen): that step's approval id and hash.
    approval_from_step: int | None = Field(None, ge=1)
    # Which session the user writes on from now on: their own, or their latest outbox message's.
    thread: Literal["main", "outbox"] | None = None
    # Use this other user's latest outbox session id (as this step's own user: a hijack attempt).
    session_from: str | None = None
    # The process is killed during this step (a forced crash in the mock world).
    expect_crash: bool = False
    # Several turns fired at the same moment, on real threads through the ASGI app.
    together: list["Step"] | None = None
    gate: Gate | None = None
    advance: float | None = Field(None, gt=0)  # seconds on the fake clock
    # {service: ...} or {alias: ...} for one booking; {provider: ...} for all of a provider's
    provider_cancels: dict[str, str] | None = None
    worker: Literal["tick"] | None = None
    restart: bool | None = None
    configure: dict[str, Any] | None = None  # settings for the runs from here on

    @model_validator(mode="after")
    def _one_action(self) -> Self:
        chosen = [name for name in ACTIONS if getattr(self, name) not in (None, False)]
        if len(chosen) != 1:
            raise ValueError(f"a step does exactly one of {ACTIONS}, got {chosen or 'none'}")
        if self.approval_from_step and self.confirm is None:
            raise ValueError("approval_from_step goes with confirm")
        if self.together is not None and any(
            s.action not in ("say", "confirm") for s in self.together
        ):
            raise ValueError("a together group holds only say and confirm turns")
        if self.gate is not None and self.together is None:
            raise ValueError("gate goes with together")
        return self

    @property
    def turns(self) -> list["Step"]:
        """The user turns this step makes: itself, or its group."""
        if self.together is not None:
            return self.together
        return [self] if self.action in ("say", "confirm") else []

    @property
    def action(self) -> str:
        return next(name for name in ACTIONS if getattr(self, name) not in (None, False))

    @property
    def text(self) -> str | None:
        return self.say if self.say is not None else self.confirm

    def describe(self) -> str:
        if self.text is not None:
            return f"{self.action} {self.text!r}"
        return f"{self.action} {getattr(self, self.action)}"


class FactSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str | None = None  # default: the scenario's user
    key: str
    value: JsonValue
    source: Literal["explicit", "inferred"] = "explicit"
    service: str | None = None
    evidence: str | None = None


class BookingSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str | None = None
    user: str | None = None  # default: the scenario's user
    provider: str
    start: datetime  # IST when no offset is given
    status: Literal["confirmed", "completed", "cancelled"] = "confirmed"

    @model_validator(mode="after")
    def _local_time(self) -> Self:
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=IST)
        return self


class Check(BaseModel):
    """One expected outcome: ``check`` names it (see errand.eval.checks), the rest are its
    parameters."""

    model_config = ConfigDict(extra="allow")

    check: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class EvalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    covers: list[str] = []
    world: str  # a mock-world scenario: scenarios/<world>.yaml
    world_overrides: dict[str, Any] = {}
    seed: int = 42
    start: datetime = START
    user: str = "u_eval"
    # The three demo users (u_demo with 60+ memories and a long history, u_asha, u_ravi).
    seed_demo: bool = False
    memories: list[FactSetup] = []
    bookings: list[BookingSetup] = []
    settings: dict[str, Any] = {}
    # purpose -> replies in order, for the scripted model
    model: dict[str, list[dict[str, Any]]] = {}
    # question id -> labels in order, or {default, match, confidence} for a rule
    decider: dict[str, Any] = {}
    # A recording (evals/recordings/<name>.json) to replay instead of the scripts above.
    recording: str | None = None
    steps: list[Step]
    expect: list[Check]
    source: Path | None = None


def load_scenarios(evals_dir: Path) -> list[EvalScenario]:
    """Every scenario in ``evals_dir/scenarios``, in file-name order, with defaults applied and
    ``extends`` resolved."""
    defaults = _read(evals_dir / "defaults.yaml") if (evals_dir / "defaults.yaml").is_file() else {}
    raw = {path: _read(path) for path in sorted((evals_dir / "scenarios").glob("*.yaml"))}
    by_name = {data.get("name"): data for data in raw.values()}
    scenarios = []
    for path, data in raw.items():
        if base := data.pop("extends", None):
            if base not in by_name:
                raise ValueError(f"{path.name} extends unknown scenario {base!r}")
            data = {**{k: v for k, v in by_name[base].items() if k != "extends"}, **data}
        decider = {**defaults.get("decider", {}), **data.get("decider", {})}
        scenario = EvalScenario.model_validate({**data, "decider": decider, "source": path})
        scenarios.append(scenario)
    names = [s.name for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(
            f"duplicate scenario names: {sorted(n for n in names if names.count(n) > 1)}"
        )
    return scenarios


def _read(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# Scripted model and decider


def scripted_model(scenario: EvalScenario) -> ScriptedModel:
    """Replies by purpose. ``{error: unavailable}`` is Ollama being down for that call, and
    ``{error: malformed}`` a reply that isn't valid JSON for the schema."""
    return ScriptedModel(
        {
            purpose: [_model_step(reply) for reply in replies]
            for purpose, replies in scenario.model.items()
        },
        name="scripted",
    )


def _model_step(reply: dict[str, Any]) -> Any:
    error = reply.get("error") if set(reply) == {"error"} else None
    if error == "unavailable":
        return ModelUnavailable("qwen2.5:7b on Ollama failed: connection refused (scripted)")
    if error == "malformed":
        return ModelOutputError("the reply isn't valid JSON (scripted)", raw='{"service": ')
    if error is not None:
        raise ValueError(f"unknown scripted model error {error!r}")
    return reply


def scripted_decider(scenario: EvalScenario) -> ScriptedDecider:
    """Labels in order, or a rule. ``{error: unavailable}`` is Vega failing for that call."""
    script: dict[str, Any] = {}
    for question_id, entry in scenario.decider.items():
        if isinstance(entry, dict):
            script[question_id] = _rule(question_id, entry)
        else:
            script[question_id] = [_outcome(question_id, item) for item in entry]
    return ScriptedDecider(script, default_confidence=0.97)


def _outcome(question_id: str, item: Any) -> Any:
    if isinstance(item, dict) and item.get("error") == "unavailable":
        return DecisionUnavailable("Vega could not be loaded: weights missing (scripted)")
    if isinstance(item, dict):
        return (_label(question_id, item["choice"]), float(item.get("confidence", 0.97)))
    return _label(question_id, item)


def _label(question_id: str, value: Any) -> str:
    if not isinstance(value, str):
        # YAML reads a bare yes/no as a boolean; labels must be quoted.
        raise ValueError(
            f"decider label for {question_id!r} must be a quoted string, got {value!r}"
        )
    return value


def _rule(question_id: str, entry: dict[str, Any]) -> Callable[[str], tuple[str, float]]:
    """``{default: label, match: [{contains: text, choice: label}], confidence: 0.9}``: the first
    match whose text appears in the decided-on text wins, otherwise the default."""
    confidence = float(entry.get("confidence", 0.95))
    matches = [
        (m["contains"], _label(question_id, m["choice"]), float(m.get("confidence", confidence)))
        for m in entry.get("match", [])
    ]
    default = _label(question_id, entry["default"])

    def rule(text: str) -> tuple[str, float]:
        for needle, choice, conf in matches:
            if needle in text:
                return choice, conf
        return default, confidence

    return rule
