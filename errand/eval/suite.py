"""Runs scenarios and judges them: expected outcomes, automatic checks and the global invariants."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from errand.config import Settings
from errand.decide import DecisionClient
from errand.eval.checks import CheckResult, automatic, run_check
from errand.eval.invariants import InvariantResult, check_invariants
from errand.eval.recording import Recording, new_recording, recording_path
from errand.eval.runner import Observation, ScenarioRunner, workspace, write_trace
from errand.eval.spec import EvalScenario, scripted_decider, scripted_model
from errand.llm import ModelClient


@dataclass
class ScenarioResult:
    name: str
    title: str
    covers: list[str]
    mode: str  # scripted | replay | live
    checks: list[CheckResult]
    invariants: list[InvariantResult]
    trace: str
    stats: dict[str, int] = field(default_factory=dict)
    drift: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and all(i.passed for i in self.invariants)


def offline_clients(
    scenario: EvalScenario, settings: Settings
) -> tuple[ModelClient, DecisionClient, str]:
    """The scenario's scripted model and decider, or the recording it names."""
    if scenario.recording:
        recording = Recording.load(recording_path(settings.evals_dir, scenario.recording))
        return recording.replay_model(), recording.replay_decider(), "replay"
    return scripted_model(scenario), scripted_decider(scenario), "scripted"


def run_scenario(
    scenario: EvalScenario,
    settings: Settings,
    *,
    model: ModelClient | None = None,
    decider: DecisionClient | None = None,
    mode: str | None = None,
    seed: int | None = None,
) -> ScenarioResult:
    """Play and judge one scenario. `seed` overrides the scenario's (the stress loop): the mock
    world's catalogue and outcomes change, and its trace is written under a seed suffix."""
    if model is None or decider is None:
        model, decider, mode = offline_clients(scenario, settings)
    started = time.perf_counter()
    with workspace() as tmp:
        runner = ScenarioRunner(
            scenario,
            workdir=Path(tmp),
            model=model,
            decider=decider,
            scenarios_dir=settings.scenarios_dir,
            seed=seed,
        )
        try:
            obs = runner.play()
            result = judge(obs, mode or "live")
            suffix = ".live" if result.mode == "live" else ""
            if seed is not None:
                suffix += f".seed{seed}"
            out = settings.eval_traces_dir / ("stress" if seed is not None else "")
            result.trace = str(write_trace(obs, out / f"{scenario.name}{suffix}.jsonl"))
        finally:
            runner.close()
    result.duration_s = round(time.perf_counter() - started, 2)
    return result


def judge(obs: Observation, mode: str) -> ScenarioResult:
    scenario = obs.scenario
    checks = [run_check(obs, c.check, c.params) for c in scenario.expect] + automatic(obs)
    return ScenarioResult(
        name=scenario.name,
        title=scenario.title,
        covers=scenario.covers,
        mode=mode,
        checks=checks,
        invariants=check_invariants(obs),
        trace="",
        stats={
            "steps": len(obs.steps),
            "runs": len(obs.runs),
            "tool_calls": len(obs.tool_calls()),
            "model_calls": len(obs.requests),
            "decisions": len(obs.events("decision")),
        },
        drift=obs.drift,
    )


def record_scenarios(
    scenarios: list[EvalScenario], settings: Settings
) -> Iterator[tuple[ScenarioResult, Path]]:
    """Run each scenario against live Ollama and Vega (loaded once) and save their outputs for
    offline replay, as evals/recordings/<scenario>.json."""
    from errand.decide import VegaDecider
    from errand.decide.questions import QUESTIONS
    from errand.decide.replay import RecordingDecider
    from errand.llm.ollama import OllamaModel
    from errand.llm.replay import RecordingModel

    vega = VegaDecider.from_settings(QUESTIONS, settings)
    vega.load()
    live = OllamaModel.from_settings(settings)
    try:
        for scenario in scenarios:
            model, decider = RecordingModel(live), RecordingDecider(vega)
            result = run_scenario(scenario, settings, model=model, decider=decider, mode="live")
            path = new_recording(scenario.name, model, decider).save(
                recording_path(settings.evals_dir, scenario.name)
            )
            yield result, path
    finally:
        live.close()
