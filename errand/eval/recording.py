"""Recordings: a live run's model replies and decisions, saved to replay a scenario offline.

``python -m errand.eval --record <scenario>`` runs the scenario against live Ollama and Vega and
writes evals/recordings/<scenario>.json. A scenario with ``recording: <scenario>`` then replays it.
"""

from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import AwareDatetime, BaseModel

from errand.decide.replay import DecisionCall, ReplayDecider
from errand.llm.replay import ModelCall, ReplayModel


class Recording(BaseModel):
    scenario: str
    recorded_at: AwareDatetime
    model: str
    decider: str
    model_calls: list[ModelCall]
    decisions: list[DecisionCall]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def replay_model(self) -> ReplayModel:
        return ReplayModel(self.model_calls, name=f"replay:{self.model}")

    def replay_decider(self) -> ReplayDecider:
        return ReplayDecider(self.decisions)


def recording_path(evals_dir: Path, name: str) -> Path:
    return evals_dir / "recordings" / f"{name}.json"


def new_recording(scenario: str, model, decider) -> Recording:
    """From a finished live run's `RecordingModel` and `RecordingDecider`."""
    return Recording(
        scenario=scenario,
        recorded_at=datetime.now().astimezone(),
        model=model.inner.name,
        decider=type(decider.inner).__name__,
        model_calls=model.calls,
        decisions=decider.calls,
    )
