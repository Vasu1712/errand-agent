"""Building an offline agent for tests: real graph, gateway and mock world; rule-based models."""

from datetime import datetime
from pathlib import Path

from errand.agent import Agent
from errand.config import Settings
from errand.contracts import IST
from errand.decide.rules import rule_decider
from errand.graph import open_checkpointer
from errand.llm.rules import rule_model
from errand.mocks import FakeClock, MockWorld, Scenario
from errand.store import Store
from errand.trace import read_trace

# A Tuesday morning.
START = datetime(2026, 11, 3, 8, 0, tzinfo=IST)


class Harness:
    def __init__(
        self, tmp_path: Path, *, model=None, decider=None, settings: dict | None = None, **scenario
    ):
        self.tmp_path = tmp_path
        self.clock = FakeClock(START)
        self.settings = Settings(
            _env_file=None,
            agent_db_path=tmp_path / "agent.db",
            world_db_path=tmp_path / "world.db",
            traces_dir=tmp_path / "traces",
            warmup_models=False,
            seed_demo_data=False,
            **(settings or {}),
        )
        self.scenario = Scenario(**scenario)
        self.model = model or rule_model()
        self.decider = decider or rule_decider()
        self.user_id = "u1"
        self.thread_id = f"{self.user_id}:test"
        self.start()

    def start(self) -> None:
        """(Re)open every component on the same files: a restart when called again."""
        self.store = Store(self.settings.agent_db_path, now=self.clock.now)
        self.world = MockWorld(
            self.scenario, db_path=self.settings.world_db_path, clock=self.clock, seed=42
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

    def stop(self) -> None:
        self.saver.conn.close()
        self.store.close()
        self.world.close()

    def say(self, message: str, **kwargs):
        self.last = self.agent.handle(
            user_id=self.user_id,
            thread_id=kwargs.pop("thread_id", self.thread_id),
            message=message,
            **kwargs,
        )
        return self.last

    def events(self, run_id: str | None = None, kind: str | None = None):
        events = read_trace(self.settings.traces_dir, run_id or self.last.run_id)
        return [e for e in events if kind is None or e.kind == kind]

    def all_events(self, kind: str | None = None):
        out = []
        for path in sorted(self.settings.traces_dir.glob("*.jsonl")):
            out += [
                e
                for e in read_trace(self.settings.traces_dir, path.stem)
                if kind is None or e.kind == kind
            ]
        return out

    def bookings(self):
        return self.world.list_bookings(self.user_id).bookings

    def book_through(
        self, first="Find a plumber in Koramangala this week, mornings only, under ₹1,500"
    ):
        """Search, pick option 1 and confirm. Returns the final turn."""
        self.say(first)
        self.say("1")
        return self.say("yes")
