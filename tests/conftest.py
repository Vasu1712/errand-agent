"""Shared fixtures. Tests run offline: any outbound network connection fails the test."""

import itertools
import os
import socket
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from errand.api import create_app
from errand.config import Settings
from errand.decide.rules import rule_decider
from errand.llm.rules import rule_model
from errand.mocks import FakeClock
from errand.store import Store
from tests.support import START

# Nothing in a test may report to LangSmith, whatever the developer's shell says.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse IPv4/IPv6 connections, loopback included, so a running Ollama can't be reached by
    accident. RuntimeError rather than OSError, so no retry or fallback path can swallow it. Tests
    marked ``live`` are exempt."""
    if request.node.get_closest_marker("live"):
        return
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.close()  # callers only clean up after OSError, so don't leak the socket
            raise RuntimeError(f"network access attempted in a test: {address!r}")

    def connect(self: socket.socket, address: object) -> None:
        refuse(self, address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        refuse(self, address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's ERRAND_* variables must not change test outcomes."""
    for name in list(os.environ):
        if name.startswith("ERRAND_"):
            monkeypatch.delenv(name)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Every writable path inside tmp_path; the static page and scenarios are the real ones."""
    return Settings(
        _env_file=None,
        agent_db_path=tmp_path / "agent.db",
        world_db_path=tmp_path / "world.db",
        traces_dir=tmp_path / "traces",
        reports_dir=tmp_path / "reports",
        warmup_models=False,
        seed_demo_data=False,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[Store]:
    """A store whose clock ticks one second per timestamp, so row order is deterministic."""
    start = datetime(2026, 11, 3, 9, 0, tzinfo=UTC)
    ticks = itertools.count()
    with Store(settings.agent_db_path, now=lambda: start + timedelta(seconds=next(ticks))) as s:
        yield s


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """The real app on the rule-based model and decider, a fake clock, and no worker thread."""
    app = create_app(
        settings,
        model=rule_model(),
        decider=rule_decider(),
        clock=FakeClock(START),
        start_worker=False,
    )
    with TestClient(app) as test_client:
        yield test_client
