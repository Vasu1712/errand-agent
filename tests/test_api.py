import asyncio
import threading

import httpx

from errand.contracts import BookingState
from errand.mocks import BookOutcome

SEARCH = "Find a plumber in Koramangala this week, mornings only, under ₹1,500"
ME = {"user_id": "u_api", "session_id": "s1"}


def chat(client, message, **extra):
    response = client.post("/chat", json={"message": message, **ME, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def force_success(client):
    with client.app.state.services.world._db.transaction() as conn:
        conn.execute(
            "INSERT INTO forced (tool, outcome) VALUES ('book', ?)", (BookOutcome.SUCCESS.value,)
        )


def test_index_serves_the_chat_and_trace_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="chat-log"' in response.text and 'id="trace-events"' in response.text
    assert 'id="user-picker"' in response.text
    assert client.get("/healthz").json() == {"status": "ok"}
    users = [u["user_id"] for u in client.get("/demo/users").json()["users"]]
    assert users == ["u_demo", "u_asha", "u_ravi"]


def test_identity_is_required_never_defaulted(client):
    for body in ({"message": "hi"}, {"message": "hi", "user_id": "u_api"}):
        assert client.post("/chat", json=body).status_code == 400
    assert client.post("/chat", json={"message": "hi", "session_id": "s1"}).status_code == 400
    for path in ("/memory", "/outbox", "/trace/run_x"):
        assert client.get(path).status_code == 400
    assert client.patch("/memory", json={"key": "preferred_area", "value": "x"}).status_code == 400
    assert client.delete("/memory", params={"key": "preferred_area"}).status_code == 400
    assert client.post("/simulate/provider_cancelled", json={}).status_code == 400


def test_a_booking_through_the_api_with_the_confirm_button(client):
    force_success(client)
    first = chat(client, SEARCH)
    assert first["thread_id"] == "u_api:s1" and first["session_id"] == "s1"
    assert first["phase"] == "presenting" and len(first["options"]) == 3

    summary = chat(client, "1")
    assert summary["phase"] == "awaiting_approval" and summary["approval"]

    stale = {"approval_id": summary["approval"]["approval_id"], "material_hash": "0" * 64}
    refused = chat(client, "Yes, book it.", confirm=stale)
    assert refused["phase"] == "awaiting_approval" and "earlier proposal" in refused["reply"]

    done = chat(client, "Yes, book it.", confirm=refused["approval"])
    assert done["phase"] == "done" and done["reply"].startswith("Booked. Confirmation")
    world = client.app.state.services.world
    assert [b.status for b in world.list_bookings("u_api").bookings] == [BookingState.CONFIRMED]

    trace = client.get(f"/trace/{done['run_id']}", params={"user_id": "u_api"}).json()
    kinds = [event["kind"] for event in trace["events"]]
    assert trace["complete"] and kinds[0] == "run.start" and kinds[-1] == "run.end"
    assert {event["user_id"] for event in trace["events"]} == {"u_api"}
    for kind in ("decision", "approval", "ledger", "tool.call", "state.transition"):
        assert kind in kinds


def test_one_users_trace_memory_and_outbox_are_invisible_to_another(client):
    mine = chat(client, "hi")
    client.patch(
        "/memory", json={"user_id": "u_api", "key": "preferred_area", "value": "HSR Layout"}
    )
    other = {"user_id": "u_other"}
    assert client.get(f"/trace/{mine['run_id']}", params=other).status_code == 404
    assert client.get("/memory", params=other).json()["facts"] == []
    assert client.get("/outbox", params=other).json()["messages"] == []
    # The same session id from another user is another thread, with none of this one's state.
    theirs = chat(client, "hi", user_id="u_other")
    assert theirs["thread_id"] == "u_other:s1" != mine["thread_id"]


def test_client_chosen_run_ids_are_validated_and_single_use(client):
    first = chat(client, "hi", run_id="run_mine_1")
    assert first["run_id"] == "run_mine_1"
    body = {"message": "hi", "run_id": "run_mine_1", **ME}
    assert client.post("/chat", json=body).status_code == 409
    assert client.post("/chat", json={**body, "run_id": "../escape"}).status_code == 422
    assert client.post("/chat", json={**ME, "message": ""}).status_code == 422
    assert (
        client.post(
            "/chat", json={"message": "hi", "user_id": "u:x", "session_id": "s"}
        ).status_code
        == 422
    )


def test_trace_lookup_errors(client):
    assert client.get("/trace/run_never_ran", params={"user_id": "u_api"}).status_code == 404
    assert client.get("/trace/..hidden", params={"user_id": "u_api"}).status_code in (400, 404)


def test_a_failure_outside_the_nodes_still_answers_and_closes_the_trace(client):
    class BrokenGraph:
        def get_state(self, config):
            class Snapshot:
                interrupts = ()
                values: dict = {}  # noqa: RUF012

            return Snapshot()

        def invoke(self, *args, **kwargs):
            raise RuntimeError("checkpoint DB is locked")

    client.app.state.services.agent.graph = BrokenGraph()
    body = chat(client, "hi")
    assert body["outcome"] == "error" and body["run_id"] in body["reply"]
    events = client.get(f"/trace/{body['run_id']}", params={"user_id": "u_api"}).json()["events"]
    assert [e["kind"] for e in events][-2:] == ["error", "run.end"]


def test_memory_can_be_inspected_corrected_and_forgotten(client):
    me = {"user_id": "u_api"}
    assert client.get("/memory", params=me).json()["facts"] == []
    client.patch("/memory", json={**me, "key": "preferred_area", "value": "HSR Layout"})
    corrected = client.patch(
        "/memory", json={**me, "key": "preferred_area", "value": "Koramangala"}
    ).json()
    assert (corrected["value"], corrected["source"], corrected["status"]) == (
        "Koramangala",
        "explicit",
        "active",
    )
    history = client.get("/memory", params={**me, "include_inactive": True}).json()["facts"]
    assert [fact["status"] for fact in history] == ["superseded", "active"]

    forgotten = client.delete("/memory", params={**me, "key": "preferred_area"})
    assert forgotten.json()["rows_forgotten"] == 2
    assert client.get("/memory", params=me).json()["facts"] == []
    assert client.delete("/memory", params={**me, "key": "preferred_area"}).status_code == 404


def test_memory_writes_are_validated(client):
    me = {"user_id": "u_api"}
    bad = client.patch("/memory", json={**me, "key": "preferred_area", "value": None})
    assert bad.status_code == 422
    assert client.patch("/memory", json={**me, "key": "Bad Key!", "value": "x"}).status_code == 422
    assert client.get("/memory", params={"user_id": "../other"}).status_code == 422


def test_a_simulated_provider_cancellation_reaches_the_outbox(client):
    force_success(client)
    chat(client, SEARCH)
    chat(client, "1")
    chat(client, "yes")

    event = client.post("/simulate/provider_cancelled", json={"user_id": "u_api"}).json()["event"]
    assert event["kind"] == "provider_cancelled"
    assert client.app.state.services.worker.run_once()["events"] == 1
    messages = client.get("/outbox", params={"user_id": "u_api"}).json()["messages"]
    payload = messages[-1]["payload"]
    assert messages[-1]["kind"] == "provider_cancelled"
    assert payload["thread_id"] == f"u_api:{payload['session_id']}"
    nope = {"user_id": "u_api", "booking_id": "b_nope"}
    assert client.post("/simulate/provider_cancelled", json=nope).status_code == 404


def test_the_app_starts_and_answers_when_vega_cannot_load(settings, monkeypatch):
    # Regression for docs/bugs.md #20: a failed Vega load at startup stopped the app from
    # starting, and a failed decision was reported to the user as a provider failure.
    from fastapi.testclient import TestClient

    from errand.api import create_app
    from errand.decide import VegaDecider
    from errand.llm.rules import rule_model
    from errand.mocks import FakeClock
    from tests.support import START

    def no_weights(self):
        raise OSError("no checkpoint at ~/.cache/huggingface")

    monkeypatch.setattr(VegaDecider, "_load", no_weights)
    warm = settings.model_copy(update={"warmup_models": True})
    app = create_app(warm, model=rule_model(), clock=FakeClock(START), start_worker=False)
    with TestClient(app) as client:
        reply = chat(client, SEARCH)
    assert reply["outcome"] == "stopped"
    assert reply["reply"].startswith("My decision model isn't working just now")


def test_one_users_slow_turn_does_not_hold_up_another_users_request(client, monkeypatch):
    # The handlers are async; the agent (Ollama, Vega, SQLite) runs in the thread pool. A handler
    # that called it on the event loop would stall every other request until the slow one ended.
    agent = client.app.state.services.agent
    entered, release = threading.Event(), threading.Event()
    real_handle = agent.handle

    def handle(**kwargs):
        if kwargs["user_id"] == "u_slow":
            entered.set()
            release.wait(10)  # a model call that takes its time
        return real_handle(**kwargs)

    monkeypatch.setattr(agent, "handle", handle)

    async def both():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://errand") as http:
            body = {"message": "hello", "session_id": "s1"}
            slow = asyncio.create_task(http.post("/chat", json={**body, "user_id": "u_slow"}))
            assert await asyncio.to_thread(entered.wait, 10)
            fast = await asyncio.wait_for(http.post("/chat", json={**body, **ME}), timeout=5)
            still_waiting = not slow.done()
            release.set()
            return fast, await slow, still_waiting

    fast, slow, still_waiting = asyncio.run(both())
    assert fast.status_code == slow.status_code == 200 and still_waiting
