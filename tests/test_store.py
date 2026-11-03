import re
import sqlite3

import pytest

from errand.store import (
    SCHEMA_VERSION,
    Action,
    ApprovalError,
    ApprovalStatus,
    FactSource,
    FactStatus,
    LedgerConflict,
    LedgerStatus,
    Store,
    UnscopedQuery,
    connect,
    material_hash,
    migrate,
)

MATERIAL = {"provider_id": "prov_1", "slot": "2026-11-05T09:00+05:30", "price_inr": 1200}
U1 = {"user_id": "u1"}


@pytest.fixture
def task(store):
    return store.tasks.create(
        user_id="u1", thread_id="u1:s1", intent=Action.BOOK, phase="collecting"
    )


# Schema and migrations


def test_migration_creates_every_table_and_is_idempotent(tmp_path):
    conn = connect(tmp_path / "agent.db")
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"tasks", "approvals", "ledger", "memory_facts", "outbox", "runs"} <= tables
    for table in ("tasks", "approvals", "ledger", "memory_facts", "outbox", "messages", "holds"):
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert "user_id" in columns, table
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_reopening_the_db_keeps_its_data(tmp_path):
    with Store(tmp_path / "agent.db") as first:
        task = first.tasks.create(user_id="u1", thread_id="u1:s1", intent=Action.CANCEL, phase="x")
    with Store(tmp_path / "agent.db") as second:
        assert second.tasks.get(task.task_id, **U1) == task


@pytest.mark.parametrize(
    ("table", "column", "enum"),
    [
        ("tasks", "intent", Action),
        ("approvals", "action", Action),
        ("approvals", "status", ApprovalStatus),
        ("ledger", "op", Action),
        ("ledger", "status", LedgerStatus),
        ("memory_facts", "source", FactSource),
        ("memory_facts", "status", FactStatus),
    ],
)
def test_check_constraints_match_the_pydantic_enums(store, table, column, enum):
    ddl = store.db.fetch_one("SELECT sql FROM sqlite_master WHERE name = ?", (table,))["sql"]
    allowed = re.search(rf"CHECK \({column} IN \(([^)]*)\)\)", ddl).group(1)
    assert {value.strip(" '") for value in allowed.split(",")} == {member.value for member in enum}


def test_values_outside_the_enums_are_rejected(store, task):
    with pytest.raises(sqlite3.IntegrityError), store.db.transaction() as conn:
        conn.execute(
            "UPDATE tasks SET intent = 'refund' WHERE task_id = ? AND user_id = ?",
            (task.task_id, "u1"),
        )


# The user scope guard


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM approvals WHERE approval_id = ?",
        "UPDATE ledger SET status = 'unknown' WHERE idempotency_key = ?",
        "SELECT user_id FROM memory_facts",  # names the column but filters nothing
        "DELETE FROM outbox WHERE message_id = ?",
        "INSERT INTO messages (thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
    ],
)
def test_a_query_on_a_users_table_without_user_id_is_refused(store, sql):
    with pytest.raises(UnscopedQuery):
        store.db.fetch_all(sql, ("x",) * sql.count("?"))
    with pytest.raises(UnscopedQuery), store.db.transaction() as conn:
        conn.execute(sql, ("x",) * sql.count("?"))


def test_a_declared_cross_user_scan_is_allowed_and_only_on_its_thread(store, task):
    with store.db.cross_user("test: count every user's tasks"):
        assert store.db.fetch_one("SELECT COUNT(*) FROM tasks")[0] == 1
    with pytest.raises(UnscopedQuery):
        store.db.fetch_one("SELECT COUNT(*) FROM tasks")


def test_one_users_rows_are_invisible_to_another(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    store.ledger.begin("idem_x", Action.BOOK, request={}, approval_id=approval.approval_id, **U1)
    other = {"user_id": "u2"}
    assert store.tasks.get(task.task_id, **other) is None
    assert store.approvals.get(approval.approval_id, **other) is None
    assert store.ledger.get("idem_x", **other) is None
    with pytest.raises(ApprovalError, match="does not exist for u2"):
        store.approvals.grant(approval.approval_id, approval.material_hash, **other)
    assert store.approvals.get(approval.approval_id, **U1).status is ApprovalStatus.PENDING


# Approvals


def test_an_approval_is_granted_then_consumed_once_for_the_same_material(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    assert approval.status is ApprovalStatus.PENDING and approval.user_id == "u1"
    assert approval.material_hash == material_hash(MATERIAL)

    store.approvals.grant(approval.approval_id, approval.material_hash, **U1)
    consumed = store.approvals.consume(approval.approval_id, approval.material_hash, **U1)
    assert consumed.status is ApprovalStatus.CONSUMED
    with pytest.raises(ApprovalError, match="consumed"):
        store.approvals.consume(approval.approval_id, approval.material_hash, **U1)


def test_material_hash_ignores_key_order_but_not_values():
    reordered = dict(reversed(list(MATERIAL.items())))
    assert material_hash(reordered) == material_hash(MATERIAL)
    assert material_hash({**MATERIAL, "price_inr": 1201}) != material_hash(MATERIAL)


def test_an_approval_cannot_be_granted_for_different_material(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    repriced = material_hash({**MATERIAL, "price_inr": 1450})
    with pytest.raises(ApprovalError, match="different material"):
        store.approvals.grant(approval.approval_id, repriced, **U1)
    assert store.approvals.get(approval.approval_id, **U1).status is ApprovalStatus.PENDING


def test_a_new_request_invalidates_the_live_approval(store, task):
    first = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    store.approvals.grant(first.approval_id, first.material_hash, **U1)
    second = store.approvals.request(
        task.task_id, Action.BOOK, {**MATERIAL, "slot": "2026-11-05T11:00+05:30"}, **U1
    )

    invalidated = store.approvals.get(first.approval_id, **U1)
    assert invalidated.status is ApprovalStatus.INVALIDATED
    assert invalidated.reason == f"superseded by {second.approval_id}"
    assert store.approvals.live(task.task_id, **U1) == second
    with pytest.raises(ApprovalError):
        store.approvals.consume(first.approval_id, first.material_hash, **U1)


def test_invalidate_drops_the_live_approval(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    invalidated = store.approvals.invalidate(task.task_id, "price changed at hold", **U1)
    assert invalidated == [approval.approval_id]
    assert store.approvals.live(task.task_id, **U1) is None
    with pytest.raises(ApprovalError, match="invalidated"):
        store.approvals.grant(approval.approval_id, approval.material_hash, **U1)


def test_schema_triggers_block_backward_moves_and_material_edits(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    store.approvals.grant(approval.approval_id, approval.material_hash, **U1)
    store.approvals.consume(approval.approval_id, approval.material_hash, **U1)

    where = "WHERE approval_id = ? AND user_id = ?"
    with (
        pytest.raises(sqlite3.IntegrityError, match="illegal status transition"),
        store.db.transaction() as conn,
    ):
        conn.execute(
            f"UPDATE approvals SET status = 'granted' {where}", (approval.approval_id, "u1")
        )
    with (
        pytest.raises(sqlite3.IntegrityError, match="material is immutable"),
        store.db.transaction() as conn,
    ):
        conn.execute(
            f"UPDATE approvals SET material = '{{}}' {where}", (approval.approval_id, "u1")
        )


# Ledger


def test_ledger_begin_is_idempotent_per_key(store, task):
    request = {"hold_id": "hold_1"}
    entry, created = store.ledger.begin(
        "book:k1", Action.BOOK, request=request, task_id=task.task_id, **U1
    )
    again, created_again = store.ledger.begin(
        "book:k1", Action.BOOK, request=request, task_id=task.task_id, **U1
    )
    assert (created, created_again) == (True, False)
    assert again == entry
    assert entry.status is LedgerStatus.IN_FLIGHT


def test_ledger_refuses_a_key_reused_for_another_request(store):
    store.ledger.begin("book:k1", Action.BOOK, request={"hold_id": "hold_1"}, **U1)
    with pytest.raises(LedgerConflict):
        store.ledger.begin("book:k1", Action.BOOK, request={"hold_id": "hold_2"}, **U1)


def test_one_ledger_entry_and_one_confirmation_per_approval_by_unique_indexes(store, task):
    approval = store.approvals.request(task.task_id, Action.BOOK, MATERIAL, **U1)
    store.ledger.begin("book:k1", Action.BOOK, request={}, approval_id=approval.approval_id, **U1)
    with pytest.raises(sqlite3.IntegrityError):
        store.ledger.begin(
            "book:k2", Action.BOOK, request={}, approval_id=approval.approval_id, **U1
        )
    indexes = {
        row["name"]
        for row in store.db.fetch_all("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert {"ledger_one_per_approval", "ledger_one_confirmed_per_approval"} <= indexes


def test_unknown_outcomes_stay_unresolved_and_a_confirmed_entry_is_final(store):
    store.ledger.begin("book:k1", Action.BOOK, request={"hold_id": "hold_1"}, **U1)
    store.ledger.record_attempt("book:k1", **U1)
    store.ledger.mark("book:k1", LedgerStatus.UNKNOWN, error="timeout", **U1)
    assert [e.idempotency_key for e in store.ledger.unresolved(**U1)] == ["book:k1"]
    assert [e.user_id for e in store.ledger.unresolved_everywhere()] == ["u1"]
    assert store.ledger.unresolved(user_id="u2") == []

    confirmed = store.ledger.mark(
        "book:k1", LedgerStatus.CONFIRMED, result={"booking_id": "bk_1"}, **U1
    )
    assert (confirmed.attempts, confirmed.result) == (1, {"booking_id": "bk_1"})
    assert store.ledger.unresolved(**U1) == []
    with pytest.raises(sqlite3.IntegrityError, match="cannot change status"):
        store.ledger.mark("book:k1", LedgerStatus.NOT_BOOKED, **U1)
    with pytest.raises(sqlite3.IntegrityError):
        store.ledger.record_attempt("book:k1", **U1)


# Memory


def test_explicit_facts_supersede_and_inferred_ones_never_override_them(store):
    inferred = store.memory.remember(
        "u1", "preferred_area", "HSR Layout", source=FactSource.INFERRED, evidence="3 past bookings"
    )
    assert inferred.status is FactStatus.UNCONFIRMED

    explicit = store.memory.remember(
        "u1", "preferred_area", "Koramangala", source=FactSource.EXPLICIT
    )
    assert explicit.status is FactStatus.ACTIVE
    ignored = store.memory.remember(
        "u1", "preferred_area", "Indiranagar", source=FactSource.INFERRED
    )
    assert ignored == explicit

    history = store.memory.list_for_user("u1", include_inactive=True)
    assert [(fact.value, fact.status) for fact in history] == [
        ("HSR Layout", FactStatus.SUPERSEDED),
        ("Koramangala", FactStatus.ACTIVE),
    ]
    assert history[0].superseded_by == explicit.fact_id
    assert store.memory.list_for_user("u1") == [explicit]


def test_service_tags_keep_separate_current_facts(store):
    store.memory.remember(
        "u1", "technician", "prov_7", source=FactSource.EXPLICIT, service_tag="electrician"
    )
    store.memory.remember(
        "u1", "technician", "prov_2", source=FactSource.EXPLICIT, service_tag="plumber"
    )
    assert {(fact.service_tag, fact.value) for fact in store.memory.list_for_user("u1")} == {
        ("electrician", "prov_7"),
        ("plumber", "prov_2"),
    }


def test_forgetting_erases_the_value_and_its_history_for_that_user_only(store):
    for user in ("u1", "u2"):
        store.memory.remember(
            user, "earliest_time", "09:00", source=FactSource.EXPLICIT, evidence="not before 9"
        )
    store.memory.remember(
        "u1", "earliest_time", "10:00", source=FactSource.EXPLICIT, evidence="not before 10"
    )
    assert store.memory.forget("u1", "earliest_time") == 2

    assert store.memory.list_for_user("u1") == []
    rows = store.memory.list_for_user("u1", include_inactive=True)
    assert [(row.status, row.value, row.evidence) for row in rows] == [
        (FactStatus.FORGOTTEN, None, None)
    ] * 2
    assert store.memory.forget("u1", "earliest_time") == 0
    assert [fact.value for fact in store.memory.list_for_user("u2")] == ["09:00"]


# Outbox


def test_outbox_dedupes_events_and_pages_by_cursor(store):
    first, created = store.outbox.enqueue(
        "u1", "provider_cancelled", "Your plumber cancelled.", dedupe_key="evt_1"
    )
    again, created_again = store.outbox.enqueue(
        "u1", "provider_cancelled", "Your plumber cancelled.", dedupe_key="evt_1"
    )
    second, _ = store.outbox.enqueue("u1", "info", "Another note")

    assert (created, created_again, again) == (True, False, first)
    assert store.outbox.since("u1") == [first, second]
    assert store.outbox.since("u1", after_id=first.message_id) == [second]
    assert store.outbox.since("someone_else") == []
    assert store.outbox.exists("u1", "evt_1") and not store.outbox.exists("u2", "evt_1")
    assert store.outbox.mark_delivered("u2", [first.message_id]) == 0  # not theirs
    assert store.outbox.mark_delivered("u1", [first.message_id]) == 1
    assert store.outbox.mark_delivered("u1", [first.message_id]) == 0


# Runs


def test_a_run_is_registered_to_one_user(store):
    store.runs.register("run_a", user_id="u1", thread_id="u1:s1", kind="chat")
    assert store.runs.get("run_a", **U1).thread_id == "u1:s1"
    assert store.runs.get("run_a", user_id="u2") is None
    with pytest.raises(ValueError):
        store.runs.register("run_b", user_id="", thread_id=None, kind="chat")
