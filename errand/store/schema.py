"""Agent DB schema, as an append-only list of migrations.

``MIGRATIONS[n]`` takes the schema from version n to n + 1 (``PRAGMA user_version``). Never edit a
migration that has shipped: append a new one. Each entry is a tuple of single SQL statements, so a
migration is applied statement by statement inside one transaction.

The triggers are a backstop for the invariants application code already enforces: an approval's
material can't change after it is requested, its status only moves forward, and a confirmed ledger
entry is final.
"""

V1 = (
    """
    CREATE TABLE tasks (
        task_id     TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        thread_id   TEXT NOT NULL,
        intent      TEXT NOT NULL CHECK (intent IN ('book', 'reschedule', 'cancel')),
        phase       TEXT NOT NULL,
        details     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details)),
        booking_id  TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX tasks_by_thread ON tasks (thread_id, created_at)",
    "CREATE INDEX tasks_by_user ON tasks (user_id, updated_at)",
    """
    CREATE TABLE approvals (
        approval_id     TEXT PRIMARY KEY,
        task_id         TEXT NOT NULL REFERENCES tasks (task_id),
        action          TEXT NOT NULL CHECK (action IN ('book', 'reschedule', 'cancel')),
        material        TEXT NOT NULL CHECK (json_valid(material)),
        material_hash   TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'granted', 'consumed', 'invalidated')),
        reason          TEXT,
        created_at      TEXT NOT NULL,
        granted_at      TEXT,
        consumed_at     TEXT,
        invalidated_at  TEXT
    )
    """,
    # At most one live (pending or granted) approval per task.
    """
    CREATE UNIQUE INDEX approvals_one_live_per_task ON approvals (task_id)
    WHERE status IN ('pending', 'granted')
    """,
    """
    CREATE TRIGGER approvals_status_moves_forward
    BEFORE UPDATE OF status ON approvals
    WHEN NOT (
        NEW.status = OLD.status
        OR (OLD.status = 'pending' AND NEW.status IN ('granted', 'invalidated'))
        OR (OLD.status = 'granted' AND NEW.status IN ('consumed', 'invalidated'))
    )
    BEGIN
        SELECT RAISE(ABORT, 'approvals: illegal status transition');
    END
    """,
    """
    CREATE TRIGGER approvals_material_is_immutable
    BEFORE UPDATE OF task_id, action, material, material_hash ON approvals
    BEGIN
        SELECT RAISE(ABORT, 'approvals: material is immutable, request a new approval instead');
    END
    """,
    """
    CREATE TABLE ledger (
        idempotency_key TEXT PRIMARY KEY,
        op              TEXT NOT NULL CHECK (op IN ('book', 'reschedule', 'cancel')),
        task_id         TEXT REFERENCES tasks (task_id),
        approval_id     TEXT REFERENCES approvals (approval_id),
        request         TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(request)),
        status          TEXT NOT NULL DEFAULT 'in_flight'
                        CHECK (status IN ('in_flight', 'confirmed', 'not_booked', 'unknown')),
        attempts        INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        result          TEXT CHECK (result IS NULL OR json_valid(result)),
        last_error      TEXT,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    # One side effect per approval: a repeated confirmation can't open a second operation.
    """
    CREATE UNIQUE INDEX ledger_one_per_approval ON ledger (approval_id)
    WHERE approval_id IS NOT NULL
    """,
    "CREATE INDEX ledger_unresolved ON ledger (status) WHERE status IN ('in_flight', 'unknown')",
    """
    CREATE TRIGGER ledger_confirmed_is_final
    BEFORE UPDATE OF status ON ledger
    WHEN OLD.status = 'confirmed' AND NEW.status <> 'confirmed'
    BEGIN
        SELECT RAISE(ABORT, 'ledger: a confirmed operation cannot change status');
    END
    """,
    """
    CREATE TABLE memory_facts (
        fact_id        TEXT PRIMARY KEY,
        user_id        TEXT NOT NULL,
        key            TEXT NOT NULL,
        value          TEXT CHECK (value IS NULL OR json_valid(value)),
        service_tag    TEXT,
        source         TEXT NOT NULL CHECK (source IN ('explicit', 'inferred')),
        status         TEXT NOT NULL
                       CHECK (status IN ('active', 'unconfirmed', 'superseded', 'forgotten')),
        evidence       TEXT,
        superseded_by  TEXT REFERENCES memory_facts (fact_id) DEFERRABLE INITIALLY DEFERRED,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        CHECK (value IS NOT NULL OR status = 'forgotten'),
        CHECK (status <> 'unconfirmed' OR source = 'inferred')
    )
    """,
    # One current (active or unconfirmed) fact per user, key and service tag. A NULL tag means the
    # fact applies to every service.
    """
    CREATE UNIQUE INDEX memory_one_current_per_key
    ON memory_facts (user_id, key, IFNULL(service_tag, ''))
    WHERE status IN ('active', 'unconfirmed')
    """,
    "CREATE INDEX memory_by_user ON memory_facts (user_id, status)",
    """
    CREATE TABLE outbox (
        message_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT NOT NULL,
        task_id       TEXT REFERENCES tasks (task_id),
        kind          TEXT NOT NULL,
        body          TEXT NOT NULL,
        payload       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload)),
        dedupe_key    TEXT UNIQUE,
        created_at    TEXT NOT NULL,
        delivered_at  TEXT
    )
    """,
    "CREATE INDEX outbox_by_user ON outbox (user_id, message_id)",
)

V2 = (
    # The conversation, for recent turns and the rolling summary. Forgetting a fact scrubs its
    # value from here as well.
    """
    CREATE TABLE messages (
        message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     TEXT NOT NULL,
        thread_id   TEXT NOT NULL,
        run_id      TEXT,
        role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content     TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX messages_by_user ON messages (user_id, message_id)",
    "CREATE INDEX messages_by_thread ON messages (thread_id, message_id)",
    # The gateway's record of what hold_slot returned. book() derives the approval material from
    # it (provider, slot, quoted price), so the call is bound to details, not to a hold id.
    """
    CREATE TABLE holds (
        hold_id           TEXT PRIMARY KEY,
        task_id           TEXT REFERENCES tasks (task_id),
        provider_id       TEXT NOT NULL,
        slot_start        TEXT NOT NULL,
        slot_end          TEXT NOT NULL,
        quoted_price_inr  INTEGER NOT NULL CHECK (quoted_price_inr >= 0),
        expires_at        TEXT NOT NULL,
        created_at        TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER holds_are_immutable
    BEFORE UPDATE ON holds
    BEGIN
        SELECT RAISE(ABORT, 'holds: a recorded hold never changes');
    END
    """,
)

V3 = (
    # Replies for the user's eyes only (one listing stored memory, guesses included) stay in the
    # conversation but are never read back into a model's context.
    "ALTER TABLE messages ADD COLUMN for_model INTEGER NOT NULL DEFAULT 1",
)

V4 = (
    # Every row says whose it is, so every query can be scoped to one user (errand/store/db.py
    # refuses one that isn't). Existing rows take their task's user.
    "ALTER TABLE approvals ADD COLUMN user_id TEXT NOT NULL DEFAULT ''",
    "UPDATE approvals SET user_id = (SELECT user_id FROM tasks WHERE tasks.task_id = approvals.task_id)",
    "ALTER TABLE ledger ADD COLUMN user_id TEXT NOT NULL DEFAULT ''",
    "UPDATE ledger SET user_id = COALESCE("
    "(SELECT user_id FROM tasks WHERE tasks.task_id = ledger.task_id), '')",
    "ALTER TABLE holds ADD COLUMN user_id TEXT NOT NULL DEFAULT ''",
    "UPDATE holds SET user_id = COALESCE("
    "(SELECT user_id FROM tasks WHERE tasks.task_id = holds.task_id), '')",
    "CREATE INDEX approvals_by_user ON approvals (user_id, task_id)",
    "CREATE INDEX ledger_by_user ON ledger (user_id, status)",
    # One confirmed side effect per approval, held by the schema and not only by code (the row
    # per approval is already unique; this names the stronger promise).
    """
    CREATE UNIQUE INDEX ledger_one_confirmed_per_approval ON ledger (approval_id)
    WHERE status = 'confirmed' AND approval_id IS NOT NULL
    """,
    # Whose each trace is: /trace answers a run only to the user it belongs to.
    """
    CREATE TABLE runs (
        run_id      TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL CHECK (user_id <> ''),
        thread_id   TEXT,
        kind        TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX runs_by_user ON runs (user_id, created_at)",
)

MIGRATIONS: tuple[tuple[str, ...], ...] = (V1, V2, V3, V4)

# The tables whose rows belong to one user. A query on one of them must say which user.
USER_TABLES = frozenset(
    {"tasks", "approvals", "ledger", "memory_facts", "outbox", "messages", "holds", "runs"}
)
SCHEMA_VERSION = len(MIGRATIONS)
