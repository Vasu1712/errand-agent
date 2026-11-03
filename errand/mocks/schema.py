"""The mock world's own SQLite schema (a separate file from the agent DB).

Keeping the world's state on disk is what lets a restarted agent find out what really happened:
bookings, holds and idempotency keys survive the agent process.
"""

V1 = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE providers (
        provider_id   TEXT PRIMARY KEY,
        service       TEXT NOT NULL,
        area          TEXT NOT NULL,
        name          TEXT NOT NULL,
        price_inr     INTEGER NOT NULL,
        rating        REAL NOT NULL,
        review_count  INTEGER NOT NULL,
        about         TEXT NOT NULL,
        technicians   TEXT NOT NULL,
        attributes    TEXT NOT NULL
    )
    """,
    "CREATE INDEX providers_by_service_area ON providers (service, area, provider_id)",
    """
    CREATE TABLE reviews (
        review_id    TEXT PRIMARY KEY,
        provider_id  TEXT NOT NULL REFERENCES providers (provider_id),
        author       TEXT NOT NULL,
        rating       INTEGER NOT NULL,
        text         TEXT NOT NULL,
        posted_on    TEXT NOT NULL
    )
    """,
    "CREATE INDEX reviews_by_provider ON reviews (provider_id)",
    """
    CREATE TABLE holds (
        hold_id           TEXT PRIMARY KEY,
        provider_id       TEXT NOT NULL REFERENCES providers (provider_id),
        slot_start        TEXT NOT NULL,
        slot_end          TEXT NOT NULL,
        quoted_price_inr  INTEGER NOT NULL,
        expires_at        TEXT NOT NULL,
        status            TEXT NOT NULL CHECK (status IN ('active', 'used', 'released')),
        created_at        TEXT NOT NULL
    )
    """,
    "CREATE INDEX holds_by_slot ON holds (provider_id, slot_start, status)",
    """
    CREATE TABLE bookings (
        booking_id       TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        provider_id      TEXT NOT NULL REFERENCES providers (provider_id),
        service          TEXT NOT NULL,
        slot_start       TEXT NOT NULL,
        slot_end         TEXT NOT NULL,
        price_inr        INTEGER NOT NULL,
        status           TEXT NOT NULL
                         CHECK (status IN ('confirmed', 'cancelled', 'cancelled_by_provider')),
        technician       TEXT,
        hold_id          TEXT UNIQUE,
        idempotency_key  TEXT UNIQUE,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    "CREATE INDEX bookings_by_user ON bookings (user_id, slot_start)",
    "CREATE INDEX bookings_by_slot ON bookings (provider_id, slot_start, status)",
    # Results of cancel and reschedule calls by idempotency key, so a repeated call replays.
    """
    CREATE TABLE ops (
        idempotency_key  TEXT PRIMARY KEY,
        op               TEXT NOT NULL CHECK (op IN ('cancel', 'reschedule')),
        booking_id       TEXT NOT NULL,
        result           TEXT NOT NULL,
        created_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE attempts (
        scope  TEXT NOT NULL,
        ref    TEXT NOT NULL,
        count  INTEGER NOT NULL,
        PRIMARY KEY (scope, ref)
    )
    """,
    # Slots someone else booked (the slot_taken outcome); never offered again.
    """
    CREATE TABLE taken (
        provider_id  TEXT NOT NULL,
        slot_start   TEXT NOT NULL,
        PRIMARY KEY (provider_id, slot_start)
    )
    """,
    """
    CREATE TABLE forced (
        seq      INTEGER PRIMARY KEY AUTOINCREMENT,
        tool     TEXT NOT NULL,
        outcome  TEXT NOT NULL,
        used_at  TEXT
    )
    """,
    """
    CREATE TABLE events (
        event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        kind         TEXT NOT NULL CHECK (kind IN ('provider_cancelled')),
        booking_id   TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        due_at       TEXT NOT NULL,
        reason       TEXT NOT NULL DEFAULT '',
        consumed_at  TEXT
    )
    """,
    "CREATE INDEX events_due ON events (due_at) WHERE consumed_at IS NULL",
)

V2 = (
    # Who has each slot. hold_slot, book and reschedule claim a slot with one conditional UPDATE
    # (WHERE status = 'free', or a lapsed hold) and check the row count, so two users can never
    # both get it: the loser's UPDATE changes nothing and it gets slot_taken.
    """
    CREATE TABLE slot_claims (
        provider_id  TEXT NOT NULL REFERENCES providers (provider_id),
        slot_start   TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('free', 'held', 'booked', 'taken')),
        hold_id      TEXT,
        held_until   TEXT,
        booking_id   TEXT,
        user_id      TEXT,
        updated_at   TEXT NOT NULL,
        PRIMARY KEY (provider_id, slot_start)
    )
    """,
    "INSERT OR IGNORE INTO slot_claims (provider_id, slot_start, status, booking_id, user_id,"
    " updated_at) SELECT provider_id, slot_start, 'booked', booking_id, user_id, updated_at"
    " FROM bookings WHERE status = 'confirmed'",
    "INSERT OR IGNORE INTO slot_claims (provider_id, slot_start, status, updated_at)"
    " SELECT provider_id, slot_start, 'taken', '' FROM taken",
    "INSERT OR IGNORE INTO slot_claims (provider_id, slot_start, status, hold_id, held_until,"
    " updated_at) SELECT provider_id, slot_start, 'held', hold_id, expires_at, created_at"
    " FROM holds WHERE status = 'active'",
    # And the schema's backstop: never two confirmed bookings for one provider and slot.
    """
    CREATE UNIQUE INDEX bookings_one_confirmed_per_slot ON bookings (provider_id, slot_start)
    WHERE status = 'confirmed'
    """,
)

WORLD_MIGRATIONS: tuple[tuple[str, ...], ...] = (V1, V2)
