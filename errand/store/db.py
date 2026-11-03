"""SQLite plumbing for the agent DB: connections, migrations, write transactions and the user
scope guard.

The guard: on a database opened with ``scoped_tables``, every statement that touches one of those
tables must name ``user_id`` (in its WHERE clause, or its columns for an INSERT), or it raises
`UnscopedQuery` before it runs. A scan that really spans users (the reconciler's sweep of open
ledger rows, an eval oracle) declares itself with ``cross_user(reason)``.
"""

import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from errand.store.schema import MIGRATIONS

_TABLES = re.compile(r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_]+)", re.I)
_WHERE = re.compile(r"\bWHERE\b", re.I)
_USER = re.compile(r"\buser_id\b")


class UnscopedQuery(Exception):
    """A query on a user's table didn't say which user: it could read or change anyone's rows."""


def check_scope(sql: str, scoped: frozenset[str]) -> None:
    tables = {table.lower() for table in _TABLES.findall(sql)} & scoped
    if not tables:
        return
    head = sql.lstrip().split(None, 1)[0].upper()
    clause = sql if head == "INSERT" else [*_WHERE.split(sql, maxsplit=1), ""][1]
    if not _USER.search(clause):
        raise UnscopedQuery(
            f"query on {', '.join(sorted(tables))} without a user_id filter:"
            f" {' '.join(sql.split())[:160]}"
        )


class _ScopedConnection:
    """The transaction's connection, checking each statement against the guard."""

    def __init__(self, conn: sqlite3.Connection, guard: Callable[[str], None]) -> None:
        self._conn, self._guard = conn, guard

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        self._guard(sql)
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: Any) -> sqlite3.Cursor:
        self._guard(sql)
        return self._conn.executemany(sql, rows)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def utcnow() -> datetime:
    return datetime.now(UTC)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection in autocommit mode; transactions are begun explicitly."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate(conn: sqlite3.Connection, migrations: Sequence[Sequence[str]] = MIGRATIONS) -> int:
    """Apply pending migrations and return the schema version. Idempotent; run on every startup.

    The version is read after taking the write lock, so two processes starting at once (the API
    and the worker) apply each migration exactly once.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for statements in migrations[version:]:
            for sql in statements:
                conn.execute(sql)
            version += 1
            conn.execute(f"PRAGMA user_version = {version}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return version


class Database:
    """One migrated connection, shared by the repositories and serialized by a lock."""

    def __init__(
        self,
        path: Path | str,
        *,
        now: Callable[[], datetime] = utcnow,
        migrations: Sequence[Sequence[str]] = MIGRATIONS,
        scoped_tables: frozenset[str] = frozenset(),
    ) -> None:
        self.path = path
        self.now = now
        self._conn = connect(path)
        self._lock = threading.RLock()
        self._depth = 0
        self._scoped = scoped_tables
        self._cross = threading.local()  # per thread: the connection is shared across threads
        self.schema_version = migrate(self._conn, migrations)

    @contextmanager
    def cross_user(self, reason: str) -> Iterator[None]:
        """Allow queries spanning users inside the block, on this thread only. `reason` says why
        (kept for debugging); every such scan is a deliberate, system-level read."""
        previous = getattr(self._cross, "reason", None)
        self._cross.reason = reason
        try:
            yield
        finally:
            self._cross.reason = previous

    def _guard(self, sql: str) -> None:
        if self._scoped and not getattr(self._cross, "reason", None):
            check_scope(sql, self._scoped)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the block in one write transaction. A nested call joins the outer transaction."""
        with self._lock:
            conn: Any = _ScopedConnection(self._conn, self._guard) if self._scoped else self._conn
            if self._depth:
                self._depth += 1
                try:
                    yield conn
                finally:
                    self._depth -= 1
                return
            # IMMEDIATE takes the write lock up front: a read-then-write block can't hit
            # SQLITE_BUSY halfway through.
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield conn
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
            finally:
                self._depth = 0

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        self._guard(sql)
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        self._guard(sql)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def scan(self, sql: str, params: Sequence[Any] = (), *, reason: str) -> list[sqlite3.Row]:
        """A deliberate read across users (see `cross_user`)."""
        with self.cross_user(reason):
            return self.fetch_all(sql, params)

    def timestamp(self) -> str:
        return self.now().isoformat(timespec="milliseconds")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
