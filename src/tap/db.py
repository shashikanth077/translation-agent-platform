"""Thin SQLite wrapper: one serialised connection, nested transactions, versioned per-service schemas.

Each service owns its tables (``Database.migrate(service, steps)``) and never joins across another
service's tables. Swapping SQLite for one Postgres database per service only touches this module.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (service TEXT PRIMARY KEY, version INTEGER NOT NULL)")

    # -- transactions -------------------------------------------------------------------------
    @contextmanager
    def tx(self) -> Iterator[Database]:
        """Serialised transaction. Re-entrant: inner ``tx()`` joins the outer one."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outer:
                    self._conn.execute("COMMIT")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        row = self.one(sql, params)
        return None if row is None else row[0]

    # -- schema -------------------------------------------------------------------------------
    def migrate(self, service: str, steps: list[str]) -> None:
        """Apply ``steps[current_version:]`` for ``service``. Append new steps, never edit old ones."""
        with self.tx():
            current = self.scalar("SELECT version FROM schema_migrations WHERE service=?", (service,)) or 0
            for i, script in enumerate(steps[current:], start=current + 1):
                with self._lock:
                    for stmt in _split_statements(script):
                        self._conn.execute(stmt)
                self.execute(
                    "INSERT INTO schema_migrations(service, version) VALUES(?, ?) "
                    "ON CONFLICT(service) DO UPDATE SET version=excluded.version",
                    (service, i),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_statements(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]
