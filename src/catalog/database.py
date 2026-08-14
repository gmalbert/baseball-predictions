"""DuckDB metadata catalog; large facts remain in partitioned Parquet."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Catalog:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[object]:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "DuckDB is required for the v2 catalog; install project dependencies"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.path))
        try:
            yield connection
        finally:
            connection.close()

    def migrate(self, migrations_dir: Path) -> list[str]:
        applied: list[str] = []
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migration "
                "(version VARCHAR PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT current_timestamp)"
            )
            existing = {
                row[0]
                for row in connection.execute("SELECT version FROM schema_migration").fetchall()
            }
            for migration in sorted(migrations_dir.glob("*.sql")):
                if migration.name in existing:
                    continue
                connection.execute("BEGIN")
                try:
                    connection.execute(migration.read_text(encoding="utf-8"))
                    connection.execute(
                        "INSERT INTO schema_migration(version) VALUES (?)", [migration.name]
                    )
                    connection.execute("COMMIT")
                    applied.append(migration.name)
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
        return applied
