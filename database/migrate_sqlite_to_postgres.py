"""
Automated SQLite to PostgreSQL Migration Tool for Server AI Assistant.
Extracts all campus intelligence records, user accounts, chat sessions,
maintenance tickets, and departmental data from SQLite (database/app.db)
and safely migrates them into the PostgreSQL database.
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")

from database.postgres_db import (
    get_postgres_config,
    get_postgres_connection,
    init_postgres_db,
    is_postgres_available,
)


def get_sqlite_conn(sqlite_path: Optional[str] = None) -> sqlite3.Connection:
    """Connect to authoritative SQLite database."""
    default_path = BASE_DIR / "database" / "app.db"
    target = Path(sqlite_path) if sqlite_path else default_path
    if not target.exists():
        raise FileNotFoundError(f"SQLite database file not found at: {target}")
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def migrate_table(sqlite_conn, pg_conn, table_name: str, primary_key: str = "id") -> int:
    """Migrate a single table from SQLite to PostgreSQL with batch streaming."""
    s_cur = sqlite_conn.cursor()
    # Check if table exists in SQLite
    table_exists = s_cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    ).fetchone()
    if not table_exists:
        return 0

    s_cur.execute(f"SELECT * FROM {table_name};")
    rows = s_cur.fetchall()
    if not rows:
        return 0

    columns = [col[0] for col in s_cur.description]
    col_names = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))

    p_cur = pg_conn.cursor()
    migrated_count = 0

    # Batch insert into PostgreSQL
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        values_batch = [[r[c] for c in columns] for r in batch]
        
        # Conflict resolution
        if table_name == "software_records":
            conflict_clause = "ON CONFLICT (source_app, record_id) DO UPDATE SET payload_json=EXCLUDED.payload_json, content_hash=EXCLUDED.content_hash, updated_at=EXCLUDED.updated_at"
        elif table_name == "users":
            conflict_clause = 'ON CONFLICT (username) DO UPDATE SET full_name=EXCLUDED.full_name, role=EXCLUDED.role, permissions=EXCLUDED.permissions'
        else:
            conflict_clause = f'ON CONFLICT ("{primary_key}") DO NOTHING'

        query = f"""
            INSERT INTO {table_name} ({col_names})
            VALUES ({placeholders})
            {conflict_clause};
        """
        p_cur.executemany(query, values_batch)
        migrated_count += len(batch)

    pg_conn.commit()
    p_cur.close()
    return migrated_count


def run_full_migration(sqlite_path: Optional[str] = None):
    """Execute complete end-to-end migration from SQLite to PostgreSQL."""
    print("=" * 70)
    print("RATHINAM CAMPUS AI ASSISTANT: SQLITE -> POSTGRESQL MIGRATION ENGINE")
    print("=" * 70)

    cfg = get_postgres_config()
    print(f"Target PostgreSQL: host={cfg['host']}, port={cfg['port']}, db={cfg['database']}, user={cfg['user']}")

    if not is_postgres_available():
        print("\n[ERROR] Could not connect to PostgreSQL server.")
        print("Please ensure your PostgreSQL service is running and accessible with the credentials in .env.")
        print(f"Connection parameters: host={cfg['host']}, port={cfg['port']}, db={cfg['database']}, user={cfg['user']}")
        print("\nTo start PostgreSQL locally, run your PostgreSQL service or Docker container:")
        print("  docker run --name campus-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=campus_ai_db -p 5432:5432 -d postgres")
        return False

    print("\n1. Initializing PostgreSQL schemas and tables...")
    init_postgres_db()
    print("   [PASS] Schemas initialized.")

    sqlite_conn = get_sqlite_conn(sqlite_path)
    pg_conn = get_postgres_connection()

    tables = [
        ("users", "id"),
        ("user_sessions", "token"),
        ("chat_sessions", "id"),
        ("chat_messages", "id"),
        ("software_records", "record_id"),
        ("sync_history", "id"),
        ("sql_file_tracking", "id"),
        ("food_items", "id"),
        ("food_inventory", "id"),
        ("food_purchase_requests", "id"),
        ("food_material_requests", "id"),
        ("food_vendors", "id"),
        ("food_procurement", "id"),
        ("cctv_cameras", "id"),
        ("cctv_nvrs", "id"),
        ("cctv_switches", "id"),
        ("cctv_biometrics", "id"),
        ("cctv_tickets", "id"),
        ("transport_vehicles", "id"),
        ("transport_fuel_entries", "id"),
        ("transport_renewals", "id"),
        ("transport_services", "id"),
    ]

    print("\n2. Migrating departmental records and data...")
    total_migrated = 0
    for tbl, pkey in tables:
        count = migrate_table(sqlite_conn, pg_conn, tbl, pkey)
        total_migrated += count
        if count > 0:
            print(f"   [MIGRATED] {tbl:28s} -> {count:5d} rows")
        else:
            print(f"   [SKIPPED]  {tbl:28s} ->     0 rows (empty in source)")

    sqlite_conn.close()
    pg_conn.close()

    print("\n" + "=" * 70)
    print(f"SUCCESS: Migration completed. {total_migrated} total rows migrated to PostgreSQL.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sqlite_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_migration(sqlite_file)
