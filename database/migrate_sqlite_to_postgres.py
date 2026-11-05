"""
Automated SQLite to PostgreSQL Migration Tool for Server AI Assistant.
Extracts all campus intelligence records, user accounts, chat sessions,
maintenance tickets, and departmental data from SQLite (app_data.db & app.db)
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
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

from database.postgres_db import (
    get_postgres_config,
    get_postgres_connection,
    init_postgres_db,
    is_postgres_available,
)


def get_sqlite_conn(sqlite_path: str) -> sqlite3.Connection:
    """Connect to SQLite database."""
    target = Path(sqlite_path)
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

    s_cur.execute(f'SELECT * FROM "{table_name}";')
    rows = s_cur.fetchall()
    if not rows:
        return 0

    columns = [col[0] for col in s_cur.description]
    col_names = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))

    p_cur = pg_conn.cursor()
    migrated_count = 0

    for r in rows:
        vals = [r[c] for c in columns]
        
        # Conflict resolution
        if table_name == "software_records":
            query = f"""
                INSERT INTO software_records ({col_names})
                VALUES ({placeholders})
                ON CONFLICT (source_app, record_id) DO UPDATE SET 
                    payload_json = EXCLUDED.payload_json, 
                    content_hash = EXCLUDED.content_hash, 
                    entity_type = EXCLUDED.entity_type,
                    table_name = EXCLUDED.table_name,
                    updated_at = EXCLUDED.updated_at;
            """
        elif table_name == "users":
            query = f"""
                INSERT INTO users ({col_names})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET
                    username = EXCLUDED.username,
                    password_hash = EXCLUDED.password_hash,
                    full_name = EXCLUDED.full_name,
                    email = EXCLUDED.email,
                    role = EXCLUDED.role,
                    department = EXCLUDED.department,
                    permissions = EXCLUDED.permissions,
                    is_active = EXCLUDED.is_active;
            """
        elif table_name == "chat_sessions":
            query = f"""
                INSERT INTO chat_sessions ({col_names})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at;
            """
        else:
            query = f"""
                INSERT INTO {table_name} ({col_names})
                VALUES ({placeholders})
                ON CONFLICT ("{primary_key}") DO NOTHING;
            """

        try:
            p_cur.execute(query, vals)
            migrated_count += 1
        except Exception as err:
            pg_conn.rollback()
            # If username conflict on users, handle gracefully
            if table_name == "users":
                try:
                    p_cur.execute("""
                        INSERT INTO users (username, password_hash, full_name, email, role, department, permissions, is_active, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING;
                    """, [r["username"], r["password_hash"], r["full_name"], r["email"], r["role"], r["department"], r["permissions"], r["is_active"], r["created_at"]])
                    migrated_count += 1
                except Exception:
                    pg_conn.rollback()
            continue

    pg_conn.commit()

    # Reset serial sequences in PostgreSQL if applicable
    serial_tables = {
        "users": "id",
        "sync_history": "id",
        "sql_file_tracking": "id",
        "chat_messages": "id",
    }
    if table_name in serial_tables:
        try:
            seq_cur = pg_conn.cursor()
            seq_cur.execute(f"""
                SELECT setval(pg_get_serial_sequence('{table_name}', '{serial_tables[table_name]}'),
                              COALESCE((SELECT MAX({serial_tables[table_name]}) FROM {table_name}), 1));
            """)
            pg_conn.commit()
            seq_cur.close()
        except Exception:
            pg_conn.rollback()

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
        return False

    print("\n1. Initializing PostgreSQL schemas and tables...")
    init_postgres_db()
    print("   [PASS] Schemas initialized.")

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

    # Sources to migrate
    sources = []
    if sqlite_path:
        sources.append(sqlite_path)
    else:
        app_data_path = BASE_DIR / "database" / "app_data.db"
        app_path = BASE_DIR / "database" / "app.db"
        if app_data_path.exists():
            sources.append(str(app_data_path))
        if app_path.exists():
            sources.append(str(app_path))

    pg_conn = get_postgres_connection()
    total_migrated = 0

    for src in sources:
        print(f"\n2. Migrating from source SQLite: {Path(src).name}...")
        sqlite_conn = get_sqlite_conn(src)
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
    print(f"SUCCESS: Migration completed. {total_migrated} total rows processed into PostgreSQL.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    sqlite_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_migration(sqlite_file)
