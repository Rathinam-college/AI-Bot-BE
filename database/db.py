"""
Database Layer for Server AI Assistant.
Authoritative Single Database Architecture: database/app.db.
Provides robust connection management, WAL mode, schema initialization,
and deterministic deduplication storage.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_project_root() -> Path:
    """Deterministically locate the workspace root containing main.py and backend."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "main.py").exists() and (parent / "backend").exists():
            return parent
    if current.parent.name in ("database", "services", "routers"):
        return current.parent.parent
    if current.parent.name == "backend":
        return current.parent.parent
    return current.parent


ROOT_DIR = find_project_root()
AUTHORITATIVE_DB_REL = "database/app.db"
AUTHORITATIVE_DB_PATH = AUTHORITATIVE_DB_REL
DEFAULT_DB_PATH = AUTHORITATIVE_DB_REL


def resolve_db_path(db_path: Optional[str] = None) -> Path:
    """
    Resolve and ensure parent directory for the database path.
    Guarantees that all non-isolated requests target <ROOT_DIR>/database/app.db.
    """
    raw = db_path or os.getenv("SQLITE_DB_PATH", AUTHORITATIVE_DB_REL)
    raw_str = str(raw).replace("\\", "/")

    # Isolated test database or explicit absolute path
    p = Path(raw)
    if "test" in raw_str.lower() or p.is_absolute():
        if not p.is_absolute():
            p = ROOT_DIR / raw
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Otherwise, authoritative database path
    target = ROOT_DIR / raw
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a SQLite connection with row factory enabled."""
    target_path = resolve_db_path(db_path)
    conn = sqlite3.connect(str(target_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize SQLite database and ensure required tables exist.
    Starts with ZERO business records and no historical data.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS software_records (
                    source_app TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'Record',
                    table_name TEXT,
                    payload_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (source_app, record_id)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_app ON software_records (source_app);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_entity ON software_records (source_app, entity_type);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    synced_at TIMESTAMP NOT NULL,
                    software_name TEXT NOT NULL,
                    fetched_count INTEGER NOT NULL,
                    inserted_count INTEGER NOT NULL,
                    updated_count INTEGER NOT NULL,
                    skipped_count INTEGER NOT NULL
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sql_file_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    department TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    discovered_count INTEGER NOT NULL,
                    inserted_count INTEGER NOT NULL,
                    updated_count INTEGER NOT NULL,
                    skipped_count INTEGER NOT NULL,
                    rejected_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    processed_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'SUCCESS',
                    error_message TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tracking_fn ON sql_file_tracking (filename);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tracking_dept ON sql_file_tracking (department);")

            # Relational domain tables for structured querying when present
            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_purchase_requests (
                    id TEXT PRIMARY KEY,
                    pr_reference TEXT,
                    order_date TEXT,
                    delivery_date TEXT,
                    vendor_id TEXT,
                    total_amount REAL,
                    status TEXT,
                    raw_json TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_food_pr_date ON food_purchase_requests(order_date);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_material_requests (
                    id TEXT PRIMARY KEY,
                    mr_number TEXT,
                    order_date TEXT,
                    target_date TEXT,
                    meal_type TEXT,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_items (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    unit TEXT,
                    min_stock_level REAL,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_inventory (
                    id TEXT PRIMARY KEY,
                    item_id TEXT,
                    current_quantity REAL,
                    min_stock_level REAL,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_vendors (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    contact_person TEXT,
                    phone TEXT,
                    address TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS food_procurement (
                    id TEXT PRIMARY KEY,
                    purchase_date TEXT,
                    vendor_name TEXT,
                    invoice_number TEXT,
                    item_name TEXT,
                    quantity REAL,
                    unit TEXT,
                    unit_price REAL,
                    tax_rate REAL,
                    total_price REAL,
                    purchase_type TEXT,
                    raw_json TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_food_proc_date ON food_procurement(purchase_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_food_proc_item ON food_procurement(item_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_food_proc_vendor ON food_procurement(vendor_name);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cctv_cameras (
                    id TEXT PRIMARY KEY,
                    camera_code TEXT,
                    camera_type TEXT,
                    device_type TEXT,
                    name TEXT,
                    block TEXT,
                    location TEXT,
                    ip_address TEXT,
                    status TEXT,
                    nvr_name TEXT,
                    raw_json TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cctv_ip ON cctv_cameras(ip_address);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cctv_nvrs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    block TEXT,
                    ip_address TEXT,
                    total_channels INTEGER,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cctv_switches (
                    id TEXT PRIMARY KEY,
                    switch_name TEXT,
                    location TEXT,
                    ip_address TEXT,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cctv_biometrics (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    block TEXT,
                    location TEXT,
                    ip_address TEXT,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cctv_tickets (
                    id TEXT PRIMARY KEY,
                    ticket_number TEXT,
                    issue_description TEXT,
                    status TEXT,
                    block TEXT,
                    date_logged TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS transport_vehicles (
                    id TEXT PRIMARY KEY,
                    vehicle_number TEXT,
                    driver_name TEXT,
                    driver_contact_no TEXT,
                    seating_capacity INTEGER,
                    make TEXT,
                    model TEXT,
                    status TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS transport_fuel_entries (
                    id TEXT PRIMARY KEY,
                    vehicle_number TEXT,
                    date TEXT,
                    liters REAL,
                    amount REAL,
                    total_km REAL,
                    bill_no TEXT,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS transport_renewals (
                    id TEXT PRIMARY KEY,
                    vehicle_number TEXT,
                    document_type TEXT,
                    new_expiry_date TEXT,
                    status TEXT,
                    amount_paid REAL,
                    raw_json TEXT
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS transport_services (
                    id TEXT PRIMARY KEY,
                    vehicle_number TEXT,
                    issue TEXT,
                    status TEXT,
                    raw_json TEXT
                );
            """)
    finally:
        conn.close()


def init_all_dbs() -> None:
    """Initialize the authoritative database without inserting any seed data."""
    init_db(str(resolve_db_path()))


def get_all_records(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all software records across all departments from authoritative database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT source_app, record_id, entity_type, table_name, payload_json, content_hash, created_at, updated_at
            FROM software_records
            ORDER BY source_app ASC, record_id ASC;
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_records_by_app(source_app: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all software records for a specific department."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT source_app, record_id, entity_type, table_name, payload_json, content_hash, created_at, updated_at
            FROM software_records
            WHERE source_app = ?
            ORDER BY record_id ASC;
        """, (source_app,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_sync_metrics_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get aggregated metrics and latest sync summary from authoritative database."""
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM software_records;")
        total_records = cur.fetchone()[0]

        cur.execute("""
            SELECT synced_at, software_name, fetched_count, inserted_count, updated_count, skipped_count
            FROM sync_history
            ORDER BY id DESC LIMIT 1;
        """)
        latest_row = cur.fetchone()

        cur.execute("""
            SELECT synced_at,
                   SUM(fetched_count) as total_fetched,
                   SUM(inserted_count) as total_inserted,
                   SUM(updated_count) as total_updated,
                   SUM(skipped_count) as total_skipped
            FROM sync_history
            WHERE synced_at = (SELECT synced_at FROM sync_history ORDER BY id DESC LIMIT 1)
            GROUP BY synced_at;
        """)
        batch_row = cur.fetchone()

        if batch_row and batch_row["synced_at"]:
            latest_batch = {
                "synced_at": batch_row["synced_at"],
                "fetched": batch_row["total_fetched"] or 0,
                "inserted": batch_row["total_inserted"] or 0,
                "updated": batch_row["total_updated"] or 0,
                "skipped": batch_row["total_skipped"] or 0,
            }
        else:
            latest_batch = {
                "synced_at": None,
                "fetched": 0,
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
            }

        cur.execute("""
            SELECT source_app, COUNT(*) as cnt
            FROM software_records
            GROUP BY source_app;
        """)
        app_counts = {row["source_app"]: row["cnt"] for row in cur.fetchall()}

        return {
            "total_records": total_records,
            "latest_sync": latest_batch,
            "app_breakdown": app_counts,
        }
    finally:
        conn.close()


def get_recent_sync_history(limit: int = 15, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get recent sync history entries."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT id, synced_at, software_name, fetched_count, inserted_count, updated_count, skipped_count
            FROM sync_history
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_recent_sql_imports(limit: int = 15, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get recent SQL file import tracking entries."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT id, filename, department, file_sha256, discovered_count,
                   inserted_count, updated_count, skipped_count, rejected_count,
                   failed_count, processed_at, status, error_message
            FROM sql_file_tracking
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def record_sql_file_tracking(
    filename: str,
    department: str,
    file_sha256: str,
    discovered: int,
    inserted: int,
    updated: int,
    skipped: int,
    rejected: int = 0,
    failed: int = 0,
    db_path: Optional[str] = None,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
) -> None:
    """Record an SQL file import event into sql_file_tracking."""
    now_ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO sql_file_tracking (
                    filename, department, file_sha256, discovered_count,
                    inserted_count, updated_count, skipped_count, rejected_count,
                    failed_count, processed_at, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                filename, department, file_sha256, discovered,
                inserted, updated, skipped, rejected,
                failed, now_ts, status, error_message
            ))
    finally:
        conn.close()


def get_sql_file_hash(filename: str, db_path: Optional[str] = None) -> Optional[str]:
    """Retrieve recorded SHA-256 hash for a processed SQL file."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT file_sha256 FROM sql_file_tracking WHERE filename = ? ORDER BY id DESC LIMIT 1;",
            (filename,)
        )
        row = cur.fetchone()
        return row["file_sha256"] if row else None
    except Exception:
        return None
    finally:
        conn.close()


def get_all_multi_db_records(department: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve software records from the authoritative database."""
    if department:
        return get_records_by_app(department, db_path=db_path)
    return get_all_records(db_path=db_path)


def get_multi_db_metrics_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute aggregated metrics directly from authoritative database."""
    return get_sync_metrics_summary(db_path=db_path)


def get_dept_db_path(department: str) -> str:
    """Returns the authoritative database path for any department."""
    return str(resolve_db_path())


def clear_database(db_path: Optional[str] = None) -> None:
    """
    Completely clear all data from the database.
    Resets tables to 0 records.
    """
    tables = [
        "software_records",
        "sync_history",
        "sql_file_tracking",
        "food_purchase_requests",
        "food_material_requests",
        "food_items",
        "food_inventory",
        "food_vendors",
        "cctv_cameras",
        "cctv_nvrs",
        "cctv_switches",
        "cctv_biometrics",
        "cctv_tickets",
        "transport_vehicles",
        "transport_fuel_entries",
        "transport_renewals",
        "transport_services",
    ]
    conn = get_connection(db_path)
    try:
        with conn:
            for t in tables:
                try:
                    conn.execute(f"DELETE FROM {t};")
                except Exception:
                    pass
            try:
                conn.execute("DELETE FROM sqlite_sequence;")
            except Exception:
                pass
        conn.execute("VACUUM;")
    finally:
        conn.close()


def clear_all_databases() -> None:
    """Purge records from authoritative database."""
    clear_database()
