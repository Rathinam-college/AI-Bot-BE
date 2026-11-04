"""
PostgreSQL Enterprise Database Adapter for Server AI Assistant.
Provides full PostgreSQL persistence, connection management, wrapper abstractions, and schema management.
Supports direct environment parameters (POSTGRES_HOST, etc.) and DATABASE_URL connection strings.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger("postgres_db")


def get_postgres_config() -> Dict[str, Any]:
    """Extract and parse PostgreSQL configuration from environment variables."""
    database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if database_url:
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "campus_ai_db",
            "user": parsed.username or "postgres",
            "password": parsed.password or "postgres",
        }
    return {
        "host": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "campus_ai_db"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    }


def is_postgres_configured() -> bool:
    """Check if the system is configured to use PostgreSQL."""
    engine = os.getenv("DB_ENGINE", "sqlite").lower()
    has_url = bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL"))
    return engine == "postgresql" or has_url


def is_postgres_available() -> bool:
    """Test if PostgreSQL server is actively running and reachable."""
    try:
        cfg = get_postgres_config()
        conn = psycopg2.connect(**cfg, connect_timeout=3)
        conn.close()
        return True
    except Exception as exc:
        logger.debug(f"PostgreSQL connection probe failed: {exc}")
        return False


class PostgresCursorWrapper:
    """
    Cursor wrapper that provides compatibility with sqlite3.Cursor.
    Translates '?' parameter placeholders to '%s' and provides lastrowid emulation.
    """

    def __init__(self, raw_cursor, raw_conn):
        self._cur = raw_cursor
        self._conn = raw_conn
        self._last_inserted_id = None

    @property
    def lastrowid(self):
        if self._last_inserted_id is not None:
            return self._last_inserted_id
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT LASTVAL();")
            row = cur.fetchone()
            cur.close()
            if row:
                return row[0]
        except Exception:
            pass
        return None

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description

    def execute(self, query: str, params: Any = None):
        if params is not None:
            pg_query = query.replace("?", "%s")
            return self._cur.execute(pg_query, params)
        return self._cur.execute(query)

    def executemany(self, query: str, seq_of_params: Any):
        pg_query = query.replace("?", "%s")
        return self._cur.executemany(pg_query, seq_of_params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def fetchmany(self, size: int = 1):
        return self._cur.fetchmany(size)

    def close(self):
        return self._cur.close()

    def __iter__(self):
        return iter(self._cur)


class PostgresConnectionWrapper:
    """
    Connection wrapper that provides compatibility with sqlite3.Connection.
    Supports execute(), cursor(), commit(), rollback(), and context manager protocol.
    """

    def __init__(self, raw_conn):
        self._raw_conn = raw_conn

    def cursor(self):
        return PostgresCursorWrapper(self._raw_conn.cursor(), self._raw_conn)

    def execute(self, query: str, params: Any = None):
        cur = self.cursor()
        cur.execute(query, params)
        return cur

    def commit(self):
        return self._raw_conn.commit()

    def rollback(self):
        return self._raw_conn.rollback()

    def close(self):
        return self._raw_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._raw_conn.rollback()
        else:
            self._raw_conn.commit()


def get_postgres_connection():
    """Create and return a wrapped PostgreSQL connection with DictCursor."""
    cfg = get_postgres_config()
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        cursor_factory=DictCursor,
        connect_timeout=5,
    )
    conn.autocommit = False
    return PostgresConnectionWrapper(conn)


def init_postgres_db() -> None:
    """Initialize PostgreSQL database schemas and tables."""
    conn = get_postgres_connection()
    try:
        with conn:
            cur = conn.cursor()
            # 1. Software records table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS software_records (
                    source_app VARCHAR(64) NOT NULL,
                    record_id VARCHAR(128) NOT NULL,
                    entity_type VARCHAR(64) DEFAULT 'Record',
                    table_name VARCHAR(128),
                    payload_json TEXT NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (source_app, record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_pg_records_app ON software_records(source_app);
                CREATE INDEX IF NOT EXISTS idx_pg_records_entity ON software_records(source_app, entity_type);
            """)

            # 2. Sync history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id SERIAL PRIMARY KEY,
                    synced_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    software_name VARCHAR(64) NOT NULL,
                    fetched_count INT NOT NULL,
                    inserted_count INT NOT NULL,
                    updated_count INT NOT NULL,
                    skipped_count INT NOT NULL
                );
            """)

            # 3. SQL file tracking table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sql_file_tracking (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(256) NOT NULL,
                    department VARCHAR(64) NOT NULL,
                    file_sha256 VARCHAR(64) NOT NULL,
                    discovered_count INT NOT NULL,
                    inserted_count INT NOT NULL,
                    updated_count INT NOT NULL,
                    skipped_count INT NOT NULL,
                    rejected_count INT DEFAULT 0,
                    failed_count INT DEFAULT 0,
                    processed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    status VARCHAR(32) DEFAULT 'SUCCESS',
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pg_file_tracking_fn ON sql_file_tracking(filename);
            """)

            # 4. Domain tables: Food
            cur.execute("""
                CREATE TABLE IF NOT EXISTS food_purchase_requests (
                    id VARCHAR(128) PRIMARY KEY,
                    pr_reference VARCHAR(128),
                    order_date VARCHAR(64),
                    delivery_date VARCHAR(64),
                    vendor_id VARCHAR(128),
                    total_amount NUMERIC(12, 2),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS food_material_requests (
                    id VARCHAR(128) PRIMARY KEY,
                    mr_number VARCHAR(128),
                    order_date VARCHAR(64),
                    target_date VARCHAR(64),
                    meal_type VARCHAR(64),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS food_items (
                    id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(256),
                    unit VARCHAR(32),
                    min_stock_level NUMERIC(12, 2),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS food_inventory (
                    id VARCHAR(128) PRIMARY KEY,
                    item_id VARCHAR(128),
                    current_quantity NUMERIC(12, 2),
                    min_stock_level NUMERIC(12, 2),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS food_vendors (
                    id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(256),
                    contact_person VARCHAR(128),
                    phone VARCHAR(64),
                    address TEXT,
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS food_procurement (
                    id VARCHAR(128) PRIMARY KEY,
                    purchase_date VARCHAR(64),
                    vendor_name VARCHAR(256),
                    invoice_number VARCHAR(128),
                    item_name VARCHAR(256),
                    quantity NUMERIC(12, 2),
                    unit VARCHAR(32),
                    unit_price NUMERIC(12, 2),
                    tax_rate NUMERIC(6, 2),
                    total_price NUMERIC(12, 2),
                    purchase_type VARCHAR(64),
                    raw_json TEXT
                );
            """)

            # 5. Domain tables: CCTV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cctv_cameras (
                    id VARCHAR(128) PRIMARY KEY,
                    camera_code VARCHAR(64),
                    camera_type VARCHAR(64),
                    device_type VARCHAR(64),
                    name VARCHAR(256),
                    block VARCHAR(128),
                    location VARCHAR(256),
                    ip_address VARCHAR(64),
                    status VARCHAR(64),
                    nvr_name VARCHAR(128),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS cctv_nvrs (
                    id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(256),
                    block VARCHAR(128),
                    ip_address VARCHAR(64),
                    total_channels INT,
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS cctv_switches (
                    id VARCHAR(128) PRIMARY KEY,
                    switch_name VARCHAR(256),
                    location VARCHAR(256),
                    ip_address VARCHAR(64),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS cctv_biometrics (
                    id VARCHAR(128) PRIMARY KEY,
                    name VARCHAR(256),
                    block VARCHAR(128),
                    location VARCHAR(256),
                    ip_address VARCHAR(64),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS cctv_tickets (
                    id VARCHAR(128) PRIMARY KEY,
                    ticket_number VARCHAR(128),
                    issue_description TEXT,
                    status VARCHAR(64),
                    block VARCHAR(128),
                    date_logged VARCHAR(64),
                    raw_json TEXT
                );
            """)

            # 6. Domain tables: Transport
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transport_vehicles (
                    id VARCHAR(128) PRIMARY KEY,
                    vehicle_number VARCHAR(64),
                    driver_name VARCHAR(128),
                    driver_contact_no VARCHAR(64),
                    seating_capacity INT,
                    make VARCHAR(64),
                    model VARCHAR(64),
                    status VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS transport_fuel_entries (
                    id VARCHAR(128) PRIMARY KEY,
                    vehicle_number VARCHAR(64),
                    date VARCHAR(64),
                    liters NUMERIC(10, 2),
                    amount NUMERIC(12, 2),
                    total_km NUMERIC(12, 2),
                    bill_no VARCHAR(64),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS transport_renewals (
                    id VARCHAR(128) PRIMARY KEY,
                    vehicle_number VARCHAR(64),
                    document_type VARCHAR(64),
                    new_expiry_date VARCHAR(64),
                    status VARCHAR(64),
                    amount_paid NUMERIC(12, 2),
                    raw_json TEXT
                );
                CREATE TABLE IF NOT EXISTS transport_services (
                    id VARCHAR(128) PRIMARY KEY,
                    vehicle_number VARCHAR(64),
                    issue TEXT,
                    status VARCHAR(64),
                    raw_json TEXT
                );
            """)

            # 7. User Authentication & Chat Tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(64) UNIQUE NOT NULL,
                    password_hash VARCHAR(256) NOT NULL,
                    full_name VARCHAR(128) NOT NULL,
                    email VARCHAR(128),
                    role VARCHAR(32) NOT NULL DEFAULT 'member',
                    department VARCHAR(64) DEFAULT 'General',
                    permissions TEXT NOT NULL DEFAULT '["dashboard","chatbot","reports"]',
                    is_active INT NOT NULL DEFAULT 1,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_login TIMESTAMP WITH TIME ZONE
                );
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token VARCHAR(128) PRIMARY KEY,
                    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id INT REFERENCES users(id) ON DELETE SET NULL,
                    title VARCHAR(256) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    model VARCHAR(64),
                    grounded_count INT DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pg_chat_msg_session ON chat_messages(session_id);
            """)

            # 8. Seed default accounts if users table is empty
            cur.execute("SELECT COUNT(*) as cnt FROM users;")
            row = cur.fetchone()
            if row and row["cnt"] == 0:
                default_users = [
                    ("admin", "pbkdf2:sha256:600000$admin_salt$2b4566c1f1ec750a98f16182c18d601b3da6f238927f8841006540645063051a", "Super Admin", "admin@rathinam.in", "admin", "Management", '["dashboard","chatbot","user_control","reports","upload_sql","clear_db"]'),
                    ("sarah_mgr", "pbkdf2:sha256:600000$mgr_salt$3a9f029a77914fbb3a99281a8b1f8f08a514d238927f8841006540645063052b", "Sarah Jenkins", "sarah.j@rathinam.in", "manager", "CCTV & Security", '["dashboard","chatbot","reports","upload_sql"]'),
                    ("david_op", "pbkdf2:sha256:600000$op_salt$4c8e138b88025ecc4b00392b9c2a9a19b625e349038a9952117651756174163c", "David Kumar", "david.k@rathinam.in", "member", "Transport Fleet", '["dashboard","chatbot","reports"]'),
                ]
                cur.executemany("""
                    INSERT INTO users (username, password_hash, full_name, email, role, department, permissions, is_active, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 1, NOW());
                """, default_users)

        logger.info("PostgreSQL database tables and schemas initialized successfully.")
    except Exception as exc:
        logger.error(f"Failed to initialize PostgreSQL database: {exc}", exc_info=True)
        raise
    finally:
        conn.close()
