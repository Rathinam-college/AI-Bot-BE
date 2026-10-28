"""
Dynamic SQL Schema Discovery and Safe Read-Only Execution Engine.
Provides schema inspection, query validation, and deterministic aggregation for app.db.
"""
import os
import re
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sql_query_engine")

DANGEROUS_SQL_PATTERNS = [
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|ATTACH|DETACH|VACUUM|REINDEX)\b",
    r"\b(PRAGMA\s+(?!table_info|index_list|foreign_key_list))\b",
    r"\b(EXEC|EXECUTE|CREATE)\b",
    r"--",
    r";.*\S", # Multiple statements
]

class SQLQueryEngine:
    """
    Safe read-only SQL query engine with automatic schema discovery.
    Strictly validates queries against known tables and prevents any write operations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = self._resolve_db_path(db_path)

    def _resolve_db_path(self, db_path: Optional[str] = None) -> str:
        if db_path:
            p = Path(db_path)
            if p.is_absolute():
                return str(p)
            root = Path(__file__).resolve().parent.parent
            return str(root / db_path)
        root = Path(__file__).resolve().parent.parent
        return str(root / "database" / "app.db")

    def discover_schema(self) -> Dict[str, Any]:
        """
        Inspect SQLite database metadata and build a structured schema catalog.
        """
        if not os.path.exists(self.db_path):
            return {"tables": {}, "views": {}}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get all tables
        cur.execute("""
            SELECT name, type, sql FROM sqlite_master 
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        entities = cur.fetchall()

        tables_meta = {}
        for ent in entities:
            tname = ent["name"]
            ttype = ent["type"]

            # Columns info
            cur.execute(f'PRAGMA table_info("{tname}");')
            cols_info = cur.fetchall()

            cols = []
            pk_cols = []
            amount_cols = []
            date_cols = []
            status_cols = []
            id_cols = []
            name_cols = []

            for col in cols_info:
                cname = col["name"]
                ctype = col["type"].upper()
                is_pk = bool(col["pk"])
                cols.append({"name": cname, "type": ctype, "pk": is_pk})
                if is_pk:
                    pk_cols.append(cname)

                clower = cname.lower()
                if any(k in clower for k in ["amount", "total", "price", "cost", "quantity", "liters", "paid"]):
                    amount_cols.append(cname)
                if any(k in clower for k in ["date", "time", "created_at", "updated_at", "synced_at", "processed_at", "expiry"]):
                    date_cols.append(cname)
                if any(k in clower for k in ["status", "state"]):
                    status_cols.append(cname)
                if any(k in clower for k in ["id", "code", "reference", "number", "ip"]):
                    id_cols.append(cname)
                if any(k in clower for k in ["name", "title", "description", "driver"]):
                    name_cols.append(cname)

            # Row count
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{tname}";')
                count = cur.fetchone()[0]
            except Exception:
                count = 0

            tables_meta[tname] = {
                "type": ttype,
                "columns": cols,
                "primary_keys": pk_cols,
                "row_count": count,
                "amount_columns": amount_cols,
                "date_columns": date_cols,
                "status_columns": status_cols,
                "id_columns": id_cols,
                "name_columns": name_cols,
            }

        conn.close()
        return tables_meta

    def get_schema_summary_text(self) -> str:
        """
        Produce a concise text summary of available tables and columns for LLM prompt context.
        """
        schema = self.discover_schema()
        lines = ["### AVAILABLE DATABASE TABLES & SCHEMAS (app.db):"]
        for tname, meta in sorted(schema.items()):
            col_list = ", ".join([f"{c['name']} ({c['type']})" for c in meta["columns"][:12]])
            more = f" ...+{len(meta['columns'])-12} more" if len(meta["columns"]) > 12 else ""
            lines.append(f"- **{tname}** ({meta['row_count']} rows): [{col_list}{more}]")
            if meta["amount_columns"]:
                lines.append(f"  * Monetary/Metric Columns: {', '.join(meta['amount_columns'])}")
            if meta["date_columns"]:
                lines.append(f"  * Date Columns: {', '.join(meta['date_columns'])}")
        return "\n".join(lines)

    def validate_query(self, query: str, allowed_tables: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validate that the query is safe, read-only, and accesses only valid tables.
        """
        q = query.strip()
        if not (q.upper().startswith("SELECT") or q.upper().startswith("WITH")):
            return False, "Query must start with SELECT or WITH."

        for pattern in DANGEROUS_SQL_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return False, f"Dangerous or disallowed SQL keyword detected: {pattern}"

        if allowed_tables:
            # Check table names in query
            table_tokens = re.findall(r"\bFROM\s+([a-zA-Z0-9_\-]+)|\bJOIN\s+([a-zA-Z0-9_\-]+)", q, re.IGNORECASE)
            for t_tuple in table_tokens:
                found_t = t_tuple[0] or t_tuple[1]
                if found_t and found_t.lower() not in [t.lower() for t in allowed_tables]:
                    return False, f"Table '{found_t}' is not recognized in the database schema."

        return True, "Valid"

    def execute_read_only_query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None, max_rows: int = 100
    ) -> Dict[str, Any]:
        """
        Execute a validated SELECT query in strict read-only mode with timeout and row limit.
        """
        schema = self.discover_schema()
        allowed_tables = list(schema.keys())

        valid, msg = self.validate_query(query, allowed_tables)
        if not valid:
            logger.warning(f"Rejected unsafe query: {query} (Reason: {msg})")
            return {"status": "error", "message": msg, "columns": [], "rows": [], "count": 0}

        # Enforce LIMIT
        clean_q = query.strip().rstrip(";")
        if not re.search(r"\bLIMIT\s+\d+", clean_q, re.IGNORECASE):
            clean_q = f"{clean_q} LIMIT {max_rows}"

        conn = None
        try:
            # Open read-only connection
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON;")

            cur = conn.cursor()
            if params:
                cur.execute(clean_q, params)
            else:
                cur.execute(clean_q)

            col_names = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(r) for r in cur.fetchall()]

            return {
                "status": "success",
                "query": clean_q,
                "columns": col_names,
                "rows": rows,
                "count": len(rows),
            }
        except Exception as exc:
            logger.error(f"Error executing safe query: {clean_q} -> {exc}")
            return {
                "status": "error",
                "message": str(exc),
                "query": clean_q,
                "columns": [],
                "rows": [],
                "count": 0,
            }
        finally:
            if conn:
                conn.close()

    def query_monthly_procurement(self, year_month: Optional[str] = "2026-10") -> Dict[str, Any]:
        """
        Deterministic, database-derived financial aggregation for Food Purchase Requests.
        """
        if not os.path.exists(self.db_path):
            return {"total_amount": 0.0, "total_count": 0, "status_breakdown": {}, "top_prs": []}

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        date_filter = f"{year_month}%" if year_month else "%"

        # 1. Totals & breakdown
        cur.execute("""
            SELECT status, COUNT(*) as count, SUM(total_amount) as sum_amt
            FROM food_purchase_requests
            WHERE order_date LIKE ? OR delivery_date LIKE ?
            GROUP BY status;
        """, (date_filter, date_filter))

        status_breakdown = {}
        total_count = 0
        total_amount = 0.0

        for r in cur.fetchall():
            st = (r["status"] or "UNKNOWN").upper()
            cnt = r["count"]
            amt = r["sum_amt"] or 0.0
            status_breakdown[st] = {"count": cnt, "amount": amt}
            total_count += cnt
            total_amount += amt

        # 2. Top purchase orders
        cur.execute("""
            SELECT pr_reference, order_date, vendor_id, total_amount, status
            FROM food_purchase_requests
            WHERE order_date LIKE ? OR delivery_date LIKE ?
            ORDER BY total_amount DESC
            LIMIT 5;
        """, (date_filter, date_filter))
        top_prs = [dict(r) for r in cur.fetchall()]

        # 3. Related departmental records in that month
        # Material Requests
        cur.execute("""
            SELECT COUNT(*) FROM food_material_requests
            WHERE order_date LIKE ? OR target_date LIKE ?;
        """, (date_filter, date_filter))
        mr_count = cur.fetchone()[0]

        # Vehicle Renewals
        cur.execute("""
            SELECT id, vehicle_number, document_type, new_expiry_date, status
            FROM transport_renewals
            WHERE new_expiry_date LIKE ?;
        """, (date_filter,))
        renewals = [dict(r) for r in cur.fetchall()]

        # CCTV Tickets
        cur.execute("""
            SELECT COUNT(*) FROM cctv_tickets
            WHERE date_logged LIKE ?;
        """, (date_filter,))
        ticket_count = cur.fetchone()[0]

        conn.close()

        return {
            "period": year_month or "All Time",
            "total_amount": total_amount,
            "total_count": total_count,
            "status_breakdown": status_breakdown,
            "top_prs": top_prs,
            "material_requests_count": mr_count,
            "vehicle_renewals": renewals,
            "cctv_tickets_count": ticket_count,
        }

sql_query_engine = SQLQueryEngine()
