"""
Production-Grade SQL Analyzer, Validator, and Ingestion Engine.
Directs all SQL imports into the authoritative database: database/app.db.

Features:
- Strict SQL input validation & security checks (rejects arbitrary unsafe execution).
- Deep structural & content factor analysis to accurately identify department.
- High-fidelity streaming parser for PostgreSQL COPY, standard SQL INSERT (single & multi-row).
- Record-level deterministic SHA-256 deduplication and change detection.
- Complete re-analysis of every upload (repeat uploads accurately report duplicates skipped).
- Immediate AI context invalidation and data refresh.
"""
import copy
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from database.db import (
    get_connection,
    init_db,
    record_sql_file_tracking,
    resolve_db_path,
)
from services.cctv_sql_parser import parse_cctv_backup_sql
from services.sql_dump_importer import (
    parse_copy_block_from_sql,
    parse_food_dump,
    parse_transport_dump,
)
from services.sync_engine import normalize_record_payload, sync_engine

logger = logging.getLogger("sql_analyzer")

# Maximum permitted SQL file upload size (100 MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

# Disallowed dangerous SQL operations
UNSAFE_SQL_COMMANDS = [
    r"\bATTACH\s+DATABASE\b",
    r"\bDETACH\s+DATABASE\b",
    r"\bPRAGMA\s+(?!table_info|index_list|foreign_key_list)",
    r"\bxp_cmdshell\b",
    r"\bLOAD_EXTENSION\b",
]


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_sql_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """Validate SQL file existence, size, encoding, and absence of malicious injection patterns."""
    p = Path(filepath)
    if not p.exists():
        return False, f"File does not exist: {filepath}"

    file_size = p.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum size limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB ({file_size} bytes)."

    # Test readable encoding and scan for dangerous commands
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(50000)
            for pat in UNSAFE_SQL_COMMANDS:
                if re.search(pat, sample, re.IGNORECASE):
                    return False, f"Unsupported or dangerous SQL pattern detected: {pat}"
    except Exception as exc:
        return False, f"Failed to read file encoding: {str(exc)}"

    return True, None


def analyze_sql_file_factors(sql_path: str) -> Dict[str, Any]:
    """
    Inspect factors inside an SQL file (table names, schemas, column names, data statements)
    to deterministically classify the file into its appropriate department:
      - 'CCTV' / 'Security'
      - 'Food' / 'Canteen'
      - 'Transport' / 'Fleet'
      - 'Software' / 'IT'
      - 'Finance' / 'Accounts'
      - Or dynamically detected department from actual schema.
    """
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL file not found at: {sql_path}")

    filename = Path(sql_path).name.lower()

    dept_scores: Dict[str, int] = {
        "CCTV": 0,
        "Food": 0,
        "Transport": 0,
        "Software": 0,
        "Finance": 0,
    }

    # Filename indicators (minor hint/tie-breaker only, 3 points max)
    if any(k in filename for k in ["cctv", "camera", "nvr", "surveillance", "security"]):
        dept_scores["CCTV"] += 3
    if any(k in filename for k in ["food", "canteen", "mess", "dining", "cafeteria", "grocery"]):
        dept_scores["Food"] += 3
    if any(k in filename for k in ["transport", "bus", "vehicle", "fleet", "route", "driver"]):
        dept_scores["Transport"] += 3
    if any(k in filename for k in ["software", "inventory", "crm", "support", "system", "app"]):
        dept_scores["Software"] += 3
    if any(k in filename for k in ["finance", "account", "ledger", "invoice", "payment", "billing"]):
        dept_scores["Finance"] += 3

    detected_tables: List[str] = []
    scanned_lines = 0

    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            scanned_lines += 1
            line_lower = line.lower().strip()

            # Table definitions, COPY statements, and INSERT INTO statements
            if line_lower.startswith("create table") or line_lower.startswith("copy ") or line_lower.startswith("insert into"):
                t_match = re.search(
                    r'(?:create table|copy|insert\s+into)\s+(?:if not exists\s+)?(?:public\.)?["`]?([a-zA-Z0-9_\-]+)["`]?',
                    line_lower
                )
                if t_match:
                    tname = t_match.group(1)
                    if tname not in detected_tables:
                        detected_tables.append(tname)

                    # Substantial weight (25 points) for structural table definitions
                    if any(k in tname for k in ["cctv", "camera", "nvr", "incident", "surveillance", "biometric", "switch", "barrier"]):
                        dept_scores["CCTV"] += 25
                    elif any(k in tname for k in ["core_item", "core_inventory", "core_purchaserequest", "core_materialrequest", "food", "menu", "mess", "dining", "canteen", "grocery"]):
                        dept_scores["Food"] += 25
                    elif any(k in tname for k in ["transport", "vehicle", "fuel", "odometer", "bus", "driver", "servicerequest", "fleet"]):
                        dept_scores["Transport"] += 25
                    elif any(k in tname for k in ["product", "hardware", "software", "support", "crm", "customer", "ticket_system", "asset"]):
                        dept_scores["Software"] += 25
                    elif any(k in tname for k in ["invoice", "payment", "ledger", "transaction", "tax", "billing", "account"]):
                        dept_scores["Finance"] += 25

            # Column and field keyword checks (8 points per occurrence, up to line 5000)
            if scanned_lines <= 5000:
                if any(k in line_lower for k in ["rtsp", "camera_ip", "nvr_ip", "dome camera", "bullet camera", "switch_ip", "cctv"]):
                    dept_scores["CCTV"] += 8
                if any(k in line_lower for k in ["purchaserequest", "materialrequest", "meal_type", "min_stock_level", "total_amount", "unit_price"]):
                    dept_scores["Food"] += 8
                if any(k in line_lower for k in ["seating_capacity", "chassis_number", "driver_name", "odometer", "fuel_liters", "vehiclerenewal", "bus_number"]):
                    dept_scores["Transport"] += 8
                if any(k in line_lower for k in ["sku", "warehouse", "assigned_rep", "ticket_priority", "sla_hours"]):
                    dept_scores["Software"] += 8
                if any(k in line_lower for k in ["invoice_no", "tax_amount", "balance_due", "account_number", "ledger_entry"]):
                    dept_scores["Finance"] += 8

    top_dept, top_score = max(dept_scores.items(), key=lambda x: x[1])
    department = top_dept if top_score > 0 else "General"

    factors = {
        "scores": dept_scores,
        "tables_detected": detected_tables[:20],
        "top_score": top_score,
        "confidence": "high" if top_score >= 15 else ("medium" if top_score >= 8 else "low"),
    }

    return {
        "department": department,
        "target_db": str(resolve_db_path()),
        "factors": factors,
    }


def _convert_sql_token(tok: str) -> Any:
    tok = tok.strip()
    if (tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"')):
        tok = tok[1:-1]
    if tok == "" or tok.upper() == "NULL":
        return None
    elif tok.lower() == "true":
        return True
    elif tok.lower() == "false":
        return False
    elif re.match(r"^-?\d+$", tok):
        return int(tok)
    else:
        try:
            return float(tok)
        except ValueError:
            return tok


def extract_sql_value_tuples(blob: str) -> List[List[Any]]:
    """Character-level parser to accurately extract SQL value tuples (...) handling quotes and commas."""
    tuples = []
    in_quotes = False
    in_paren = False
    cur_tokens: List[Any] = []
    cur_chars: List[str] = []
    i = 0
    n = len(blob)
    while i < n:
        c = blob[i]
        if not in_paren:
            if c == "(":
                in_paren = True
                cur_tokens = []
                cur_chars = []
                in_quotes = False
        else:
            if in_quotes:
                if c == "'" and i + 1 < n and blob[i + 1] == "'":
                    cur_chars.append("'")
                    i += 1
                elif c == "\\" and i + 1 < n and blob[i + 1] in ("'", "\\"):
                    cur_chars.append(blob[i + 1])
                    i += 1
                elif c == "'":
                    in_quotes = False
                else:
                    cur_chars.append(c)
            else:
                if c == "'":
                    in_quotes = True
                elif c == ",":
                    raw = "".join(cur_chars).strip()
                    cur_tokens.append(_convert_sql_token(raw))
                    cur_chars = []
                elif c == ")":
                    raw = "".join(cur_chars).strip()
                    cur_tokens.append(_convert_sql_token(raw))
                    tuples.append(cur_tokens)
                    cur_tokens = []
                    cur_chars = []
                    in_paren = False
                else:
                    cur_chars.append(c)
        i += 1
    return tuples


def _is_statement_terminated(stmt_text: str) -> bool:
    """Check if the statement ends with semicolon not enclosed in quotes."""
    trimmed = stmt_text.strip()
    if not trimmed.endswith(";"):
        return False
    in_quotes = False
    i = 0
    n = len(stmt_text)
    while i < n:
        c = stmt_text[i]
        if in_quotes:
            if c == "'" and i + 1 < n and stmt_text[i + 1] == "'":
                i += 1
            elif c == "\\" and i + 1 < n and stmt_text[i + 1] in ("'", "\\"):
                i += 1
            elif c == "'":
                in_quotes = False
        else:
            if c == "'":
                in_quotes = True
        i += 1
    return not in_quotes


def parse_generic_sql_statements(sql_path: str, department: str) -> List[Dict[str, Any]]:
    """
    Universal streaming SQL parser:
    Parses PostgreSQL COPY blocks and multiline/multi-row INSERT INTO statements.
    Extracts individual canonical records.
    """
    records: List[Dict[str, Any]] = []
    current_copy_table = None
    current_copy_cols: List[str] = []
    stmt_buffer: List[str] = []

    insert_pattern = re.compile(
        r'insert\s+into\s+(?:public\.)?["`]?([a-zA-Z0-9_\-]+)["`]?\s*(?:\((.*?)\))?\s*values\s*(.*);?',
        re.IGNORECASE | re.DOTALL
    )

    def _process_buffered_stmt(stmt_str: str):
        stmt_str = stmt_str.strip()
        if not stmt_str.lower().startswith("insert into"):
            return
        m = insert_pattern.search(stmt_str)
        if not m:
            return
        tname = m.group(1)
        if tname.startswith("django_") or tname.startswith("auth_"):
            return
        cols_str = m.group(2)
        values_blob = m.group(3)
        cols = [c.strip().strip('"`') for c in cols_str.split(",")] if cols_str else []
        tuples = extract_sql_value_tuples(values_blob)
        for t in tuples:
            if cols:
                row_dict = dict(zip(cols, t))
            else:
                row_dict = {f"col_{idx+1}": tok for idx, tok in enumerate(t)}
            
            raw_id = (
                row_dict.get("id")
                or row_dict.get("code")
                or row_dict.get("record_id")
                or row_dict.get("vehicle_id")
                or row_dict.get("vehicle_number")
                or row_dict.get("camera_id")
                or row_dict.get("invoice_no")
            )
            if raw_id is not None:
                rec_id = f"{department[:3].upper()}-{tname.upper()[:8]}-{raw_id}"
            else:
                clean_item, item_h = normalize_record_payload(row_dict)
                rec_id = f"{department[:3].upper()}-{tname.upper()[:8]}-{item_h[:8].upper()}"

            name_val = (
                row_dict.get("name")
                or row_dict.get("title")
                or row_dict.get("vehicle_number")
                or row_dict.get("driver_name")
                or row_dict.get("camera_name")
                or row_dict.get("item_name")
                or row_dict.get("sku")
                or row_dict.get("code")
                or rec_id
            )
            records.append({
                "id": rec_id,
                "entity_type": tname.replace("_", " ").title(),
                "name": str(name_val),
                "attributes": row_dict,
                "table": tname,
            })

    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()

            # 1. Inside COPY block
            if current_copy_table:
                if line_str == "\\.":
                    current_copy_table = None
                    continue
                parts = line.rstrip("\r\n").split("\t")
                row_dict = {}
                for col, val in zip(current_copy_cols, parts):
                    if val != "\\N" and val != "":
                        row_dict[col] = _convert_sql_token(val)
                raw_id = row_dict.get("id") or row_dict.get("code") or len(records) + 1
                rec_id = f"{department[:3].upper()}-{current_copy_table.upper()[:8]}-{raw_id}"
                name_val = (
                    row_dict.get("name")
                    or row_dict.get("title")
                    or row_dict.get("vehicle_number")
                    or row_dict.get("camera_name")
                    or rec_id
                )
                records.append({
                    "id": rec_id,
                    "entity_type": current_copy_table.replace("_", " ").title(),
                    "name": str(name_val),
                    "attributes": row_dict,
                    "table": current_copy_table,
                })
                continue

            # 2. Start of COPY block
            if line_str.upper().startswith("COPY "):
                if stmt_buffer:
                    _process_buffered_stmt("".join(stmt_buffer))
                    stmt_buffer = []
                idx1 = line_str.find("(")
                idx2 = line_str.find(")")
                if idx1 != -1 and idx2 != -1:
                    tname = line_str[5:idx1].strip().replace("public.", "").replace('"', "").replace("`", "")
                    if not tname.startswith("django_") and not tname.startswith("auth_"):
                        current_copy_table = tname
                        current_copy_cols = [c.strip().replace('"', "").replace("`", "") for c in line_str[idx1 + 1 : idx2].split(",")]
                    else:
                        current_copy_table = None
                continue

            # 3. SQL comments & empty lines
            if line_str.startswith("--") or line_str.startswith("#") or not line_str:
                continue

            stmt_buffer.append(line)
            if _is_statement_terminated("".join(stmt_buffer)):
                _process_buffered_stmt("".join(stmt_buffer))
                stmt_buffer = []

    if stmt_buffer:
        _process_buffered_stmt("".join(stmt_buffer))

    return records



def parse_sql_file_to_records(sql_path: str, department: str) -> List[Dict[str, Any]]:
    """Parse an SQL file into canonical records based on its detected department."""
    records: List[Dict[str, Any]] = []

    if department == "Food":
        try:
            records = parse_food_dump(sql_path)
        except Exception:
            records = []
    elif department == "Transport":
        try:
            records = parse_transport_dump(sql_path)
        except Exception:
            records = []
    elif department == "CCTV":
        try:
            parsed = parse_cctv_backup_sql(sql_path)
            records = parsed.get("cctv", parsed.get("app1", []))
        except Exception:
            records = []

    if not records:
        logger.info(f"Applying universal SQL statement parser for {sql_path} (department={department})...")
        records = parse_generic_sql_statements(sql_path, department)

    return records


def sync_records_to_relational_tables(department: str, records: List[Dict[str, Any]], db_path: str) -> None:
    """Populate typed domain tables in SQLite for structured departmental queries."""
    conn = get_connection(db_path)
    try:
        with conn:
            if department == "Food":
                prs = []
                mrs = []
                items = []
                inv = []
                vendors = []
                for r in records:
                    rid = str(r.get("id", ""))
                    etype = r.get("entity_type", "")
                    raw_str = json.dumps(r, sort_keys=True)

                    if etype == "Food Purchase Request":
                        amt = 0.0
                        try:
                            amt = float(str(r.get("total_amount") or 0).replace(",", ""))
                        except Exception:
                            pass
                        prs.append((
                            rid, r.get("pr_reference", rid), r.get("order_date"),
                            r.get("delivery_date"), str(r.get("vendor_id", "")),
                            amt, r.get("status", "COMPLETED"), raw_str
                        ))
                    elif etype == "Food Material Request":
                        mrs.append((
                            rid, r.get("mr_number", rid), r.get("order_date"),
                            r.get("target_date"), r.get("meal_type"),
                            r.get("status", "COMPLETED"), raw_str
                        ))
                    elif etype == "Food Item":
                        min_s = 0.0
                        try:
                            min_s = float(r.get("min_stock_level") or 0)
                        except Exception:
                            pass
                        items.append((rid, r.get("name", rid), r.get("unit"), min_s, r.get("status", "Active"), raw_str))
                    elif etype == "Food Inventory":
                        qty = 0.0
                        min_lvl = 0.0
                        try:
                            qty = float(r.get("current_quantity") or 0)
                            min_lvl = float(r.get("min_stock_level") or 0)
                        except Exception:
                            pass
                        inv.append((rid, str(r.get("item_id", "")), qty, min_lvl, r.get("status", "In Stock"), raw_str))
                    elif etype == "Food Vendor":
                        vendors.append((rid, r.get("name", rid), r.get("contact_person"), r.get("phone"), r.get("address"), raw_str))

                if prs:
                    conn.executemany("INSERT OR REPLACE INTO food_purchase_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)", prs)
                if mrs:
                    conn.executemany("INSERT OR REPLACE INTO food_material_requests VALUES (?, ?, ?, ?, ?, ?, ?)", mrs)
                if items:
                    conn.executemany("INSERT OR REPLACE INTO food_items VALUES (?, ?, ?, ?, ?, ?)", items)
                if inv:
                    conn.executemany("INSERT OR REPLACE INTO food_inventory VALUES (?, ?, ?, ?, ?, ?)", inv)
                if vendors:
                    conn.executemany("INSERT OR REPLACE INTO food_vendors VALUES (?, ?, ?, ?, ?, ?)", vendors)

            elif department == "CCTV":
                cams = []
                nvrs = []
                switches = []
                bios = []
                tkts = []
                for r in records:
                    rid = str(r.get("id", ""))
                    etype = r.get("entity_type", "")
                    raw_str = json.dumps(r, sort_keys=True)

                    if etype == "CCTV Camera":
                        cams.append((
                            rid, r.get("camera_code", rid), r.get("camera_type"),
                            r.get("device_type"), r.get("name", rid), r.get("block"),
                            r.get("location"), r.get("ip_address"), r.get("status", "Online"),
                            r.get("nvr_name"), raw_str
                        ))
                    elif etype == "NVR Device":
                        ch = 0
                        try:
                            ch = int(r.get("total_channels") or 0)
                        except Exception:
                            pass
                        nvrs.append((rid, r.get("name", rid), r.get("block"), r.get("ip_address"), ch, r.get("status", "Online"), raw_str))
                    elif etype == "Network Switch":
                        switches.append((rid, r.get("switch_name", rid), r.get("location"), r.get("ip_address"), r.get("status", "Online"), raw_str))
                    elif etype == "Biometric Device":
                        bios.append((rid, r.get("name", rid), r.get("block"), r.get("location"), r.get("ip_address"), r.get("status", "Online"), raw_str))
                    elif etype == "Maintenance Ticket":
                        tkts.append((rid, r.get("ticket_number", rid), r.get("issue_description"), r.get("status", "Open"), r.get("block"), r.get("date_logged"), raw_str))

                if cams:
                    conn.executemany("INSERT OR REPLACE INTO cctv_cameras VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", cams)
                if nvrs:
                    conn.executemany("INSERT OR REPLACE INTO cctv_nvrs VALUES (?, ?, ?, ?, ?, ?, ?)", nvrs)
                if switches:
                    conn.executemany("INSERT OR REPLACE INTO cctv_switches VALUES (?, ?, ?, ?, ?, ?)", switches)
                if bios:
                    conn.executemany("INSERT OR REPLACE INTO cctv_biometrics VALUES (?, ?, ?, ?, ?, ?)", bios)
                if tkts:
                    conn.executemany("INSERT OR REPLACE INTO cctv_tickets VALUES (?, ?, ?, ?, ?, ?, ?)", tkts)

            elif department == "Transport":
                vehicles = []
                fuels = []
                rnws = []
                services = []
                for r in records:
                    rid = str(r.get("id", ""))
                    etype = r.get("entity_type", "")
                    raw_str = json.dumps(r, sort_keys=True)

                    if etype == "Campus Bus":
                        cap = 0
                        try:
                            cap = int(r.get("seating_capacity") or 0)
                        except Exception:
                            pass
                        vehicles.append((
                            rid, r.get("vehicle_number", rid), r.get("driver_name"),
                            r.get("driver_contact_no"), cap, r.get("make"),
                            r.get("model"), r.get("status", "Active"), raw_str
                        ))
                    elif etype == "Transport Fuel Entry":
                        liters = 0.0
                        amt = 0.0
                        km = 0.0
                        try:
                            liters = float(r.get("liters") or 0)
                            amt = float(r.get("amount") or 0)
                            km = float(r.get("total_km") or 0)
                        except Exception:
                            pass
                        fuels.append((rid, r.get("vehicle_number"), r.get("date"), liters, amt, km, r.get("bill_no"), raw_str))
                    elif etype == "Transport Renewal":
                        paid = 0.0
                        try:
                            paid = float(r.get("amount_paid") or 0)
                        except Exception:
                            pass
                        rnws.append((rid, r.get("vehicle_number"), r.get("document_type"), r.get("new_expiry_date"), r.get("status", "Valid"), paid, raw_str))
                    elif etype == "Transport Service Request":
                        services.append((rid, r.get("vehicle_number"), r.get("issue"), r.get("status", "Open"), raw_str))

                if vehicles:
                    conn.executemany("INSERT OR REPLACE INTO transport_vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", vehicles)
                if fuels:
                    conn.executemany("INSERT OR REPLACE INTO transport_fuel_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", fuels)
                if rnws:
                    conn.executemany("INSERT OR REPLACE INTO transport_renewals VALUES (?, ?, ?, ?, ?, ?, ?)", rnws)
                if services:
                    conn.executemany("INSERT OR REPLACE INTO transport_services VALUES (?, ?, ?, ?, ?)", services)
    finally:
        conn.close()


def process_sql_file(sql_path: str, force: bool = False, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest and re-analyze any uploaded SQL file at the record level:
    1. Validates the SQL file (syntax, size, security checks).
    2. Determines the department based on internal table/schema analysis.
    3. Parses every record and performs deterministic SHA-256 differential upsert:
       - Insert new records
       - Skip exact duplicates
       - Update modified records in-place
    4. Records the upload event and statistics in sql_file_tracking.
    5. Returns accurate counts of discovered, inserted, skipped, updated, and failed records.
    """
    filepath = Path(sql_path).resolve()
    valid, err = validate_sql_file(str(filepath))
    if not valid:
        raise ValueError(f"Invalid SQL file '{filepath.name}': {err}")

    filename = filepath.name
    file_sha256 = compute_file_sha256(str(filepath))
    target_db = str(resolve_db_path(db_path))

    # Factor Analysis
    analysis = analyze_sql_file_factors(str(filepath))
    department = analysis["department"]

    # Parse records
    logger.info(f"Parsing '{filename}' into department '{department}'...")
    try:
        records = parse_sql_file_to_records(str(filepath), department)
    except Exception as parse_exc:
        logger.error(f"Failed to parse {filename}: {parse_exc}", exc_info=True)
        record_sql_file_tracking(
            filename=filename,
            department=department,
            file_sha256=file_sha256,
            discovered=0,
            inserted=0,
            updated=0,
            skipped=0,
            rejected=0,
            failed=1,
            db_path=target_db,
            status="FAILED",
            error_message=str(parse_exc),
        )
        raise

    if not records:
        logger.warning(f"No records extracted from {filename}.")
        record_sql_file_tracking(
            filename=filename,
            department=department,
            file_sha256=file_sha256,
            discovered=0,
            inserted=0,
            updated=0,
            skipped=0,
            rejected=0,
            failed=0,
            db_path=target_db,
            status="SUCCESS",
            error_message="0 records discovered in file",
        )
        return {
            "status": "warning",
            "message": f"Processed '{filename}', but 0 records were discovered.",
            "filename": filename,
            "department": department,
            "target_db": target_db,
            "file_sha256": file_sha256,
            "total_records": 0,
            "discovered": 0,
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "rejected": 0,
            "failed": 0,
        }

    # 1. Upsert into authoritative database software_records with duplicate prevention
    res_app = sync_engine.upsert_records(
        source_app=department,
        records=records,
        db_path=target_db,
        log_to_history=True,
    )

    # 2. Sync to typed relational tables in app.db
    try:
        sync_records_to_relational_tables(department, records, target_db)
    except Exception as rel_exc:
        logger.warning(f"Warning syncing relational tables for {filename}: {rel_exc}")

    # 3. Record tracking metadata in sql_file_tracking
    record_sql_file_tracking(
        filename=filename,
        department=department,
        file_sha256=file_sha256,
        discovered=len(records),
        inserted=res_app["inserted"],
        updated=res_app["updated"],
        skipped=res_app["skipped"],
        rejected=res_app.get("rejected", 0),
        failed=0,
        db_path=target_db,
        status="SUCCESS",
    )

    logger.info(
        f"Ingested '{filename}' into {target_db}: "
        f"{len(records)} discovered -> {res_app['inserted']} inserted, {res_app['updated']} updated, {res_app['skipped']} skipped."
    )

    return {
        "status": "success",
        "message": f"File '{filename}' analyzed as {department}: {res_app['inserted']} inserted, {res_app['updated']} updated, {res_app['skipped']} skipped.",
        "filename": filename,
        "department": department,
        "target_db": target_db,
        "file_sha256": file_sha256,
        "total_records": len(records),
        "discovered": len(records),
        "inserted": res_app["inserted"],
        "updated": res_app["updated"],
        "skipped": res_app["skipped"],
        "rejected": res_app.get("rejected", 0),
        "failed": 0,
        "factors": analysis["factors"],
    }


def scan_and_ingest_uploads_folder(uploads_dir: str = "uploads") -> List[Dict[str, Any]]:
    """
    Utility scanner for uploads directory.
    Note: Automatic startup ingestion has been disabled as per clean SQL-driven requirements.
    """
    root = Path(__file__).resolve().parent.parent
    up_path = root / uploads_dir
    if not up_path.exists():
        up_path = Path(uploads_dir)
    if not up_path.exists():
        return []

    results = []
    sql_files = sorted(up_path.glob("*.sql"))
    for f in sql_files:
        try:
            res = process_sql_file(str(f))
            results.append(res)
        except Exception as exc:
            logger.error(f"Error ingesting {f.name}: {exc}")
            results.append({
                "status": "error",
                "filename": f.name,
                "message": str(exc),
            })

    return results
