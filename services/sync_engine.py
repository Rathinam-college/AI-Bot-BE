"""
Deterministic Differential Upsert Engine & Data Ingestion Synchronization.
Implements:
- Record-level SHA-256 deduplication and change detection
- Department isolation via (source_app, record_id) composite primary key
- Atomic transactions with rollback safety
- Zero duplicate records
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import httpx

from database.db import get_connection, init_db, resolve_db_path
from services.auth_manager import auth_manager

logger = logging.getLogger("sync_engine")


def normalize_record_payload(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Deterministically normalize record payload and compute SHA-256 content hash.
    - Strips extraneous whitespace from string values
    - Sorts nested keys
    - Normalizes null / empty representations
    """
    clean: Dict[str, Any] = {}
    for k, v in sorted(record.items(), key=lambda x: str(x[0])):
        if isinstance(v, str):
            clean[k] = v.strip()
        elif isinstance(v, dict):
            clean[k] = {str(dk): (dv.strip() if isinstance(dv, str) else dv) for dk, dv in sorted(v.items())}
        elif isinstance(v, list):
            clean[k] = [(it.strip() if isinstance(it, str) else it) for it in v]
        else:
            clean[k] = v

    canonical_json = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return clean, content_hash


class SyncEngine:
    """
    Authoritative Differential Upsert Engine.
    Executes Rule A (Insert), Rule B (Skip unchanged), and Rule C (Update changed).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        init_db(self.db_path)

    @staticmethod
    def compute_content_hash(record: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 hash of record JSON content with sorted keys."""
        _, h = normalize_record_payload(record)
        return h

    def upsert_records(
        self,
        source_app: str,
        records: List[Dict[str, Any]],
        db_path: Optional[str] = None,
        sync_timestamp: Optional[str] = None,
        log_to_history: bool = True,
    ) -> Dict[str, int]:
        """
        Execute Differential Upsert for a list of records belonging to source_app:
        - Rule A (INSERT): If (source_app, record_id) not in DB -> INSERT row, count as inserted.
        - Rule B (SKIP): If (source_app, record_id) in DB AND stored_hash == new_hash -> DO NOTHING, count as skipped.
        - Rule C (UPDATE): If (source_app, record_id) in DB AND stored_hash != new_hash -> UPDATE payload, hash, updated_at, count as updated.
        - Rule D (Isolation): Records with identical IDs in different departments are kept strictly separate.
        """
        target_db = db_path or self.db_path
        init_db(target_db)
        now_ts = sync_timestamp or datetime.now(timezone.utc).isoformat()
        conn = get_connection(target_db)

        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        rejected_count = 0
        fetched_count = len(records)

        try:
            with conn:
                cur = conn.cursor()
                for item in records:
                    if not isinstance(item, dict):
                        rejected_count += 1
                        continue

                    # Resolve stable business primary key
                    raw_id = (
                        item.get("id")
                        if item.get("id") is not None
                        else item.get("record_id", item.get("code", item.get("_id", item.get("key"))))
                    )
                    entity_type = str(item.get("entity_type") or "Record").strip()
                    table_name = str(item.get("table") or item.get("table_name") or "").strip() or None

                    if raw_id is not None and str(raw_id).strip():
                        record_id = str(raw_id).strip()
                    else:
                        # Composite deterministic key based on content fingerprint
                        clean_item, item_hash = normalize_record_payload(item)
                        prefix = "".join(w[0] for w in entity_type.split() if w)[:4].upper() or "REC"
                        record_id = f"{prefix}-{item_hash[:12].upper()}"

                    # Normalize payload and compute SHA-256 hash
                    clean_payload, new_hash = normalize_record_payload(item)
                    canonical_json = json.dumps(clean_payload, sort_keys=True, ensure_ascii=False)

                    cur.execute(
                        "SELECT content_hash FROM software_records WHERE source_app = ? AND record_id = ?;",
                        (source_app, record_id),
                    )
                    row = cur.fetchone()

                    if row is None:
                        # Rule A: INSERT new record
                        cur.execute(
                            """
                            INSERT INTO software_records (
                                source_app, record_id, entity_type, table_name,
                                payload_json, content_hash, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            (source_app, record_id, entity_type, table_name, canonical_json, new_hash, now_ts, now_ts),
                        )
                        inserted_count += 1
                    else:
                        stored_hash = row["content_hash"]
                        if stored_hash == new_hash:
                            # Rule B: SKIP exact duplicate
                            skipped_count += 1
                        else:
                            # Rule C: UPDATE existing record with changed content
                            cur.execute(
                                """
                                UPDATE software_records
                                SET payload_json = ?, content_hash = ?, updated_at = ?,
                                    entity_type = ?, table_name = ?
                                WHERE source_app = ? AND record_id = ?;
                                """,
                                (canonical_json, new_hash, now_ts, entity_type, table_name, source_app, record_id),
                            )
                            updated_count += 1

                if log_to_history:
                    cur.execute(
                        """
                        INSERT INTO sync_history (
                            synced_at, software_name, fetched_count, inserted_count, updated_count, skipped_count
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        (now_ts, source_app, fetched_count, inserted_count, updated_count, skipped_count),
                    )
        finally:
            conn.close()

        return {
            "fetched": fetched_count,
            "inserted": inserted_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "rejected": rejected_count,
        }

    async def fetch_app_data(
        self,
        app_key: str,
        client: httpx.AsyncClient,
        data_path: str = "/api/data",
    ) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        """Fetch records from one software using auth manager retry logic."""
        cfg = auth_manager.get_app_config(app_key)
        app_name = cfg["name"]
        url_path = f"/{app_key}{data_path}"

        try:
            resp = await auth_manager.fetch_with_auth_retry(app_key, url_path, client)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("records", data.get("items", data.get("data", [])))
            else:
                records = []

            return (app_name, records, None)

        except Exception as exc:
            logger.error(f"Failed to fetch data from {app_name} ({app_key}): {exc}")
            return (app_name, [], str(exc))

    async def sync_all_targets(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """Concurrently fetch and upsert records from all configured target apps."""
        sync_timestamp = datetime.now(timezone.utc).isoformat()
        target_apps = ["app1", "app2", "app3"]

        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = [self.fetch_app_data(app_key, client) for app_key in target_apps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        totals = {"fetched": 0, "inserted": 0, "updated": 0, "skipped": 0}
        per_software: Dict[str, Dict[str, int]] = {}
        failed_apps: List[str] = []

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Unexpected exception in sync task: {res}")
                continue

            app_name, records, err_msg = res
            if err_msg:
                failed_apps.append(f"{app_name}: {err_msg}")
                per_software[app_name] = {"fetched": 0, "inserted": 0, "updated": 0, "skipped": 0, "error": err_msg}
                continue

            upsert_res = self.upsert_records(
                source_app=app_name,
                records=records,
                db_path=db_path,
                sync_timestamp=sync_timestamp,
            )

            per_software[app_name] = upsert_res
            totals["fetched"] += upsert_res["fetched"]
            totals["inserted"] += upsert_res["inserted"]
            totals["updated"] += upsert_res["updated"]
            totals["skipped"] += upsert_res["skipped"]

        status = "partial" if failed_apps and len(failed_apps) < len(target_apps) else ("failed" if len(failed_apps) == len(target_apps) else "success")

        return {
            "status": status,
            "synced_at": sync_timestamp,
            "totals": totals,
            "per_software": per_software,
            "failed_apps": failed_apps,
        }


# Authoritative singleton instances
sync_engine = SyncEngine()
