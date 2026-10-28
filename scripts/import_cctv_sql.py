import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_sync_metrics_summary, init_db
from services.cctv_sql_parser import parse_cctv_backup_sql
from services.sync_engine import sync_engine


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    if len(sys.argv) > 1:
        sql_file = sys.argv[1]
    else:
        candidates = [
            "uploads/cctv_backup_2026-10-22.sql",
            "../uploads/cctv_backup_2026-10-22.sql",
            "cctv_backup_2026-10-22.sql",
            "uploads/cctv_backup_2026-10-21.sql",
            "../uploads/cctv_backup_2026-10-21.sql",
            "cctv_backup_2026-10-21.sql",
            "uploads/cctv_backup_2026-10-17.sql",
            "../uploads/cctv_backup_2026-10-17.sql",
            "cctv_backup_2026-10-17.sql",
        ]
        sql_file = next((c for c in candidates if os.path.exists(c)), None)
    if not sql_file or not os.path.exists(sql_file):
        print(f"Error: SQL dump file '{sql_file}' not found.")
        sys.exit(1)

    print(f"Parsing CCTV PostgreSQL dump: {sql_file}...")
    parsed = parse_cctv_backup_sql(sql_file)

    app1_name = os.getenv("APP1_NAME", "Software 1 (CCTV Surveillance)")
    app2_name = os.getenv("APP2_NAME", "Software 2 (Network & Racks)")
    app3_name = os.getenv("APP3_NAME", "Software 3 (Maintenance & Tickets)")

    print(f"\n--- Ingesting Records into SQLite (database/app_data.db) ---")
    s1 = sync_engine.upsert_records(app1_name, parsed["app1"])
    print(f"✓ {app1_name}: {s1['fetched']} fetched | {s1['inserted']} inserted | {s1['updated']} updated | {s1['skipped']} skipped")

    s2 = sync_engine.upsert_records(app2_name, parsed["app2"])
    print(f"✓ {app2_name}: {s2['fetched']} fetched | {s2['inserted']} inserted | {s2['updated']} updated | {s2['skipped']} skipped")

    s3 = sync_engine.upsert_records(app3_name, parsed["app3"])
    print(f"✓ {app3_name}: {s3['fetched']} fetched | {s3['inserted']} inserted | {s3['updated']} updated | {s3['skipped']} skipped")

    tot_ins = s1["inserted"] + s2["inserted"] + s3["inserted"]
    tot_upd = s1["updated"] + s2["updated"] + s3["updated"]
    tot_skp = s1["skipped"] + s2["skipped"] + s3["skipped"]

    metrics = get_sync_metrics_summary()
    print(f"\nIngestion Complete!")
    print(f"Total Processed: {s1['fetched'] + s2['fetched'] + s3['fetched']} records")
    print(f"Inserted: {tot_ins} | Updated: {tot_upd} | Skipped: {tot_skp}")
    print(f"Total Rows in SQLite: {metrics['total_records']}")
    for app, cnt in metrics["app_breakdown"].items():
        print(f"  - {app}: {cnt} rows")


if __name__ == "__main__":
    main()
