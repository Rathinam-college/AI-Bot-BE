import logging
import os
from typing import Any, Dict, List, Optional
from services.cctv_sql_parser import parse_cctv_backup_sql

logger = logging.getLogger("sql_dump_importer")


def parse_copy_block_from_sql(sql_path: str, target_tables: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fast streaming parser for PostgreSQL dumps using COPY <table> (...) FROM stdin;
    """
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL file not found at {sql_path}")

    results: Dict[str, List[Dict[str, Any]]] = {t: [] for t in target_tables}
    current_table: Optional[str] = None
    current_cols: List[str] = []

    with open(sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("COPY "):
                idx1 = line_str.find("(")
                idx2 = line_str.find(")")
                if idx1 != -1 and idx2 != -1:
                    tname = line_str[5:idx1].strip().replace("public.", "").replace('"', "")
                    if tname in results:
                        current_table = tname
                        current_cols = [c.strip().replace('"', "") for c in line_str[idx1+1:idx2].split(",")]
                    else:
                        current_table = None
            elif current_table and line_str == "\\.":
                current_table = None
            elif current_table:
                # Tab-delimited values
                parts = line.rstrip("\r\n").split("\t")
                row_dict = {}
                for col, val in zip(current_cols, parts):
                    if val == "\\N" or val == "":
                        row_dict[col] = None
                    elif val == "t":
                        row_dict[col] = True
                    elif val == "f":
                        row_dict[col] = False
                    else:
                        row_dict[col] = val
                results[current_table].append(row_dict)

    return results


def parse_food_dump(sql_path: str) -> List[Dict[str, Any]]:
    """
    Extract canonical records from food.sql for Food Department:
    - Items (Commodities / Groceries)
    - Inventory Stock Levels
    - Material Requests
    - Purchase Requests
    - Food Vendors / Suppliers
    """
    target_tables = [
        "core_item",
        "core_inventory",
        "core_materialrequest",
        "core_purchaserequest",
        "core_vendor",
        "core_organizationunit"
    ]
    raw = parse_copy_block_from_sql(sql_path, target_tables)
    records = []

    # 1. Food Items
    for item in raw.get("core_item", []):
        rec_id = f"FOOD-ITEM-{item['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Food Item",
            "name": item.get("name", rec_id),
            "unit": item.get("unit"),
            "category_id": item.get("category_id"),
            "min_stock_level": item.get("min_stock_level"),
            "status": "Active"
        })

    # 2. Inventory Levels
    for inv in raw.get("core_inventory", []):
        rec_id = f"FOOD-INV-{inv['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Food Inventory",
            "item_id": inv.get("item_id"),
            "current_quantity": inv.get("current_quantity"),
            "min_stock_level": inv.get("min_stock_level"),
            "last_updated": inv.get("last_updated"),
            "department_id": inv.get("department_id"),
            "status": "In Stock" if float(inv.get("current_quantity") or 0) > 0 else "Low Stock"
        })

    # 3. Material Requests
    for mr in raw.get("core_materialrequest", []):
        rec_id = f"MR-{mr.get('mr_number') or mr['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Food Material Request",
            "mr_number": mr.get("mr_number"),
            "order_date": mr.get("order_date"),
            "target_date": mr.get("target_date"),
            "meal_type": mr.get("meal_type"),
            "status": mr.get("status") or "PENDING"
        })

    # 4. Purchase Requests
    for pr in raw.get("core_purchaserequest", []):
        rec_id = f"PR-{pr.get('pr_reference') or pr['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Food Purchase Request",
            "pr_reference": pr.get("pr_reference"),
            "order_date": pr.get("order_date"),
            "delivery_date": pr.get("delivery_date"),
            "status": pr.get("status") or "SUBMITTED",
            "total_amount": pr.get("total_amount"),
            "vendor_id": pr.get("vendor_id")
        })

    # 5. Food Vendors
    for v in raw.get("core_vendor", []):
        if not v.get("name") or v.get("name") == "Opening Stock":
            continue
        rec_id = f"FOOD-VND-{v['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Food Vendor",
            "name": v.get("name"),
            "contact_person": v.get("contact_person"),
            "phone": v.get("phone"),
            "email": v.get("email"),
            "address": v.get("address"),
            "status": "Active" if v.get("is_active") else "Inactive"
        })

    return records


def parse_transport_dump(sql_path: str) -> List[Dict[str, Any]]:
    """
    Extract canonical records from Transport.sql for Transport Department:
    - Vehicles / Campus Buses (with driver, seating, registration, make, model)
    - Odometer Logs
    - Document Renewals (FC, Insurance, Permit)
    - Service Requests (Maintenance, Repair, Testing)
    - Fuel Data Entries
    - Transport Vendors
    """
    target_tables = [
        "transport_department_vehicle",
        "transport_department_vehicleodometerlog",
        "transport_department_vehiclerenewal",
        "transport_department_servicerequest",
        "transport_department_fueldata",
        "transport_department_vendor",
        "transport_department_institution"
    ]
    raw = parse_copy_block_from_sql(sql_path, target_tables)
    records = []

    # Map vehicle ID to Vehicle Number
    veh_map = {}
    for v in raw.get("transport_department_vehicle", []):
        veh_map[str(v["id"])] = v.get("vehicle_number", f"Vehicle-{v['id']}")

    # 1. Vehicles / Buses
    for v in raw.get("transport_department_vehicle", []):
        vnum = v.get("vehicle_number", f"Vehicle-{v['id']}")
        rec_id = f"BUS-{vnum.replace(' ', '-')}"
        records.append({
            "id": rec_id,
            "entity_type": "Campus Bus",
            "vehicle_number": vnum,
            "make": v.get("make"),
            "model": v.get("model"),
            "seating_capacity": v.get("seating_capacity"),
            "driver_name": v.get("driver_name"),
            "driver_contact_no": v.get("driver_contact_no"),
            "chassis_number": v.get("chassis_number"),
            "fuel_type": v.get("fuel_type") or "Diesel",
            "status": "Active" if v.get("is_active") else "Inactive",
            "is_tracking": v.get("is_tracking")
        })

    # 2. Document Renewals (FC, Insurance, Tax)
    for r in raw.get("transport_department_vehiclerenewal", []):
        vnum = veh_map.get(str(r.get("vehicle_id")), f"Vehicle-{r.get('vehicle_id')}")
        rec_id = f"TR-RNW-{r['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Transport Renewal",
            "vehicle_number": vnum,
            "document_type": r.get("document_type"),
            "new_expiry_date": r.get("new_expiry_date"),
            "amount_paid": r.get("amount_paid"),
            "status": "Valid" if r.get("new_expiry_date") and r.get("new_expiry_date") > "2026-10" else "Pending Renewal"
        })

    # 3. Service Requests
    for s in raw.get("transport_department_servicerequest", []):
        rec_id = s.get("request_id") or f"TR-SR-{s['id']}"
        vnum = veh_map.get(str(s.get("vehicle_id")), s.get("cab_no"))
        records.append({
            "id": rec_id,
            "entity_type": "Transport Service Request",
            "vehicle_number": vnum,
            "date": s.get("date"),
            "nature_of_problem": s.get("nature_of_problem"),
            "estimated_repair_cost": s.get("estimated_repair_cost"),
            "kms_reading": s.get("kms_reading"),
            "status": "Completed" if s.get("last_service_date") else "In Service"
        })

    # 4. Fuel Data
    for f in raw.get("transport_department_fueldata", []):
        vnum = veh_map.get(str(f.get("vehicle_id")), f"Vehicle-{f.get('vehicle_id')}")
        rec_id = f"TR-FUEL-{f['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Transport Fuel Entry",
            "vehicle_number": vnum,
            "date": f.get("date"),
            "liters": f.get("liters"),
            "amount": f.get("amount"),
            "bill_no": f.get("bill_no"),
            "current_km": f.get("current_km"),
            "total_km": f.get("total_km")
        })

    # 5. Odometer Logs (Recent snapshot of 50 logs)
    for o in raw.get("transport_department_vehicleodometerlog", [])[-50:]:
        vnum = veh_map.get(str(o.get("vehicle_id")), f"Vehicle-{o.get('vehicle_id')}")
        rec_id = f"TR-ODO-{o['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Transport Odometer Log",
            "vehicle_number": vnum,
            "log_date": o.get("log_date"),
            "morning_930_km": o.get("morning_930"),
            "evening_330_km": o.get("evening_330"),
            "remarks": o.get("remarks")
        })

    # 6. Transport Vendors
    for tv in raw.get("transport_department_vendor", []):
        rec_id = f"TR-VND-{tv['id']}"
        records.append({
            "id": rec_id,
            "entity_type": "Transport Vendor",
            "name": tv.get("name"),
            "contact_number": tv.get("contact_number"),
            "email": tv.get("email"),
            "address": tv.get("address"),
            "gst_number": tv.get("gst_number")
        })

    return records


def ingest_all_uploaded_dumps(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest all uploaded SQL dump files found in uploads/ directory into SQLite
    using the dynamic Factor Analysis and Differential Upsert Engine.
    Zero hardcoded file names.
    """
    from services.sql_analyzer import process_sql_file
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    up_path = root / "uploads"
    if not up_path.exists():
        up_path = Path("uploads")

    summary = {
        "status": "success",
        "files_processed": 0,
        "details": [],
        "overall_total_discovered": 0,
        "overall_total_inserted": 0,
        "overall_total_updated": 0,
        "overall_total_skipped": 0,
    }

    if not up_path.exists():
        summary["message"] = "Uploads directory does not exist. 0 files processed."
        return summary

    sql_files = sorted(up_path.glob("*.sql"))
    if not sql_files:
        summary["message"] = "No .sql files found in uploads directory."
        return summary

    for f in sql_files:
        try:
            res = process_sql_file(str(f), db_path=db_path)
            summary["files_processed"] += 1
            summary["details"].append(res)
            summary["overall_total_discovered"] += res.get("discovered", 0)
            summary["overall_total_inserted"] += res.get("inserted", 0)
            summary["overall_total_updated"] += res.get("updated", 0)
            summary["overall_total_skipped"] += res.get("skipped", 0)
        except Exception as exc:
            logger.error(f"Error processing {f.name}: {exc}")
            summary["details"].append({"filename": f.name, "status": "error", "error": str(exc)})

    summary["message"] = f"Processed {summary['files_processed']} SQL dump files successfully."
    return summary
