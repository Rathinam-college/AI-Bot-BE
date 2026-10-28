import os
import re
from typing import Any, Dict, List, Tuple


def get_table_columns(sql_text: str, table_name: str) -> List[str]:
    pattern = rf'CREATE TABLE public\.{table_name}\s*\((.*?)\n\);'
    match = re.search(pattern, sql_text, re.DOTALL | re.IGNORECASE)
    if not match:
        pattern = rf'CREATE TABLE public\."{table_name}"\s*\((.*?)\n\);'
        match = re.search(pattern, sql_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    body = match.group(1)
    cols = []
    for line in body.split("\n"):
        line = line.strip().rstrip(",")
        if not line or line.upper().startswith("CONSTRAINT"):
            continue
        col_match = re.match(r'^("?[a-zA-Z0-9_]+"?)\s+', line)
        if col_match:
            cols.append(col_match.group(1).replace('"', ""))
    return cols


def parse_sql_values(val_str: str) -> List[Any]:
    val_str = val_str.strip()
    tokens = []
    current = []
    in_quotes = False
    i = 0
    n = len(val_str)

    while i < n:
        c = val_str[i]
        if in_quotes:
            if c == "'" and i + 1 < n and val_str[i + 1] == "'":
                current.append("'")
                i += 1
            elif c == "\\" and i + 1 < n and val_str[i + 1] == "'":
                current.append("'")
                i += 1
            elif c == "'":
                in_quotes = False
            else:
                current.append(c)
        else:
            if c == "'":
                in_quotes = True
            elif c == ",":
                raw_token = "".join(current).strip()
                tokens.append(convert_token(raw_token))
                current = []
            else:
                current.append(c)
        i += 1

    if current or tokens:
        raw_token = "".join(current).strip()
        tokens.append(convert_token(raw_token))

    return tokens


def convert_token(tok: str) -> Any:
    if tok == "NULL" or tok == "":
        return None
    elif tok == "true":
        return True
    elif tok == "false":
        return False
    elif re.match(r"^-?\d+$", tok):
        return int(tok)
    else:
        try:
            return float(tok)
        except ValueError:
            return tok


def parse_cctv_backup_sql(sql_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"SQL file not found at {sql_file_path}")

    with open(sql_file_path, "r", encoding="utf-8", errors="ignore") as f:
        sql_text = f.read()

    target_tables = [
        "cctv_camera",
        "cctv_nvr",
        "cctv_biometric",
        "cctv_networkswitch",
        "cctv_rack",
        "cctv_camerarelocation",
        "maintenance_ticket",
        "maintenance_project",
    ]

    table_cols = {t: get_table_columns(sql_text, t) for t in target_tables}
    table_records = {t: [] for t in target_tables}

    lines = sql_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        for t in target_tables:
            prefix = f"INSERT INTO public.{t} VALUES"
            prefix_q = f'INSERT INTO public."{t}" VALUES'
            if line.startswith(prefix) or line.startswith(prefix_q):
                full_stmt = line
                while not full_stmt.rstrip().endswith(";") and i + 1 < len(lines):
                    i += 1
                    full_stmt += " " + lines[i]

                idx = full_stmt.find("VALUES")
                paren_start = full_stmt.find("(", idx)
                paren_end = full_stmt.rfind(")")
                if paren_start != -1 and paren_end != -1:
                    tuple_part = full_stmt[paren_start + 1 : paren_end]
                    vals = parse_sql_values(tuple_part)
                    cols = table_cols[t]
                    if len(vals) == len(cols):
                        obj = dict(zip(cols, vals))
                        table_records[t].append(obj)
        i += 1

    # Map into Software 1, 2, 3
    # Software 1: CCTV Surveillance (Cameras, NVRs, Biometrics)
    app1_records = []
    for cam in table_records["cctv_camera"]:
        cid = cam.get("cameraId") or f"CAM-{cam['id']:03d}"
        app1_records.append({
            "id": f"CAM-{cam['id']:03d}",
            "entity_type": "CCTV Camera",
            "camera_code": cid,
            "name": cam.get("name"),
            "site_name": cam.get("siteName"),
            "ip_address": cam.get("ipAddress"),
            "dvr_nvr_details": cam.get("dvrNvrDetails"),
            "block": cam.get("block"),
            "floor": str(cam.get("floor")),
            "room": cam.get("room"),
            "brand": cam.get("brand"),
            "model": cam.get("model"),
            "device_type": cam.get("deviceType"),
            "status": cam.get("status"),
            "mac_address": cam.get("macAddress"),
            "gateway": cam.get("gateway"),
            "serial_number": cam.get("serialNumber"),
            "remarks": cam.get("remarks"),
        })

    for nvr in table_records["cctv_nvr"]:
        app1_records.append({
            "id": f"NVR-{nvr['id']:03d}",
            "entity_type": "NVR / Storage",
            "nvr_name": nvr.get("nvrName"),
            "ip_address": nvr.get("ipAddress"),
            "location": nvr.get("location"),
            "brand": nvr.get("brand"),
            "model": nvr.get("model"),
            "channel_count": nvr.get("channel"),
            "hard_disk_capacity": nvr.get("hardDisk"),
            "serial_number": nvr.get("serialNumber"),
            "status": nvr.get("status"),
            "block": nvr.get("block"),
            "division": nvr.get("divisionName"),
            "remarks": nvr.get("remarks"),
        })

    for bio in table_records["cctv_biometric"]:
        app1_records.append({
            "id": f"BIO-{bio['id']:03d}",
            "entity_type": "Biometric Device",
            "device_name": bio.get("name"),
            "location": bio.get("location"),
            "auth_type": bio.get("type"),
            "brand": bio.get("brand"),
            "model": bio.get("model"),
            "ip_address": bio.get("ipAddress"),
            "status": bio.get("status"),
            "sync_status": bio.get("syncStatus"),
            "block": bio.get("block"),
            "division": bio.get("divisionName"),
            "serial_number": bio.get("serialNumber"),
            "remarks": bio.get("remarks"),
        })

    # Software 2: Network Infrastructure & Racks
    app2_records = []
    for sw in table_records["cctv_networkswitch"]:
        app2_records.append({
            "id": f"SW-{sw['id']:03d}",
            "entity_type": "Network Switch",
            "switch_name": sw.get("name"),
            "location": sw.get("location"),
            "brand": sw.get("brand"),
            "model": sw.get("model"),
            "port_count": sw.get("portCount"),
            "serial_number": sw.get("serialNumber"),
            "status": sw.get("status"),
            "block": sw.get("block"),
            "floor": str(sw.get("floor")),
            "room": sw.get("room"),
            "division": sw.get("divisionName"),
            "mac_address": sw.get("macAddress"),
            "remarks": sw.get("remarks"),
        })

    for rk in table_records["cctv_rack"]:
        app2_records.append({
            "id": f"RACK-{rk['id']:03d}",
            "entity_type": "Server Rack",
            "rack_name": rk.get("name"),
            "location": rk.get("location"),
            "brand": rk.get("brand"),
            "model": rk.get("model"),
            "u_space": rk.get("uSpace"),
            "serial_number": rk.get("serialNumber"),
            "status": rk.get("status"),
            "block": rk.get("block"),
            "floor": str(rk.get("floor")),
            "remarks": rk.get("remarks"),
        })

    for rel in table_records["cctv_camerarelocation"]:
        app2_records.append({
            "id": f"RELOC-{rel['id']:03d}",
            "entity_type": "Camera Relocation Event",
            "old_location": rel.get("oldLocation"),
            "new_location": rel.get("newLocation"),
            "ip_address": rel.get("ipAddress"),
            "date": rel.get("date"),
            "reason": rel.get("reason"),
            "remarks": rel.get("remarks"),
        })

    # Software 3: Maintenance Tickets & Projects
    app3_records = []
    for tkt in table_records["maintenance_ticket"]:
        app3_records.append({
            "id": f"TKT-{tkt['id']:04d}",
            "entity_type": "Maintenance Ticket",
            "ticket_number": f"TKT-{tkt['id']:04d}",
            "issue_description": tkt.get("issueDescription"),
            "status": tkt.get("status"),
            "category": tkt.get("category"),
            "action_taken": tkt.get("actionTaken"),
            "ticket_device": tkt.get("ticketDevice"),
            "location": tkt.get("location"),
            "block": tkt.get("block"),
            "floor": str(tkt.get("floor")),
            "raised_by_name": tkt.get("raisedByName"),
            "instruction_by": tkt.get("instructionBy"),
            "created_date": str(tkt.get("createdDate")),
            "completed_date": str(tkt.get("completedDate")),
            "total_time": tkt.get("totalTime"),
            "remarks": tkt.get("remarks"),
        })

    for prj in table_records["maintenance_project"]:
        app3_records.append({
            "id": f"PROJ-{prj['id']:03d}",
            "entity_type": "Maintenance Project",
            "project_name": prj.get("name"),
            "client_name": prj.get("client_name"),
            "description": prj.get("description"),
            "status": prj.get("status"),
            "start_date": str(prj.get("start_date")),
            "end_date": str(prj.get("end_date")),
            "bill_number": prj.get("bill_number"),
            "instruction_by": prj.get("instructionBy"),
        })

    cctv_all = app1_records + app2_records + app3_records

    return {
        "cctv": cctv_all,
        "app1": cctv_all,
        "app2": [],
        "app3": [],
        "cctv_devices": app1_records,
        "cctv_network": app2_records,
        "cctv_tickets": app3_records,
        "raw_counts": {t: len(recs) for t, recs in table_records.items()},
    }


if __name__ == "__main__":
    result = parse_cctv_backup_sql("cctv_backup_2026-10-21.sql")
    print("Raw Counts:")
    for t, cnt in result["raw_counts"].items():
        print(f"  {t}: {cnt}")
    print(f"\nTotal CCTV Records: {len(result['cctv'])}")

