import copy
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from services.cctv_sql_parser import parse_cctv_backup_sql

mock_router = APIRouter()

# --- Mock State Storage ---
mock_state = {
    "current_dataset_state": 0,  # 0 = clean/empty or custom SQL dump; 1-4 = automated test states
    "custom_records": {
        "app1": [],
        "app2": [],
        "app3": [],
    },
    "invalidated_tokens": set(),
    "simulated_failures": set(),
    "valid_tokens": {
        "app1": "mock_token_app1_secure_session",
        "app2": "mock_token_app2_secure_session",
        "app3": "mock_token_app3_secure_session",
    },
    "credentials": {
        "app1": {"username": "admin_app1", "password": "secret_app1_pass"},
        "app2": {"username": "admin_app2", "password": "secret_app2_pass"},
        "app3": {"username": "admin_app3", "password": "secret_app3_pass"},
    },
}


def generate_food_dataset() -> List[Dict[str, Any]]:
    """Generate realistic campus Food & Cafeteria records."""
    counters = [
        {"id": "FOOD-CNT-001", "entity_type": "Food Counter", "name": "Main Food Court", "location": "Block A - Ground Floor", "type": "Multi-Cuisine", "status": "Open", "operating_hours": "07:30 - 21:30"},
        {"id": "FOOD-CNT-002", "entity_type": "Food Counter", "name": "South Dining Hall", "location": "Hostel Block 1", "type": "Student Mess", "status": "Open", "operating_hours": "07:00 - 22:00"},
        {"id": "FOOD-CNT-003", "entity_type": "Food Counter", "name": "North Dining Hall", "location": "Hostel Block 2", "type": "Student Mess", "status": "Open", "operating_hours": "07:00 - 22:00"},
        {"id": "FOOD-CNT-004", "entity_type": "Food Counter", "name": "Campus Cafe & Bakery", "location": "Zia Campus Central", "type": "Bakery & Cafe", "status": "Open", "operating_hours": "08:30 - 20:00"},
        {"id": "FOOD-CNT-005", "entity_type": "Food Counter", "name": "Fresh Juice & Snacks Hub", "location": "Grand Hall Quadrangle", "type": "Juice & Snacks", "status": "Open", "operating_hours": "09:00 - 19:30"},
    ]
    menu_items = [
        {"id": "FOOD-MENU-001", "entity_type": "Menu Item", "item_name": "South Indian Breakfast Combo", "category": "Breakfast", "price": 40.0, "dietary": "Vegetarian", "counter": "Main Food Court", "availability": "In Stock", "calories": 380},
        {"id": "FOOD-MENU-002", "entity_type": "Menu Item", "item_name": "Executive Veg Lunch Thali", "category": "Lunch", "price": 80.0, "dietary": "Vegetarian", "counter": "South Dining Hall", "availability": "In Stock", "calories": 650},
        {"id": "FOOD-MENU-003", "entity_type": "Menu Item", "item_name": "Special Chicken Biryani Combo", "category": "Lunch", "price": 140.0, "dietary": "Non-Vegetarian", "counter": "Main Food Court", "availability": "In Stock", "calories": 780},
        {"id": "FOOD-MENU-004", "entity_type": "Menu Item", "item_name": "Paneer Butter Masala with 3 Rotis", "category": "Dinner", "price": 95.0, "dietary": "Vegetarian", "counter": "North Dining Hall", "availability": "In Stock", "calories": 520},
        {"id": "FOOD-MENU-005", "entity_type": "Menu Item", "item_name": "Chapati & Mixed Veg Kurma", "category": "Dinner", "price": 50.0, "dietary": "Vegetarian", "counter": "South Dining Hall", "availability": "In Stock", "calories": 410},
        {"id": "FOOD-MENU-006", "entity_type": "Menu Item", "item_name": "Filter Coffee / Masala Tea", "category": "Beverage", "price": 15.0, "dietary": "Vegetarian", "counter": "Campus Cafe & Bakery", "availability": "In Stock", "calories": 110},
        {"id": "FOOD-MENU-007", "entity_type": "Menu Item", "item_name": "Fresh Seasonal Fruit Juice", "category": "Beverage", "price": 35.0, "dietary": "Vegetarian", "counter": "Fresh Juice & Snacks Hub", "availability": "In Stock", "calories": 160},
        {"id": "FOOD-MENU-008", "entity_type": "Menu Item", "item_name": "Crispy Veg Samosa & Chutney", "category": "Snacks", "price": 20.0, "dietary": "Vegetarian", "counter": "Fresh Juice & Snacks Hub", "availability": "In Stock", "calories": 240},
    ]
    tokens = [
        {"id": f"FOOD-TKN-{i:03d}", "entity_type": "Food Token", "token_number": 1000 + i, "counter": "Main Food Court" if i % 2 == 0 else "South Dining Hall", "meal_type": "Breakfast" if i % 3 == 0 else ("Lunch" if i % 2 == 0 else "Dinner"), "amount": 40.0 + (i * 5), "payment_status": "Paid", "token_status": "Redeemed" if i < 15 else "Active", "created_date": "2026-10-21"}
        for i in range(1, 21)
    ]
    inventory = [
        {"id": "FOOD-INV-001", "entity_type": "Food Inventory", "item": "Sona Masoori Rice", "quantity": 450, "unit": "kg", "reorder_level": 100, "status": "In Stock", "supplier": "Premier Agro Stores"},
        {"id": "FOOD-INV-002", "entity_type": "Food Inventory", "item": "Refined Sunflower Oil", "quantity": 120, "unit": "L", "reorder_level": 30, "status": "In Stock", "supplier": "Kovai Refineries"},
        {"id": "FOOD-INV-003", "entity_type": "Food Inventory", "item": "Toor Dal Premium", "quantity": 85, "unit": "kg", "reorder_level": 25, "status": "In Stock", "supplier": "Sri Lakshmi Traders"},
        {"id": "FOOD-INV-004", "entity_type": "Food Inventory", "item": "Pasteurized Milk", "quantity": 140, "unit": "L", "reorder_level": 40, "status": "Fresh Stock", "supplier": "Aavin Dairy"},
        {"id": "FOOD-INV-005", "entity_type": "Food Inventory", "item": "Fresh Farm Vegetables", "quantity": 90, "unit": "kg", "reorder_level": 20, "status": "In Stock", "supplier": "Udumalai Farms"},
    ]
    return counters + menu_items + tokens + inventory


def generate_transport_dataset() -> List[Dict[str, Any]]:
    """Generate realistic campus Transport & Bus fleet records."""
    buses = [
        {"id": "TRN-BUS-001", "entity_type": "Campus Bus", "bus_number": "Bus 01", "registration_no": "TN-38-BZ-4501", "capacity": 54, "make_model": "Ashok Leyland Viking", "route_name": "Route 01: Gandhipuram", "assigned_driver": "Murugan K", "gps_status": "Active", "status": "On Route", "current_location": "Lakshmi Mills Junction"},
        {"id": "TRN-BUS-002", "entity_type": "Campus Bus", "bus_number": "Bus 02", "registration_no": "TN-38-BZ-4502", "capacity": 54, "make_model": "Tata Starbus Ultra", "route_name": "Route 02: Hope College", "assigned_driver": "Senthil Kumar P", "gps_status": "Active", "status": "On Route", "current_location": "Singanallur"},
        {"id": "TRN-BUS-003", "entity_type": "Campus Bus", "bus_number": "Bus 03", "registration_no": "TN-38-BZ-4503", "capacity": 54, "make_model": "Ashok Leyland Viking", "route_name": "Route 03: Ondipudur", "assigned_driver": "Rajendran M", "gps_status": "Active", "status": "On Route", "current_location": "Trichy Road"},
        {"id": "TRN-BUS-004", "entity_type": "Campus Bus", "bus_number": "Bus 04", "registration_no": "TN-38-BZ-4504", "capacity": 54, "make_model": "Tata Starbus Ultra", "route_name": "Route 04: Saravanampatti", "assigned_driver": "Palanisamy V", "gps_status": "Active", "status": "On Route", "current_location": "Ganapathy"},
        {"id": "TRN-BUS-005", "entity_type": "Campus Bus", "bus_number": "Bus 05", "registration_no": "TN-38-BZ-4505", "capacity": 54, "make_model": "Ashok Leyland Viking", "route_name": "Route 05: Pollachi", "assigned_driver": "Velumani R", "gps_status": "Active", "status": "Maintenance", "current_location": "Campus Workshop (Brake Service)"},
        {"id": "TRN-BUS-006", "entity_type": "Campus Bus", "bus_number": "Bus 06", "registration_no": "TN-38-BZ-4506", "capacity": 54, "make_model": "Ashok Leyland Viking", "route_name": "Route 06: Thudiyalur", "assigned_driver": "Karthik N", "gps_status": "Active", "status": "On Route", "current_location": "Town Hall"},
        {"id": "TRN-BUS-007", "entity_type": "Campus Bus", "bus_number": "Bus 07", "registration_no": "TN-38-BZ-4507", "capacity": 54, "make_model": "Tata Starbus Ultra", "route_name": "Route 07: Ukkadam", "assigned_driver": "Muthusamy S", "gps_status": "Active", "status": "On Route", "current_location": "Sundarapuram"},
        {"id": "TRN-BUS-008", "entity_type": "Campus Bus", "bus_number": "Bus 08", "registration_no": "TN-38-BZ-4508", "capacity": 32, "make_model": "Force Traveller", "route_name": "Internal Shuttle", "assigned_driver": "Anand P", "gps_status": "Active", "status": "Active", "current_location": "Campus Ring Road"},
    ]
    routes = [
        {"id": "TRN-RT-001", "entity_type": "Bus Route", "route_code": "R-01", "route_name": "Gandhipuram Express", "start_point": "Gandhipuram Central Stand", "stops": ["Gandhipuram", "Lakshmi Mills", "Ramanathapuram", "Rathinam Campus"], "departure_time": "07:15 AM", "status": "Active", "assigned_bus": "Bus 01"},
        {"id": "TRN-RT-002", "entity_type": "Bus Route", "route_code": "R-02", "route_name": "Hope College / Peelamedu", "start_point": "Hope College", "stops": ["Hope College", "Hopes", "Singanallur", "Eachanari", "Rathinam Campus"], "departure_time": "07:20 AM", "status": "Active", "assigned_bus": "Bus 02"},
        {"id": "TRN-RT-003", "entity_type": "Bus Route", "route_code": "R-03", "route_name": "Ondipudur / Trichy Road", "start_point": "Ondipudur Flyover", "stops": ["Ondipudur", "Singanallur", "Podanur Junction", "Rathinam Campus"], "departure_time": "07:25 AM", "status": "Active", "assigned_bus": "Bus 03"},
        {"id": "TRN-RT-004", "entity_type": "Bus Route", "route_code": "R-04", "route_name": "Saravanampatti / Sathy Rd", "start_point": "Saravanampatti Checkpost", "stops": ["Saravanampatti", "Ganapathy", "Cross Cut Road", "Rathinam Campus"], "departure_time": "07:10 AM", "status": "Active", "assigned_bus": "Bus 04"},
        {"id": "TRN-RT-005", "entity_type": "Bus Route", "route_code": "R-05", "route_name": "Pollachi Bypass", "start_point": "Pollachi New Bus Stand", "stops": ["Pollachi", "Kinathukadavu", "Eachanari", "Rathinam Campus"], "departure_time": "07:00 AM", "status": "Active", "assigned_bus": "Bus 05"},
        {"id": "TRN-RT-006", "entity_type": "Bus Route", "route_code": "R-06", "route_name": "Thudiyalur / Mettupalayam", "start_point": "Thudiyalur Bus Stand", "stops": ["Thudiyalur", "Saibaba Colony", "Flyover", "Rathinam Campus"], "departure_time": "07:15 AM", "status": "Active", "assigned_bus": "Bus 06"},
        {"id": "TRN-RT-007", "entity_type": "Bus Route", "route_code": "R-07", "route_name": "Ukkadam / Kuniyamuthur", "start_point": "Ukkadam Bus Stand", "stops": ["Ukkadam", "Kuniyamuthur", "Sundarapuram", "Rathinam Campus"], "departure_time": "07:35 AM", "status": "Active", "assigned_bus": "Bus 07"},
    ]
    drivers = [
        {"id": "TRN-DRV-001", "entity_type": "Transport Driver", "name": "Murugan K", "license_no": "DL-TN38-20120042", "phone": "9842100011", "assigned_bus": "Bus 01", "experience_years": 12, "status": "On Duty"},
        {"id": "TRN-DRV-002", "entity_type": "Transport Driver", "name": "Senthil Kumar P", "license_no": "DL-TN38-20140089", "phone": "9842100022", "assigned_bus": "Bus 02", "experience_years": 9, "status": "On Duty"},
        {"id": "TRN-DRV-003", "entity_type": "Transport Driver", "name": "Rajendran M", "license_no": "DL-TN38-20100015", "phone": "9842100033", "assigned_bus": "Bus 03", "experience_years": 14, "status": "On Duty"},
        {"id": "TRN-DRV-004", "entity_type": "Transport Driver", "name": "Palanisamy V", "license_no": "DL-TN38-20160074", "phone": "9842100044", "assigned_bus": "Bus 04", "experience_years": 8, "status": "On Duty"},
        {"id": "TRN-DRV-005", "entity_type": "Transport Driver", "name": "Velumani R", "license_no": "DL-TN38-20130091", "phone": "9842100055", "assigned_bus": "Bus 05", "experience_years": 11, "status": "At Workshop"},
    ]
    passes = [
        {"id": f"TRN-PASS-{i:03d}", "entity_type": "Transport Pass", "pass_number": f"PASS-2026-{100+i}", "student_name": f"Student {i}", "department": "Computer Science" if i % 2 == 0 else "Information Technology", "route": f"Route 0{((i-1)%7)+1}", "valid_until": "2027-05-31", "status": "Active"}
        for i in range(1, 11)
    ]
    return buses + routes + drivers + passes


def load_cctv_sql_data(sql_file_path: Optional[str] = None) -> Dict[str, int]:
    """Parse CCTV SQL file if provided and populate mock records for testing."""
    if not sql_file_path or not os.path.exists(sql_file_path):
        return {
            "app1": len(mock_state["custom_records"]["app1"]),
            "app2": len(mock_state["custom_records"]["app2"]),
            "app3": len(mock_state["custom_records"]["app3"]),
            "total": len(mock_state["custom_records"]["app1"]) + len(mock_state["custom_records"]["app2"]) + len(mock_state["custom_records"]["app3"]),
        }

    try:
        parsed = parse_cctv_backup_sql(sql_file_path)
        mock_state["custom_records"]["app1"] = parsed.get("cctv", parsed.get("app1", []))
        mock_state["current_dataset_state"] = 0
        return {
            "app1": len(mock_state["custom_records"]["app1"]),
            "app2": len(mock_state["custom_records"]["app2"]),
            "app3": len(mock_state["custom_records"]["app3"]),
            "total": len(mock_state["custom_records"]["app1"]) + len(mock_state["custom_records"]["app2"]) + len(mock_state["custom_records"]["app3"]),
            "file": sql_file_path,
        }
    except Exception as exc:
        return {"error": str(exc), "total": 0}



# Mock server starts clean in state 0 (no preloaded demo data)


class LoginRequest(BaseModel):
    username: str
    password: str


# --- Deterministic Dataset Generators ---
def generate_base_datasets():
    """
    Generate initial 100 records:
      - App 1 (Inventory): 40 records (APP1-ITEM-001 to APP1-ITEM-040)
      - App 2 (CRM): 35 records (APP2-CUST-001 to APP2-CUST-035)
      - App 3 (Support): 25 records (APP3-TCKT-001 to APP3-TCKT-025)
    Total = 40 + 35 + 25 = 100 records.
    """
    app1_records = [
        {
            "id": f"APP1-ITEM-{i:03d}",
            "sku": f"SKU-{1000 + i}",
            "name": f"Enterprise Product #{i}",
            "category": "Hardware" if i % 2 == 0 else "Accessories",
            "stock": 100 + (i * 5),
            "unit_price": 29.99 + (i * 2.5),
            "warehouse": "Primary-East" if i % 3 == 0 else "Secondary-West",
        }
        for i in range(1, 41)
    ]

    app2_records = [
        {
            "id": f"APP2-CUST-{i:03d}",
            "account_name": f"Global Client Corp {i}",
            "contact_email": f"client{i}@corp.example.com",
            "tier": "Enterprise" if i % 3 == 0 else "Standard",
            "status": "active",
            "total_spend": 5000.0 + (i * 250.0),
            "assigned_rep": f"Agent {((i - 1) % 5) + 1}",
        }
        for i in range(1, 36)
    ]

    app3_records = [
        {
            "id": f"APP3-TCKT-{i:03d}",
            "title": f"System Incident Ticket #{i}",
            "priority": "critical" if i % 5 == 0 else ("high" if i % 2 == 0 else "normal"),
            "status": "open" if i % 2 == 0 else "resolved",
            "assigned_team": "Infrastructure" if i % 3 == 0 else "Application",
            "sla_hours": 4 if i % 5 == 0 else 24,
        }
        for i in range(1, 26)
    ]

    return app1_records, app2_records, app3_records


def generate_modified_datasets():
    """
    Generate modified 105 records for Test Case 3 & 4:
      - 5 new records:
          * APP1-ITEM-041, APP1-ITEM-042, APP1-ITEM-043 (3 new)
          * APP2-CUST-036 (1 new)
          * APP3-TCKT-026 (1 new)
      - 3 modified records:
          * APP1-ITEM-001: stock changed from 105 to 999, unit_price changed
          * APP2-CUST-002: status changed from 'active' to 'vip_priority'
          * APP3-TCKT-003: priority changed from 'normal' to 'escalated'
      - 97 identical records (unchanged).
    Total = 97 + 3 + 5 = 105 records.
    """
    app1_records, app2_records, app3_records = generate_base_datasets()

    # Modify 1 in App 1
    app1_records[0]["stock"] = 999
    app1_records[0]["unit_price"] = 199.99

    # Add 3 new to App 1
    for i in range(41, 44):
        app1_records.append({
            "id": f"APP1-ITEM-{i:03d}",
            "sku": f"SKU-{1000 + i}",
            "name": f"Enterprise Product #{i} [NEW]",
            "category": "Peripherals",
            "stock": 50,
            "unit_price": 49.99,
            "warehouse": "Central-Hub",
        })

    # Modify 1 in App 2
    app2_records[1]["status"] = "vip_priority"
    app2_records[1]["tier"] = "Premier-Platinum"

    # Add 1 new to App 2
    app2_records.append({
        "id": "APP2-CUST-036",
        "account_name": "Newly Acquired MegaCorp 36",
        "contact_email": "leadership@megacorp36.example.com",
        "tier": "Enterprise",
        "status": "onboarding",
        "total_spend": 25000.0,
        "assigned_rep": "Executive Director",
    })

    # Modify 1 in App 3
    app3_records[2]["priority"] = "escalated"
    app3_records[2]["sla_hours"] = 1

    # Add 1 new to App 3
    app3_records.append({
        "id": "APP3-TCKT-026",
        "title": "New Urgent Ticket #26",
        "priority": "critical",
        "status": "escalated",
        "assigned_team": "Core-Platform",
        "sla_hours": 2,
    })

    return app1_records, app2_records, app3_records


def get_current_records_for_apps():
    """Retrieve datasets based on current test state (0 = clean/custom, 1-2 = base 100 records, 3-4 = modified 105 records)."""
    state = mock_state["current_dataset_state"]
    if state == 0:
        return (
            copy.deepcopy(mock_state["custom_records"]["app1"]),
            copy.deepcopy(mock_state["custom_records"]["app2"]),
            copy.deepcopy(mock_state["custom_records"]["app3"]),
        )
    elif state in (1, 2):
        base1, base2, base3 = generate_base_datasets()
        return base1, base2, base3
    else:
        return generate_modified_datasets()


@mock_router.post("/{app_key}/api/records")
async def add_custom_record_to_app(app_key: str, record: Dict[str, Any]):
    """Add an individual record to a specific target software."""
    if app_key not in mock_state["custom_records"]:
        raise HTTPException(status_code=400, detail=f"Invalid app key: {app_key}")
    mock_state["custom_records"][app_key].append(record)
    return {
        "status": "success",
        "message": f"Record added to {app_key}",
        "total": len(mock_state["custom_records"][app_key]),
    }


@mock_router.delete("/mock/records")
async def clear_all_mock_records():
    """Reset all target applications to empty state."""
    mock_state["custom_records"] = {"app1": [], "app2": [], "app3": []}
    mock_state["current_dataset_state"] = 0
    return {"status": "success", "message": "All target software records cleared (0 records)."}


# --- Generic Auth & Data Handlers ---
def handle_app_login(app_key: str, req: LoginRequest):
    creds = mock_state["credentials"][app_key]
    if req.username != creds["username"] or req.password != creds["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials for {app_key}",
        )
    # Remove from invalidated tokens if re-authenticating
    mock_state["invalidated_tokens"].discard(app_key)
    return {"token": mock_state["valid_tokens"][app_key], "token_type": "Bearer"}


def verify_app_token(app_key: str, authorization: Optional[str]):
    if app_key in mock_state["simulated_failures"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulated internal server error for {app_key}",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    token = authorization.split("Bearer ")[1].strip()
    valid_token = mock_state["valid_tokens"][app_key]

    if app_key in mock_state["invalidated_tokens"] or token != valid_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token expired or unauthorized for {app_key}",
        )


# --- Software 1 Endpoints (CCTV) ---
@mock_router.post("/app1/auth/login")
async def app1_login(req: LoginRequest):
    return handle_app_login("app1", req)


@mock_router.get("/app1/api/data")
async def app1_data(authorization: Optional[str] = Header(None)):
    verify_app_token("app1", authorization)
    app1_records, _, _ = get_current_records_for_apps()
    return {"records": app1_records, "total": len(app1_records), "app": "CCTV"}


# --- Software 2 Endpoints (Food) ---
@mock_router.post("/app2/auth/login")
async def app2_login(req: LoginRequest):
    return handle_app_login("app2", req)


@mock_router.get("/app2/api/data")
async def app2_data(authorization: Optional[str] = Header(None)):
    verify_app_token("app2", authorization)
    _, app2_records, _ = get_current_records_for_apps()
    return {"records": app2_records, "total": len(app2_records), "app": "Food"}


# --- Software 3 Endpoints (Transport) ---
@mock_router.post("/app3/auth/login")
async def app3_login(req: LoginRequest):
    return handle_app_login("app3", req)


@mock_router.get("/app3/api/data")
async def app3_data(authorization: Optional[str] = Header(None)):
    verify_app_token("app3", authorization)
    _, _, app3_records = get_current_records_for_apps()
    return {"records": app3_records, "total": len(app3_records), "app": "Transport"}


# --- Mock Testing Control Endpoints ---
@mock_router.post("/mock/set-state/{state_num}")
async def set_mock_state(state_num: int):
    """Set dataset state to 0 (clean/empty), 1, 2, 3, or 4 for sequential sync tests."""
    if state_num not in (0, 1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="State must be 0, 1, 2, 3, or 4")
    mock_state["current_dataset_state"] = state_num
    return {"message": f"Dataset state set to {state_num}", "current_state": state_num}


@mock_router.post("/mock/invalidate-token/{app_key}")
async def invalidate_mock_token(app_key: str):
    """Simulate token expiration/invalidation to trigger 401 on next fetch."""
    if app_key not in mock_state["valid_tokens"]:
        raise HTTPException(status_code=400, detail=f"Unknown app_key: {app_key}")
    mock_state["invalidated_tokens"].add(app_key)
    return {"message": f"Token for {app_key} invalidated. Next fetch will return 401."}


@mock_router.post("/mock/simulate-failure/{app_key}")
async def simulate_mock_failure(app_key: str, enable: bool = True):
    """Toggle simulated service failure for partial degradation testing."""
    if enable:
        mock_state["simulated_failures"].add(app_key)
    else:
        mock_state["simulated_failures"].discard(app_key)
    return {"message": f"Simulated failure for {app_key} set to {enable}"}


@mock_router.post("/mock/load-cctv-backup")
async def reload_cctv_backup():
    """Manually reload records from cctv_backup_2026-10-17.sql into mock target apps."""
    counts = load_cctv_sql_data()
    return {"status": "success", "counts": counts}


@mock_router.get("/mock/status")
async def get_mock_status():
    app1, app2, app3 = get_current_records_for_apps()
    return {
        "dataset_state": mock_state["current_dataset_state"],
        "records_count": {
            "app1": len(app1),
            "app2": len(app2),
            "app3": len(app3),
            "total": len(app1) + len(app2) + len(app3),
        },
        "invalidated_tokens": list(mock_state["invalidated_tokens"]),
        "simulated_failures": list(mock_state["simulated_failures"]),
    }

