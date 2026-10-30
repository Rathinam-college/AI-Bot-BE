import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.db import init_all_dbs, list_users

@pytest.fixture(scope="module")
def client():
    init_all_dbs()
    with TestClient(app) as c:
        yield c

def test_login_success_and_me(client):
    # Test admin login
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "token" in data
    assert data["user"]["role"] == "admin"
    token = data["token"]

    # Test me endpoint
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["user"]["username"] == "admin"

def test_login_invalid_credentials(client):
    res = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert res.status_code == 401

def test_user_control_permissions(client):
    # Login as admin
    admin_res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    admin_token = admin_res.json()["token"]

    # Create new team member
    create_res = client.post(
        "/api/users",
        json={
            "username": "test_operator",
            "password": "testpass123",
            "full_name": "Test Operator",
            "email": "test@rathinam.ac.in",
            "role": "member",
            "department": "Transport",
            "permissions": ["dashboard", "chatbot"]
        },
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert create_res.status_code == 200
    created_user = create_res.json()["user"]
    user_id = created_user["id"]
    assert created_user["username"] == "test_operator"

    # Member login
    member_res = client.post("/api/auth/login", json={"username": "test_operator", "password": "testpass123"})
    assert member_res.status_code == 200
    member_token = member_res.json()["token"]

    # Member tries to access /api/users -> should be forbidden (403)
    forbidden_res = client.get("/api/users", headers={"Authorization": f"Bearer {member_token}"})
    assert forbidden_res.status_code == 403

    # Clean up test user
    del_res = client.delete(f"/api/users/{user_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert del_res.status_code == 200

def test_chat_sessions_and_report(client):
    # Create chat session
    sess_res = client.post("/api/chat/sessions", json={"title": "Camera Status Check"})
    assert sess_res.status_code == 200
    sess = sess_res.json()["session"]
    sess_id = sess["id"]

    # List sessions
    list_res = client.get("/api/chat/sessions")
    assert list_res.status_code == 200
    assert any(s["id"] == sess_id for s in list_res.json()["sessions"])

    # Test report generation endpoint
    report_res = client.get(f"/api/chat/sessions/{sess_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert report_data["status"] == "success"
    assert "metadata" in report_data

    # Delete session
    del_sess = client.delete(f"/api/chat/sessions/{sess_id}")
    assert del_sess.status_code == 200

def test_dashboard_summary(client):
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "system" in data
    assert "cctv" in data
    assert "food" in data
    assert "transport" in data
