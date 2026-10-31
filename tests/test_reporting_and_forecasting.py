import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from database.db import init_all_dbs
from main import app
from services.reporting_engine import reporting_engine


@pytest.fixture(scope="module")
def client():
    init_all_dbs()
    with TestClient(app) as c:
        yield c


def test_reporting_engine_available_months():
    """Verify reporting engine discovers all available audit months."""
    months = reporting_engine.get_available_months()
    assert isinstance(months, list)
    assert len(months) >= 1
    # Check YYYY-MM format
    for m in months:
        assert len(m) == 7
        assert m[4] == "-"


def test_reporting_engine_monthly_audit_report():
    """Verify consolidated executive monthly audit generation."""
    report = reporting_engine.generate_monthly_audit_report()
    assert report["status"] == "success"
    assert "audit_period" in report
    assert "executive_summary" in report
    assert "departments" in report

    exec_sum = report["executive_summary"]
    assert "total_campus_spend" in exec_sum
    assert "department_spending" in exec_sum
    assert "campus_operational_health_score" in exec_sum
    assert exec_sum["campus_operational_health_score"] >= 0.0

    depts = report["departments"]
    assert "food" in depts
    assert "transport" in depts
    assert "cctv" in depts


def test_reporting_engine_csv_export():
    """Verify structured CSV export generation for auditing."""
    csv_text = reporting_engine.generate_monthly_audit_csv()
    assert isinstance(csv_text, str)
    assert len(csv_text) > 100
    assert "RATHINAM CAMPUS EXECUTIVE AUDIT REPORT" in csv_text
    assert "EXECUTIVE SUMMARY KPI" in csv_text
    assert "Total Campus Operational Spend" in csv_text


def test_reporting_engine_predictive_forecasting():
    """Verify time-series and demand forecasting generation."""
    forecast = reporting_engine.predict_procurement_and_demand()
    assert forecast["status"] == "success"
    assert "forecast_period" in forecast
    assert "projected_requirements" in forecast
    assert "executive_narrative" in forecast

    reqs = forecast["projected_requirements"]
    assert "total_campus_budget" in reqs
    assert "food_procurement_spend" in reqs
    assert "transport_fuel_spend" in reqs
    assert "trend_direction" in reqs
    assert reqs["total_campus_budget"] > 0
    assert len(forecast["executive_narrative"]) > 20


def test_api_available_months_endpoint(client):
    """Verify GET /api/reports/available-months returns valid month list."""
    res = client.get("/api/reports/available-months")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "months" in data
    assert len(data["months"]) >= 1


def test_api_monthly_audit_endpoint(client):
    """Verify GET /api/reports/monthly-audit returns structured audit JSON."""
    res = client.get("/api/reports/monthly-audit")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "executive_summary" in data
    assert "departments" in data


def test_api_monthly_audit_csv_download(client):
    """Verify GET /api/reports/monthly-audit/csv returns downloadable CSV stream."""
    res = client.get("/api/reports/monthly-audit/csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers.get("content-type", "")
    assert "attachment" in res.headers.get("content-disposition", "")
    assert len(res.text) > 100
    assert "EXECUTIVE SUMMARY KPI" in res.text


def test_api_forecasting_endpoint(client):
    """Verify GET /api/reports/forecasting returns predictive budget metrics."""
    res = client.get("/api/reports/forecasting")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "projected_requirements" in data
    assert "executive_narrative" in data

