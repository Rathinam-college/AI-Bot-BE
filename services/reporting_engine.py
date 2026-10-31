"""
Executive Reporting & Forecasting Engine for Server AI Assistant.
Provides:
  1. Automated monthly audit reports across campus departments (Food, Transport, CCTV)
  2. CSV export generation for auditing and financial tracking
  3. Time-series statistical and AI forecasting for upcoming month procurement and budget demand
"""

import csv
import io
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.db import resolve_db_path

logger = logging.getLogger("reporting_engine")


class ReportingEngine:
    """Enterprise Audit Reporting & Predictive Forecasting Engine."""

    def __init__(self):
        pass

    def _get_connection(self, db_path: Optional[str] = None) -> sqlite3.Connection:
        actual_path = str(resolve_db_path(db_path))
        conn = sqlite3.connect(actual_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_available_months(self, db_path: Optional[str] = None) -> List[str]:
        """Detect all available YYYY-MM months present in procurement, PRs, and fuel logs."""
        conn = self._get_connection(db_path)
        months = set()
        try:
            cur = conn.cursor()
            # Check food procurement
            try:
                cur.execute("SELECT DISTINCT substr(purchase_date, 1, 7) as mo FROM food_procurement WHERE purchase_date IS NOT NULL AND purchase_date != '';")
                for r in cur.fetchall():
                    if r["mo"] and len(r["mo"]) == 7:
                        months.add(r["mo"])
            except Exception:
                pass

            # Check purchase requests
            try:
                cur.execute("SELECT DISTINCT substr(order_date, 1, 7) as mo FROM food_purchase_requests WHERE order_date IS NOT NULL AND order_date != '';")
                for r in cur.fetchall():
                    if r["mo"] and len(r["mo"]) == 7:
                        months.add(r["mo"])
            except Exception:
                pass

            # Check fuel logs
            try:
                cur.execute("SELECT DISTINCT substr(date, 1, 7) as mo FROM transport_fuel_entries WHERE date IS NOT NULL AND date != '';")
                for r in cur.fetchall():
                    if r["mo"] and len(r["mo"]) == 7:
                        months.add(r["mo"])
            except Exception:
                pass
        finally:
            conn.close()

        sorted_months = sorted(list(months), reverse=True)
        if not sorted_months:
            # Fallback to current year-month
            now_mo = datetime.now(timezone.utc).strftime("%Y-%m")
            return [now_mo]
        return sorted_months

    def generate_monthly_audit_report(
        self,
        month: Optional[str] = None,
        department: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive monthly financial & operational audit across campus services.
        If month is omitted, defaults to latest available month.
        """
        avail_months = self.get_available_months(db_path)
        selected_month = (month or "").strip()
        if not selected_month or len(selected_month) != 7:
            selected_month = avail_months[0]

        dept_filter = (department or "").strip().title() if department else None
        conn = self._get_connection(db_path)
        try:
            cur = conn.cursor()

            # -------------------------------------------------------------
            # 1. FOOD & CAFETERIA AUDIT
            # -------------------------------------------------------------
            food_audit = {
                "total_procurement_spend": 0.0,
                "procurement_line_items": 0,
                "total_prs_amount": 0.0,
                "prs_count": 0,
                "pr_status_breakdown": {},
                "top_vendors": [],
                "top_commodities": [],
            }

            if not dept_filter or dept_filter == "Food":
                # Procurement items
                try:
                    cur.execute(
                        """
                        SELECT COUNT(*) as cnt, COALESCE(SUM(total_price), 0.0) as total
                        FROM food_procurement
                        WHERE substr(purchase_date, 1, 7) = ?;
                        """,
                        (selected_month,),
                    )
                    row = cur.fetchone()
                    if row:
                        food_audit["procurement_line_items"] = row["cnt"]
                        food_audit["total_procurement_spend"] = round(row["total"], 2)

                    # Top vendors
                    cur.execute(
                        """
                        SELECT vendor_name, COUNT(*) as items, COALESCE(SUM(total_price), 0.0) as total_val
                        FROM food_procurement
                        WHERE substr(purchase_date, 1, 7) = ? AND vendor_name IS NOT NULL AND vendor_name != ''
                        GROUP BY vendor_name
                        ORDER BY total_val DESC LIMIT 5;
                        """,
                        (selected_month,),
                    )
                    food_audit["top_vendors"] = [
                        {"vendor": r["vendor_name"], "items_count": r["items"], "amount": round(r["total_val"], 2)}
                        for r in cur.fetchall()
                    ]

                    # Top commodities
                    cur.execute(
                        """
                        SELECT item_name, COALESCE(unit, 'Units') as unit, COALESCE(SUM(quantity), 0.0) as qty, COALESCE(SUM(total_price), 0.0) as total_val
                        FROM food_procurement
                        WHERE substr(purchase_date, 1, 7) = ? AND item_name IS NOT NULL AND item_name != ''
                        GROUP BY item_name
                        ORDER BY total_val DESC LIMIT 6;
                        """,
                        (selected_month,),
                    )
                    food_audit["top_commodities"] = [
                        {"item": r["item_name"], "unit": r["unit"], "quantity": round(r["qty"], 1), "amount": round(r["total_val"], 2)}
                        for r in cur.fetchall()
                    ]
                except Exception as exc:
                    logger.warning(f"Error reading food procurement in audit: {exc}")

                # Purchase Requests
                try:
                    cur.execute(
                        """
                        SELECT COUNT(*) as cnt, COALESCE(SUM(total_amount), 0.0) as total
                        FROM food_purchase_requests
                        WHERE substr(order_date, 1, 7) = ?;
                        """,
                        (selected_month,),
                    )
                    pr_row = cur.fetchone()
                    if pr_row:
                        food_audit["prs_count"] = pr_row["cnt"]
                        food_audit["total_prs_amount"] = round(pr_row["total"], 2)

                    cur.execute(
                        """
                        SELECT COALESCE(status, 'Completed') as st, COUNT(*) as cnt, COALESCE(SUM(total_amount), 0.0) as total
                        FROM food_purchase_requests
                        WHERE substr(order_date, 1, 7) = ?
                        GROUP BY st;
                        """,
                        (selected_month,),
                    )
                    food_audit["pr_status_breakdown"] = {
                        r["st"]: {"count": r["cnt"], "amount": round(r["total"], 2)}
                        for r in cur.fetchall()
                    }
                except Exception as exc:
                    logger.warning(f"Error reading food PRs in audit: {exc}")

            # -------------------------------------------------------------
            # 2. TRANSPORT FLEET AUDIT
            # -------------------------------------------------------------
            transport_audit = {
                "total_fuel_spend": 0.0,
                "total_fuel_liters": 0.0,
                "fuel_logs_count": 0,
                "avg_cost_per_liter": 0.0,
                "total_fleet_km": 0.0,
                "vehicles_fuel_breakdown": [],
                "renewals_spend": 0.0,
            }

            if not dept_filter or dept_filter == "Transport":
                try:
                    cur.execute(
                        """
                        SELECT COUNT(*) as cnt, COALESCE(SUM(amount), 0.0) as total_amt, COALESCE(SUM(liters), 0.0) as total_liters, COALESCE(SUM(total_km), 0.0) as total_km
                        FROM transport_fuel_entries
                        WHERE substr(date, 1, 7) = ?;
                        """,
                        (selected_month,),
                    )
                    f_row = cur.fetchone()
                    if f_row and f_row["cnt"]:
                        transport_audit["fuel_logs_count"] = f_row["cnt"]
                        transport_audit["total_fuel_spend"] = round(f_row["total_amt"], 2)
                        transport_audit["total_fuel_liters"] = round(f_row["total_liters"], 1)
                        transport_audit["total_fleet_km"] = round(f_row["total_km"], 1)
                        if f_row["total_liters"] > 0:
                            transport_audit["avg_cost_per_liter"] = round(f_row["total_amt"] / f_row["total_liters"], 2)

                    # Vehicle breakdown
                    cur.execute(
                        """
                        SELECT vehicle_number, COUNT(*) as logs, COALESCE(SUM(amount), 0.0) as total_amt, COALESCE(SUM(liters), 0.0) as liters
                        FROM transport_fuel_entries
                        WHERE substr(date, 1, 7) = ? AND vehicle_number IS NOT NULL
                        GROUP BY vehicle_number
                        ORDER BY total_amt DESC;
                        """,
                        (selected_month,),
                    )
                    transport_audit["vehicles_fuel_breakdown"] = [
                        {"vehicle": r["vehicle_number"], "logs": r["logs"], "amount": round(r["total_amt"], 2), "liters": round(r["liters"], 1)}
                        for r in cur.fetchall()
                    ]

                    # Document Renewals in that month
                    cur.execute(
                        """
                        SELECT COALESCE(SUM(amount_paid), 0.0) as total_ren
                        FROM transport_renewals
                        WHERE substr(new_expiry_date, 1, 7) = ?;
                        """,
                        (selected_month,),
                    )
                    ren_row = cur.fetchone()
                    if ren_row:
                        transport_audit["renewals_spend"] = round(ren_row["total_ren"], 2)
                except Exception as exc:
                    logger.warning(f"Error reading transport in audit: {exc}")

            # -------------------------------------------------------------
            # 3. CCTV & SECURITY INFRASTRUCTURE AUDIT
            # -------------------------------------------------------------
            cctv_audit = {
                "total_cameras": 0,
                "online_cameras": 0,
                "camera_uptime_pct": 100.0,
                "total_tickets": 0,
                "open_tickets": 0,
                "completed_tickets": 0,
                "ticket_resolution_rate_pct": 100.0,
            }

            if not dept_filter or dept_filter == "Cctv":
                try:
                    cur.execute("SELECT status, count(*) as cnt FROM cctv_cameras GROUP BY status;")
                    cam_stats = {r["status"]: r["cnt"] for r in cur.fetchall()}
                    tot_cam = sum(cam_stats.values())
                    on_cam = cam_stats.get("Online", cam_stats.get("online", tot_cam))
                    cctv_audit["total_cameras"] = tot_cam
                    cctv_audit["online_cameras"] = on_cam
                    cctv_audit["camera_uptime_pct"] = round((on_cam / tot_cam * 100.0) if tot_cam > 0 else 100.0, 1)

                    # Tickets logged or active
                    cur.execute("SELECT status, count(*) as cnt FROM cctv_tickets GROUP BY status;")
                    t_stats = {r["status"]: r["cnt"] for r in cur.fetchall()}
                    tot_t = sum(t_stats.values())
                    comp_t = t_stats.get("Completed", t_stats.get("completed", 0))
                    open_t = t_stats.get("Open", t_stats.get("open", 0))
                    cctv_audit["total_tickets"] = tot_t
                    cctv_audit["open_tickets"] = open_t
                    cctv_audit["completed_tickets"] = comp_t
                    cctv_audit["ticket_resolution_rate_pct"] = round((comp_t / tot_t * 100.0) if tot_t > 0 else 100.0, 1)
                except Exception as exc:
                    logger.warning(f"Error reading CCTV in audit: {exc}")

            # -------------------------------------------------------------
            # 4. OVERALL EXECUTIVE METRICS CONSOLIDATION
            # -------------------------------------------------------------
            total_spend = round(
                food_audit["total_procurement_spend"]
                + transport_audit["total_fuel_spend"]
                + transport_audit["renewals_spend"],
                2,
            )

            # Health index score (weighted average of camera uptime & ticket resolution)
            health_score = round(
                (cctv_audit["camera_uptime_pct"] * 0.6) + (cctv_audit["ticket_resolution_rate_pct"] * 0.4),
                1,
            )

            month_label = selected_month
            month_names = {
                "01": "January", "02": "February", "03": "March", "04": "April",
                "05": "May", "06": "June", "07": "July", "08": "August",
                "09": "September", "10": "October", "11": "November", "12": "December"
            }
            if "-" in selected_month:
                yr, mo = selected_month.split("-", 1)
                month_label = f"{month_names.get(mo, mo)} {yr}"

            return {
                "status": "success",
                "audit_period": {
                    "month_key": selected_month,
                    "month_label": month_label,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "available_months": avail_months,
                },
                "executive_summary": {
                    "total_campus_spend": total_spend,
                    "total_line_items": food_audit["procurement_line_items"] + transport_audit["fuel_logs_count"],
                    "department_spending": {
                        "Food": food_audit["total_procurement_spend"],
                        "Transport": round(transport_audit["total_fuel_spend"] + transport_audit["renewals_spend"], 2),
                        "CCTV": 0.0,
                    },
                    "campus_operational_health_score": health_score,
                },
                "departments": {
                    "food": food_audit,
                    "transport": transport_audit,
                    "cctv": cctv_audit,
                },
            }
        finally:
            conn.close()

    def generate_monthly_audit_csv(
        self,
        month: Optional[str] = None,
        department: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> str:
        """Export monthly audit figures and item ledger into structured CSV format."""
        report = self.generate_monthly_audit_report(month=month, department=department, db_path=db_path)
        selected_month = report["audit_period"]["month_key"]
        month_label = report["audit_period"]["month_label"]
        exec_sum = report["executive_summary"]
        food = report["departments"]["food"]
        trans = report["departments"]["transport"]
        cctv = report["departments"]["cctv"]

        output = io.StringIO()
        writer = csv.writer(output)

        # Header block
        writer.writerow(["================================================================"])
        writer.writerow([f"RATHINAM CAMPUS EXECUTIVE AUDIT REPORT - {month_label.upper()}"])
        writer.writerow([f"Report Month: {selected_month}", f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"])
        writer.writerow(["================================================================"])
        writer.writerow([])

        # Summary KPIs
        writer.writerow(["EXECUTIVE SUMMARY KPI", "VALUE"])
        writer.writerow(["Total Campus Operational Spend (INR)", f"Rs. {exec_sum['total_campus_spend']:,.2f}"])
        writer.writerow(["Food Procurement Spend", f"Rs. {exec_sum['department_spending']['Food']:,.2f}"])
        writer.writerow(["Transport Fleet Spend", f"Rs. {exec_sum['department_spending']['Transport']:,.2f}"])
        writer.writerow(["Operational Health Score", f"{exec_sum['campus_operational_health_score']}%"])
        writer.writerow(["CCTV Camera Uptime", f"{cctv['camera_uptime_pct']}% ({cctv['online_cameras']}/{cctv['total_cameras']} Online)"])
        writer.writerow(["Service Tickets Resolution Rate", f"{cctv['ticket_resolution_rate_pct']}% ({cctv['completed_tickets']}/{cctv['total_tickets']} Resolved)"])
        writer.writerow([])

        # Top Food Vendors
        writer.writerow(["TOP FOOD VENDORS", "ITEMS PURCHASED", "TOTAL INVOICE VALUE (INR)"])
        for v in food.get("top_vendors", []):
            writer.writerow([v["vendor"], v["items_count"], f"Rs. {v['amount']:,.2f}"])
        writer.writerow([])

        # Top Food Commodities
        writer.writerow(["COMMODITY ITEM", "QUANTITY", "UNIT", "TOTAL VALUE (INR)"])
        for item in food.get("top_commodities", []):
            writer.writerow([item["item"], item["quantity"], item["unit"], f"Rs. {item['amount']:,.2f}"])
        writer.writerow([])

        # Fleet Fuel Logs
        writer.writerow(["VEHICLE NUMBER", "FUEL LOGS", "LITERS", "TOTAL FUEL SPEND (INR)"])
        for v in trans.get("vehicles_fuel_breakdown", []):
            writer.writerow([v["vehicle"], v["logs"], f"{v['liters']} L", f"Rs. {v['amount']:,.2f}"])
        writer.writerow([])

        # Detailed Food Procurement Ledger for the month
        writer.writerow(["DETAILED PROCUREMENT LEDGER"])
        writer.writerow(["Purchase Date", "Invoice No", "Vendor", "Item Name", "Quantity", "Unit", "Unit Price", "Total Price (INR)"])

        conn = self._get_connection(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT purchase_date, invoice_number, vendor_name, item_name, quantity, unit, unit_price, total_price
                FROM food_procurement
                WHERE substr(purchase_date, 1, 7) = ?
                ORDER BY purchase_date ASC LIMIT 500;
                """,
                (selected_month,),
            )
            for r in cur.fetchall():
                writer.writerow([
                    r["purchase_date"] or "",
                    r["invoice_number"] or "",
                    r["vendor_name"] or "",
                    r["item_name"] or "",
                    r["quantity"] or 0,
                    r["unit"] or "",
                    r["unit_price"] or 0,
                    r["total_price"] or 0,
                ])
        except Exception:
            pass
        finally:
            conn.close()

        return output.getvalue()

    def predict_procurement_and_demand(
        self,
        department: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict next month's procurement demand and estimated budget based on
        historical time-series trends from authoritative SQLite records.
        """
        conn = self._get_connection(db_path)
        dept_filter = (department or "").strip().title() if department else None

        try:
            cur = conn.cursor()

            # 1. Historical Food Monthly Spend Series
            cur.execute(
                """
                SELECT substr(purchase_date, 1, 7) as mo, COUNT(*) as cnt, COALESCE(SUM(total_price), 0.0) as total
                FROM food_procurement
                WHERE purchase_date IS NOT NULL AND purchase_date != ''
                GROUP BY mo
                ORDER BY mo ASC;
                """
            )
            food_monthly = [
                {"month": r["mo"], "items_count": r["cnt"], "total_spend": round(r["total"], 2)}
                for r in cur.fetchall()
            ]

            # 2. Historical Transport Monthly Fuel Spend Series
            cur.execute(
                """
                SELECT substr(date, 1, 7) as mo, COUNT(*) as cnt, COALESCE(SUM(amount), 0.0) as total, COALESCE(SUM(liters), 0.0) as liters
                FROM transport_fuel_entries
                WHERE date IS NOT NULL AND date != ''
                GROUP BY mo
                ORDER BY mo ASC;
                """
            )
            fuel_monthly = [
                {"month": r["mo"], "logs_count": r["cnt"], "fuel_spend": round(r["total"], 2), "liters": round(r["liters"], 1)}
                for r in cur.fetchall()
            ]

            # 3. Low stock inventory items requiring restocking
            low_stock_items = []
            try:
                cur.execute(
                    """
                    SELECT fi.item_id, COALESCE(ft.name, fi.item_id) as item_name, fi.current_quantity, fi.min_stock_level, COALESCE(ft.unit, 'Units') as unit, fi.status
                    FROM food_inventory fi
                    LEFT JOIN food_items ft ON fi.item_id = ft.id OR fi.item_id = replace(ft.id, 'FOOD-ITEM-', '')
                    WHERE fi.current_quantity <= (fi.min_stock_level * 1.25)
                    ORDER BY fi.current_quantity ASC LIMIT 10;
                    """
                )
                low_stock_items = [
                    {
                        "item_id": r["item_id"],
                        "item_name": r["item_name"],
                        "current_stock": r["current_quantity"],
                        "min_stock_level": r["min_stock_level"],
                        "unit": r["unit"],
                        "reorder_priority": "CRITICAL" if r["current_quantity"] <= r["min_stock_level"] else "WARNING",
                    }
                    for r in cur.fetchall()
                ]
            except Exception as exc:
                logger.warning(f"Error querying low stock items: {exc}")

            # -------------------------------------------------------------
            # Statistical Time-Series Projection
            # -------------------------------------------------------------
            # Compute 3-month moving average and trend velocity
            food_spends = [m["total_spend"] for m in food_monthly if m["total_spend"] > 0]
            if len(food_spends) >= 3:
                recent_avg = sum(food_spends[-3:]) / 3.0
                last_val = food_spends[-1]
                growth_rate = ((last_val - food_spends[-2]) / food_spends[-2]) if food_spends[-2] > 0 else 0.0
                # Dampen growth rate to 15% max for conservative financial estimation
                damped_growth = max(min(growth_rate, 0.15), -0.15)
                projected_food_spend = round(recent_avg * (1.0 + damped_growth), 2)
            elif food_spends:
                recent_avg = sum(food_spends) / len(food_spends)
                projected_food_spend = round(recent_avg, 2)
                growth_rate = 0.0
            else:
                projected_food_spend = 7500000.0  # Fallback campus dining baseline
                growth_rate = 0.0

            # Transport fuel projections
            fuel_spends = [m["fuel_spend"] for m in fuel_monthly if m["fuel_spend"] > 0]
            fuel_liters = [m["liters"] for m in fuel_monthly if m["liters"] > 0]
            if fuel_spends:
                projected_fuel_spend = round(sum(fuel_spends[-3:]) / min(len(fuel_spends), 3), 2)
                projected_fuel_liters = round(sum(fuel_liters[-3:]) / min(len(fuel_liters), 3), 1)
            else:
                projected_fuel_spend = 125000.0
                projected_fuel_liters = 1200.0

            projected_total_budget = round(projected_food_spend + projected_fuel_spend, 2)

            # Determine next month label
            latest_mo = food_monthly[-1]["month"] if food_monthly else "2026-10"
            try:
                yr_int, mo_int = map(int, latest_mo.split("-"))
                next_mo_int = mo_int + 1
                next_yr_int = yr_int
                if next_mo_int > 12:
                    next_mo_int = 1
                    next_yr_int += 1
                next_month_key = f"{next_yr_int:04d}-{next_mo_int:02d}"
            except Exception:
                next_month_key = "2026-11"

            # Executive narrative summary
            trend_direction = "upward" if growth_rate > 0.02 else ("downward" if growth_rate < -0.02 else "stable")
            narrative = (
                f"Based on historical run-rates across {len(food_monthly)} recorded months, campus procurement demand for "
                f"**{next_month_key}** is trending **{trend_direction}** ({growth_rate:+.1%}). "
                f"The projected operational budget requirement is **₹{projected_total_budget:,.2f}** "
                f"(comprising **₹{projected_food_spend:,.2f}** for Food & Cafeteria procurement and "
                f"**₹{projected_fuel_spend:,.2f}** across ~{projected_fuel_liters:,.0f} Liters of campus fleet fuel). "
                f"{len(low_stock_items)} inventory commodities have been flagged for immediate reordering to prevent dining shortages."
            )

            return {
                "status": "success",
                "forecast_period": {
                    "baseline_latest_month": latest_mo,
                    "target_forecast_month": next_month_key,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                "projected_requirements": {
                    "total_campus_budget": projected_total_budget,
                    "food_procurement_spend": projected_food_spend,
                    "transport_fuel_spend": projected_fuel_spend,
                    "transport_fuel_liters": projected_fuel_liters,
                    "trend_direction": trend_direction,
                    "projected_growth_rate_pct": round(growth_rate * 100, 2),
                },
                "historical_trends": {
                    "food_monthly": food_monthly,
                    "fuel_monthly": fuel_monthly,
                },
                "restocking_recommendations": low_stock_items,
                "executive_narrative": narrative,
            }
        finally:
            conn.close()


reporting_engine = ReportingEngine()
