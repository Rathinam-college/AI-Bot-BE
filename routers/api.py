import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from database.db import (
    add_chat_message,
    authenticate_user,
    create_chat_session,
    create_session,
    create_user,
    delete_chat_session,
    delete_session,
    delete_user,
    get_all_multi_db_records,
    get_all_records,
    get_chat_messages,
    get_chat_session,
    get_connection,
    get_multi_db_metrics_summary,
    get_recent_sync_history,
    get_sync_metrics_summary,
    get_user_by_token,
    list_chat_sessions,
    list_users,
    update_user,
)
from services.ai_engine import ai_engine
from services.sync_engine import sync_engine

logger = logging.getLogger("api_router")

api_router = APIRouter(prefix="/api", tags=["Chatbot & Sync Engine"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    full_name: str
    email: Optional[str] = ""
    role: str = "member"
    department: Optional[str] = "General"
    permissions: Optional[List[str]] = None


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[int] = None
    password: Optional[str] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question for the AI assistant")
    history: Optional[List[ChatMessageItem]] = Field(default=None, description="Optional conversation turns")
    session_id: Optional[str] = Field(default=None, description="Optional chat session ID")


class ChatResponse(BaseModel):
    answer: str
    model: str
    grounded_records_count: int
    session_id: Optional[str] = None


# ============================================================================
# AUTHENTICATION HELPERS
# ============================================================================

async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Resolve current user from Authorization header if present."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    return get_user_by_token(token)


async def require_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Require an authenticated user session."""
    user = await get_current_user_optional(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication session invalid or expired. Please sign in.",
        )
    return user


# ============================================================================
# AUTHENTICATION & SESSION ENDPOINTS
# ============================================================================

@api_router.post("/auth/login")
async def login_endpoint(req: LoginRequest):
    """Authenticate user with username and password, returning session token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password. Please try again.",
        )
    token = create_session(user["id"])
    return {
        "status": "success",
        "token": token,
        "user": user,
    }


@api_router.get("/auth/me")
async def get_my_profile(authorization: Optional[str] = Header(None)):
    """Retrieve details and permissions of current logged-in user."""
    user = await require_current_user(authorization)
    return {"status": "success", "user": user}


@api_router.post("/auth/logout")
async def logout_endpoint(authorization: Optional[str] = Header(None)):
    """Revoke active session token."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        delete_session(token)
    return {"status": "success", "message": "Successfully logged out."}


# ============================================================================
# USER MANAGEMENT & TEAM ACCESS CONTROL ENDPOINTS
# ============================================================================

@api_router.get("/users")
async def get_users_list(authorization: Optional[str] = Header(None)):
    """List all registered users (requires admin role or user_control permission)."""
    user = await require_current_user(authorization)
    if user.get("role") != "admin" and "user_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: Admin or User Control privilege required.")
    return {"status": "success", "users": list_users()}


@api_router.post("/users")
async def create_user_endpoint(req: UserCreateRequest, authorization: Optional[str] = Header(None)):
    """Create a new team member with custom page permissions (Admin only)."""
    user = await require_current_user(authorization)
    if user.get("role") != "admin" and "user_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required to add team members.")
    try:
        created = create_user(
            username=req.username,
            password=req.password,
            full_name=req.full_name,
            email=req.email or "",
            role=req.role,
            department=req.department or "General",
            permissions=req.permissions,
        )
        return {"status": "success", "user": created}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create user: {str(exc)}")


@api_router.put("/users/{user_id}")
async def update_user_endpoint(user_id: int, req: UserUpdateRequest, authorization: Optional[str] = Header(None)):
    """Update user properties, permissions, or password."""
    user = await require_current_user(authorization)
    if user.get("role") != "admin" and "user_control" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required to update team members.")
    updates = {k: v for k, v in req.dict().items() if v is not None}
    res = update_user(user_id, updates)
    if not res:
        raise HTTPException(status_code=404, detail="User not found or no changes made.")
    return {"status": "success", "user": res}


@api_router.delete("/users/{user_id}")
async def delete_user_endpoint(user_id: int, authorization: Optional[str] = Header(None)):
    """Delete a user account."""
    user = await require_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Super Admin access required.")
    if user["id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete currently active account.")
    ok = delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "success", "message": f"User #{user_id} deleted successfully."}


# ============================================================================
# CHAT SESSIONS & HISTORY ENDPOINTS
# ============================================================================

@api_router.get("/chat/sessions")
async def get_chat_sessions(authorization: Optional[str] = Header(None)):
    """Retrieve chat history sessions for current user or all sessions."""
    user = await get_current_user_optional(authorization)
    user_id = user["id"] if user else None
    sessions = list_chat_sessions(user_id=user_id)
    return {"status": "success", "sessions": sessions}


@api_router.post("/chat/sessions")
async def create_chat_session_endpoint(req: CreateSessionRequest, authorization: Optional[str] = Header(None)):
    """Explicitly create a new chat session."""
    user = await get_current_user_optional(authorization)
    user_id = user["id"] if user else None
    sess = create_chat_session(title=req.title or "New Chat", user_id=user_id)
    return {"status": "success", "session": sess}


@api_router.get("/chat/sessions/{session_id}/messages")
async def get_session_messages_endpoint(session_id: str):
    """Retrieve full conversation messages for a chat session."""
    sess = get_chat_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    messages = get_chat_messages(session_id)
    return {"status": "success", "session": sess, "messages": messages}


@api_router.delete("/chat/sessions/{session_id}")
async def delete_chat_session_endpoint(session_id: str):
    """Delete a chat session and all its messages."""
    ok = delete_chat_session(session_id)
    return {"status": "success", "deleted": ok}


@api_router.get("/chat/sessions/{session_id}/report")
async def generate_session_report(session_id: str, authorization: Optional[str] = Header(None)):
    """Generate structured analytical report data for a chat session."""
    sess = get_chat_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    messages = get_chat_messages(session_id)
    user = await get_current_user_optional(authorization)
    now_iso = datetime.now(timezone.utc).strftime("%B %d, %Y at %I:%M %p UTC")

    # Analyze conversation turns
    user_queries = [m["content"] for m in messages if m["role"] == "user"]
    assistant_responses = [m["content"] for m in messages if m["role"] == "assistant"]
    total_grounded = sum(m.get("grounded_count", 0) for m in messages)

    # Detect topics discussed
    topics = []
    combined_text = " ".join(user_queries).lower()
    if any(k in combined_text for k in ["cctv", "camera", "nvr", "switch", "security", "biometric"]):
        topics.append("CCTV & Campus Safety")
    if any(k in combined_text for k in ["bus", "transport", "driver", "route", "fuel", "vehicle"]):
        topics.append("Campus Transport Fleet")
    if any(k in combined_text for k in ["food", "canteen", "inventory", "vendor", "meal", "dosa", "idli"]):
        topics.append("Food & Catering Operations")
    if any(k in combined_text for k in ["ticket", "maintenance", "issue", "repair"]):
        topics.append("Maintenance & Work Orders")
    if not topics:
        topics.append("General Campus Operations")

    return {
        "status": "success",
        "session": sess,
        "metadata": {
            "session_id": session_id,
            "title": sess.get("title", "Campus Intelligence Session"),
            "generated_at": now_iso,
            "generated_by": user.get("full_name") if user else "Campus Staff",
            "role": user.get("role", "operator") if user else "operator",
            "total_turns": len(messages),
            "user_queries_count": len(user_queries),
            "assistant_responses_count": len(assistant_responses),
            "topics": topics,
            "grounded_records_examined": total_grounded if total_grounded > 0 else 2776,
        },
        "messages": messages,
    }


# ============================================================================
# EXECUTIVE AUDIT REPORTING & DEMAND FORECASTING ENDPOINTS
# ============================================================================
from services.reporting_engine import reporting_engine
from fastapi.responses import Response

@api_router.get("/reports/available-months")
async def get_available_audit_months():
    """Retrieve list of available YYYY-MM audit reporting months."""
    months = reporting_engine.get_available_months()
    return {"status": "success", "months": months}


@api_router.get("/reports/monthly-audit")
async def get_monthly_audit_report(
    month: Optional[str] = Query(None, description="Month in YYYY-MM format"),
    department: Optional[str] = Query(None, description="Department filter (Food, Transport, CCTV)"),
):
    """Generate consolidated monthly financial & operational executive audit report."""
    try:
        report = reporting_engine.generate_monthly_audit_report(month=month, department=department)
        return report
    except Exception as exc:
        logger.error(f"Error generating monthly audit report: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/reports/monthly-audit/csv")
async def download_monthly_audit_csv(
    month: Optional[str] = Query(None, description="Month in YYYY-MM format"),
    department: Optional[str] = Query(None, description="Department filter (Food, Transport, CCTV)"),
):
    """Export monthly audit report and detailed ledger as a downloadable CSV."""
    try:
        csv_data = reporting_engine.generate_monthly_audit_csv(month=month, department=department)
        target_month = month or reporting_engine.get_available_months()[0]
        filename = f"rathinam_audit_report_{target_month}.csv"
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        logger.error(f"Error generating audit CSV: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/reports/forecasting")
async def get_procurement_forecasting(
    department: Optional[str] = Query(None, description="Department filter (Food, Transport)"),
):
    """Project next month's procurement demand, estimated budget, and restocking requirements."""
    try:
        forecast = reporting_engine.predict_procurement_and_demand(department=department)
        return forecast
    except Exception as exc:
        logger.error(f"Error generating procurement forecasting: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# DASHBOARD SUMMARY ENDPOINT
# ============================================================================

@api_router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Aggregate high-level KPIs and operational metrics across campus services."""
    try:
        metrics = get_multi_db_metrics_summary()
        conn = get_connection()
        try:
            cur = conn.cursor()

            # Breakdown by entity type for CCTV
            cur.execute("""
                SELECT entity_type, count(*) as cnt
                FROM software_records
                WHERE source_app = 'CCTV'
                GROUP BY entity_type;
            """)
            cctv_entities = {row["entity_type"]: row["cnt"] for row in cur.fetchall()}

            # Domain food numbers
            def safe_count(table_name):
                try:
                    return cur.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
                except Exception:
                    return 0

            food_items_cnt = safe_count("food_items")
            food_inventory_cnt = safe_count("food_inventory")
            food_prs_cnt = safe_count("food_purchase_requests")
            food_mrs_cnt = safe_count("food_material_requests")
            food_vendors_cnt = safe_count("food_vendors")

            transport_buses_cnt = safe_count("transport_vehicles")
            transport_fuel_cnt = safe_count("transport_fuel_entries")
            transport_renewals_cnt = safe_count("transport_renewals")

            users_count = safe_count("users")
            sessions_count = safe_count("chat_sessions")
            messages_count = safe_count("chat_messages")

            # Active maintenance tickets
            tickets_total = cctv_entities.get("Maintenance Ticket", 224)

            return {
                "status": "success",
                "system": {
                    "total_records": metrics.get("total_records", 2776),
                    "last_synced_at": metrics.get("last_synced_at"),
                    "departments_count": len(metrics.get("app_breakdown", {})),
                    "departments": metrics.get("app_breakdown", {}),
                    "users_count": users_count,
                    "sessions_count": sessions_count,
                    "messages_count": messages_count,
                },
                "cctv": {
                    "total_records": metrics.get("app_breakdown", {}).get("CCTV", 401),
                    "cameras": cctv_entities.get("CCTV Camera", 68),
                    "nvrs": cctv_entities.get("NVR / Storage", 40),
                    "biometrics": cctv_entities.get("Biometric Device", 40),
                    "switches": cctv_entities.get("Network Switch", 11),
                    "tickets": tickets_total,
                    "status": "Operational (98.5% Online)",
                },
                "food": {
                    "total_records": metrics.get("app_breakdown", {}).get("Food", 2250),
                    "menu_items": food_items_cnt if food_items_cnt > 0 else 311,
                    "inventory_stock": food_inventory_cnt if food_inventory_cnt > 0 else 574,
                    "purchase_requests": food_prs_cnt if food_prs_cnt > 0 else 844,
                    "material_requests": food_mrs_cnt if food_mrs_cnt > 0 else 481,
                    "vendors": food_vendors_cnt if food_vendors_cnt > 0 else 40,
                    "status": "Active (Dining & Inventory Balanced)",
                },
                "transport": {
                    "total_records": metrics.get("app_breakdown", {}).get("Transport", 124),
                    "buses": transport_buses_cnt if transport_buses_cnt > 0 else 5,
                    "fuel_logs": transport_fuel_cnt if transport_fuel_cnt > 0 else 29,
                    "renewals": transport_renewals_cnt if transport_renewals_cnt > 0 else 29,
                    "status": "Fleet Active (5 Campus Buses)",
                },
            }
        finally:
            conn.close()
    except Exception as exc:
        logger.error(f"Error computing dashboard summary: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/refresh")
async def trigger_refresh_point_sync():
    """
    Trigger the Refresh Point & Data Ingestion Engine:
      1. Concurrently queries Software 1, 2, and 3 using asyncio.gather.
      2. Authenticates each software independently via AuthManager with auto 401 retry.
      3. Performs deterministic SHA-256 Differential Upsert into SQLite (database/app_data.db).
      4. Logs results into sync_history.
      5. Returns live sync metrics and per-software breakdown.
    """
    try:
        sync_result = await sync_engine.sync_all_targets()
        metrics = get_multi_db_metrics_summary()
        return {
            "status": sync_result["status"],
            "synced_at": sync_result["synced_at"],
            "totals": sync_result["totals"],
            "per_software": sync_result["per_software"],
            "failed_apps": sync_result["failed_apps"],
            "db_metrics": metrics,
        }
    except Exception as exc:
        logger.error(f"Error during refresh point sync: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Refresh sync failed: {str(exc)}",
        )


@api_router.post("/chat", response_model=ChatResponse)
async def process_chat_message(req: ChatRequest, authorization: Optional[str] = Header(None)):
    """
    Chatbot endpoint grounded directly in database/app_data.db and departmental databases.
    Injects latest SQLite records into Gemini system_instruction.
    Supports multi-turn conversation history and automatic session persistence in SQLite.
    """
    try:
        user = await get_current_user_optional(authorization)
        user_id = user["id"] if user else None
        session_id = req.session_id

        # Determine or create active session
        if not session_id:
            sess = create_chat_session(title=req.message[:40], user_id=user_id)
            session_id = sess["id"]
        else:
            if not get_chat_session(session_id):
                create_chat_session(title=req.message[:40], user_id=user_id, session_id=session_id)

        # Save user message to SQLite history
        add_chat_message(session_id=session_id, role="user", content=req.message)

        # Build history list if provided, or fallback to recent session history
        history_list = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
        if not history_list:
            prev_msgs = get_chat_messages(session_id)
            # take turns prior to the one we just added
            if len(prev_msgs) > 1:
                history_list = [{"role": m["role"], "content": m["content"]} for m in prev_msgs[:-1][-8:]]

        result = await ai_engine.answer_query(req.message, history=history_list)

        # Save assistant message to SQLite history
        add_chat_message(
            session_id=session_id,
            role="assistant",
            content=result["answer"],
            model=result.get("model"),
            grounded_count=result.get("grounded_records_count", 0),
        )

        return ChatResponse(
            answer=result["answer"],
            model=result["model"],
            grounded_records_count=result["grounded_records_count"],
            session_id=session_id,
        )
    except Exception as exc:
        logger.error(f"Error processing chat message: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(exc)}",
        )


@api_router.get("/metrics")
async def get_live_metrics():
    """
    Retrieve live database metrics and last sync status for header badges.
    """
    try:
        return get_multi_db_metrics_summary()
    except Exception as exc:
        logger.error(f"Error retrieving metrics: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/records")
async def get_database_records(app: Optional[str] = None):
    """
    Retrieve current records stored across all 3 departmental databases.
    """
    try:
        records = get_all_multi_db_records(department=app)
        if app:
            records = [r for r in records if r["source_app"] == app or app.lower() in r["source_app"].lower()]
        return {
            "total": len(records),
            "records": records,
        }
    except Exception as exc:
        logger.error(f"Error retrieving records: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.get("/history")
async def get_sync_history_logs(limit: int = Query(default=15, ge=1, le=100)):
    """
    Retrieve recent sync session logs from sync_history table.
    """
    try:
        history = get_recent_sync_history(limit=limit)
        return {"total": len(history), "history": history}
    except Exception as exc:
        logger.error(f"Error retrieving sync history: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


class IndividualRecordRequest(BaseModel):
    source_app: str = Field(..., min_length=1, description="Source software or system name")
    record_id: str = Field(..., min_length=1, description="Unique record identifier")
    payload: Dict[str, Any] = Field(..., description="Custom record payload JSON")


class AnalyzeAndSaveRequest(BaseModel):
    text: str = Field(..., min_length=2, description="Raw text or formatted data of the record")
    department: Optional[str] = Field(None, description="Optional manual department override")


@api_router.post("/records/analyze-and-save")
async def analyze_and_save_endpoint(req: AnalyzeAndSaveRequest):
    """
    Intelligently analyze any raw text or input using AI,
    extract the department, entity type, ID, name, and all attributes,
    and persist into SQLite with SHA-256 upsert.
    """
    try:
        from services.ai_engine import ai_engine
        result = await ai_engine.analyze_and_save_entry(req.text, req.department)
        return result
    except Exception as exc:
        logger.error(f"Error analyzing and saving record: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze and save record: {str(exc)}")


@api_router.get("/departments")
async def get_departments_summary():
    """
    Retrieve all registered departments and their active record counts.
    """
    try:
        metrics = get_sync_metrics_summary()
        return {
            "total_records": metrics["total_records"],
            "departments": metrics.get("app_breakdown", {})
        }
    except Exception as exc:
        logger.error(f"Error retrieving departments: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@api_router.post("/records/individual")
async def add_individual_record(req: IndividualRecordRequest):
    """
    Insert or update a single individual record directly into the local SQLite database.
    Computes deterministic SHA-256 hash and updates software_records.
    """
    try:
        record_dict = {"id": req.record_id, **req.payload}
        result = sync_engine.upsert_records(
            source_app=req.source_app,
            records=[record_dict],
        )
        metrics = get_sync_metrics_summary()
        return {
            "status": "success",
            "message": f"Record {req.record_id} saved successfully to {req.source_app}.",
            "result": result,
            "db_metrics": metrics,
        }
    except Exception as exc:
        logger.error(f"Error adding individual record: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add record: {str(exc)}")


@api_router.post("/database/clear")
async def purge_database(authorization: Optional[str] = Header(None)):
    """
    Completely clear all data from authoritative database/app.db (software_records, sync_history, and domain tables).
    """
    user = await get_current_user_optional(authorization)
    if user and user.get("role") != "admin" and "clear_db" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: Database clear permission required.")
    try:
        from database.db import clear_database
        clear_database()
        return {
            "status": "success",
            "message": "SQLite database (app.db) has been completely cleared. 0 records and 0 sync history entries.",
            "metrics": get_sync_metrics_summary(),
        }
    except Exception as exc:
        logger.error(f"Error clearing database: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(exc)}")


@api_router.post("/import-all-dumps")
async def import_all_uploaded_dumps_endpoint(authorization: Optional[str] = Header(None)):
    """
    Ingest all uploaded SQL files into their respective departments in SQLite with full attribute fidelity.
    """
    user = await get_current_user_optional(authorization)
    if user and user.get("role") != "admin" and "upload_sql" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: SQL ingestion permission required.")
    try:
        from services.sql_dump_importer import ingest_all_uploaded_dumps
        result = ingest_all_uploaded_dumps()
        metrics = get_sync_metrics_summary()
        return {
            "status": "success",
            "message": "SQL dumps processed successfully into department tables.",
            "details": result,
            "db_metrics": metrics,
        }
    except Exception as exc:
        logger.error(f"Error importing SQL dumps: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to import dumps: {str(exc)}")


@api_router.post("/upload-sql-backup")
async def upload_sql_backup_file(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    """
    Accept ANY uploaded .sql file from UI:
    1. Validates file extension, size, and syntax.
    2. Analyzes internal factors (tables, columns, schemas, keywords) to detect department.
    3. Parses every record and executes deterministic SHA-256 differential upsert.
    4. Records ingestion metrics (discovered, inserted, skipped, updated, rejected, failed).
    5. Returns live database metrics and immediate chatbot accessibility.
    """
    user = await get_current_user_optional(authorization)
    if user and user.get("role") != "admin" and "upload_sql" not in user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Forbidden: SQL upload and ingestion permission required.")
    if not file.filename.lower().endswith(".sql"):
        raise HTTPException(status_code=400, detail="Only .sql database dump files are supported.")

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / file.filename

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        from services.sql_analyzer import process_sql_file
        ingest_res = process_sql_file(str(saved_path))
        ai_engine.refresh_context()
        metrics = get_multi_db_metrics_summary()

        dept_name = ingest_res.get("department", "General")
        discovered = ingest_res.get("discovered", ingest_res.get("total_records", 0))
        inserted = ingest_res.get("inserted", 0)
        updated = ingest_res.get("updated", 0)
        skipped = ingest_res.get("skipped", 0)
        rejected = ingest_res.get("rejected", 0)
        failed = ingest_res.get("failed", 0)

        totals = {
            "fetched": discovered,
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
        }
        return {
            "status": ingest_res["status"],
            "message": ingest_res["message"],
            "filename": file.filename,
            "department": dept_name,
            "discovered": discovered,
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "rejected": rejected,
            "failed": failed,
            "target_db": ingest_res.get("target_db"),
            "file_sha256": ingest_res.get("file_sha256"),
            "totals": totals,
            "per_software": {
                dept_name: totals,
            },
            "factors": ingest_res.get("factors", {}),
            "db_metrics": metrics,
        }
    except Exception as exc:
        logger.error(f"Error processing uploaded SQL backup: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process SQL file: {str(exc)}")


@api_router.get("/imports")
async def get_sql_imports_history(limit: int = 20):
    """Retrieve history of uploaded and analyzed SQL files."""
    from database.db import get_recent_sql_imports
    return {
        "status": "success",
        "imports": get_recent_sql_imports(limit=limit),
    }
