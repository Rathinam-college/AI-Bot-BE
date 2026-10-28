import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from database.db import (
    get_all_multi_db_records,
    get_all_records,
    get_multi_db_metrics_summary,
    get_recent_sync_history,
    get_sync_metrics_summary,
)
from services.ai_engine import ai_engine
from services.sync_engine import sync_engine

logger = logging.getLogger("api_router")

api_router = APIRouter(prefix="/api", tags=["Chatbot & Sync Engine"])


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question for the AI assistant")
    history: Optional[List[ChatMessageItem]] = Field(default=None, description="Optional conversation turns")


class ChatResponse(BaseModel):
    answer: str
    model: str
    grounded_records_count: int


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
async def process_chat_message(req: ChatRequest):
    """
    Chatbot endpoint grounded directly in database/app_data.db and departmental databases.
    Injects latest SQLite records into Gemini system_instruction.
    Supports multi-turn conversation history.
    """
    try:
        history_list = [{"role": h.role, "content": h.content} for h in req.history] if req.history else []
        result = await ai_engine.answer_query(req.message, history=history_list)
        return ChatResponse(
            answer=result["answer"],
            model=result["model"],
            grounded_records_count=result["grounded_records_count"],
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
async def purge_database():
    """
    Completely clear all data from authoritative database/app.db (software_records, sync_history, and domain tables).
    """
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
async def import_all_uploaded_dumps_endpoint():
    """
    Ingest all uploaded SQL files into their respective departments in SQLite with full attribute fidelity.
    """
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
async def upload_sql_backup_file(file: UploadFile = File(...)):
    """
    Accept ANY uploaded .sql file from UI:
    1. Validates file extension, size, and syntax.
    2. Analyzes internal factors (tables, columns, schemas, keywords) to detect department.
    3. Parses every record and executes deterministic SHA-256 differential upsert.
    4. Records ingestion metrics (discovered, inserted, skipped, updated, rejected, failed).
    5. Returns live database metrics and immediate chatbot accessibility.
    """
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
