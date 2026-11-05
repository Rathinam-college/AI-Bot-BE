import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

# Load environment configuration from candidate locations
for env_path in [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=True)
load_dotenv(override=False)

from database.db import get_active_db_engine, init_all_dbs
from database.postgres_db import get_postgres_config, is_postgres_available
from routers.api import api_router
from routers.mock_target_apps import mock_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure clean database tables exist
    active_engine = get_active_db_engine()
    if active_engine == "postgresql":
        logger.info("[STARTUP] Initializing authoritative PostgreSQL enterprise database (campus_ai_db)...")
    else:
        logger.info("[STARTUP] Initializing SQLite database (database/app.db)...")
    init_all_dbs()
    logger.info(f"[STARTUP] Chatbot backend startup completed with active engine: {active_engine.upper()}.")

    # Check Gemini AI integration status
    from services.ai_engine import ai_engine
    api_key, var_name = ai_engine._get_api_key()
    if api_key:
        logger.info(f"[STARTUP AI CHECK] Gemini API key configured via '{var_name}': {ai_engine._mask_key(api_key)}. Candidate models: {ai_engine.get_candidate_models()}")
    else:
        logger.warning("[STARTUP AI CHECK] No Gemini API key configured. Running in local grounding fallback mode.")

    yield
    logger.info("Chatbot backend application shutting down.")


app = FastAPI(
    title="Rathinam Server AI Assistant - Consolidated Backend",
    description="Single Centralized FastAPI Backend integrating Authentication, Chatbot, Department Data, Domain Management, and Analytics with PostgreSQL Persistence.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include primary API routes
app.include_router(api_router)

# Mount internal mock routes directly into the single backend (eliminates separate port 8085 mock server)
app.include_router(mock_router)


# ============================================================================
# SYSTEM HEALTH & TELEMETRY ENDPOINT
# ============================================================================
@app.get("/health", tags=["System Health"])
@app.get("/api/health", tags=["System Health"])
async def health_check():
    """Consolidated Single Backend Health Check & Engine Telemetry."""
    engine = get_active_db_engine()
    pg_ok = is_postgres_available()
    cfg = get_postgres_config()

    return {
        "status": "healthy",
        "service": "Rathinam Server AI Assistant",
        "architecture": "Single Consolidated Backend",
        "port": int(os.getenv("CHATBOT_PORT", "8000")),
        "database": {
            "engine": engine,
            "connected": pg_ok if engine == "postgresql" else True,
            "name": cfg.get("database") if engine == "postgresql" else "app.db",
            "host": cfg.get("host") if engine == "postgresql" else "local",
            "port": cfg.get("port") if engine == "postgresql" else None,
        },
        "modules": {
            "auth_and_users": "active",
            "chatbot_and_history": "active",
            "department_data": "active",
            "food_and_inventory": "active",
            "transport_and_fleet": "active",
            "maintenance_tickets": "active",
            "file_and_sql_uploads": "active",
            "administrative_analytics": "active",
            "integrated_target_services": "active_internal",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Serve frontend static assets directly from single backend
FRONTEND_DIR = BASE_DIR.parent / "AI-Bot-FE"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR.parent / "static"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the primary frontend interface."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Chatbot API server running. frontend/index.html not found."}


if __name__ == "__main__":
    port = int(os.getenv("CHATBOT_PORT", "8000"))
    logger.info(f"Starting Consolidated Single Backend on http://127.0.0.1:{port}...")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False, app_dir=str(BASE_DIR))
