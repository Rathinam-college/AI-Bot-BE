import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

from database.db import init_db, init_all_dbs
from routers.api import api_router
from routers.mock_target_apps import mock_router

# Load environment configuration from candidate locations
for env_path in [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
load_dotenv(override=False)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure clean SQLite database tables exist without any seed data
    logger.info("Initializing authoritative SQLite database (database/app.db)...")
    init_all_dbs()
    logger.info("Chatbot application startup completed with fresh, clean database.")

    # Check Gemini AI integration status
    from services.ai_engine import ai_engine
    api_key, var_name = ai_engine._get_api_key()
    if api_key:
        logger.info(f"[STARTUP AI CHECK] Gemini API key configured via '{var_name}': {ai_engine._mask_key(api_key)}. Candidate models: {ai_engine.get_candidate_models()}")
    else:
        logger.warning("[STARTUP AI CHECK] No Gemini API key configured (GEMINI_API_KEY/GOOGLE_API_KEY). Running in local SQLite grounding fallback mode.")

    yield
    logger.info("Chatbot application shutting down.")


app = FastAPI(
    title="Server AI Assistant",
    description="Production-ready AI Chatbot with Dedicated Refresh Point Sync Engine & SQLite Persistence",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local cross-origin API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

# Include API routes
app.include_router(api_router)


@app.api_route("/mock/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def forward_mock_request(full_path: str, request: Request):
    """Forward /mock requests from UI directly to Target Server base URL on port 8085."""
    target_base = os.getenv("TARGET_SERVER_BASE_URL", "http://127.0.0.1:8085").rstrip("/")
    target_url = f"{target_base}/mock/{full_path}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target_url,
                params=dict(request.query_params),
                content=body if body else None,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as exc:
        logger.warning(f"Failed to forward mock request to {target_url}: {exc}")
        return JSONResponse(status_code=502, content={"detail": f"Target mock server unreachable: {exc}"})


# Mount mock router as secondary fallback
app.include_router(mock_router)

# Serve frontend files
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
    logger.info(f"Starting Chatbot Server on http://127.0.0.1:{port}...")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False, app_dir=str(BASE_DIR))
