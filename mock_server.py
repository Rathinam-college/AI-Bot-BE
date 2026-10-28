"""
Mock Target Server hosting Software 1, Software 2, and Software 3
on port 8080 sharing the same server environment with distinct authentication
and distinct path-routed endpoints.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

from routers.mock_target_apps import mock_router

load_dotenv()

app = FastAPI(
    title="Target Enterprise Applications Mock Server",
    description="Simulates Software 1 (Inventory), Software 2 (CRM), and Software 3 (Support) on port 8080",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_router)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Target Applications Multi-Software Server",
        "port": 8080,
        "endpoints": {
            "app1": {"login": "/app1/auth/login", "data": "/app1/api/data"},
            "app2": {"login": "/app2/auth/login", "data": "/app2/api/data"},
            "app3": {"login": "/app3/auth/login", "data": "/app3/api/data"},
        },
    }


if __name__ == "__main__":
    port = int(os.getenv("MOCK_SERVER_PORT", "8085"))
    print(f"Starting Target Applications Mock Server on http://127.0.0.1:{port}...")
    uvicorn.run("mock_server:app", host="127.0.0.1", port=port, log_level="info", app_dir=str(BASE_DIR))
