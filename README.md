# ⚙️ AI-Bot Backend Service (`AI-Bot-BE`)

Production-ready Backend API & Data Synchronization Engine for Rathinam College AI Assistant platform.

---

## 🌟 Key Architecture & Features

1. **Refresh Point Sync Engine (`POST /api/refresh`)**:
   - Concurrently ingests data from enterprise systems (CCTV, Food Tracker, Transport) using `asyncio.gather` and `httpx.AsyncClient`.
   - **Deterministic SHA-256 Differential Upsert**:
     - **INSERT**: New records stored in SQLite and increment `inserted` counter.
     - **SKIP**: Records with identical content hash are skipped without touching storage.
     - **UPDATE**: Records with modified payloads are updated and increment `updated` counter.
   - **Independent Auth Manager**: Per-app bearer tokens with automatic single-retry token re-acquisition upon receiving `401 Unauthorized`.
   - **Fault Tolerance**: Partial degradation handling keeps healthy endpoints syncing even if one target system is down.

2. **Ground-Truth AI Chatbot (`POST /api/chat`)**:
   - Hallucination-proof reasoning powered by Google Gemini (`gemini-flash-lite-latest` or `gemini-2.5-flash`).
   - Grounded strictly on authoritative SQLite database records (`database/app_data.db`).
   - Built-in fallback response mechanism when external LLM credentials are not configured.

3. **Multi-System Mock Server (Port 8085)**:
   - Built-in simulation server hosting Software 1 (CCTV), Software 2 (Food Tracker), and Software 3 (Transport) with isolated authentication.

---

## 📁 Repository Structure

```
.
├── main.py                   # Chatbot FastAPI Application (Port 8000)
├── mock_server.py            # Target Applications Mock Server (Port 8085)
├── requirements.txt          # Production dependencies
├── .env.example              # Template environment configuration
├── database/
│   ├── __init__.py
│   └── db.py                 # SQLite schema, connection pooling & metrics queries
├── routers/
│   ├── __init__.py
│   ├── api.py                # Core REST APIs (/api/refresh, /api/chat, /api/metrics, etc.)
│   └── mock_target_apps.py   # Multi-application mock router for Software 1, 2, and 3
├── services/
│   ├── __init__.py
│   ├── ai_engine.py          # Google GenAI Gemini + Hallucination guard & grounding
│   ├── auth_manager.py       # Independent token management & automatic 401 retry
│   ├── cctv_sql_parser.py    # PostgreSQL SQL dump parser (cameras, tickets, switches, NVRs)
│   └── sync_engine.py        # Parallel ingestion & deterministic SHA-256 diff upsert engine
├── scripts/
│   ├── __init__.py
│   └── import_cctv_sql.py    # Standalone ingestion utility for CCTV backup dumps
├── tests/
│   ├── __init__.py
│   ├── test_sync_engine.py   # 4 sequential refresh cycles, 401 retry & failover test cases
│   └── test_e2e_integration.py # End-to-end integration verification
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY if desired
```

### 3. Run Backend Services
In Terminal 1 (Mock Server):
```bash
python mock_server.py
# Runs on http://127.0.0.1:8085
```

In Terminal 2 (Chatbot Server):
```bash
python main.py
# Runs on http://127.0.0.1:8000
```

---

## 🧪 Running Automated Tests

```bash
pytest -v tests/test_sync_engine.py
```

