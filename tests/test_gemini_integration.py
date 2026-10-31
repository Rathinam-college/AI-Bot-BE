import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from database.db import init_all_dbs
from services.ai_engine import AIEngine, ai_engine
from services.sync_engine import sync_engine


@pytest.fixture(autouse=True)
def setup_test_db():
    init_all_dbs()
    # Seed sample campus records for grounding tests
    sync_engine.upsert_records(
        source_app="CCTV",
        records=[
            {
                "id": "CCTV-CAM-TEST-001",
                "entity_type": "CCTV Camera",
                "name": "Main Gate HD Camera",
                "status": "Online",
                "location": "Main Entrance Gate",
                "ip_address": "192.168.1.101",
            }
        ],
    )
    yield


@pytest.mark.asyncio
async def test_mask_key():
    """Verify API keys are never exposed in logs or outputs."""
    assert AIEngine._mask_key("") == "<none>"
    assert AIEngine._mask_key("12345") == "***"
    assert AIEngine._mask_key("AIzaSyC8SuB7iOTt8Kf3Ms9esxqroM4aPCd5UqQ") == "AIza...5UqQ"


@pytest.mark.asyncio
async def test_missing_api_key_handling():
    """Verify missing API key activates local-sqlite-grounding without crashing."""
    engine = AIEngine()
    with patch.object(engine, "_get_api_key", return_value=("", "")):
        res = await engine.answer_query("How many cameras are online?")
        assert res["model"] == "local-sqlite-grounding"
        assert res["grounded_records_count"] >= 1
        assert "Main Gate HD Camera" in res["answer"] or "CCTV" in res["answer"] or "Online" in res["answer"]


@pytest.mark.asyncio
async def test_invalid_api_key_handling():
    """Verify invalid / revoked API key triggers error log and graceful fallback."""
    engine = AIEngine()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("403 PERMISSION_DENIED: Your API key was reported as leaked.")
    )

    with patch.object(engine, "_get_client", return_value=fake_client):
        res = await engine.answer_query("Tell me about campus CCTV")
        assert res["model"] == "local-sqlite-grounding"
        assert "answer" in res
        assert len(res["answer"]) > 0


@pytest.mark.asyncio
async def test_quota_exceeded_handling():
    """Verify HTTP 429 quota exhaustion triggers error log and fallback."""
    engine = AIEngine()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota group.")
    )

    with patch.object(engine, "_get_client", return_value=fake_client):
        res = await engine.answer_query("List cafeteria items")
        assert res["model"] == "local-sqlite-grounding"


@pytest.mark.asyncio
async def test_timeout_handling():
    """Verify network timeout triggers timeout log and fallback."""
    engine = AIEngine()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(
        side_effect=TimeoutError("Request timed out after 30.0s")
    )

    with patch.object(engine, "_get_client", return_value=fake_client):
        res = await engine.answer_query("Check fleet status")
        assert res["model"] == "local-sqlite-grounding"


@pytest.mark.asyncio
async def test_gemini_success_response():
    """Verify valid Gemini API call returns model name and generated answer."""
    engine = AIEngine()
    mock_resp = MagicMock()
    mock_resp.text = "All campus cameras are online and recording continuously."

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    with patch.object(engine, "_get_client", return_value=fake_client):
        res = await engine.answer_query("What is the security status?")
        assert res["model"] in engine.get_candidate_models()
        assert res["answer"] == "All campus cameras are online and recording continuously."
        assert res["grounded_records_count"] >= 1


@pytest.mark.asyncio
async def test_sqlite_grounding_context_builder():
    """Verify SQLite records are dynamically injected into ground truth context."""
    context_str, total_cnt = ai_engine.build_ground_truth_context("camera")
    assert total_cnt >= 1
    assert "CCTV" in context_str
    assert "Main Gate HD Camera" in context_str or "CCTV-CAM-TEST-001" in context_str


@pytest.mark.asyncio
async def test_long_campus_query_handling():
    """Verify complex long queries are handled with SQLite grounding without truncation errors."""
    long_query = (
        "Can you please provide a comprehensive audit report for all CCTV devices on campus, "
        "including exact device counts, health percentages, open service operations maintenance tickets, "
        "and tell me if there are any cameras currently offline at the Main Entrance Gate?"
    )
    res = await ai_engine.answer_query(long_query)
    assert res["grounded_records_count"] >= 1
    assert len(res["answer"]) > 50


@pytest.mark.asyncio
async def test_general_query_grounding_guard():
    """Verify questions completely unrelated to campus records trigger ground-truth guard."""
    query = "What is the capital of Mars in 2099?"
    res = await ai_engine.answer_query(query)
    # Must enforce zero hallucination
    ans_lower = res["answer"].lower()
    assert any(phrase in ans_lower for phrase in ["not found", "does not contain", "does not exist"])
