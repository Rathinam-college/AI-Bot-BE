
"""
Comprehensive Automated Verification Test Suite for:
  1. The 4 sequential Refresh Point Sync test cases (asserting exact inserted/updated/skipped counts)
  2. Independent Auth Manager with automatic 401 retry on token expiration
  3. Parallel fetching with partial degradation tolerance
  4. Ground-truth Hallucination Prevention in the Chatbot engine
"""

import asyncio
import os
import sys
import pytest
import httpx

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database.db import clear_database, get_all_records, get_connection, init_db
from routers.mock_target_apps import (
    generate_base_datasets,
    generate_modified_datasets,
    mock_router,
    mock_state,
)
from services.ai_engine import ai_engine
from services.auth_manager import auth_manager
from services.sync_engine import SyncEngine, sync_engine

TEST_DB_PATH = "database/test_app_data.db"


@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Ensure a fresh test database for each test run."""
    init_db(TEST_DB_PATH)
    clear_database(TEST_DB_PATH)
    auth_manager.clear_all_tokens()
    mock_state["current_dataset_state"] = 1
    mock_state["invalidated_tokens"].clear()
    mock_state["simulated_failures"].clear()
    yield
    clear_database(TEST_DB_PATH)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass


class TestSequentialSyncEngine:
    """
    Verifies the 4 exact test cases specified in the system requirements:
      - 1st refresh: 100 records fetched -> 100 inserted, 0 updated, 0 skipped.
      - 2nd refresh: Same 100 records fetched -> 0 inserted, 0 updated, 100 skipped.
      - 3rd refresh: 105 records fetched (5 new, 3 modified, 97 identical) -> 5 inserted, 3 updated, 97 skipped.
      - 4th refresh: Same 105 records fetched -> 0 inserted, 0 updated, 105 skipped.
    """

    def test_four_sequential_refreshes_unit(self):
        init_db(TEST_DB_PATH)
        clear_database(TEST_DB_PATH)
        engine = SyncEngine(db_path=TEST_DB_PATH)

        base_app1, base_app2, base_app3 = generate_base_datasets()
        mod_app1, mod_app2, mod_app3 = generate_modified_datasets()

        # Total counts verification
        total_base_records = len(base_app1) + len(base_app2) + len(base_app3)
        assert total_base_records == 100, f"Expected 100 base records, got {total_base_records}"

        total_mod_records = len(mod_app1) + len(mod_app2) + len(mod_app3)
        assert total_mod_records == 105, f"Expected 105 modified records, got {total_mod_records}"

        # -------------------------------------------------------------------------
        # 1st Refresh: 100 records fetched -> 100 inserted, 0 updated, 0 skipped
        # -------------------------------------------------------------------------
        res1_app1 = engine.upsert_records("Software 1 (Inventory)", base_app1, db_path=TEST_DB_PATH)
        res1_app2 = engine.upsert_records("Software 2 (CRM)", base_app2, db_path=TEST_DB_PATH)
        res1_app3 = engine.upsert_records("Software 3 (Support)", base_app3, db_path=TEST_DB_PATH)

        total1_fetched = res1_app1["fetched"] + res1_app2["fetched"] + res1_app3["fetched"]
        total1_inserted = res1_app1["inserted"] + res1_app2["inserted"] + res1_app3["inserted"]
        total1_updated = res1_app1["updated"] + res1_app2["updated"] + res1_app3["updated"]
        total1_skipped = res1_app1["skipped"] + res1_app2["skipped"] + res1_app3["skipped"]

        print(f"\n[Case 1] Fetched: {total1_fetched}, Inserted: {total1_inserted}, Updated: {total1_updated}, Skipped: {total1_skipped}")
        assert total1_fetched == 100
        assert total1_inserted == 100
        assert total1_updated == 0
        assert total1_skipped == 0

        # Verify DB row count
        all_rows = get_all_records(TEST_DB_PATH)
        assert len(all_rows) == 100

        # -------------------------------------------------------------------------
        # 2nd Refresh: Same 100 records fetched -> 0 inserted, 0 updated, 100 skipped
        # -------------------------------------------------------------------------
        res2_app1 = engine.upsert_records("Software 1 (Inventory)", base_app1, db_path=TEST_DB_PATH)
        res2_app2 = engine.upsert_records("Software 2 (CRM)", base_app2, db_path=TEST_DB_PATH)
        res2_app3 = engine.upsert_records("Software 3 (Support)", base_app3, db_path=TEST_DB_PATH)

        total2_fetched = res2_app1["fetched"] + res2_app2["fetched"] + res2_app3["fetched"]
        total2_inserted = res2_app1["inserted"] + res2_app2["inserted"] + res2_app3["inserted"]
        total2_updated = res2_app1["updated"] + res2_app2["updated"] + res2_app3["updated"]
        total2_skipped = res2_app1["skipped"] + res2_app2["skipped"] + res2_app3["skipped"]

        print(f"[Case 2] Fetched: {total2_fetched}, Inserted: {total2_inserted}, Updated: {total2_updated}, Skipped: {total2_skipped}")
        assert total2_fetched == 100
        assert total2_inserted == 0
        assert total2_updated == 0
        assert total2_skipped == 100

        # Total rows in DB remain 100
        assert len(get_all_records(TEST_DB_PATH)) == 100

        # -------------------------------------------------------------------------
        # 3rd Refresh: 105 records fetched (5 new, 3 modified, 97 identical) ->
        #              5 inserted, 3 updated, 97 skipped
        # -------------------------------------------------------------------------
        res3_app1 = engine.upsert_records("Software 1 (Inventory)", mod_app1, db_path=TEST_DB_PATH)
        res3_app2 = engine.upsert_records("Software 2 (CRM)", mod_app2, db_path=TEST_DB_PATH)
        res3_app3 = engine.upsert_records("Software 3 (Support)", mod_app3, db_path=TEST_DB_PATH)

        total3_fetched = res3_app1["fetched"] + res3_app2["fetched"] + res3_app3["fetched"]
        total3_inserted = res3_app1["inserted"] + res3_app2["inserted"] + res3_app3["inserted"]
        total3_updated = res3_app1["updated"] + res3_app2["updated"] + res3_app3["updated"]
        total3_skipped = res3_app1["skipped"] + res3_app2["skipped"] + res3_app3["skipped"]

        print(f"[Case 3] Fetched: {total3_fetched}, Inserted: {total3_inserted}, Updated: {total3_updated}, Skipped: {total3_skipped}")
        assert total3_fetched == 105
        assert total3_inserted == 5
        assert total3_updated == 3
        assert total3_skipped == 97

        # Total rows in DB is now 105 (100 initial + 5 new)
        assert len(get_all_records(TEST_DB_PATH)) == 105

        # -------------------------------------------------------------------------
        # 4th Refresh: Same 105 records fetched -> 0 inserted, 0 updated, 105 skipped
        # -------------------------------------------------------------------------
        res4_app1 = engine.upsert_records("Software 1 (Inventory)", mod_app1, db_path=TEST_DB_PATH)
        res4_app2 = engine.upsert_records("Software 2 (CRM)", mod_app2, db_path=TEST_DB_PATH)
        res4_app3 = engine.upsert_records("Software 3 (Support)", mod_app3, db_path=TEST_DB_PATH)

        total4_fetched = res4_app1["fetched"] + res4_app2["fetched"] + res4_app3["fetched"]
        total4_inserted = res4_app1["inserted"] + res4_app2["inserted"] + res4_app3["inserted"]
        total4_updated = res4_app1["updated"] + res4_app2["updated"] + res4_app3["updated"]
        total4_skipped = res4_app1["skipped"] + res4_app2["skipped"] + res4_app3["skipped"]

        print(f"[Case 4] Fetched: {total4_fetched}, Inserted: {total4_inserted}, Updated: {total4_updated}, Skipped: {total4_skipped}")
        assert total4_fetched == 105
        assert total4_inserted == 0
        assert total4_updated == 0
        assert total4_skipped == 105

        assert len(get_all_records(TEST_DB_PATH)) == 105


@pytest.mark.asyncio
async def test_auth_manager_and_401_retry():
    """
    Verifies that:
      1. Login successfully acquires token and caches it in-memory.
      2. If an API returns 401 Unauthorized, the token is automatically invalidated,
         re-login is performed once, and the request succeeds seamlessly.
    """
    from fastapi import FastAPI
    mock_state["current_dataset_state"] = 1
    mock_app = FastAPI()
    mock_app.include_router(mock_router)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://127.0.0.1:8080") as client:
        # Step 1: Initial login
        token = await auth_manager.login("app1", client)
        assert token == "mock_token_app1_secure_session"
        assert auth_manager.AUTH_TOKENS["app1"] == token

        # Step 2: Invalidate token in mock server to simulate expiration
        mock_state["invalidated_tokens"].add("app1")

        # Step 3: Fetch with auth retry
        resp = await auth_manager.fetch_with_auth_retry("app1", "/app1/api/data", client)
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert len(data["records"]) == 40
        print("\n[Auth Retry Test] 401 caught, token automatically re-authenticated, data fetched successfully!")


@pytest.mark.asyncio
async def test_partial_degradation_tolerance():
    """
    Verifies that if Software 3 fails (returns 500), Software 1 and 2
    still synchronize successfully without crashing the sync engine.
    """
    from fastapi import FastAPI
    mock_app = FastAPI()
    mock_app.include_router(mock_router)

    # Reset state and clean test database
    clear_database(TEST_DB_PATH)
    mock_state["simulated_failures"].clear()
    mock_state["simulated_failures"].add("app3")

    engine = SyncEngine(db_path=TEST_DB_PATH)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://127.0.0.1:8080") as client:
        # Fetch Software 1
        name1, recs1, err1 = await engine.fetch_app_data("app1", client)
        assert err1 is None
        assert len(recs1) == 40

        # Fetch Software 2
        name2, recs2, err2 = await engine.fetch_app_data("app2", client)
        assert err2 is None
        assert len(recs2) == 35

        # Fetch Software 3 (simulated error)
        name3, recs3, err3 = await engine.fetch_app_data("app3", client)
        assert err3 is not None
        assert len(recs3) == 0

        # Upsert the successful ones
        r1 = engine.upsert_records(name1, recs1, db_path=TEST_DB_PATH)
        r2 = engine.upsert_records(name2, recs2, db_path=TEST_DB_PATH)

        assert r1["inserted"] == 40
        assert r2["inserted"] == 35

        # Database contains records from apps 1 and 2
        rows = get_all_records(TEST_DB_PATH)
        assert len(rows) == 75
        print("\n[Partial Degradation Test] Software 3 failed gracefully; Software 1 and 2 synchronized cleanly.")


@pytest.mark.asyncio
async def test_hallucination_prevention_policy():
    """
    Verifies that the Chatbot enforces strict hallucination prevention:
      - When database is empty -> directs user to click 'Refresh Point Sync'.
      - When querying non-existent records -> notifies user that record does not exist
        in app_data.db and instructs to refresh.
      - When querying existing records -> accurately responds with ground-truth details.
    """
    clear_database(TEST_DB_PATH)
    # 1. Empty database test
    ans1 = await ai_engine.answer_query("What is the stock of product SKU-9999?", db_path=TEST_DB_PATH)
    assert "Refresh Point Sync" in ans1["answer"]
    assert ans1["grounded_records_count"] == 0

    # 2. Populate 1 record in database
    engine = SyncEngine(db_path=TEST_DB_PATH)
    engine.upsert_records(
        "Software 1 (Inventory)",
        [{"id": "APP1-ITEM-001", "name": "Titanium Server Rack", "stock": 42}],
        db_path=TEST_DB_PATH,
    )

    # 3. Query existing record
    ans2 = await ai_engine.answer_query("Show me Titanium Server Rack", db_path=TEST_DB_PATH)
    assert "Titanium Server Rack" in ans2["answer"] or "APP1-ITEM-001" in ans2["answer"]

    # 4. Query non-existent record -> must not hallucinate
    ans3 = await ai_engine.answer_query("Can you tell me about the Flying Hoverboard?", db_path=TEST_DB_PATH)
    ans3_lower = ans3["answer"].lower()
    assert (
        "refresh point sync" in ans3_lower
        or "not found" in ans3_lower
        or "not been recorded" in ans3_lower
        or "does not exist" in ans3_lower
    )
    print("\n[Hallucination Guard Test] Accurately guided user to Refresh Point Sync on missing records!")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SYNC ENGINE & CHATBOT AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    suite = TestSequentialSyncEngine()
    suite.test_four_sequential_refreshes_unit()

    asyncio.run(test_auth_manager_and_401_retry())
    asyncio.run(test_partial_degradation_tolerance())
    asyncio.run(test_hallucination_prevention_policy())
    print("\nALL 4 SYNC TEST CASES & SAFETY CRITERIA PASSED WITH 100% ACCURACY!")
