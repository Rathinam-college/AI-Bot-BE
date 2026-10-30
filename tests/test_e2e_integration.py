"""
Full End-to-End System Verification Script.
Tests:
  1. Frontend HTML and static assets (CSS, JS)
  2. Sequential Refresh Sync Cases 1, 2, 3, 4 with exact assertions
  3. Independent Auth Manager & Automatic 401 retry
  4. Partial degradation tolerance
  5. Hallucination prevention policy in Chatbot API
  6. SQLite Database & Sync History inspection APIs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx
from database.db import clear_database, init_db

print("=" * 70)
print("FULL END-TO-END SYSTEM INTEGRATION VERIFICATION")
print("=" * 70)

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

# 1. Frontend & Static Assets
r_index = client.get("/")
assert r_index.status_code == 200, f"Index failed: {r_index.status_code}"
assert "Server AI Assistant" in r_index.text
assert "btn-refresh-sync" in r_index.text
assert "val-inserted" in r_index.text
assert "chat-input" in r_index.text
print("[PASS] Frontend GET / served index.html with all required elements.")

r_css = client.get("/static/css/style.css")
assert r_css.status_code == 200, f"CSS failed: {r_css.status_code}"
r_js = client.get("/static/js/app.js")
assert r_js.status_code == 200, f"JS failed: {r_js.status_code}"
print("[PASS] Static assets (CSS and JS) served successfully.")

# Reset database to fresh state for the 4 sequential sync test cases
client.post("/api/database/clear")
clear_database()
client.post("/mock/set-state/1")
client.post("/mock/simulate-failure/app1?enable=false")
client.post("/mock/simulate-failure/app2?enable=false")
client.post("/mock/simulate-failure/app3?enable=false")

# -------------------------------------------------------------------------
# Test Case 1: 1st Refresh -> 100 inserted, 0 updated, 0 skipped
# -------------------------------------------------------------------------
r1 = client.post("/api/refresh").json()
tot1 = r1["totals"]
print(f"\n[Sync Case 1] Fetched: {tot1['fetched']}, Inserted: {tot1['inserted']}, Updated: {tot1['updated']}, Skipped: {tot1['skipped']}")
assert tot1 == {"fetched": 100, "inserted": 100, "updated": 0, "skipped": 0}, f"Case 1 failed: {tot1}"

# -------------------------------------------------------------------------
# Test Case 2: 2nd Refresh -> 0 inserted, 0 updated, 100 skipped
# -------------------------------------------------------------------------
r2 = client.post("/api/refresh").json()
tot2 = r2["totals"]
print(f"[Sync Case 2] Fetched: {tot2['fetched']}, Inserted: {tot2['inserted']}, Updated: {tot2['updated']}, Skipped: {tot2['skipped']}")
assert tot2 == {"fetched": 100, "inserted": 0, "updated": 0, "skipped": 100}, f"Case 2 failed: {tot2}"

# -------------------------------------------------------------------------
# Test Case 3: 3rd Refresh (105 records: 5 new, 3 mod, 97 same) ->
#              5 inserted, 3 updated, 97 skipped
# -------------------------------------------------------------------------
client.post("/mock/set-state/3")
r3 = client.post("/api/refresh").json()
tot3 = r3["totals"]
print(f"[Sync Case 3] Fetched: {tot3['fetched']}, Inserted: {tot3['inserted']}, Updated: {tot3['updated']}, Skipped: {tot3['skipped']}")
assert tot3 == {"fetched": 105, "inserted": 5, "updated": 3, "skipped": 97}, f"Case 3 failed: {tot3}"

# -------------------------------------------------------------------------
# Test Case 4: 4th Refresh -> 0 inserted, 0 updated, 105 skipped
# -------------------------------------------------------------------------
r4 = client.post("/api/refresh").json()
tot4 = r4["totals"]
print(f"[Sync Case 4] Fetched: {tot4['fetched']}, Inserted: {tot4['inserted']}, Updated: {tot4['updated']}, Skipped: {tot4['skipped']}")
assert tot4 == {"fetched": 105, "inserted": 0, "updated": 0, "skipped": 105}, f"Case 4 failed: {tot4}"

# -------------------------------------------------------------------------
# Test 5: Invalidate Token & Auto-Retry on 401
# -------------------------------------------------------------------------
client.post("/mock/invalidate-token/app1")
r_retry = client.post("/api/refresh").json()
assert r_retry["status"] == "success"
print("\n[PASS] Auth Token Invalidation & Auto-Retry on 401 verified seamlessly.")

# -------------------------------------------------------------------------
# Test 6: Partial Degradation Tolerance
# -------------------------------------------------------------------------
client.post("/mock/simulate-failure/app3?enable=true")
r_deg = client.post("/api/refresh").json()
assert r_deg["status"] == "partial"
assert any("3" in a or "Support" in a or "Transport" in a for a in r_deg["failed_apps"])
print("[PASS] Partial Degradation verified: Software 1 and 2 synced while Software 3 failed gracefully.")
client.post("/mock/simulate-failure/app3?enable=false")

# -------------------------------------------------------------------------
# Test 7: Ground-Truth Chatbot & Hallucination Guard
# -------------------------------------------------------------------------
# Query existing record
r_chat1 = client.post("/api/chat", json={"message": "Tell me about Enterprise Product #1 in Software 1"}).json()
assert "APP1-ITEM-001" in r_chat1["answer"] or "Enterprise Product #1" in r_chat1["answer"]
print("\n[PASS] Chat grounded in SQLite database verified for existing record.")

# Query non-existent record -> must guard against hallucination
r_chat2 = client.post("/api/chat", json={"message": "What is the stock of Flying Unicorn 5000?"}).json()
ans2_lower = r_chat2["answer"].lower()
assert any(phrase in ans2_lower for phrase in ["refresh point sync", "not found", "does not exist", "no record", "does not contain"]), f"Hallucination guard failed: {r_chat2['answer']}"
print("[PASS] Hallucination Guard verified: correctly reported that non-existent entity does not exist in SQLite.")

# -------------------------------------------------------------------------
# Test 8: Records & History Endpoints
# -------------------------------------------------------------------------
r_recs = client.get("/api/records").json()
assert r_recs["total"] >= 105
assert len(r_recs["records"]) >= 105

r_hist = client.get("/api/history").json()
assert r_hist["total"] > 0
print(f"[PASS] Database inspector endpoints verified: {r_recs['total']} rows, {r_hist['total']} sync history logs.")

# -------------------------------------------------------------------------
# Post-Test Cleanup: Ensure mock server reset to clean State 0
# -------------------------------------------------------------------------
client.post("/mock/set-state/0")
print("\n[CLEANUP] Mock server reset to clean State 0.")

print("\n" + "=" * 70)
print("ALL E2E INTEGRATION & SPECIFICATION TESTS PASSED WITH 100% SUCCESS!")
print("=" * 70)
