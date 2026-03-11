"""Quick end-to-end Supabase smoke test. Run once, then delete."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.database.auth import sign_up, sign_in, get_current_user, sign_out
from src.database.db_profiles import get_user_profile
from src.database.db_analyses import save_analysis, list_analyses, delete_analysis

def ok(label, result):
    status = "PASS" if result["success"] else "FAIL"
    print(f"  [{status}] {label}")
    if result.get("error"):
        print(f"         Error: {result['error']}")
    return result

print("=" * 50)
print("  SUPABASE SMOKE TEST")
print("=" * 50)

# 1. Sign up
print("\n1. Sign up...")
r = ok("sign_up", sign_up("test@stylomex.dev", "TestPass123!", "Test User"))

# 2. Sign in
print("\n2. Sign in...")
r = ok("sign_in", sign_in("test@stylomex.dev", "TestPass123!"))
if not r["success"]:
    print("Cannot continue without sign in.")
    sys.exit(1)

token = r["data"]["access_token"]
uid = r["data"]["user_id"]
print(f"  User ID: {uid}")

# 3. Get current user
print("\n3. Get current user...")
r = ok("get_user", get_current_user(token))
if r["data"]:
    print(f"  Email: {r['data']['email']}")
    print(f"  Name:  {r['data']['name']}")

# 4. Check auto-created profile
print("\n4. Check profile (auto-created by trigger)...")
r = ok("get_profile", get_user_profile(token, uid))
if r["data"]:
    print(f"  Name:  {r['data'].get('name')}")
    print(f"  Model: {r['data'].get('preferred_model')}")

# 5. Save a mock analysis
print("\n5. Save mock analysis...")
mock = {
    "metadata": {"model_used": "gemma3:1b", "processing_mode": "enhanced", "file_info": []},
    "user_profile": {"name": "Test User"},
    "consolidated_analysis": {"test": True},
    "readability_metrics": {"flesch": 62.3},
    "confidence_report": {"overall": 0.82},
    "style_fingerprint_summary": "Test fingerprint",
    "most_distinctive_traits": ["trait1"],
    "key_traits": ["trait2"],
    "rewrite_directive": "Keep it simple",
    "do_not_lose": ["clarity"],
    "avoid_in_rewrite": ["jargon"],
}
r = ok("save_analysis", save_analysis(token, uid, mock))
analysis_id = r["data"]["id"] if r["data"] else None

# 6. List analyses
print("\n6. List analyses...")
r = ok("list_analyses", list_analyses(token, uid))
if r["data"]:
    print(f"  Found {len(r['data'])} analysis(es)")
    for a in r["data"]:
        print(f"    - {a['analysis_name']} | {a['created_at']}")

# 7. Cleanup
if analysis_id:
    print("\n7. Delete test analysis...")
    ok("delete_analysis", delete_analysis(token, uid, analysis_id))

# 8. Sign out
print("\n8. Sign out...")
ok("sign_out", sign_out())

print("\n" + "=" * 50)
print("  SMOKE TEST COMPLETE")
print("=" * 50)
