"""
Test Supabase authentication flow.
Run: python tests/test_supabase_auth.py

This will:
  1. Test connection to Supabase
  2. Sign up a test user
  3. Sign in with that user
  4. Fetch the user's info
  5. Fetch the auto-created profile
  6. Sign out
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("  SUPABASE AUTH TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Test connection
    # ------------------------------------------------------------------
    print("\n[1/6] Testing Supabase connection...")
    try:
        from src.database.supabase_client import get_supabase_client
        client = get_supabase_client()
        print("  ✓ Connected to Supabase successfully!")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  → Make sure your .env file has SUPABASE_URL and SUPABASE_ANON_KEY set.")
        return

    # ------------------------------------------------------------------
    # Step 2: Sign up
    # ------------------------------------------------------------------
    print("\n[2/6] Testing sign up...")
    from src.database.auth import sign_up, sign_in, sign_out, get_current_user

    test_email = input("  Enter a test email (a real one — Supabase sends a confirmation): ").strip()
    test_password = input("  Enter a test password (min 6 chars): ").strip()
    test_name = input("  Enter a display name: ").strip() or "Test User"

    result = sign_up(test_email, test_password, test_name)
    if result["success"]:
        print(f"  ✓ Signed up! User ID: {result['data']['user_id']}")
        print("  → Check your email for a confirmation link (if email confirmation is enabled).")
        print("  → If you have email confirmation DISABLED in Supabase dashboard, you can proceed.")
    else:
        print(f"  ✗ Sign up failed: {result['error']}")
        print("  → If the user already exists, that's OK — try signing in next.")

    # ------------------------------------------------------------------
    # Step 3: Sign in
    # ------------------------------------------------------------------
    print("\n[3/6] Testing sign in...")
    result = sign_in(test_email, test_password)
    if result["success"]:
        print(f"  ✓ Signed in! User ID: {result['data']['user_id']}")
        access_token = result["data"]["access_token"]
        user_id = result["data"]["user_id"]
    else:
        print(f"  ✗ Sign in failed: {result['error']}")
        print("  → If you just signed up, you may need to confirm your email first.")
        print("  → To disable email confirmation: Supabase Dashboard → Authentication → Settings → Toggle off 'Enable email confirmations'")
        return

    # ------------------------------------------------------------------
    # Step 4: Get current user
    # ------------------------------------------------------------------
    print("\n[4/6] Testing get current user...")
    result = get_current_user(access_token)
    if result["success"]:
        print(f"  ✓ User info retrieved:")
        print(f"    Email: {result['data']['email']}")
        print(f"    Name:  {result['data']['name']}")
    else:
        print(f"  ✗ Failed: {result['error']}")

    # ------------------------------------------------------------------
    # Step 5: Fetch auto-created profile
    # ------------------------------------------------------------------
    print("\n[5/6] Testing profile auto-creation...")
    from src.database.db_profiles import get_user_profile
    result = get_user_profile(access_token, user_id)
    if result["success"]:
        print(f"  ✓ Profile found in database:")
        profile = result["data"]
        print(f"    Name: {profile.get('name')}")
        print(f"    Preferred model: {profile.get('preferred_model')}")
        print(f"    Created at: {profile.get('created_at')}")
    else:
        print(f"  ✗ Profile fetch failed: {result['error']}")
        print("  → The database trigger may not have run. Check migration.sql was executed.")

    # ------------------------------------------------------------------
    # Step 6: Sign out
    # ------------------------------------------------------------------
    print("\n[6/6] Testing sign out...")
    result = sign_out()
    if result["success"]:
        print("  ✓ Signed out successfully!")
    else:
        print(f"  ✗ Sign out failed: {result['error']}")

    print("\n" + "=" * 60)
    print("  AUTH TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
